import argparse
import csv
import glob
import os
import pickle
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

Polygon = List[Tuple[int, int]]


def _normalized_label(value: object) -> str:
    return str(value).strip().lower().replace("-", " ").replace("_", " ")


def resolve_car_class_id(model: YOLO, default: int = 2) -> int:
    names_obj = getattr(model, "names", {})

    if isinstance(names_obj, list):
        names = {i: name for i, name in enumerate(names_obj)}
    elif isinstance(names_obj, dict):
        names = {int(k): v for k, v in names_obj.items()}
    else:
        names = {}

    for class_id, label in names.items():
        if _normalized_label(label) == "car":
            return int(class_id)

    for class_id, label in names.items():
        if "car" in _normalized_label(label):
            return int(class_id)

    if len(names) == 1:
        return int(next(iter(names.keys())))

    if default in names:
        return int(default)

    return int(min(names.keys())) if names else int(default)


def extract_year_hint(path_value: Optional[str]) -> Optional[int]:
    if not isinstance(path_value, str) or not path_value:
        return None
    match = re.search(r"(19|20)\d{2}", path_value)
    if not match:
        return None
    return int(match.group(0))


def normalize_polygon(raw: Any) -> Optional[Polygon]:
    if not isinstance(raw, list):
        return None
    out: Polygon = []
    for pt in raw:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            out.append((int(pt[0]), int(pt[1])))
    return out if len(out) >= 3 else None


def normalize_polygons(raw: Any) -> List[Polygon]:
    if not isinstance(raw, list):
        return []
    out: List[Polygon] = []
    for poly in raw:
        p = normalize_polygon(poly)
        if p:
            out.append(p)
    return out


def select_profile_for_frame(frame_path: str, profiles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not profiles:
        return None
    if len(profiles) == 1:
        return profiles[0]

    year = extract_year_hint(frame_path)
    if year is None:
        return profiles[0]

    exact = [p for p in profiles if p.get("year_hint") == year]
    if exact:
        return exact[0]

    # Allow a truly generic profile only when no exact year exists.
    generic = [p for p in profiles if p.get("year_hint") is None]
    if generic:
        return generic[0]

    # Important safety: do not fall back to another year (e.g., 2024 on 2026 images).
    return None


def load_forbidden_zone_profiles(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        zones = normalize_polygons(data)
        return [{"name": "default", "zones": zones, "reference_image": None, "year_hint": None}]

    if isinstance(data, dict) and "zones" in data:
        zones = normalize_polygons(data.get("zones"))
        ref = data.get("reference_image") if isinstance(data.get("reference_image"), str) else None
        return [{
            "name": "default",
            "zones": zones,
            "reference_image": ref,
            "year_hint": extract_year_hint(ref),
        }]

    if isinstance(data, dict) and "profiles" in data and isinstance(data["profiles"], list):
        profiles = []
        active = data.get("active_profile") if isinstance(data.get("active_profile"), str) else None
        for i, p in enumerate(data["profiles"]):
            if not isinstance(p, dict):
                continue
            name = p.get("name") if isinstance(p.get("name"), str) else f"profile_{i+1}"
            zones = normalize_polygons(p.get("zones"))
            ref = p.get("reference_image") if isinstance(p.get("reference_image"), str) else None
            profiles.append({
                "name": name,
                "zones": zones,
                "reference_image": ref,
                "year_hint": extract_year_hint(ref) or extract_year_hint(name),
                "is_active": active == name,
            })
        profiles.sort(key=lambda x: (0 if x.get("is_active") else 1, str(x.get("name", ""))))
        return profiles

    raise ValueError(f"Unsupported forbidden zones format: {type(data)}")


def load_work_zone_profiles(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        zone = normalize_polygon(data)
        return [{"name": "default", "zone": zone, "reference_image": None, "year_hint": None}] if zone else []

    if isinstance(data, dict) and "zone" in data:
        zone = normalize_polygon(data.get("zone"))
        ref = data.get("reference_image") if isinstance(data.get("reference_image"), str) else None
        return [{
            "name": "default",
            "zone": zone,
            "reference_image": ref,
            "year_hint": extract_year_hint(ref),
        }] if zone else []

    if isinstance(data, dict) and "profiles" in data and isinstance(data["profiles"], list):
        profiles = []
        active = data.get("active_profile") if isinstance(data.get("active_profile"), str) else None
        for i, p in enumerate(data["profiles"]):
            if not isinstance(p, dict):
                continue
            name = p.get("name") if isinstance(p.get("name"), str) else f"profile_{i+1}"
            zone = normalize_polygon(p.get("zone"))
            if not zone:
                print(f"Warning: work-zone profile '{name}' ignored (missing/invalid polygon)")
                continue
            ref = p.get("reference_image") if isinstance(p.get("reference_image"), str) else None
            profiles.append({
                "name": name,
                "zone": zone,
                "reference_image": ref,
                "year_hint": extract_year_hint(ref) or extract_year_hint(name),
                "is_active": active == name,
            })
        profiles.sort(key=lambda x: (0 if x.get("is_active") else 1, str(x.get("name", ""))))
        return profiles

    raise ValueError(f"Unsupported work zone format: {type(data)}")


def to_np_poly(poly: Polygon, dtype=np.int32) -> np.ndarray:
    return np.array(poly, dtype=dtype)


def poly_area(poly: np.ndarray) -> float:
    return float(cv2.contourArea(poly.astype(np.float32)))


def intersection_area(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    area, _ = cv2.intersectConvexConvex(poly_a.astype(np.float32), poly_b.astype(np.float32))
    return float(area)


def collect_images(folder: str, include_subfolders: Optional[List[str]] = None) -> List[str]:
    image_files: List[str] = []
    selected = [s.strip() for s in (include_subfolders or []) if s.strip()]

    if selected:
        for sub in selected:
            sub_path = os.path.join(folder, sub)
            if not os.path.isdir(sub_path):
                print(f"Subfolder not found: {sub_path}")
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                image_files.extend(glob.glob(os.path.join(sub_path, "**", ext), recursive=True))
    else:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))

    return sorted(image_files)


def to_image_relpath(image_path: str, input_root: str) -> str:
    abs_image = os.path.abspath(image_path)
    abs_root = os.path.abspath(input_root)
    try:
        relpath = os.path.relpath(abs_image, abs_root)
    except ValueError:
        relpath = os.path.basename(abs_image)
    return relpath.replace("\\", "/")


def output_path_for_image(image_path: str, input_root: str, output_folder: str) -> str:
    relpath = to_image_relpath(image_path, input_root)
    safe_rel = relpath.replace("/", "__").replace(":", "_")
    return os.path.join(output_folder, f"ModeleFinal_{safe_rel}")


def cleanup_selected_outputs(image_files: List[str], input_root: str, output_folder: str) -> int:
    removed = 0
    for img_path in image_files:
        new_out = output_path_for_image(img_path, input_root, output_folder)
        if os.path.exists(new_out):
            os.remove(new_out)
            removed += 1

        legacy_out = os.path.join(output_folder, f"ModeleFinal_{os.path.basename(img_path)}")
        if legacy_out != new_out and os.path.exists(legacy_out):
            os.remove(legacy_out)
            removed += 1

    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modele Final simplifie - uniquement zones rouge (interdite) et bleue (parking)"
    )
    parser.add_argument("--input-folder", default="../DATA")
    parser.add_argument("--include-subfolders", nargs="*", default=None)
    parser.add_argument("--forbidden-zones", default="zones_interdites.pkl")
    parser.add_argument("--work-zone", default="parking_zone.pkl")

    parser.add_argument("--model-path", default="yolov8m.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--forbidden-overlap-threshold", type=float, default=0.25)
    parser.add_argument("--forbidden-min-overlap-pixels", type=float, default=40.0)
    parser.add_argument("--output-folder", default="resultats_modele_final")
    parser.add_argument("--csv-path", default="resultats_modele_final.csv")
    parser.add_argument("--history-csv", default="resultats_modele_final_history.csv")
    parser.add_argument("--cleanup", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def _resolve_path(path_value: str) -> str:
        if os.path.isabs(path_value):
            return path_value
        return os.path.normpath(os.path.join(script_dir, path_value))

    args.input_folder = _resolve_path(args.input_folder)
    args.forbidden_zones = _resolve_path(args.forbidden_zones)
    args.work_zone = _resolve_path(args.work_zone)
    args.model_path = _resolve_path(args.model_path)

    args.output_folder = _resolve_path(args.output_folder)
    args.csv_path = _resolve_path(args.csv_path)
    args.history_csv = _resolve_path(args.history_csv)

    os.makedirs(args.output_folder, exist_ok=True)

    print(f"Loading YOLO model: {args.model_path}")
    model = YOLO(args.model_path)
    car_class_id = resolve_car_class_id(model)
    print(f"Using car class id: {car_class_id}")

    forbidden_profiles = load_forbidden_zone_profiles(args.forbidden_zones)
    print(f"Loaded {len(forbidden_profiles)} forbidden profile(s)")

    work_profiles = load_work_zone_profiles(args.work_zone)
    print(f"Loaded {len(work_profiles)} work profile(s)")

    image_files = collect_images(args.input_folder, args.include_subfolders)
    print(f"Found {len(image_files)} image(s)")
    if not image_files:
        print("No images found. Exiting.")
        return

    if args.cleanup:
        removed = cleanup_selected_outputs(image_files, args.input_folder, args.output_folder)
        print(f"Removed {removed} old output image(s)")

    with open(args.csv_path, mode="w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "timestamp",
            "image",
            "total_cars",
            "cars_in_forbidden",
            "cars_legal",
            "line_delta_dx",
            "line_delta_dy",
            "alignment_source",
            "alignment_confidence",
            "uncertain",
        ])

        history_rows = []
        last_forbidden_profile = None
        last_work_profile = None

        for frame_idx, img_path in enumerate(image_files):
            print(f"[{frame_idx + 1}/{len(image_files)}] Processing: {img_path}")

            frame = cv2.imread(img_path)
            if frame is None:
                print("  ERROR: Could not read image")
                continue

            forbidden_profile = select_profile_for_frame(img_path, forbidden_profiles)
            work_profile = select_profile_for_frame(img_path, work_profiles)

            forbidden_zones = forbidden_profile.get("zones", []) if forbidden_profile else []
            work_zone = work_profile.get("zone") if work_profile else None

            fp_name = forbidden_profile.get("name") if forbidden_profile else "none"
            wp_name = work_profile.get("name") if work_profile else "none"
            if fp_name != last_forbidden_profile:
                print(f"  Forbidden profile: {fp_name}")
                last_forbidden_profile = fp_name
            if wp_name != last_work_profile:
                print(f"  Work profile: {wp_name}")
                last_work_profile = wp_name

            results = model(frame, conf=args.conf, verbose=False)

            cars = []
            det_confidences: List[float] = []
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy() if getattr(r.boxes, "conf", None) is not None else np.ones(len(boxes), dtype=float)
                for box, cls, det_conf in zip(boxes, classes, confs):
                    if int(cls) != car_class_id:
                        continue
                    x1, y1, x2, y2 = box.astype(int)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    det_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    cars.append({
                        "bbox": (x1, y1, x2, y2),
                        "poly": det_poly,
                        "center": (cx, cy),
                        "area": max(1.0, poly_area(det_poly)),
                    })
                    det_confidences.append(float(det_conf))

            total_cars = len(cars)
            forbidden_np = [to_np_poly(poly, dtype=np.int32) for poly in forbidden_zones]

            cars_in_forbidden = 0
            for car in cars:
                det_poly = car["poly"]
                det_area = car["area"]
                max_forbidden_overlap = 0.0
                max_forbidden_overlap_px = 0.0
                for fpoly in forbidden_np:
                    overlap_area = intersection_area(det_poly, fpoly)
                    max_forbidden_overlap_px = max(max_forbidden_overlap_px, overlap_area)
                    max_forbidden_overlap = max(
                        max_forbidden_overlap,
                        overlap_area / det_area,
                    )
                if (
                    max_forbidden_overlap >= args.forbidden_overlap_threshold
                    and max_forbidden_overlap_px >= args.forbidden_min_overlap_pixels
                ):
                    cars_in_forbidden += 1
                    car["forbidden"] = True
                else:
                    car["forbidden"] = False

            cars_legal = total_cars - cars_in_forbidden
            uncertain = 1 if total_cars == 0 else 0
            frame_confidence = float(np.mean(det_confidences)) if det_confidences else 0.0

            output_frame = frame.copy()

            for car in cars:
                x1, y1, x2, y2 = car["bbox"]
                cx, cy = car["center"]
                if car.get("forbidden", False):
                    color = (0, 0, 255)
                    label = "FORBIDDEN"
                else:
                    color = (0, 255, 255)
                    label = "CAR"

                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(output_frame, (cx, cy), 3, color, -1)
                cv2.putText(
                    output_frame,
                    label,
                    (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            for fpoly in forbidden_np:
                cv2.polylines(output_frame, [fpoly], True, (0, 0, 255), 2)

            if work_zone:
                wz_np = to_np_poly(work_zone, dtype=np.int32)
                cv2.polylines(output_frame, [wz_np], True, (255, 0, 0), 2)

            cv2.rectangle(output_frame, (15, 15), (820, 70), (0, 0, 0), -1)
            cv2.putText(
                output_frame,
                f"Total: {total_cars}  Forbidden: {cars_in_forbidden}  Legal: {cars_legal}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
            )

            output_name = output_path_for_image(img_path, args.input_folder, args.output_folder)
            cv2.imwrite(output_name, output_frame)
            print(f"  Saved: {output_name}")

            rel_image = to_image_relpath(img_path, args.input_folder)
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                rel_image,
                total_cars,
                cars_in_forbidden,
                cars_legal,
                "0.0",
                "0.0",
                "disabled",
                f"{frame_confidence:.4f}",
                uncertain,
            ]
            writer.writerow(row)
            history_rows.append(row)

    history_exists = os.path.exists(args.history_csv)
    with open(args.history_csv, mode="a", newline="", encoding="utf-8") as f_hist:
        hist_writer = csv.writer(f_hist)
        if not history_exists:
            hist_writer.writerow([
                "timestamp",
                "image",
                "total_cars",
                "cars_in_forbidden",
                "cars_legal",
                "line_delta_dx",
                "line_delta_dy",
                "alignment_source",
                "alignment_confidence",
                "uncertain",
            ])
        for row in history_rows:
            hist_writer.writerow(row)

    print("=== Processing complete ===")
    print("Done!")


if __name__ == "__main__":
    main()
