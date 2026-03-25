import argparse
import csv
import glob
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from nettoyer_check_manuel import clean_manual_csv


Polygon = List[Tuple[int, int]]


@dataclass
class TunedParams:
    conf: float
    occupancy_threshold: float
    illegal_slot_overlap_threshold: float
    forbidden_overlap_threshold: float
    min_matches: int
    min_inliers: int
    min_inlier_ratio: float


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def tune_params_from_manual_check(args: argparse.Namespace) -> TunedParams:
    tuned = TunedParams(
        conf=args.conf,
        occupancy_threshold=args.occupancy_threshold,
        illegal_slot_overlap_threshold=args.illegal_slot_overlap_threshold,
        forbidden_overlap_threshold=args.forbidden_overlap_threshold,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
    )

    check_csv = args.manual_check_csv
    if not check_csv or not os.path.exists(check_csv):
        return tuned

    try:
        clean_manual_csv(check_csv)
    except Exception as e:
        print(f"Nettoyage auto check manuel ignore ({e})")

    by_image = {}
    with open(check_csv, mode="r", newline="", encoding="utf-8") as f:
        dict_reader = csv.DictReader(f)
        has_err10_col = dict_reader.fieldnames is not None and "err10" in dict_reader.fieldnames

        if has_err10_col:
            for row in dict_reader:
                img = row.get("image", "").strip()
                if not img:
                    continue
                # Keep latest annotation when an image appears multiple times.
                by_image[img] = {
                    "err1": _safe_int(row.get("err1", "0"), 0),
                    "err2": _safe_int(row.get("err2", "0"), 0),
                    "err10": _safe_int(row.get("err10", "0"), 0),
                }
        else:
            f.seek(0)
            raw_reader = csv.reader(f)
            next(raw_reader, None)  # header
            for row in raw_reader:
                # Legacy schema: image, err1..err9, err10?, places_detectees, coords
                if not row:
                    continue
                img = row[0].strip()
                if not img:
                    continue
                err1 = _safe_int(row[1], 0) if len(row) > 1 else 0
                err2 = _safe_int(row[2], 0) if len(row) > 2 else 0
                err10 = _safe_int(row[10], 0) if len(row) > 10 else 0
                by_image[img] = {
                    "err1": err1,
                    "err2": err2,
                    "err10": err10,
                }

    if not by_image:
        return tuned

    n = len(by_image)
    err1_total = 0
    err2_total = 0
    err10_total = 0

    for row in by_image.values():
        err1_total += int(row["err1"])
        err2_total += int(row["err2"])
        err10_total += int(row["err10"])

    err1_avg = err1_total / max(n, 1)
    err2_avg = err2_total / max(n, 1)
    err10_avg = err10_total / max(n, 1)

    if err1_avg >= 1.0:
        tuned.conf = max(0.10, tuned.conf - 0.10)
        tuned.occupancy_threshold = max(0.12, tuned.occupancy_threshold - 0.08)

    if err1_avg >= 2.0:
        tuned.conf = max(0.08, tuned.conf - 0.05)
        tuned.occupancy_threshold = max(0.10, tuned.occupancy_threshold - 0.03)

    if err2_avg >= 0.5:
        tuned.conf = min(0.70, tuned.conf + 0.05)

    if err10_avg >= 0.5:
        tuned.min_matches = max(tuned.min_matches, 40)
        tuned.min_inliers = max(tuned.min_inliers, 18)
        tuned.min_inlier_ratio = max(tuned.min_inlier_ratio, 0.55)

    if err10_avg >= 1.0:
        tuned.min_matches = max(tuned.min_matches, 50)
        tuned.min_inliers = max(tuned.min_inliers, 22)
        tuned.min_inlier_ratio = max(tuned.min_inlier_ratio, 0.62)

    print("\nCalibration depuis check manuel:")
    print(f"- images annotees: {n}")
    print(f"- moyenne err1 (voitures non detectees): {err1_avg:.2f}")
    print(f"- moyenne err2 (fausses detections): {err2_avg:.2f}")
    print(f"- moyenne err10 (erreur cadrage place): {err10_avg:.2f}")
    print(
        f"- params utilises: conf={tuned.conf:.2f}, occupancy={tuned.occupancy_threshold:.2f}, "
        f"min_matches={tuned.min_matches}, min_inliers={tuned.min_inliers}, min_inlier_ratio={tuned.min_inlier_ratio:.2f}"
    )

    return tuned


def load_parking_slots(path: str) -> List[Polygon]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        return data

    raise ValueError(f"Format de parking_slots non supporte: {type(data)}")


def load_forbidden_zones(path: str) -> List[Polygon]:
    if not os.path.exists(path):
        return []

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "zones" in data and isinstance(data["zones"], list):
            return data["zones"]

    raise ValueError(f"Format de zones interdites non supporte: {type(data)}")


def load_optional_zone_polygon(path: str) -> Polygon | None:
    if not path or not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "zone" in data:
        return data["zone"]

    return None


def to_np_poly(poly: Polygon, dtype=np.int32) -> np.ndarray:
    return np.array(poly, dtype=dtype)


def poly_area(poly: np.ndarray) -> float:
    return float(cv2.contourArea(poly.astype(np.float32)))


def intersection_area(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    area, _ = cv2.intersectConvexConvex(
        poly_a.astype(np.float32), poly_b.astype(np.float32)
    )
    return float(area)


def preprocess_for_orb(gray: np.ndarray, use_clahe: bool, use_normalize: bool) -> np.ndarray:
    processed = gray
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
    if use_normalize:
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
    return processed


def build_orb_mask(shape: tuple[int, int], ignore_top_px: int) -> np.ndarray | None:
    if ignore_top_px <= 0:
        return None

    h, w = shape
    top = min(ignore_top_px, h)
    if top >= h:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:, :] = 255
    return mask


def is_homography_reasonable(
    h: np.ndarray | None,
    max_translation: float,
    min_scale: float,
    max_scale: float,
    max_perspective: float,
) -> bool:
    if h is None:
        return False

    if abs(float(h[2, 2])) < 1e-9:
        return False

    hn = h / h[2, 2]

    tx = abs(float(hn[0, 2]))
    ty = abs(float(hn[1, 2]))

    scale_x = float(np.linalg.norm(hn[0:2, 0]))
    scale_y = float(np.linalg.norm(hn[0:2, 1]))

    persp_x = abs(float(hn[2, 0]))
    persp_y = abs(float(hn[2, 1]))

    if tx > max_translation or ty > max_translation:
        return False
    if not (min_scale < scale_x < max_scale and min_scale < scale_y < max_scale):
        return False
    if persp_x > max_perspective or persp_y > max_perspective:
        return False

    return True


def compute_homography_or_none(
    ref_image: np.ndarray,
    cur_image: np.ndarray,
    min_matches: int,
    min_inliers: int,
    min_inlier_ratio: float,
    max_translation: float,
    min_scale: float,
    max_scale: float,
    max_perspective: float,
    orb_ignore_top_px: int,
    orb_use_clahe: bool,
    orb_use_normalize: bool,
) -> Tuple[np.ndarray | None, int, float, int]:
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)

    ref_gray = preprocess_for_orb(ref_gray, use_clahe=orb_use_clahe, use_normalize=orb_use_normalize)
    cur_gray = preprocess_for_orb(cur_gray, use_clahe=orb_use_clahe, use_normalize=orb_use_normalize)

    orb_mask = build_orb_mask(ref_gray.shape, ignore_top_px=orb_ignore_top_px)

    orb = cv2.ORB_create(3500)
    kp1, des1 = orb.detectAndCompute(ref_gray, orb_mask)
    kp2, des2 = orb.detectAndCompute(cur_gray, orb_mask)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, 0, 0.0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    match_count = len(good)
    if match_count < min_matches:
        return None, 0, 0.0, match_count

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if h is None or mask is None:
        return None, 0, 0.0, match_count

    inliers = int(mask.sum())
    inlier_ratio = inliers / max(match_count, 1)

    if inliers < min_inliers or inlier_ratio < min_inlier_ratio:
        return None, inliers, inlier_ratio, match_count

    if not is_homography_reasonable(
        h,
        max_translation=max_translation,
        min_scale=min_scale,
        max_scale=max_scale,
        max_perspective=max_perspective,
    ):
        return None, inliers, inlier_ratio, match_count

    return h, inliers, inlier_ratio, match_count


def warp_polygons(polygons: List[Polygon], h: np.ndarray | None) -> List[Polygon]:
    if h is None:
        return [poly.copy() for poly in polygons]

    warped = []
    for poly in polygons:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, h).reshape(-1, 2)
        warped.append([(int(x), int(y)) for x, y in dst])
    return warped


def collect_images(folder: str) -> List[str]:
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(image_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modele final v1: recalage camera + detection de stationnement illegal"
    )
    parser.add_argument("--input-folder", default="../Model 1/image_de_depart_pour_analyse")
    parser.add_argument("--parking-slots", default="../Model 1/parking_slots.pkl")
    parser.add_argument("--parking-zone", default="../Model 3/detection_zone_2.pkl")
    parser.add_argument("--forbidden-zones", default="zones_interdites.pkl")
    parser.add_argument("--model-path", default="yolov8m.pt")
    parser.add_argument("--reference-image", default="")
    parser.add_argument("--output-folder", default="resultats_modele_final")
    parser.add_argument("--csv-path", default="resultats_modele_final.csv")
    parser.add_argument("--manual-check-csv", default="check_manuel_results.csv")

    parser.add_argument("--conf", type=float, default=0.22)
    parser.add_argument("--occupancy-threshold", type=float, default=0.22)
    # Lower sensitivity for illegal flagging: require stronger evidence.
    parser.add_argument("--illegal-slot-overlap-threshold", type=float, default=0.05)
    parser.add_argument("--forbidden-overlap-threshold", type=float, default=0.25)

    parser.add_argument("--min-matches", type=int, default=25)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.35)
    parser.add_argument("--max-translation", type=float, default=50.0)
    parser.add_argument("--min-scale", type=float, default=0.8)
    parser.add_argument("--max-scale", type=float, default=1.2)
    parser.add_argument("--max-perspective", type=float, default=0.002)
    parser.add_argument("--orb-ignore-top-px", type=int, default=100)
    parser.add_argument("--orb-no-clahe", action="store_true")
    parser.add_argument("--orb-no-normalize", action="store_true")
    parser.add_argument("--keep-last-good-h", action="store_true")
    parser.add_argument("--temporal-smooth-alpha", type=float, default=0.2)
    parser.add_argument("--disable-alignment", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tuned = tune_params_from_manual_check(args)

    os.makedirs(args.output_folder, exist_ok=True)

    image_files = collect_images(args.input_folder)
    if not image_files:
        print(f"Aucune image trouvee dans {args.input_folder}")
        return

    parking_slots_ref = load_parking_slots(args.parking_slots)
    forbidden_ref = load_forbidden_zones(args.forbidden_zones)
    zone_ref = load_optional_zone_polygon(args.parking_zone)

    model = YOLO(args.model_path)

    if args.reference_image:
        reference_path = args.reference_image
    else:
        reference_path = image_files[0]

    reference_image = cv2.imread(reference_path)
    if reference_image is None:
        print(f"Image de reference introuvable: {reference_path}")
        return

    with open(args.csv_path, mode="w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "image",
                "free_places",
                "occupied_places",
                "total_places",
                "cars_detected",
                "illegal_parked",
                "alignment_ok",
                "match_count",
                "inliers",
                "inlier_ratio",
            ]
        )

        last_good_h = None
        temporal_alpha = float(np.clip(args.temporal_smooth_alpha, 0.0, 1.0))
        orb_use_clahe = not args.orb_no_clahe
        orb_use_normalize = not args.orb_no_normalize

        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Lecture impossible: {img_path}")
                continue

            if args.disable_alignment:
                h = None
                alignment_ok = 0
                inliers = 0
                inlier_ratio = 0.0
                match_count = 0
            else:
                h, inliers, inlier_ratio, match_count = compute_homography_or_none(
                    reference_image,
                    frame,
                    min_matches=tuned.min_matches,
                    min_inliers=tuned.min_inliers,
                    min_inlier_ratio=tuned.min_inlier_ratio,
                    max_translation=args.max_translation,
                    min_scale=args.min_scale,
                    max_scale=args.max_scale,
                    max_perspective=args.max_perspective,
                    orb_ignore_top_px=args.orb_ignore_top_px,
                    orb_use_clahe=orb_use_clahe,
                    orb_use_normalize=orb_use_normalize,
                )

                if h is not None and last_good_h is not None and temporal_alpha > 0.0:
                    h = (1.0 - temporal_alpha) * last_good_h + temporal_alpha * h

                if h is not None:
                    last_good_h = h.copy()
                elif args.keep_last_good_h and last_good_h is not None:
                    h = last_good_h.copy()

                alignment_ok = int(h is not None)

            parking_slots_cur = warp_polygons(parking_slots_ref, h)
            forbidden_cur = warp_polygons(forbidden_ref, h)
            zone_cur = warp_polygons([zone_ref], h)[0] if zone_ref is not None else None

            # YOLO: classe 2 = voiture dans COCO
            results = model(frame, conf=tuned.conf, verbose=False)

            cars = []
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                for box, cls in zip(boxes, classes):
                    if int(cls) != 2:
                        continue
                    x1, y1, x2, y2 = box.astype(int)
                    det_poly = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        dtype=np.int32,
                    )
                    cars.append(
                        {
                            "bbox": (x1, y1, x2, y2),
                            "poly": det_poly,
                            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                            "area": max(1.0, poly_area(det_poly)),
                        }
                    )

            # Occupation place par place
            free_places = 0
            occupied_places = 0
            slot_np = [to_np_poly(poly, dtype=np.int32) for poly in parking_slots_cur]

            for poly in slot_np:
                slot_area = max(1.0, poly_area(poly))
                is_occupied = False

                for car in cars:
                    ratio = intersection_area(poly, car["poly"]) / slot_area
                    if ratio > tuned.occupancy_threshold:
                        is_occupied = True
                        break

                if is_occupied:
                    occupied_places += 1
                    color = (0, 0, 255)
                else:
                    free_places += 1
                    color = (0, 255, 0)

                cv2.polylines(frame, [poly], True, color, 2)

            # Stationnement illegal
            forbidden_np = [to_np_poly(poly, dtype=np.int32) for poly in forbidden_cur]
            zone_np = to_np_poly(zone_cur, dtype=np.int32) if zone_cur is not None else None

            illegal_count = 0
            for car in cars:
                det_poly = car["poly"]
                det_area = car["area"]
                cx, cy = car["center"]

                max_slot_overlap = 0.0
                for slot in slot_np:
                    max_slot_overlap = max(
                        max_slot_overlap,
                        intersection_area(det_poly, slot) / det_area,
                    )

                max_forbidden_overlap = 0.0
                for fpoly in forbidden_np:
                    max_forbidden_overlap = max(
                        max_forbidden_overlap,
                        intersection_area(det_poly, fpoly) / det_area,
                    )

                in_zone = True
                if zone_np is not None:
                    in_zone = cv2.pointPolygonTest(zone_np, (float(cx), float(cy)), False) >= 0

                center_in_any_slot = False
                for slot in slot_np:
                    if cv2.pointPolygonTest(slot.astype(np.float32), (float(cx), float(cy)), False) >= 0:
                        center_in_any_slot = True
                        break

                is_illegal = False
                if max_forbidden_overlap >= tuned.forbidden_overlap_threshold:
                    is_illegal = True
                elif in_zone and (not center_in_any_slot) and max_slot_overlap < tuned.illegal_slot_overlap_threshold:
                    is_illegal = True

                x1, y1, x2, y2 = car["bbox"]
                if is_illegal:
                    illegal_count += 1
                    color = (0, 0, 255)
                    tag = "ILLEGAL"
                else:
                    color = (255, 255, 0)
                    tag = "LEGAL"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                cv2.putText(
                    frame,
                    tag,
                    (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            for fpoly in forbidden_np:
                cv2.polylines(frame, [fpoly], True, (0, 0, 255), 2)

            if zone_np is not None:
                cv2.polylines(frame, [zone_np], True, (255, 0, 0), 2)

            # Cartouche resume
            cv2.rectangle(frame, (15, 15), (880, 170), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Free: {free_places}/{len(slot_np)}  Occupied: {occupied_places}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Cars: {len(cars)}  Illegal: {illegal_count}",
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 200, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Alignment ok: {alignment_ok}  matches: {match_count}  inliers: {inliers}  ratio: {inlier_ratio:.2f}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            file_name = os.path.basename(img_path)
            output_name = os.path.join(args.output_folder, f"ModeleFinal_{file_name}")
            cv2.imwrite(output_name, frame)

            writer.writerow(
                [
                    file_name,
                    free_places,
                    occupied_places,
                    len(slot_np),
                    len(cars),
                    illegal_count,
                    alignment_ok,
                    match_count,
                    inliers,
                    f"{inlier_ratio:.4f}",
                ]
            )

            print(
                f"OK {file_name} | free={free_places} occupied={occupied_places} illegal={illegal_count} align={alignment_ok}"
            )

    print("\nTermine.")
    print(f"CSV: {args.csv_path}")
    print(f"Images: {args.output_folder}")


if __name__ == "__main__":
    main()
