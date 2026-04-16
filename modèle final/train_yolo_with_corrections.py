#!/usr/bin/env python3
"""Export manual corrections to a YOLO dataset and fine-tune the detector.

The correction GUI stores click-based annotations in manual_review_annotations.json.
This script converts those corrections into a small YOLO dataset, then fine-tunes
an Ultralytics YOLO model and stores the updated weights for the next pipeline run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../DATA").resolve()
ANNOTATIONS_JSON = BASE_DIR / "manual_review_annotations.json"
DEFAULT_OUTPUT_MODEL = BASE_DIR / "parking_detector_corrections.pt"
DEFAULT_BASE_MODEL = BASE_DIR / "yolov8m.pt"
DEFAULT_DATASET_DIR = BASE_DIR / "yolo_correction_dataset"
DEFAULT_REVIEWED_CSV = BASE_DIR / "manual_review_done.csv"
DEFAULT_DECISIONS_CSV = BASE_DIR / "training_decisions_history.csv"
DEFAULT_REPORT_JSON = BASE_DIR / "training_last_report.json"
DEFAULT_REPORT_HTML = BASE_DIR / "training_last_report.html"
MIN_IMAGES_FOR_TRAINING = 8
MIN_BOXES_FOR_TRAINING = 40

Point = Tuple[int, int, str, float, float]
Box = Tuple[float, float, float, float]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO from manual corrections")
    parser.add_argument("--annotations-json", default=str(ANNOTATIONS_JSON))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--output-model", default=str(DEFAULT_OUTPUT_MODEL))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--reviewed-csv", default=str(DEFAULT_REVIEWED_CSV))
    parser.add_argument("--decisions-csv", default=str(DEFAULT_DECISIONS_CSV))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-html", default=str(DEFAULT_REPORT_HTML))
    parser.add_argument("--min-improvement-ratio", type=float, default=0.98)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--close-mosaic", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_annotations(path: Path) -> Dict[str, List[Point]]:
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    images = payload.get("images", {}) if isinstance(payload, dict) else {}
    if not isinstance(images, dict):
        return {}

    result: Dict[str, List[Point]] = {}
    for image_key, items in images.items():
        if not isinstance(image_key, str) or not isinstance(items, list):
            continue
        points: List[Point] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                x = int(item.get("x", 0))
                y = int(item.get("y", 0))
                point_type = str(item.get("type", "legal"))
                box_w = float(item.get("box_w", 0) or 0)
                box_h = float(item.get("box_h", 0) or 0)
            except Exception:
                continue
            points.append((x, y, point_type, box_w, box_h))
        if points:
            result[image_key] = points
    return result


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    if not image_path.exists():
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def run_yolo_predictions(model: YOLO, image_path: Path, conf: float) -> List[Box]:
    car_class_id = resolve_car_class_id(model)
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    boxes: List[Box] = []
    for result in results:
        if getattr(result, "boxes", None) is None:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy() if getattr(result.boxes, "xyxy", None) is not None else []
        classes = result.boxes.cls.cpu().numpy() if getattr(result.boxes, "cls", None) is not None else []
        for box, cls in zip(xyxy, classes):
            if int(cls) != car_class_id:
                continue
            x1, y1, x2, y2 = box.astype(float)
            boxes.append((x1, y1, x2, y2))
    return boxes


def mean_car_count(model_path: Path, image_paths: Sequence[Path], conf: float) -> float:
    if not image_paths:
        return 0.0
    model = YOLO(str(model_path))
    car_class_id = resolve_car_class_id(model)
    totals = 0
    valid = 0
    for image_path in image_paths:
        if not image_path.exists():
            continue
        results = model.predict(source=str(image_path), conf=conf, verbose=False)
        cars = 0
        for result in results:
            if getattr(result, "boxes", None) is None:
                continue
            classes = result.boxes.cls.cpu().numpy() if getattr(result.boxes, "cls", None) is not None else []
            for cls in classes:
                if int(cls) == car_class_id:
                    cars += 1
        totals += cars
        valid += 1
    return (totals / valid) if valid > 0 else 0.0


def car_count_for_image(model: YOLO, image_path: Path, conf: float, car_class_id: int) -> Optional[int]:
    if not image_path.exists():
        return None
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    cars = 0
    for result in results:
        if getattr(result, "boxes", None) is None:
            continue
        classes = result.boxes.cls.cpu().numpy() if getattr(result.boxes, "cls", None) is not None else []
        for cls in classes:
            if int(cls) == car_class_id:
                cars += 1
    return cars


def mae_on_reviewed(model_path: Path, reviewed_csv: Path, data_dir: Path, conf: float) -> Tuple[float, int]:
    if not reviewed_csv.exists():
        return float("inf"), 0

    model = YOLO(str(model_path))
    car_class_id = resolve_car_class_id(model)
    errors: List[float] = []

    with open(reviewed_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_rel = (row.get("image") or "").strip()
            if not image_rel:
                continue
            try:
                target_total = int(row.get("corrected_total", "0") or 0)
            except Exception:
                continue

            image_path = data_dir / image_rel
            pred_total = car_count_for_image(model, image_path, conf, car_class_id)
            if pred_total is None:
                continue
            errors.append(abs(float(pred_total - target_total)))

    if not errors:
        return float("inf"), 0

    return sum(errors) / float(len(errors)), len(errors)


def box_center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_size(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def clamp_box(x_center: float, y_center: float, box_w: float, box_h: float, img_w: int, img_h: int) -> Box:
    half_w = box_w / 2.0
    half_h = box_h / 2.0
    x1 = max(0.0, x_center - half_w)
    y1 = max(0.0, y_center - half_h)
    x2 = min(float(img_w - 1), x_center + half_w)
    y2 = min(float(img_h - 1), y_center + half_h)
    if x2 <= x1:
        x2 = min(float(img_w - 1), x1 + max(8.0, box_w))
    if y2 <= y1:
        y2 = min(float(img_h - 1), y1 + max(8.0, box_h))
    return (x1, y1, x2, y2)


def nearest_box_index(boxes: Sequence[Box], point: Tuple[int, int]) -> Tuple[Optional[int], float]:
    if not boxes:
        return None, float("inf")
    px, py = point
    best_idx = None
    best_dist = float("inf")
    for idx, box in enumerate(boxes):
        cx, cy = box_center(box)
        dist = math.hypot(cx - px, cy - py)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx, best_dist


def deduplicate_box(boxes: List[Box], candidate: Box, min_center_distance: float = 28.0) -> bool:
    cx, cy = box_center(candidate)
    for box in boxes:
        ocx, ocy = box_center(box)
        if math.hypot(cx - ocx, cy - ocy) < min_center_distance:
            return True
    return False


def apply_corrections_to_boxes(boxes: List[Box], points: Sequence[Point], img_w: int, img_h: int) -> List[Box]:
    working = list(boxes)

    predicted_sizes = [box_size(box) for box in working]
    if predicted_sizes:
        template_w = float(median([w for w, _ in predicted_sizes]))
        template_h = float(median([h for _, h in predicted_sizes]))
    else:
        template_w = max(48.0, img_w * 0.07)
        template_h = max(28.0, img_h * 0.04)

    for x, y, point_type, _, _ in points:
        if point_type == "false_positive":
            idx, dist = nearest_box_index(working, (x, y))
            if idx is not None and dist < max(template_w, template_h) * 1.5:
                working.pop(idx)

    for x, y, point_type, point_w, point_h in points:
        if point_type not in {"legal", "illegal", "missed"}:
            continue
        local_w = float(point_w) if point_w and point_w > 4 else template_w
        local_h = float(point_h) if point_h and point_h > 4 else template_h
        candidate = clamp_box(float(x), float(y), local_w, local_h, img_w, img_h)
        if not deduplicate_box(working, candidate):
            working.append(candidate)

    return working


def split_items(items: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if not items:
        return [], []
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    n_val = min(n_val, len(shuffled) - 1) if len(shuffled) > 1 else 0
    val_items = shuffled[:n_val]
    train_items = shuffled[n_val:]
    if not train_items:
        train_items = val_items[:]
        val_items = []
    return train_items, val_items


def write_yolo_label(path: Path, boxes: Sequence[Box], img_w: int, img_h: int) -> None:
    lines = []
    for box in boxes:
        x1, y1, x2, y2 = box
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        x_center = x1 + width / 2.0
        y_center = y1 + height / 2.0
        lines.append(
            f"0 {x_center / img_w:.6f} {y_center / img_h:.6f} {width / img_w:.6f} {height / img_h:.6f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def build_dataset(
    annotations: Dict[str, List[Point]],
    data_dir: Path,
    dataset_dir: Path,
    model: YOLO,
    conf: float,
    val_ratio: float,
    seed: int,
) -> Tuple[int, int, int, Path, List[Path]]:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    for split in ("train", "val"):
        ensure_dir(images_dir / split)
        ensure_dir(labels_dir / split)

    items = sorted(annotations.items(), key=lambda kv: kv[0])
    if not items:
        return 0, 0, 0, dataset_dir / "data.yaml", []

    train_items, val_items = split_items([item[0] for item in items], val_ratio, seed)
    split_lookup = {image: "train" for image in train_items}
    split_lookup.update({image: "val" for image in val_items})

    exported = 0
    kept_train = 0
    kept_val = 0
    total_boxes = 0
    probe_images: List[Path] = []

    for image_relpath, points in items:
        split = split_lookup.get(image_relpath, "train")
        source_image = data_dir / image_relpath
        size = load_image_size(source_image)
        if size is None:
            continue
        img_w, img_h = size

        predicted_boxes = run_yolo_predictions(model, source_image, conf=conf)
        corrected_boxes = apply_corrections_to_boxes(predicted_boxes, points, img_w, img_h)
        total_boxes += len(corrected_boxes)

        destination_image = images_dir / split / Path(image_relpath)
        destination_label = labels_dir / split / Path(image_relpath).with_suffix(".txt")
        ensure_dir(destination_image.parent)
        ensure_dir(destination_label.parent)
        shutil.copy2(source_image, destination_image)
        write_yolo_label(destination_label, corrected_boxes, img_w, img_h)
        if len(probe_images) < 20:
            probe_images.append(source_image)

        exported += 1
        if split == "train":
            kept_train += 1
        else:
            kept_val += 1

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {dataset_dir.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write("  0: car\n")

    return kept_train, kept_val, total_boxes, yaml_path, probe_images


def copy_model(source: Path, target: Path) -> None:
    ensure_dir(target.parent)
    if source != target:
        shutil.copy2(source, target)


def append_decision_log(
    decisions_csv: Path,
    status: str,
    reason: str,
    output_model: Path,
    train_images: int = 0,
    val_images: int = 0,
    total_boxes: int = 0,
    baseline_mean: Optional[float] = None,
    candidate_mean: Optional[float] = None,
    baseline_mae: Optional[float] = None,
    candidate_mae: Optional[float] = None,
    n_eval_base: int = 0,
    n_eval_candidate: int = 0,
    min_improvement_ratio: float = 0.98,
) -> None:
    decisions_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = decisions_csv.exists()
    with open(decisions_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "status",
                "reason",
                "output_model",
                "train_images",
                "val_images",
                "total_boxes",
                "baseline_mean",
                "candidate_mean",
                "baseline_mae",
                "candidate_mae",
                "n_eval_base",
                "n_eval_candidate",
                "min_improvement_ratio",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": status,
                "reason": reason,
                "output_model": str(output_model),
                "train_images": train_images,
                "val_images": val_images,
                "total_boxes": total_boxes,
                "baseline_mean": "" if baseline_mean is None else f"{baseline_mean:.4f}",
                "candidate_mean": "" if candidate_mean is None else f"{candidate_mean:.4f}",
                "baseline_mae": "" if baseline_mae is None else f"{baseline_mae:.4f}",
                "candidate_mae": "" if candidate_mae is None else f"{candidate_mae:.4f}",
                "n_eval_base": n_eval_base,
                "n_eval_candidate": n_eval_candidate,
                "min_improvement_ratio": f"{min_improvement_ratio:.4f}",
            }
        )


def _safe_float(value: Optional[float]) -> str:
        if value is None:
                return "-"
        return f"{float(value):.4f}"


def _safe_int(value: Optional[int]) -> str:
        if value is None:
                return "-"
        return str(int(value))


def load_recent_decisions(decisions_csv: Path, limit: int = 15) -> List[Dict[str, str]]:
        if not decisions_csv.exists():
                return []
        with open(decisions_csv, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        return rows[-limit:]


def _as_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def write_training_report(
    report_json: Path,
    report_html: Path,
    payload: Dict[str, object],
    recent_decisions: List[Dict[str, str]],
) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_html.parent.mkdir(parents=True, exist_ok=True)

    serializable = dict(payload)
    serializable["recent_decisions"] = recent_decisions

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    decision = str(payload.get("decision_status", "unknown"))
    decision_color = "#2e7d32" if decision == "accepted" else ("#c62828" if decision == "rejected" else "#1565c0")

    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    params = payload.get("parameters", {}) if isinstance(payload.get("parameters"), dict) else {}
    dataset = payload.get("dataset", {}) if isinstance(payload.get("dataset"), dict) else {}
    status_counts = {"accepted": 0, "rejected": 0, "skipped": 0, "failed": 0}
    labels: List[str] = []
    baseline_mae_series: List[Optional[float]] = []
    candidate_mae_series: List[Optional[float]] = []
    baseline_mean_series: List[Optional[float]] = []
    candidate_mean_series: List[Optional[float]] = []

    rows_html = []
    for row in recent_decisions:
        status = str(row.get("status", "")).strip().lower()
        if status in status_counts:
            status_counts[status] += 1

        labels.append(str(row.get("timestamp", ""))[-8:])
        baseline_mae_series.append(_as_float(row.get("baseline_mae")))
        candidate_mae_series.append(_as_float(row.get("candidate_mae")))
        baseline_mean_series.append(_as_float(row.get("baseline_mean")))
        candidate_mean_series.append(_as_float(row.get("candidate_mean")))

        status_class = f"status-{status}" if status in status_counts else ""
        rows_html.append(
            "<tr>"
            f"<td>{row.get('timestamp', '')}</td>"
            f"<td><span class='pill {status_class}'>{row.get('status', '')}</span></td>"
            f"<td>{row.get('reason', '')}</td>"
            f"<td>{row.get('baseline_mean', '')}</td>"
            f"<td>{row.get('candidate_mean', '')}</td>"
            f"<td>{row.get('baseline_mae', '')}</td>"
            f"<td>{row.get('candidate_mae', '')}</td>"
            "</tr>"
        )
    decisions_table = "\n".join(rows_html) if rows_html else "<tr><td colspan='7'>No history yet</td></tr>"

    baseline_mae = _as_float(metrics.get("baseline_mae"))
    candidate_mae = _as_float(metrics.get("candidate_mae"))
    baseline_mean = _as_float(metrics.get("baseline_mean"))
    candidate_mean = _as_float(metrics.get("candidate_mean"))
    mae_delta = None if baseline_mae is None or candidate_mae is None else (candidate_mae - baseline_mae)
    mae_delta_text = "-" if mae_delta is None else (f"{mae_delta:+.4f}")

    trend_symbol = "→"
    trend_text = "stable"
    trend_color = "#64748b"
    hist_candidate_mae = [v for v in candidate_mae_series if v is not None]
    if len(hist_candidate_mae) >= 2:
        trend_delta = hist_candidate_mae[-1] - hist_candidate_mae[-2]
        if trend_delta < -1e-6:
            trend_symbol = "↓"
            trend_text = "amelioration MAE"
            trend_color = "#2e7d32"
        elif trend_delta > 1e-6:
            trend_symbol = "↑"
            trend_text = "degradation MAE"
            trend_color = "#c62828"

    min_improvement_ratio = _as_float(params.get("min_improvement_ratio"))
    target_mae = None if baseline_mae is None or min_improvement_ratio is None else (baseline_mae * min_improvement_ratio)
    proximity_pct: Optional[float] = None
    if target_mae is not None and target_mae > 0 and candidate_mae is not None and candidate_mae > 0:
        proximity_pct = max(0.0, min(100.0, (target_mae / candidate_mae) * 100.0))
    proximity_text = "-" if proximity_pct is None else f"{proximity_pct:.1f}%"
    gauge_fill = 0.0 if proximity_pct is None else proximity_pct
    if gauge_fill >= 95:
        gauge_color = "#2e7d32"
    elif gauge_fill >= 70:
        gauge_color = "#ea580c"
    else:
        gauge_color = "#c62828"

    recommendations: List[str] = []
    if decision == "accepted":
        recommendations = [
            "Conserver ce modele comme reference active pour les prochaines analyses.",
            "Continuer les corrections sur les cas limites pour renforcer la robustesse.",
            "Relancer un entrainement uniquement apres un nouveau lot significatif de corrections.",
        ]
    elif decision == "rejected":
        recommendations.append("Prioriser les corrections de faux positifs sur les images les plus surchargees.")
        if baseline_mae is not None and candidate_mae is not None and candidate_mae > baseline_mae:
            recommendations.append("Le candidat reste moins precis: ajouter des corrections cibles plutot que de gros volumes redondants.")
        if baseline_mean is not None and candidate_mean is not None:
            if candidate_mean > baseline_mean * 1.10:
                recommendations.append("Le candidat sur-detecte: augmenter legerement le seuil de confiance de detection pour les prochains tests.")
            elif candidate_mean < baseline_mean * 0.90:
                recommendations.append("Le candidat sous-detecte: ajouter des annotations sur les voitures partiellement visibles.")
        recommendations.append("Relancer l'entrainement apres un lot de corrections variees (conditions lumineuses et angles differents).")
    elif decision == "skipped":
        recommendations = [
            "Ajouter plus de corrections avant le prochain entrainement.",
            "Verifier que manual_review_annotations.json contient bien des images corrigees exploitables.",
        ]
    else:
        recommendations = [
            "Verifier les chemins de donnees et le modele de base.",
            "Consulter les logs pour identifier la cause exacte de l'echec.",
        ]

    recommendations_html = "".join(f"<li>{item}</li>" for item in recommendations)

    chart_data = {
        "labels": labels,
        "baselineMAE": baseline_mae_series,
        "candidateMAE": candidate_mae_series,
        "baselineMean": baseline_mean_series,
        "candidateMean": candidate_mean_series,
    }
    chart_json = json.dumps(chart_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang=\"fr\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Training Report</title>
    <style>
        :root {{ --bg:#f3f7fb; --card:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --ok:#2e7d32; --bad:#c62828; --skip:#1565c0; --fail:#7b1fa2; }}
        * {{ box-sizing: border-box; }}
        body {{ margin:0; font-family:Segoe UI, Arial, sans-serif; background: radial-gradient(circle at top right, #e9f2ff 0%, var(--bg) 45%); color:var(--ink); }}
        .wrap {{ max-width:1280px; margin:0 auto; padding:24px; }}
        .card {{ background:var(--card); border-radius:14px; padding:18px; margin-bottom:14px; border:1px solid #dbe3ef; box-shadow:0 8px 24px rgba(15,23,42,.05); }}
        .head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }}
        .title {{ margin:0; font-size:28px; }}
        .subtitle {{ margin:6px 0 0; color:var(--muted); }}
        .status {{ display:inline-block; padding:8px 12px; border-radius:999px; color:#fff; font-weight:700; background:{decision_color}; text-transform:uppercase; letter-spacing:.5px; }}
        .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; }}
        .kpi {{ border:1px solid var(--line); border-radius:10px; padding:10px; background:#f8fbff; }}
        .kpi .k {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
        .kpi .v {{ font-size:22px; font-weight:700; margin-top:3px; }}
        .trend {{ font-size:28px; font-weight:800; line-height:1; }}
        .gauge {{ margin-top:8px; background:#e2e8f0; border-radius:999px; height:14px; overflow:hidden; }}
        .gauge-fill {{ height:100%; width:{gauge_fill:.1f}%; background:{gauge_color}; transition:width .4s ease; }}
        .actions {{ margin:0; padding-left:20px; color:#334155; }}
        .actions li {{ margin:6px 0; }}
        .charts {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
        .chart-card {{ border:1px solid var(--line); border-radius:10px; padding:10px; background:#fbfdff; }}
        .chart-title {{ font-size:14px; margin:0 0 6px; color:#334155; }}
        svg {{ width:100%; height:210px; display:block; background:#fff; border:1px solid #edf2f7; border-radius:8px; }}
        .legend {{ display:flex; gap:10px; margin-top:8px; font-size:12px; color:#475569; }}
        .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:4px; }}
        table {{ width:100%; border-collapse:collapse; }}
        th, td {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; font-size:12px; }}
        th {{ background:#f8fafc; color:#334155; }}
        .pill {{ display:inline-block; padding:3px 8px; border-radius:999px; font-size:11px; font-weight:700; text-transform:uppercase; }}
        .status-accepted {{ background:#dcfce7; color:#166534; }}
        .status-rejected {{ background:#fee2e2; color:#991b1b; }}
        .status-skipped {{ background:#dbeafe; color:#1d4ed8; }}
        .status-failed {{ background:#f3e8ff; color:#6b21a8; }}
        @media (max-width: 900px) {{ .charts {{ grid-template-columns:1fr; }} }}
    </style>
</head>
<body>
    <div class=\"wrap\">
        <div class=\"card\">
            <div class=\"head\">
                <div>
                    <h1 class=\"title\">Compte rendu entrainement YOLO</h1>
                    <p class=\"subtitle\">Run: {payload.get('run_started', '')} | Decision reason: {payload.get('decision_reason', '')}</p>
                </div>
                <div class=\"status\">{decision}</div>
            </div>
        </div>

        <div class=\"card\">
            <div class=\"kpis\">
                <div class=\"kpi\"><div class=\"k\">Train images</div><div class=\"v\">{dataset.get('train_images', '-')}</div></div>
                <div class=\"kpi\"><div class=\"k\">Val images</div><div class=\"v\">{dataset.get('val_images', '-')}</div></div>
                <div class=\"kpi\"><div class=\"k\">Total boxes</div><div class=\"v\">{dataset.get('total_boxes', '-')}</div></div>
                <div class=\"kpi\"><div class=\"k\">Baseline MAE</div><div class=\"v\">{_safe_float(metrics.get('baseline_mae'))}</div></div>
                <div class=\"kpi\"><div class=\"k\">Candidate MAE</div><div class=\"v\">{_safe_float(metrics.get('candidate_mae'))}</div></div>
                <div class=\"kpi\"><div class=\"k\">Delta MAE (cand-base)</div><div class=\"v\">{mae_delta_text}</div></div>
            </div>
        </div>

        <div class=\"card\">
            <div class=\"kpis\">
                <div class=\"kpi\">
                    <div class=\"k\">Tendance MAE (historique recent)</div>
                    <div class=\"trend\" style=\"color:{trend_color}\">{trend_symbol}</div>
                    <div class=\"v\" style=\"font-size:16px; color:{trend_color}\">{trend_text}</div>
                </div>
                <div class=\"kpi\">
                    <div class=\"k\">Proximite d'acceptation (MAE)</div>
                    <div class=\"v\">{proximity_text}</div>
                    <div class=\"gauge\"><div class=\"gauge-fill\"></div></div>
                    <div style=\"margin-top:6px; color:#64748b; font-size:12px;\">Cible MAE: {_safe_float(target_mae)} | Candidat: {_safe_float(candidate_mae)}</div>
                </div>
            </div>
        </div>

        <div class=\"card\">
            <div class=\"kpis\">
                <div class=\"kpi\"><div class=\"k\">Accepted (history)</div><div class=\"v\">{status_counts['accepted']}</div></div>
                <div class=\"kpi\"><div class=\"k\">Rejected (history)</div><div class=\"v\">{status_counts['rejected']}</div></div>
                <div class=\"kpi\"><div class=\"k\">Skipped (history)</div><div class=\"v\">{status_counts['skipped']}</div></div>
                <div class=\"kpi\"><div class=\"k\">Failed (history)</div><div class=\"v\">{status_counts['failed']}</div></div>
            </div>
        </div>

        <div class=\"card\">
            <div class=\"charts\">
                <div class=\"chart-card\">
                    <h3 class=\"chart-title\">Evolution MAE (baseline vs candidate)</h3>
                    <svg id=\"chartMae\"></svg>
                    <div class=\"legend\">
                        <span><span class=\"dot\" style=\"background:#2563eb\"></span>Baseline MAE</span>
                        <span><span class=\"dot\" style=\"background:#dc2626\"></span>Candidate MAE</span>
                    </div>
                </div>
                <div class=\"chart-card\">
                    <h3 class=\"chart-title\">Evolution Mean Count (baseline vs candidate)</h3>
                    <svg id=\"chartMean\"></svg>
                    <div class=\"legend\">
                        <span><span class=\"dot\" style=\"background:#0f766e\"></span>Baseline mean</span>
                        <span><span class=\"dot\" style=\"background:#ea580c\"></span>Candidate mean</span>
                    </div>
                </div>
            </div>
        </div>

        <div class=\"card\">
            <h3 style=\"margin-top:0\">Historique recent des decisions</h3>
            <table>
                <thead><tr><th>timestamp</th><th>status</th><th>reason</th><th>baseline_mean</th><th>candidate_mean</th><th>baseline_mae</th><th>candidate_mae</th></tr></thead>
                <tbody>{decisions_table}</tbody>
            </table>
        </div>

        <div class=\"card\">
            <h3 style=\"margin-top:0\">Parametres d'entrainement</h3>
            <div class=\"kpis\">
                <div class=\"kpi\"><div class=\"k\">epochs</div><div class=\"v\">{params.get('epochs', '-')}</div></div>
                <div class=\"kpi\"><div class=\"k\">batch</div><div class=\"v\">{params.get('batch', '-')}</div></div>
                <div class=\"kpi\"><div class=\"k\">imgsz</div><div class=\"v\">{params.get('imgsz', '-')}</div></div>
                <div class=\"kpi\"><div class=\"k\">lr0</div><div class=\"v\">{params.get('lr0', '-')}</div></div>
                <div class=\"kpi\"><div class=\"k\">freeze</div><div class=\"v\">{params.get('freeze', '-')}</div></div>
                <div class=\"kpi\"><div class=\"k\">min_improvement_ratio</div><div class=\"v\">{params.get('min_improvement_ratio', '-')}</div></div>
            </div>
        </div>

        <div class=\"card\">
            <h3 style=\"margin-top:0\">Actions recommandees</h3>
            <ul class=\"actions\">
                {recommendations_html}
            </ul>
        </div>
    </div>

    <script>
        const DATA = {chart_json};

        function drawLineChart(svgId, labels, s1, s2, color1, color2) {{
            const svg = document.getElementById(svgId);
            if (!svg) return;
            const width = svg.clientWidth || 560;
            const height = svg.clientHeight || 210;
            const pad = 30;
            const usableW = width - pad * 2;
            const usableH = height - pad * 2;

            const merged = [];
            for (let i = 0; i < Math.max(s1.length, s2.length); i++) {{
                if (typeof s1[i] === 'number') merged.push(s1[i]);
                if (typeof s2[i] === 'number') merged.push(s2[i]);
            }}

            if (!merged.length) {{
                svg.innerHTML = `<text x="${{width/2}}" y="${{height/2}}" text-anchor="middle" fill="#94a3b8">Pas de donnees</text>`;
                return;
            }}

            const minY = Math.min(...merged);
            const maxY = Math.max(...merged);
            const span = Math.max(1e-6, maxY - minY);
            const toX = (i, n) => pad + (usableW * i / Math.max(1, n - 1));
            const toY = (v) => height - pad - ((v - minY) / span) * usableH;

            function pathFor(series) {{
                let p = '';
                let started = false;
                for (let i = 0; i < series.length; i++) {{
                    const v = series[i];
                    if (typeof v !== 'number') continue;
                    const x = toX(i, series.length);
                    const y = toY(v);
                    if (!started) {{ p += `M ${{x}} ${{y}}`; started = true; }}
                    else {{ p += ` L ${{x}} ${{y}}`; }}
                }}
                return p;
            }}

            const p1 = pathFor(s1);
            const p2 = pathFor(s2);
            svg.innerHTML = `
                <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="#ffffff"/>
                <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height-pad}}" stroke="#cbd5e1" />
                <line x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}" stroke="#cbd5e1" />
                <path d="${{p1}}" fill="none" stroke="${{color1}}" stroke-width="2" />
                <path d="${{p2}}" fill="none" stroke="${{color2}}" stroke-width="2" />
                <text x="${{pad}}" y="${{pad-8}}" fill="#64748b" font-size="11">min=${{minY.toFixed(2)}} | max=${{maxY.toFixed(2)}}</text>
            `;
        }}

        drawLineChart('chartMae', DATA.labels || [], DATA.baselineMAE || [], DATA.candidateMAE || [], '#2563eb', '#dc2626');
        drawLineChart('chartMean', DATA.labels || [], DATA.baselineMean || [], DATA.candidateMean || [], '#0f766e', '#ea580c');
    </script>
</body>
</html>
"""

    with open(report_html, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    args = parse_args()

    annotations_path = Path(args.annotations_json)
    data_dir = Path(args.data_dir)
    base_model = Path(args.base_model)
    output_model = Path(args.output_model)
    dataset_dir = Path(args.dataset_dir)
    reviewed_csv = Path(args.reviewed_csv)
    decisions_csv = Path(args.decisions_csv)
    report_json = Path(args.report_json)
    report_html = Path(args.report_html)
    deployed_model_source = output_model if output_model.exists() else base_model

    run_started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def emit_report(
        decision_status: str,
        decision_reason: str,
        train_images: int = 0,
        val_images: int = 0,
        total_boxes: int = 0,
        baseline_mean: Optional[float] = None,
        candidate_mean: Optional[float] = None,
        baseline_mae: Optional[float] = None,
        candidate_mae: Optional[float] = None,
        n_eval_base: Optional[int] = None,
        n_eval_candidate: Optional[int] = None,
    ) -> None:
        payload: Dict[str, object] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_started": run_started,
            "decision_status": decision_status,
            "decision_reason": decision_reason,
            "dataset": {
                "train_images": train_images,
                "val_images": val_images,
                "total_boxes": total_boxes,
                "dataset_dir": str(dataset_dir),
            },
            "metrics": {
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "baseline_mae": baseline_mae,
                "candidate_mae": candidate_mae,
                "n_eval_base": n_eval_base,
                "n_eval_candidate": n_eval_candidate,
            },
            "parameters": {
                "base_model": str(base_model),
                "output_model": str(output_model),
                "reference_model": str(deployed_model_source),
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "lr0": args.lr0,
                "freeze": args.freeze,
                "patience": args.patience,
                "close_mosaic": args.close_mosaic,
                "conf": args.conf,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "min_improvement_ratio": args.min_improvement_ratio,
            },
        }
        recent = load_recent_decisions(decisions_csv, limit=15)
        write_training_report(report_json, report_html, payload, recent)

    annotations = load_annotations(annotations_path)
    if not annotations:
        if deployed_model_source.exists():
            copy_model(deployed_model_source, output_model)
            print(f"Aucune annotation de correction trouvee, modele recopie: {output_model}")
            append_decision_log(
                decisions_csv,
                status="skipped",
                reason="no_annotations",
                output_model=output_model,
                min_improvement_ratio=args.min_improvement_ratio,
            )
            emit_report(decision_status="skipped", decision_reason="no_annotations")
        else:
            print("Aucune annotation de correction trouvee et aucun modele de reference introuvable.")
            append_decision_log(
                decisions_csv,
                status="failed",
                reason="no_annotations_and_no_reference_model",
                output_model=output_model,
                min_improvement_ratio=args.min_improvement_ratio,
            )
            emit_report(decision_status="failed", decision_reason="no_annotations_and_no_reference_model")
        return

    if not data_dir.exists():
        print(f"Dossier DATA introuvable: {data_dir}")
        append_decision_log(
            decisions_csv,
            status="failed",
            reason="data_dir_not_found",
            output_model=output_model,
            min_improvement_ratio=args.min_improvement_ratio,
        )
        emit_report(decision_status="failed", decision_reason="data_dir_not_found")
        return

    ensure_dir(dataset_dir)
    if not deployed_model_source.exists():
        print(f"Modele source introuvable: {deployed_model_source}")
        append_decision_log(
            decisions_csv,
            status="failed",
            reason="reference_model_not_found",
            output_model=output_model,
            min_improvement_ratio=args.min_improvement_ratio,
        )
        emit_report(decision_status="failed", decision_reason="reference_model_not_found")
        return

    print(f"Chargement du modele de reference: {deployed_model_source}")
    model = YOLO(str(deployed_model_source))

    train_count, val_count, total_boxes, yaml_path, probe_images = build_dataset(
        annotations=annotations,
        data_dir=data_dir,
        dataset_dir=dataset_dir,
        model=model,
        conf=args.conf,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if train_count + val_count == 0:
        copy_model(deployed_model_source, output_model)
        print("Aucune image exploitable pour l'entrainement, modele conserve.")
        append_decision_log(
            decisions_csv,
            status="skipped",
            reason="no_usable_images",
            output_model=output_model,
            train_images=train_count,
            val_images=val_count,
            total_boxes=total_boxes,
            min_improvement_ratio=args.min_improvement_ratio,
        )
        emit_report(
            decision_status="skipped",
            decision_reason="no_usable_images",
            train_images=train_count,
            val_images=val_count,
            total_boxes=total_boxes,
        )
        return

    if (train_count + val_count) < MIN_IMAGES_FOR_TRAINING or total_boxes < MIN_BOXES_FOR_TRAINING:
        copy_model(deployed_model_source, output_model)
        print(
            "Pas assez de donnees corrigees pour un entrainement stable "
            f"(images={train_count + val_count}, boxes={total_boxes}). Modele conserve."
        )
        append_decision_log(
            decisions_csv,
            status="skipped",
            reason="insufficient_training_data",
            output_model=output_model,
            train_images=train_count,
            val_images=val_count,
            total_boxes=total_boxes,
            min_improvement_ratio=args.min_improvement_ratio,
        )
        emit_report(
            decision_status="skipped",
            decision_reason="insufficient_training_data",
            train_images=train_count,
            val_images=val_count,
            total_boxes=total_boxes,
        )
        return

    print(f"Dataset exporte: train={train_count} | val={val_count} | boxes={total_boxes}")
    print(f"Data YAML: {yaml_path}")

    try:
        model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            freeze=args.freeze,
            close_mosaic=args.close_mosaic,
            patience=max(1, min(args.patience, args.epochs)),
            cos_lr=True,
            project=str(BASE_DIR / "training_runs"),
            name="parking_corrections",
            exist_ok=True,
            verbose=False,
        )
        save_dir_value = getattr(getattr(model, "trainer", None), "save_dir", None)
        best_path = Path(save_dir_value) / "weights" / "best.pt" if save_dir_value else None
        if best_path is not None and best_path.exists():
            baseline_mean = mean_car_count(deployed_model_source, probe_images, conf=args.conf)
            candidate_mean = mean_car_count(best_path, probe_images, conf=args.conf)
            baseline_mae, n_eval_base = mae_on_reviewed(deployed_model_source, reviewed_csv, data_dir, conf=args.conf)
            candidate_mae, n_eval_candidate = mae_on_reviewed(best_path, reviewed_csv, data_dir, conf=args.conf)

            collapse_detected = baseline_mean > 0 and candidate_mean < baseline_mean * 0.4
            regression_detected = (
                n_eval_base >= 5
                and n_eval_candidate >= 5
                and candidate_mae > baseline_mae * args.min_improvement_ratio
            )

            if collapse_detected or regression_detected:
                copy_model(deployed_model_source, output_model)
                print(
                    "Modele fine-tune rejete (anti-regression): "
                    f"count baseline={baseline_mean:.2f}, candidate={candidate_mean:.2f}; "
                    f"MAE baseline={baseline_mae:.2f} (n={n_eval_base}), "
                    f"candidate={candidate_mae:.2f} (n={n_eval_candidate}). "
                    "Modele de reference restaure."
                )
                append_decision_log(
                    decisions_csv,
                    status="rejected",
                    reason="regression_guard",
                    output_model=output_model,
                    train_images=train_count,
                    val_images=val_count,
                    total_boxes=total_boxes,
                    baseline_mean=baseline_mean,
                    candidate_mean=candidate_mean,
                    baseline_mae=baseline_mae,
                    candidate_mae=candidate_mae,
                    n_eval_base=n_eval_base,
                    n_eval_candidate=n_eval_candidate,
                    min_improvement_ratio=args.min_improvement_ratio,
                )
                emit_report(
                    decision_status="rejected",
                    decision_reason="regression_guard",
                    train_images=train_count,
                    val_images=val_count,
                    total_boxes=total_boxes,
                    baseline_mean=baseline_mean,
                    candidate_mean=candidate_mean,
                    baseline_mae=baseline_mae,
                    candidate_mae=candidate_mae,
                    n_eval_base=n_eval_base,
                    n_eval_candidate=n_eval_candidate,
                )
            else:
                copy_model(best_path, output_model)
                print(
                    f"Modele fine-tune sauvegarde: {output_model} "
                    f"(count baseline={baseline_mean:.2f}, candidate={candidate_mean:.2f}; "
                    f"MAE baseline={baseline_mae:.2f} n={n_eval_base}, "
                    f"candidate={candidate_mae:.2f} n={n_eval_candidate})"
                )
                append_decision_log(
                    decisions_csv,
                    status="accepted",
                    reason="passed_guards",
                    output_model=output_model,
                    train_images=train_count,
                    val_images=val_count,
                    total_boxes=total_boxes,
                    baseline_mean=baseline_mean,
                    candidate_mean=candidate_mean,
                    baseline_mae=baseline_mae,
                    candidate_mae=candidate_mae,
                    n_eval_base=n_eval_base,
                    n_eval_candidate=n_eval_candidate,
                    min_improvement_ratio=args.min_improvement_ratio,
                )
                emit_report(
                    decision_status="accepted",
                    decision_reason="passed_guards",
                    train_images=train_count,
                    val_images=val_count,
                    total_boxes=total_boxes,
                    baseline_mean=baseline_mean,
                    candidate_mean=candidate_mean,
                    baseline_mae=baseline_mae,
                    candidate_mae=candidate_mae,
                    n_eval_base=n_eval_base,
                    n_eval_candidate=n_eval_candidate,
                )
        else:
            copy_model(deployed_model_source, output_model)
            print("Entrainement termine mais best.pt introuvable, modele precedent conserve.")
            append_decision_log(
                decisions_csv,
                status="failed",
                reason="best_checkpoint_missing",
                output_model=output_model,
                train_images=train_count,
                val_images=val_count,
                total_boxes=total_boxes,
                min_improvement_ratio=args.min_improvement_ratio,
            )
            emit_report(
                decision_status="failed",
                decision_reason="best_checkpoint_missing",
                train_images=train_count,
                val_images=val_count,
                total_boxes=total_boxes,
            )
    except Exception as exc:
        copy_model(deployed_model_source, output_model)
        print(f"Echec de l'entrainement, modele precedent conserve. Detail: {exc}")
        append_decision_log(
            decisions_csv,
            status="failed",
            reason=f"exception:{exc}",
            output_model=output_model,
            train_images=train_count,
            val_images=val_count,
            total_boxes=total_boxes,
            min_improvement_ratio=args.min_improvement_ratio,
        )
        emit_report(
            decision_status="failed",
            decision_reason=f"exception:{exc}",
            train_images=train_count,
            val_images=val_count,
            total_boxes=total_boxes,
        )


if __name__ == "__main__":
    main()
