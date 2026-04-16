#!/usr/bin/env python3
"""Train a YOLO model on a prepared dataset using transfer learning."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch YOLO training")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--weights", default="yolov8n.pt", help="Initial transfer-learning weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="", help="Examples: cpu, 0, 0,1")
    parser.add_argument("--project", default="training_runs")
    parser.add_argument("--name", default="batch_yolo")
    parser.add_argument(
        "--publish-model-path",
        default="parking_detector_corrections.pt",
        help="Where to copy the best checkpoint after training",
    )
    parser.add_argument("--report-json", default="training_batch_last_report.json")
    parser.add_argument("--report-html", default="training_batch_last_report.html")
    return parser.parse_args()


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _read_last_metrics(results_csv: Path) -> dict[str, float | int | None]:
    if not results_csv.exists():
        return {}

    last_row: dict[str, str] | None = None
    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                last_row = row

    if not last_row:
        return {}

    return {
        "epoch": int(float(last_row.get("epoch", "0") or 0)),
        "val_precision": _to_float(last_row.get("metrics/precision(B)", "")),
        "val_recall": _to_float(last_row.get("metrics/recall(B)", "")),
        "val_map50": _to_float(last_row.get("metrics/mAP50(B)", "")),
        "val_map50_95": _to_float(last_row.get("metrics/mAP50-95(B)", "")),
        "train_box_loss": _to_float(last_row.get("train/box_loss", "")),
        "train_cls_loss": _to_float(last_row.get("train/cls_loss", "")),
        "train_dfl_loss": _to_float(last_row.get("train/dfl_loss", "")),
        "val_box_loss": _to_float(last_row.get("val/box_loss", "")),
        "val_cls_loss": _to_float(last_row.get("val/cls_loss", "")),
        "val_dfl_loss": _to_float(last_row.get("val/dfl_loss", "")),
    }


def _extract_split_metrics(eval_results: object) -> dict[str, float | None]:
    box = getattr(eval_results, "box", None)
    if box is None:
        return {
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
        }

    return {
        "precision": _to_float(str(getattr(box, "mp", ""))),
        "recall": _to_float(str(getattr(box, "mr", ""))),
        "map50": _to_float(str(getattr(box, "map50", ""))),
        "map50_95": _to_float(str(getattr(box, "map", ""))),
    }


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _write_reports(
    report_json: Path,
    report_html: Path,
    run_dir: Path,
    data_yaml: Path,
    weights_path: Path,
    publish_model_path: Path | None,
    replaced_previous_model: bool,
    metrics: dict[str, float | int | None],
) -> None:
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "batch_training",
        "run_dir": str(run_dir),
        "data_yaml": str(data_yaml),
        "best_weights": str(weights_path),
        "publish_model_path": str(publish_model_path) if publish_model_path else None,
        "replaced_previous_model": replaced_previous_model,
        "metrics": metrics,
    }
    report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    html = f"""<!doctype html>
<html lang=\"fr\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Rapport Batch Training YOLO</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f5f7fb; color: #1f2937; }}
    .card {{ background: white; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 4px 14px rgba(0,0,0,0.08); }}
    h1 {{ margin: 0 0 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
    .metric {{ padding: 10px 12px; border-radius: 8px; background: #eef2ff; }}
    .label {{ font-size: 12px; color: #374151; }}
    .value {{ font-size: 20px; font-weight: 700; margin-top: 3px; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>Rapport Batch Training YOLO</h1>
    <div>Généré le: <strong>{payload['generated_at']}</strong></div>
    <div>Run: <code>{run_dir}</code></div>
    <div>Dataset: <code>{data_yaml}</code></div>
    <div>Poids best: <code>{weights_path}</code></div>
        <div>Modèle publié: <code>{publish_model_path if publish_model_path else 'N/A'}</code></div>
        <div>Ancien modèle remplacé: <strong>{'oui' if replaced_previous_model else 'non'}</strong></div>
  </div>

  <div class=\"card\">
    <h2>Indices Qualité (Train / Validation)</h2>
    <div class=\"grid\">
      <div class=\"metric\"><div class=\"label\">Epoch final</div><div class=\"value\">{_fmt(metrics.get('epoch'), 0)}</div></div>
      <div class=\"metric\"><div class=\"label\">Précision train</div><div class=\"value\">{_fmt(metrics.get('train_precision'))}</div></div>
      <div class=\"metric\"><div class=\"label\">Précision val</div><div class=\"value\">{_fmt(metrics.get('val_precision'))}</div></div>
      <div class=\"metric\"><div class=\"label\">Rappel train</div><div class=\"value\">{_fmt(metrics.get('train_recall'))}</div></div>
      <div class=\"metric\"><div class=\"label\">Rappel val</div><div class=\"value\">{_fmt(metrics.get('val_recall'))}</div></div>
      <div class=\"metric\"><div class=\"label\">mAP50 train</div><div class=\"value\">{_fmt(metrics.get('train_map50'))}</div></div>
      <div class=\"metric\"><div class=\"label\">mAP50 val</div><div class=\"value\">{_fmt(metrics.get('val_map50'))}</div></div>
      <div class=\"metric\"><div class=\"label\">mAP50-95 train</div><div class=\"value\">{_fmt(metrics.get('train_map50_95'))}</div></div>
      <div class=\"metric\"><div class=\"label\">mAP50-95 val</div><div class=\"value\">{_fmt(metrics.get('val_map50_95'))}</div></div>
    </div>
  </div>

  <div class=\"card\">
    <h2>Indices Losses</h2>
    <div class=\"grid\">
      <div class=\"metric\"><div class=\"label\">train box loss</div><div class=\"value\">{_fmt(metrics.get('train_box_loss'))}</div></div>
      <div class=\"metric\"><div class=\"label\">train cls loss</div><div class=\"value\">{_fmt(metrics.get('train_cls_loss'))}</div></div>
      <div class=\"metric\"><div class=\"label\">train dfl loss</div><div class=\"value\">{_fmt(metrics.get('train_dfl_loss'))}</div></div>
      <div class=\"metric\"><div class=\"label\">val box loss</div><div class=\"value\">{_fmt(metrics.get('val_box_loss'))}</div></div>
      <div class=\"metric\"><div class=\"label\">val cls loss</div><div class=\"value\">{_fmt(metrics.get('val_cls_loss'))}</div></div>
      <div class=\"metric\"><div class=\"label\">val dfl loss</div><div class=\"value\">{_fmt(metrics.get('val_dfl_loss'))}</div></div>
    </div>
  </div>
</body>
</html>
"""
    report_html.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset yaml not found: {data_yaml}")

    model = YOLO(args.weights)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        device=args.device if args.device else None,
    )

    run_dir = Path(results.save_dir).resolve()
    best_weights = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"
    metrics = _read_last_metrics(results_csv)

    publish_model_path = Path(args.publish_model_path).resolve() if args.publish_model_path else None
    replaced_previous_model = False
    if publish_model_path is not None:
        previous_exists = publish_model_path.exists()
        publish_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, publish_model_path)
        replaced_previous_model = previous_exists

    best_model = YOLO(str(best_weights))
    train_eval = best_model.val(data=str(data_yaml), split="train", verbose=False)
    val_eval = best_model.val(data=str(data_yaml), split="val", verbose=False)

    train_split_metrics = _extract_split_metrics(train_eval)
    val_split_metrics = _extract_split_metrics(val_eval)

    metrics.update(
        {
            "train_precision": train_split_metrics["precision"],
            "train_recall": train_split_metrics["recall"],
            "train_map50": train_split_metrics["map50"],
            "train_map50_95": train_split_metrics["map50_95"],
            "val_precision": val_split_metrics["precision"] if val_split_metrics["precision"] is not None else metrics.get("val_precision"),
            "val_recall": val_split_metrics["recall"] if val_split_metrics["recall"] is not None else metrics.get("val_recall"),
            "val_map50": val_split_metrics["map50"] if val_split_metrics["map50"] is not None else metrics.get("val_map50"),
            "val_map50_95": val_split_metrics["map50_95"] if val_split_metrics["map50_95"] is not None else metrics.get("val_map50_95"),
        }
    )

    report_json = Path(args.report_json).resolve()
    report_html = Path(args.report_html).resolve()

    _write_reports(
        report_json=report_json,
        report_html=report_html,
        run_dir=run_dir,
        data_yaml=data_yaml,
        weights_path=best_weights,
        publish_model_path=publish_model_path,
        replaced_previous_model=replaced_previous_model,
        metrics=metrics,
    )

    print("Training complete")
    print(f"- run name: {args.name}")
    print(f"- project dir: {Path(args.project).resolve()}")
    print(f"- best checkpoint: {best_weights}")
    print(f"- published model: {publish_model_path if publish_model_path else 'N/A'}")
    print(f"- replaced previous model: {'yes' if replaced_previous_model else 'no'}")
    print(f"- last checkpoint: {run_dir / 'weights' / 'last.pt'}")
    print(f"- report json: {report_json}")
    print(f"- report html: {report_html}")
    print(f"- precision train/val: {_fmt(metrics.get('train_precision'))} / {_fmt(metrics.get('val_precision'))}")
    print(f"- losses train_box/val_box: {_fmt(metrics.get('train_box_loss'))} / {_fmt(metrics.get('val_box_loss'))}")


if __name__ == "__main__":
    main()
