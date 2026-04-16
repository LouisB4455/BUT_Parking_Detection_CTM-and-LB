#!/usr/bin/env python3
"""Run a compact 3-run hyperparameter grid and rank experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3 YOLO training experiments and rank results")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--project", default="training_runs")
    parser.add_argument("--prefix", default="grid3")
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--summary-csv", default="training_grid3_summary.csv")
    parser.add_argument("--best-json", default="training_grid3_best.json")
    return parser.parse_args()


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _read_last_row(results_csv: Path) -> dict[str, str]:
    last: dict[str, str] | None = None
    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                last = row
    return last or {}


def _read_run_report(report_json: Path) -> dict[str, float | int | None]:
    if not report_json.exists():
        return {}

    try:
        payload = json.loads(report_json.read_text(encoding="utf-8"))
    except Exception:
        return {}

    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    if not isinstance(metrics, dict):
        return {}

    return {
        "val_map50_95": _to_float(str(metrics.get("val_map50_95", 0))),
        "val_precision": _to_float(str(metrics.get("val_precision", 0))),
        "val_recall": _to_float(str(metrics.get("val_recall", 0))),
        "val_box_loss": _to_float(str(metrics.get("val_box_loss", 0))),
    }


def _run_one(train_script: Path, params: dict[str, str], device: str, workers: int) -> None:
    cmd = [
        sys.executable,
        str(train_script),
        "--data",
        params["data"],
        "--weights",
        params["weights"],
        "--epochs",
        params["epochs"],
        "--imgsz",
        params["imgsz"],
        "--batch",
        params["batch"],
        "--patience",
        params["patience"],
        "--project",
        params["project"],
        "--name",
        params["name"],
        "--workers",
        str(workers),
        "--report-json",
        params["report_json"],
        "--report-html",
        params["report_html"],
        "--publish-model-path",
        "",
    ]
    if device:
        cmd.extend(["--device", device])

    print("\n=== RUN", params["name"], "===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Training failed for run: {params['name']}")


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset yaml not found: {data_yaml}")

    base = {
        "data": str(data_yaml),
        "project": args.project,
        "patience": "25",
    }

    runs = [
        {
            **base,
            "name": f"{args.prefix}_n_640_b16",
            "weights": "yolov8n.pt",
            "epochs": "100",
            "imgsz": "640",
            "batch": "16",
        },
        {
            **base,
            "name": f"{args.prefix}_s_640_b16",
            "weights": "yolov8s.pt",
            "epochs": "120",
            "imgsz": "640",
            "batch": "16",
        },
        {
            **base,
            "name": f"{args.prefix}_s_832_b8",
            "weights": "yolov8s.pt",
            "epochs": "120",
            "imgsz": "832",
            "batch": "8",
        },
    ]

    train_script = Path(__file__).resolve().parents[2] / "modèle final" / "train_batch_yolo.py"

    for run in runs:
        run["report_json"] = f"{run['name']}_report.json"
        run["report_html"] = f"{run['name']}_report.html"
        _run_one(train_script, run, device=args.device, workers=args.workers)

    scored: list[dict[str, object]] = []
    for run in runs:
        report_json = Path(run["report_json"])
        last = _read_run_report(report_json)
        if not last:
            continue

        val_map = float(last.get("val_map50_95", 0.0) or 0.0)
        val_precision = float(last.get("val_precision", 0.0) or 0.0)
        val_recall = float(last.get("val_recall", 0.0) or 0.0)
        val_box_loss = float(last.get("val_box_loss", 0.0) or 0.0)

        score = (0.7 * val_map) + (0.2 * val_precision) + (0.1 * val_recall) - (0.05 * val_box_loss)

        scored.append(
            {
                "run": run["name"],
                "weights": run["weights"],
                "epochs": int(run["epochs"]),
                "imgsz": int(run["imgsz"]),
                "batch": int(run["batch"]),
                "val_map50_95": val_map,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_box_loss": val_box_loss,
                "score": score,
            }
        )

    scored.sort(key=lambda x: float(x["score"]), reverse=True)

    summary_csv = Path(args.summary_csv).resolve()
    if not scored:
        print("Aucune métrique exploitable trouvée. Vérifie les fichiers *_report.json.")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "run",
            "weights",
            "epochs",
            "imgsz",
            "batch",
            "val_map50_95",
            "val_precision",
            "val_recall",
            "val_box_loss",
            "score",
        ])
        for i, row in enumerate(scored, start=1):
            writer.writerow(
                [
                    i,
                    row["run"],
                    row["weights"],
                    row["epochs"],
                    row["imgsz"],
                    row["batch"],
                    f"{float(row['val_map50_95']):.6f}",
                    f"{float(row['val_precision']):.6f}",
                    f"{float(row['val_recall']):.6f}",
                    f"{float(row['val_box_loss']):.6f}",
                    f"{float(row['score']):.6f}",
                ]
            )

    best = scored[0] if scored else {}
    best_json = Path(args.best_json).resolve()
    best_json.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nGrid 3 runs complete")
    print(f"- summary: {summary_csv}")
    print(f"- best: {best_json}")
    if best:
        print(f"- winner: {best['run']} (score={float(best['score']):.4f}, mAP50-95={float(best['val_map50_95']):.4f})")


if __name__ == "__main__":
    main()
