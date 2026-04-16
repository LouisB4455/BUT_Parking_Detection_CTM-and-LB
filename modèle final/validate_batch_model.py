#!/usr/bin/env python3
"""Run batch inference on unknown images and save annotated outputs + summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch validation on unknown images")
    parser.add_argument("--model", required=True, help="Path to trained YOLO weights")
    parser.add_argument("--images-dir", required=True, help="Folder of unknown images")
    parser.add_argument("--output-dir", default="validation_outputs", help="Where to save annotated images and CSV")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--target-class",
        default="car",
        help="Class name to count in summary (set empty string to count all classes)",
    )
    return parser.parse_args()


def iter_images(images_dir: Path):
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def resolve_target_class_id(model: YOLO, target_class: str) -> int | None:
    if not target_class:
        return None

    names = model.names
    if isinstance(names, dict):
        for cls_id, cls_name in names.items():
            if str(cls_name).strip().lower() == target_class.strip().lower():
                return int(cls_id)
    elif isinstance(names, list):
        for cls_id, cls_name in enumerate(names):
            if str(cls_name).strip().lower() == target_class.strip().lower():
                return cls_id

    return None


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).resolve()
    images_dir = Path(args.images_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    model = YOLO(str(model_path))
    target_class_id = resolve_target_class_id(model, args.target_class)

    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[list[str]] = []

    image_files = sorted(iter_images(images_dir))
    if not image_files:
        raise RuntimeError("No image found in --images-dir")

    for idx, img_path in enumerate(image_files, start=1):
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )
        result = results[0]

        rel_path = img_path.relative_to(images_dir)
        out_img = annotated_dir / rel_path
        out_img.parent.mkdir(parents=True, exist_ok=True)

        plotted = result.plot()
        cv2.imwrite(str(out_img), plotted)

        count = 0
        conf_mean = 0.0

        if result.boxes is not None and result.boxes.cls is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []

            selected = []
            if target_class_id is None:
                selected = list(range(len(classes)))
            else:
                selected = [i for i, cls_id in enumerate(classes) if cls_id == target_class_id]

            count = len(selected)
            if selected and len(confs) > 0:
                conf_mean = float(sum(float(confs[i]) for i in selected) / len(selected))

        summary_rows.append([rel_path.as_posix(), str(count), f"{conf_mean:.4f}"])
        print(f"[{idx}/{len(image_files)}] {rel_path.as_posix()} -> count={count}")

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "detected_count", "mean_confidence"])
        writer.writerows(summary_rows)

    print("Validation batch complete")
    print(f"- annotated images: {annotated_dir}")
    print(f"- summary csv: {csv_path}")


if __name__ == "__main__":
    main()
