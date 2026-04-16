#!/usr/bin/env python3
"""Prepare a YOLO dataset split (train/val/test) from images and labels folders."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Pair:
    image_path: Path
    label_path: Path
    rel_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO train/val/test folders and dataset.yaml")
    parser.add_argument("--images-dir", required=True, help="Source folder containing all images")
    parser.add_argument("--labels-dir", required=True, help="Source folder containing YOLO .txt labels")
    parser.add_argument("--output-dir", default="batch_dataset", help="Output dataset root folder")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-names",
        default="car",
        help="Comma-separated class names used by YOLO dataset.yaml (example: car,bus,truck)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if an image has no matching label file",
    )
    return parser.parse_args()


def iter_images(images_dir: Path) -> Iterable[Path]:
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def find_label_path(image_path: Path, rel_path: Path, labels_dir: Path) -> Path | None:
    candidate_relative = (labels_dir / rel_path).with_suffix(".txt")
    if candidate_relative.exists():
        return candidate_relative

    candidate_flat = labels_dir / f"{image_path.stem}.txt"
    if candidate_flat.exists():
        return candidate_flat

    return None


def split_items(items: list[Pair], train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[list[Pair], list[Pair], list[Pair]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("Ratios must sum to > 0")

    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:n_train + n_val + n_test]
    return train_items, val_items, test_items


def copy_split(items: list[Pair], output_dir: Path, split: str) -> int:
    count = 0
    for pair in items:
        out_img = output_dir / "images" / split / pair.rel_path
        out_lbl = output_dir / "labels" / split / pair.rel_path.with_suffix(".txt")

        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(pair.image_path, out_img)
        shutil.copy2(pair.label_path, out_lbl)
        count += 1
    return count


def write_dataset_yaml(output_dir: Path, class_names: list[str]) -> Path:
    yaml_path = output_dir / "dataset.yaml"
    names_yaml = ", ".join(f'"{name}"' for name in class_names)

    content = "\n".join(
        [
            f"path: {output_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(class_names)}",
            f"names: [{names_yaml}]",
            "",
        ]
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

    pairs: list[Pair] = []
    missing_labels: list[Path] = []

    for image_path in iter_images(images_dir):
        rel_path = image_path.relative_to(images_dir)
        label_path = find_label_path(image_path, rel_path, labels_dir)
        if label_path is None:
            missing_labels.append(rel_path)
            continue
        pairs.append(Pair(image_path=image_path, label_path=label_path, rel_path=rel_path))

    if args.strict and missing_labels:
        sample = "\n".join(str(p) for p in missing_labels[:20])
        raise RuntimeError(f"Missing labels for {len(missing_labels)} images. Sample:\n{sample}")

    if not pairs:
        raise RuntimeError("No valid image/label pair found")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    train_items, val_items, test_items = split_items(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    n_train = copy_split(train_items, output_dir, "train")
    n_val = copy_split(val_items, output_dir, "val")
    n_test = copy_split(test_items, output_dir, "test")

    classes = [c.strip() for c in args.class_names.split(",") if c.strip()]
    if not classes:
        raise ValueError("No valid class names provided")

    yaml_path = write_dataset_yaml(output_dir, classes)

    print("Dataset preparation complete")
    print(f"- valid pairs: {len(pairs)}")
    print(f"- missing labels ignored: {len(missing_labels)}")
    print(f"- train: {n_train}")
    print(f"- val: {n_val}")
    print(f"- test: {n_test}")
    print(f"- yaml: {yaml_path}")


if __name__ == "__main__":
    main()
