#!/usr/bin/env python3
"""Generate offline augmented YOLO samples from source images/labels."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline augmentation for YOLO dataset")
    parser.add_argument("--images-dir", required=True, help="Folder of source images")
    parser.add_argument("--labels-dir", required=True, help="Folder of source YOLO labels")
    parser.add_argument("--output-images-dir", required=True, help="Output folder for augmented images")
    parser.add_argument("--output-labels-dir", required=True, help="Output folder for augmented labels")
    parser.add_argument("--variations", type=int, default=10, help="Number of augmented variants per source image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Also copy original image/label to output",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a source image has no matching label",
    )
    return parser.parse_args()


def iter_images(images_dir: Path):
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


def read_labels(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []
    rows: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except Exception:
            continue
        rows.append([float(cls_id), x, y, w, h])
    return rows


def write_labels(label_path: Path, rows: list[list[float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cls_id, x, y, w, h in rows:
        lines.append(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def apply_photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.75, 1.35)
    beta = rng.uniform(-45.0, 45.0)
    gamma = rng.uniform(0.75, 1.35)

    work = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    look_up = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    work = cv2.LUT(work, look_up)

    h, w = work.shape[:2]
    shadow_strength = rng.uniform(0.0, 0.45)
    if shadow_strength > 0.05:
        mask = np.ones((h, w), dtype=np.float32)
        y0 = int(h * rng.uniform(0.25, 0.6))
        y1 = int(h * rng.uniform(0.75, 1.0))
        x0 = int(w * rng.uniform(0.0, 0.3))
        x1 = int(w * rng.uniform(0.7, 1.0))
        cv2.rectangle(mask, (x0, y0), (x1, y1), 1.0 - shadow_strength, thickness=-1)
        work = np.clip(work.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)

    return work


def horizontal_flip_labels(rows: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for cls_id, x, y, w, h in rows:
        out.append([cls_id, 1.0 - x, y, w, h])
    return out


def save_augmented_sample(
    out_images_dir: Path,
    out_labels_dir: Path,
    rel_path: Path,
    stem_suffix: str,
    image: np.ndarray,
    labels: list[list[float]],
) -> None:
    rel_parent = rel_path.parent
    new_stem = f"{rel_path.stem}_{stem_suffix}"

    out_img = out_images_dir / rel_parent / f"{new_stem}.jpg"
    out_lbl = out_labels_dir / rel_parent / f"{new_stem}.txt"

    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_img), image)
    write_labels(out_lbl, labels)


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    out_images_dir = Path(args.output_images_dir).resolve()
    out_labels_dir = Path(args.output_labels_dir).resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")
    if args.variations < 1:
        raise ValueError("--variations must be >= 1")

    if out_images_dir.exists():
        shutil.rmtree(out_images_dir)
    if out_labels_dir.exists():
        shutil.rmtree(out_labels_dir)

    rng = random.Random(args.seed)

    total_sources = 0
    total_written = 0
    missing_labels = 0

    for image_path in iter_images(images_dir):
        rel_path = image_path.relative_to(images_dir)
        label_path = find_label_path(image_path, rel_path, labels_dir)

        if label_path is None:
            missing_labels += 1
            if args.strict:
                raise RuntimeError(f"Missing label for image: {rel_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        labels = read_labels(label_path)
        total_sources += 1

        if args.include_original:
            save_augmented_sample(out_images_dir, out_labels_dir, rel_path, "orig", image, labels)
            total_written += 1

        for i in range(1, args.variations + 1):
            if i == 1:
                aug_image = cv2.flip(image, 1)
                aug_labels = horizontal_flip_labels(labels)
            else:
                aug_image = apply_photometric(image, rng)
                aug_labels = labels

            save_augmented_sample(
                out_images_dir,
                out_labels_dir,
                rel_path,
                f"aug{i:02d}",
                aug_image,
                aug_labels,
            )
            total_written += 1

    print("Offline augmentation complete")
    print(f"- source images used: {total_sources}")
    print(f"- missing labels ignored: {missing_labels}")
    print(f"- generated images: {total_written}")
    print(f"- output images: {out_images_dir}")
    print(f"- output labels: {out_labels_dir}")


if __name__ == "__main__":
    main()
