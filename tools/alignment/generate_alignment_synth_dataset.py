import argparse
import glob
import os

import cv2
import numpy as np

from alignment_ml_utils import extract_alignment_features


def collect_images(folder: str) -> list[str]:
    image_files: list[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(image_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic camera-shift dataset for alignment model"
    )
    parser.add_argument("--input-folder", default="../DATA")
    parser.add_argument("--max-images", type=int, default=400)
    parser.add_argument("--aug-per-image", type=int, default=10)
    parser.add_argument("--max-shift", type=float, default=50.0)
    parser.add_argument("--output", default="alignment_synth_dataset.npz")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def random_shift_warp(img: np.ndarray, max_shift: float, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    dx = float(rng.uniform(-max_shift, max_shift))
    dy = float(rng.uniform(-max_shift, max_shift))

    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    warped = cv2.warpAffine(
        img,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped, dx, dy


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    image_files = collect_images(args.input_folder)
    if not image_files:
        print(f"No images found in {args.input_folder}")
        return

    if len(image_files) > args.max_images:
        idx = rng.choice(len(image_files), size=args.max_images, replace=False)
        image_files = [image_files[i] for i in sorted(idx)]

    x_rows = []
    y_rows = []

    for img_path in image_files:
        base = cv2.imread(img_path)
        if base is None:
            continue

        for _ in range(args.aug_per_image):
            warped, dx, dy = random_shift_warp(base, args.max_shift, rng)
            feat = extract_alignment_features(base, warped)
            x_rows.append(feat)
            y_rows.append([dx, dy])

    if not x_rows:
        print("Dataset generation failed: no valid samples")
        return

    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)

    np.savez_compressed(args.output, x=x, y=y)
    print(f"Dataset generated: {args.output}")
    print(f"Samples: {x.shape[0]}")
    print(f"Features: {x.shape[1]}")


if __name__ == "__main__":
    main()
