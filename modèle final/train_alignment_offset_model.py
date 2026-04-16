import argparse
import os

import numpy as np

from alignment_ml_utils import (
    fit_linear_translation_model,
    predict_translation,
    save_alignment_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train linear camera-shift model from synthetic alignment dataset"
    )
    parser.add_argument("--dataset", default="alignment_synth_dataset.npz")
    parser.add_argument("--output", default="alignment_offset_model.pkl")
    parser.add_argument("--reg-lambda", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mae(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(a - b), axis=0)


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}")
        return

    data = np.load(args.dataset)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)

    if x.shape[0] < 100:
        print(f"Not enough samples: {x.shape[0]} (need >=100)")
        return

    rng = np.random.default_rng(args.seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    x = x[idx]
    y = y[idx]

    n_val = max(1, int(x.shape[0] * args.val_ratio))
    x_val, y_val = x[:n_val], y[:n_val]
    x_train, y_train = x[n_val:], y[n_val:]

    model = fit_linear_translation_model(x_train, y_train, reg_lambda=args.reg_lambda)

    y_hat_train = np.array([predict_translation(model, xi)[:2] for xi in x_train], dtype=np.float32)
    y_hat_val = np.array([predict_translation(model, xi)[:2] for xi in x_val], dtype=np.float32)

    train_mae = mae(y_train, y_hat_train)
    val_mae = mae(y_val, y_hat_val)

    model["train_mae"] = train_mae.tolist()
    model["val_mae"] = val_mae.tolist()
    model["n_train"] = int(x_train.shape[0])
    model["n_val"] = int(x_val.shape[0])

    save_alignment_model(model, args.output)

    print(f"Model saved: {args.output}")
    print(f"Train samples: {x_train.shape[0]} | Val samples: {x_val.shape[0]}")
    print(f"MAE train dx/dy: {train_mae[0]:.2f} / {train_mae[1]:.2f} px")
    print(f"MAE val   dx/dy: {val_mae[0]:.2f} / {val_mae[1]:.2f} px")


if __name__ == "__main__":
    main()
