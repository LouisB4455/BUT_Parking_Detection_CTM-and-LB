import pickle
from typing import Any, Tuple

import cv2
import numpy as np


FEATURE_NAMES = [
    "phase_dx",
    "phase_dy",
    "phase_response",
    "absdiff_mean",
    "absdiff_std",
    "edge_diff_mean",
]


def _to_gray_small(image_bgr: np.ndarray, size: Tuple[int, int] = (320, 180)) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def extract_alignment_features(ref_image: np.ndarray, cur_image: np.ndarray) -> np.ndarray:
    ref = _to_gray_small(ref_image)
    cur = _to_gray_small(cur_image)

    ref_f = ref.astype(np.float32)
    cur_f = cur.astype(np.float32)

    h, w = ref.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(ref_f, cur_f, window)

    absdiff = cv2.absdiff(ref, cur)
    edges_ref = cv2.Canny(ref, 80, 160)
    edges_cur = cv2.Canny(cur, 80, 160)
    edge_diff = cv2.absdiff(edges_ref, edges_cur)

    feat = np.array(
        [
            float(dx),
            float(dy),
            float(response),
            float(np.mean(absdiff)),
            float(np.std(absdiff)),
            float(np.mean(edge_diff)),
        ],
        dtype=np.float32,
    )
    return feat


def save_alignment_model(model: dict[str, Any], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_alignment_model(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        model = pickle.load(f)

    if not isinstance(model, dict):
        raise ValueError("Alignment model format invalid")
    if model.get("type") != "linear_translation_v1":
        raise ValueError("Unsupported alignment model type")

    return model


def predict_translation(model: dict[str, Any], features: np.ndarray) -> tuple[float, float, float]:
    x = features.astype(np.float32)
    x_mean = np.asarray(model["x_mean"], dtype=np.float32)
    x_std = np.asarray(model["x_std"], dtype=np.float32)
    weights = np.asarray(model["weights"], dtype=np.float32)
    bias = np.asarray(model["bias"], dtype=np.float32)

    x_norm = (x - x_mean) / np.maximum(x_std, 1e-6)
    y = x_norm @ weights + bias

    phase_response = float(np.clip(features[2], 0.0, 1.0))
    return float(y[0]), float(y[1]), phase_response


def fit_linear_translation_model(
    x: np.ndarray,
    y: np.ndarray,
    reg_lambda: float = 1e-3,
) -> dict[str, Any]:
    if x.ndim != 2 or y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("Expected x:(n,d), y:(n,2)")

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std = np.maximum(x_std, 1e-6)
    x_norm = (x - x_mean) / x_std

    d = x_norm.shape[1]
    xtx = x_norm.T @ x_norm
    reg = reg_lambda * np.eye(d, dtype=np.float32)
    w = np.linalg.solve(xtx + reg, x_norm.T @ y)
    b = np.mean(y - x_norm @ w, axis=0)

    model = {
        "type": "linear_translation_v1",
        "feature_names": FEATURE_NAMES,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "weights": w.tolist(),
        "bias": b.tolist(),
    }
    return model
