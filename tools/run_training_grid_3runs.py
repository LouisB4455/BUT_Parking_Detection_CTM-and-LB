#!/usr/bin/env python3
"""Root-level wrapper to launch the optional 3-run grid script from tools/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _normalize_data_arg(args: list[str], model_dir: Path) -> list[str]:
    fixed = list(args)

    if "--data" in fixed:
        idx = fixed.index("--data")
        if idx + 1 < len(fixed):
            data_value = fixed[idx + 1]
            candidate = Path(data_value)
            if not candidate.is_absolute():
                expected = model_dir / data_value
                if not expected.exists():
                    raise FileNotFoundError(
                        f"Dataset YAML introuvable: {expected}. "
                        "Lance d'abord prepare_yolo_dataset.py vers batch_dataset."
                    )
    else:
        default_data = model_dir / "batch_dataset" / "dataset.yaml"
        if not default_data.exists():
            raise FileNotFoundError(
                f"Dataset YAML introuvable: {default_data}. "
                "Lance d'abord prepare_yolo_dataset.py vers batch_dataset."
            )
        fixed.extend(["--data", "batch_dataset/dataset.yaml"])

    return fixed


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    model_dir = root_dir / "modèle final"
    target_script = root_dir / "tools" / "training" / "run_training_grid_3runs.py"

    if not target_script.exists():
        raise FileNotFoundError(f"Script target introuvable: {target_script}")

    user_args = sys.argv[1:]
    user_args = _normalize_data_arg(user_args, model_dir)

    cmd = [sys.executable, str(target_script), *user_args]
    print("[INFO] Lancement:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(model_dir))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
