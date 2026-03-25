import argparse
import csv
import os
from typing import Dict, List


CANONICAL_FIELDS = [
    "image",
    "err1",
    "err2",
    "err3",
    "err4",
    "err5",
    "err6",
    "err7",
    "err8",
    "err9",
    "err10",
    "places_detectees",
    "coords",
]


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _from_dict_row(row: Dict[str, str]) -> Dict[str, str]:
    out = {k: "0" for k in CANONICAL_FIELDS}
    out["coords"] = ""

    out["image"] = (row.get("image") or "").strip()
    for i in range(1, 11):
        out[f"err{i}"] = str(_safe_int(row.get(f"err{i}", "0"), 0))

    out["places_detectees"] = str(_safe_int(row.get("places_detectees", "0"), 0))
    out["coords"] = (row.get("coords") or "").strip()
    return out


def _from_legacy_row(row: List[str]) -> Dict[str, str]:
    out = {k: "0" for k in CANONICAL_FIELDS}
    out["coords"] = ""

    if not row:
        return out

    out["image"] = row[0].strip()
    for i in range(1, 10):
        idx = i
        out[f"err{i}"] = str(_safe_int(row[idx], 0)) if len(row) > idx else "0"

    # Legacy files may miss err10.
    out["err10"] = str(_safe_int(row[10], 0)) if len(row) > 10 else "0"
    out["places_detectees"] = str(_safe_int(row[11], 0)) if len(row) > 11 else "0"
    out["coords"] = row[12].strip() if len(row) > 12 else ""
    return out


def clean_manual_csv(csv_path: str) -> tuple[int, int]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    by_image: Dict[str, Dict[str, str]] = {}

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        dict_reader = csv.DictReader(f)
        has_err10 = dict_reader.fieldnames is not None and "err10" in dict_reader.fieldnames

        if has_err10:
            for row in dict_reader:
                normalized = _from_dict_row(row)
                img = normalized["image"]
                if img:
                    by_image[img] = normalized
        else:
            f.seek(0)
            raw_reader = csv.reader(f)
            next(raw_reader, None)
            for row in raw_reader:
                normalized = _from_legacy_row(row)
                img = normalized["image"]
                if img:
                    by_image[img] = normalized

    rows = [by_image[k] for k in sorted(by_image.keys())]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), len(by_image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nettoie check_manuel_results.csv")
    parser.add_argument("--csv", default="check_manuel_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kept_rows, unique_images = clean_manual_csv(args.csv)
    print(f"CSV nettoye: {args.csv}")
    print(f"Lignes conservees: {kept_rows}")
    print(f"Images uniques: {unique_images}")


if __name__ == "__main__":
    main()
