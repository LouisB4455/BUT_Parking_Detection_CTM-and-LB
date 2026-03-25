import csv
import json
import os
import re

HTML_FILE = "monitoring_officiel.html"
RESULT_CSV = "resultats_modele_final.csv"
MANUAL_CSV = "check_manuel_results.csv"


def to_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def normalize_manual_image(name: str) -> str:
    base = os.path.basename(name or "")
    prefix = "ModeleFinal_"
    if base.startswith(prefix):
        return base[len(prefix) :]
    return base


def load_result_rows():
    rows = []
    with open(RESULT_CSV, mode="r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "image": r.get("image", ""),
                    "free": to_int(r.get("free_places", 0)),
                    "occ": to_int(r.get("occupied_places", 0)),
                    "total": to_int(r.get("total_places", 0)),
                    "cars": to_int(r.get("cars_detected", 0)),
                    "illegal": to_int(r.get("illegal_parked", 0)),
                    "inlier": float(r.get("inlier_ratio", 0.0) or 0.0),
                }
            )
    return rows


def load_manual_rows():
    # keep latest pass per image
    latest = {}
    with open(MANUAL_CSV, mode="r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            image = normalize_manual_image(r.get("image", ""))
            if not image:
                continue
            latest[image] = {
                "image": image,
                "err1": to_int(r.get("err1", 0)),
                "err2": to_int(r.get("err2", 0)),
                "err3": to_int(r.get("err3", 0)),
                "err4": to_int(r.get("err4", 0)),
                "err5": to_int(r.get("err5", 0)),
                "err6": to_int(r.get("err6", 0)),
                "err7": to_int(r.get("err7", 0)),
                "err8": to_int(r.get("err8", 0)),
                "err9": to_int(r.get("err9", 0)),
                "err10": to_int(r.get("err10", 0)),
                "places": to_int(r.get("places_detectees", 0)),
            }

    return [latest[k] for k in sorted(latest.keys())]


def main():
    if not os.path.exists(HTML_FILE):
        raise FileNotFoundError(f"Fichier introuvable: {HTML_FILE}")
    if not os.path.exists(RESULT_CSV):
        raise FileNotFoundError(f"CSV introuvable: {RESULT_CSV}")
    if not os.path.exists(MANUAL_CSV):
        raise FileNotFoundError(f"CSV introuvable: {MANUAL_CSV}")

    result_rows = load_result_rows()
    manual_rows = load_manual_rows()

    with open(HTML_FILE, mode="r", encoding="utf-8") as f:
        html = f.read()

    result_js = "const RESULT_ROWS = " + json.dumps(result_rows, ensure_ascii=False, indent=2) + ";"
    manual_js = "const MANUAL_ROWS = " + json.dumps(manual_rows, ensure_ascii=False, indent=2) + ";"

    html, n1 = re.subn(
        r"const RESULT_ROWS = \[(.|\n)*?\];",
        result_js,
        html,
        count=1,
    )
    html, n2 = re.subn(
        r"const MANUAL_ROWS = \[(.|\n)*?\];",
        manual_js,
        html,
        count=1,
    )

    if n1 != 1 or n2 != 1:
        raise RuntimeError("Impossible de trouver les blocs RESULT_ROWS/MANUAL_ROWS dans le HTML.")

    with open(HTML_FILE, mode="w", encoding="utf-8") as f:
        f.write(html)

    print("monitoring_officiel.html mis a jour.")
    print(f"- lignes resultats: {len(result_rows)}")
    print(f"- lignes check manuel (uniques): {len(manual_rows)}")


if __name__ == "__main__":
    main()
