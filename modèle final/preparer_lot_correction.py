import argparse
import csv
import math
import os
from dataclasses import dataclass


DEFAULT_RESULTS_CSV = "resultats_modele_final.csv"
DEFAULT_QUEUE_TXT = "manual_review_queue.txt"
DEFAULT_QUEUE_REPORT = "manual_review_queue_report.csv"
DEFAULT_REVIEWED_CSV = "manual_review_done.csv"
DEFAULT_REVIEWED_TXT = "manual_review_done.txt"


@dataclass
class RowScore:
    image: str
    score: float
    confidence: float
    reason: str


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_reviewed_images(reviewed_csv: str, reviewed_txt: str) -> set[str]:
    reviewed: set[str] = set()

    if reviewed_csv and os.path.exists(reviewed_csv):
        with open(reviewed_csv, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image = (row.get("image") or "").strip()
                if image:
                    reviewed.add(image)

    if reviewed_txt and os.path.exists(reviewed_txt):
        with open(reviewed_txt, mode="r", encoding="utf-8") as f:
            for line in f:
                image = line.strip()
                if image:
                    reviewed.add(image)

    return reviewed


def compute_score(row: dict) -> RowScore:
    image = row.get("image", "").strip()
    illegal = _safe_int(row.get("cars_in_forbidden", row.get("illegal_parked", "0")), 0)
    cars = _safe_int(row.get("total_cars", row.get("cars_detected", "0")), 0)
    confidence = _safe_float(row.get("alignment_confidence", "0"), 0.0)
    alignment_source = (row.get("alignment_source", "") or "").strip().lower()
    uncertain_frame = _safe_int(row.get("uncertain", row.get("uncertain_frame", "0")), 0)

    # Higher score = higher priority for manual review.
    # Core rule: low confidence drives priority.
    score = (1.0 - max(0.0, min(1.0, confidence))) * 100.0
    reasons = []

    if uncertain_frame == 1:
        score += 40
        reasons.append("uncertain_frame")

    if alignment_source == "fallback":
        score += 20
        reasons.append("fallback_alignment")

    if illegal > 0:
        score += 10 + min(illegal, 5) * 2
        reasons.append(f"illegal_{illegal}")

    if cars == 0:
        score += 15
        reasons.append("zero_car")

    if not reasons:
        reasons.append("low_confidence")

    return RowScore(image=image, score=score, confidence=confidence, reason=";".join(reasons))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a prioritized manual review queue")
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--queue-txt", default=DEFAULT_QUEUE_TXT)
    parser.add_argument("--queue-report", default=DEFAULT_QUEUE_REPORT)
    parser.add_argument("--reviewed-csv", default=DEFAULT_REVIEWED_CSV)
    parser.add_argument("--reviewed-txt", default=DEFAULT_REVIEWED_TXT)
    parser.add_argument("--max-images", type=int, default=120)
    parser.add_argument("--low-confidence-percent", type=float, default=10.0)
    parser.add_argument("--fill-to-max", action="store_true", help="After selecting low-confidence quota, fill remaining slots by priority")
    parser.add_argument("--include-reviewed", action="store_true", help="Include already reviewed images in the queue")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.results_csv):
        raise FileNotFoundError(f"CSV resultats introuvable: {args.results_csv}")

    scored = []
    with open(args.results_csv, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = compute_score(row)
            if item.image:
                scored.append(item)

    if not scored:
        raise RuntimeError("Aucune ligne exploitable dans le CSV resultats")

    reviewed = set()
    if not args.include_reviewed:
        reviewed = _load_reviewed_images(args.reviewed_csv, args.reviewed_txt)

    scored.sort(key=lambda x: x.score, reverse=True)

    selected = []
    selected_set = set()

    # Mandatory bucket: 10% least confident images.
    by_conf = sorted(scored, key=lambda x: x.confidence)
    mandatory_count = max(1, int(math.ceil(len(by_conf) * (args.low_confidence_percent / 100.0))))
    for item in by_conf:
        if len(selected) >= mandatory_count:
            break
        if item.image in reviewed:
            continue
        if item.image in selected_set:
            continue
        selected.append(item)
        selected_set.add(item.image)

    # Optional fill: disabled by default to keep manual review focused.
    if args.fill_to_max:
        for item in scored:
            if len(selected) >= args.max_images:
                break
            if item.image in reviewed:
                continue
            if item.image in selected_set:
                continue
            selected.append(item)
            selected_set.add(item.image)

    with open(args.queue_txt, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(item.image + "\n")

    with open(args.queue_report, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "score", "confidence", "reason"])
        for item in selected:
            writer.writerow([item.image, f"{item.score:.2f}", f"{item.confidence:.4f}", item.reason])

    print("File de correction manuelle preparee.")
    print(f"- images total: {len(scored)}")
    print(f"- quota faible confiance: {mandatory_count} ({args.low_confidence_percent:.1f}%)")
    print(f"- deja revues exclues: {len(reviewed)}")
    print(f"- images selectionnees: {len(selected)}")
    print(f"- queue: {args.queue_txt}")
    print(f"- rapport: {args.queue_report}")


if __name__ == "__main__":
    main()
