import argparse
import csv
import os
from datetime import datetime


DEFAULT_QUEUE_TXT = "manual_review_queue.txt"
DEFAULT_REVIEWED_CSV = "manual_review_done.csv"
DEFAULT_REVIEWED_TXT = "manual_review_done.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive a manual review queue as already reviewed")
    parser.add_argument("--queue-txt", default=DEFAULT_QUEUE_TXT)
    parser.add_argument("--reviewed-csv", default=DEFAULT_REVIEWED_CSV)
    parser.add_argument("--reviewed-txt", default=DEFAULT_REVIEWED_TXT)
    return parser.parse_args()


def load_queue(queue_txt: str) -> list[str]:
    if not os.path.exists(queue_txt):
        raise FileNotFoundError(f"Queue introuvable: {queue_txt}")

    images: list[str] = []
    with open(queue_txt, mode="r", encoding="utf-8") as f:
        for line in f:
            image = line.strip()
            if image:
                images.append(image)
    return images


def load_existing_reviewed(reviewed_csv: str, reviewed_txt: str) -> set[str]:
    reviewed: set[str] = set()

    if os.path.exists(reviewed_csv):
        with open(reviewed_csv, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image = (row.get("image") or "").strip()
                if image:
                    reviewed.add(image)

    if os.path.exists(reviewed_txt):
        with open(reviewed_txt, mode="r", encoding="utf-8") as f:
            for line in f:
                image = line.strip()
                if image:
                    reviewed.add(image)

    return reviewed


def append_reviewed(reviewed_csv: str, reviewed_txt: str, images: list[str]) -> int:
    existing = load_existing_reviewed(reviewed_csv, reviewed_txt)
    new_images = [img for img in images if img not in existing]

    if not new_images:
        return 0

    csv_exists = os.path.exists(reviewed_csv)
    with open(reviewed_csv, mode="a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if not csv_exists:
            writer.writerow(["timestamp", "image", "status"])
        for image in new_images:
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image, "reviewed"])

    with open(reviewed_txt, mode="a", encoding="utf-8") as f_txt:
        for image in new_images:
            f_txt.write(image + "\n")

    return len(new_images)


def main() -> None:
    args = parse_args()
    images = load_queue(args.queue_txt)
    added = append_reviewed(args.reviewed_csv, args.reviewed_txt, images)

    print("Lot de correction archive.")
    print(f"- queue: {args.queue_txt}")
    print(f"- nouvelles images archivees: {added}")
    print(f"- reviewed csv: {args.reviewed_csv}")
    print(f"- reviewed txt: {args.reviewed_txt}")


if __name__ == "__main__":
    main()
