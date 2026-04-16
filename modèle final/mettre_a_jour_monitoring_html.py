#!/usr/bin/env python3
import csv
import json
import os
import re
from datetime import datetime

HISTORY_CSV = "resultats_modele_final_history.csv"
RESULT_CSV = "resultats_modele_final.csv"
HTML_FILE = "monitoring_final_simple.html"
EXPECTED_FIELDS = {
    "timestamp",
    "image",
    "total_cars",
    "cars_in_forbidden",
    "cars_legal",
    "alignment_source",
    "alignment_confidence",
    "uncertain",
    "processing_seconds",
}

CURRENT_SCHEMA_FIELDS = [
    "timestamp",
    "image",
    "total_cars",
    "cars_in_forbidden",
    "cars_legal",
    "line_delta_dx",
    "line_delta_dy",
    "alignment_source",
    "alignment_confidence",
    "uncertain",
    "processing_seconds",
]


def to_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def parse_capture_timestamp(image_value):
    if not image_value:
        return ""

    text = str(image_value).replace("\\", "/")
    match = re.search(r"(\d{4}-\d{2}-\d{2})/(\d{4}-\d{2}-\d{2}) (\d{2})_(\d{2})_(\d{2})", text)
    if not match:
        return ""

    date_part = match.group(2)
    hour = match.group(3)
    minute = match.group(4)
    second = match.group(5)
    return f"{date_part} {hour}:{minute}:{second}"


def build_result(row):
    image_value = row.get("image", "")
    capture_timestamp = parse_capture_timestamp(image_value)
    timestamp_value = row.get("timestamp", "")
    display_timestamp = capture_timestamp or timestamp_value
    sort_key = display_timestamp or timestamp_value or image_value

    return {
        "timestamp": timestamp_value,
        "capture_timestamp": capture_timestamp,
        "display_timestamp": display_timestamp,
        "sort_key": sort_key,
        "image": image_value,
        "total_cars": to_int(row.get("total_cars", 0)),
        "cars_in_forbidden": to_int(row.get("cars_in_forbidden", 0)),
        "cars_legal": to_int(row.get("cars_legal", 0)),
        "alignment_source": row.get("alignment_source", "none"),
        "alignment_confidence": row.get("alignment_confidence", ""),
        "uncertain": to_int(row.get("uncertain", 0)),
        "processing_seconds": float(row.get("processing_seconds", 0.0) or 0.0),
    }


def write_results_csv(csv_path, results):
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CURRENT_SCHEMA_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "timestamp": result.get("timestamp", ""),
                "image": result.get("image", ""),
                "total_cars": result.get("total_cars", 0),
                "cars_in_forbidden": result.get("cars_in_forbidden", 0),
                "cars_legal": result.get("cars_legal", 0),
                "line_delta_dx": result.get("line_delta_dx", "0.0"),
                "line_delta_dy": result.get("line_delta_dy", "0.0"),
                "alignment_source": result.get("alignment_source", "none"),
                "alignment_confidence": result.get("alignment_confidence", ""),
                "uncertain": result.get("uncertain", 0),
                "processing_seconds": f"{float(result.get('processing_seconds', 0.0) or 0.0):.6f}",
            })


def csv_has_expected_schema(path):
    if not os.path.exists(path):
        return False

    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
    return EXPECTED_FIELDS.issubset(fieldnames)


def load_results_from_csv(csv_path):
    """Load results from a specific CSV file"""
    results = []
    
    if not os.path.exists(csv_path):
        return results
    
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        sample = f.readline()
        f.seek(0)

        if sample and EXPECTED_FIELDS.issubset(set(sample.strip().split(","))):
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    results.append(build_result(row))
                except Exception:
                    continue
        else:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                mapped = {
                    field: row[i].strip() if i < len(row) else ""
                    for i, field in enumerate(CURRENT_SCHEMA_FIELDS)
                }
                try:
                    results.append(build_result(mapped))
                except Exception:
                    continue

    results.sort(key=lambda item: item.get("sort_key", ""))
    
    return results


def load_results():
    """Load both recent and historical results"""
    # Load recent analysis (current subfolder only)
    recent = load_results_from_csv(RESULT_CSV)
    
    # Load all history (cumulative)
    history = load_results_from_csv(HISTORY_CSV)
    if history and not csv_has_expected_schema(HISTORY_CSV):
        write_results_csv(HISTORY_CSV, history)
    
    # Deduplicate history by keeping only the latest entry per image
    if history:
        history_dict = {}
        for result in history:
            img = result.get("image", "")
            if img:
                history_dict[img] = result
        
        # Convert back to list and re-sort
        dedup_history = list(history_dict.values())
        dedup_history.sort(key=lambda item: item.get("sort_key", ""))
        history = dedup_history
    
    return {
        "recent": recent,
        "alltime": history
    }


def main():
    if not os.path.exists(HTML_FILE):
        print(f"ERROR: {HTML_FILE} not found")
        return
    
    results_data = load_results()
    recent = results_data["recent"]
    alltime = results_data["alltime"]
    
    # Get stats from recent analysis
    stats = {
        "totalCars": 0,
        "forbiddenCars": 0,
        "legalCars": 0
    }
    
    if recent:
        stats["totalCars"] = recent[-1]["total_cars"]
        stats["forbiddenCars"] = recent[-1]["cars_in_forbidden"]
        stats["legalCars"] = recent[-1]["cars_legal"]
    
    last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare data JSON with both recent and historical results
    data = {
        "lastUpdate": last_update,
        "recent": recent,
        "alltime": alltime,
        "stats": stats
    }
    
    # Read HTML
    with open(HTML_FILE, "r", encoding="utf-8") as f:
        html = f.read()
    
    # Replace DATA object
    data_json = json.dumps(data, ensure_ascii=False, indent=2)
    new_data_line = f"        const DATA = {data_json};"
    
    html = re.sub(
        r"const DATA = \{[\s\S]*?\};",
        new_data_line.strip(),
        html,
        count=1
    )
    
    # Write back
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"[OK] {HTML_FILE} updated")
    print(f"   - resultats recents: {len(recent)}")
    print(f"   - resultats all time: {len(alltime)}")
    print(f"   - dernier total: {stats['totalCars']}")
    print(f"   - derniere maj: {last_update}")


if __name__ == "__main__":
    main()
