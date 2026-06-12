from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "datasets" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GOOD_DIR = PROCESSED_ROOT / "good"
BAD_DIR = PROCESSED_ROOT / "bad"


def rename_class(class_name: str, class_dir: Path) -> tuple[list[dict], int]:
    files = sorted(
        [p for p in class_dir.iterdir() if p.is_file()],
        key=lambda p: p.name,
    )
    total = len(files)

    # Build records with original names and extensions
    records: list[tuple[int, str, str]] = []
    for idx, path in enumerate(files, start=1):
        ext = path.suffix.lower()
        records.append((idx, path.name, ext))

    # Stage 1: rename original → unique temp name (avoid any collision)
    temp_map: list[tuple[int, str, str, Path]] = []
    for idx, orig_name, ext in records:
        src = class_dir / orig_name
        temp_name = f"_tmp_{idx:07d}{ext}"
        dst = class_dir / temp_name
        src.rename(dst)
        temp_map.append((idx, orig_name, ext, dst))

    # Stage 2: rename temp → final name
    mapping: list[dict] = []
    for idx, orig_name, ext, temp_path in temp_map:
        new_name = f"{class_name}_{idx}{ext}"
        new_path = class_dir / new_name
        temp_path.rename(new_path)
        mapping.append({
            "class": class_name,
            "original_filename": orig_name,
            "new_filename": new_name,
            "extension": ext.lstrip("."),
        })

    return mapping, total


def main() -> None:
    print(f"Processing: {GOOD_DIR}")
    good_mapping, good_total = rename_class("good", GOOD_DIR)
    print(f"Processing: {BAD_DIR}")
    bad_mapping, bad_total = rename_class("bad", BAD_DIR)

    # Re-read to verify and fill original names from sorted re-read
    all_mapping = good_mapping + bad_mapping

    # Verify final counts
    good_files_now = sorted([p for p in GOOD_DIR.iterdir() if p.is_file()], key=lambda p: p.name)
    bad_files_now = sorted([p for p in BAD_DIR.iterdir() if p.is_file()], key=lambda p: p.name)

    if len(good_files_now) != good_total:
        print(f"ERROR: Good files count mismatch: before={good_total}, after={len(good_files_now)}")
        sys.exit(1)
    if len(bad_files_now) != bad_total:
        print(f"ERROR: Bad files count mismatch: before={bad_total}, after={len(bad_files_now)}")
        sys.exit(1)

    # Check for duplicate filenames
    all_names = [p.name for p in good_files_now] + [p.name for p in bad_files_now]
    if len(all_names) != len(set(all_names)):
        print("ERROR: Duplicate filenames detected after rename")
        sys.exit(1)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate report JSON
    report = {
        "total_good_images": good_total,
        "total_bad_images": bad_total,
        "renamed_good_count": len(good_files_now),
        "renamed_bad_count": len(bad_files_now),
        "good_files": [
            {"original": m["original_filename"], "new": m["new_filename"]} for m in good_mapping
        ],
        "bad_files": [
            {"original": m["original_filename"], "new": m["new_filename"]} for m in bad_mapping
        ],
    }
    report_path = OUTPUTS_DIR / "dataset_rename_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")

    # Generate CSV mapping
    csv_path = OUTPUTS_DIR / "dataset_rename_mapping.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "original_filename", "new_filename", "extension"])
        writer.writeheader()
        for item in all_mapping:
            writer.writerow(item)
    print(f"Mapping saved: {csv_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Good images renamed: {good_total}")
    print(f"Bad images renamed: {bad_total}")
    print(f"Total renamed: {good_total + bad_total}")
    print(f"{'='*50}")

    # Show first 10 and last 10 good files
    print(f"\nFirst 10 good files:")
    for p in good_files_now[:10]:
        print(f"  {p.name}")
    print(f"\nFirst 10 bad files:")
    for p in bad_files_now[:10]:
        print(f"  {p.name}")

    print(f"\nGood total: {len(good_files_now)}")
    print(f"Bad total: {len(bad_files_now)}")


if __name__ == "__main__":
    main()
