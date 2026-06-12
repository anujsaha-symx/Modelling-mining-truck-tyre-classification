from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean, stdev

from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.transforms import LetterboxResize
from src.utils.common import PROJECT_ROOT, ensure_dir

PROCESSED_ROOT = PROJECT_ROOT / "datasets" / "processed"
RESIZED_ROOT = PROJECT_ROOT / "datasets" / "resized"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CLASS_NAMES = ["good", "bad"]
TARGET_SIZE = 224


def process_class(
    class_name: str,
    letterbox: LetterboxResize,
) -> tuple[list[dict], list[str]]:
    src_dir = PROCESSED_ROOT / class_name
    dst_dir = RESIZED_ROOT / class_name
    ensure_dir(dst_dir)

    records: list[dict] = []
    errors: list[str] = []

    paths = sorted([p for p in src_dir.iterdir() if p.is_file()], key=lambda p: p.name)

    for path in paths:
        try:
            with Image.open(path) as img:
                orig_size = img.size
                out = letterbox(img)
                out_size = out.size

            out_path = dst_dir / path.name
            out.save(out_path, quality=95)

            src_w, src_h = orig_size
            tgt_w, tgt_h = letterbox.target_width, letterbox.target_height
            scale = min(tgt_w / src_w, tgt_h / src_h)
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            pad_left = (tgt_w - new_w) // 2
            pad_top = (tgt_h - new_h) // 2
            pad_right = tgt_w - new_w - pad_left
            pad_bottom = tgt_h - new_h - pad_top

            records.append({
                "filename": path.name,
                "class": class_name,
                "original_width": src_w,
                "original_height": src_h,
                "resized_width": out_size[0],
                "resized_height": out_size[1],
                "pad_top": pad_top,
                "pad_bottom": pad_bottom,
                "pad_left": pad_left,
                "pad_right": pad_right,
            })
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    return records, errors


def save_visualizations(records: list[dict], class_name: str, seed: int = 42) -> None:
    vis_dir = OUTPUTS_ROOT / "resize_visualization"
    ensure_dir(vis_dir)

    rng = random.Random(seed)
    class_records = [r for r in records if r["class"] == class_name]
    if not class_records:
        return

    sample_size = min(20, len(class_records))
    sampled = rng.sample(class_records, sample_size)

    for rec in sampled:
        src_path = PROCESSED_ROOT / class_name / rec["filename"]
        dst_path = RESIZED_ROOT / class_name / rec["filename"]
        if not src_path.exists() or not dst_path.exists():
            continue

        src_img = Image.open(src_path).convert("RGB")
        dst_img = Image.open(dst_path).convert("RGB")

        src_w, src_h = src_img.size
        dst_w, dst_h = dst_img.size

        vis_w = max(src_w, dst_w) + 40
        total_w = vis_w * 2 + 20
        total_h = max(src_h, dst_h) + 60
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

        canvas.paste(src_img, (10, 30))
        canvas.paste(dst_img, (vis_w + 10, 30))

        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
        draw.text((10, 5), "Original", fill=(0, 0, 0))
        draw.text((vis_w + 10, 5), "Letterbox 224x224", fill=(0, 0, 0))

        out_name = f"comparison_{class_name}_{rec['filename'].rsplit('.', 1)[0]}.jpg"
        canvas.save(vis_dir / out_name, quality=90)


def main() -> None:
    start = time.time()

    letterbox = LetterboxResize(TARGET_SIZE)

    all_records: list[dict] = []
    all_errors: list[str] = []

    for class_name in CLASS_NAMES:
        records, errors = process_class(class_name, letterbox)
        all_records.extend(records)
        all_errors.extend(errors)

    elapsed = time.time() - start

    good_records = [r for r in all_records if r["class"] == "good"]
    bad_records = [r for r in all_records if r["class"] == "bad"]

    # Validation
    good_src = [p for p in (PROCESSED_ROOT / "good").iterdir() if p.is_file()]
    bad_src = [p for p in (PROCESSED_ROOT / "bad").iterdir() if p.is_file()]
    good_dst = [p for p in (RESIZED_ROOT / "good").iterdir() if p.is_file()]
    bad_dst = [p for p in (RESIZED_ROOT / "bad").iterdir() if p.is_file()]

    assert len(good_src) == len(good_dst), f"Good count mismatch: {len(good_src)} vs {len(good_dst)}"
    assert len(bad_src) == len(bad_dst), f"Bad count mismatch: {len(bad_src)} vs {len(bad_dst)}"

    src_names = {p.name for p in good_src} | {p.name for p in bad_src}
    dst_names = {p.name for p in good_dst} | {p.name for p in bad_dst}
    assert src_names == dst_names, "Filenames differ between source and destination"

    all_224 = all(r["resized_width"] == TARGET_SIZE and r["resized_height"] == TARGET_SIZE for r in all_records)
    assert all_224, "Not all output images are 224x224"

    # Padding statistics
    pads = {
        "pad_top": [r["pad_top"] for r in all_records],
        "pad_bottom": [r["pad_bottom"] for r in all_records],
        "pad_left": [r["pad_left"] for r in all_records],
        "pad_right": [r["pad_right"] for r in all_records],
    }
    padding_statistics = {}
    for key, values in pads.items():
        stats = {
            "min": int(min(values)),
            "max": int(max(values)),
            "mean": round(mean(values), 2),
        }
        if len(values) > 1:
            stats["std"] = round(stdev(values), 2)
        else:
            stats["std"] = 0.0
        padding_statistics[key] = stats
    padding_statistics["images_with_padding"] = sum(
        1 for r in all_records if r["pad_top"] > 0 or r["pad_left"] > 0
    )
    padding_statistics["images_without_padding"] = sum(
        1 for r in all_records if r["pad_top"] == 0 and r["pad_left"] == 0
    )

    # Report JSON
    report = {
        "target_size": [TARGET_SIZE, TARGET_SIZE],
        "good_images_processed": len(good_records),
        "bad_images_processed": len(bad_records),
        "total_images_processed": len(all_records),
        "failed_images": len(all_errors),
        "failed_details": all_errors[:50],
        "padding_statistics": padding_statistics,
        "processing_time_seconds": round(elapsed, 2),
    }
    report_path = OUTPUTS_ROOT / "resize_report.json"
    ensure_dir(report_path.parent)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # CSV summary
    csv_path = OUTPUTS_ROOT / "resized_dataset_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "class", "original_width", "original_height",
            "resized_width", "resized_height",
            "pad_top", "pad_bottom", "pad_left", "pad_right",
        ])
        writer.writeheader()
        writer.writerows(all_records)

    # Visualizations
    save_visualizations(all_records, "good")
    save_visualizations(all_records, "bad")

    # Print summary
    print(f"Good images processed: {len(good_records)}")
    print(f"Bad images processed: {len(bad_records)}")
    print(f"Total images processed: {len(all_records)}")
    print(f"All output images verified as {TARGET_SIZE}x{TARGET_SIZE}.")
    if all_errors:
        print(f"Failed: {len(all_errors)}")
        for err in all_errors[:5]:
            print(f"  {err}")

    print(f"\nFiles created:")
    print(f"  {report_path}")
    print(f"  {csv_path}")
    print(f"  {OUTPUTS_ROOT / 'resize_visualization/'} (20 comparison images)")


if __name__ == "__main__":
    main()
