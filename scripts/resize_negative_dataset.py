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
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.transforms import LetterboxResize
from src.utils.common import PROJECT_ROOT, ensure_dir

NEGATIVE_ROOT = PROJECT_ROOT / "datasets" / "negative"
RESIZED_NEGATIVE_ROOT = PROJECT_ROOT / "datasets" / "resized" / "negative"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
TARGET_SIZE = 224

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(root: Path) -> list[Path]:
    paths = []
    for p in sorted(root.rglob("*"), key=lambda p: (str(p.relative_to(root)), p.name)):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return paths


def process_negative_images() -> tuple[list[dict], list[str]]:
    ensure_dir(RESIZED_NEGATIVE_ROOT)

    letterbox = LetterboxResize(TARGET_SIZE)

    src_paths = collect_images(NEGATIVE_ROOT)
    records: list[dict] = []
    errors: list[str] = []

    for idx, src_path in enumerate(src_paths, start=1):
        orig_ext = src_path.suffix.lower()
        new_name = f"negative_{idx}{orig_ext}"
        dst_path = RESIZED_NEGATIVE_ROOT / new_name

        rel_path = src_path.relative_to(NEGATIVE_ROOT)

        try:
            with Image.open(src_path) as img:
                orig_w, orig_h = img.size
                out_img = letterbox(img)
                out_w, out_h = out_img.size

            out_img.save(dst_path, quality=95)

            tgt_w, tgt_h = letterbox.target_width, letterbox.target_height
            scale = min(tgt_w / orig_w, tgt_h / orig_h)
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            pad_left = (tgt_w - new_w) // 2
            pad_top = (tgt_h - new_h) // 2
            pad_right = tgt_w - new_w - pad_left
            pad_bottom = tgt_h - new_h - pad_top

            records.append({
                "original_path": str(rel_path),
                "new_filename": new_name,
                "extension": orig_ext,
                "original_width": orig_w,
                "original_height": orig_h,
                "resized_width": out_w,
                "resized_height": out_h,
                "pad_top": pad_top,
                "pad_bottom": pad_bottom,
                "pad_left": pad_left,
                "pad_right": pad_right,
            })
        except Exception as exc:
            errors.append(f"{rel_path}: {exc}")

    return records, errors


def save_mapping_csv(records: list[dict]) -> Path:
    csv_path = OUTPUTS_ROOT / "negative_resize_mapping.csv"
    ensure_dir(csv_path.parent)
    fieldnames = [
        "original_path", "new_filename", "extension",
        "original_width", "original_height",
        "resized_width", "resized_height",
        "pad_top", "pad_bottom", "pad_left", "pad_right",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return csv_path


def save_report_json(records: list[dict], errors: list[str], elapsed: float) -> Path:
    report = {
        "target_size": [TARGET_SIZE, TARGET_SIZE],
        "negative_images_processed": len(records),
        "failed_images": len(errors),
        "failed_details": errors[:50],
        "image_count_before": len(records) + len(errors),
        "image_count_after": len(records),
        "processing_time_seconds": round(elapsed, 2),
    }
    report_path = OUTPUTS_ROOT / "negative_resize_report.json"
    ensure_dir(report_path.parent)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report_path


def validate_outputs(records: list[dict]) -> None:
    dst_files = sorted(RESIZED_NEGATIVE_ROOT.iterdir(), key=lambda p: p.name)
    assert len(dst_files) == len(records), (
        f"Output file count {len(dst_files)} != records count {len(records)}"
    )

    filenames = [r["new_filename"] for r in records]
    assert len(filenames) == len(set(filenames)), "Duplicate filenames detected"

    for rec in records:
        assert rec["resized_width"] == TARGET_SIZE, (
            f"Width mismatch for {rec['new_filename']}: {rec['resized_width']}"
        )
        assert rec["resized_height"] == TARGET_SIZE, (
            f"Height mismatch for {rec['new_filename']}: {rec['resized_height']}"
        )

    for dst in dst_files:
        with Image.open(dst) as img:
            assert img.size == (TARGET_SIZE, TARGET_SIZE), (
                f"{dst.name} is {img.size}, expected {TARGET_SIZE}x{TARGET_SIZE}"
            )


def save_visualizations(records: list[dict], seed: int = 42) -> None:
    vis_dir = OUTPUTS_ROOT / "resize_visualization_negative"
    ensure_dir(vis_dir)

    rng = random.Random(seed)
    sample_size = min(20, len(records))
    if sample_size == 0:
        return

    sampled = rng.sample(records, sample_size)

    for rec in sampled:
        src_path = NEGATIVE_ROOT / rec["original_path"]
        dst_path = RESIZED_NEGATIVE_ROOT / rec["new_filename"]
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

        stem = rec["new_filename"].rsplit(".", 1)[0]
        out_name = f"comparison_{stem}.jpg"
        canvas.save(vis_dir / out_name, quality=90)


def main() -> None:
    start = time.time()

    records, errors = process_negative_images()

    elapsed = time.time() - start

    csv_path = save_mapping_csv(records)
    report_path = save_report_json(records, errors, elapsed)

    # Validation
    try:
        validate_outputs(records)
        validated = True
    except AssertionError as e:
        print(f"VALIDATION FAILED: {e}")
        validated = False

    # Visualizations
    save_visualizations(records)

    # Print summary
    print(f"Negative images processed: {len(records)}")
    print(f"Failed images: {len(errors)}")
    print(f"Output folder:")
    print(f"  {RESIZED_NEGATIVE_ROOT}")
    if validated:
        print(f"All images verified as {TARGET_SIZE}x{TARGET_SIZE}.")
    else:
        print("VALIDATION FAILED.")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors[:10]:
            print(f"  {err}")

    print(f"\nTotal images resized: {len(records)}")
    print(f"First 10 renamed files:")
    for r in records[:10]:
        print(f"  {r['original_path']} -> {r['new_filename']}")
    print(f"  ...")

    print(f"\nFiles created:")
    print(f"  {csv_path}")
    print(f"  {report_path}")
    print(f"  {OUTPUTS_ROOT / 'resize_visualization_negative/'} (comparison images)")


if __name__ == "__main__":
    main()
