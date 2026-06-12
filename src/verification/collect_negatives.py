from __future__ import annotations

import argparse
import hashlib
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.common import NEGATIVE_ROOT, PROCESSED_ROOT, VERIFICATION_ROOT, ensure_dir, prepare_output_dirs, save_json, set_seed
from src.verification.dataset_builder import create_verification_splits


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CATEGORIES = [
    "road",
    "machinery",
    "truck_body",
    "construction_site",
    "people",
    "buildings",
    "nature",
    "miscellaneous",
]

KEYWORDS = {
    "road": ["road", "street", "highway", "pavement", "traffic", "asphalt"],
    "machinery": ["machine", "machinery", "engine", "factory", "industrial", "equipment", "robot"],
    "truck_body": ["truck", "lorry", "trailer", "bus", "vehicle_body", "cab", "container"],
    "construction_site": ["construction", "site", "crane", "bulldozer", "excavator", "cement", "scaffold"],
    "people": ["people", "person", "human", "worker", "crowd", "portrait", "face"],
    "buildings": ["building", "house", "office", "tower", "city", "architecture", "warehouse"],
    "nature": ["nature", "tree", "forest", "animal", "grass", "mountain", "river", "sky"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def infer_category(path: Path) -> str:
    lowered = str(path).lower().replace("-", "_")
    for category, keywords in KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "miscellaneous"


def valid_image(path: Path) -> tuple[bool, tuple[int, int] | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image = image.convert("RGB")
            return True, image.size
    except Exception:
        return False, None


def build_positive_hashes() -> set[str]:
    hashes = set()
    for class_name in ["good", "bad"]:
        for path in (PROCESSED_ROOT / class_name).glob("*"):
            if path.is_file():
                hashes.add(sha256_file(path))
    return hashes


def save_sample_grid(image_paths: list[Path], output_path: Path, sample_size: int = 16, seed: int = 42) -> None:
    if not image_paths:
        return

    rng = random.Random(seed)
    selected = image_paths if len(image_paths) <= sample_size else rng.sample(image_paths, sample_size)
    columns = 4
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(rows, columns)

    for axis in axes.flatten():
        axis.axis("off")

    for axis, path in zip(axes.flatten(), selected):
        with Image.open(path) as image:
            image = ImageOps.contain(image.convert("RGB"), (256, 256))
            axis.imshow(image)
            axis.set_title(path.parent.name.replace("_", " "), fontsize=8)
            axis.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_summary(report: dict, output_path: Path) -> None:
    lines = [
        "# Negative Dataset Summary",
        "",
        "## Overview",
        "",
        f"- Total ingested images: {report['total_images']}",
        f"- Unique kept images: {report['unique_images']}",
        f"- Duplicate count: {report['duplicate_count']}",
        f"- Corruption count: {report['corruption_count']}",
        f"- Overlap with tyre dataset removed: {report['tyre_overlap_count']}",
        "",
        "## Image Size Statistics",
        "",
        f"- Width min/mean/max: {report['image_size_statistics']['width']['min']} / {report['image_size_statistics']['width']['mean']} / {report['image_size_statistics']['width']['max']}",
        f"- Height min/mean/max: {report['image_size_statistics']['height']['min']} / {report['image_size_statistics']['height']['mean']} / {report['image_size_statistics']['height']['max']}",
        "",
        "## Diversity Analysis",
        "",
    ]

    for category, count in report["category_distribution"].items():
        lines.append(f"- {category}: {count}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Categories are inferred from folder and file names when collecting source images.",
            "- Files are deduplicated with SHA-256 hashing.",
            "- Corrupted and unreadable files are excluded from the organized dataset.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def collect_negatives(input_roots: list[Path], output_root: Path = NEGATIVE_ROOT, seed: int = 42) -> dict:
    prepare_output_dirs()
    set_seed(seed)

    if output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(output_root)

    for category in CATEGORIES:
        ensure_dir(output_root / category)

    positive_hashes = build_positive_hashes()
    seen_hashes: dict[str, Path] = {}
    duplicate_count = 0
    corruption_count = 0
    tyre_overlap_count = 0
    total_images = 0
    kept_paths: list[Path] = []
    widths: list[int] = []
    heights: list[int] = []
    source_distribution = Counter()
    category_distribution = Counter()
    duplicate_examples: dict[str, list[str]] = defaultdict(list)

    for root in input_roots:
        if not root.exists():
            continue
        if root.resolve() == output_root.resolve():
            continue
        for source_path in iter_images(root):
            total_images += 1
            is_valid, size = valid_image(source_path)
            if not is_valid or size is None:
                corruption_count += 1
                continue

            file_hash = sha256_file(source_path)
            if file_hash in positive_hashes:
                tyre_overlap_count += 1
                continue

            if file_hash in seen_hashes:
                duplicate_count += 1
                duplicate_examples[str(seen_hashes[file_hash])].append(str(source_path))
                continue

            category = infer_category(source_path)
            destination = output_root / category / f"{file_hash}{source_path.suffix.lower()}"
            shutil.copy2(source_path, destination)

            seen_hashes[file_hash] = destination
            kept_paths.append(destination)
            widths.append(size[0])
            heights.append(size[1])
            source_distribution[root.name or str(root)] += 1
            category_distribution[category] += 1

    if not kept_paths:
        raise ValueError("No valid negative images were collected.")

    size_stats = {
        "width": {
            "min": int(min(widths)),
            "mean": round(float(np.mean(widths)), 2),
            "max": int(max(widths)),
        },
        "height": {
            "min": int(min(heights)),
            "mean": round(float(np.mean(heights)), 2),
            "max": int(max(heights)),
        },
    }

    report = {
        "input_roots": [str(path.resolve()) for path in input_roots if path.exists()],
        "output_root": str(output_root.resolve()),
        "total_images": total_images,
        "unique_images": len(kept_paths),
        "duplicate_count": duplicate_count,
        "corruption_count": corruption_count,
        "tyre_overlap_count": tyre_overlap_count,
        "image_size_statistics": size_stats,
        "category_distribution": {category: int(category_distribution.get(category, 0)) for category in CATEGORIES},
        "source_distribution": dict(source_distribution),
        "duplicate_examples": duplicate_examples,
    }

    save_json(VERIFICATION_ROOT / "negative_dataset_report.json", report)
    save_sample_grid(kept_paths, VERIFICATION_ROOT / "negative_samples_grid.jpg", seed=seed)
    write_summary(report, VERIFICATION_ROOT / "negative_dataset_summary.md")

    split_summary = create_verification_splits(negative_roots=[output_root], seed=seed)
    report["verification_splits"] = split_summary
    save_json(VERIFICATION_ROOT / "negative_dataset_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and clean non-tyre images for verifier training.")
    parser.add_argument("--input-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, default=NEGATIVE_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect_negatives(args.input_roots, output_root=args.output_root, seed=args.seed)
    print(f"Collected {report['unique_images']} negative images into {args.output_root}")
    print(f"Duplicates removed: {report['duplicate_count']}")
    print(f"Corrupted skipped: {report['corruption_count']}")
    print(f"Tyre overlaps removed: {report['tyre_overlap_count']}")


if __name__ == "__main__":
    main()
