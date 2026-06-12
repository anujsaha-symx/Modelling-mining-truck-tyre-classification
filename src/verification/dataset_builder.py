from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.common import (
    NEGATIVE_ROOT,
    PROCESSED_ROOT,
    VERIFICATION_SPLITS_ROOT,
    ensure_dir,
    save_json,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_TO_IDX = {"non_tyre": 0, "tyre": 1}


def collect_images(root: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_records(positive_root: Path, negative_roots: list[Path], negatives_per_positive: float = 1.0) -> pd.DataFrame:
    positive_paths = collect_images(positive_root / "good") + collect_images(positive_root / "bad")
    if not positive_paths:
        raise FileNotFoundError(f"No positive tyre images found under {positive_root}")

    negative_paths: list[Path] = []
    for root in negative_roots:
        negative_paths.extend(collect_images(root))

    if not negative_paths:
        raise FileNotFoundError(
            "No negative images found. Provide one or more folders with non-tyre images via --negative-roots."
        )

    positive_hashes = {sha256_file(path) for path in positive_paths}
    filtered_negative_paths = []
    overlap_count = 0
    for path in negative_paths:
        if sha256_file(path) in positive_hashes:
            overlap_count += 1
            continue
        filtered_negative_paths.append(path)

    negative_paths = filtered_negative_paths

    if not negative_paths:
        raise ValueError("All negative images overlap with tyre dataset content after hash verification.")

    max_negatives = min(len(negative_paths), int(len(positive_paths) * negatives_per_positive))
    negative_paths = negative_paths[:max_negatives]

    records = []
    for path in positive_paths:
        records.append({"filepath": str(path), "label": "tyre", "label_idx": LABEL_TO_IDX["tyre"]})
    for path in negative_paths:
        records.append({"filepath": str(path), "label": "non_tyre", "label_idx": LABEL_TO_IDX["non_tyre"]})

    frame = pd.DataFrame(records)
    frame.attrs["positive_negative_overlap_removed"] = overlap_count
    return frame


def verify_no_overlap(*frames: pd.DataFrame) -> None:
    split_sets = [set(frame["filepath"].tolist()) for frame in frames]
    for idx, current in enumerate(split_sets):
        for other_idx, other in enumerate(split_sets[idx + 1 :], start=idx + 1):
            overlap = current.intersection(other)
            if overlap:
                raise ValueError(
                    f"Found filepath overlap between verification splits {idx} and {other_idx}: {next(iter(overlap))}"
                )


def summarize_split(name: str, frame: pd.DataFrame) -> dict:
    counts = frame["label"].value_counts().sort_index().to_dict()
    return {
        "split": name,
        "samples": int(len(frame)),
        "class_counts": {key: int(value) for key, value in counts.items()},
        "class_ratio": {key: round(value / len(frame), 4) for key, value in counts.items()},
    }


def create_verification_splits(
    negative_roots: list[Path],
    positive_root: Path = PROCESSED_ROOT,
    output_dir: Path = VERIFICATION_SPLITS_ROOT,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    negatives_per_positive: float = 1.0,
) -> dict:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test sizes must sum to 1.0")

    ensure_dir(output_dir)
    dataframe = build_records(positive_root, negative_roots, negatives_per_positive=negatives_per_positive)

    train_frame, temp_frame = train_test_split(
        dataframe,
        test_size=(1.0 - train_size),
        stratify=dataframe["label_idx"],
        random_state=seed,
    )
    relative_val = val_size / (val_size + test_size)
    val_frame, test_frame = train_test_split(
        temp_frame,
        test_size=(1.0 - relative_val),
        stratify=temp_frame["label_idx"],
        random_state=seed,
    )

    train_frame = train_frame.sort_values("filepath").reset_index(drop=True)
    val_frame = val_frame.sort_values("filepath").reset_index(drop=True)
    test_frame = test_frame.sort_values("filepath").reset_index(drop=True)
    verify_no_overlap(train_frame, val_frame, test_frame)

    train_frame.to_csv(output_dir / "train.csv", index=False)
    val_frame.to_csv(output_dir / "val.csv", index=False)
    test_frame.to_csv(output_dir / "test.csv", index=False)

    summary = {
        "positive_root": str(positive_root.resolve()),
        "negative_roots": [str(path.resolve()) for path in negative_roots],
        "total_samples": int(len(dataframe)),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "seed": seed,
        "negatives_per_positive": negatives_per_positive,
        "positive_negative_overlap_removed": int(dataframe.attrs.get("positive_negative_overlap_removed", 0)),
        "splits": [
            summarize_split("train", train_frame),
            summarize_split("val", val_frame),
            summarize_split("test", test_frame),
        ],
        "overlap_verified": True,
    }
    save_json(output_dir / "split_stats.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verification dataset splits from tyre and non-tyre images.")
    parser.add_argument("--positive-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--negative-roots", type=Path, nargs="+", default=[NEGATIVE_ROOT])
    parser.add_argument("--output-dir", type=Path, default=VERIFICATION_SPLITS_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negatives-per-positive", type=float, default=1.0)
    args = parser.parse_args()

    summary = create_verification_splits(
        negative_roots=args.negative_roots,
        positive_root=args.positive_root,
        output_dir=args.output_dir,
        seed=args.seed,
        negatives_per_positive=args.negatives_per_positive,
    )
    print(f"Created verification splits in {args.output_dir}")
    for split in summary["splits"]:
        print(f"{split['split']}: {split['samples']} samples | {split['class_counts']}")


if __name__ == "__main__":
    main()
