from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.common import CLASS_TO_IDX, PROCESSED_ROOT, SPLITS_ROOT, ensure_dir, save_json


def build_dataset_index(dataset_root: Path) -> pd.DataFrame:
    records = []
    for label_name, label_idx in CLASS_TO_IDX.items():
        class_dir = dataset_root / label_name
        for path in sorted(class_dir.glob("*")):
            if path.is_file():
                records.append(
                    {
                        "filepath": str(path.resolve()),
                        "label": label_name,
                        "label_idx": label_idx,
                    }
                )

    if not records:
        raise FileNotFoundError(f"No files found under {dataset_root}")

    return pd.DataFrame(records)


def verify_no_overlap(*frames: pd.DataFrame) -> None:
    sets = [set(frame["filepath"].tolist()) for frame in frames]
    for idx, current in enumerate(sets):
        for other_idx, other in enumerate(sets[idx + 1 :], start=idx + 1):
            overlap = current.intersection(other)
            if overlap:
                raise ValueError(
                    f"Found filepath overlap between splits {idx} and {other_idx}: {next(iter(overlap))}"
                )


def summarize_split(name: str, frame: pd.DataFrame) -> dict:
    class_counts = frame["label"].value_counts().sort_index().to_dict()
    return {
        "split": name,
        "samples": int(len(frame)),
        "class_counts": {key: int(value) for key, value in class_counts.items()},
        "class_ratio": {
            key: round(value / len(frame), 4) for key, value in class_counts.items()
        },
    }


def create_splits(
    dataset_root: Path = PROCESSED_ROOT,
    output_dir: Path = SPLITS_ROOT,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test sizes must sum to 1.0")

    ensure_dir(output_dir)

    dataframe = build_dataset_index(dataset_root)
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

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_frame.to_csv(train_path, index=False)
    val_frame.to_csv(val_path, index=False)
    test_frame.to_csv(test_path, index=False)

    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "total_samples": int(len(dataframe)),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "seed": seed,
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
    parser = argparse.ArgumentParser(description="Create stratified dataset splits for tyre classification.")
    parser.add_argument("--dataset-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=SPLITS_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = create_splits(dataset_root=args.dataset_root, output_dir=args.output_dir, seed=args.seed)
    print(f"Created splits in {args.output_dir}")
    for split in summary["splits"]:
        print(f"{split['split']}: {split['samples']} samples | {split['class_counts']}")


if __name__ == "__main__":
    main()
