from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.common import ANNOTATED_ROOT, SPLITS_ROOT, ensure_dir, save_json


ANNOTATED_CLASS_TO_LABEL = {
    "good": 1,
    "bad": 1,
    "negative": 0,
}


def build_dataset_index(dataset_root: Path) -> pd.DataFrame:
    records = []
    for class_name, label in ANNOTATED_CLASS_TO_LABEL.items():
        class_dir = dataset_root / class_name
        for path in sorted(class_dir.glob("*")):
            if path.is_file():
                records.append(
                    {
                        "filepath": str(path.resolve()),
                        "label": label,
                        "class_name": class_name,
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
    return {
        "count": int(len(frame)),
        "good": int((frame["class_name"] == "good").sum()),
        "bad": int((frame["class_name"] == "bad").sum()),
        "negative": int((frame["class_name"] == "negative").sum()),
    }


def create_splits(
    dataset_root: Path = ANNOTATED_ROOT,
    output_dir: Path = SPLITS_ROOT,
    train_size: float = 0.70,
    val_size: float = 0.10,
    test_size: float = 0.20,
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
        stratify=dataframe["label"],
        random_state=seed,
    )

    relative_val = val_size / (val_size + test_size)
    val_frame, test_frame = train_test_split(
        temp_frame,
        test_size=(1.0 - relative_val),
        stratify=temp_frame["label"],
        random_state=seed,
    )

    train_frame = train_frame.sort_values("filepath").reset_index(drop=True)
    val_frame = val_frame.sort_values("filepath").reset_index(drop=True)
    test_frame = test_frame.sort_values("filepath").reset_index(drop=True)

    verify_no_overlap(train_frame, val_frame, test_frame)

    csv_columns = ["filepath", "label", "class_name"]
    train_frame[csv_columns].to_csv(output_dir / "train.csv", index=False)
    val_frame[csv_columns].to_csv(output_dir / "val.csv", index=False)
    test_frame[csv_columns].to_csv(output_dir / "test.csv", index=False)

    summary = {
        "total_images": int(len(dataframe)),
        "train": summarize_split("train", train_frame),
        "val": summarize_split("val", val_frame),
        "test": summarize_split("test", test_frame),
    }

    save_json(output_dir / "split_stats.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create stratified dataset splits for tyre classification from annotated dataset."
    )
    parser.add_argument("--dataset-root", type=Path, default=ANNOTATED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=SPLITS_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = create_splits(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print(f"Total images: {summary['total_images']}")
    print()
    for split_name in ["train", "val", "test"]:
        s = summary[split_name]
        print(f"{split_name}: {s['count']} | good={s['good']}, bad={s['bad']}, negative={s['negative']}")

    print()
    train_df = pd.read_csv(args.output_dir / "train.csv")
    val_df = pd.read_csv(args.output_dir / "val.csv")
    test_df = pd.read_csv(args.output_dir / "test.csv")

    train_paths = set(train_df["filepath"].tolist())
    val_paths = set(val_df["filepath"].tolist())
    test_paths = set(test_df["filepath"].tolist())

    print("Leakage check:")
    print(f"  train vs val:  {len(train_paths & val_paths)} overlapping")
    print(f"  train vs test: {len(train_paths & test_paths)} overlapping")
    print(f"  val vs test:   {len(val_paths & test_paths)} overlapping")

    all_split_paths = train_paths | val_paths | test_paths
    all_annotated = set()
    for class_name in ANNOTATED_CLASS_TO_LABEL:
        for p in (args.dataset_root / class_name).glob("*"):
            if p.is_file():
                all_annotated.add(str(p.resolve()))

    missing = all_annotated - all_split_paths
    extra = all_split_paths - all_annotated
    if missing or extra:
        print(f"  Missing from splits: {len(missing)}, Extra in splits: {len(extra)}")
    else:
        print("  Every annotated image appears exactly once. OK")
    print(f"\nFinal counts: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")


if __name__ == "__main__":
    main()
