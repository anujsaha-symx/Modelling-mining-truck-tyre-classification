from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def create_stratified_splits(
    dataset_df: pd.DataFrame,
    splits_root: Path,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    if dataset_df.empty:
        raise ValueError("Cannot create splits from an empty dataset.")
    splits_root.mkdir(parents=True, exist_ok=True)
    split_df = dataset_df[["processed_filepath", "label", "source_dataset"]].rename(
        columns={"processed_filepath": "filepath"}
    )
    train_val_df, test_df = train_test_split(
        split_df,
        test_size=0.15,
        stratify=split_df["label"],
        random_state=random_state,
    )
    val_fraction = 0.15 / 0.85
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        stratify=train_val_df["label"],
        random_state=random_state,
    )
    outputs = {"train": train_df, "val": val_df, "test": test_df}
    for split_name, dataframe in outputs.items():
        output_path = splits_root / f"{split_name}.csv"
        dataframe.sort_values(by="filepath").to_csv(output_path, index=False)
        print(f"Saved {split_name} split with {len(dataframe)} rows to {output_path}")
    metadata = {
        split_name: {
            "total": int(len(dataframe)),
            "class_distribution": dataframe["label"].value_counts().to_dict(),
            "source_distribution": dataframe["source_dataset"].value_counts().to_dict(),
        }
        for split_name, dataframe in outputs.items()
    }
    (splits_root / "split_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return outputs