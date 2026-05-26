from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path
import pandas as pd

LOGGER = logging.getLogger(__name__)

def load_validated_dataset(validated_csv_path: Path) -> pd.DataFrame:
    if not validated_csv_path.exists():
        raise FileNotFoundError(f"Validated dataset CSV not found: {validated_csv_path}")
    dataframe = pd.read_csv(validated_csv_path)
    if dataframe.empty:
        raise ValueError("Validated dataset CSV is empty.")
    required_columns = {
        "source_dataset",
        "label",
        "raw_filepath",
        "filename",
        "extension",
        "processed_filepath",
        "width",
        "height",
        "image_mode",
        "file_size_bytes",
        "content_hash",
    }
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Validated dataset CSV is missing required columns: {sorted(missing_columns)}")
    return dataframe

def deduplicate_dataset(
    validated_csv_path: Path,
    processed_root: Path,
    metrics_root: Path,
) -> tuple[pd.DataFrame, dict]:
    dataset_df = load_validated_dataset(validated_csv_path)
    dataset_df = dataset_df.sort_values(
        by=["content_hash", "source_dataset", "label", "raw_filepath", "filename"],
        kind="mergesort",
    ).reset_index(drop=True)
    duplicate_groups_df = dataset_df.groupby("content_hash", sort=True).agg(
        duplicate_count=("content_hash", "size"),
        labels=("label", lambda values: sorted(set(values))),
        source_datasets=("source_dataset", lambda values: sorted(set(values))),
        raw_filepaths=("raw_filepath", list),
    )
    duplicate_groups_df = duplicate_groups_df[duplicate_groups_df["duplicate_count"] > 1].reset_index()
    keep_mask = ~dataset_df.duplicated(subset=["content_hash"], keep="first")
    canonical_df = dataset_df[keep_mask].copy()
    removed_df = dataset_df[~keep_mask].copy()
    LOGGER.info(
        "Duplicate-content analysis complete. Groups: %d | Removed images: %d | Canonical images kept: %d",
        len(duplicate_groups_df),
        len(removed_df),
        len(canonical_df),
    )
    for label_dir in (processed_root / "good", processed_root / "bad"):
        if label_dir.exists():
            shutil.rmtree(label_dir)
        label_dir.mkdir(parents=True, exist_ok=True)
    refreshed_paths: list[str] = []
    for row in canonical_df.itertuples(index=False):
        raw_path = Path(row.raw_filepath)
        destination_name = f"{row.source_dataset}_{row.label}_{raw_path.stem}{raw_path.suffix.lower()}"
        destination_path = processed_root / row.label / destination_name
        shutil.copy2(raw_path, destination_path)
        refreshed_paths.append(str(destination_path.resolve()))
    canonical_df.loc[:, "processed_filepath"] = refreshed_paths
    deduplicated_dataset_path = metrics_root / "deduplicated_dataset.csv"
    canonical_df.to_csv(deduplicated_dataset_path, index=False)
    summary = {
        "total_validated_images": int(len(dataset_df)),
        "duplicate_groups_found": int(len(duplicate_groups_df)),
        "removed_duplicates": int(len(removed_df)),
        "remaining_duplicates": int(canonical_df["content_hash"].duplicated().sum()),
        "canonical_images_retained": int(len(canonical_df)),
        "label_conflict_groups": int(
            duplicate_groups_df["labels"].apply(lambda labels: len(labels) > 1).sum()
        ),
        "duplicate_groups": duplicate_groups_df.to_dict("records"),
        "removed_duplicate_records": removed_df[
            ["content_hash", "source_dataset", "label", "raw_filepath", "processed_filepath"]
        ].to_dict("records"),
    }
    LOGGER.info("Saved deduplicated dataset manifest to %s", deduplicated_dataset_path)
    return canonical_df, summary