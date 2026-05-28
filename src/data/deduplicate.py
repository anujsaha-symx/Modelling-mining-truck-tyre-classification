from __future__ import annotations
import json
import shutil
from pathlib import Path
import pandas as pd

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
    print(
        f"Duplicate-content analysis complete. Groups: {len(duplicate_groups_df)} | "
        f"Removed images: {len(removed_df)} | Canonical images kept: {len(canonical_df)}"
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
    return canonical_df, summary

def run_leakage_audit(
    deduplicated_df: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    dedup_summary: dict,
    output_path: Path,
) -> dict:
    hash_lookup = deduplicated_df[["processed_filepath", "content_hash"]].rename(
        columns={"processed_filepath": "filepath"}
    )
    enriched_splits: dict[str, pd.DataFrame] = {}
    for split_name, split_df in split_frames.items():
        enriched = split_df.merge(hash_lookup, on="filepath", how="left", validate="many_to_one")
        if enriched["content_hash"].isna().any():
            missing = enriched[enriched["content_hash"].isna()]["filepath"].tolist()
            raise ValueError(f"Split {split_name} contains files missing from deduplicated manifest: {missing[:5]}")
        enriched_splits[split_name] = enriched
    split_names = sorted(enriched_splits)
    hash_overlap_details: list[dict[str, object]] = []
    filepath_overlap_details: list[dict[str, object]] = []
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            left_df = enriched_splits[left_name]
            right_df = enriched_splits[right_name]
            shared_hashes = sorted(set(left_df["content_hash"]).intersection(right_df["content_hash"]))
            if shared_hashes:
                hash_overlap_details.append(
                    {
                        "split_pair": [left_name, right_name],
                        "shared_content_hashes": shared_hashes,
                        "shared_content_hash_count": len(shared_hashes),
                    }
                )
            shared_filepaths = sorted(set(left_df["filepath"]).intersection(right_df["filepath"]))
            if shared_filepaths:
                filepath_overlap_details.append(
                    {
                        "split_pair": [left_name, right_name],
                        "shared_filepaths": shared_filepaths,
                        "shared_filepath_count": len(shared_filepaths),
                    }
                )
    audit = {
        "duplicate_groups_found": int(dedup_summary.get("duplicate_groups_found", 0)),
        "removed_duplicates": int(dedup_summary.get("removed_duplicates", 0)),
        "remaining_duplicates": int(deduplicated_df["content_hash"].duplicated().sum()),
        "duplicate_groups": dedup_summary.get("duplicate_groups", []),
        "removed_duplicate_records": dedup_summary.get("removed_duplicate_records", []),
        "content_hash_leakage_detected": bool(hash_overlap_details),
        "filepath_leakage_detected": bool(filepath_overlap_details),
        "leakage_free": not hash_overlap_details and not filepath_overlap_details,
        "content_hash_overlap_details": hash_overlap_details,
        "filepath_overlap_details": filepath_overlap_details,
    }
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit