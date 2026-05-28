from __future__ import annotations
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

@dataclass(frozen=True)
class RawDatasetSpec:
    name: str
    root: Path
def scan_raw_datasets(dataset_specs: Iterable[RawDatasetSpec]) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for spec in dataset_specs:
        print(f"Scanning dataset: {spec.name}")
        for label_dir in sorted(spec.root.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name.lower()
            for file_path in label_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                    records.append(
                        {
                            "source_dataset": spec.name,
                            "label": label,
                            "raw_filepath": str(file_path.resolve()),
                            "filename": file_path.name,
                            "extension": file_path.suffix.lower(),
                        }
                    )
    if not records:
        raise FileNotFoundError("No supported image files were found in the configured raw datasets.")
    dataframe = pd.DataFrame(records)
    print(f"Discovered {len(dataframe)} image candidates.")
    return dataframe

def _compute_file_hash(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def validate_and_merge_datasets(
    dataset_specs: Iterable[RawDatasetSpec],
    processed_root: Path,
    metrics_root: Path,
) -> tuple[pd.DataFrame, dict]:
    processed_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    for label_dir in (processed_root / "good", processed_root / "bad"):
        if label_dir.exists():
            shutil.rmtree(label_dir)
    raw_df = scan_raw_datasets(dataset_specs)
    duplicate_filename_counts = raw_df["filename"].value_counts()
    duplicate_filenames = duplicate_filename_counts[duplicate_filename_counts > 1].to_dict()
    valid_records: list[dict[str, object]] = []
    invalid_records: list[dict[str, object]] = []
    hash_counter: Counter[str] = Counter()
    for record in tqdm(raw_df.to_dict("records"), desc="Validating images", unit="image"):
        file_path = Path(record["raw_filepath"])
        try:
            with Image.open(file_path) as image:
                image.verify()
            with Image.open(file_path) as image:
                width, height = image.size
                image_mode = image.mode
            file_hash = _compute_file_hash(file_path)
            hash_counter[file_hash] += 1
            destination_name = f"{record['source_dataset']}_{record['label']}_{file_path.stem}{file_path.suffix.lower()}"
            destination_path = processed_root / str(record["label"]) / destination_name
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, destination_path)
            valid_records.append(
                {
                    **record,
                    "processed_filepath": str(destination_path.resolve()),
                    "width": width,
                    "height": height,
                    "image_mode": image_mode,
                    "file_size_bytes": file_path.stat().st_size,
                    "content_hash": file_hash,
                }
            )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            invalid_records.append({**record, "error": str(exc)})
    valid_df = pd.DataFrame(valid_records)
    invalid_df = pd.DataFrame(
        invalid_records,
        columns=["source_dataset", "label", "raw_filepath", "filename", "extension", "error"],
    )
    duplicate_content_counts = Counter({file_hash: count for file_hash, count in hash_counter.items() if count > 1})
    summary = {
        "total_discovered": int(len(raw_df)),
        "total_valid": int(len(valid_df)),
        "total_invalid": int(len(invalid_df)),
        "class_distribution": raw_df["label"].value_counts().to_dict(),
        "valid_class_distribution": valid_df["label"].value_counts().to_dict() if not valid_df.empty else {},
        "source_distribution": raw_df["source_dataset"].value_counts().to_dict(),
        "duplicate_filenames": duplicate_filenames,
        "duplicate_filename_count": int(sum(count - 1 for count in duplicate_filenames.values())),
        "duplicate_content_hashes": dict(duplicate_content_counts),
        "duplicate_content_count": int(sum(count - 1 for count in duplicate_content_counts.values())),
        "image_width": {
            "min": int(valid_df["width"].min()) if not valid_df.empty else None,
            "max": int(valid_df["width"].max()) if not valid_df.empty else None,
        },
        "image_height": {
            "min": int(valid_df["height"].min()) if not valid_df.empty else None,
            "max": int(valid_df["height"].max()) if not valid_df.empty else None,
        },
    }
    valid_output = metrics_root / "validated_dataset.csv"
    invalid_output = metrics_root / "invalid_files.csv"
    summary_output = metrics_root / "validation_summary.json"
    valid_df.to_csv(valid_output, index=False)
    invalid_df.to_csv(invalid_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Validation complete. Valid: {len(valid_df)} | Invalid: {len(invalid_df)}")
    print(f"Merged valid images into {processed_root}")
    return valid_df, summary