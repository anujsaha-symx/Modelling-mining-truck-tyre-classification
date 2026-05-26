from __future__ import annotations
import logging
from pathlib import Path
from src.data.create_splits import create_stratified_splits
from src.data.deduplicate import deduplicate_dataset, run_leakage_audit
from src.data.dataset_validator import RawDatasetSpec, validate_and_merge_datasets
from src.data.eda import generate_eda_reports
from src.data.prepare_yolo_dataset import prepare_yolo_dataset

PROJECT_ROOT = Path(__file__).resolve().parent

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def run() -> None:
    _, summary = validate_and_merge_datasets(
        dataset_specs=build_dataset_specs(),
        processed_root=PROCESSED_ROOT,
        metrics_root=METRICS_ROOT,
    )
    deduplicated_df, dedup_summary = deduplicate_dataset(
        validated_csv_path=METRICS_ROOT / "validated_dataset.csv",
        processed_root=PROCESSED_ROOT,
        metrics_root=METRICS_ROOT,
    )
    split_frames = create_stratified_splits(dataset_df=deduplicated_df, splits_root=SPLITS_ROOT)
    prepare_yolo_dataset(splits_root=SPLITS_ROOT, yolo_root=YOLO_ROOT)
    generate_eda_reports(dataset_df=deduplicated_df, summary={**summary, **dedup_summary, **leakage_audit}, output_dir=FIGURES_ROOT)
    logging.getLogger(__name__).info("Data pipeline completed successfully.")

if __name__ == "__main__":
    run()