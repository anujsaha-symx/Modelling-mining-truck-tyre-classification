from __future__ import annotations
import logging
import shutil
from pathlib import Path
from typing import NamedTuple
import pandas as pd
import yaml
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
PSEUDO_BOX = (0, 0.5, 0.5, 0.9, 0.9)

class PreparedDataset(NamedTuple):
    data_yaml_path: Path
    summary_path: Path
def _clean_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
def _write_yolo_label(label_path: Path) -> None:
    class_id, x_center, y_center, width, height = PSEUDO_BOX
    label_path.write_text(
        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
        encoding="utf-8",
    )
def _build_output_stem(row, used_names: dict[str, int]) -> str:
    base_stem = f"{Path(row.filepath).parent.name}_{Path(row.filepath).stem}"
    seen_count = used_names.get(base_stem, 0)
    used_names[base_stem] = seen_count + 1
    if seen_count == 0:
        return base_stem
    return f"{base_stem}_{seen_count + 1}"

def prepare_yolo_dataset(splits_root: Path, yolo_root: Path) -> PreparedDataset:
    images_root = yolo_root / "images"
    labels_root = yolo_root / "labels"
    used_names: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        _clean_directory(images_root / split_name)
        _clean_directory(labels_root / split_name)
    for split_name in ("train", "val", "test"):
        split_csv = splits_root / f"{split_name}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
        dataframe = pd.read_csv(split_csv)
        split_counts[split_name] = len(dataframe)
        LOGGER.info("Preparing YOLO %s split with %d images.", split_name, len(dataframe))
        for row in tqdm(dataframe.itertuples(index=False), total=len(dataframe), desc=f"YOLO {split_name}", unit="image"):
            source_path = Path(row.filepath)
            if not source_path.exists():
                raise FileNotFoundError(f"Split file does not exist: {source_path}")
            output_stem = _build_output_stem(row, used_names)
            destination_image = images_root / split_name / f"{output_stem}{source_path.suffix.lower()}"
            destination_label = labels_root / split_name / f"{output_stem}.txt"
            shutil.copy2(source_path, destination_image)
            _write_yolo_label(destination_label)
    data_yaml_path = yolo_root / "data.yaml"
    yaml_content = {
        "path": str(yolo_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "tyre"},
    }
    data_yaml_path.write_text(yaml.safe_dump(yaml_content, sort_keys=False), encoding="utf-8")
    summary_path = yolo_root / "dataset_summary.yaml"
    summary_payload = {
        "pseudo_box": {
            "class_id": PSEUDO_BOX[0],
            "x_center": PSEUDO_BOX[1],
            "y_center": PSEUDO_BOX[2],
            "width": PSEUDO_BOX[3],
            "height": PSEUDO_BOX[4],
        },
        "class_names": {0: "tyre"},
        "split_counts": split_counts,
    }
    summary_path.write_text(yaml.safe_dump(summary_payload, sort_keys=False), encoding="utf-8")
    LOGGER.info("Created YOLO dataset config at %s", data_yaml_path)
    return PreparedDataset(data_yaml_path=data_yaml_path, summary_path=summary_path)