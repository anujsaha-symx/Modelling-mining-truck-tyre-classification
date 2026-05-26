from __future__ import annotations
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("good", "bad")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

@dataclass(frozen=True)
class ClassificationManifestSummary:
    split_name: str
    sample_count: int
    crop_count: int
    fallback_count: int
    class_distribution: dict[str, int]
class RandomBrightnessContrast:
    def __init__(self, brightness_range: tuple[float, float] = (0.8, 1.2), contrast_range: tuple[float, float] = (0.8, 1.2), p: float = 0.5) -> None:
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        brightness_factor = random.uniform(*self.brightness_range)
        contrast_factor = random.uniform(*self.contrast_range)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        return image
class RandomMotionBlur:
    def __init__(self, kernel_sizes: tuple[int, ...] = (3, 5, 7), p: float = 0.3) -> None:
        self.kernel_sizes = kernel_sizes
        self.p = p
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        kernel_size = random.choice(self.kernel_sizes)
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        if random.random() < 0.5:
            kernel[kernel_size // 2, :] = 1.0
        else:
            kernel[:, kernel_size // 2] = 1.0
        kernel /= kernel_size
        image_array = np.array(image)
        blurred = cv2.filter2D(image_array, -1, kernel)
        return Image.fromarray(blurred)

def build_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            RandomBrightnessContrast(),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            RandomMotionBlur(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
def build_eval_transforms(image_size: int = 224) -> transforms.Compose:
    resize_size = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
def _index_crops(crops_root: Path) -> dict[str, Path]:
    if not crops_root.exists():
        return {}
    crop_map: dict[str, tuple[float, Path]] = {}
    for crop_path in crops_root.glob("*_crop_*.jpg"):
        prefix = crop_path.stem.split("_crop_", maxsplit=1)[0]
        mtime = crop_path.stat().st_mtime
        current = crop_map.get(prefix)
        if current is None or mtime >= current[0]:
            crop_map[prefix] = (mtime, crop_path)
    return {prefix: path for prefix, (_, path) in crop_map.items()}

def _resolve_input_path(source_filepath: Path, label: str, crop_lookup: dict[str, Path]) -> tuple[Path, str]:
    crop_key = f"{label}_{source_filepath.stem}"
    crop_path = crop_lookup.get(crop_key)
    if crop_path is not None and crop_path.exists():
        return crop_path, "crop"
    return source_filepath, "processed"

def load_deduplicated_manifest(deduplicated_manifest_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(deduplicated_manifest_path)
    if dataframe.empty:
        raise ValueError(f"Deduplicated manifest is empty: {deduplicated_manifest_path}")

    required_columns = {"processed_filepath", "label", "content_hash", "source_dataset"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Deduplicated manifest missing columns: {sorted(missing_columns)}")
    return dataframe

def build_classification_manifests(
    splits_root: Path,
    deduplicated_manifest_path: Path,
    crops_root: Path,
    max_samples_by_split: dict[str, int | None] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, ClassificationManifestSummary]]:
    manifest_df = load_deduplicated_manifest(deduplicated_manifest_path)
    deduplicated_lookup = manifest_df[["processed_filepath", "content_hash"]].rename(columns={"processed_filepath": "filepath"})
    crop_lookup = _index_crops(crops_root)
    manifests: dict[str, pd.DataFrame] = {}
    summaries: dict[str, ClassificationManifestSummary] = {}
    for split_name in ("train", "val", "test"):
        split_path = splits_root / f"{split_name}.csv"
        split_df = pd.read_csv(split_path)
        split_df = split_df.merge(deduplicated_lookup, on="filepath", how="left", validate="many_to_one")
        if split_df["content_hash"].isna().any():
            raise ValueError(f"Split {split_name} contains paths missing from deduplicated manifest.")
        records: list[dict[str, object]] = []
        crop_count = 0
        fallback_count = 0
        for row in split_df.itertuples(index=False):
            source_filepath = Path(row.filepath)
            if not source_filepath.exists():
                raise FileNotFoundError(f"Input file not found for classification: {source_filepath}")
            image_path, input_source = _resolve_input_path(source_filepath=source_filepath, label=row.label, crop_lookup=crop_lookup)
            if input_source == "crop":
                crop_count += 1
            else:
                fallback_count += 1
            records.append(
                {
                    "split": split_name,
                    "image_path": str(image_path.resolve()),
                    "source_filepath": str(source_filepath.resolve()),
                    "label": row.label,
                    "label_id": CLASS_TO_INDEX[row.label],
                    "source_dataset": row.source_dataset,
                    "content_hash": row.content_hash,
                    "input_source": input_source,
                }
            )
        manifest = pd.DataFrame.from_records(records)
        if max_samples_by_split and max_samples_by_split.get(split_name):
            manifest = manifest.head(int(max_samples_by_split[split_name])).copy()
        manifests[split_name] = manifest
        summaries[split_name] = ClassificationManifestSummary(
            split_name=split_name,
            sample_count=int(len(manifest)),
            crop_count=int((manifest["input_source"] == "crop").sum()),
            fallback_count=int((manifest["input_source"] == "processed").sum()),
            class_distribution={key: int(value) for key, value in manifest["label"].value_counts().to_dict().items()},
        )
        LOGGER.info(
            "Prepared %s classification manifest with %d samples (%d crops, %d fallbacks).",
            split_name,
            len(manifest),
            summaries[split_name].crop_count,
            summaries[split_name].fallback_count,
        )
    return manifests, summaries

class TyreClassificationDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform: transforms.Compose) -> None:
        if manifest.empty:
            raise ValueError("Classification dataset manifest is empty.")
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
    def __len__(self) -> int:
        return len(self.manifest)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, str]]:
        row = self.manifest.iloc[index]
        image_path = Path(row["image_path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        metadata = {
            "image_path": row["image_path"],
            "source_filepath": row["source_filepath"],
            "label": row["label"],
            "content_hash": row["content_hash"],
            "input_source": row["input_source"],
        }
        return tensor, int(row["label_id"]), metadata