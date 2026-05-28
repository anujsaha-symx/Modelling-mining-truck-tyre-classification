from __future__ import annotations
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

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
    def __init__(self, kernel_sizes: tuple[int, ...] = (3, 5, 7, 9), p: float = 0.4) -> None:
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

class RandomShadow:
    def __init__(self, shadow_strength: tuple[float, float] = (0.2, 0.7), p: float = 0.5) -> None:
        self.shadow_strength = shadow_strength
        self.p = p
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        image_array = np.array(image).astype(np.float32)
        height, width = image_array.shape[:2]
        shadow_mask = np.ones((height, width), dtype=np.float32)
        x_start = random.randint(0, width // 2)
        shadow_width = random.randint(width // 4, width // 2)
        shadow_mask[:, x_start : min(x_start + shadow_width, width)] = random.uniform(*self.shadow_strength)
        if random.random() < 0.3:
            kernel_size = random.choice([15, 25, 35])
            shadow_mask = cv2.GaussianBlur(shadow_mask, (kernel_size, kernel_size), 0)
        for channel in range(3):
            image_array[:, :, channel] *= shadow_mask
        return Image.fromarray(image_array.clip(0, 255).astype(np.uint8))
class RandomDust:
    def __init__(self, dust_count_range: tuple[int, int] = (8, 25), max_dust_size: int = 14, p: float = 0.4) -> None:
        self.dust_count_range = dust_count_range
        self.max_dust_size = max_dust_size
        self.p = p
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        image_array = np.array(image).astype(np.float32)
        height, width = image_array.shape[:2]
        dust_count = random.randint(*self.dust_count_range)
        for _ in range(dust_count):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            size = random.randint(1, self.max_dust_size)
            intensity = random.uniform(0.4, 0.85)
            is_dark = random.random() < 0.5
            cv2.circle(image_array, (x, y), size, (intensity * 255 if is_dark else 255, intensity * 255 if is_dark else 255, intensity * 255 if is_dark else 255), -1)
            if random.random() < 0.3:
                blur_size = random.choice([3, 5])
                cv2.GaussianBlur(image_array[ max(0, y - size) : min(height, y + size), max(0, x - size) : min(width, x + size) ], (blur_size, blur_size), 0)
        return Image.fromarray(image_array.clip(0, 255).astype(np.uint8))
class RandomPerspectiveDistortion:
    def __init__(self, distortion_scale: float = 0.06, p: float = 0.35) -> None:
        self.distortion_scale = distortion_scale
        self.p = p
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        width, height = image.size
        distortion = int(min(width, height) * self.distortion_scale)
        source_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        destination_points = np.float32(
            [
                [random.randint(0, distortion), random.randint(0, distortion)],
                [width - random.randint(0, distortion), random.randint(0, distortion)],
                [random.randint(0, distortion), height - random.randint(0, distortion)],
                [width - random.randint(0, distortion), height - random.randint(0, distortion)],
            ]
        )
        matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        image_array = np.array(image)
        warped = cv2.warpPerspective(image_array, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(warped)
class RandomJPEGCompression:
    def __init__(self, quality_range: tuple[int, int] = (40, 85), p: float = 0.35) -> None:
        self.quality_range = quality_range
        self.p = p
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
class RandomCoarseDropout:
    def __init__(self, hole_size: tuple[int, int] = (15, 55), hole_count: int = 3, fill_value: int = 128, p: float = 0.3) -> None:
        self.hole_size = hole_size
        self.hole_count = hole_count
        self.fill_value = fill_value
        self.p = p
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        for _ in range(self.hole_count):
            hole_height = random.randint(*self.hole_size)
            hole_width = random.randint(*self.hole_size)
            x = random.randint(0, max(width - hole_width, 1))
            y = random.randint(0, max(height - hole_height, 1))
            image_array[y : y + hole_height, x : x + hole_width] = self.fill_value
        return Image.fromarray(image_array)
class RandomLowLight:
    def __init__(self, gamma_range: tuple[float, float] = (1.5, 3.5), p: float = 0.35) -> None:
        self.gamma_range = gamma_range
        self.p = p
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        gamma = random.uniform(*self.gamma_range)
        image_array = np.array(image).astype(np.float32) / 255.0
        darkened = np.power(image_array, gamma) * 255.0
        noise = np.random.normal(0, 5.0, darkened.shape).astype(np.float32)
        return Image.fromarray((darkened + noise).clip(0, 255).astype(np.uint8))
    
def build_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0), ratio=(0.80, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            RandomPerspectiveDistortion(),
            RandomBrightnessContrast(),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            RandomMotionBlur(),
            RandomJPEGCompression(),
            RandomDust(),
            RandomShadow(),
            RandomLowLight(),
            RandomCoarseDropout(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
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

def verify_classification_leakage(
    splits_root: Path,
    deduplicated_manifest_path: Path,
    output_path: Path,
) -> dict[str, object]:
    deduplicated_df = load_deduplicated_manifest(deduplicated_manifest_path)
    hash_lookup = deduplicated_df[["processed_filepath", "content_hash"]].rename(columns={"processed_filepath": "filepath"})
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        split_path = splits_root / f"{split_name}.csv"
        split_df = pd.read_csv(split_path)
        enriched_df = split_df.merge(hash_lookup, on="filepath", how="left", validate="many_to_one")
        if enriched_df["content_hash"].isna().any():
            missing_paths = enriched_df[enriched_df["content_hash"].isna()]["filepath"].tolist()
            raise ValueError(f"Split {split_name} contains paths missing from deduplicated manifest: {missing_paths[:5]}")
        split_frames[split_name] = enriched_df
    filepath_overlap_details: list[dict[str, object]] = []
    hash_overlap_details: list[dict[str, object]] = []
    split_names = tuple(split_frames)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            left_frame = split_frames[left_name]
            right_frame = split_frames[right_name]
            filepath_overlap = sorted(set(left_frame["filepath"]).intersection(right_frame["filepath"]))
            if filepath_overlap:
                filepath_overlap_details.append(
                    {
                        "split_pair": [left_name, right_name],
                        "shared_filepaths": filepath_overlap,
                        "shared_filepath_count": len(filepath_overlap),
                    }
                )
            hash_overlap = sorted(set(left_frame["content_hash"]).intersection(right_frame["content_hash"]))
            if hash_overlap:
                hash_overlap_details.append(
                    {
                        "split_pair": [left_name, right_name],
                        "shared_content_hashes": hash_overlap,
                        "shared_content_hash_count": len(hash_overlap),
                    }
                )
    audit = {
        "filepath_leakage_detected": bool(filepath_overlap_details),
        "content_hash_leakage_detected": bool(hash_overlap_details),
        "leakage_free": not filepath_overlap_details and not hash_overlap_details,
        "filepath_overlap_details": filepath_overlap_details,
        "content_hash_overlap_details": hash_overlap_details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["leakage_free"]:
        raise ValueError("Classification leakage verification failed.")
    return audit

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
        print(f"Prepared {split_name} classification manifest with {len(manifest)} samples ({summaries[split_name].crop_count} crops, {summaries[split_name].fallback_count} fallbacks).")
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