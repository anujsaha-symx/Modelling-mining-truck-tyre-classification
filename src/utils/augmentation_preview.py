from __future__ import annotations
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data.classification_dataset import build_train_transforms  # noqa: E402

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Preview augmentations on sample images.")
    parser.add_argument("--image", default=None, help="Path to a single input image. If omitted, samples from processed dataset.")
    parser.add_argument("--output", default=None, help="Output path or directory.")
    parser.add_argument("--image-size", type=int, default=224, help="Target image size.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of augmented versions to show.")
    return parser.parse_args()

def _generate_preview_grid(image: Image.Image, num_samples: int, image_size: int, output_path: Path) -> None:
    transform = build_train_transforms(image_size=image_size)
    total_cols = num_samples + 1
    figure, axes = plt.subplots(1, total_cols, figsize=(total_cols * 3, 3.5))
    axes[0].imshow(np.array(image))
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")
    for sample_index in range(num_samples):
        augmented = transform(image)
        augmented_image = augmented.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        augmented_image = std * augmented_image + mean
        augmented_image = augmented_image.clip(0, 1)
        axes[sample_index + 1].imshow(augmented_image)
        axes[sample_index + 1].set_title(f"Aug #{sample_index + 1}", fontsize=10)
        axes[sample_index + 1].axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved augmentation preview to {output_path}")

def run(args: Namespace | None = None) -> None:
    args = args or parse_args()
    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / "classification" / "augmentation_preview"
    if args.image:
        image_path = Path(args.image)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        output_path = output_dir / f"{image_path.stem}_preview.png"
        _generate_preview_grid(image=image, num_samples=args.num_samples, image_size=args.image_size, output_path=output_path)
        return
    processed_root = PROJECT_ROOT / "datasets" / "processed"
    sample_images = []
    for label in ("good", "bad"):
        label_dir = processed_root / label
        images = sorted(label_dir.glob("*.jpg")) + sorted(label_dir.glob("*.png"))
        if images:
            sample_images.append((label, images[0]))
            if len(images) > 1:
                sample_images.append((label, images[len(images) // 2]))
    if not sample_images:
        print(f"No sample images found in {processed_root}", file=sys.stderr)
        return
    for label, img_path in sample_images:
        with Image.open(img_path) as image:
            image = image.convert("RGB")
        output_path = output_dir / f"sample_{label}_{img_path.stem}_preview.png"
        _generate_preview_grid(image=image, num_samples=args.num_samples, image_size=args.image_size, output_path=output_path)
    print(f"Generated {len(sample_images)} augmentation previews in {output_dir}")

if __name__ == "__main__":
    run()