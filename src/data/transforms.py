from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class LetterboxResize:
    size: int | tuple[int, int] = 224
    fill: tuple[int, int, int] = (0, 0, 0)
    interpolation: InterpolationMode = InterpolationMode.BILINEAR

    def __post_init__(self) -> None:
        if isinstance(self.size, int):
            self.target_width = self.size
            self.target_height = self.size
        else:
            self.target_height, self.target_width = self.size

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        src_width, src_height = image.size
        scale = min(self.target_width / src_width, self.target_height / src_height)
        new_width = max(1, int(round(src_width * scale)))
        new_height = max(1, int(round(src_height * scale)))

        pil_resampling = getattr(Image, "Resampling", Image)
        resample_map = {
            InterpolationMode.NEAREST: pil_resampling.NEAREST,
            InterpolationMode.BILINEAR: pil_resampling.BILINEAR,
            InterpolationMode.BICUBIC: pil_resampling.BICUBIC,
            InterpolationMode.BOX: pil_resampling.BOX,
            InterpolationMode.HAMMING: pil_resampling.HAMMING,
            InterpolationMode.LANCZOS: pil_resampling.LANCZOS,
        }
        resized = image.resize((new_width, new_height), resample=resample_map[self.interpolation])

        pad_left = (self.target_width - new_width) // 2
        pad_top = (self.target_height - new_height) // 2
        pad_right = self.target_width - new_width - pad_left
        pad_bottom = self.target_height - new_height - pad_top

        return ImageOps.expand(
            resized,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
        )


def build_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            LetterboxResize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8, fill=(0, 0, 0)),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.08,
                hue=0.02,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                p=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            LetterboxResize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
