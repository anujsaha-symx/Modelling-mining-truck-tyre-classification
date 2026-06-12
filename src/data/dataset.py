from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.utils.common import CLASS_TO_IDX


class TyreClassificationDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform=None):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]
        image_path = Path(row["filepath"])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_name = str(row["label"])
        label_idx = int(row["label_idx"])

        return {
            "image": image,
            "label": label_idx,
            "filepath": str(image_path),
            "label_name": label_name,
        }


def load_split_dataframe(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame["filepath"] = frame["filepath"].astype(str)
    frame["label"] = frame["label"].astype(str)
    frame["label_idx"] = frame["label_idx"].astype(int)
    return frame
