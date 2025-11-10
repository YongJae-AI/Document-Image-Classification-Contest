from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DocumentDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        transforms: Optional[Callable] = None,
        is_train: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.is_train = is_train
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        self.image_col = "ID"
        self.target_col = "target"

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_id = str(row[self.image_col])
        image_path = self.image_dir / image_id

        # Optional cache: prefer cached PNG/JPG with same basename
        if self.cache_dir is not None:
            base = Path(image_id).stem
            for ext in (".png", ".jpg", ".jpeg"):
                cand = self.cache_dir / f"{base}{ext}"
                if cand.exists():
                    image_path = cand
                    break

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = None
        if self.is_train and self.target_col in row:
            label = int(row[self.target_col])

        if self.transforms:
            if label is not None:
                augmented = self.transforms(image=image, label=label)
            else:
                augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0

        if label is not None:
            return image, label

        return image, image_id
