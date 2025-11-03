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
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.is_train = is_train

        self.image_col = "ID"
        self.target_col = "target"

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_path = self.image_dir / row[self.image_col]

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0

        if self.is_train and self.target_col in row:
            label = int(row[self.target_col])
            return image, label

        return image, row[self.image_col]

