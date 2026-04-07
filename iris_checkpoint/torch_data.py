from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import IMG_SIZE


class IrisImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        subject_to_idx: dict[str, int] | None = None,
        size: int = IMG_SIZE,
    ):
        self.paths = df["path"].tolist()
        self.subjects = df["subject_id"].astype(str).tolist()
        if subject_to_idx is None:
            uniq = sorted(set(self.subjects))
            subject_to_idx = {s: i for i, s in enumerate(uniq)}
        self.subject_to_idx = subject_to_idx
        self.labels = [subject_to_idx[s] for s in self.subjects]
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        p = self.paths[i]
        im = Image.open(p).convert("L").resize((self.size, self.size))
        x = torch.from_numpy(np.array(im, dtype=np.float32) / 255.0).unsqueeze(0)
        return x, self.labels[i]
