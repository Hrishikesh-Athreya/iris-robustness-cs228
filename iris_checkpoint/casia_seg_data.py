"""Segmentation dataset for CASIA-Iris-Interval (with IRISSEG-CC GT).

GT is parametric: pupil + iris circles plus two eyelid circles. We rasterize
on the fly at the target training resolution by scaling the circle parameters
from native (280, 320) to ``image_size``.

Subject-disjoint splits are inherited from the CASIA-Interval verification
manifest (built by ``13_build_casia_manifests.py``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .casia_data import IrisGTCircles, load_casia_interval_gt, rasterize_iris_mask
from .config import (
    CASIA_INTERVAL_MANIFEST,
    CASIA_INTERVAL_NATIVE,
    SEG_INPUT_SIZE,
)


def _scale_circles(c: IrisGTCircles, sx: float, sy: float) -> IrisGTCircles:
    """Scale a circle set by (sx, sy). Use the geometric-mean scale for radius."""
    sr = float(np.sqrt(sx * sy))
    upper = (c.upper_cx, c.upper_cy, c.upper_r)
    lower = (c.lower_cx, c.lower_cy, c.lower_r)
    if upper[2] is not None:
        upper = (upper[0] * sx, upper[1] * sy, upper[2] * sr)
    if lower[2] is not None:
        lower = (lower[0] * sx, lower[1] * sy, lower[2] * sr)
    return replace(
        c,
        pupil_cx=c.pupil_cx * sx,
        pupil_cy=c.pupil_cy * sy,
        pupil_r=c.pupil_r * sr,
        iris_cx=c.iris_cx * sx,
        iris_cy=c.iris_cy * sy,
        iris_r=c.iris_r * sr,
        upper_cx=upper[0],
        upper_cy=upper[1],
        upper_r=upper[2],
        lower_cx=lower[0],
        lower_cy=lower[1],
        lower_r=lower[2],
    )


class CASIASegDataset(Dataset):
    """CASIA-Interval images + on-the-fly rasterized GT iris masks (eyelid-cut)."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        image_size: int = SEG_INPUT_SIZE,
        augment: bool = False,
    ) -> None:
        df = manifest[(manifest["split"] == split) & (manifest["has_gt"])].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No GT-paired CASIA-Interval samples in split={split}")
        self.df = df
        self.size = int(image_size)
        self.augment = augment
        self._h_native, self._w_native = CASIA_INTERVAL_NATIVE

    def __len__(self) -> int:
        return len(self.df)

    def _load_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("L").resize((self.size, self.size), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        circles = load_casia_interval_gt(row["img_key"])
        if circles is None:
            y = np.zeros((self.size, self.size), dtype=np.float32)
        else:
            sx = self.size / float(self._w_native)
            sy = self.size / float(self._h_native)
            scaled = _scale_circles(circles, sx, sy)
            mask = rasterize_iris_mask(scaled, self.size, self.size, use_eyelids=True)
            y = (mask > 127).astype(np.float32)
        return x, y

    def _augment(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            x = np.ascontiguousarray(x[:, ::-1])
            y = np.ascontiguousarray(y[:, ::-1])
        if np.random.rand() < 0.5:
            gain = np.random.uniform(0.85, 1.15)
            bias = np.random.uniform(-0.05, 0.05)
            x = np.clip(x * gain + bias, 0.0, 1.0)
        return x, y

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._load_pair(idx)
        if self.augment:
            x, y = self._augment(x, y)
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y).unsqueeze(0)


def load_casia_interval_seg_manifest(path: Path = CASIA_INTERVAL_MANIFEST) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "has_gt" not in df.columns:
        raise ValueError("manifest missing 'has_gt' column; rebuild with 13_build_casia_manifests.py")
    return df
