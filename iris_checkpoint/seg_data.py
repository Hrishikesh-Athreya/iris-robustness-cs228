"""Dataset / dataloader for IRISSEG-EP UBIRIS binary masks.

Pairs each UBIRIS image with its OperatorA mask. Subject-disjoint splits are
inherited from the existing verification manifest (manifest.csv), so no
seg-train subject ever appears in verification val/test.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import MANIFEST_PATH, SEG_GT_DIR, SEG_INPUT_SIZE, SEG_MANIFEST_PATH

_IMG_KEY_RE = re.compile(r"(C\d+_S\d+_I\d+)\.")


def _img_key(path: str) -> str | None:
    m = _IMG_KEY_RE.search(Path(path).name)
    return m.group(1) if m else None


def build_seg_manifest(verification_manifest: Path = MANIFEST_PATH) -> pd.DataFrame:
    """Join verification manifest with available IRISSEG-EP masks.

    Output columns: image_path, mask_path, subject_id, split, has_mask.
    Splits are inherited from the verification manifest so segmentation training
    cannot leak into verification val/test.
    """
    df = pd.read_csv(verification_manifest)
    df["img_key"] = df["path"].apply(_img_key)
    masks = {}
    for f in Path(SEG_GT_DIR).iterdir():
        if not f.suffix.lower() == ".tiff":
            continue
        m = re.match(r"OperatorA_(C\d+_S\d+_I\d+)\.tiff", f.name)
        if m:
            masks[m.group(1)] = str(f)
    df["mask_path"] = df["img_key"].map(masks)
    df["has_mask"] = df["mask_path"].notna()
    out = df.rename(columns={"path": "image_path"})[
        ["image_path", "mask_path", "subject_id", "split", "has_mask"]
    ]
    return out


def save_seg_manifest(out_path: Path = SEG_MANIFEST_PATH) -> pd.DataFrame:
    df = build_seg_manifest()
    df.to_csv(out_path, index=False)
    return df


class IRISSEGDataset(Dataset):
    """UBIRIS image + IRISSEG-EP binary mask pairs for a given split.

    Loads only rows with available masks. Resizes both image and mask to
    `image_size` (square) for FISNet-lite training.
    """

    def __init__(
        self,
        seg_manifest: pd.DataFrame,
        split: str,
        image_size: int = SEG_INPUT_SIZE,
        augment: bool = False,
    ) -> None:
        self.df = seg_manifest[
            (seg_manifest["split"] == split) & (seg_manifest["has_mask"])
        ].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No masked samples in split={split}")
        self.size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def _load_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.df.iloc[idx]
        im = Image.open(row["image_path"]).convert("L").resize(
            (self.size, self.size), Image.BILINEAR
        )
        mk = Image.open(row["mask_path"]).convert("L").resize(
            (self.size, self.size), Image.NEAREST
        )
        x = np.asarray(im, dtype=np.float32) / 255.0
        y = (np.asarray(mk, dtype=np.uint8) > 127).astype(np.float32)
        return x, y

    def _augment(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # cheap, deterministic-friendly augmentation: hflip + small intensity jitter
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
        x_t = torch.from_numpy(x).unsqueeze(0)  # (1, H, W)
        y_t = torch.from_numpy(y).unsqueeze(0)  # (1, H, W)
        return x_t, y_t
