#!/usr/bin/env python3
"""Run CASIA-trained FISNet-lite over CASIA-Interval and CASIA-Lamp.

Predicts binary iris masks at the model resolution and upsamples to each
dataset's native sensor size:
  - CASIA-Interval: (280, 320)
  - CASIA-Lamp:     (480, 640)

Predicted masks are PNGs keyed by ``<image stem>.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iris_checkpoint.config import (
    CASIA_FISNET_PATH,
    CASIA_INTERVAL_MANIFEST,
    CASIA_INTERVAL_NATIVE,
    CASIA_INTERVAL_SEG_PRED_DIR,
    CASIA_LAMP_MANIFEST,
    CASIA_LAMP_NATIVE,
    CASIA_LAMP_SEG_PRED_DIR,
    SEG_INPUT_SIZE,
)
from iris_checkpoint.device import device_summary, pick_device
from iris_checkpoint.segmenter import FISNetLite


def _load_image(path: str, size: int) -> torch.Tensor:
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def predict_dataset(
    model: torch.nn.Module,
    df: pd.DataFrame,
    out_dir: Path,
    native_hw: tuple[int, int],
    size: int,
    batch_size: int,
    threshold: float,
    device: torch.device,
    overwrite: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    todo: list[tuple[str, Path]] = []
    for _, row in df.iterrows():
        out_path = out_dir / (Path(row["path"]).stem + ".png")
        if not overwrite and out_path.exists():
            continue
        todo.append((row["path"], out_path))
    print(f"  to predict: {len(todo)} (existing skipped)")
    if not todo:
        return
    H_n, W_n = native_hw
    model.eval()
    with torch.no_grad():
        for i in range(0, len(todo), batch_size):
            chunk = todo[i : i + batch_size]
            batch = torch.stack([_load_image(p, size) for p, _ in chunk]).to(device)
            logits = model(batch)
            logits = F.interpolate(
                logits, size=(H_n, W_n), mode="bilinear", align_corners=False
            )
            prob = torch.sigmoid(logits).cpu().numpy()
            for k, (_, out_path) in enumerate(chunk):
                m = (prob[k, 0] > threshold).astype(np.uint8) * 255
                Image.fromarray(m, mode="L").save(out_path)
            if (i // batch_size) % 50 == 0:
                print(f"    predicted {min(i + batch_size, len(todo))}/{len(todo)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["interval", "lamp"], choices=["interval", "lamp"])
    args = ap.parse_args()

    if not CASIA_FISNET_PATH.exists():
        print(f"CASIA FISNet checkpoint not found at {CASIA_FISNET_PATH}", file=sys.stderr)
        sys.exit(1)

    device = pick_device()
    print(f"Device: {device_summary(device)}")
    ckpt = torch.load(CASIA_FISNET_PATH, map_location=device)
    size = int(ckpt.get("image_size", SEG_INPUT_SIZE))
    base_c = int(ckpt.get("base_channels", 32))
    val_dice = ckpt.get("val_dice")
    print(f"Loaded CASIA FISNet: size={size} base_c={base_c} val_dice={val_dice:.4f} epoch={ckpt.get('epoch')}")
    model = FISNetLite(base_channels=base_c).to(device)
    model.load_state_dict(ckpt["model"])

    if "interval" in args.datasets:
        print("\n--- CASIA-Interval ---")
        df = pd.read_csv(CASIA_INTERVAL_MANIFEST)
        print(f"  manifest rows: {len(df)}")
        predict_dataset(
            model, df, CASIA_INTERVAL_SEG_PRED_DIR, CASIA_INTERVAL_NATIVE,
            size, args.batch_size, args.threshold, device, args.overwrite,
        )
        print(f"  -> {CASIA_INTERVAL_SEG_PRED_DIR}")

    if "lamp" in args.datasets:
        print("\n--- CASIA-Lamp ---")
        df = pd.read_csv(CASIA_LAMP_MANIFEST)
        print(f"  manifest rows: {len(df)}")
        predict_dataset(
            model, df, CASIA_LAMP_SEG_PRED_DIR, CASIA_LAMP_NATIVE,
            size, args.batch_size, args.threshold, device, args.overwrite,
        )
        print(f"  -> {CASIA_LAMP_SEG_PRED_DIR}")


if __name__ == "__main__":
    main()
