#!/usr/bin/env python3
"""Run trained FISNet-lite over the full UBIRIS corpus and save predicted
binary iris masks (PNG, 300x400) into data/seg_pred/. Skips images that
already have a prediction so the script is resumable.
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
    MANIFEST_PATH,
    SEG_INPUT_SIZE,
    SEG_MODEL_PATH,
    SEG_NATIVE_SIZE,
    SEG_PRED_DIR,
)
from iris_checkpoint.device import device_summary, pick_device
from iris_checkpoint.segmenter import FISNetLite


def _load_image(path: str, size: int) -> torch.Tensor:
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def _predict_to_native(
    model: torch.nn.Module,
    paths: list[str],
    device: torch.device,
    size: int,
    native_hw: tuple[int, int],
    batch_size: int,
    threshold: float,
) -> list[np.ndarray]:
    masks_out: list[np.ndarray] = []
    H_native, W_native = native_hw
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            batch = torch.stack([_load_image(p, size) for p in chunk]).to(device)
            logits = model(batch)
            logits = F.interpolate(
                logits, size=(H_native, W_native), mode="bilinear", align_corners=False
            )
            prob = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)
            for k in range(prob.shape[0]):
                masks_out.append((prob[k, 0] > threshold).astype(np.uint8) * 255)
    return masks_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true", help="re-predict masks even if files exist")
    args = ap.parse_args()

    if not SEG_MODEL_PATH.exists():
        print(f"Segmenter checkpoint not found at {SEG_MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    print(f"Manifest: {len(df)} images")

    device = pick_device()
    print(f"Device: {device_summary(device)}")
    ckpt = torch.load(SEG_MODEL_PATH, map_location=device)
    size = int(ckpt.get("image_size", SEG_INPUT_SIZE))
    base_c = int(ckpt.get("base_channels", 32))
    print(
        f"Loaded segmenter: image_size={size}  base_channels={base_c}  "
        f"val_dice={ckpt.get('val_dice'):.4f}  epoch={ckpt.get('epoch')}"
    )
    model = FISNetLite(base_channels=base_c).to(device)
    model.load_state_dict(ckpt["model"])

    SEG_PRED_DIR.mkdir(parents=True, exist_ok=True)

    todo: list[tuple[int, str, Path]] = []
    for idx, row in df.iterrows():
        img_path = row["path"]
        out_name = Path(img_path).stem + ".png"
        out_path = SEG_PRED_DIR / out_name
        if not args.overwrite and out_path.exists():
            continue
        todo.append((idx, img_path, out_path))

    print(f"To predict: {len(todo)} images (existing skipped)")
    if not todo:
        return

    batch = args.batch_size
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        paths = [t[1] for t in chunk]
        masks = _predict_to_native(
            model, paths, device, size, SEG_NATIVE_SIZE, batch, args.threshold
        )
        for (_, _, out_path), m in zip(chunk, masks):
            Image.fromarray(m, mode="L").save(out_path)
        if (i // batch) % 20 == 0:
            print(f"  predicted {min(i + batch, len(todo))}/{len(todo)}")

    print(f"Done. Masks in {SEG_PRED_DIR}")


if __name__ == "__main__":
    main()
