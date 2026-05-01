#!/usr/bin/env python3
"""Generate Daugman rubber-sheet strips for the entire corpus.

For each image in manifest.csv:
  1. Load image (grayscale).
  2. Load the predicted mask from data/seg_pred/ (FISNet-lite output).
  3. Fit pupil + iris circles, unwrap to STRIP_HEIGHT x STRIP_WIDTH strip.
  4. Save strip as PNG and emit a row in strip_manifest.csv.

Images whose mask is too small/degenerate are dropped from the strip manifest
(logged for diagnostics).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iris_checkpoint.config import (
    MANIFEST_PATH,
    SEG_PRED_DIR,
    STRIP_DIR,
    STRIP_HEIGHT,
    STRIP_MANIFEST_PATH,
    STRIP_WIDTH,
)
from iris_checkpoint.parallel_util import worker_cap
from iris_checkpoint.rubber_sheet import unwrap_from_mask


def _process_one(row: dict, strip_h: int, strip_w: int) -> dict:
    img_path = row["path"]
    mask_path = SEG_PRED_DIR / (Path(img_path).stem + ".png")
    if not mask_path.exists():
        return {**row, "strip_path": None, "ok": False, "reason": "no_mask"}
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L"))
    strip, circles = unwrap_from_mask(img, mask, strip_h=strip_h, strip_w=strip_w)
    if strip is None:
        return {**row, "strip_path": None, "ok": False, "reason": "fit_fail"}
    out_name = Path(img_path).stem + ".png"
    out_path = STRIP_DIR / out_name
    Image.fromarray(np.clip(strip, 0, 255).astype(np.uint8)).save(out_path)
    return {
        **row,
        "strip_path": str(out_path),
        "ok": True,
        "reason": "",
        "pupil_r": circles.pupil_r,
        "iris_r": circles.iris_r,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    workers = args.workers if args.workers > 0 else worker_cap(8)

    df = pd.read_csv(MANIFEST_PATH).to_dict(orient="records")
    STRIP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building {len(df)} strips with {workers} workers...")

    rows: list[dict] = []
    if workers <= 1:
        for r in df:
            rows.append(_process_one(r, STRIP_HEIGHT, STRIP_WIDTH))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_one, r, STRIP_HEIGHT, STRIP_WIDTH) for r in df]
            for i, fut in enumerate(futs):
                rows.append(fut.result())
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(df)}")

    out = pd.DataFrame(rows)
    out.to_csv(STRIP_MANIFEST_PATH, index=False)
    ok = int(out["ok"].sum())
    print(f"Strips written: {ok}/{len(out)}  manifest: {STRIP_MANIFEST_PATH}")
    if (~out["ok"]).any():
        print("Failure reasons:")
        print(out[~out["ok"]]["reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
