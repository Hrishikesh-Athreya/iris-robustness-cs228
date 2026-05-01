#!/usr/bin/env python3
"""Build Daugman rubber-sheet strips for CASIA datasets.

Generates STRIP_HEIGHT x STRIP_WIDTH grayscale strips using either:
  * FISNet-lite predicted mask -> circle fit -> unwrap (Interval + Lamp).
  * IRISSEG-CC GT circles -> direct unwrap (Interval only, GT-oracle ablation).

Outputs:
  data/casia_interval_strips/<key>.png         (FISNet)
  data/casia_interval_strips_gt/<key>.png      (GT circles)
  data/casia_lamp_strips/<key>.png             (FISNet)
  strip_manifest_casia_interval.csv            (one row per image with both
                                                 strip_path (FISNet) and
                                                 strip_path_gt where available)
  strip_manifest_casia_lamp.csv
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

from iris_checkpoint.casia_data import load_casia_interval_gt, to_iris_circles
from iris_checkpoint.config import (
    CASIA_INTERVAL_MANIFEST,
    CASIA_INTERVAL_SEG_PRED_DIR,
    CASIA_INTERVAL_STRIP_DIR,
    CASIA_INTERVAL_STRIP_GT_DIR,
    CASIA_INTERVAL_STRIP_MANIFEST,
    CASIA_LAMP_MANIFEST,
    CASIA_LAMP_SEG_PRED_DIR,
    CASIA_LAMP_STRIP_DIR,
    CASIA_LAMP_STRIP_MANIFEST,
    STRIP_HEIGHT,
    STRIP_WIDTH,
)
from iris_checkpoint.rubber_sheet import unwrap_from_mask, unwrap_iris


def _save_strip(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def _process_interval(row: dict, sh: int, sw: int) -> dict:
    img_path = row["path"]
    key = Path(img_path).stem
    out: dict = {**row}
    out["strip_path"] = None
    out["strip_path_gt"] = None
    out["fisnet_ok"] = False
    out["gt_ok"] = False
    out["reason_fisnet"] = ""
    out["reason_gt"] = ""

    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)

    # FISNet pathway
    mask_path = CASIA_INTERVAL_SEG_PRED_DIR / f"{key}.png"
    if not mask_path.exists():
        out["reason_fisnet"] = "no_mask"
    else:
        mask = np.asarray(Image.open(mask_path).convert("L"))
        strip, _ = unwrap_from_mask(img, mask, strip_h=sh, strip_w=sw)
        if strip is None:
            out["reason_fisnet"] = "fit_fail"
        else:
            sp = CASIA_INTERVAL_STRIP_DIR / f"{key}.png"
            _save_strip(strip, sp)
            out["strip_path"] = str(sp)
            out["fisnet_ok"] = True

    # GT pathway
    if not row.get("has_gt", False):
        out["reason_gt"] = "no_gt"
    else:
        circles = load_casia_interval_gt(key)
        if circles is None:
            out["reason_gt"] = "gt_load_fail"
        else:
            strip = unwrap_iris(img, to_iris_circles(circles), strip_h=sh, strip_w=sw)
            sp_gt = CASIA_INTERVAL_STRIP_GT_DIR / f"{key}.png"
            _save_strip(strip, sp_gt)
            out["strip_path_gt"] = str(sp_gt)
            out["gt_ok"] = True

    return out


def _process_lamp(row: dict, sh: int, sw: int) -> dict:
    img_path = row["path"]
    key = Path(img_path).stem
    out: dict = {**row, "strip_path": None, "fisnet_ok": False, "reason_fisnet": ""}
    mask_path = CASIA_LAMP_SEG_PRED_DIR / f"{key}.png"
    if not mask_path.exists():
        out["reason_fisnet"] = "no_mask"
        return out
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L"))
    strip, _ = unwrap_from_mask(img, mask, strip_h=sh, strip_w=sw)
    if strip is None:
        out["reason_fisnet"] = "fit_fail"
        return out
    sp = CASIA_LAMP_STRIP_DIR / f"{key}.png"
    _save_strip(strip, sp)
    out["strip_path"] = str(sp)
    out["fisnet_ok"] = True
    return out


def _run(rows, fn, label):
    print(f"  {label}: {len(rows)} images", flush=True)
    out = []
    for i, r in enumerate(rows):
        out.append(fn(r, STRIP_HEIGHT, STRIP_WIDTH))
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(rows)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["interval", "lamp"], choices=["interval", "lamp"])
    args = ap.parse_args()

    if "interval" in args.datasets:
        print("--- CASIA-Interval strips ---")
        CASIA_INTERVAL_STRIP_DIR.mkdir(parents=True, exist_ok=True)
        CASIA_INTERVAL_STRIP_GT_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(CASIA_INTERVAL_MANIFEST).to_dict(orient="records")
        rows = _run(df, _process_interval, "Interval")
        out = pd.DataFrame(rows)
        out.to_csv(CASIA_INTERVAL_STRIP_MANIFEST, index=False)
        print(f"  FISNet ok: {int(out['fisnet_ok'].sum())}/{len(out)}")
        print(f"  GT     ok: {int(out['gt_ok'].sum())}/{len(out)}")
        if (~out["fisnet_ok"]).any():
            print(f"  FISNet failures: {out[~out['fisnet_ok']]['reason_fisnet'].value_counts().to_dict()}")
        print(f"  manifest -> {CASIA_INTERVAL_STRIP_MANIFEST}")

    if "lamp" in args.datasets:
        print("\n--- CASIA-Lamp strips ---")
        CASIA_LAMP_STRIP_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(CASIA_LAMP_MANIFEST).to_dict(orient="records")
        rows = _run(df, _process_lamp, "Lamp")
        out = pd.DataFrame(rows)
        out.to_csv(CASIA_LAMP_STRIP_MANIFEST, index=False)
        print(f"  FISNet ok: {int(out['fisnet_ok'].sum())}/{len(out)}")
        if (~out["fisnet_ok"]).any():
            print(f"  FISNet failures: {out[~out['fisnet_ok']]['reason_fisnet'].value_counts().to_dict()}")
        print(f"  manifest -> {CASIA_LAMP_STRIP_MANIFEST}")


if __name__ == "__main__":
    main()
