#!/usr/bin/env python3
"""Robustness sweep on CASIA-Iris-Lamp.

End-to-end: degrade each input image, then re-segment with the CASIA-trained
FISNet-lite, rubber-sheet, and re-score with all three pipelines:
  baseline_cnn (Lamp), strip_cnn (Lamp), iriscode

Outputs results/robustness_casia_lamp.csv.
"""

from __future__ import annotations

import argparse
import csv
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
    CASIA_BASELINE_LAMP_PATH,
    CASIA_FISNET_PATH,
    CASIA_LAMP_MANIFEST,
    CASIA_LAMP_NATIVE,
    CASIA_ROBUSTNESS_LAMP_PATH,
    CASIA_STRIPCNN_LAMP_PATH,
    IMG_SIZE,
    RANDOM_SEED,
    RESULTS_DIR,
    SEG_INPUT_SIZE,
    STRIP_HEIGHT,
    STRIP_WIDTH,
)
from iris_checkpoint.degradations import DEGRADERS, SWEEPS
from iris_checkpoint.device import device_summary, pick_device
from iris_checkpoint.iriscode import (
    IrisCodeParams,
    default_strip_mask,
    encode_strip,
    hamming_distance,
)
from iris_checkpoint.metrics import eer_and_accuracy
from iris_checkpoint.model import IrisEmbeddingCNN
from iris_checkpoint.rubber_sheet import unwrap_from_mask
from iris_checkpoint.segmenter import FISNetLite


def sample_pairs(df, n, seed):
    rng = np.random.default_rng(seed)
    by_sub = df.groupby("subject_id")["path"].apply(list).to_dict()
    subs = [s for s, ps in by_sub.items() if len(ps) >= 2]
    gen, imp = [], []
    while len(gen) < n:
        s = subs[rng.integers(0, len(subs))]
        a, b = rng.choice(len(by_sub[s]), size=2, replace=False)
        gen.append((by_sub[s][int(a)], by_sub[s][int(b)]))
    all_subs = list(by_sub.keys())
    while len(imp) < n:
        s1, s2 = rng.choice(len(all_subs), size=2, replace=False)
        s1, s2 = all_subs[int(s1)], all_subs[int(s2)]
        p1 = by_sub[s1][rng.integers(0, len(by_sub[s1]))]
        p2 = by_sub[s2][rng.integers(0, len(by_sub[s2]))]
        imp.append((p1, p2))
    return gen, imp


def load_gray_uint8(p):
    return np.asarray(Image.open(p).convert("L"))


def predict_mask_native(seg, img_uint8, device, size, native):
    H, W = native
    pil = Image.fromarray(img_uint8).resize((size, size), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = seg(x)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
    return (prob > 0.5).astype(np.uint8) * 255


def degrade_one(img, axis, intensity):
    fn = DEGRADERS[axis]
    if axis in ("noise", "specular"):
        return fn(img, intensity, seed=int(intensity * 1000) + 7)
    return fn(img, intensity)


def cnn_embed_imgs(model, imgs, device, size):
    tensors = []
    for im in imgs:
        pil = Image.fromarray(im).resize((size, size))
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).unsqueeze(0))
    x = torch.stack(tensors).to(device)
    model.eval()
    with torch.no_grad():
        return model(x).float().cpu().numpy()


def cnn_embed_strips(model, strips, device):
    tensors = []
    for s in strips:
        if s.dtype == np.uint8:
            arr = s.astype(np.float32) / 255.0
        else:
            arr = s.astype(np.float32) / (s.max() + 1e-6) if s.max() > 1.5 else s.astype(np.float32)
        tensors.append(torch.from_numpy(arr).unsqueeze(0))
    x = torch.stack(tensors).to(device)
    model.eval()
    with torch.no_grad():
        return model(x).float().cpu().numpy()


def cosine_pair_scores(z, pairs):
    a = np.stack([z[p[0]] for p in pairs])
    b = np.stack([z[p[1]] for p in pairs])
    return (a * b).sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--axes", nargs="*", default=list(SWEEPS.keys()))
    args = ap.parse_args()

    torch.manual_seed(RANDOM_SEED)
    df = pd.read_csv(CASIA_LAMP_MANIFEST)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    gen_pairs, imp_pairs = sample_pairs(df_test, args.n_pairs, RANDOM_SEED + 1)
    all_paths = sorted({p for pair in gen_pairs + imp_pairs for p in pair})
    print(f"Lamp test sample: {len(gen_pairs)} gen, {len(imp_pairs)} imp; {len(all_paths)} unique imgs")

    device = pick_device()
    print(f"Device: {device_summary(device)}")

    raw = {p: load_gray_uint8(p) for p in all_paths}

    if not CASIA_FISNET_PATH.exists():
        print(f"Missing CASIA FISNet: {CASIA_FISNET_PATH}", file=sys.stderr)
        sys.exit(1)
    seg_ckpt = torch.load(CASIA_FISNET_PATH, map_location=device)
    seg = FISNetLite(base_channels=int(seg_ckpt.get("base_channels", 32))).to(device)
    seg.load_state_dict(seg_ckpt["model"])
    seg_size = int(seg_ckpt.get("image_size", SEG_INPUT_SIZE))

    base = None
    if CASIA_BASELINE_LAMP_PATH.exists():
        b_ckpt = torch.load(CASIA_BASELINE_LAMP_PATH, map_location=device)
        base = IrisEmbeddingCNN(embed_dim=b_ckpt.get("embed_dim", 128)).to(device)
        base.load_state_dict(b_ckpt["backbone"])

    strip_cnn = None
    if CASIA_STRIPCNN_LAMP_PATH.exists():
        s_ckpt = torch.load(CASIA_STRIPCNN_LAMP_PATH, map_location=device)
        strip_cnn = IrisEmbeddingCNN(embed_dim=s_ckpt.get("embed_dim", 128)).to(device)
        strip_cnn.load_state_dict(s_ckpt["backbone"])

    code_params = IrisCodeParams(n_scales=2, base_wavelength=18.0, mult=1.6, sigma_on_f=0.5)
    strip_mask = default_strip_mask(STRIP_HEIGHT, STRIP_WIDTH, fraction_kept=0.7)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = CASIA_ROBUSTNESS_LAMP_PATH
    done: set[tuple[str, float]] = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = {(str(r.axis), float(r.intensity)) for r in prev.itertuples()}
        print(f"resume: {len(done)} (axis, intensity) pairs already in {out_csv}")
    else:
        with open(out_csv, "w", newline="") as f:
            csv.writer(f).writerow(["axis", "intensity", "pipeline", "eer", "far_at_eer", "frr_at_eer", "n_genuine", "n_impostor"])

    for axis in args.axes:
        for level in SWEEPS[axis]:
            if (axis, float(level)) in done:
                print(f"--- axis={axis} intensity={level}: skip (already done)", flush=True)
                continue
            print(f"\n--- axis={axis} intensity={level} ---", flush=True)
            degraded = {p: degrade_one(raw[p], axis, level) for p in all_paths}

            base_z = {}
            if base is not None:
                paths = list(degraded.keys())
                z = cnn_embed_imgs(base, [degraded[p] for p in paths], device, IMG_SIZE)
                base_z = dict(zip(paths, z))

            strips = {}
            for p in all_paths:
                m = predict_mask_native(seg, degraded[p], device, seg_size, CASIA_LAMP_NATIVE)
                s, _ = unwrap_from_mask(degraded[p].astype(np.float32), m, STRIP_HEIGHT, STRIP_WIDTH)
                if s is not None:
                    strips[p] = s

            valid_gen = [(a, b) for a, b in gen_pairs if a in strips and b in strips]
            valid_imp = [(a, b) for a, b in imp_pairs if a in strips and b in strips]

            strip_z = {}
            if strip_cnn is not None and strips:
                paths = list(strips.keys())
                z = cnn_embed_strips(strip_cnn, [strips[p] for p in paths], device)
                strip_z = dict(zip(paths, z))

            codes = {p: encode_strip(strips[p], code_params) for p in strips}

            rows = []
            if base is not None:
                gs = cosine_pair_scores(base_z, gen_pairs)
                ims = cosine_pair_scores(base_z, imp_pairs)
                r = eer_and_accuracy(gs, ims)
                rows.append(("baseline_cnn", r["eer"], r["far_at_eer"], r["frr_at_eer"], len(gs), len(ims)))
            if strip_cnn is not None and valid_gen and valid_imp:
                gs = cosine_pair_scores(strip_z, valid_gen)
                ims = cosine_pair_scores(strip_z, valid_imp)
                r = eer_and_accuracy(gs, ims)
                rows.append(("strip_cnn", r["eer"], r["far_at_eer"], r["frr_at_eer"], len(gs), len(ims)))
            if codes and valid_gen and valid_imp:
                gs = np.array([1.0 - hamming_distance(codes[a], codes[b], strip_mask, strip_mask, max_shift=16) for a, b in valid_gen])
                ims = np.array([1.0 - hamming_distance(codes[a], codes[b], strip_mask, strip_mask, max_shift=16) for a, b in valid_imp])
                r = eer_and_accuracy(gs, ims)
                rows.append(("iriscode", r["eer"], r["far_at_eer"], r["frr_at_eer"], len(gs), len(ims)))

            with open(out_csv, "a", newline="") as f:
                w = csv.writer(f)
                for name, eer, far, frr, ng, ni in rows:
                    w.writerow([axis, level, name, eer, far, frr, ng, ni])
                    print(f"  {name:14s}  EER={eer:.4f}", flush=True)

    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
