#!/usr/bin/env python3
"""Unified evaluation across CASIA pipelines, on a single dataset at a time.

For ``--dataset interval`` we evaluate up to five pipelines:
    baseline_cnn, strip_cnn (FISNet strips), iriscode (FISNet strips),
    strip_cnn_gt (GT strips), iriscode_gt (GT strips)

For ``--dataset lamp`` we evaluate the first three (no GT available).

Pair sampling is anchored on the verification manifest (whole-frame paths)
and identical pairs are re-mapped to the corresponding strip / GT-strip paths
for the strip pipelines (pairs whose strips are missing are dropped).

Output: results/metrics_casia_<dataset>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iris_checkpoint.config import (
    CASIA_BASELINE_INTERVAL_PATH,
    CASIA_BASELINE_LAMP_PATH,
    CASIA_INTERVAL_MANIFEST,
    CASIA_INTERVAL_STRIP_MANIFEST,
    CASIA_LAMP_MANIFEST,
    CASIA_LAMP_STRIP_MANIFEST,
    CASIA_METRICS_INTERVAL_PATH,
    CASIA_METRICS_LAMP_PATH,
    CASIA_STRIPCNN_INTERVAL_PATH,
    CASIA_STRIPCNN_LAMP_PATH,
    IMG_SIZE,
    RANDOM_SEED,
    RESULTS_DIR,
    STRIP_HEIGHT,
    STRIP_WIDTH,
)
from iris_checkpoint.device import device_summary, pick_device
from iris_checkpoint.iriscode import (
    IrisCodeParams,
    default_strip_mask,
    encode_strip,
    hamming_distance,
)
from iris_checkpoint.metrics import compute_far_frr_curve, eer_and_accuracy
from iris_checkpoint.model import IrisEmbeddingCNN
from iris_checkpoint.parallel_util import thread_map, worker_cap


def sample_pairs(df, n_genuine, n_impostor, seed, path_col="path"):
    rng = np.random.default_rng(seed)
    by_sub = df.groupby("subject_id")[path_col].apply(list).to_dict()
    subs = [s for s, ps in by_sub.items() if len(ps) >= 2]
    gen, imp = [], []
    while len(gen) < n_genuine:
        s = subs[rng.integers(0, len(subs))]
        a, b = rng.choice(len(by_sub[s]), size=2, replace=False)
        gen.append((by_sub[s][int(a)], by_sub[s][int(b)]))
    all_subs = list(by_sub.keys())
    while len(imp) < n_impostor:
        s1, s2 = rng.choice(len(all_subs), size=2, replace=False)
        s1, s2 = all_subs[int(s1)], all_subs[int(s2)]
        p1 = by_sub[s1][rng.integers(0, len(by_sub[s1]))]
        p2 = by_sub[s2][rng.integers(0, len(by_sub[s2]))]
        imp.append((p1, p2))
    return gen, imp


def _load_gray(path, size):
    im = Image.open(path).convert("L")
    if size is not None:
        im = im.resize((size, size))
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).unsqueeze(0)


def _load_strip(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0).unsqueeze(0)


def embed_unique(backbone, paths, device, loader, batch_size, load_threads):
    backbone.eval()
    out = {}
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            tensors = thread_map(loader, chunk, max_workers=load_threads)
            x = torch.stack(tensors).to(device)
            z = backbone(x).float().cpu().numpy()
            for p, row in zip(chunk, z):
                out[p] = row
    return out


def cnn_pair_scores(zmap, pairs):
    a = np.stack([zmap[p[0]] for p in pairs])
    b = np.stack([zmap[p[1]] for p in pairs])
    return (a * b).sum(axis=1)


def encode_strips_to_codes(paths, params, load_threads):
    def _enc(p):
        s = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        return encode_strip(s, params)
    codes = thread_map(_enc, paths, max_workers=load_threads)
    return dict(zip(paths, codes))


def iriscode_pair_similarity(cmap, pairs, strip_h, strip_w, max_shift):
    mask = default_strip_mask(strip_h, strip_w, fraction_kept=0.7)
    out = []
    for a, b in pairs:
        hd = hamming_distance(cmap[a], cmap[b], mask, mask, max_shift=max_shift)
        out.append(1.0 - hd)
    return np.array(out, dtype=np.float64)


def evaluate_pipeline(name, val_pairs, test_pairs, score_fn):
    val_gen, val_imp = val_pairs
    test_gen, test_imp = test_pairs
    if not val_gen or not val_imp or not test_gen or not test_imp:
        return None
    val_g = score_fn(val_gen)
    val_i = score_fn(val_imp)
    val_out = eer_and_accuracy(val_g, val_i)
    tau = float(val_out["threshold"])
    test_g = score_fn(test_gen)
    test_i = score_fn(test_imp)
    far = float((test_i >= tau).mean())
    frr = float((test_g < tau).mean())
    acc = float(((test_g >= tau).sum() + (test_i < tau).sum()) / (len(test_g) + len(test_i)))
    test_eer = eer_and_accuracy(test_g, test_i)
    all_scores = np.concatenate([test_g, test_i])
    lo, hi = float(all_scores.min()), float(all_scores.max())
    pad = max(1e-6, (hi - lo) * 0.05)
    thr = np.linspace(lo - pad, hi + pad, 400)
    far_c, frr_c = compute_far_frr_curve(test_g, test_i, thr, higher_is_genuine=True)
    return {
        "name": name,
        "val_eer": float(val_out["eer"]),
        "val_threshold": tau,
        "test_far": far,
        "test_frr": frr,
        "test_accuracy": acc,
        "test_eer": float(test_eer["eer"]),
        "n_genuine_pairs": int(len(test_g)),
        "n_impostor_pairs": int(len(test_i)),
        "roc_far": far_c.tolist(),
        "roc_tpr": (1.0 - frr_c).tolist(),
        "roc_thresholds": thr.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["interval", "lamp"], required=True)
    ap.add_argument("--n-pairs", type=int, default=4000)
    ap.add_argument("--infer-batch-size", type=int, default=128)
    ap.add_argument("--load-threads", type=int, default=0)
    ap.add_argument("--max-shift", type=int, default=16)
    args = ap.parse_args()
    load_threads = args.load_threads if args.load_threads > 0 else worker_cap(12)

    if args.dataset == "interval":
        manifest_path = CASIA_INTERVAL_MANIFEST
        strip_manifest_path = CASIA_INTERVAL_STRIP_MANIFEST
        baseline_path = CASIA_BASELINE_INTERVAL_PATH
        strip_cnn_path = CASIA_STRIPCNN_INTERVAL_PATH
        out_path = CASIA_METRICS_INTERVAL_PATH
        has_gt = True
    else:
        manifest_path = CASIA_LAMP_MANIFEST
        strip_manifest_path = CASIA_LAMP_STRIP_MANIFEST
        baseline_path = CASIA_BASELINE_LAMP_PATH
        strip_cnn_path = CASIA_STRIPCNN_LAMP_PATH
        out_path = CASIA_METRICS_LAMP_PATH
        has_gt = False

    torch.manual_seed(RANDOM_SEED)
    df = pd.read_csv(manifest_path)
    df_strip = pd.read_csv(strip_manifest_path) if strip_manifest_path.exists() else None

    if df_strip is not None:
        df_strip = df_strip[df_strip["fisnet_ok"] & df_strip["strip_path"].notna()].copy()
        path_to_strip = dict(zip(df_strip["path"], df_strip["strip_path"]))
        if has_gt and "strip_path_gt" in df_strip.columns:
            df_gt = df_strip[df_strip["strip_path_gt"].notna()]
            path_to_strip_gt = dict(zip(df_gt["path"], df_gt["strip_path_gt"]))
        else:
            path_to_strip_gt = {}
    else:
        path_to_strip = None
        path_to_strip_gt = {}

    device = pick_device()
    print(f"Device: {device_summary(device)}  dataset={args.dataset}")

    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    val_pairs_img = sample_pairs(val_df, args.n_pairs, args.n_pairs, seed=RANDOM_SEED, path_col="path")
    test_pairs_img = sample_pairs(test_df, args.n_pairs, args.n_pairs, seed=RANDOM_SEED + 1, path_col="path")

    results: dict = {}

    # ----- baseline whole-frame CNN -----
    if baseline_path.exists():
        print("=== baseline_cnn ===")
        ckpt = torch.load(baseline_path, map_location=device)
        backbone = IrisEmbeddingCNN(embed_dim=ckpt.get("embed_dim", 128)).to(device)
        backbone.load_state_dict(ckpt["backbone"])
        all_paths = list({p for pair in val_pairs_img[0] + val_pairs_img[1] + test_pairs_img[0] + test_pairs_img[1] for p in pair})
        zmap = embed_unique(backbone, all_paths, device, lambda p: _load_gray(p, IMG_SIZE), args.infer_batch_size, load_threads)
        r = evaluate_pipeline("baseline_cnn", val_pairs_img, test_pairs_img, lambda pairs: cnn_pair_scores(zmap, pairs))
        if r is not None:
            results["baseline_cnn"] = r
    else:
        print(f"[skip] baseline checkpoint missing: {baseline_path}")

    def remap(pairs, mapping):
        out = [(mapping.get(a), mapping.get(b)) for a, b in pairs]
        return [m for m in out if m[0] is not None and m[1] is not None]

    # ----- FISNet strip pipelines -----
    if path_to_strip:
        val_g_s = remap(val_pairs_img[0], path_to_strip)
        val_i_s = remap(val_pairs_img[1], path_to_strip)
        test_g_s = remap(test_pairs_img[0], path_to_strip)
        test_i_s = remap(test_pairs_img[1], path_to_strip)
        print(f"FISNet strip pairs: val={len(val_g_s)}/{len(val_i_s)}  test={len(test_g_s)}/{len(test_i_s)}")

        if strip_cnn_path.exists():
            print("=== strip_cnn (FISNet strips) ===")
            ckpt = torch.load(strip_cnn_path, map_location=device)
            sm = IrisEmbeddingCNN(embed_dim=ckpt.get("embed_dim", 128)).to(device)
            sm.load_state_dict(ckpt["backbone"])
            all_s = list({p for pair in val_g_s + val_i_s + test_g_s + test_i_s for p in pair})
            zmap_s = embed_unique(sm, all_s, device, _load_strip, args.infer_batch_size, load_threads)
            r = evaluate_pipeline("strip_cnn", (val_g_s, val_i_s), (test_g_s, test_i_s),
                                  lambda pairs: cnn_pair_scores(zmap_s, pairs))
            if r is not None:
                results["strip_cnn"] = r
        else:
            print(f"[skip] strip_cnn checkpoint missing: {strip_cnn_path}")

        print("=== iriscode (FISNet strips) ===")
        params = IrisCodeParams(n_scales=2, base_wavelength=18.0, mult=1.6, sigma_on_f=0.5)
        all_s = list({p for pair in val_g_s + val_i_s + test_g_s + test_i_s for p in pair})
        cmap = encode_strips_to_codes(all_s, params, load_threads)
        r = evaluate_pipeline("iriscode", (val_g_s, val_i_s), (test_g_s, test_i_s),
                              lambda pairs: iriscode_pair_similarity(cmap, pairs, STRIP_HEIGHT, STRIP_WIDTH, args.max_shift))
        if r is not None:
            results["iriscode"] = r

    # ----- GT strip pipelines (Interval only) -----
    if has_gt and path_to_strip_gt:
        val_g_gt = remap(val_pairs_img[0], path_to_strip_gt)
        val_i_gt = remap(val_pairs_img[1], path_to_strip_gt)
        test_g_gt = remap(test_pairs_img[0], path_to_strip_gt)
        test_i_gt = remap(test_pairs_img[1], path_to_strip_gt)
        print(f"GT strip pairs: val={len(val_g_gt)}/{len(val_i_gt)}  test={len(test_g_gt)}/{len(test_i_gt)}")

        # GT iriscode (cheap; reuses FISNet-trained strip-CNN under "GT_strip_cnn" name).
        # We *don't* retrain a separate strip-CNN on GT strips because the strip
        # geometry is identical (Daugman polar) -- only the circle source changes.
        # The strip-CNN trained on FISNet strips is robust to small fit error and
        # can be reused as an oracle-input evaluator (this is a common protocol).
        if strip_cnn_path.exists():
            print("=== strip_cnn (GT strips) ===")
            ckpt = torch.load(strip_cnn_path, map_location=device)
            sm = IrisEmbeddingCNN(embed_dim=ckpt.get("embed_dim", 128)).to(device)
            sm.load_state_dict(ckpt["backbone"])
            all_s = list({p for pair in val_g_gt + val_i_gt + test_g_gt + test_i_gt for p in pair})
            zmap_gt = embed_unique(sm, all_s, device, _load_strip, args.infer_batch_size, load_threads)
            r = evaluate_pipeline("strip_cnn_gt", (val_g_gt, val_i_gt), (test_g_gt, test_i_gt),
                                  lambda pairs: cnn_pair_scores(zmap_gt, pairs))
            if r is not None:
                results["strip_cnn_gt"] = r

        print("=== iriscode (GT strips) ===")
        params = IrisCodeParams(n_scales=2, base_wavelength=18.0, mult=1.6, sigma_on_f=0.5)
        all_s = list({p for pair in val_g_gt + val_i_gt + test_g_gt + test_i_gt for p in pair})
        cmap_gt = encode_strips_to_codes(all_s, params, load_threads)
        r = evaluate_pipeline("iriscode_gt", (val_g_gt, val_i_gt), (test_g_gt, test_i_gt),
                              lambda pairs: iriscode_pair_similarity(cmap_gt, pairs, STRIP_HEIGHT, STRIP_WIDTH, args.max_shift))
        if r is not None:
            results["iriscode_gt"] = r

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    print("Summary:")
    for name, r in results.items():
        print(f"  {name:14s}  test_EER={r['test_eer']:.4f}  FAR@tau={r['test_far']:.4f}  FRR@tau={r['test_frr']:.4f}  acc={r['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
