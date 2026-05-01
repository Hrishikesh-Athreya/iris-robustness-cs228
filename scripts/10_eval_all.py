#!/usr/bin/env python3
"""Evaluate all three pipelines on identical val/test pair samples.

Pipelines:
  * baseline_cnn  -- whole-frame CNN from Checkpoint 1 (cosine similarity).
  * strip_cnn     -- same architecture trained on rubber-sheet strips.
  * iriscode      -- log-Gabor IrisCode + Hamming similarity.

For each pipeline we sweep thresholds on the val split to pick tau*, then
report test FAR/FRR/EER and accuracy. ROC arrays are stored for plotting.
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
    ADVANCED_METRICS_PATH,
    IMG_SIZE,
    MANIFEST_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    RESULTS_DIR,
    STRIP_CNN_PATH,
    STRIP_HEIGHT,
    STRIP_MANIFEST_PATH,
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


# ---- pair sampling -----------------------------------------------------------


def sample_pairs(
    df: pd.DataFrame,
    n_genuine: int,
    n_impostor: int,
    seed: int,
    path_col: str = "path",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
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


# ---- CNN scoring (whole-frame OR strip) --------------------------------------


def _load_gray(path: str, size: int | None) -> torch.Tensor:
    im = Image.open(path).convert("L")
    if size is not None:
        im = im.resize((size, size))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _load_strip(path: str) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def embed_unique(
    backbone: torch.nn.Module,
    paths: list[str],
    device: torch.device,
    loader,
    batch_size: int,
    load_threads: int,
) -> dict[str, np.ndarray]:
    backbone.eval()
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            tensors = thread_map(loader, chunk, max_workers=load_threads)
            x = torch.stack(tensors).to(device)
            z = backbone(x).float().cpu().numpy()
            for p, row in zip(chunk, z):
                out[p] = row
    return out


def cnn_pair_scores(zmap: dict[str, np.ndarray], pairs: list[tuple[str, str]]) -> np.ndarray:
    a = np.stack([zmap[p[0]] for p in pairs])
    b = np.stack([zmap[p[1]] for p in pairs])
    return (a * b).sum(axis=1)


# ---- IrisCode scoring --------------------------------------------------------


def encode_strips_to_codes(
    paths: list[str], params: IrisCodeParams, load_threads: int
) -> dict[str, np.ndarray]:
    def _enc(p: str) -> np.ndarray:
        s = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        return encode_strip(s, params)

    codes = thread_map(_enc, paths, max_workers=load_threads)
    return dict(zip(paths, codes))


def iriscode_pair_similarity(
    cmap: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    strip_h: int,
    strip_w: int,
    max_shift: int,
) -> np.ndarray:
    mask = default_strip_mask(strip_h, strip_w, fraction_kept=0.7)
    sims = []
    for a, b in pairs:
        hd = hamming_distance(cmap[a], cmap[b], mask, mask, max_shift=max_shift)
        sims.append(1.0 - hd)  # higher = more similar
    return np.array(sims, dtype=np.float64)


# ---- evaluator harness -------------------------------------------------------


def evaluate_pipeline(
    name: str,
    val_pairs: tuple[list, list],
    test_pairs: tuple[list, list],
    score_fn,
) -> dict:
    """Score pairs with score_fn(pairs) -> ndarray; pick tau* on val EER, report on test."""
    val_gen, val_imp = val_pairs
    test_gen, test_imp = test_pairs

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

    # ROC: TPR=1-FRR vs FAR over a fine threshold sweep on test scores.
    all_scores = np.concatenate([test_g, test_i])
    lo, hi = float(all_scores.min()), float(all_scores.max())
    pad = max(1e-6, (hi - lo) * 0.05)
    thr = np.linspace(lo - pad, hi + pad, 400)
    far_c, frr_c = compute_far_frr_curve(test_g, test_i, thr, higher_is_genuine=True)
    tpr_c = 1.0 - frr_c

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
        "roc_tpr": tpr_c.tolist(),
        "roc_thresholds": thr.tolist(),
    }


# ---- main --------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=4000)
    ap.add_argument("--infer-batch-size", type=int, default=128)
    ap.add_argument("--load-threads", type=int, default=0)
    ap.add_argument("--max-shift", type=int, default=16)
    args = ap.parse_args()
    load_threads = args.load_threads if args.load_threads > 0 else worker_cap(12)

    torch.manual_seed(RANDOM_SEED)
    rng_seed_offsets = {"val": 0, "test": 1}

    df_full = pd.read_csv(MANIFEST_PATH)
    df_strip = pd.read_csv(STRIP_MANIFEST_PATH) if STRIP_MANIFEST_PATH.exists() else None
    if df_strip is not None:
        df_strip = df_strip[df_strip["ok"] & df_strip["strip_path"].notna()].copy()
        # join image_path -> strip_path
        df_strip["image_path"] = df_strip["path"]
        path_to_strip = dict(zip(df_strip["image_path"], df_strip["strip_path"]))
    else:
        path_to_strip = None

    device = pick_device()
    print(f"Device: {device_summary(device)}")

    # Sample whole-frame pairs (used by baseline_cnn). Same indices map to
    # strips/codes via path_to_strip below for the other two pipelines.
    val_df = df_full[df_full["split"] == "val"].reset_index(drop=True)
    test_df = df_full[df_full["split"] == "test"].reset_index(drop=True)
    val_pairs_img = sample_pairs(
        val_df, args.n_pairs, args.n_pairs, seed=RANDOM_SEED + rng_seed_offsets["val"], path_col="path"
    )
    test_pairs_img = sample_pairs(
        test_df, args.n_pairs, args.n_pairs, seed=RANDOM_SEED + rng_seed_offsets["test"], path_col="path"
    )

    results: dict[str, dict] = {}

    # --- baseline CNN (whole frame, 128x128) ---
    if MODEL_PATH.exists():
        print("=== baseline_cnn (whole-frame) ===")
        ckpt = torch.load(MODEL_PATH, map_location=device)
        base = IrisEmbeddingCNN(embed_dim=ckpt.get("embed_dim", 128)).to(device)
        base.load_state_dict(ckpt["backbone"])
        all_paths = list({p for pair in val_pairs_img[0] + val_pairs_img[1] + test_pairs_img[0] + test_pairs_img[1] for p in pair})
        zmap = embed_unique(base, all_paths, device, lambda p: _load_gray(p, IMG_SIZE), args.infer_batch_size, load_threads)
        results["baseline_cnn"] = evaluate_pipeline(
            "baseline_cnn",
            val_pairs_img,
            test_pairs_img,
            score_fn=lambda pairs: cnn_pair_scores(zmap, pairs),
        )
    else:
        print(f"[skip] baseline checkpoint missing at {MODEL_PATH}")

    # --- strip pipelines: need strip_path mapping ---
    if path_to_strip is not None:
        def _to_strip_pairs(pairs):
            mapped = [(path_to_strip.get(a), path_to_strip.get(b)) for a, b in pairs]
            return [m for m in mapped if m[0] is not None and m[1] is not None]

        val_gen_s = _to_strip_pairs(val_pairs_img[0])
        val_imp_s = _to_strip_pairs(val_pairs_img[1])
        test_gen_s = _to_strip_pairs(test_pairs_img[0])
        test_imp_s = _to_strip_pairs(test_pairs_img[1])
        print(
            f"strip pairs (after dropping seg failures): "
            f"val gen/imp={len(val_gen_s)}/{len(val_imp_s)}  "
            f"test gen/imp={len(test_gen_s)}/{len(test_imp_s)}"
        )

        # --- strip_cnn ---
        if STRIP_CNN_PATH.exists():
            print("=== strip_cnn ===")
            ckpt = torch.load(STRIP_CNN_PATH, map_location=device)
            strip_model = IrisEmbeddingCNN(embed_dim=ckpt.get("embed_dim", 128)).to(device)
            strip_model.load_state_dict(ckpt["backbone"])
            all_strips = list({p for pair in val_gen_s + val_imp_s + test_gen_s + test_imp_s for p in pair})
            zmap_s = embed_unique(strip_model, all_strips, device, _load_strip, args.infer_batch_size, load_threads)
            results["strip_cnn"] = evaluate_pipeline(
                "strip_cnn",
                (val_gen_s, val_imp_s),
                (test_gen_s, test_imp_s),
                score_fn=lambda pairs: cnn_pair_scores(zmap_s, pairs),
            )
        else:
            print(f"[skip] strip_cnn checkpoint missing at {STRIP_CNN_PATH}")

        # --- iriscode ---
        print("=== iriscode (Daugman log-Gabor + Hamming) ===")
        params = IrisCodeParams(n_scales=2, base_wavelength=18.0, mult=1.6, sigma_on_f=0.5)
        all_strips = list({p for pair in val_gen_s + val_imp_s + test_gen_s + test_imp_s for p in pair})
        cmap = encode_strips_to_codes(all_strips, params, load_threads)
        results["iriscode"] = evaluate_pipeline(
            "iriscode",
            (val_gen_s, val_imp_s),
            (test_gen_s, test_imp_s),
            score_fn=lambda pairs: iriscode_pair_similarity(
                cmap, pairs, STRIP_HEIGHT, STRIP_WIDTH, args.max_shift
            ),
        )
    else:
        print("[skip] strip pipelines (run 07 + 08 first)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ADVANCED_METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {ADVANCED_METRICS_PATH}")
    print("Summary:")
    for name, r in results.items():
        print(
            f"  {name:14s}  test_EER={r['test_eer']:.4f}  "
            f"FAR@tau={r['test_far']:.4f}  FRR@tau={r['test_frr']:.4f}  "
            f"acc={r['test_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
