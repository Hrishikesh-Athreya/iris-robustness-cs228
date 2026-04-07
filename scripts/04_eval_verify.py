#!/usr/bin/env python3
"""Evaluate verification FAR/FRR/EER on val (threshold) and test (report)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iris_checkpoint.config import FIG_DIR, MANIFEST_PATH, METRICS_PATH, MODEL_PATH, RANDOM_SEED
from iris_checkpoint.dataset import load_manifest
from iris_checkpoint.device import device_summary, pick_device
from iris_checkpoint.metrics import compute_far_frr_curve, eer_and_accuracy
from iris_checkpoint.model import IrisEmbeddingCNN
from iris_checkpoint.parallel_util import thread_map, worker_cap


def sample_pairs(
    df: pd.DataFrame,
    n_genuine: int,
    n_impostor: int,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    rng = np.random.default_rng(seed)
    by_sub = df.groupby("subject_id")["path"].apply(list).to_dict()
    subs = [s for s, ps in by_sub.items() if len(ps) >= 2]
    gen: list[tuple[str, str]] = []
    tries = 0
    while len(gen) < n_genuine and tries < n_genuine * 50:
        tries += 1
        s = subs[rng.integers(0, len(subs))]
        a, b = rng.choice(len(by_sub[s]), size=2, replace=False)
        gen.append((by_sub[s][int(a)], by_sub[s][int(b)]))
    all_subs = list(by_sub.keys())
    imp: list[tuple[str, str]] = []
    tries = 0
    while len(imp) < n_impostor and tries < n_impostor * 50:
        tries += 1
        s1, s2 = rng.choice(len(all_subs), size=2, replace=False)
        s1, s2 = all_subs[int(s1)], all_subs[int(s2)]
        p1 = by_sub[s1][rng.integers(0, len(by_sub[s1]))]
        p2 = by_sub[s2][rng.integers(0, len(by_sub[s2]))]
        imp.append((p1, p2))
    return gen, imp


def _load_gray_tensor(path: str, size: int) -> torch.Tensor:
    from PIL import Image

    im = Image.open(path).convert("L").resize((size, size))
    return torch.from_numpy(np.array(im, dtype=np.float32) / 255.0).unsqueeze(0)


def embed_unique_paths(
    backbone: torch.nn.Module,
    unique_paths: list[str],
    device: torch.device,
    size: int,
    infer_batch_size: int,
    load_threads: int,
) -> dict[str, np.ndarray]:
    """One forward pass per mini-batch; parallel PIL decode inside batch."""
    backbone.eval()
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(unique_paths), infer_batch_size):
            chunk = unique_paths[i : i + infer_batch_size]
            tensors = thread_map(
                lambda p: _load_gray_tensor(p, size),
                chunk,
                max_workers=load_threads,
            )
            x = torch.stack(tensors).to(device)
            z = backbone(x).float().cpu().numpy()
            for p, row in zip(chunk, z):
                out[p] = row
    return out


def pair_cosine_scores(
    zmap: dict[str, np.ndarray],
    paths_a: list[str],
    paths_b: list[str],
) -> np.ndarray:
    za = np.stack([zmap[p] for p in paths_a])
    zb = np.stack([zmap[p] for p in paths_b])
    return (za * zb).sum(axis=1)


def scores_for_pairs(
    backbone: torch.nn.Module,
    paths_a: list[str],
    paths_b: list[str],
    device: torch.device,
    size: int,
    infer_batch_size: int,
    load_threads: int,
) -> np.ndarray:
    uniq = list(dict.fromkeys(list(paths_a) + list(paths_b)))
    zmap = embed_unique_paths(
        backbone, uniq, device, size, infer_batch_size, load_threads
    )
    return pair_cosine_scores(zmap, paths_a, paths_b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-batch-size", type=int, default=128)
    ap.add_argument(
        "--load-threads",
        type=int,
        default=0,
        help="Parallel image decodes per batch (0=auto).",
    )
    args = ap.parse_args()
    load_threads = args.load_threads if args.load_threads > 0 else worker_cap(12)

    torch.manual_seed(RANDOM_SEED)
    _SPLIT_SEED_OFFSET = {"val": 0, "test": 1, "train": 2}

    df = load_manifest(MANIFEST_PATH)
    device = pick_device()
    print(f"Device: {device_summary(device)}")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    backbone = IrisEmbeddingCNN(embed_dim=ckpt.get("embed_dim", 128)).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    size = 128

    n_pairs = min(8000, max(500, len(df) * 2))

    def run_split(split_name: str) -> dict:
        sdf = df[df["split"] == split_name].reset_index(drop=True)
        gen_paths, imp_paths = sample_pairs(
            sdf, n_pairs, n_pairs, seed=RANDOM_SEED + _SPLIT_SEED_OFFSET.get(split_name, 0)
        )
        ga, gb = zip(*gen_paths) if gen_paths else ([], [])
        ia, ib = zip(*imp_paths) if imp_paths else ([], [])
        la, lb = list(ga), list(gb)
        li_a, li_b = list(ia), list(ib)
        sg = (
            scores_for_pairs(
                backbone,
                la,
                lb,
                device,
                size,
                args.infer_batch_size,
                load_threads,
            )
            if gen_paths
            else np.array([])
        )
        si = (
            scores_for_pairs(
                backbone,
                li_a,
                li_b,
                device,
                size,
                args.infer_batch_size,
                load_threads,
            )
            if imp_paths
            else np.array([])
        )
        return {"scores_genuine": sg, "scores_impostor": si}

    val_res = run_split("val")
    out_val = eer_and_accuracy(
        val_res["scores_genuine"], val_res["scores_impostor"]
    )
    t_star = out_val["threshold"]

    test_res = run_split("test")
    g, i = test_res["scores_genuine"], test_res["scores_impostor"]
    if len(g) and len(i):
        far = float((i >= t_star).mean())
        frr = float((g < t_star).mean())
        acc = float(((g >= t_star).sum() + (i < t_star).sum()) / (len(g) + len(i)))
    else:
        far = frr = acc = float("nan")

    test_eer = eer_and_accuracy(g, i)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "val_eer": out_val["eer"],
        "val_threshold": t_star,
        "test_far_at_val_threshold": far,
        "test_frr_at_val_threshold": frr,
        "test_accuracy_at_val_threshold": acc,
        "test_eer": test_eer["eer"],
        "n_genuine_pairs": int(len(g)),
        "n_impostor_pairs": int(len(i)),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))

    lo = min(float(g.min()), float(i.min()))
    hi = max(float(g.max()), float(i.max()))
    pad = max(1e-6, (hi - lo) * 0.05)
    thresholds = np.linspace(lo - pad, hi + pad, 400)
    far_c, frr_c = compute_far_frr_curve(g, i, thresholds, higher_is_genuine=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(thresholds, far_c, label="FAR")
    ax.plot(thresholds, frr_c, label="FRR")
    ax.axvline(t_star, color="gray", linestyle="--", label="Val EER threshold")
    ax.set_xlabel("Cosine similarity threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Test split: FAR vs FRR trade-off")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "far_frr_tradeoff.pdf")
    fig.savefig(FIG_DIR / "far_frr_tradeoff.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
