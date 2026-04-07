#!/usr/bin/env python3
"""Train embedding CNN with triplet loss (works with subject-disjoint val/test)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iris_checkpoint.config import IMG_SIZE, MANIFEST_PATH, MODEL_PATH, RANDOM_SEED
from iris_checkpoint.dataset import load_manifest
from iris_checkpoint.device import device_summary, pick_device
from iris_checkpoint.model import IrisEmbeddingCNN
from iris_checkpoint.parallel_util import thread_map, worker_cap


def build_triplet_batch(
    df: pd.DataFrame,
    by_sub: dict[str, list[str]],
    batch_subjects: int,
    rng: np.random.Generator,
    load_threads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    eligible = [s for s, ps in by_sub.items() if len(ps) >= 2]
    if len(eligible) < batch_subjects + 1:
        return None
    pick = rng.choice(len(eligible), size=batch_subjects + 1, replace=False)
    subs = [eligible[int(i)] for i in pick]
    anchor_subs = subs[:-1]
    neg_candidates = [s for s in eligible if s not in set(anchor_subs)]
    if not neg_candidates:
        neg_candidates = eligible  # fallback
    anchors, positives, negatives = [], [], []
    for s in anchor_subs:
        idx = rng.choice(len(by_sub[s]), size=2, replace=False)
        a, p = by_sub[s][int(idx[0])], by_sub[s][int(idx[1])]
        neg_sub = neg_candidates[int(rng.integers(0, len(neg_candidates)))]
        neg_pool = by_sub[neg_sub]
        n = neg_pool[int(rng.integers(0, len(neg_pool)))]
        anchors.append(a)
        positives.append(p)
        negatives.append(n)
    from PIL import Image

    def load_path(p: str) -> torch.Tensor:
        im = Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE))
        return torch.from_numpy(np.array(im, dtype=np.float32) / 255.0).unsqueeze(0)

    paths = anchors + positives + negatives
    tensors = thread_map(load_path, paths, max_workers=load_threads)
    n = len(anchors)
    A = torch.stack(tensors[:n])
    P = torch.stack(tensors[n : 2 * n])
    N = torch.stack(tensors[2 * n :])
    return A, P, N


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-subjects", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps-per-epoch", type=int, default=80)
    ap.add_argument(
        "--load-threads",
        type=int,
        default=0,
        help="Threads for parallel PIL decode per batch (0=auto).",
    )
    args = ap.parse_args()

    load_threads = args.load_threads if args.load_threads > 0 else worker_cap(12)

    torch.manual_seed(RANDOM_SEED)

    df = load_manifest(MANIFEST_PATH)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    if len(train_df) < 20:
        print("Not enough training images; rebuild manifest.", file=sys.stderr)
        sys.exit(1)

    by_sub = train_df.groupby("subject_id")["path"].apply(list).to_dict()
    device = pick_device()
    print(f"Device: {device_summary(device)} (set IRIS_DEVICE=cpu if MPS errors)")
    backbone = IrisEmbeddingCNN(embed_dim=128).to(device)
    opt = torch.optim.Adam(backbone.parameters(), lr=args.lr)
    rng = np.random.default_rng(RANDOM_SEED)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(args.epochs):
        backbone.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            batch = build_triplet_batch(
                train_df, by_sub, args.batch_subjects, rng, load_threads
            )
            if batch is None:
                continue
            A, P, N = [t.to(device) for t in batch]
            opt.zero_grad()
            za, zp, zn = backbone(A), backbone(P), backbone(N)
            loss = F.triplet_margin_loss(za, zp, zn, margin=0.35, p=2)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        avg = float(np.mean(losses)) if losses else 0.0
        print(f"epoch {epoch+1}/{args.epochs}  triplet_loss={avg:.4f}")
        if avg < best_loss and avg > 0:
            best_loss = avg
            torch.save(
                {"backbone": backbone.state_dict(), "embed_dim": 128},
                MODEL_PATH,
            )
    final_path = MODEL_PATH.with_name("baseline_cnn_final.pt")
    torch.save({"backbone": backbone.state_dict(), "embed_dim": 128}, final_path)
    if best_loss == float("inf"):
        # no improvement was ever saved; use final as the main checkpoint
        torch.save({"backbone": backbone.state_dict(), "embed_dim": 128}, MODEL_PATH)
    print(f"Saved best backbone to {MODEL_PATH}, final to {final_path}")


if __name__ == "__main__":
    main()
