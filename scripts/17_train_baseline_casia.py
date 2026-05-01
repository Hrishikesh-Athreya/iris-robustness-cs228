#!/usr/bin/env python3
"""Train baseline whole-frame IrisEmbeddingCNN per CASIA dataset (Interval or Lamp).

Triplet loss on grayscale, resized images. Subject-disjoint val/test inherited
from the CASIA manifest; identities are <class>_<L|R>.
"""

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

from iris_checkpoint.config import (
    CASIA_BASELINE_INTERVAL_PATH,
    CASIA_BASELINE_LAMP_PATH,
    CASIA_INTERVAL_MANIFEST,
    CASIA_LAMP_MANIFEST,
    IMG_SIZE,
    RANDOM_SEED,
)
from iris_checkpoint.device import device_summary, pick_device
from iris_checkpoint.model import IrisEmbeddingCNN
from iris_checkpoint.parallel_util import thread_map, worker_cap


def build_triplet_batch(by_sub, batch_subjects, rng, load_threads, img_size):
    eligible = [s for s, ps in by_sub.items() if len(ps) >= 2]
    if len(eligible) < batch_subjects + 1:
        return None
    pick = rng.choice(len(eligible), size=batch_subjects + 1, replace=False)
    subs = [eligible[int(i)] for i in pick]
    anchor_subs = subs[:-1]
    neg_candidates = [s for s in eligible if s not in set(anchor_subs)] or eligible
    anchors, positives, negatives = [], [], []
    for s in anchor_subs:
        idx = rng.choice(len(by_sub[s]), size=2, replace=False)
        anchors.append(by_sub[s][int(idx[0])])
        positives.append(by_sub[s][int(idx[1])])
        ns = neg_candidates[int(rng.integers(0, len(neg_candidates)))]
        negatives.append(by_sub[ns][int(rng.integers(0, len(by_sub[ns])))])
    from PIL import Image

    def load_path(p):
        im = Image.open(p).convert("L").resize((img_size, img_size))
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).unsqueeze(0)

    paths = anchors + positives + negatives
    tensors = thread_map(load_path, paths, max_workers=load_threads)
    n = len(anchors)
    return torch.stack(tensors[:n]), torch.stack(tensors[n : 2 * n]), torch.stack(tensors[2 * n :])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["interval", "lamp"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-subjects", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps-per-epoch", type=int, default=80)
    ap.add_argument("--load-threads", type=int, default=0)
    args = ap.parse_args()
    load_threads = args.load_threads if args.load_threads > 0 else worker_cap(12)

    if args.dataset == "interval":
        manifest_path = CASIA_INTERVAL_MANIFEST
        out_path = CASIA_BASELINE_INTERVAL_PATH
    else:
        manifest_path = CASIA_LAMP_MANIFEST
        out_path = CASIA_BASELINE_LAMP_PATH

    torch.manual_seed(RANDOM_SEED)
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    print(f"[{args.dataset}] train rows: {len(train_df)}  identities: {train_df['subject_id'].nunique()}")

    by_sub = train_df.groupby("subject_id")["path"].apply(list).to_dict()
    device = pick_device()
    print(f"Device: {device_summary(device)}")
    backbone = IrisEmbeddingCNN(embed_dim=128).to(device)
    opt = torch.optim.Adam(backbone.parameters(), lr=args.lr)
    rng = np.random.default_rng(RANDOM_SEED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(args.epochs):
        backbone.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            batch = build_triplet_batch(by_sub, args.batch_subjects, rng, load_threads, IMG_SIZE)
            if batch is None:
                continue
            A, P, N = [t.to(device) for t in batch]
            opt.zero_grad()
            za, zp, zn = backbone(A), backbone(P), backbone(N)
            loss = F.triplet_margin_loss(za, zp, zn, margin=0.35, p=2)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        avg = float(np.mean(losses)) if losses else 0.0
        print(f"epoch {epoch + 1}/{args.epochs}  triplet_loss={avg:.4f}", flush=True)
        if avg > 0 and avg < best:
            best = avg
            torch.save({"backbone": backbone.state_dict(), "embed_dim": 128, "dataset": args.dataset}, out_path)
    if best == float("inf"):
        torch.save({"backbone": backbone.state_dict(), "embed_dim": 128, "dataset": args.dataset}, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
