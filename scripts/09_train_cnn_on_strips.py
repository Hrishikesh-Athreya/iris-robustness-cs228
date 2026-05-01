#!/usr/bin/env python3
"""Retrain IrisEmbeddingCNN on rubber-sheet strips with the same triplet loss.

This is the strip-CNN arm of the comparison: same backbone and loss as the
Checkpoint 1 baseline, but the input is a normalized iris strip rather than
a whole-frame image. Single-variable ablation vs the baseline.
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
    RANDOM_SEED,
    RESULTS_DIR,
    STRIP_CNN_PATH,
    STRIP_HEIGHT,
    STRIP_MANIFEST_PATH,
    STRIP_WIDTH,
)
from iris_checkpoint.device import device_summary, pick_device
from iris_checkpoint.model import IrisEmbeddingCNN
from iris_checkpoint.parallel_util import thread_map, worker_cap


def _load_strip(p: str, theta_shift: int = 0) -> torch.Tensor:
    im = Image.open(p).convert("L")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    if theta_shift:
        arr = np.roll(arr, theta_shift, axis=1)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


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
    neg_candidates = [s for s in eligible if s not in set(anchor_subs)] or eligible

    anchors, positives, negatives = [], [], []
    pos_shifts, neg_shifts = [], []
    for s in anchor_subs:
        idx = rng.choice(len(by_sub[s]), size=2, replace=False)
        a, p = by_sub[s][int(idx[0])], by_sub[s][int(idx[1])]
        neg_sub = neg_candidates[int(rng.integers(0, len(neg_candidates)))]
        n = by_sub[neg_sub][int(rng.integers(0, len(by_sub[neg_sub])))]
        anchors.append(a)
        positives.append(p)
        negatives.append(n)
        # Cyclic theta-shift augmentation: simulates eye rotation.
        pos_shifts.append(int(rng.integers(-32, 33)))
        neg_shifts.append(int(rng.integers(-32, 33)))

    anchor_tensors = thread_map(lambda x: _load_strip(x, 0), anchors, max_workers=load_threads)
    pos_tensors = thread_map(
        lambda ps: _load_strip(ps[0], ps[1]),
        list(zip(positives, pos_shifts)),
        max_workers=load_threads,
    )
    neg_tensors = thread_map(
        lambda ns: _load_strip(ns[0], ns[1]),
        list(zip(negatives, neg_shifts)),
        max_workers=load_threads,
    )
    A = torch.stack(anchor_tensors)
    P = torch.stack(pos_tensors)
    N = torch.stack(neg_tensors)
    return A, P, N


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-subjects", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps-per-epoch", type=int, default=80)
    ap.add_argument("--load-threads", type=int, default=0)
    args = ap.parse_args()
    load_threads = args.load_threads if args.load_threads > 0 else worker_cap(12)

    torch.manual_seed(RANDOM_SEED)
    if not STRIP_MANIFEST_PATH.exists():
        print(f"Strip manifest missing: {STRIP_MANIFEST_PATH}. Run 08 first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(STRIP_MANIFEST_PATH)
    df = df[df["ok"] & df["strip_path"].notna()].copy()
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    print(f"strip-CNN train rows: {len(train_df)}  ({STRIP_HEIGHT}x{STRIP_WIDTH})")
    if len(train_df) < 100:
        print("Not enough train strips; check segmenter outputs.", file=sys.stderr)
        sys.exit(1)

    by_sub = train_df.groupby("subject_id")["strip_path"].apply(list).to_dict()
    device = pick_device()
    print(f"Device: {device_summary(device)}")
    backbone = IrisEmbeddingCNN(embed_dim=128).to(device)
    opt = torch.optim.Adam(backbone.parameters(), lr=args.lr)
    rng = np.random.default_rng(RANDOM_SEED)

    STRIP_CNN_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "strip_cnn_training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "triplet_loss"])

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
            losses.append(float(loss.item()))
        avg = float(np.mean(losses)) if losses else 0.0
        print(f"epoch {epoch + 1}/{args.epochs}  triplet_loss={avg:.4f}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, avg])

        if avg > 0 and avg < best_loss:
            best_loss = avg
            torch.save({"backbone": backbone.state_dict(), "embed_dim": 128}, STRIP_CNN_PATH)

    if best_loss == float("inf"):
        torch.save({"backbone": backbone.state_dict(), "embed_dim": 128}, STRIP_CNN_PATH)
    print(f"Saved best strip-CNN to {STRIP_CNN_PATH}")


if __name__ == "__main__":
    main()
