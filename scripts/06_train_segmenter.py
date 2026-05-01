#!/usr/bin/env python3
"""Train FISNet-lite for binary iris segmentation on IRISSEG-EP UBIRIS GT.

Inherits subject-disjoint splits from manifest.csv so segmentation training
never sees verification val/test subjects.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iris_checkpoint.config import (
    RANDOM_SEED,
    RESULTS_DIR,
    SEG_INPUT_SIZE,
    SEG_MANIFEST_PATH,
    SEG_MODEL_PATH,
)
from iris_checkpoint.device import device_summary, pick_device
from iris_checkpoint.seg_data import IRISSEGDataset, save_seg_manifest
from iris_checkpoint.segmenter import (
    FISNetLite,
    bce_dice_loss,
    count_params,
    dice_score,
    iou_score,
)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    losses, dices, ious = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(float(bce_dice_loss(logits, y).item()))
            dices.append(dice_score(logits, y))
            ious.append(iou_score(logits, y))
    n = max(1, len(losses))
    return sum(losses) / n, sum(dices) / n, sum(ious) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=SEG_INPUT_SIZE)
    ap.add_argument("--base-channels", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(RANDOM_SEED)

    if not SEG_MANIFEST_PATH.exists():
        print(f"Building seg manifest at {SEG_MANIFEST_PATH}...")
        df = save_seg_manifest()
    else:
        import pandas as pd

        df = pd.read_csv(SEG_MANIFEST_PATH)

    n_train = int(df[(df["split"] == "train") & df["has_mask"]].shape[0])
    n_val = int(df[(df["split"] == "val") & df["has_mask"]].shape[0])
    n_test = int(df[(df["split"] == "test") & df["has_mask"]].shape[0])
    print(f"seg pairs: train={n_train}  val={n_val}  test={n_test}")

    train_ds = IRISSEGDataset(df, "train", image_size=args.image_size, augment=True)
    val_ds = IRISSEGDataset(df, "val", image_size=args.image_size, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    device = pick_device()
    print(f"Device: {device_summary(device)}")
    model = FISNetLite(base_channels=args.base_channels).to(device)
    print(f"FISNet-lite params: {count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    SEG_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "seg_training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr"]
        )

    best_dice = -1.0
    for epoch in range(args.epochs):
        model.train()
        ep_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = bce_dice_loss(logits, y)
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.item()))
        sched.step()

        train_loss = sum(ep_losses) / max(1, len(ep_losses))
        val_loss, val_dice, val_iou = evaluate(model, val_loader, device)
        lr_now = opt.param_groups[0]["lr"]
        print(
            f"epoch {epoch + 1:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_dice={val_dice:.4f}  val_iou={val_iou:.4f}  lr={lr_now:.2e}"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch + 1, train_loss, val_loss, val_dice, val_iou, lr_now]
            )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "image_size": args.image_size,
                    "base_channels": args.base_channels,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "epoch": epoch + 1,
                },
                SEG_MODEL_PATH,
            )

    print(f"Best val Dice: {best_dice:.4f}")
    print(f"Saved best segmenter to {SEG_MODEL_PATH}")


if __name__ == "__main__":
    main()
