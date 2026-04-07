#!/usr/bin/env python3
"""Exploratory analysis figures for the checkpoint report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iris_checkpoint.config import FIG_DIR, MANIFEST_PATH
from iris_checkpoint.dataset import image_stats_worker, load_manifest
from iris_checkpoint.parallel_util import parallel_map_process, worker_cap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Process workers for image stats (0=auto, 1=serial).",
    )
    ap.add_argument(
        "--stats-sample",
        type=int,
        default=2000,
        help="Max images for per-pixel statistic computation.",
    )
    args = ap.parse_args()

    workers = args.workers if args.workers > 0 else worker_cap(8)
    if args.workers == 1:
        workers = 1

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_manifest(MANIFEST_PATH)
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)

    # Per-subject counts
    counts = df.groupby("subject_id").size().rename("n_images").reset_index()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(counts["n_images"], bins=20, color="steelblue", edgecolor="white")
    ax.set_xlabel("Images per subject")
    ax.set_ylabel("Count of subjects")
    ax.set_title("Univariate: distribution of samples per subject")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_hist_samples_per_subject.pdf")
    fig.savefig(FIG_DIR / "eda_hist_samples_per_subject.png", dpi=200)
    plt.close(fig)

    sample_n = min(args.stats_sample, len(df))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=sample_n, replace=False)
    paths_rows = [
        (str(df.iloc[i]["path"]), str(df.iloc[i]["subject_id"]), str(df.iloc[i]["split"]))
        for i in idx
    ]
    paths_only = [p for p, _, _ in paths_rows]

    if workers == 1:
        stats_raw = [image_stats_worker(p) for p in paths_only]
    else:
        stats_raw = parallel_map_process(
            image_stats_worker, paths_only, max_workers=workers
        )

    stats = []
    for st, (_, sid, sp) in zip(stats_raw, paths_rows):
        row = dict(st)
        row["subject_id"] = sid
        row["split"] = sp
        stats.append(row)
    sdf = pd.DataFrame(stats)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.6))
    for ax, col, title in zip(
        axes,
        ["mean_intensity", "std_intensity", "blur_proxy_inv"],
        ["Mean intensity", "Std (contrast)", "Laplacian var (sharpness proxy)"],
    ):
        ax.boxplot(sdf[col], vert=True)
        ax.set_title(title)
        ax.set_xticks([])
    fig.suptitle("Univariate: ROI statistics (subsample)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_boxplot_stats.pdf")
    fig.savefig(FIG_DIR / "eda_boxplot_stats.png", dpi=200)
    plt.close(fig)

    corr = sdf[["mean_intensity", "std_intensity", "blur_proxy_inv"]].corr()
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
    )
    ax.set_title("Bivariate: correlation of image statistics")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_corr_stats.pdf")
    fig.savefig(FIG_DIR / "eda_corr_stats.png", dpi=200)
    plt.close(fig)

    top = counts.sort_values("n_images", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(top["subject_id"].astype(str), top["n_images"], color="darkseagreen")
    ax.invert_yaxis()
    ax.set_xlabel("Number of images")
    ax.set_title("Class balance: top 25 subjects by image count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_balance_top_subjects.pdf")
    fig.savefig(FIG_DIR / "eda_balance_top_subjects.png", dpi=200)
    plt.close(fig)

    split_counts = df["split"].value_counts()
    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.bar(split_counts.index, split_counts.values, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Images")
    ax.set_title("Train / val / test (subject-disjoint)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_split_counts.pdf")
    fig.savefig(FIG_DIR / "eda_split_counts.png", dpi=200)
    plt.close(fig)

    print(f"EDA figures written to {FIG_DIR} (stats workers={workers}, n={sample_n})")


if __name__ == "__main__":
    main()
