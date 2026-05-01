#!/usr/bin/env python3
"""Generate CASIA-only figures and LaTeX snippet for the Checkpoint 2 report.

Reads:
  results/metrics_casia_interval.json
  results/metrics_casia_lamp.json
  results/robustness_casia_lamp.csv
  manifest_casia_interval.csv, manifest_casia_lamp.csv

Writes:
  report/figs/roc_casia_interval.pdf,  roc_casia_lamp.pdf
  report/figs/eer_compare_datasets.pdf
  report/figs/robustness_lamp_<axis>.pdf
  report/figs/seg_examples_casia.pdf
  report/figs/strips_examples_casia.pdf
  report/results_inc.tex   (all metric macros)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iris_checkpoint.casia_data import (
    load_casia_interval_gt,
    rasterize_iris_mask,
)
from iris_checkpoint.config import (
    CASIA_FISNET_PATH,
    CASIA_INTERVAL_MANIFEST,
    CASIA_INTERVAL_NATIVE,
    CASIA_INTERVAL_STRIP_MANIFEST,
    CASIA_LAMP_MANIFEST,
    CASIA_LAMP_NATIVE,
    CASIA_METRICS_INTERVAL_PATH,
    CASIA_METRICS_LAMP_PATH,
    CASIA_ROBUSTNESS_LAMP_PATH,
    FIG_DIR,
    LATEX_SNIPPET_PATH,
    RANDOM_SEED,
    RESULTS_DIR,
    SEG_INPUT_SIZE,
)
from iris_checkpoint.device import pick_device
from iris_checkpoint.segmenter import FISNetLite

PIPELINE_LABEL = {
    "baseline_cnn": "Whole-frame CNN",
    "strip_cnn": "FISNet + Strip CNN",
    "iriscode": "FISNet + IrisCode",
    "strip_cnn_gt": "GT + Strip CNN",
    "iriscode_gt": "GT + IrisCode",
}
PIPELINE_COLOR = {
    "baseline_cnn": "#888888",
    "strip_cnn": "#1f77b4",
    "iriscode": "#d62728",
    "strip_cnn_gt": "#1f77b4",
    "iriscode_gt": "#d62728",
}
PIPELINE_LS = {
    "baseline_cnn": "-",
    "strip_cnn": "-",
    "iriscode": "-",
    "strip_cnn_gt": "--",
    "iriscode_gt": "--",
}


def plot_roc(adv: dict, title: str, out_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for name in ["baseline_cnn", "strip_cnn", "iriscode", "strip_cnn_gt", "iriscode_gt"]:
        if name not in adv:
            continue
        far = np.array(adv[name]["roc_far"])
        tpr = np.array(adv[name]["roc_tpr"])
        order = np.argsort(far)
        ax.plot(
            far[order], tpr[order],
            label=f"{PIPELINE_LABEL[name]} (EER={adv[name]['test_eer']:.3f})",
            color=PIPELINE_COLOR[name], linestyle=PIPELINE_LS[name], lw=1.4,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=0.7)
    ax.set_xlabel("False Acceptance Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{out_stem}.pdf")
    fig.savefig(FIG_DIR / f"{out_stem}.png", dpi=200)
    plt.close(fig)


def plot_eer_compare(adv_int: dict, adv_lamp: dict) -> None:
    pipelines = ["baseline_cnn", "strip_cnn", "iriscode"]
    labels = [PIPELINE_LABEL[p] for p in pipelines]
    int_eer = [adv_int.get(p, {}).get("test_eer", np.nan) for p in pipelines]
    lamp_eer = [adv_lamp.get(p, {}).get("test_eer", np.nan) for p in pipelines]

    x = np.arange(len(pipelines))
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    b1 = ax.bar(x - width / 2, int_eer, width, label="Interval (clean NIR)",
                color="#4c72b0")
    b2 = ax.bar(x + width / 2, lamp_eer, width, label="Lamp (illum-variable NIR)",
                color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test EER (lower is better)")
    ax.set_title("EER per pipeline across CASIA datasets")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    for bs, vs in [(b1, int_eer), (b2, lamp_eer)]:
        for b, v in zip(bs, vs):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                        ha="center", fontsize=7)
    ax.set_ylim(0, max([v for v in int_eer + lamp_eer if np.isfinite(v)] + [0.1]) * 1.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eer_compare_datasets.pdf")
    fig.savefig(FIG_DIR / "eer_compare_datasets.png", dpi=200)
    plt.close(fig)


def plot_robustness_lamp(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    for axis, gdf in df.groupby("axis"):
        fig, ax = plt.subplots(figsize=(4.5, 2.8))
        for name, sdf in gdf.groupby("pipeline"):
            sdf = sdf.sort_values("intensity")
            ax.plot(sdf["intensity"], sdf["eer"], "-o",
                    label=PIPELINE_LABEL.get(name, name),
                    color=PIPELINE_COLOR.get(name, None), lw=1.3, markersize=4)
        ax.set_xlabel(f"{axis} intensity")
        ax.set_ylabel("EER")
        ax.set_title(f"CASIA-Lamp robustness: EER vs {axis}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"robustness_lamp_{axis}.pdf")
        fig.savefig(FIG_DIR / f"robustness_lamp_{axis}.png", dpi=200)
        plt.close(fig)


def plot_seg_examples(n: int = 4) -> None:
    if not CASIA_FISNET_PATH.exists():
        return
    device = pick_device()
    ckpt = torch.load(CASIA_FISNET_PATH, map_location=device)
    model = FISNetLite(base_channels=int(ckpt.get("base_channels", 32))).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    size = int(ckpt.get("image_size", SEG_INPUT_SIZE))

    df = pd.read_csv(CASIA_INTERVAL_MANIFEST)
    df_test = df[(df["split"] == "test") & df["has_gt"]].reset_index(drop=True)
    if len(df_test) < n:
        return
    rng = np.random.default_rng(RANDOM_SEED)
    pick = rng.choice(len(df_test), size=n, replace=False)

    fig, axes = plt.subplots(n, 3, figsize=(7.0, 2.0 * n))
    H, W = CASIA_INTERVAL_NATIVE
    for row, idx in enumerate(pick):
        rr = df_test.iloc[idx]
        img = np.asarray(Image.open(rr["path"]).convert("L"))
        c = load_casia_interval_gt(rr["img_key"])
        gt = rasterize_iris_mask(c, H, W, use_eyelids=False) if c is not None else np.zeros_like(img)

        pil = Image.fromarray(img).resize((size, size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logits).cpu().numpy()[0, 0] > 0.5).astype(np.uint8) * 255

        for col, (im, ttl) in enumerate(zip([img, gt, pred],
                                            ["input", "GT (IRISSEG-CC)", "FISNet-lite"])):
            axes[row, col].imshow(im, cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(ttl, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seg_examples_casia.pdf")
    fig.savefig(FIG_DIR / "seg_examples_casia.png", dpi=200)
    plt.close(fig)


def plot_strip_examples(n: int = 3) -> None:
    if not CASIA_INTERVAL_STRIP_MANIFEST.exists():
        return
    df = pd.read_csv(CASIA_INTERVAL_STRIP_MANIFEST)
    df = df[df["fisnet_ok"] & df["strip_path"].notna() & (df["split"] == "test")].reset_index(drop=True)
    if len(df) < n:
        return
    rng = np.random.default_rng(RANDOM_SEED + 2)
    pick = rng.choice(len(df), size=n, replace=False)
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 1.6 * n), gridspec_kw={"width_ratios": [1.0, 2.5]})
    for row, idx in enumerate(pick):
        r = df.iloc[idx]
        img = np.asarray(Image.open(r["path"]).convert("L"))
        strip = np.asarray(Image.open(r["strip_path"]).convert("L"))
        axes[row, 0].imshow(img, cmap="gray"); axes[row, 0].axis("off")
        axes[row, 1].imshow(strip, cmap="gray", aspect="auto"); axes[row, 1].axis("off")
        if row == 0:
            axes[row, 0].set_title("CASIA-Interval frame", fontsize=9)
            axes[row, 1].set_title("rubber-sheet strip 64x512", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strips_examples_casia.pdf")
    fig.savefig(FIG_DIR / "strips_examples_casia.png", dpi=200)
    plt.close(fig)


# --- LaTeX snippet ----------------------------------------------------------

PIPE_PREFIX = {
    "baseline_cnn": "Base",
    "strip_cnn": "Strip",
    "iriscode": "Iris",
    "strip_cnn_gt": "StripGT",
    "iriscode_gt": "IrisGT",
}


def _emit_pipeline(lines, ds_prefix: str, adv: dict, pipe: str):
    r = adv.get(pipe, {})
    pf = PIPE_PREFIX[pipe]
    def cmd(name, val):
        return rf"\renewcommand{{\{name}}}{{{val}}}"
    eer = r.get("test_eer", float("nan"))
    far = r.get("test_far", float("nan"))
    frr = r.get("test_frr", float("nan"))
    acc = r.get("test_accuracy", float("nan"))
    ng = r.get("n_genuine_pairs", 0)
    ni = r.get("n_impostor_pairs", 0)
    lines.append(cmd(f"{ds_prefix}{pf}TestEER", f"{eer:.4f}"))
    lines.append(cmd(f"{ds_prefix}{pf}TestFAR", f"{far:.4f}"))
    lines.append(cmd(f"{ds_prefix}{pf}TestFRR", f"{frr:.4f}"))
    lines.append(cmd(f"{ds_prefix}{pf}TestAcc", f"{acc:.4f}"))
    lines.append(cmd(f"{ds_prefix}{pf}NGen", f"{ng}"))
    lines.append(cmd(f"{ds_prefix}{pf}NImp", f"{ni}"))


def write_latex(adv_int, adv_lamp) -> None:
    lines: list[str] = []
    def cmd(name, val):
        return rf"\renewcommand{{\{name}}}{{{val}}}"

    # dataset stats
    df_int = pd.read_csv(CASIA_INTERVAL_MANIFEST)
    df_lamp = pd.read_csv(CASIA_LAMP_MANIFEST)
    lines.append(cmd("NIntImgs", f"{len(df_int):,}"))
    lines.append(cmd("NIntIds", f"{df_int['subject_id'].nunique():,}"))
    lines.append(cmd("NIntClasses", f"{df_int['class_id'].nunique():,}"))
    lines.append(cmd("NLampImgs", f"{len(df_lamp):,}"))
    lines.append(cmd("NLampIds", f"{df_lamp['subject_id'].nunique():,}"))
    lines.append(cmd("NLampClasses", f"{df_lamp['class_id'].nunique():,}"))
    lines.append(cmd("IntGTCoverage", f"{100*df_int['has_gt'].mean():.2f}\\%"))

    # segmentation summary (last epoch from training log)
    seg_log = RESULTS_DIR / "seg_training_log_casia.csv"
    if seg_log.exists():
        sl = pd.read_csv(seg_log)
        if not sl.empty:
            best = sl.loc[sl["val_dice"].idxmax()]
            lines.append(cmd("CasiaSegDice", f"{best['val_dice']:.4f}"))
            lines.append(cmd("CasiaSegIoU", f"{best['val_iou']:.4f}"))
            lines.append(cmd("CasiaSegEpochs", f"{int(sl['epoch'].max())}"))

    # per-dataset, per-pipeline metrics
    for pipe in ["baseline_cnn", "strip_cnn", "iriscode", "strip_cnn_gt", "iriscode_gt"]:
        _emit_pipeline(lines, "Int", adv_int, pipe)
    for pipe in ["baseline_cnn", "strip_cnn", "iriscode"]:
        _emit_pipeline(lines, "Lamp", adv_lamp, pipe)

    LATEX_SNIPPET_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEX_SNIPPET_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {LATEX_SNIPPET_PATH}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    adv_int = json.loads(CASIA_METRICS_INTERVAL_PATH.read_text()) if CASIA_METRICS_INTERVAL_PATH.exists() else {}
    adv_lamp = json.loads(CASIA_METRICS_LAMP_PATH.read_text()) if CASIA_METRICS_LAMP_PATH.exists() else {}

    if adv_int:
        plot_roc(adv_int, "ROC: CASIA-Iris-Interval (clean NIR)", "roc_casia_interval")
    if adv_lamp:
        plot_roc(adv_lamp, "ROC: CASIA-Iris-Lamp (illum. variable)", "roc_casia_lamp")
    if adv_int or adv_lamp:
        plot_eer_compare(adv_int, adv_lamp)

    plot_robustness_lamp(CASIA_ROBUSTNESS_LAMP_PATH)
    plot_seg_examples()
    plot_strip_examples()
    write_latex(adv_int, adv_lamp)
    print("CASIA plots + LaTeX snippet generated.")


if __name__ == "__main__":
    main()
