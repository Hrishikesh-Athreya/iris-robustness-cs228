#!/usr/bin/env python3
"""Generate Checkpoint 2 figures, error grids, and LaTeX snippet.

Reads:
  results/metrics_advanced.json           (from 10_eval_all.py)
  results/robustness_sweep.csv            (from 11_robustness_sweep.py)
  metrics.json                            (Checkpoint 1 baseline)
  manifest.csv, strip_manifest.csv

Writes:
  report/figs/architecture_diagram.pdf    (TikZ standalone, also kept as .tex)
  report/figs/roc_compare.pdf             (ROC overlay all pipelines)
  report/figs/eer_bar.pdf                 (EER bar chart)
  report/figs/robustness_<axis>.pdf       (one per degradation axis)
  report/figs/failure_grid.pdf            (worst genuine + worst impostor)
  report/figs/seg_examples.pdf            (input / GT / pred for a few images)
  report/results_inc.tex                  (LaTeX numbers for main.tex)
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

from iris_checkpoint.config import (
    ADVANCED_METRICS_PATH,
    FIG_DIR,
    IMG_SIZE,
    LATEX_SNIPPET_PATH,
    MANIFEST_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    RESULTS_DIR,
    SEG_GT_DIR,
    SEG_INPUT_SIZE,
    SEG_MODEL_PATH,
    SEG_NATIVE_SIZE,
    STRIP_CNN_PATH,
    STRIP_HEIGHT,
    STRIP_MANIFEST_PATH,
    STRIP_WIDTH,
)
from iris_checkpoint.device import pick_device
from iris_checkpoint.iriscode import (
    IrisCodeParams,
    default_strip_mask,
    encode_strip,
    hamming_distance,
)
from iris_checkpoint.model import IrisEmbeddingCNN
from iris_checkpoint.segmenter import FISNetLite


PIPELINE_LABEL = {
    "baseline_cnn": "Baseline CNN (whole frame)",
    "strip_cnn": "Strip CNN (FISNet seg + rubber-sheet)",
    "iriscode": "IrisCode (Gabor + Hamming)",
}
PIPELINE_COLOR = {
    "baseline_cnn": "#888888",
    "strip_cnn": "#1f77b4",
    "iriscode": "#d62728",
}


# ---- ROC overlay -------------------------------------------------------------


def plot_roc_compare(adv: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for name, label in PIPELINE_LABEL.items():
        if name not in adv:
            continue
        far = np.array(adv[name]["roc_far"])
        tpr = np.array(adv[name]["roc_tpr"])
        order = np.argsort(far)
        ax.plot(far[order], tpr[order], label=f"{label} (EER={adv[name]['test_eer']:.3f})", color=PIPELINE_COLOR[name])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=0.8)
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1 - FRR)")
    ax.set_title("ROC: pipelines compared on test split")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_compare.pdf")
    fig.savefig(FIG_DIR / "roc_compare.png", dpi=200)
    plt.close(fig)


# ---- EER bar chart -----------------------------------------------------------


def plot_eer_bar(adv: dict) -> None:
    names = [n for n in PIPELINE_LABEL if n in adv]
    eers = [adv[n]["test_eer"] for n in names]
    labels = [PIPELINE_LABEL[n] for n in names]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    bars = ax.bar(range(len(names)), eers, color=[PIPELINE_COLOR[n] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("Test EER (lower is better)")
    ax.set_title("Equal Error Rate by pipeline")
    for b, v in zip(bars, eers):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, max(eers) * 1.25 + 0.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eer_bar.pdf")
    fig.savefig(FIG_DIR / "eer_bar.png", dpi=200)
    plt.close(fig)


# ---- Robustness curves -------------------------------------------------------


def plot_robustness(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    for axis, gdf in df.groupby("axis"):
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        for name, sdf in gdf.groupby("pipeline"):
            sdf = sdf.sort_values("intensity")
            ax.plot(sdf["intensity"], sdf["eer"], "-o",
                    label=PIPELINE_LABEL.get(name, name),
                    color=PIPELINE_COLOR.get(name, None))
        ax.set_xlabel(f"{axis} intensity")
        ax.set_ylabel("EER")
        ax.set_title(f"Robustness: EER vs {axis}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"robustness_{axis}.pdf")
        fig.savefig(FIG_DIR / f"robustness_{axis}.png", dpi=200)
        plt.close(fig)


# ---- Segmentation examples ---------------------------------------------------


def plot_seg_examples(n: int = 4) -> None:
    if not SEG_MODEL_PATH.exists():
        return
    device = pick_device()
    ckpt = torch.load(SEG_MODEL_PATH, map_location=device)
    model = FISNetLite(base_channels=int(ckpt.get("base_channels", 32))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    size = int(ckpt.get("image_size", SEG_INPUT_SIZE))

    df = pd.read_csv(MANIFEST_PATH)
    df["imgkey"] = df["path"].str.extract(r"(C\d+_S\d+_I\d+)\.", expand=False)
    rng = np.random.default_rng(RANDOM_SEED)
    # pick n test-split images that have GT masks
    df_test = df[df["split"] == "test"].copy()
    gt_keys = set(p.stem.replace("OperatorA_", "") for p in Path(SEG_GT_DIR).iterdir() if p.suffix.lower() == ".tiff")
    df_test = df_test[df_test["imgkey"].isin(gt_keys)].reset_index(drop=True)
    if len(df_test) < n:
        return
    pick = rng.choice(len(df_test), size=n, replace=False)

    fig, axes = plt.subplots(n, 3, figsize=(7.0, 2.0 * n))
    H, W = SEG_NATIVE_SIZE
    for row, idx in enumerate(pick):
        img_p = df_test.iloc[idx]["path"]
        key = df_test.iloc[idx]["imgkey"]
        gt_p = SEG_GT_DIR / f"OperatorA_{key}.tiff"
        img = np.asarray(Image.open(img_p).convert("L"))
        gt = np.asarray(Image.open(gt_p).convert("L"))
        # predict
        pil = Image.fromarray(img).resize((size, size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logits).cpu().numpy()[0, 0] > 0.5).astype(np.uint8) * 255

        for col, (im, ttl) in enumerate(zip([img, gt, pred], ["input", "GT (IRISSEG-EP)", "FISNet-lite"])):
            axes[row, col].imshow(im, cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(ttl, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seg_examples.pdf")
    fig.savefig(FIG_DIR / "seg_examples.png", dpi=200)
    plt.close(fig)


# ---- Failure grid ------------------------------------------------------------


def plot_failure_grid(adv: dict, k: int = 4) -> None:
    """Worst genuine (lowest similarity) and worst impostor (highest similarity)
    for the strip_cnn pipeline. Shows source images and predicted strips.
    """
    if "strip_cnn" not in adv:
        return
    if not STRIP_MANIFEST_PATH.exists():
        return
    if not STRIP_CNN_PATH.exists():
        return

    # Re-score a fresh test pair sample using strip_cnn.
    device = pick_device()
    s_ckpt = torch.load(STRIP_CNN_PATH, map_location=device)
    model = IrisEmbeddingCNN(embed_dim=s_ckpt.get("embed_dim", 128)).to(device)
    model.load_state_dict(s_ckpt["backbone"])
    model.eval()

    df_strip = pd.read_csv(STRIP_MANIFEST_PATH)
    df_strip = df_strip[df_strip["ok"] & df_strip["strip_path"].notna()].copy()
    df_strip = df_strip[df_strip["split"] == "test"].reset_index(drop=True)
    if len(df_strip) < 2:
        return

    rng = np.random.default_rng(RANDOM_SEED + 1)
    by_sub = df_strip.groupby("subject_id")[["path", "strip_path"]].apply(
        lambda d: list(zip(d["path"], d["strip_path"]))
    ).to_dict()
    subs = [s for s, ps in by_sub.items() if len(ps) >= 2]
    if not subs:
        return

    gen, imp = [], []
    n = 1500
    while len(gen) < n:
        s = subs[rng.integers(0, len(subs))]
        a, b = rng.choice(len(by_sub[s]), size=2, replace=False)
        gen.append(by_sub[s][int(a)] + by_sub[s][int(b)])
    all_subs = list(by_sub.keys())
    while len(imp) < n:
        s1, s2 = rng.choice(len(all_subs), size=2, replace=False)
        a = by_sub[all_subs[int(s1)]][rng.integers(0, len(by_sub[all_subs[int(s1)]]))]
        b = by_sub[all_subs[int(s2)]][rng.integers(0, len(by_sub[all_subs[int(s2)]]))]
        imp.append(a + b)

    def _score(pairs):
        all_strips = sorted({p[1] for p in pairs} | {p[3] for p in pairs})
        zmap = {}
        bs = 64
        with torch.no_grad():
            for i in range(0, len(all_strips), bs):
                chunk = all_strips[i : i + bs]
                tens = torch.stack([
                    torch.from_numpy(np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0).unsqueeze(0)
                    for p in chunk
                ]).to(device)
                z = model(tens).float().cpu().numpy()
                for p, row in zip(chunk, z):
                    zmap[p] = row
        scores = []
        for img_a, str_a, img_b, str_b in pairs:
            scores.append(float((zmap[str_a] * zmap[str_b]).sum()))
        return np.array(scores)

    g_scores = _score(gen)
    i_scores = _score(imp)

    # worst genuine: lowest score; worst impostor: highest score
    worst_gen_idx = np.argsort(g_scores)[:k]
    worst_imp_idx = np.argsort(-i_scores)[:k]

    fig, axes = plt.subplots(2 * k, 2, figsize=(5.0, 1.6 * 2 * k))
    for row, idx in enumerate(worst_gen_idx):
        img_a, str_a, img_b, str_b = gen[idx]
        axes[row, 0].imshow(np.asarray(Image.open(img_a).convert("L")), cmap="gray")
        axes[row, 0].set_title(f"genuine score={g_scores[idx]:.2f}", fontsize=7)
        axes[row, 1].imshow(np.asarray(Image.open(img_b).convert("L")), cmap="gray")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    for row, idx in enumerate(worst_imp_idx):
        rr = k + row
        img_a, str_a, img_b, str_b = imp[idx]
        axes[rr, 0].imshow(np.asarray(Image.open(img_a).convert("L")), cmap="gray")
        axes[rr, 0].set_title(f"impostor score={i_scores[idx]:.2f}", fontsize=7)
        axes[rr, 1].imshow(np.asarray(Image.open(img_b).convert("L")), cmap="gray")
        axes[rr, 0].axis("off")
        axes[rr, 1].axis("off")
    fig.suptitle(f"Strip-CNN failure cases (top {k} worst genuine / impostor)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "failure_grid.pdf")
    fig.savefig(FIG_DIR / "failure_grid.png", dpi=200)
    plt.close(fig)


# ---- LaTeX snippet -----------------------------------------------------------


def write_latex(adv: dict) -> None:
    base = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    lines = []
    def cmd(name: str, val) -> str:
        # \renewcommand so we override any \providecommand defaults from main.tex.
        return rf"\renewcommand{{\{name}}}{{{val}}}"

    # baseline (Checkpoint 1) numbers
    lines.append(cmd("BaseValEER", f"{base.get('val_eer', float('nan')):.4f}"))
    lines.append(cmd("BaseTestEER", f"{base.get('test_eer', float('nan')):.4f}"))
    lines.append(cmd("BaseTestFAR", f"{base.get('test_far_at_val_threshold', float('nan')):.4f}"))
    lines.append(cmd("BaseTestFRR", f"{base.get('test_frr_at_val_threshold', float('nan')):.4f}"))
    lines.append(cmd("BaseTestAcc", f"{base.get('test_accuracy_at_val_threshold', float('nan')):.4f}"))

    for key, prefix in [("strip_cnn", "Strip"), ("iriscode", "Iris")]:
        r = adv.get(key, {})
        lines.append(cmd(f"{prefix}TestEER", f"{r.get('test_eer', float('nan')):.4f}"))
        lines.append(cmd(f"{prefix}TestFAR", f"{r.get('test_far', float('nan')):.4f}"))
        lines.append(cmd(f"{prefix}TestFRR", f"{r.get('test_frr', float('nan')):.4f}"))
        lines.append(cmd(f"{prefix}TestAcc", f"{r.get('test_accuracy', float('nan')):.4f}"))
        lines.append(cmd(f"{prefix}NGen", f"{r.get('n_genuine_pairs', 0)}"))
        lines.append(cmd(f"{prefix}NImp", f"{r.get('n_impostor_pairs', 0)}"))

    # legacy aliases used by current main.tex (keep so old build still works)
    lines.append(cmd("ValEER", f"{base.get('val_eer', float('nan')):.4f}"))
    lines.append(cmd("TestFAR", f"{base.get('test_far_at_val_threshold', float('nan')):.4f}"))
    lines.append(cmd("TestFRR", f"{base.get('test_frr_at_val_threshold', float('nan')):.4f}"))
    lines.append(cmd("TestAcc", f"{base.get('test_accuracy_at_val_threshold', float('nan')):.4f}"))
    lines.append(cmd("TestEER", f"{base.get('test_eer', float('nan')):.4f}"))
    lines.append(cmd("NGenPairs", f"{base.get('n_genuine_pairs', 0)}"))
    lines.append(cmd("NImpPairs", f"{base.get('n_impostor_pairs', 0)}"))

    LATEX_SNIPPET_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEX_SNIPPET_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {LATEX_SNIPPET_PATH}")


# ---- main --------------------------------------------------------------------


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    adv = json.loads(ADVANCED_METRICS_PATH.read_text()) if ADVANCED_METRICS_PATH.exists() else {}
    if not adv:
        print(f"warn: {ADVANCED_METRICS_PATH} not found - run 10 first")

    plot_roc_compare(adv)
    plot_eer_bar(adv)
    plot_robustness(RESULTS_DIR / "robustness_sweep.csv")
    plot_seg_examples()
    plot_failure_grid(adv)
    write_latex(adv)
    print("Plots and LaTeX snippet generated.")


if __name__ == "__main__":
    main()
