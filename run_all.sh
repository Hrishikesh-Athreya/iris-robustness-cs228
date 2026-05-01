#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# --- Checkpoint 1 baseline (already produces metrics.json + far_frr_tradeoff.pdf) ---
python3 scripts/01_build_manifest.py
python3 scripts/02_eda.py
python3 scripts/03_train_baseline.py --epochs 15
python3 scripts/04_eval_verify.py
python3 scripts/05_write_latex_snippets.py

# --- Checkpoint 2: segmentation, rubber-sheet, IrisCode, comparison ---
python3 scripts/06_train_segmenter.py --epochs 40
python3 scripts/07_infer_masks.py
python3 scripts/08_build_strip_dataset.py
python3 scripts/09_train_cnn_on_strips.py --epochs 20
python3 scripts/10_eval_all.py
python3 scripts/11_robustness_sweep.py
python3 scripts/12_make_plots_and_report.py

cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo "Done: report/main.pdf"
