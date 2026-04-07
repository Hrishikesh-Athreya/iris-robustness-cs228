#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/01_build_manifest.py
python3 scripts/02_eda.py
python3 scripts/03_train_baseline.py --epochs 15
python3 scripts/04_eval_verify.py
python3 scripts/05_write_latex_snippets.py
cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo "Done: report/main.pdf"
