# CS 228 — Project Checkpoint 1 (Iris robustness)

Technical report sources live in `report/`; reproducible code lives in `iris_checkpoint/` and `scripts/`.

## Environment

- Python 3.10+ recommended (tested on 3.11).
- LaTeX with `pdflatex` for the PDF report.

```bash
cd checkpoint_deliverable
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## End-to-end pipeline

From `checkpoint_deliverable/`:

```bash
# 1) Manifest — auto-detects data/archive (2)/CLASSES_400_300_Part1 (UBIRIS.V2)
python scripts/01_build_manifest.py

# 2) EDA figures -> report/figs/ (multiprocess image stats by default)
python scripts/02_eda.py

# 3) Train triplet embedding CNN -> checkpoints/baseline_cnn.pt
# Uses CUDA if present, else Apple MPS on M-series Macs, else CPU.
# export IRIS_DEVICE=cpu   # if MPS misbehaves on an op
python scripts/03_train_baseline.py --epochs 15

# 4) Verification metrics + FAR/FRR curve (batched GPU/MPS inference + threaded loads)
python scripts/04_eval_verify.py

# 5) Inject numbers into LaTeX
python scripts/05_write_latex_snippets.py
```

**Apple M4 / M-series:** PyTorch accelerates this project via **MPS (Metal GPU)**, not the Apple Neural Engine (that path is Core ML). Training and evaluation call `pick_device()` accordingly.

**Parallelism:** EDA uses `ProcessPoolExecutor` for per-image statistics; training and eval use `ThreadPoolExecutor` for parallel PIL decoding. Tune with `--workers` (EDA), `--load-threads` (train/eval), and `--infer-batch-size` (eval).

## Build the PDF

```bash
cd report
./build_report.sh
```

Or manually (run **twice** so references resolve):

```bash
cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

`main.tex` sets `\renewcommand{\ttdefault}{cmtt}` so **TeX Live basic** (no Courier metrics) still builds; output is `report/main.pdf`.

## Authors

Hrishikesh Athreya, Rohan Hareesh.

## Repository layout

| Path | Purpose |
|------|---------|
| `iris_checkpoint/` | Dataset helpers, metrics, model |
| `scripts/` | CLI stages 01–05 |
| `report/main.tex` | IEEE-style two-column report |
| `report/figs/` | Figures from `02_eda.py` and `04_eval_verify.py` |
| `manifest.csv` | Generated image index (gitignored by default) |
| `metrics.json` | Evaluation output |
| `report/results_inc.tex` | Auto-generated `\\newcommand` metrics for LaTeX |
