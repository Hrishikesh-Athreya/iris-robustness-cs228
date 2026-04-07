# CS 228 — Iris Robustness

Reproducible code for iris verification under quality degradation using UBIRIS.V2.

## Environment

- Python 3.10+ recommended (tested on 3.11).

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline

```bash
# 1) Manifest — auto-detects data/archive (2)/CLASSES_400_300_Part1 (UBIRIS.V2)
python scripts/01_build_manifest.py

# 2) EDA figures (multiprocess image stats by default)
python scripts/02_eda.py

# 3) Train triplet embedding CNN -> checkpoints/baseline_cnn.pt
# Uses CUDA if present, else Apple MPS on M-series Macs, else CPU.
python scripts/03_train_baseline.py --epochs 15

# 4) Verification metrics + FAR/FRR curve
python scripts/04_eval_verify.py
```

## Repository Layout

| Path | Purpose |
|------|---------|
| `iris_checkpoint/` | Dataset helpers, metrics, model |
| `scripts/` | CLI stages 01–04 |
| `requirements.txt` | Python dependencies |

## Authors

Hrishikesh Athreya, Rohan Hareesh.
