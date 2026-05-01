# CS 228 — Iris Robustness (Checkpoints 1 & 2)

Reproducible code for iris verification under quality degradation on **UBIRIS.V2**, comparing:
- **Baseline CNN** (Checkpoint 1): whole-frame triplet-trained embedding.
- **FISNet-lite + strip CNN** (Checkpoint 2): fusion U-Net iris segmenter trained on IRISSEG-EP, Daugman rubber-sheet polar unwrap, and the same triplet CNN trained on the normalized strips.
- **IrisCode** (Checkpoint 2): classical Daugman log-Gabor binary code with Hamming-distance matching on the same strips.

All three pipelines share the same subject-disjoint splits and pair sampling, so head-to-head FAR / FRR / EER and ROC are directly comparable.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

```
data/archive (2)/
  CLASSES_400_300_Part1/   # UBIRIS.V2 Part 1 -- 5,799 images, 260 classes
  ubiris_seg/ubiris/       # IRISSEG-EP UBIRIS GT -- 2,250 binary masks (OperatorA)
```

## Running the pipeline

```bash
# Checkpoint 1 ----------------------------------------------------------------
python scripts/01_build_manifest.py
python scripts/02_eda.py
python scripts/03_train_baseline.py --epochs 15
python scripts/04_eval_verify.py
python scripts/05_write_latex_snippets.py

# Checkpoint 2 ----------------------------------------------------------------
python scripts/06_train_segmenter.py --epochs 40        # FISNet-lite -> checkpoints/fisnet_lite.pt
python scripts/07_infer_masks.py                        # full-corpus masks -> data/seg_pred/
python scripts/08_build_strip_dataset.py                # rubber-sheet strips -> data/strips/
python scripts/09_train_cnn_on_strips.py --epochs 20    # strip CNN -> checkpoints/strip_cnn.pt
python scripts/10_eval_all.py                           # 3-pipeline FAR/FRR/EER -> results/metrics_advanced.json
python scripts/11_robustness_sweep.py                   # blur/noise/illum/specular/off-angle sweep -> results/robustness_sweep.csv
python scripts/12_make_plots_and_report.py              # all plots + results_inc.tex

# Report build ----------------------------------------------------------------
cd report && ./build_report.sh
```

Or run end-to-end with `./run_all.sh`.

## Repository layout

| Path | Purpose |
|------|---------|
| `iris_checkpoint/config.py` | All paths, sizes, seeds. |
| `iris_checkpoint/dataset.py` | UBIRIS / CASIA / synthetic discovery + manifest builder. |
| `iris_checkpoint/torch_data.py` | Whole-frame `Dataset` for baseline CNN. |
| `iris_checkpoint/model.py` | `IrisEmbeddingCNN` (used by both baseline and strip CNN). |
| `iris_checkpoint/metrics.py` | FAR/FRR/EER, FAR-FRR curve. |
| `iris_checkpoint/segmenter.py` | **FISNet-lite** fusion U-Net + Dice / IoU losses. |
| `iris_checkpoint/seg_data.py` | IRISSEG-EP dataset and split-aware seg manifest builder. |
| `iris_checkpoint/rubber_sheet.py` | Circle fit + Daugman polar unwrap. |
| `iris_checkpoint/iriscode.py` | Log-Gabor encoder + Hamming matcher. |
| `iris_checkpoint/degradations.py` | Blur / noise / illumination / specular / off-angle generators. |
| `iris_checkpoint/device.py` | CUDA → MPS → CPU device picker. |
| `iris_checkpoint/parallel_util.py` | Thread / process map helpers. |
| `scripts/01_..05_*.py` | Checkpoint 1 stages. |
| `scripts/06_..12_*.py` | Checkpoint 2 stages. |
| `report/` | IEEEtran LaTeX source, figs, build script. |
| `results/` | Saved checkpoints, training logs, metrics JSON, robustness CSV. |
| `manifest.csv` | Subject-disjoint 70/15/15 split for verification. |
| `seg_manifest.csv` | Image $\rightarrow$ IRISSEG-EP mask join (auto-built). |
| `strip_manifest.csv` | Image $\rightarrow$ rubber-sheet strip path (built by 08). |

## Notes

- **Device**: `IRIS_DEVICE=cpu` (or `mps`/`cuda`) overrides the auto-pick.
- **TeX Live basic**: report uses `\renewcommand{\ttdefault}{cmtt}` and `\renewcommand{\sfdefault}{cmss}` to avoid missing Helvetica/Courier `tfm` files.
- Large data (`data/archive (2)/`, `data/seg_pred/`, `data/strips/`) is gitignored.

## Authors

Hrishikesh Athreya, Rohan Hareesh.
