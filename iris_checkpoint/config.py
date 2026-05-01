from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
# Default UBIRIS.V2 layout when unzipped under data/archive (2)/
DEFAULT_UBIRIS_CLASSES_DIR = DATA_DIR / "archive (2)" / "CLASSES_400_300_Part1"
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figs"
MANIFEST_PATH = ROOT / "manifest.csv"
METRICS_PATH = ROOT / "metrics.json"
LATEX_SNIPPET_PATH = REPORT_DIR / "results_inc.tex"
MODEL_PATH = ROOT / "checkpoints" / "baseline_cnn.pt"

IMG_SIZE = 128
RANDOM_SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
# test = remainder

# --- Checkpoint 2: segmentation, rubber-sheet, IrisCode ----------------------
RESULTS_DIR = ROOT / "results"

# Legacy (UBIRIS, kept for archive; main pipeline now NIR/CASIA) ----
SEG_GT_DIR = DATA_DIR / "archive (2)" / "ubiris_seg" / "ubiris"
SEG_INPUT_SIZE = 256
SEG_NATIVE_SIZE = (300, 400)
SEG_MODEL_PATH = ROOT / "checkpoints" / "fisnet_lite.pt"
SEG_PRED_DIR = ROOT / "data" / "seg_pred"
SEG_MANIFEST_PATH = ROOT / "seg_manifest.csv"
STRIP_HEIGHT = 64
STRIP_WIDTH = 512
STRIP_DIR = ROOT / "data" / "strips"
STRIP_MANIFEST_PATH = ROOT / "strip_manifest.csv"
STRIP_CNN_PATH = ROOT / "checkpoints" / "strip_cnn.pt"
ADVANCED_METRICS_PATH = RESULTS_DIR / "metrics_advanced.json"

# --- CASIA NIR datasets (Checkpoint 2 main results) -------------------------
CASIA_INTERVAL_DIR = DATA_DIR / "CASIA-Iris-Interval"
CASIA_LAMP_DIR = DATA_DIR / "CASIA-Iris-Lamp"

# IRISSEG-CC GT (Halmstad) for CASIA-IrisV3-Interval: 2,655 paired .mat circles.
CASIA_INTERVAL_GT_ROOT = DATA_DIR / "CASIA-IrisV3-Interval_groundtruth"
CASIA_INTERVAL_GT_SEG = CASIA_INTERVAL_GT_ROOT / "CASIA-IrisV3-Interval_manual_segmentation"
CASIA_INTERVAL_GT_OCC = CASIA_INTERVAL_GT_ROOT / "CASIA-IrisV3-Interval_manual_occlusion_eyelids"

# Native sensor sizes (rows, cols).
CASIA_INTERVAL_NATIVE = (280, 320)
CASIA_LAMP_NATIVE = (480, 640)

# Per-dataset manifests, predicted masks, strip outputs, checkpoints, metrics.
CASIA_INTERVAL_MANIFEST = ROOT / "manifest_casia_interval.csv"
CASIA_LAMP_MANIFEST = ROOT / "manifest_casia_lamp.csv"
CASIA_INTERVAL_SEG_PRED_DIR = DATA_DIR / "casia_interval_seg_pred"
CASIA_LAMP_SEG_PRED_DIR = DATA_DIR / "casia_lamp_seg_pred"
CASIA_INTERVAL_STRIP_DIR = DATA_DIR / "casia_interval_strips"             # FISNet-strips
CASIA_INTERVAL_STRIP_GT_DIR = DATA_DIR / "casia_interval_strips_gt"       # GT-strips (oracle)
CASIA_LAMP_STRIP_DIR = DATA_DIR / "casia_lamp_strips"
CASIA_INTERVAL_STRIP_MANIFEST = ROOT / "strip_manifest_casia_interval.csv"
CASIA_LAMP_STRIP_MANIFEST = ROOT / "strip_manifest_casia_lamp.csv"

CASIA_FISNET_PATH = ROOT / "checkpoints" / "fisnet_lite_casia.pt"
CASIA_BASELINE_INTERVAL_PATH = ROOT / "checkpoints" / "baseline_cnn_casia_interval.pt"
CASIA_BASELINE_LAMP_PATH = ROOT / "checkpoints" / "baseline_cnn_casia_lamp.pt"
CASIA_STRIPCNN_INTERVAL_PATH = ROOT / "checkpoints" / "strip_cnn_casia_interval.pt"
CASIA_STRIPCNN_LAMP_PATH = ROOT / "checkpoints" / "strip_cnn_casia_lamp.pt"
CASIA_METRICS_INTERVAL_PATH = RESULTS_DIR / "metrics_casia_interval.json"
CASIA_METRICS_LAMP_PATH = RESULTS_DIR / "metrics_casia_lamp.json"
CASIA_ROBUSTNESS_LAMP_PATH = RESULTS_DIR / "robustness_casia_lamp.csv"
