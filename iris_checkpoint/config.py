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
