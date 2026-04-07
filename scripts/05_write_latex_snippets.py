#!/usr/bin/env python3
"""Write results_inc.tex for \\input in the report from metrics.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iris_checkpoint.config import LATEX_SNIPPET_PATH, METRICS_PATH


def esc(s: str) -> str:
    return s.replace("_", "\\_")


def main() -> None:
    with open(METRICS_PATH) as f:
        m = json.load(f)

    lines = [
        f"\\newcommand{{\\ValEER}}{{{m['val_eer']:.4f}}}",
        f"\\newcommand{{\\TestFAR}}{{{m['test_far_at_val_threshold']:.4f}}}",
        f"\\newcommand{{\\TestFRR}}{{{m['test_frr_at_val_threshold']:.4f}}}",
        f"\\newcommand{{\\TestAcc}}{{{m['test_accuracy_at_val_threshold']:.4f}}}",
        f"\\newcommand{{\\TestEER}}{{{m['test_eer']:.4f}}}",
        f"\\newcommand{{\\NGenPairs}}{{{m['n_genuine_pairs']}}}",
        f"\\newcommand{{\\NImpPairs}}{{{m['n_impostor_pairs']}}}",
    ]
    LATEX_SNIPPET_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEX_SNIPPET_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {LATEX_SNIPPET_PATH}")


if __name__ == "__main__":
    main()
