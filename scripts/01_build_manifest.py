#!/usr/bin/env python3
"""Build subject-disjoint manifest.csv (UBIRIS.V2, CASIA, or synthetic)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iris_checkpoint.config import DATA_DIR, DEFAULT_UBIRIS_CLASSES_DIR, MANIFEST_PATH
from iris_checkpoint.dataset import (
    build_manifest_casia,
    build_manifest_ubiris,
    synthesize_demo_dataset,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="If no dataset flag is given and data/archive (2)/CLASSES_400_300_Part1 exists, UBIRIS is used by default."
    )
    ap.add_argument(
        "--ubiris-root",
        type=Path,
        default=None,
        help="UBIRIS.V2 image root (e.g. CLASSES_400_300_Part1).",
    )
    ap.add_argument(
        "--casia-root",
        type=Path,
        default=None,
        help="CASIA-IrisV3-Interval (or similar) root.",
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate demo images under data/synthetic_demo/.",
    )
    args = ap.parse_args()

    if args.synthetic:
        out = DATA_DIR / "synthetic_demo"
        df = synthesize_demo_dataset(out)
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(MANIFEST_PATH, index=False)
        print(f"Wrote synthetic data under {out} and manifest {MANIFEST_PATH}")
        print(df.groupby("split").size())
        return

    if args.casia_root is not None:
        df = build_manifest_casia(args.casia_root.resolve())
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(MANIFEST_PATH, index=False)
        print(f"CASIA: wrote {len(df)} rows to {MANIFEST_PATH}")
        print(df.groupby("split").size())
        return

    ub = args.ubiris_root
    if ub is None and DEFAULT_UBIRIS_CLASSES_DIR.is_dir():
        ub = DEFAULT_UBIRIS_CLASSES_DIR
    if ub is None:
        print(
            "No default UBIRIS folder found. Use --ubiris-root, --casia-root, or --synthetic.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = build_manifest_ubiris(ub.resolve())
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"UBIRIS.V2: wrote {len(df)} rows from {ub} -> {MANIFEST_PATH}")
    print(df.groupby("split").size())


if __name__ == "__main__":
    main()
