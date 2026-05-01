"""Build CASIA-Interval and CASIA-Lamp manifests with subject-disjoint splits.

Identity convention: ``<class>_<L|R>`` — left/right eyes are separate subjects.
Annotates each Interval row with whether IRISSEG-CC GT exists for that frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iris_checkpoint.casia_data import build_casia_manifest, load_casia_interval_gt
from iris_checkpoint.config import (
    CASIA_INTERVAL_DIR,
    CASIA_INTERVAL_MANIFEST,
    CASIA_LAMP_DIR,
    CASIA_LAMP_MANIFEST,
)


def _summarize(name: str, df) -> None:
    n_subj = df["subject_id"].nunique()
    n_class = df["class_id"].nunique()
    print(f"\n--- {name} ---")
    print(f"  images:        {len(df):,}")
    print(f"  raw subjects:  {n_class:,}")
    print(f"  L/R identities:{n_subj:,}")
    by_split_imgs = df["split"].value_counts().to_dict()
    by_split_subj = df.groupby("split")["subject_id"].nunique().to_dict()
    for s in ["train", "val", "test"]:
        print(f"  {s:>5}: {by_split_imgs.get(s, 0):>6} images / {by_split_subj.get(s, 0):>4} ids")
    if "has_gt" in df.columns:
        cov = df["has_gt"].mean()
        print(f"  GT coverage:   {df['has_gt'].sum()}/{len(df)} = {100*cov:.2f}%")


def main() -> None:
    # CASIA-Interval (with GT annotation)
    df_int = build_casia_manifest(CASIA_INTERVAL_DIR)
    df_int["has_gt"] = df_int["img_key"].apply(lambda k: load_casia_interval_gt(k) is not None)
    df_int.to_csv(CASIA_INTERVAL_MANIFEST, index=False)
    _summarize("CASIA-Iris-Interval", df_int)
    print(f"  wrote -> {CASIA_INTERVAL_MANIFEST}")

    # CASIA-Lamp (no GT)
    df_lamp = build_casia_manifest(CASIA_LAMP_DIR)
    df_lamp.to_csv(CASIA_LAMP_MANIFEST, index=False)
    _summarize("CASIA-Iris-Lamp", df_lamp)
    print(f"  wrote -> {CASIA_LAMP_MANIFEST}")


if __name__ == "__main__":
    main()
