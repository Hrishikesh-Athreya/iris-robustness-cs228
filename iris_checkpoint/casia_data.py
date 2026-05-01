"""CASIA-Iris helpers: identity = (subject, eye), GT loading, mask rasterization.

Filenames follow CASIA-IrisV3 convention: ``S<class><L|R><img>.jpg`` (e.g.
``S1001L01.jpg``). Left and right irises of the same person are *different*
identities (different texture), so the canonical subject id used downstream
is ``<class>_<eye>`` (e.g. ``1001_L``).

GT (IRISSEG-CC, Halmstad) for CASIA-IrisV3-Interval lives next door:
  ``manual_segmentation/<key>.mat``           --> CC_PUPIL, RADIO_PUPIL,
                                                CC_SCLERA, RADIO_SCLERA
  ``manual_occlusion_eyelids/<key>_eyelids_circles.mat``
                                              --> upper_eyelid, lower_eyelid

The segmentation file is MATLAB v7.3 (HDF5); the occlusion file is older v5.
We try scipy first and fall back to h5py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CASIA_INTERVAL_DIR,
    CASIA_INTERVAL_GT_OCC,
    CASIA_INTERVAL_GT_SEG,
    CASIA_LAMP_DIR,
    RANDOM_SEED,
    TRAIN_FRAC,
    VAL_FRAC,
)

CASIA_FN_RE = re.compile(r"S(\d+)([LR])(\d+)\.(?:jpg|jpeg|bmp|png|tif|tiff)$", re.IGNORECASE)


def parse_casia_filename(name: str) -> tuple[str, str, str] | None:
    """Return (class_id, eye, frame) or None."""
    m = CASIA_FN_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2).upper(), m.group(3)


def casia_subject_id(class_id: str, eye: str) -> str:
    """Canonical identity: '<class>_<L|R>'."""
    return f"{class_id}_{eye.upper()}"


def discover_casia_images(root: Path) -> list[tuple[Path, str]]:
    """Recursively find CASIA-style images under root, return (path, identity).

    Identity is `<class>_<eye>`. Skips hidden files and any file that doesn't
    match the SXXXX(L|R)YY.<ext> pattern.
    """
    out: list[tuple[Path, str]] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        parsed = parse_casia_filename(p.name)
        if parsed is None:
            continue
        cls, eye, _ = parsed
        out.append((p, casia_subject_id(cls, eye)))
    return out


def subject_disjoint_split(subjects: list[str], seed: int = RANDOM_SEED) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    u = sorted(set(subjects))
    rng.shuffle(u)
    n = len(u)
    n_tr = int(n * TRAIN_FRAC)
    n_va = int(n * VAL_FRAC)
    smap: dict[str, str] = {}
    for i, s in enumerate(u):
        if i < n_tr:
            smap[s] = "train"
        elif i < n_tr + n_va:
            smap[s] = "val"
        else:
            smap[s] = "test"
    return smap


def build_casia_manifest(root: Path) -> pd.DataFrame:
    pairs = discover_casia_images(root)
    if not pairs:
        raise FileNotFoundError(f"No CASIA-format images under {root}")
    smap = subject_disjoint_split([sid for _, sid in pairs])
    rows = []
    for p, sid in pairs:
        m = CASIA_FN_RE.match(p.name)
        cls, eye, _ = m.group(1), m.group(2).upper(), m.group(3)
        rows.append(
            {
                "path": str(p),
                "subject_id": sid,
                "class_id": cls,
                "eye": eye,
                "split": smap[sid],
                "img_key": p.stem,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- GT loading


@dataclass
class IrisGTCircles:
    """Parametric IRISSEG-CC ground truth in image (row, col) coordinates."""

    pupil_cy: float  # row
    pupil_cx: float  # col
    pupil_r: float
    iris_cy: float   # row
    iris_cx: float   # col
    iris_r: float
    upper_cx: float | None   # col
    upper_cy: float | None   # row
    upper_r: float | None
    lower_cx: float | None
    lower_cy: float | None
    lower_r: float | None


def _load_mat_any(path: Path) -> dict[str, np.ndarray]:
    """Load .mat that may be either v7.3 (HDF5) or older (v5). Returns dict
    of squeezed numpy arrays, excluding scipy/MATLAB header keys.
    """
    # try scipy first (older format, very fast)
    try:
        from scipy.io import loadmat as _loadmat

        m = _loadmat(path)
        return {
            k: np.array(v).squeeze()
            for k, v in m.items()
            if not k.startswith("__")
        }
    except NotImplementedError:
        pass
    except Exception:
        pass
    # fall back to h5py for v7.3
    import h5py

    with h5py.File(path, "r") as f:
        return {k: np.array(f[k]).squeeze() for k in f.keys()}


def load_casia_interval_gt(img_key: str) -> IrisGTCircles | None:
    """Look up segmentation + occlusion .mat for a given image key (e.g.
    'S1001L01'). Returns None if either GT file is missing or malformed.
    """
    seg_path = Path(CASIA_INTERVAL_GT_SEG) / f"{img_key}.mat"
    occ_path = Path(CASIA_INTERVAL_GT_OCC) / f"{img_key}_eyelids_circles.mat"
    if not seg_path.exists():
        return None
    try:
        seg = _load_mat_any(seg_path)
        # CC_PUPIL = [row, col], RADIO_* scalars
        cc_p = np.atleast_1d(seg["CC_PUPIL"]).astype(np.float64).ravel()
        cc_s = np.atleast_1d(seg["CC_SCLERA"]).astype(np.float64).ravel()
        rp = float(np.atleast_1d(seg["RADIO_PUPIL"]).ravel()[0])
        rs = float(np.atleast_1d(seg["RADIO_SCLERA"]).ravel()[0])
    except Exception:
        return None
    upper = lower = (None, None, None)
    if occ_path.exists():
        try:
            occ = _load_mat_any(occ_path)
            ue = np.atleast_1d(occ["upper_eyelid"]).astype(np.float64).ravel()
            le = np.atleast_1d(occ["lower_eyelid"]).astype(np.float64).ravel()
            upper = (float(ue[0]), float(ue[1]), float(ue[2]))
            lower = (float(le[0]), float(le[1]), float(le[2]))
        except Exception:
            pass
    return IrisGTCircles(
        pupil_cy=float(cc_p[0]),
        pupil_cx=float(cc_p[1]),
        pupil_r=rp,
        iris_cy=float(cc_s[0]),
        iris_cx=float(cc_s[1]),
        iris_r=rs,
        upper_cx=upper[0],
        upper_cy=upper[1],
        upper_r=upper[2],
        lower_cx=lower[0],
        lower_cy=lower[1],
        lower_r=lower[2],
    )


# ---------------------------------------------------------------- rasterize


def rasterize_iris_mask(
    circles: IrisGTCircles,
    height: int,
    width: int,
    use_eyelids: bool = True,
) -> np.ndarray:
    """Return a binary (uint8 0/255) iris mask matching the IRISSEG-CC
    plotter convention:
        iris = inside sclera ∧ outside pupil ∧ inside both eyelid circles
    where "inside eyelid circle" means the iris-visible region per the
    Halmstad plotter.

    Coordinates: rows 1..height, cols 1..width in the .m file (1-based);
    we use 0-based indexing here (matters only at the half-pixel level).
    """
    yy, xx = np.indices((height, width)).astype(np.float64)
    iris_outer = (xx - circles.iris_cx) ** 2 + (yy - circles.iris_cy) ** 2 <= circles.iris_r ** 2
    iris_inner = (xx - circles.pupil_cx) ** 2 + (yy - circles.pupil_cy) ** 2 >= circles.pupil_r ** 2
    mask = iris_outer & iris_inner
    if use_eyelids and circles.upper_r is not None and circles.lower_r is not None:
        upper = (xx - circles.upper_cx) ** 2 + (yy - circles.upper_cy) ** 2 < circles.upper_r ** 2
        lower = (xx - circles.lower_cx) ** 2 + (yy - circles.lower_cy) ** 2 < circles.lower_r ** 2
        mask = mask & upper & lower
    return (mask.astype(np.uint8)) * 255


def to_iris_circles(c: IrisGTCircles):
    """Convert a parametric GT to the rubber_sheet.IrisCircles dataclass for
    use with ``unwrap_iris``."""
    from .rubber_sheet import IrisCircles

    return IrisCircles(
        pupil_cx=float(c.pupil_cx),
        pupil_cy=float(c.pupil_cy),
        pupil_r=float(c.pupil_r),
        iris_cx=float(c.iris_cx),
        iris_cy=float(c.iris_cy),
        iris_r=float(c.iris_r),
    )


def gt_eyelid_mask_only(
    circles: IrisGTCircles,
    height: int,
    width: int,
) -> np.ndarray:
    """Return a binary mask of the *non-occluded* eyelid region (1 = visible).

    Used to feed the IrisCode noise mask along the rubber-sheet strip.
    """
    yy, xx = np.indices((height, width)).astype(np.float64)
    if circles.upper_r is None or circles.lower_r is None:
        return np.ones((height, width), dtype=np.uint8)
    upper = (xx - circles.upper_cx) ** 2 + (yy - circles.upper_cy) ** 2 < circles.upper_r ** 2
    lower = (xx - circles.lower_cx) ** 2 + (yy - circles.lower_cy) ** 2 < circles.lower_r ** 2
    return (upper & lower).astype(np.uint8)
