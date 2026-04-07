"""Build image manifest from CASIA, UBIRIS.V2, or synthetic demo data."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import DATA_DIR, IMG_SIZE, RANDOM_SEED, TRAIN_FRAC, VAL_FRAC

# UBIRIS.V2 filenames: C{class}_S{session}_I{image}.tiff (also segmentation exports).
UBIRIS_SUBJECT_RE = re.compile(r"C(\d+)_S(\d+)_I(\d+)\.", re.IGNORECASE)
_IMAGE_EXTS = {".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff"}


def _subject_from_casia_path(path: Path) -> str | None:
    """Infer subject id from common CASIA-IrisV3-Interval paths (e.g. .../024/024_2/024_2_1.jpg)."""
    parts = path.parts
    for p in parts:
        if re.fullmatch(r"\d{3}", p):
            return p
    m = re.match(r"^(\d{3})[_/]", path.name)
    if m:
        return m.group(1)
    m = re.search(r"S(\d+)[_L]", path.name, re.I)
    if m:
        return m.group(1).zfill(3)[-3:] if len(m.group(1)) >= 3 else m.group(1).zfill(3)
    return None


def discover_casia_images(root: Path) -> list[tuple[Path, str]]:
    root = root.resolve()
    out: list[tuple[Path, str]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        sid = _subject_from_casia_path(p.relative_to(root))
        if sid is None:
            # fallback: top-level numeric folder under root
            rel = p.relative_to(root)
            if len(rel.parts) > 0 and rel.parts[0].isdigit():
                sid = rel.parts[0][:3].zfill(3)[-3:]
            else:
                continue
        out.append((p, sid))
    return out


def subject_from_ubiris_filename(name: str) -> str | None:
    m = UBIRIS_SUBJECT_RE.search(name)
    return m.group(1) if m else None


def discover_ubiris_v2_images(root: Path) -> list[tuple[Path, str]]:
    """Scan root recursively for UBIRIS.V2-style iris images; subject = class id C*."""
    root = root.resolve()
    out: list[tuple[Path, str]] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in _IMAGE_EXTS:
            continue
        sid = subject_from_ubiris_filename(p.name)
        if sid is None:
            continue
        out.append((p, sid))
    return out


def build_manifest_ubiris(ubiris_root: Path) -> pd.DataFrame:
    pairs = discover_ubiris_v2_images(ubiris_root)
    if not pairs:
        raise FileNotFoundError(f"No UBIRIS-style images under {ubiris_root}")
    subjects = [sid for _, sid in pairs]
    smap = subject_disjoint_split(subjects)
    rows = []
    for path, sid in pairs:
        rows.append(
            {
                "path": str(path),
                "subject_id": sid,
                "split": smap[sid],
            }
        )
    return pd.DataFrame(rows)


def subject_disjoint_split(
    subjects: list[str], seed: int = RANDOM_SEED
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    u = sorted(set(subjects))
    rng.shuffle(u)
    n = len(u)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_s = set(u[:n_train])
    val_s = set(u[n_train : n_train + n_val])
    test_s = set(u[n_train + n_val :])
    split_map: dict[str, str] = {}
    for s in u:
        if s in train_s:
            split_map[s] = "train"
        elif s in val_s:
            split_map[s] = "val"
        else:
            split_map[s] = "test"
    return split_map


def build_manifest_casia(casia_root: Path) -> pd.DataFrame:
    pairs = discover_casia_images(casia_root)
    if not pairs:
        raise FileNotFoundError(f"No images found under {casia_root}")
    subjects = [sid for _, sid in pairs]
    smap = subject_disjoint_split(subjects)
    rows = []
    for path, sid in pairs:
        rows.append(
            {
                "path": str(path),
                "subject_id": sid,
                "split": smap[sid],
            }
        )
    return pd.DataFrame(rows)


def synthesize_demo_dataset(
    out_root: Path,
    n_subjects: int = 80,
    images_per_subject: int = 16,
    size: int = IMG_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create a small grayscale corpus for pipeline testing (not a substitute for CASIA)."""
    rng = np.random.default_rng(seed)
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in range(n_subjects):
        sid = f"{s:03d}"
        subdir = out_root / sid / f"{sid}_1"
        subdir.mkdir(parents=True, exist_ok=True)
        base_phase = rng.uniform(0, 2 * np.pi)
        base_freq = rng.uniform(10.0, 22.0)
        for k in range(images_per_subject):
            u, v = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
            r = np.sqrt(u**2 + v**2) + 1e-6
            jitter = rng.normal(0, 0.08, size=(size, size))
            img = (
                np.sin(base_freq * r * np.pi + base_phase)
                * np.cos((base_freq * 0.7) * u * np.pi + k * 0.15)
                + jitter
            )
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
            fp = subdir / f"{sid}_1_{k+1}.jpg"
            Image.fromarray(img).save(fp, quality=95)
            rows.append({"path": str(fp), "subject_id": sid})
    subjects = [r["subject_id"] for r in rows]
    smap = subject_disjoint_split(subjects)
    for r in rows:
        r["split"] = smap[r["subject_id"]]
    return pd.DataFrame(rows)


def load_manifest(path: Path | None = None) -> pd.DataFrame:
    p = path or (DATA_DIR.parent / "manifest.csv")
    return pd.read_csv(p)


def image_stats_worker(path: str) -> dict[str, float]:
    """Top-level helper for multiprocessing (must stay picklable)."""
    return image_stats(path)


def image_stats(path: str) -> dict[str, float]:
    im = np.array(Image.open(path).convert("L"), dtype=np.float32)
    # Discrete Laplacian variance as a blur proxy (higher → sharper).
    lap = (
        -4 * im
        + np.roll(im, 1, 0)
        + np.roll(im, -1, 0)
        + np.roll(im, 1, 1)
        + np.roll(im, -1, 1)
    )
    var_lap = float(np.var(lap))
    return {
        "mean_intensity": float(im.mean()),
        "std_intensity": float(im.std()),
        "blur_proxy_inv": var_lap,
    }
