"""Controlled image degradations for robustness sweeps.

Each function takes a uint8 grayscale image (HxW or HxWxC) and a single scalar
intensity, and returns a uint8 image of the same shape. Intensities are chosen
so that 0 means "no degradation" and larger values mean stronger degradation.
"""

from __future__ import annotations

import numpy as np


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur. sigma=0 returns the input unchanged."""
    if sigma <= 1e-6:
        return image.copy()
    radius = max(1, int(round(3 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    g /= g.sum()
    img = image.astype(np.float32)
    if img.ndim == 2:
        # along x then y
        out = np.apply_along_axis(lambda row: np.convolve(row, g, mode="same"), 1, img)
        out = np.apply_along_axis(lambda col: np.convolve(col, g, mode="same"), 0, out)
    else:
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            tmp = np.apply_along_axis(lambda row: np.convolve(row, g, mode="same"), 1, img[..., c])
            out[..., c] = np.apply_along_axis(lambda col: np.convolve(col, g, mode="same"), 0, tmp)
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_noise(image: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Additive Gaussian noise (sigma in raw pixel units, 0..50ish)."""
    if sigma <= 1e-6:
        return image.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, size=image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def illumination_shift(image: np.ndarray, delta: float) -> np.ndarray:
    """Multiplicative gain on intensities. delta in [-0.5, 0.5] ish:
    0 = no change; negative = darker; positive = brighter.
    """
    if abs(delta) < 1e-6:
        return image.copy()
    gain = 1.0 + float(delta)
    out = image.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def specular_reflection(image: np.ndarray, intensity: float, seed: int = 0) -> np.ndarray:
    """Drop a few bright disks onto the image to simulate reflections.

    intensity in [0, 1]; 0 means none, 1 means up to ~5 disks of radius up to
    8% of min image dimension.
    """
    if intensity <= 1e-6:
        return image.copy()
    rng = np.random.default_rng(seed)
    out = image.astype(np.float32).copy()
    H, W = out.shape[:2]
    n_disks = max(1, int(round(5 * intensity)))
    rmax = max(2, int(0.08 * min(H, W) * intensity))
    yy, xx = np.indices((H, W))
    for _ in range(n_disks):
        cx = int(rng.integers(0, W))
        cy = int(rng.integers(0, H))
        r = int(rng.integers(2, rmax + 1))
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = d2 <= r * r
        if out.ndim == 2:
            out[mask] = 255.0
        else:
            out[mask, ...] = 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def off_angle_warp(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Approximate an off-angle gaze with a horizontal squeeze (affine).

    angle_deg in [0, 30]; 0 = identity. We compress the x axis by cos(angle)
    around the image center.
    """
    if abs(angle_deg) < 1e-6:
        return image.copy()
    s = float(np.cos(np.deg2rad(angle_deg)))
    H, W = image.shape[:2]
    cx = (W - 1) / 2.0
    yy, xx = np.indices((H, W)).astype(np.float32)
    # Inverse map: source x for each dest x.
    src_x = (xx - cx) / s + cx
    src_y = yy
    src_x = np.clip(src_x, 0, W - 1.001)
    x0 = np.floor(src_x).astype(np.int32)
    x1 = x0 + 1
    wx = (src_x - x0).astype(np.float32)
    img = image.astype(np.float32)
    if img.ndim == 2:
        Ia = img[src_y.astype(np.int32), x0]
        Ib = img[src_y.astype(np.int32), x1]
        out = Ia * (1 - wx) + Ib * wx
    else:
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            Ia = img[src_y.astype(np.int32), x0, c]
            Ib = img[src_y.astype(np.int32), x1, c]
            out[..., c] = Ia * (1 - wx) + Ib * wx
    return np.clip(out, 0, 255).astype(np.uint8)


SWEEPS: dict[str, list[float]] = {
    "blur": [0.0, 1.0, 2.0, 3.0, 4.0],
    "noise": [0.0, 5.0, 10.0, 20.0, 30.0],
    "illumination": [-0.4, -0.2, 0.0, 0.2, 0.4],
    "specular": [0.0, 0.25, 0.5, 0.75, 1.0],
    "off_angle": [0.0, 5.0, 10.0, 20.0, 30.0],
}

DEGRADERS = {
    "blur": gaussian_blur,
    "noise": gaussian_noise,
    "illumination": illumination_shift,
    "specular": specular_reflection,
    "off_angle": off_angle_warp,
}
