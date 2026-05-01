"""Daugman rubber-sheet polar normalization.

Given a binary iris mask, fit two circles (pupil inner boundary, iris outer
boundary), then unwrap the iris annulus into a fixed-size polar strip
(STRIP_HEIGHT x STRIP_WIDTH) using bilinear sampling. The strip is the
normalized template that both the IrisCode encoder and the strip-CNN consume.

Notes:
  * IRISSEG-EP UBIRIS GT is binary (iris vs not), so the pupil is *inside*
    the mask's hole. We approximate the inner (pupil) boundary by fitting a
    circle to the inner contour of the iris ring.
  * If the iris region in the mask is too small/degenerate, returns None and
    the caller should fall back (skip the sample, or use a default
    centroid-based circle).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IrisCircles:
    pupil_cx: float
    pupil_cy: float
    pupil_r: float
    iris_cx: float
    iris_cy: float
    iris_r: float


def _fit_circle_least_squares(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float] | None:
    """Algebraic least-squares circle fit (Kasa 1976). Robust enough here."""
    if len(xs) < 5:
        return None
    A = np.column_stack([2 * xs, 2 * ys, np.ones_like(xs)])
    b = xs * xs + ys * ys
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return float(cx), float(cy), float(np.sqrt(r2))


def _outer_contour_points(mask: np.ndarray) -> np.ndarray:
    """Pixels on the iris outer boundary: iris pixels with at least one
    background neighbour (4-connected).
    """
    m = mask.astype(bool)
    pad = np.pad(m, 1, mode="constant", constant_values=False)
    n = (
        pad[:-2, 1:-1].astype(np.int8)
        + pad[2:, 1:-1].astype(np.int8)
        + pad[1:-1, :-2].astype(np.int8)
        + pad[1:-1, 2:].astype(np.int8)
    )
    outer = m & (n < 4)
    ys, xs = np.where(outer)
    return np.column_stack([xs, ys])  # (x, y)


def fit_iris_circles(mask: np.ndarray) -> IrisCircles | None:
    """Fit pupil (inner) and iris (outer) circles from a binary iris mask.

    The mask is expected to be {0, 1} or {0, 255}. We treat any non-zero pixel
    as iris. Returns None if the iris region is too small or the fit fails.
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 200:
        return None

    contour = _outer_contour_points(m)
    if len(contour) < 20:
        return None
    fit_outer = _fit_circle_least_squares(contour[:, 0].astype(np.float64), contour[:, 1].astype(np.float64))
    if fit_outer is None:
        return None
    cx_o, cy_o, r_o = fit_outer

    # Inner (pupil) boundary: pixels just OUTSIDE the iris on the *inside* of
    # the ring. Approximate by taking the centroid of the *non-iris* region
    # that lies within the fitted outer disk and fitting a circle to its rim.
    yy, xx = np.indices(m.shape)
    inside_outer = (xx - cx_o) ** 2 + (yy - cy_o) ** 2 <= (0.95 * r_o) ** 2
    hole = inside_outer & (m == 0)
    if hole.sum() < 50:
        # Fallback: place a small pupil at the iris center.
        return IrisCircles(cx_o, cy_o, max(3.0, 0.25 * r_o), cx_o, cy_o, r_o)

    hys, hxs = np.where(hole)
    # Hole boundary = hole pixels with an iris neighbour
    pad = np.pad(hole, 1, mode="constant", constant_values=False)
    n_iris = (
        (pad[:-2, 1:-1] != hole).astype(np.int8)
        + (pad[2:, 1:-1] != hole).astype(np.int8)
        + (pad[1:-1, :-2] != hole).astype(np.int8)
        + (pad[1:-1, 2:] != hole).astype(np.int8)
    )
    rim = hole & (n_iris > 0)
    rys, rxs = np.where(rim)
    if len(rxs) < 10:
        return IrisCircles(float(hxs.mean()), float(hys.mean()), max(3.0, 0.25 * r_o), cx_o, cy_o, r_o)

    fit_inner = _fit_circle_least_squares(rxs.astype(np.float64), rys.astype(np.float64))
    if fit_inner is None:
        return IrisCircles(float(hxs.mean()), float(hys.mean()), max(3.0, 0.25 * r_o), cx_o, cy_o, r_o)
    cx_i, cy_i, r_i = fit_inner

    # Sanity bounds.
    r_i = float(np.clip(r_i, 0.05 * r_o, 0.85 * r_o))
    return IrisCircles(cx_i, cy_i, r_i, cx_o, cy_o, r_o)


def unwrap_iris(
    image: np.ndarray,
    circles: IrisCircles,
    strip_h: int = 64,
    strip_w: int = 512,
) -> np.ndarray:
    """Daugman rubber-sheet polar unwrap from raw image + circles to a strip.

    For each (theta, r) in the strip, we sample a point linearly between the
    pupil and iris boundaries:
        x(r,theta) = (1-r) * pupil_x(theta) + r * iris_x(theta)
        y(r,theta) = (1-r) * pupil_y(theta) + r * iris_y(theta)
    where pupil_x(theta) = pupil_cx + pupil_r*cos(theta), etc., and r in [0,1].
    """
    H, W = image.shape[:2]
    img = image.astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)

    thetas = np.linspace(0.0, 2.0 * np.pi, strip_w, endpoint=False)
    rs = np.linspace(0.0, 1.0, strip_h)

    cos_t = np.cos(thetas)[None, :]  # (1, strip_w)
    sin_t = np.sin(thetas)[None, :]

    px = circles.pupil_cx + circles.pupil_r * cos_t  # (1, strip_w)
    py = circles.pupil_cy + circles.pupil_r * sin_t
    ix = circles.iris_cx + circles.iris_r * cos_t
    iy = circles.iris_cy + circles.iris_r * sin_t

    R = rs[:, None]  # (strip_h, 1)
    sample_x = (1 - R) * px + R * ix  # (strip_h, strip_w)
    sample_y = (1 - R) * py + R * iy

    sample_x = np.clip(sample_x, 0, W - 1.001)
    sample_y = np.clip(sample_y, 0, H - 1.001)

    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = sample_x - x0
    wy = sample_y - y0

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]
    out = (
        Ia * (1 - wx) * (1 - wy)
        + Ic * wx * (1 - wy)
        + Ib * (1 - wx) * wy
        + Id * wx * wy
    )
    return out.astype(np.float32)


def unwrap_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    strip_h: int = 64,
    strip_w: int = 512,
) -> tuple[np.ndarray | None, IrisCircles | None]:
    """Convenience: fit circles from mask, then unwrap. Returns (None, None)
    if circle fitting fails.
    """
    circles = fit_iris_circles(mask)
    if circles is None:
        return None, None
    strip = unwrap_iris(image, circles, strip_h=strip_h, strip_w=strip_w)
    return strip, circles
