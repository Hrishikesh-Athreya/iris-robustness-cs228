"""Daugman-style IrisCode: 1D log-Gabor encoder + Hamming-distance matcher.

Encoder:
  * Take rubber-sheet strip (H x W), encode each row with a bank of 1D
    log-Gabor filters along theta. Quantize the real and imaginary parts
    of the response by sign to produce a binary code (2 bits per filter
    per (row, theta) location, packed into a uint8 code map).

Matcher:
  * Hamming distance between two codes, with cyclic shift along theta to
    handle eye rotation. Mask bits exclude saturated rows (top/bottom of
    strip near pupil/sclera) and any provided per-pixel noise mask.

This is a working implementation, not a faithful re-creation of every
parameter from Daugman 1993. Tunable parameters (n_scales, base wavelength,
sigma_on_f) are exposed at the top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---- log-Gabor filter bank ---------------------------------------------------


def _log_gabor_kernel(width: int, wavelength: float, sigma_on_f: float = 0.5) -> np.ndarray:
    """1D log-Gabor frequency-domain kernel for a row of length `width`."""
    f = np.fft.fftfreq(width)  # cycles/sample, in [-0.5, 0.5)
    f_abs = np.abs(f)
    f_abs[f_abs == 0] = 1e-9  # avoid log(0)
    f0 = 1.0 / wavelength
    kernel = np.exp(-((np.log(f_abs / f0)) ** 2) / (2.0 * (np.log(sigma_on_f)) ** 2))
    kernel[0] = 0.0  # zero DC
    return kernel


@dataclass
class IrisCodeParams:
    n_scales: int = 1
    base_wavelength: float = 18.0
    mult: float = 1.6
    sigma_on_f: float = 0.5


def encode_strip(strip: np.ndarray, params: IrisCodeParams = IrisCodeParams()) -> np.ndarray:
    """Encode a rubber-sheet strip into a binary IrisCode.

    Args:
      strip: (H, W) float or uint8 array.
      params: filter-bank settings.

    Returns:
      code: (n_scales*2, H, W) uint8 with values in {0, 1}.
        Channels are interleaved [Re_s0, Im_s0, Re_s1, Im_s1, ...].
    """
    s = np.asarray(strip, dtype=np.float32)
    if s.dtype != np.float32:
        s = s.astype(np.float32)
    if s.max() > 1.5:
        s = s / 255.0
    H, W = s.shape

    code_channels = []
    for k in range(params.n_scales):
        wavelength = params.base_wavelength * (params.mult ** k)
        kernel = _log_gabor_kernel(W, wavelength, params.sigma_on_f)
        F = np.fft.fft(s, axis=1)
        resp = np.fft.ifft(F * kernel[None, :], axis=1)
        re_bit = (resp.real >= 0).astype(np.uint8)
        im_bit = (resp.imag >= 0).astype(np.uint8)
        code_channels.append(re_bit)
        code_channels.append(im_bit)
    return np.stack(code_channels, axis=0)


# ---- noise / strip mask ------------------------------------------------------


def default_strip_mask(strip_h: int, strip_w: int, fraction_kept: float = 0.7) -> np.ndarray:
    """Suppress rows near the inner pupillary boundary and outer sclera fringe
    where reflection / eyelashes are most likely. Keeps the central
    `fraction_kept` of the radial range.
    """
    keep = np.zeros((strip_h, strip_w), dtype=np.uint8)
    margin = int(round(strip_h * (1.0 - fraction_kept) / 2.0))
    keep[margin : strip_h - margin, :] = 1
    return keep


# ---- matcher -----------------------------------------------------------------


def hamming_distance(
    code_a: np.ndarray,
    code_b: np.ndarray,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
    max_shift: int = 16,
) -> float:
    """Best (minimum) normalized Hamming distance over cyclic theta-shifts.

    code_*: (C, H, W) uint8 in {0,1}. mask_*: (H, W) uint8 in {0,1}; if None,
    full mask is used.
    """
    assert code_a.shape == code_b.shape
    H, W = code_a.shape[-2:]
    if mask_a is None:
        mask_a = np.ones((H, W), dtype=np.uint8)
    if mask_b is None:
        mask_b = np.ones((H, W), dtype=np.uint8)

    best = 1.0
    for shift in range(-max_shift, max_shift + 1):
        b_shift = np.roll(code_b, shift, axis=-1)
        mb_shift = np.roll(mask_b, shift, axis=-1)
        m = (mask_a & mb_shift)[None, :, :]  # (1, H, W)
        diff = np.bitwise_xor(code_a, b_shift)
        diff_masked = diff * m
        n_valid = int(m.sum() * code_a.shape[0])
        if n_valid == 0:
            continue
        hd = float(diff_masked.sum()) / float(n_valid)
        if hd < best:
            best = hd
    return best


def hamming_similarity(
    code_a: np.ndarray,
    code_b: np.ndarray,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
    max_shift: int = 16,
) -> float:
    """Convenience: 1 - HD so higher = more similar (matches eval code that
    expects `higher_is_genuine=True`).
    """
    return 1.0 - hamming_distance(code_a, code_b, mask_a, mask_b, max_shift)
