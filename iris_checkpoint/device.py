"""Pick the best torch device (CUDA, Apple MPS, or CPU)."""

from __future__ import annotations

import os

import torch


def pick_device(explicit: str | None = None) -> torch.device:
    """
    Order: explicit arg > env IRIS_DEVICE > CUDA > MPS > CPU.
    Apple Silicon: PyTorch uses the GPU via MPS (Metal), not the Neural Engine
    (Core ML is separate). MPS accelerates this CNN baseline when supported.
    """
    if explicit:
        return torch.device(explicit)
    env = os.environ.get("IRIS_DEVICE", "").strip().lower()
    if env == "cpu":
        return torch.device("cpu")
    if env == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if env == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_summary(dev: torch.device) -> str:
    if dev.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if dev.type == "mps":
        return "mps (Apple GPU / Metal)"
    return "cpu"
