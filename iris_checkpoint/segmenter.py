"""FISNet-lite: lightweight fusion U-Net for binary iris segmentation.

The "fusion" follows the multi-scale aggregation idea from FISNet/IrisParseNet:
each decoder block aggregates features from *multiple* encoder levels (the
same-resolution skip plus a downsampled shallower skip), not just the
canonical U-Net same-level skip. Trained on IRISSEG-EP UBIRIS GT (binary
masks); produces an iris-vs-not probability map per pixel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class FISNetLite(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.enc1 = _conv_block(1, c)
        self.enc2 = _conv_block(c, c * 2)
        self.enc3 = _conv_block(c * 2, c * 4)
        self.enc4 = _conv_block(c * 4, c * 8)
        self.bottleneck = _conv_block(c * 8, c * 16)
        self.pool = nn.MaxPool2d(2)

        # Fusion decoder: each block fuses {upsampled deeper, same-level skip,
        # downsampled shallower skip}.
        self.dec4 = _conv_block(c * 16 + c * 8 + c * 4, c * 8)
        self.dec3 = _conv_block(c * 8 + c * 4 + c * 2, c * 4)
        self.dec2 = _conv_block(c * 4 + c * 2 + c, c * 2)
        self.dec1 = _conv_block(c * 2 + c, c)
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        def up(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            return F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)

        def down(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            return F.adaptive_avg_pool2d(t, ref.shape[-2:])

        d4 = self.dec4(torch.cat([up(b, e4), e4, down(e3, e4)], dim=1))
        d3 = self.dec3(torch.cat([up(d4, e3), e3, down(e2, e3)], dim=1))
        d2 = self.dec2(torch.cat([up(d3, e2), e2, down(e1, e2)], dim=1))
        d1 = self.dec1(torch.cat([up(d2, e1), e1], dim=1))
        return self.head(d1)


# --- losses & metrics ---------------------------------------------------------


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(2, 3))
    denom = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def dice_score(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5, eps: float = 1e-6) -> float:
    p = (torch.sigmoid(logits) > thresh).float()
    inter = (p * target).sum(dim=(2, 3))
    denom = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return float(((2 * inter + eps) / (denom + eps)).mean().item())


def iou_score(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5, eps: float = 1e-6) -> float:
    p = (torch.sigmoid(logits) > thresh).float()
    inter = (p * target).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter
    return float(((inter + eps) / (union + eps)).mean().item())


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
