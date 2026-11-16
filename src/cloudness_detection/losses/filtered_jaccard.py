"""Filtered Jaccard Loss (Cloud-Net+, FJL1).

Статья: arXiv:2001.08768. На пустой маске (S ≈ 0) берётся inverted Jaccard,
иначе обычный soft Jaccard. Переключение — sigmoid, m=1000, порог S=0.5
(в пикселях, не в доле): почти любой облачный патч идёт в soft Jaccard.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _jaccard_per_image(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    """1 − soft IoU, вектор длины B.

    Args:
        pred: Вероятности Bx1xHxW.
        target: Маска Bx1xHxW.
        eps: Стабилизация.
    """
    dims = (1, 2, 3)
    inter = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - inter
    return 1.0 - (inter + eps) / (union + eps)


class FilteredJaccardLoss(nn.Module):
    """FJL1: soft Jaccard + inverted Jaccard на clear-кадрах."""

    def __init__(self, m: float = 1000.0, cutoff: float = 0.5, eps: float = 1e-6) -> None:
        """Args:
            m: Крутизна sigmoid (в статье 1000).
            cutoff: Порог суммы пикселей GT (в статье 0.5).
            eps: Стабилизация Jaccard.
        """
        super().__init__()
        self.m = m
        self.cutoff = cutoff
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: Bx1xHxW без sigmoid.
            target: Bx1xHxW в 0..1.
        """
        probs = torch.sigmoid(logits)
        jaccard = _jaccard_per_image(probs, target, self.eps)
        inverted = _jaccard_per_image(1.0 - probs, 1.0 - target, self.eps)
        foreground = target.sum(dim=(1, 2, 3))
        # LP = 1 на пустой маске, ~0 как только есть хотя бы пиксель облака.
        lowpass = torch.sigmoid(-self.m * (foreground - self.cutoff))
        return (inverted * lowpass + jaccard * (1.0 - lowpass)).mean()
