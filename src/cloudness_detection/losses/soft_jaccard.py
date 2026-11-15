"""Soft Jaccard / Dice+BCE (идея Cloud-Net+ FJL-lite)."""

from __future__ import annotations

import torch
import torch.nn as nn


def soft_jaccard_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """1 − soft IoU по батчу.

    Args:
        logits: Bx1xHxW без sigmoid.
        target: Bx1xHxW в 0..1.
        eps: Стабилизация.

    Returns:
        Скаляр loss.
    """
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims) - inter
    jaccard = (inter + eps) / (union + eps)
    return 1.0 - jaccard.mean()


class SoftJaccardLoss(nn.Module):
    """Только soft Jaccard (FJL-lite без focal-обёртки)."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: Bx1xHxW.
            target: Bx1xHxW.
        """
        return soft_jaccard_loss(logits, target)


class DiceBCELoss(nn.Module):
    """BCE + soft Dice — стабильный дефолт для U-Net/SegFormer."""

    def __init__(self, jaccard_weight: float = 0.0) -> None:
        """Args:
            jaccard_weight: Если >0, добавляем soft Jaccard (FJL-lite mix).
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: Bx1xHxW без sigmoid.
            target: Bx1xHxW в 0..1.
        """
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)
        loss = bce + dice.mean()
        if self.jaccard_weight > 0:
            loss = loss + self.jaccard_weight * soft_jaccard_loss(logits, target)
        return loss
