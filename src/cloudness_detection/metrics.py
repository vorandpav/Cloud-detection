"""Метрики сегментации и кейса (%, keep/delete)."""

from __future__ import annotations

import numpy as np


def dice_binary(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Dice для бинарных масок 0/1.

    Args:
        pred: Предсказанная маска.
        target: Истина.
        eps: Стабилизация знаменателя.
    """
    pred_b = pred.astype(bool).ravel()
    target_b = target.astype(bool).ravel()
    inter = float(np.logical_and(pred_b, target_b).sum())
    return float((2.0 * inter + eps) / (pred_b.sum() + target_b.sum() + eps))


def iou_binary(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """IoU (Jaccard) для бинарных масок.

    Args:
        pred: Предсказанная маска.
        target: Истина.
        eps: Стабилизация знаменателя.
    """
    pred_b = pred.astype(bool).ravel()
    target_b = target.astype(bool).ravel()
    inter = float(np.logical_and(pred_b, target_b).sum())
    union = float(np.logical_or(pred_b, target_b).sum())
    return float((inter + eps) / (union + eps))


def mae_cloud_percent(pred_pct: np.ndarray, true_pct: np.ndarray) -> float:
    """MAE по проценту облачности.

    Args:
        pred_pct: Предсказанные проценты.
        true_pct: Истинные проценты.
    """
    return float(np.mean(np.abs(pred_pct - true_pct)))
