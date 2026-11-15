"""Простой baseline: порог по яркости → маска → % облачности.

Нужен, чтобы в презентации было кого бить нейросетью и McNemar.
"""

from __future__ import annotations

import numpy as np


def brightness_cloud_mask(
    rgb: np.ndarray,
    brightness_thr: float = 0.78,
    sat_max: float = 0.35,
) -> np.ndarray:
    """Грубая маска «похоже на облако» по яркости и низкой насыщенности.

    Args:
        rgb: Картинка CxHxW или HxWxC, значения 0..1.
        brightness_thr: Порог яркости (формула Rec.601).
        sat_max: Максимальная «насыщенность» (max-min по каналам),
            выше которой пиксель скорее цветная земля, не облако.

    Returns:
        Бинарная маска HxW с dtype float32 (0/1).
    """
    if rgb.ndim != 3:
        raise ValueError(f"Ожидали 3D массив, получили shape={rgb.shape}")

    if rgb.shape[0] in (3, 4):
        r, g, b = rgb[0], rgb[1], rgb[2]
    elif rgb.shape[2] in (3, 4):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    else:
        raise ValueError(f"Не понял формат каналов: {rgb.shape}")

    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    mask = (brightness >= brightness_thr) & (sat <= sat_max)
    return mask.astype(np.float32)


def cloud_percent_from_mask(mask: np.ndarray) -> float:
    """Переводит маску в процент облачности 0..100.

    Args:
        mask: HxW или 1xHxW с значениями 0/1.

    Returns:
        Процент облачности.
    """
    return float(mask.mean() * 100.0)


def decide_reject(cloud_pct: float, threshold: float = 75.0) -> bool:
    """Решение кейса: удалять ли снимок.

    Args:
        cloud_pct: Оценка облачности в процентах.
        threshold: Порог из ТЗ (по умолчанию 75).

    Returns:
        True, если снимок надо отсеять.
    """
    return cloud_pct > threshold
