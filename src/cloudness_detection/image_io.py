"""Чтение каналов Landsat/масок в общий вид 0..1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_raster01(path: Path, percentile_stretch: bool = True) -> np.ndarray:
    """Читает TIFF/PNG/JPEG как float32 HxW в диапазоне примерно 0..1.

    Для uint8-масок — деление на 255.
    Для uint16 каналов Landsat — растяжение по перцентилям 2..98
    (иначе числа ~10k и с JPEG Photos не сравнимы).

    Args:
        path: Путь к файлу.
        percentile_stretch: Если True и динамический диапазон широкий,
            тянем 2–98 перцентиль в 0..1. Для бинарных масок лучше
            ``False`` (но маски uint8 и так идут через /255).

    Returns:
        Массив HxW float32.
    """
    with Image.open(path) as img:
        arr = np.asarray(img, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Ожидали одноканальный растр, shape={arr.shape} у {path}")

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError(f"Пустой растр: {path}")

    vmax = float(finite.max())
    if vmax <= 1.0:
        return arr
    if vmax <= 255.0:
        return arr / 255.0

    if not percentile_stretch:
        return np.clip(arr / 65535.0, 0.0, 1.0)

    nonzero = finite[finite > 0]
    sample = nonzero if nonzero.size > 0 else finite
    lo, hi = np.percentile(sample, [2.0, 98.0])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
