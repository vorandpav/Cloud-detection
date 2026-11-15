"""Нормализация цвета публичных RGB к домену Photos.

Три простых режима:
1. percentile stretch 2–98 по каналам (уже близко к ``image_io.load_raster01``);
2. match mean/std каналов к референсу (Photos);
3. Reinhard-like перенос в Lab (линейный match μ/σ по Lab-каналам).

Для train: кэш статистик Photos в ``outputs/eda/deep/photos_ref_stats.json``;
флаг CLI ``--color-match-photos`` в ``train-mix``. На Kaggle Cloud-Adapter
этот match **не** применять — у DINOv2 свой препроцесс.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

# Корень репо: .../src/cloudness_detection/color_norm.py → parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHOTOS_REF_STATS = _REPO_ROOT / "outputs" / "eda" / "deep" / "photos_ref_stats.json"

# Fallback, если json ещё не посчитан (seed 42, 80 Photos).
_FALLBACK_PHOTOS_STATS: dict[str, list[float]] = {
    "mean_rgb": [0.45349547266960144, 0.5447981357574463, 0.7044731974601746],
    "std_rgb": [0.20009629428386688, 0.19534730911254883, 0.1615404188632965],
    "mean_lab": [57.05459213256836, 3.065854787826538, -24.345277786254883],
    "std_lab": [18.75975227355957, 5.1958160400390625, 9.936075210571289],
}


def percentile_stretch(
    rgb: np.ndarray,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
) -> np.ndarray:
    """Тянет каждый канал RGB в 0..1 по перцентилям.

    Args:
        rgb: HxWx3 float или uint8.
        lo_pct: Нижний перцентиль.
        hi_pct: Верхний перцентиль.

    Returns:
        HxWx3 float32 в 0..1.
    """
    arr = rgb.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(3):
        ch = arr[..., c]
        finite = ch[np.isfinite(ch)]
        if finite.size == 0:
            out[..., c] = 0.0
            continue
        lo, hi = np.percentile(finite, [lo_pct, hi_pct])
        if hi <= lo:
            out[..., c] = 0.0
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return out


def match_mean_std(
    rgb: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Подгоняет mean/std каналов изображения к референсу Photos.

    Args:
        rgb: HxWx3 в 0..1.
        ref_mean: (3,) целевые средние.
        ref_std: (3,) целевые std.
        eps: Защита от деления на ноль.

    Returns:
        HxWx3 float32, клип в 0..1.
    """
    arr = rgb.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    src_mean = arr.reshape(-1, 3).mean(axis=0)
    src_std = arr.reshape(-1, 3).std(axis=0)
    out = (arr - src_mean) / np.maximum(src_std, eps) * ref_std + ref_mean
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Грубый RGB→Lab через XYZ (D65), без ICC.

    Args:
        rgb: HxWx3 в 0..1.

    Returns:
        HxWx3 Lab.
    """
    arr = np.clip(rgb.astype(np.float64), 0.0, 1.0)
    # sRGB linear
    mask = arr <= 0.04045
    lin = np.empty_like(arr)
    lin[mask] = arr[mask] / 12.92
    lin[~mask] = ((arr[~mask] + 0.055) / 1.055) ** 2.4
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = lin @ m.T
    # D65 white
    xyz_n = xyz / np.array([0.95047, 1.0, 1.08883])
    eps = 216 / 24389
    kappa = 24389 / 27

    def f(t: np.ndarray) -> np.ndarray:
        out = np.empty_like(t)
        small = t <= eps
        out[small] = (kappa * t[small] + 16.0) / 116.0
        out[~small] = np.cbrt(t[~small])
        return out

    fxyz = f(xyz_n)
    lab = np.empty_like(fxyz)
    lab[..., 0] = 116.0 * fxyz[..., 1] - 16.0
    lab[..., 1] = 500.0 * (fxyz[..., 0] - fxyz[..., 1])
    lab[..., 2] = 200.0 * (fxyz[..., 1] - fxyz[..., 2])
    return lab


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Обратное Lab→sRGB.

    Args:
        lab: HxWx3.

    Returns:
        HxWx3 float32 0..1.
    """
    lab = lab.astype(np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = lab[..., 1] / 500.0 + fy
    fz = fy - lab[..., 2] / 200.0
    eps = 216 / 24389
    kappa = 24389 / 27

    def finv(t: np.ndarray) -> np.ndarray:
        t3 = t**3
        out = np.empty_like(t)
        big = t3 > eps
        out[big] = t3[big]
        out[~big] = (116.0 * t[~big] - 16.0) / kappa
        return out

    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1)
    xyz *= np.array([0.95047, 1.0, 1.08883])
    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float64,
    )
    lin = xyz @ m_inv.T
    lin = np.clip(lin, 0.0, None)
    mask = lin <= 0.0031308
    srgb = np.empty_like(lin)
    srgb[mask] = 12.92 * lin[mask]
    srgb[~mask] = 1.055 * np.power(lin[~mask], 1.0 / 2.4) - 0.055
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def reinhard_lab_transfer(
    rgb: np.ndarray,
    ref_lab_mean: np.ndarray,
    ref_lab_std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Reinhard-like: match μ/σ в Lab к референсу Photos.

    Args:
        rgb: HxWx3 в 0..1.
        ref_lab_mean: (3,) Lab mean референса.
        ref_lab_std: (3,) Lab std референса.
        eps: Защита от деления на ноль.

    Returns:
        HxWx3 float32 0..1.
    """
    arr = rgb.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    lab = _rgb_to_lab(arr)
    src_mean = lab.reshape(-1, 3).mean(axis=0)
    src_std = lab.reshape(-1, 3).std(axis=0)
    lab_out = (lab - src_mean) / np.maximum(src_std, eps) * ref_lab_std + ref_lab_mean
    return _lab_to_rgb(lab_out)


def histogram_match_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Одномерный histogram matching канала к референсу.

    Args:
        source: Плоский или HxW float 0..1.
        reference: Плоский или HxW float 0..1.

    Returns:
        Массив той же формы, что source.
    """
    src = source.ravel()
    ref = reference.ravel()
    src_values, src_idx, src_counts = np.unique(src, return_inverse=True, return_counts=True)
    ref_values, ref_counts = np.unique(ref, return_counts=True)
    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_counts).astype(np.float64)
    ref_cdf /= ref_cdf[-1]
    interp = np.interp(src_cdf, ref_cdf, ref_values)
    return interp[src_idx].reshape(source.shape).astype(np.float32)


def histogram_match_rgb(rgb: np.ndarray, ref_rgb: np.ndarray) -> np.ndarray:
    """Histogram matching по каждому RGB-каналу.

    Args:
        rgb: HxWx3 source.
        ref_rgb: HxWx3 reference (Photos).

    Returns:
        HxWx3 float32 0..1.
    """
    src = rgb.astype(np.float32)
    ref = ref_rgb.astype(np.float32)
    if src.max() > 1.5:
        src = src / 255.0
    if ref.max() > 1.5:
        ref = ref / 255.0
    out = np.empty_like(src)
    for c in range(3):
        out[..., c] = histogram_match_channel(src[..., c], ref[..., c])
    return np.clip(out, 0.0, 1.0)


def ref_stats_from_images(rgbs: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Считает mean/std RGB и Lab по списку картинок Photos.

    Args:
        rgbs: Список HxWx3 в 0..1.

    Returns:
        Словарь с ключами mean_rgb, std_rgb, mean_lab, std_lab.
    """
    pixels = []
    lab_pixels = []
    for rgb in rgbs:
        arr = rgb.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        # прореживание, чтобы не раздувать память
        flat = arr.reshape(-1, 3)[::8]
        pixels.append(flat)
        lab_pixels.append(_rgb_to_lab(arr).reshape(-1, 3)[::8])
    pix = np.concatenate(pixels, axis=0)
    lab = np.concatenate(lab_pixels, axis=0)
    return {
        "mean_rgb": pix.mean(axis=0).astype(np.float32),
        "std_rgb": pix.std(axis=0).astype(np.float32),
        "mean_lab": lab.mean(axis=0).astype(np.float32),
        "std_lab": lab.std(axis=0).astype(np.float32),
    }


@lru_cache(maxsize=4)
def load_photos_ref_stats(path: str | None = None) -> dict[str, np.ndarray]:
    """Читает кэш μ/σ Photos (RGB + Lab) для color-match в train.

    Args:
        path: JSON-файл; ``None`` → ``outputs/eda/deep/photos_ref_stats.json``.

    Returns:
        Словарь mean_rgb / std_rgb / mean_lab / std_lab как float32 (3,).

    Raises:
        ValueError: Если в JSON нет нужных ключей и нет fallback.
    """
    stats_path = Path(path) if path else DEFAULT_PHOTOS_REF_STATS
    raw: dict
    if stats_path.is_file():
        raw = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        raw = dict(_FALLBACK_PHOTOS_STATS)
    out: dict[str, np.ndarray] = {}
    for key in ("mean_rgb", "std_rgb", "mean_lab", "std_lab"):
        if key not in raw:
            raise ValueError(f"В {stats_path} нет ключа {key}")
        out[key] = np.asarray(raw[key], dtype=np.float32)
    return out


def apply_color_match_hwc(
    rgb_hwc: np.ndarray,
    method: str = "meanstd",
    *,
    stats: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Color-match HxWx3 RGB float к статистикам Photos.

    Args:
        rgb_hwc: HxWx3 в 0..1 (или uint8).
        method: ``meanstd`` или ``reinhard``.
        stats: Готовые статистики; None → load_photos_ref_stats().

    Returns:
        HxWx3 float32 0..1.
    """
    ref = stats if stats is not None else load_photos_ref_stats()
    method_l = method.lower().strip()
    if method_l in {"meanstd", "mean_std", "match_mean_std"}:
        return match_mean_std(rgb_hwc, ref["mean_rgb"], ref["std_rgb"])
    if method_l in {"reinhard", "reinhard_lab", "lab"}:
        return reinhard_lab_transfer(rgb_hwc, ref["mean_lab"], ref["std_lab"])
    raise ValueError(f"Неизвестный color-match method: {method}")


def apply_color_match_chw(
    rgb_chw: np.ndarray,
    method: str = "meanstd",
    *,
    stats: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """То же, что ``apply_color_match_hwc``, для layout CxHxW.

    Args:
        rgb_chw: (3, H, W) float.
        method: ``meanstd`` или ``reinhard``.
        stats: Опциональные статистики Photos.

    Returns:
        (3, H, W) float32.
    """
    hwc = np.transpose(rgb_chw, (1, 2, 0))
    out = apply_color_match_hwc(hwc, method=method, stats=stats)
    return np.transpose(out, (2, 0, 1)).astype(np.float32)
