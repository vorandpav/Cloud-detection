"""Сравнение цветовых распределений Photos vs 38-Cloud для слайда «почему этот датасет»."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cloudness_detection.seed import DEFAULT_SEED

ROOT = Path(__file__).resolve().parents[2]
PHOTOS_STATS = ROOT / "outputs" / "eda" / "photos" / "photos_catalog.csv"
CLOUD_STATS = ROOT / "outputs" / "eda" / "38cloud" / "sample_stats.csv"
OUT_DIR = ROOT / "outputs" / "eda" / "compare"


def main() -> None:
    """Рисует наложенные гистограммы и пишет вывод для презентации."""
    if not PHOTOS_STATS.exists():
        raise FileNotFoundError(f"Сначала запусти eda-photos. Нет файла {PHOTOS_STATS}")
    if not CLOUD_STATS.exists():
        raise FileNotFoundError(f"Сначала запусти eda-38cloud. Нет файла {CLOUD_STATS}")

    photos = pd.read_csv(PHOTOS_STATS)
    cloud = pd.read_csv(CLOUD_STATS)
    photos_c = photos.dropna(subset=["mean_r", "mean_g", "mean_b", "mean_brightness"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    pairs = [
        ("mean_r", "mean R"),
        ("mean_g", "mean G"),
        ("mean_b", "mean B"),
        ("mean_brightness", "яркость"),
    ]
    for ax, (col, title) in zip(axes.ravel(), pairs):
        ax.hist(
            photos_c[col],
            bins=25,
            density=True,
            alpha=0.55,
            color="#4c78a8",
            label="Photos (СПУТНИКС)",
        )
        ax.hist(
            cloud[col],
            bins=25,
            density=True,
            alpha=0.55,
            color="#f58518",
            label="38-Cloud RGB",
        )
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Сравнение цветовых распределений (сид {DEFAULT_SEED})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "color_hist_overlay.png", dpi=160)
    plt.close(fig)

    # Простая дистанция между средними цветами — для текста, не «доказательство».
    photo_mean = photos_c[["mean_r", "mean_g", "mean_b"]].mean().to_numpy()
    cloud_mean = cloud[["mean_r", "mean_g", "mean_b"]].mean().to_numpy()
    l2 = float(np.linalg.norm(photo_mean - cloud_mean))

    lines = [
        "# Почему 38-Cloud, а не «первый попавшийся»",
        "",
        "## Факты сравнения",
        "",
        f"- Photos: цветовая подвыборка **{len(photos_c)}** кадров",
        f"- 38-Cloud: EDA-выборка **{len(cloud)}** RGB-патчей",
        f"- Средний RGB Photos: ({photo_mean[0]:.3f}, {photo_mean[1]:.3f}, {photo_mean[2]:.3f})",
        f"- Средний RGB 38-Cloud: ({cloud_mean[0]:.3f}, {cloud_mean[1]:.3f}, {cloud_mean[2]:.3f})",
        f"- L2 между средними RGB: **{l2:.3f}** (порядок величины для слайда, не метрика качества)",
        "",
        "## Почему выбрали 38-Cloud",
        "",
        "1. **RGB-режим**: у Photos только RGB; у 38-Cloud можно отбросить NIR и учить так же.",
        "2. **Ручные маски**: честный GT, а не QA60 / s2cloudless.",
        "3. **Сопоставимый визуальный контент**: земля/вода/облака на оптике; распределения яркости пересекаются (см. график).",
        "4. **Удобный размер**: патчи 384², не 50+ GB как KappaSet / полный CloudSEN12.",
        "",
        "## Почему не взяли другое",
        "",
        "- **CloudSEN12 / KappaSet** — Sentinel-2 MSI, другой домен и тяжелый download.",
        "- **Авторазметка Photos через GEE** — другой сенсор (~200 м, без калибровки), шумные псевдо-метки.",
        "- **Только эвристика по белизне** — путает снег/пустыню/блики; годится как baseline, не как GT.",
        "",
        "## Что всё равно остаётся",
        "",
        "Domain gap (Landsat 30 м vs кубсат ~200 м) никуда не делся → калибруем порог 75% на 40 ручных Photos.",
        "",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Готово → {OUT_DIR}")


if __name__ == "__main__":
    main()
