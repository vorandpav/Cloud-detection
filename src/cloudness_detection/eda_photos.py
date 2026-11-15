"""EDA по датасету Photos (СПУТНИКС): карта, время, цвет, прокси-белизна."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from cloudness_detection.kml_meta import iter_photo_meta, lat_band
from cloudness_detection.seed import DEFAULT_SEED, set_global_seed

ROOT = Path(__file__).resolve().parents[2]
PHOTOS_DIR = ROOT / "Photos"
OUT_DIR = ROOT / "outputs" / "eda" / "photos"
COLOR_SAMPLE = 400


def color_stats(jpg_path: Path) -> dict[str, float]:
    """Средние R/G/B, яркость и доля ярких пикселей на уменьшенном кадре.

    Args:
        jpg_path: Путь к JPEG.

    Returns:
        Словарь со статистиками в диапазоне 0..1 для средних каналов
        и 0..1 для white_frac.
    """
    with Image.open(jpg_path) as img:
        rgb = img.convert("RGB")
        rgb = rgb.resize((160, 120), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return {
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "mean_brightness": float(brightness.mean()),
        "white_frac": float((brightness >= 0.78).mean()),
        "sat_proxy": float(np.mean(np.max(arr, axis=2) - np.min(arr, axis=2))),
    }


def build_catalog() -> pd.DataFrame:
    """Собирает таблицу метаданных (+ цвет на подвыборке).

    Returns:
        DataFrame по всем кадрам с KML. Цветовые колонки заполнены
        для случайной подвыборки размера ``COLOR_SAMPLE``.
    """
    set_global_seed(DEFAULT_SEED)
    rng = np.random.default_rng(DEFAULT_SEED)
    metas = iter_photo_meta(PHOTOS_DIR)
    rows = []
    for meta in metas:
        rows.append(
            {
                "stem": meta.stem,
                "board_id": meta.board_id,
                "captured_at": meta.captured_at,
                "year": meta.captured_at.year,
                "month": meta.captured_at.month,
                "center_lon": meta.center_lon,
                "center_lat": meta.center_lat,
                "lat_band": lat_band(meta.center_lat),
                "jpg_path": str(meta.jpg_path),
            }
        )
    df = pd.DataFrame(rows)

    sample_idx = rng.choice(len(df), size=min(COLOR_SAMPLE, len(df)), replace=False)
    color_cols = [
        "mean_r",
        "mean_g",
        "mean_b",
        "mean_brightness",
        "white_frac",
        "sat_proxy",
    ]
    for col in color_cols:
        df[col] = np.nan

    print(f"Цветовая статистика на {len(sample_idx)} кадрах…")
    for i, idx in enumerate(sample_idx, start=1):
        stats = color_stats(Path(df.at[idx, "jpg_path"]))
        for key, value in stats.items():
            df.at[idx, key] = value
        if i % 50 == 0:
            print(f"  {i}/{len(sample_idx)}")
    return df


def plot_map(df: pd.DataFrame, path: Path) -> None:
    """Точки центров кадров на lon/lat."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sc = ax.scatter(
        df["center_lon"],
        df["center_lat"],
        c=df["month"],
        cmap="twilight",
        s=8,
        alpha=0.55,
        linewidths=0,
    )
    ax.set_xlabel("Долгота")
    ax.set_ylabel("Широта")
    ax.set_title("Photos: центры кадров (цвет = месяц)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="месяц")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_month_board(df: pd.DataFrame, path: Path) -> None:
    """Сколько кадров по месяцам и топ бортов."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    month_counts = df["month"].value_counts().sort_index()
    axes[0].bar(month_counts.index.astype(str), month_counts.values, color="#3b6ea5")
    axes[0].set_title("Кадров по месяцам")
    axes[0].set_xlabel("месяц")
    axes[0].set_ylabel("число кадров")

    top = df["board_id"].value_counts().head(12)
    axes[1].barh(top.index[::-1], top.values[::-1], color="#5b8c5a")
    axes[1].set_title("Топ-12 бортов по числу кадров")
    axes[1].set_xlabel("число кадров")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_color(df: pd.DataFrame, path: Path) -> None:
    """RGB-средние и прокси-белизна на цветовой подвыборке."""
    sub = df.dropna(subset=["mean_r", "mean_g", "mean_b", "white_frac"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(sub["mean_r"], sub["mean_b"], c=sub["mean_g"], cmap="viridis", s=18, alpha=0.7)
    axes[0].set_xlabel("mean R")
    axes[0].set_ylabel("mean B")
    axes[0].set_title("Цветовое облако кадров (цвет точки = mean G)")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", alpha=0.4)

    axes[1].hist(sub["white_frac"], bins=20, color="#8a6d3b", edgecolor="white")
    axes[1].set_title("Прокси «белизна» (доля ярких пикселей)")
    axes[1].set_xlabel("white_frac")
    axes[1].set_ylabel("число кадров в выборке")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_lat_white(df: pd.DataFrame, path: Path) -> None:
    """Связь широты и белизны — намёк на снег/полярные сцены."""
    sub = df.dropna(subset=["white_frac"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(sub["center_lat"], sub["white_frac"], s=16, alpha=0.65, c="#4c78a8")
    ax.set_xlabel("Широта центра")
    ax.set_ylabel("white_frac")
    ax.set_title("Широта vs прокси-белизна (осторожно: снег ≠ облако)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(df: pd.DataFrame, path: Path) -> None:
    """Короткий markdown-summary для презентации."""
    color_n = int(df["mean_r"].notna().sum())
    lines = [
        "# EDA: Photos (СПУТНИКС)",
        "",
        f"- Кадров с валидным KML: **{len(df)}**",
        f"- Уникальных бортов: **{df['board_id'].nunique()}**",
        f"- Диапазон дат: **{df['captured_at'].min()} → {df['captured_at'].max()}**",
        f"- Широта: **{df['center_lat'].min():.1f}…{df['center_lat'].max():.1f}**",
        f"- Долгота: **{df['center_lon'].min():.1f}…{df['center_lon'].max():.1f}**",
        f"- Цветовая подвыборка (сид {DEFAULT_SEED}): **{color_n}** кадров",
        "",
        "## Широтные корзины",
        "",
        df["lat_band"].value_counts().to_string(),
        "",
        "## Заметки для кейса",
        "",
        "- Меток облаков нет: любые «% облачности» ниже — только после модели или ручной разметки.",
        "- Прокси-белизна путает снег, пустыню, sunglint и облака — это feature для стратификации, не GT.",
        "- Датасет глобальный: domain shift к Landsat/Sentinel неизбежен, но RGB-патчи 38-Cloud ближе, чем полный MSI CloudSEN12.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Строит графики и summary в ``outputs/eda/photos``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Собираю каталог…")
    df = build_catalog()
    df.to_csv(OUT_DIR / "photos_catalog.csv", index=False)

    plot_map(df, OUT_DIR / "map_centers.png")
    plot_month_board(df, OUT_DIR / "month_and_boards.png")
    plot_color(df, OUT_DIR / "color_and_white.png")
    plot_lat_white(df, OUT_DIR / "lat_vs_white.png")
    write_summary(df, OUT_DIR / "summary.md")

    meta = {
        "n_frames": int(len(df)),
        "n_boards": int(df["board_id"].nunique()),
        "color_sample": int(df["mean_r"].notna().sum()),
        "seed": DEFAULT_SEED,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Готово → {OUT_DIR}")


if __name__ == "__main__":
    main()
