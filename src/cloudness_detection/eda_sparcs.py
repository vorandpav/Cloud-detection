"""EDA по SPARCS: классы масок, % облаков, снег vs облако, RGB-превью."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from cloudness_detection.seed import DEFAULT_SEED, set_global_seed

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "eda" / "sparcs"


def resolve_sparcs_dir() -> Path:
    """Ищет папку SPARCS: ``data/sparcs`` или ``data/sending``.

    Returns:
        Путь к папке с ``*_mask.png``.

    Raises:
        FileNotFoundError: Если нигде нет масок.
    """
    candidates = [ROOT / "data" / "sparcs", ROOT / "data" / "sending"]
    for path in candidates:
        if path.is_dir() and any(path.glob("*_mask.png")):
            return path
    raise FileNotFoundError(
        "Не нашёл SPARCS. Положи файлы в data/sparcs/ (или data/sending/)."
    )

# USGS SPARCS mask codes
CLASS_NAMES = {
    0: "shadow",
    1: "shadow_water",
    2: "water",
    3: "snow",
    4: "land",
    5: "cloud",
    6: "flooded",
}


def list_scenes(data_dir: Path) -> list[str]:
    """Возвращает список stem сцен по файлам ``*_mask.png``.

    Args:
        data_dir: Папка SPARCS.
    """
    masks = sorted(data_dir.glob("*_mask.png"))
    return [p.name.replace("_mask.png", "") for p in masks]


def cloud_and_snow_frac(mask: np.ndarray) -> tuple[float, float]:
    """Доля облаков и снега на маске.

    Args:
        mask: HxW с кодами SPARCS.
    """
    n = mask.size
    return float((mask == 5).sum() / n), float((mask == 3).sum() / n)


def rgb_stats(photo_path: Path) -> dict[str, float]:
    """Средние R/G/B и яркость по превью ``*_photo.png``.

    Args:
        photo_path: Путь к RGB-превью сцены.
    """
    with Image.open(photo_path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return {
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "mean_brightness": float(brightness.mean()),
    }


def build_table(data_dir: Path) -> tuple[pd.DataFrame, dict[int, int]]:
    """Собирает per-scene таблицу и глобальные счётчики классов.

    Args:
        data_dir: Папка SPARCS.
    """
    scenes = list_scenes(data_dir)
    if not scenes:
        raise FileNotFoundError(f"Нет *_mask.png в {data_dir}")

    rows: list[dict] = []
    class_counts = {k: 0 for k in CLASS_NAMES}

    for stem in scenes:
        mask_path = data_dir / f"{stem}_mask.png"
        photo_path = data_dir / f"{stem}_photo.png"
        with Image.open(mask_path) as img:
            mask = np.asarray(img)
            # Иногда PNG маска в RGB — берём один канал.
            if mask.ndim == 3:
                mask = mask[:, :, 0]

        for code in CLASS_NAMES:
            class_counts[code] += int((mask == code).sum())

        cloud_f, snow_f = cloud_and_snow_frac(mask)
        row = {
            "scene": stem,
            "h": int(mask.shape[0]),
            "w": int(mask.shape[1]),
            "cloud_frac": cloud_f,
            "snow_frac": snow_f,
            "reject75": int(cloud_f > 0.75),
        }
        if photo_path.exists():
            row.update(rgb_stats(photo_path))
        rows.append(row)

    return pd.DataFrame(rows), class_counts


def plot_class_bars(counts: dict[int, int], path: Path) -> None:
    """Глобальный баланс классов по пикселям."""
    labels = [CLASS_NAMES[k] for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]
    total = sum(values) or 1
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, [v / total for v in values], color="#4c78a8")
    ax.set_ylabel("доля пикселей")
    ax.set_title("SPARCS: баланс классов (все сцены)")
    ax.tick_params(axis="x", rotation=25)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v/total:.1%}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cloud_snow(df: pd.DataFrame, path: Path) -> None:
    """Cloud fraction, порог 75%, и snow vs cloud."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(df["cloud_frac"] * 100.0, bins=16, color="#4c78a8", edgecolor="white")
    axes[0].axvline(75, color="#c44e52", linestyle="--", label="порог 75%")
    axes[0].set_title("SPARCS: % облачности по сценам")
    axes[0].set_xlabel("% облаков")
    axes[0].legend()

    axes[1].scatter(df["snow_frac"] * 100.0, df["cloud_frac"] * 100.0, s=40, alpha=0.8, c="#f58518")
    axes[1].set_xlabel("% снега")
    axes[1].set_ylabel("% облаков")
    axes[1].set_title("Снег vs облака (важный fail-mode)")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_examples(data_dir: Path, df: pd.DataFrame, path: Path) -> None:
    """Сетка: ясно / облачно / снег — photo + маска."""
    set_global_seed(DEFAULT_SEED)
    picks: list[tuple[str, str]] = []

    clear = df.sort_values("cloud_frac").head(1)
    cloudy = df.sort_values("cloud_frac", ascending=False).head(1)
    snowy = df.sort_values("snow_frac", ascending=False).head(1)
    picks.append((clear.iloc[0]["scene"], "мало облаков"))
    picks.append((cloudy.iloc[0]["scene"], "много облаков"))
    picks.append((snowy.iloc[0]["scene"], "много снега"))

    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    for row_i, (stem, title) in enumerate(picks):
        photo = np.asarray(Image.open(data_dir / f"{stem}_photo.png").convert("RGB"))
        mask = np.asarray(Image.open(data_dir / f"{stem}_mask.png"))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        axes[row_i, 0].imshow(photo)
        axes[row_i, 0].set_title(f"{title}\n{stem[:28]}…")
        axes[row_i, 0].axis("off")
        axes[row_i, 1].imshow(mask, cmap="tab10", vmin=0, vmax=6)
        axes[row_i, 1].set_title("mask (5=cloud, 3=snow)")
        axes[row_i, 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_summary(df: pd.DataFrame, counts: dict[int, int], path: Path) -> None:
    """Текст для презентации: зачем SPARCS, чем отличается от 38-Cloud."""
    total_px = sum(counts.values()) or 1
    lines = [
        "# EDA: SPARCS",
        "",
        f"- Сцен: **{len(df)}**",
        f"- Типичный размер: **{df['h'].mode().iloc[0]}×{df['w'].mode().iloc[0]}**",
        f"- Средний cloud_frac: **{df['cloud_frac'].mean()*100:.1f}%**",
        f"- Сцен с cloud_frac > 75%: **{int(df['reject75'].sum())} / {len(df)}**",
        f"- Средний snow_frac: **{df['snow_frac'].mean()*100:.1f}%**",
        f"- Пикселей cloud: **{counts[5]/total_px:.1%}**, snow: **{counts[3]/total_px:.1%}**",
        "",
        "## Зачем SPARCS (и почему не 95-Cloud)",
        "",
        "- Это **другой** набор: не расширение 38-Cloud, а валидационный Landsat с "
        "**snow / shadow / water / cloud** — удобно показывать fail-modes.",
        "- 95-Cloud — по сути больше тех же патчей, что 38-Cloud; для кейса мало новой информации.",
        "- SPARCS оставляем как **hold-out / error analysis**, основной train — 38-Cloud RGB.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Строит EDA SPARCS в ``outputs/eda/sparcs``."""
    data_dir = resolve_sparcs_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Читаю SPARCS из {data_dir}…")
    df, counts = build_table(data_dir)
    df.to_csv(OUT_DIR / "scenes.csv", index=False)

    plot_class_bars(counts, OUT_DIR / "class_balance.png")
    plot_cloud_snow(df, OUT_DIR / "cloud_and_snow.png")
    plot_examples(data_dir, df, OUT_DIR / "examples_grid.png")
    write_summary(df, counts, OUT_DIR / "summary.md")

    meta = {
        "data_dir": str(data_dir),
        "n_scenes": int(len(df)),
        "mean_cloud_frac": float(df["cloud_frac"].mean()),
        "mean_snow_frac": float(df["snow_frac"].mean()),
        "reject75_scenes": int(df["reject75"].sum()),
        "seed": DEFAULT_SEED,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Готово → {OUT_DIR}")


if __name__ == "__main__":
    main()
