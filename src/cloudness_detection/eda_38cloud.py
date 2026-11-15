"""EDA по 38-Cloud: доля облаков, цвет RGB, почему датасет подходит."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cloudness_detection.image_io import load_raster01
from cloudness_detection.seed import DEFAULT_SEED, set_global_seed

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "38-cloud"
TRAIN_DIR = DATA_DIR / "38-Cloud_training"
OUT_DIR = ROOT / "outputs" / "eda" / "38cloud"
SAMPLE_N = 800


def band_path(split_dir: Path, band: str, patch_name: str) -> Path:
    """Собирает путь к каналу или маске патча.

    Args:
        split_dir: ``38-Cloud_training`` или ``38-Cloud_test``.
        band: ``red`` / ``green`` / ``blue`` / ``gt``.
        patch_name: Имя без префикса, как в CSV
            (``patch_1_1_by_1_LC08_...``).

    Returns:
        Путь к ``.TIF``.
    """
    prefix = "gt" if band == "gt" else band
    folder = f"train_{band}" if "training" in split_dir.name else f"test_{band}"
    if band == "gt":
        folder = "train_gt"
    return split_dir / folder / f"{prefix}_{patch_name}.TIF"


def is_informative(red: np.ndarray, min_nonzero: float = 0.2) -> bool:
    """Отсекает почти чёрные патчи с краёв сцены Landsat.

    Args:
        red: Канал red в 0..1.
        min_nonzero: Минимальная доля пикселей ярче малого порога.

    Returns:
        True, если патч стоит брать в статистику.
    """
    return float((red > 0.02).mean()) >= min_nonzero


def sample_training_stats(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Считает cloud fraction и RGB-средние на случайных train-патчах.

    Args:
        n: Сколько патчей взять.
        rng: Генератор с зафиксированным сидом.

    Returns:
        DataFrame со статистиками по патчам.
    """
    nonempty_csv = DATA_DIR / "training_patches_38-cloud_nonempty.csv"
    patches_csv = TRAIN_DIR / "training_patches_38-Cloud.csv"
    if nonempty_csv.exists():
        names = pd.read_csv(nonempty_csv)["name"].tolist()
    else:
        names = pd.read_csv(patches_csv)["name"].tolist()

    chosen = rng.choice(names, size=min(n, len(names)), replace=False)
    rows: list[dict[str, float | str]] = []
    skipped_black = 0

    for i, name in enumerate(chosen, start=1):
        red_p = band_path(TRAIN_DIR, "red", name)
        green_p = band_path(TRAIN_DIR, "green", name)
        blue_p = band_path(TRAIN_DIR, "blue", name)
        gt_p = band_path(TRAIN_DIR, "gt", name)
        if not (red_p.exists() and green_p.exists() and blue_p.exists() and gt_p.exists()):
            raise FileNotFoundError(f"Нет файлов для патча {name}")

        red = load_raster01(red_p, percentile_stretch=True)
        if not is_informative(red):
            skipped_black += 1
            continue
        green = load_raster01(green_p, percentile_stretch=True)
        blue = load_raster01(blue_p, percentile_stretch=True)
        gt = load_raster01(gt_p, percentile_stretch=False)
        cloud = gt >= 0.5
        brightness = 0.299 * red + 0.587 * green + 0.114 * blue
        rows.append(
            {
                "name": name,
                "cloud_frac": float(cloud.mean()),
                "mean_r": float(red.mean()),
                "mean_g": float(green.mean()),
                "mean_b": float(blue.mean()),
                "mean_brightness": float(brightness.mean()),
                "white_frac": float((brightness >= 0.78).mean()),
            }
        )
        if i % 100 == 0:
            print(f"  {i}/{len(chosen)}")

    print(f"Пропущено почти чёрных патчей: {skipped_black}")
    return pd.DataFrame(rows)


def plot_cloud_hist(df: pd.DataFrame, path: Path) -> None:
    """Гистограмма доли облаков и порог кейса 75%."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["cloud_frac"] * 100.0, bins=25, color="#4c78a8", edgecolor="white")
    ax.axvline(75, color="#c44e52", linestyle="--", linewidth=2, label="порог кейса 75%")
    ax.set_xlabel("% облачности на патче")
    ax.set_ylabel("число патчей")
    ax.set_title("38-Cloud train: распределение cloud fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reject_balance(df: pd.DataFrame, path: Path) -> None:
    """Сколько патчей ушло бы в delete при пороге 75%."""
    reject = (df["cloud_frac"] > 0.75).value_counts().rename({False: "keep ≤75%", True: "delete >75%"})
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(reject.index.astype(str), reject.values, color=["#5b8c5a", "#c44e52"])
    ax.set_title("38-Cloud: решение keep/delete при пороге 75%")
    ax.set_ylabel("число патчей в выборке")
    for i, v in enumerate(reject.values):
        ax.text(i, v + max(reject.values) * 0.01, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rgb_cloud(df: pd.DataFrame, path: Path) -> None:
    """Цвет патча vs доля облаков."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        df["mean_r"],
        df["mean_b"],
        c=df["cloud_frac"] * 100.0,
        cmap="coolwarm",
        s=18,
        alpha=0.75,
    )
    ax.set_xlabel("mean R")
    ax.set_ylabel("mean B")
    ax.set_title("38-Cloud: цвет патча (цвет точки = % облаков)")
    fig.colorbar(sc, ax=ax, label="% облаков")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(df: pd.DataFrame, path: Path) -> None:
    """Короткий текст для слайдов."""
    reject_rate = float((df["cloud_frac"] > 0.75).mean())
    lines = [
        "# EDA: 38-Cloud",
        "",
        f"- Патчей в выборке EDA (сид {DEFAULT_SEED}): **{len(df)}**",
        f"- Размер патча: **384×384**, каналы RGB (+ NIR в датасете, для обучения Photos не нужен)",
        f"- Средний cloud_frac: **{df['cloud_frac'].mean()*100:.1f}%**",
        f"- Медиана cloud_frac: **{df['cloud_frac'].median()*100:.1f}%**",
        f"- Доля патчей с cloud_frac > 75%: **{reject_rate*100:.1f}%**",
        "",
        "## Почему берём 38-Cloud",
        "",
        "- Есть **ручные** пиксельные маски облаков (не псевдо-метки GEE).",
        "- Можно учить на **RGB only** — ближе к камерам СПУТНИКС, чем Sentinel MSI.",
        "- Патчи компактные, удобно гонять на RTX 4060 / Kaggle T4.",
        "- Это стандартный бенчмарк статьи Cloud-Net — удобно цитировать.",
        "",
        "## Ограничения (честно на слайд)",
        "",
        "- Landsat 8, не кубсат: другая оптика, калибровка, ~30 м vs ~200 м у Photos.",
        "- Поэтому после pretrain нужен микросет Photos (40 кадров) для калибровки порога.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Строит EDA 38-Cloud в ``outputs/eda/38cloud``."""
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(
            f"Нет {TRAIN_DIR}. Положи 38-Cloud в data/38-cloud/ как в README."
        )

    set_global_seed(DEFAULT_SEED)
    rng = np.random.default_rng(DEFAULT_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Считаю статистики по train-патчам…")
    df = sample_training_stats(SAMPLE_N, rng)
    df.to_csv(OUT_DIR / "sample_stats.csv", index=False)

    plot_cloud_hist(df, OUT_DIR / "cloud_fraction_hist.png")
    plot_reject_balance(df, OUT_DIR / "reject75_balance.png")
    plot_rgb_cloud(df, OUT_DIR / "rgb_vs_cloud.png")
    write_summary(df, OUT_DIR / "summary.md")

    meta = {
        "n_sampled": int(len(df)),
        "mean_cloud_frac": float(df["cloud_frac"].mean()),
        "reject75_rate": float((df["cloud_frac"] > 0.75).mean()),
        "seed": DEFAULT_SEED,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Готово → {OUT_DIR}")


if __name__ == "__main__":
    main()
