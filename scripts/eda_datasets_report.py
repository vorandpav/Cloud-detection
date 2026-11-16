"""Воспроизводимый EDA по доступным датасетам.

Обходит Photos, 38-Cloud, SPARCS (sending), HRC_WHU, GF1, S2 thumbs;
сохраняет sample PNG, гистограммы cloud fraction и summary.json в
``outputs/eda/datasets/``. Сид фиксирован.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cloudness_detection.image_io import load_raster01
from cloudness_detection.seed import DEFAULT_SEED, set_global_seed

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "eda" / "datasets"
SEED = DEFAULT_SEED
SAMPLE_N_STATS = 400  # макс. файлов на датасет для гистограмм


def ensure_dir(path: Path) -> Path:
    """Создаёт папку, если её нет.

    Args:
        path: Целевая директория.

    Returns:
        Тот же path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def dir_size_mb(path: Path) -> float | None:
    """Считает размер директории в мегабайтах.

    Args:
        path: Корень обхода.

    Returns:
        Размер в MB или None, если пути нет.
    """
    if not path.exists():
        return None
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return round(total / (1024 * 1024), 1)


def save_rgb_png(rgb: np.ndarray, path: Path) -> None:
    """Пишет RGB float/uint8 массив в PNG.

    Args:
        rgb: HxWx3 в 0..1 или 0..255.
        path: Куда сохранить.
    """
    if rgb.dtype != np.uint8:
        arr = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = rgb
    Image.fromarray(arr).save(path)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Накладывает бинарную маску красным полупрозрачным слоем.

    Args:
        rgb: HxWx3 в 0..1.
        mask: HxW bool / 0..1.
        alpha: Доля красного слоя.

    Returns:
        HxWx3 float 0..1.
    """
    out = np.clip(rgb.astype(np.float32), 0.0, 1.0).copy()
    cloud = mask.astype(bool)
    if cloud.shape[:2] != out.shape[:2]:
        raise ValueError(f"Размер маски {cloud.shape} != RGB {out.shape}")
    red = np.zeros_like(out)
    red[..., 0] = 1.0
    out[cloud] = (1.0 - alpha) * out[cloud] + alpha * red[cloud]
    return out


def hist_cloud_frac(values: list[float], title: str, out_path: Path) -> None:
    """Рисует гистограмму доли облаков.

    Args:
        values: Доли 0..1 по файлам/патчам.
        title: Заголовок графика.
        out_path: PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(values, bins=20, range=(0.0, 1.0), color="#4C78A8", edgecolor="white")
    ax.set_xlabel("Cloud fraction")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def pick_indices(n: int, k: int, rng: random.Random) -> list[int]:
    """Выбирает до k индексов без повторов.

    Args:
        n: Длина списка.
        k: Сколько взять.
        rng: random.Random с сидом.

    Returns:
        Отсортированный список индексов.
    """
    if n <= 0:
        return []
    k = min(k, n)
    return sorted(rng.sample(range(n), k))


def eda_photos(out_dir: Path, rng: random.Random) -> dict:
    """EDA целевых Photos + выборки labeling/photos_40.

    Args:
        out_dir: ``outputs/eda/datasets/photos``.
        rng: Генератор случайности.

    Returns:
        Словарь статистик для summary.json.
    """
    ensure_dir(out_dir)
    photos_dir = ROOT / "Photos"
    label_dir = ROOT / "labeling" / "photos_40" / "images"
    jpgs = sorted(photos_dir.glob("*.jpg"))
    label_imgs = sorted(label_dir.glob("*.jpg")) if label_dir.exists() else []

    widths: list[int] = []
    heights: list[int] = []
    sample_idx = pick_indices(len(jpgs), 3, rng)
    for i, idx in enumerate(sample_idx, start=1):
        with Image.open(jpgs[idx]) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
            widths.append(img.width)
            heights.append(img.height)
        save_rgb_png(rgb, out_dir / f"sample_{i}.png")

    # ещё размеры с небольшой подвыборки
    size_idx = pick_indices(len(jpgs), min(SAMPLE_N_STATS, len(jpgs)), rng)
    for idx in size_idx:
        with Image.open(jpgs[idx]) as img:
            widths.append(img.width)
            heights.append(img.height)

    if label_imgs:
        with Image.open(label_imgs[0]) as img:
            save_rgb_png(
                np.asarray(img.convert("RGB"), dtype=np.uint8),
                out_dir / "sample_labeling_01.png",
            )

    # простой прокси «белизны» на подвыборке (масок нет)
    white_fracs: list[float] = []
    for idx in pick_indices(len(jpgs), min(120, len(jpgs)), rng):
        with Image.open(jpgs[idx]) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        bright = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        white_fracs.append(float((bright >= 0.78).mean()))
    if white_fracs:
        hist_cloud_frac(
            white_fracs,
            "Photos: white_frac proxy (bright>=0.78), n=%d" % len(white_fracs),
            out_dir / "white_frac_hist.png",
        )

    return {
        "name": "photos",
        "path": str(photos_dir.relative_to(ROOT)).replace("\\", "/"),
        "n_jpg": len(jpgs),
        "n_labeling_40": len(label_imgs),
        "size_mb": dir_size_mb(photos_dir),
        "width_mean": float(np.mean(widths)) if widths else None,
        "height_mean": float(np.mean(heights)) if heights else None,
        "white_frac_mean": float(np.mean(white_fracs)) if white_fracs else None,
        "notes": "Нет пиксельных масок; white_frac — грубый прокси облачности.",
        "ok": True,
    }


def _38cloud_band(split_dir: Path, band: str, patch_name: str) -> Path:
    """Путь к каналу/маске 38-Cloud.

    Args:
        split_dir: Train/test директория.
        band: red/green/blue/gt.
        patch_name: Имя патча без префикса.

    Returns:
        Путь к TIF.
    """
    prefix = "gt" if band == "gt" else band
    if "training" in split_dir.name:
        folder = "train_gt" if band == "gt" else f"train_{band}"
    else:
        folder = "test_gt" if band == "gt" else f"test_{band}"
    return split_dir / folder / f"{prefix}_{patch_name}.TIF"


def eda_38cloud(out_dir: Path, rng: random.Random) -> dict:
    """EDA 38-Cloud (RGB + GT).

    Args:
        out_dir: Выходная папка.
        rng: Генератор.

    Returns:
        Статистики.
    """
    ensure_dir(out_dir)
    data_dir = ROOT / "data" / "38-cloud"
    train_dir = data_dir / "38-Cloud_training"
    nonempty = data_dir / "training_patches_38-cloud_nonempty.csv"
    if not train_dir.exists():
        return {"name": "38-cloud", "ok": False, "error": "нет data/38-cloud"}

    import pandas as pd

    if nonempty.exists():
        names = pd.read_csv(nonempty)["name"].tolist()
    else:
        names = pd.read_csv(train_dir / "training_patches_38-Cloud.csv")["name"].tolist()

    n_gt = len(list((train_dir / "train_gt").glob("*.TIF"))) if (train_dir / "train_gt").exists() else 0
    cloud_fracs: list[float] = []
    samples_saved = 0
    chosen = rng.sample(names, k=min(SAMPLE_N_STATS, len(names)))

    for name in chosen:
        red_p = _38cloud_band(train_dir, "red", name)
        green_p = _38cloud_band(train_dir, "green", name)
        blue_p = _38cloud_band(train_dir, "blue", name)
        gt_p = _38cloud_band(train_dir, "gt", name)
        if not all(p.exists() for p in (red_p, green_p, blue_p, gt_p)):
            continue
        red = load_raster01(red_p)
        if float((red > 0.02).mean()) < 0.2:
            continue
        green = load_raster01(green_p)
        blue = load_raster01(blue_p)
        gt = load_raster01(gt_p, percentile_stretch=False)
        cloud = gt >= 0.5
        cloud_fracs.append(float(cloud.mean()))
        if samples_saved < 3:
            rgb = np.stack([red, green, blue], axis=-1)
            samples_saved += 1
            save_rgb_png(rgb, out_dir / f"sample_{samples_saved}.png")
            save_rgb_png(
                overlay_mask(rgb, cloud),
                out_dir / f"sample_{samples_saved}_overlay.png",
            )

    if cloud_fracs:
        hist_cloud_frac(
            cloud_fracs,
            f"38-Cloud train cloud fraction (n={len(cloud_fracs)})",
            out_dir / "cloud_fraction_hist.png",
        )

    return {
        "name": "38-cloud",
        "path": "data/38-cloud",
        "n_train_gt": n_gt,
        "n_stats": len(cloud_fracs),
        "cloud_frac_mean": float(np.mean(cloud_fracs)) if cloud_fracs else None,
        "cloud_frac_median": float(np.median(cloud_fracs)) if cloud_fracs else None,
        "size_mb": dir_size_mb(data_dir),
        "ok": True,
    }


def eda_sparcs(out_dir: Path, rng: random.Random) -> dict:
    """EDA SPARCS из data/sending (photo.png + mask.png).

    Args:
        out_dir: Выход.
        rng: Генератор.

    Returns:
        Статистики.
    """
    ensure_dir(out_dir)
    send_dir = ROOT / "data" / "sending"
    if not send_dir.exists():
        return {"name": "sparcs", "ok": False, "error": "нет data/sending"}

    photos = sorted(send_dir.glob("*_photo.png"))
    masks = {p.name.replace("_mask.png", ""): p for p in send_dir.glob("*_mask.png")}
    cloud_fracs: list[float] = []
    samples_saved = 0

    for idx in pick_indices(len(photos), len(photos), rng):
        photo_path = photos[idx]
        stem = photo_path.name.replace("_photo.png", "")
        mask_path = masks.get(stem)
        with Image.open(photo_path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        if mask_path is None:
            continue
        with Image.open(mask_path) as mimg:
            mask_arr = np.asarray(mimg)
        # USGS SPARCS: 5 = cloud; binary/simplified маски — {0,1}
        unique = np.unique(mask_arr)
        if set(unique.tolist()) <= {0, 1}:
            cloud = mask_arr >= 1
        else:
            cloud = mask_arr == 5
        cloud_fracs.append(float(cloud.mean()))
        if samples_saved < 3:
            samples_saved += 1
            save_rgb_png(rgb, out_dir / f"sample_{samples_saved}.png")
            save_rgb_png(
                overlay_mask(rgb, cloud),
                out_dir / f"sample_{samples_saved}_overlay.png",
            )

    if cloud_fracs:
        hist_cloud_frac(
            cloud_fracs,
            f"SPARCS cloud fraction (n={len(cloud_fracs)})",
            out_dir / "cloud_fraction_hist.png",
        )

    return {
        "name": "sparcs",
        "path": "data/sending",
        "n_scenes": len(photos),
        "n_masks": len(masks),
        "n_data_tif": len(list(send_dir.glob("*_data.tif"))),
        "cloud_frac_mean": float(np.mean(cloud_fracs)) if cloud_fracs else None,
        "cloud_frac_median": float(np.median(cloud_fracs)) if cloud_fracs else None,
        "size_mb": dir_size_mb(send_dir),
        "notes": "RGB из *_photo.png; *_data.tif — 10 каналов, Pillow не читает.",
        "ok": True,
    }


def eda_mmseg_style(
    name: str,
    root: Path,
    out_dir: Path,
    rng: random.Random,
    splits: tuple[str, ...] = ("train", "test", "val"),
) -> dict:
    """EDA датасетов вида img_dir/ann_dir (HRC, GF1).

    Args:
        name: Короткое имя папки вывода.
        root: Корень датасета.
        out_dir: Куда писать артефакты.
        rng: Генератор.
        splits: Какие split-папки смотреть.

    Returns:
        Статистики.
    """
    ensure_dir(out_dir)
    if not root.exists():
        return {"name": name, "ok": False, "error": f"нет {root}"}

    img_dir = root / "img_dir"
    ann_dir = root / "ann_dir"
    pairs: list[tuple[Path, Path]] = []
    split_counts: dict[str, int] = {}
    for split in splits:
        idir = img_dir / split
        adir = ann_dir / split
        if not idir.exists():
            continue
        files = sorted(
            [p for p in idir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".tif", ".tiff"}]
        )
        split_counts[split] = len(files)
        for img_path in files:
            mask_path = adir / img_path.name
            if mask_path.exists():
                pairs.append((img_path, mask_path))
            else:
                # иногда расширение маски другое
                alt = adir / (img_path.stem + ".png")
                if alt.exists():
                    pairs.append((img_path, alt))

    cloud_fracs: list[float] = []
    samples_saved = 0
    sample_pairs = [pairs[i] for i in pick_indices(len(pairs), min(SAMPLE_N_STATS, len(pairs)), rng)]

    for img_path, mask_path in sample_pairs:
        with Image.open(img_path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(mask_path) as mimg:
            mask_arr = np.asarray(mimg)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
        cloud = mask_arr > 0
        cloud_fracs.append(float(cloud.mean()))
        if samples_saved < 3:
            samples_saved += 1
            save_rgb_png(rgb, out_dir / f"sample_{samples_saved}.png")
            save_rgb_png(
                overlay_mask(rgb, cloud),
                out_dir / f"sample_{samples_saved}_overlay.png",
            )

    if cloud_fracs:
        hist_cloud_frac(
            cloud_fracs,
            f"{name} cloud fraction (n={len(cloud_fracs)})",
            out_dir / "cloud_fraction_hist.png",
        )

    # размер кадра
    w = h = None
    if pairs:
        with Image.open(pairs[0][0]) as img:
            w, h = img.size

    return {
        "name": name,
        "path": str(root.relative_to(ROOT)).replace("\\", "/"),
        "split_counts": split_counts,
        "n_pairs": len(pairs),
        "image_size": [w, h] if w else None,
        "cloud_frac_mean": float(np.mean(cloud_fracs)) if cloud_fracs else None,
        "cloud_frac_median": float(np.median(cloud_fracs)) if cloud_fracs else None,
        "size_mb": dir_size_mb(root),
        "ok": True,
    }


def s2_rgb_from_subscene(sub: np.ndarray) -> np.ndarray:
    """Собирает RGB (B4,B3,B2) из 13-канального S2 subscene и тянет 2–98%.

    Args:
        sub: HxWx13 float TOA reflectance.

    Returns:
        HxWx3 float 0..1 после percentile stretch по каждому каналу.
    """
    # Порядок каталога: B01..B12 с B8A — RGB = B04, B03, B02 → индексы 3,2,1.
    bands = [sub[..., 3], sub[..., 2], sub[..., 1]]
    out = np.empty((*sub.shape[:2], 3), dtype=np.float32)
    for i, ch in enumerate(bands):
        finite = ch[np.isfinite(ch)]
        if finite.size == 0:
            out[..., i] = 0.0
            continue
        lo, hi = np.percentile(finite, [2.0, 98.0])
        if hi <= lo:
            out[..., i] = 0.0
        else:
            out[..., i] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return out


def eda_s2_catalogue(out_dir: Path, rng: random.Random) -> dict:
    """EDA Sentinel-2 Cloud Mask Catalogue: RGB + overlay CLOUD-маски.

    Берёт сцены с ненулевой облачностью, чтобы на слайдах был виден cloud + mask.

    Args:
        out_dir: ``outputs/eda/datasets/s2_catalogue``.
        rng: Генератор.

    Returns:
        Статистики.
    """
    ensure_dir(out_dir)
    root = ROOT / "data" / "s2_catalogue"
    masks_dir = root / "masks"
    subs_dir = root / "subscenes"
    if not masks_dir.exists() or not subs_dir.exists():
        return {"name": "s2_catalogue", "ok": False, "error": "нет masks/subscenes"}

    mask_files = sorted(masks_dir.glob("*.npy"))
    cloud_fracs: list[float] = []
    cloudy_stems: list[tuple[str, float]] = []

    for mask_path in mask_files:
        mask = np.load(mask_path)
        # CLEAR / CLOUD / SHADOW → канал 1 = cloud
        cloud = mask[..., 1] if mask.ndim == 3 else mask > 0
        frac = float(np.asarray(cloud, dtype=np.float32).mean())
        cloud_fracs.append(frac)
        if 0.12 <= frac <= 0.88:
            cloudy_stems.append((mask_path.stem, frac))

    if not cloudy_stems:
        # запасной путь: любые ненулевые
        cloudy_stems = [
            (mask_files[i].stem, cloud_fracs[i])
            for i in range(len(mask_files))
            if cloud_fracs[i] > 0.05
        ]

    sample_pool = cloudy_stems if cloudy_stems else [(p.stem, 0.0) for p in mask_files]
    # ближе к 0.45 — и облака, и земля видны на overlay
    ranked = sorted(sample_pool, key=lambda x: abs(x[1] - 0.45))
    pick_from = ranked[: max(12, min(40, len(ranked)))]
    chosen = rng.sample(pick_from, k=min(3, len(pick_from)))
    chosen = sorted(chosen, key=lambda x: x[0])

    for i, (stem, frac) in enumerate(chosen, start=1):
        sub_path = subs_dir / f"{stem}.npy"
        mask_path = masks_dir / f"{stem}.npy"
        if not sub_path.exists():
            raise FileNotFoundError(f"Нет subscene для {stem}")
        sub = np.load(sub_path)
        mask = np.load(mask_path)
        cloud = mask[..., 1] if mask.ndim == 3 else mask > 0
        rgb = s2_rgb_from_subscene(sub)
        save_rgb_png(rgb, out_dir / f"sample_{i}.png")
        save_rgb_png(overlay_mask(rgb, cloud), out_dir / f"sample_{i}_overlay.png")
        (out_dir / f"sample_{i}_meta.txt").write_text(
            f"stem={stem}\ncloud_frac={frac:.4f}\n",
            encoding="utf-8",
        )

    if cloud_fracs:
        hist_cloud_frac(
            cloud_fracs,
            f"S2 Catalogue cloud fraction (n={len(cloud_fracs)})",
            out_dir / "cloud_fraction_hist.png",
        )

    almost_empty = float(np.mean([f < 0.05 for f in cloud_fracs])) if cloud_fracs else None
    almost_full = float(np.mean([f > 0.95 for f in cloud_fracs])) if cloud_fracs else None

    return {
        "name": "s2_catalogue",
        "path": "data/s2_catalogue",
        "n_masks": len(mask_files),
        "n_subscenes": len(list(subs_dir.glob("*.npy"))),
        "cloud_frac_mean": float(np.mean(cloud_fracs)) if cloud_fracs else None,
        "cloud_frac_median": float(np.median(cloud_fracs)) if cloud_fracs else None,
        "almost_empty_frac": almost_empty,
        "almost_full_frac": almost_full,
        "sample_stems": [s for s, _ in chosen],
        "size_mb": dir_size_mb(root),
        "notes": "RGB=B4,B3,B2 + percentile 2–98; overlay = CLOUD channel.",
        "ok": True,
    }


def eda_s2_thumbs(out_dir: Path, rng: random.Random) -> dict:
    """EDA превью Sentinel-2 Cloud Mask Catalogue (без масок).

    Args:
        out_dir: Выход.
        rng: Генератор.

    Returns:
        Статистики.
    """
    ensure_dir(out_dir)
    root = ROOT / "data" / "s2_catalogue_thumbs"
    thumbs_dir = root / "thumbnails"
    if not thumbs_dir.exists():
        # иногда файлы прямо в корне
        thumbs_dir = root
    thumbs = sorted(thumbs_dir.glob("*.png"))
    if not thumbs:
        return {"name": "s2_catalogue_thumbs", "ok": False, "error": "нет png"}

    for i, idx in enumerate(pick_indices(len(thumbs), 3, rng), start=1):
        with Image.open(thumbs[idx]) as img:
            save_rgb_png(
                np.asarray(img.convert("RGB"), dtype=np.uint8),
                out_dir / f"sample_{i}.png",
            )

    widths: list[int] = []
    heights: list[int] = []
    for idx in pick_indices(len(thumbs), min(100, len(thumbs)), rng):
        with Image.open(thumbs[idx]) as img:
            widths.append(img.width)
            heights.append(img.height)

    return {
        "name": "s2_catalogue_thumbs",
        "path": "data/s2_catalogue_thumbs",
        "n_thumbs": len(thumbs),
        "width_mean": float(np.mean(widths)) if widths else None,
        "height_mean": float(np.mean(heights)) if heights else None,
        "size_mb": dir_size_mb(root),
        "notes": "Только RGB-превью, масок в этом срезе нет.",
        "ok": True,
    }


def main() -> None:
    """Запускает EDA по всем доступным датасетам и пишет summary.json."""
    set_global_seed(SEED)
    rng = random.Random(SEED)
    ensure_dir(OUT_ROOT)

    results: list[dict] = []
    errors: list[str] = []

    steps = [
        ("photos", lambda: eda_photos(OUT_ROOT / "photos", rng)),
        ("38-cloud", lambda: eda_38cloud(OUT_ROOT / "38-cloud", rng)),
        ("sparcs", lambda: eda_sparcs(OUT_ROOT / "sparcs", rng)),
        (
            "hrc_whu",
            lambda: eda_mmseg_style(
                "hrc_whu", ROOT / "data" / "hrc_whu", OUT_ROOT / "hrc_whu", rng, ("train", "test")
            ),
        ),
        (
            "gf12ms_whu_gf1",
            lambda: eda_mmseg_style(
                "gf12ms_whu_gf1",
                ROOT / "data" / "gf12ms_whu_gf1",
                OUT_ROOT / "gf12ms_whu_gf1",
                rng,
                ("train", "val"),
            ),
        ),
        ("s2_catalogue", lambda: eda_s2_catalogue(OUT_ROOT / "s2_catalogue", rng)),
        ("s2_catalogue_thumbs", lambda: eda_s2_thumbs(OUT_ROOT / "s2_catalogue_thumbs", rng)),
    ]

    for label, fn in steps:
        print(f"[eda] {label} ...")
        try:
            info = fn()
            results.append(info)
            if not info.get("ok", False):
                errors.append(f"{label}: {info.get('error', 'fail')}")
                print(f"  WARN {info}")
            else:
                print(f"  OK { {k: info[k] for k in info if k in ('n_jpg','n_scenes','n_pairs','n_thumbs','n_stats','cloud_frac_mean','size_mb')} }")
        except Exception as exc:  # noqa: BLE001 — отчёт по каждому сету
            msg = f"{label}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            results.append({"name": label, "ok": False, "error": str(exc)})
            print(f"  FAIL {msg}")

    summary = {
        "seed": SEED,
        "datasets": results,
        "errors": errors,
    }
    summary_path = OUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    if errors:
        print("Errors:", errors)


if __name__ == "__main__":
    main()
