"""Глубокий сравнительный EDA: Photos vs публичные сеты.

Артефакты → ``outputs/eda/deep/``.
Сид фиксирован (DEFAULT_SEED).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

from cloudness_detection.color_norm import (
    histogram_match_rgb,
    match_mean_std,
    percentile_stretch,
    ref_stats_from_images,
    reinhard_lab_transfer,
)
from cloudness_detection.image_io import load_raster01
from cloudness_detection.kml_meta import iter_photo_meta
from cloudness_detection.seed import DEFAULT_SEED, set_global_seed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "eda" / "deep"
SEED = DEFAULT_SEED

# объёмы подвыборок: баланс скорости и стабильности тестов
N_PHOTOS_COLOR = 240
N_PUBLIC_COLOR = 180
N_CLOUD_FRAC = 400
NEAR_EMPTY = 0.05
NEAR_FULL = 0.95


def ensure(path: Path) -> Path:
    """Создаёт директорию."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def to01_rgb(arr: np.ndarray) -> np.ndarray:
    """Приводит RGB к float 0..1."""
    a = arr.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def resize_rgb(rgb: np.ndarray, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Уменьшает RGB для быстрых гистограмм/KS."""
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img = img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def channel_means(rgb: np.ndarray) -> tuple[float, float, float]:
    """Средние R/G/B по кадру."""
    return float(rgb[..., 0].mean()), float(rgb[..., 1].mean()), float(rgb[..., 2].mean())


def hsv_hue_mean(rgb: np.ndarray) -> float:
    """Средний hue (0..1) через Pillow-совместимый HSV."""
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).convert("HSV")
    h = np.asarray(img, dtype=np.float32)[..., 0] / 255.0
    return float(h.mean())


def pick(n: int, k: int, rng: random.Random) -> list[int]:
    """До k индексов без повторов."""
    if n <= 0:
        return []
    return sorted(rng.sample(range(n), min(k, n)))


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_photos_samples(rng: random.Random) -> dict:
    """Цвет + geo из Photos (KML). Масок нет — cloud_frac = white_frac proxy."""
    photos_dir = ROOT / "Photos"
    jpgs = sorted(photos_dir.glob("*.jpg"))
    rows = []
    for idx in pick(len(jpgs), N_PHOTOS_COLOR, rng):
        path = jpgs[idx]
        with Image.open(path) as img:
            rgb = to01_rgb(np.asarray(img.convert("RGB")))
        small = resize_rgb(rgb)
        mr, mg, mb = channel_means(small)
        bright = 0.299 * small[..., 0] + 0.587 * small[..., 1] + 0.114 * small[..., 2]
        rows.append(
            {
                "dataset": "photos",
                "stem": path.stem,
                "mean_r": mr,
                "mean_g": mg,
                "mean_b": mb,
                "std_r": float(small[..., 0].std()),
                "std_g": float(small[..., 1].std()),
                "std_b": float(small[..., 2].std()),
                "hue_mean": hsv_hue_mean(small),
                "cloud_frac": float((bright >= 0.78).mean()),
                "rgb_small": small,
            }
        )

    geos = []
    for meta in iter_photo_meta(photos_dir):
        geos.append(
            {
                "stem": meta.stem,
                "lon": meta.center_lon,
                "lat": meta.center_lat,
                "year": meta.captured_at.year,
                "month": meta.captured_at.month,
                "board_id": meta.board_id,
            }
        )
    return {"color_rows": rows, "geo": geos, "n_total": len(jpgs)}


def load_38cloud(rng: random.Random) -> dict:
    """38-Cloud train RGB + GT + scene id из имени патча."""
    data_dir = ROOT / "data" / "38-cloud"
    train_dir = data_dir / "38-Cloud_training"
    nonempty = data_dir / "training_patches_38-cloud_nonempty.csv"
    if nonempty.exists():
        names = pd.read_csv(nonempty)["name"].tolist()
    else:
        names = pd.read_csv(train_dir / "training_patches_38-Cloud.csv")["name"].tolist()

    def band_path(band: str, name: str) -> Path:
        folder = "train_gt" if band == "gt" else f"train_{band}"
        prefix = "gt" if band == "gt" else band
        return train_dir / folder / f"{prefix}_{name}.TIF"

    # scene id: у патчей часто префикс вроде patch_xxx_... или LC8...
    scene_re = re.compile(r"(LC8\d+|L\d+|scene[_-]?\w+)", re.I)

    cloud_fracs: list[float] = []
    rows = []
    scene_ids: list[str] = []
    chosen = [names[i] for i in pick(len(names), min(N_CLOUD_FRAC, len(names)), rng)]

    for name in chosen:
        rp, gp, bp, gp_mask = (
            band_path("red", name),
            band_path("green", name),
            band_path("blue", name),
            band_path("gt", name),
        )
        if not all(p.exists() for p in (rp, gp, bp, gp_mask)):
            continue
        red = load_raster01(rp)
        if float((red > 0.02).mean()) < 0.2:
            continue
        green = load_raster01(gp)
        blue = load_raster01(bp)
        gt = load_raster01(gp_mask, percentile_stretch=False)
        cloud = float((gt >= 0.5).mean())
        cloud_fracs.append(cloud)
        m = scene_re.search(name)
        scene_ids.append(m.group(1) if m else name.split("_")[0])
        if len(rows) < N_PUBLIC_COLOR:
            rgb = resize_rgb(np.stack([red, green, blue], axis=-1))
            mr, mg, mb = channel_means(rgb)
            rows.append(
                {
                    "dataset": "38-cloud",
                    "stem": name,
                    "mean_r": mr,
                    "mean_g": mg,
                    "mean_b": mb,
                    "std_r": float(rgb[..., 0].std()),
                    "std_g": float(rgb[..., 1].std()),
                    "std_b": float(rgb[..., 2].std()),
                    "hue_mean": hsv_hue_mean(rgb),
                    "cloud_frac": cloud,
                    "rgb_small": rgb,
                }
            )

    return {
        "color_rows": rows,
        "cloud_fracs": cloud_fracs,
        "scene_ids": scene_ids,
        "n_total": len(names),
    }


def load_sparcs(rng: random.Random) -> dict:
    """SPARCS RGB + mask + MTL lat/lon/date."""
    send = ROOT / "data" / "sending"
    photos = sorted(send.glob("*_photo.png"))
    masks = {p.name.replace("_mask.png", ""): p for p in send.glob("*_mask.png")}
    mtls = {p.name.replace("_mtl.txt", ""): p for p in send.glob("*_mtl.txt")}

    cloud_fracs: list[float] = []
    rows = []
    geos = []

    for idx in pick(len(photos), len(photos), rng):
        photo_path = photos[idx]
        stem = photo_path.name.replace("_photo.png", "")
        mask_path = masks.get(stem)
        with Image.open(photo_path) as img:
            rgb_full = to01_rgb(np.asarray(img.convert("RGB")))
        if mask_path is None:
            continue
        with Image.open(mask_path) as mimg:
            mask_arr = np.asarray(mimg)
        if set(np.unique(mask_arr).tolist()) <= {0, 1}:
            cloud = mask_arr >= 1
        else:
            # USGS SPARCS: 5 = cloud (не 2=water)
            cloud = mask_arr == 5
        frac = float(cloud.mean())
        cloud_fracs.append(frac)
        rgb = resize_rgb(rgb_full)
        mr, mg, mb = channel_means(rgb)
        rows.append(
            {
                "dataset": "sparcs",
                "stem": stem,
                "mean_r": mr,
                "mean_g": mg,
                "mean_b": mb,
                "std_r": float(rgb[..., 0].std()),
                "std_g": float(rgb[..., 1].std()),
                "std_b": float(rgb[..., 2].std()),
                "hue_mean": hsv_hue_mean(rgb),
                "cloud_frac": frac,
                "rgb_small": rgb,
            }
        )

        # MTL: префикс сцены до _NN_
        scene = stem.rsplit("_", 1)[0]
        mtl = mtls.get(scene)
        if mtl and mtl.exists():
            text = mtl.read_text(encoding="utf-8", errors="ignore")
            vals: dict[str, float | str] = {}
            for line in text.splitlines():
                line = line.strip()
                if "CORNER_UL_LAT_PRODUCT" in line:
                    vals["ul_lat"] = float(line.split("=")[1].strip())
                elif "CORNER_LR_LAT_PRODUCT" in line:
                    vals["lr_lat"] = float(line.split("=")[1].strip())
                elif "CORNER_UL_LON_PRODUCT" in line:
                    vals["ul_lon"] = float(line.split("=")[1].strip())
                elif "CORNER_LR_LON_PRODUCT" in line:
                    vals["lr_lon"] = float(line.split("=")[1].strip())
                elif line.startswith("DATE_ACQUIRED"):
                    vals["date"] = line.split("=")[1].strip()
            if {"ul_lat", "lr_lat", "ul_lon", "lr_lon"} <= vals.keys():
                geos.append(
                    {
                        "stem": stem,
                        "lat": 0.5 * (float(vals["ul_lat"]) + float(vals["lr_lat"])),
                        "lon": 0.5 * (float(vals["ul_lon"]) + float(vals["lr_lon"])),
                        "date": vals.get("date"),
                        "scene": scene,
                    }
                )

    return {
        "color_rows": rows,
        "cloud_fracs": cloud_fracs,
        "geo": geos,
        "n_total": len(photos),
    }


def load_mmseg(name: str, root: Path, rng: random.Random, splits: tuple[str, ...]) -> dict:
    """HRC / GF1 в раскладке img_dir / ann_dir."""
    pairs: list[tuple[Path, Path]] = []
    for split in splits:
        idir = root / "img_dir" / split
        adir = root / "ann_dir" / split
        if not idir.exists():
            continue
        for img_path in sorted(idir.iterdir()):
            if img_path.suffix.lower() not in {".png", ".jpg", ".tif", ".tiff"}:
                continue
            mask_path = adir / img_path.name
            if not mask_path.exists():
                mask_path = adir / (img_path.stem + ".png")
            if mask_path.exists():
                pairs.append((img_path, mask_path))

    cloud_fracs: list[float] = []
    rows = []
    for idx in pick(len(pairs), min(N_CLOUD_FRAC, len(pairs)), rng):
        img_path, mask_path = pairs[idx]
        with Image.open(img_path) as img:
            rgb_full = to01_rgb(np.asarray(img.convert("RGB")))
        with Image.open(mask_path) as mimg:
            mask_arr = np.asarray(mimg)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
        frac = float((mask_arr > 0).mean())
        cloud_fracs.append(frac)
        if len(rows) < N_PUBLIC_COLOR:
            rgb = resize_rgb(rgb_full)
            mr, mg, mb = channel_means(rgb)
            rows.append(
                {
                    "dataset": name,
                    "stem": img_path.stem,
                    "mean_r": mr,
                    "mean_g": mg,
                    "mean_b": mb,
                    "std_r": float(rgb[..., 0].std()),
                    "std_g": float(rgb[..., 1].std()),
                    "std_b": float(rgb[..., 2].std()),
                    "hue_mean": hsv_hue_mean(rgb),
                    "cloud_frac": frac,
                    "rgb_small": rgb,
                }
            )

    return {
        "color_rows": rows,
        "cloud_fracs": cloud_fracs,
        "n_total": len(pairs),
        "geo_note": "метаданных lat/lon в pack нет",
    }


def s2_rgb(sub: np.ndarray) -> np.ndarray:
    """B4,B3,B2 + percentile stretch."""
    bands = np.stack([sub[..., 3], sub[..., 2], sub[..., 1]], axis=-1)
    return percentile_stretch(bands)


def load_s2(rng: random.Random) -> dict:
    """S2 catalogue masks + subscenes + classification_tags."""
    root = ROOT / "data" / "s2_catalogue"
    masks_dir = root / "masks"
    subs_dir = root / "subscenes"
    tags_path = root / "classification_tags.csv"
    mask_files = sorted(masks_dir.glob("*.npy"))

    cloud_fracs: list[float] = []
    stems_frac: list[tuple[str, float]] = []
    for mp in mask_files:
        mask = np.load(mp)
        frac = float(mask[..., 1].mean())
        cloud_fracs.append(frac)
        stems_frac.append((mp.stem, frac))

    # цвет: подвыборка сцен (полные npy тяжёлые)
    color_idx = pick(len(stems_frac), min(N_PUBLIC_COLOR, len(stems_frac)), rng)
    rows = []
    for i in color_idx:
        stem, frac = stems_frac[i]
        sub = np.load(subs_dir / f"{stem}.npy")
        rgb = resize_rgb(s2_rgb(sub))
        mr, mg, mb = channel_means(rgb)
        rows.append(
            {
                "dataset": "s2",
                "stem": stem,
                "mean_r": mr,
                "mean_g": mg,
                "mean_b": mb,
                "std_r": float(rgb[..., 0].std()),
                "std_g": float(rgb[..., 1].std()),
                "std_b": float(rgb[..., 2].std()),
                "hue_mean": hsv_hue_mean(rgb),
                "cloud_frac": frac,
                "rgb_small": rgb,
            }
        )

    tags = None
    if tags_path.exists():
        tags = pd.read_csv(tags_path)

    # MGRS tile / time из имени сцены Sentinel-2
    tile_re = re.compile(r"_T([0-9]{2}[A-Z]{3})_")
    date_re = re.compile(r"_(\d{8}T\d{6})_")
    tiles, dates = [], []
    for stem, _ in stems_frac:
        tm = tile_re.search(stem)
        dm = date_re.search(stem)
        if tm:
            tiles.append(tm.group(1))
        if dm:
            dates.append(dm.group(1)[:8])

    return {
        "color_rows": rows,
        "cloud_fracs": cloud_fracs,
        "n_total": len(mask_files),
        "tags": tags,
        "tiles": tiles,
        "dates": dates,
        "geo_note": "точных lat/lon нет; есть MGRS tile и дата в имени + tags CSV",
    }


# ---------------------------------------------------------------------------
# plots / stats
# ---------------------------------------------------------------------------


def cloud_summary(fracs: list[float] | np.ndarray, label: str) -> dict:
    """Сводка по cloud fraction."""
    arr = np.asarray(fracs, dtype=np.float64)
    if arr.size == 0:
        return {"dataset": label, "n": 0}
    return {
        "dataset": label,
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "near_empty_frac": float((arr < NEAR_EMPTY).mean()),
        "near_full_frac": float((arr > NEAR_FULL).mean()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def plot_cloud_hists(by_ds: dict[str, list[float]], out_path: Path) -> None:
    """Общая гистограмма cloud fraction по источникам."""
    n = len(by_ds)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, (name, vals) in zip(axes, by_ds.items()):
        ax.hist(vals, bins=20, range=(0, 1), color="#4C78A8", edgecolor="white")
        ax.set_title(f"{name} (n={len(vals)})")
        ax.set_xlim(0, 1)
        ax.set_xlabel("cloud frac")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Cloud fraction by source")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_rgb_box(color_df: pd.DataFrame, out_path: Path) -> None:
    """Boxplot mean R/G/B по датасетам."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title in zip(axes, ["mean_r", "mean_g", "mean_b"], ["R", "G", "B"]):
        data = [color_df.loc[color_df["dataset"] == ds, col].values for ds in color_df["dataset"].unique()]
        ax.boxplot(data, tick_labels=list(color_df["dataset"].unique()), showfliers=False)
        ax.set_title(f"mean {title}")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Per-image channel means")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_rgb_kde(color_df: pd.DataFrame, out_path: Path) -> None:
    """Наложенные hist mean brightness / R."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ds, sub in color_df.groupby("dataset"):
        bright = 0.299 * sub["mean_r"] + 0.587 * sub["mean_g"] + 0.114 * sub["mean_b"]
        axes[0].hist(bright, bins=25, density=True, alpha=0.45, label=ds)
        axes[1].hist(sub["mean_r"], bins=25, density=True, alpha=0.45, label=ds)
    axes[0].set_title("mean brightness")
    axes[1].set_title("mean R")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_stat_tests(color_df: pd.DataFrame, cloud_by_ds: dict[str, list[float]]) -> pd.DataFrame:
    """KS + Mann-Whitney: Photos vs каждый публичный сет."""
    photo = color_df[color_df["dataset"] == "photos"]
    rows = []
    publics = [d for d in color_df["dataset"].unique() if d != "photos"]
    metrics = ["mean_r", "mean_g", "mean_b", "hue_mean"]
    for ds in publics:
        other = color_df[color_df["dataset"] == ds]
        for col in metrics:
            a = photo[col].dropna().to_numpy()
            b = other[col].dropna().to_numpy()
            if len(a) < 5 or len(b) < 5:
                continue
            ks = stats.ks_2samp(a, b)
            mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append(
                {
                    "compare": f"photos_vs_{ds}",
                    "metric": col,
                    "ks_stat": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "mw_stat": float(mw.statistic),
                    "mw_pvalue": float(mw.pvalue),
                    "photos_mean": float(a.mean()),
                    "other_mean": float(b.mean()),
                    "mean_diff": float(a.mean() - b.mean()),
                }
            )
        # cloud frac
        a_c = np.asarray(cloud_by_ds.get("photos", []), dtype=np.float64)
        b_c = np.asarray(cloud_by_ds.get(ds, []), dtype=np.float64)
        if a_c.size >= 5 and b_c.size >= 5:
            ks = stats.ks_2samp(a_c, b_c)
            mw = stats.mannwhitneyu(a_c, b_c, alternative="two-sided")
            rows.append(
                {
                    "compare": f"photos_vs_{ds}",
                    "metric": "cloud_frac",
                    "ks_stat": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "mw_stat": float(mw.statistic),
                    "mw_pvalue": float(mw.pvalue),
                    "photos_mean": float(a_c.mean()),
                    "other_mean": float(b_c.mean()),
                    "mean_diff": float(a_c.mean() - b_c.mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_geo(photos_geo: list[dict], sparcs_geo: list[dict], out_path: Path) -> None:
    """Scatter lat-lon Photos + SPARCS."""
    fig, ax = plt.subplots(figsize=(9, 5))
    if photos_geo:
        ax.scatter(
            [g["lon"] for g in photos_geo],
            [g["lat"] for g in photos_geo],
            s=4,
            alpha=0.35,
            label=f"Photos (n={len(photos_geo)})",
            c="#4C78A8",
        )
    if sparcs_geo:
        ax.scatter(
            [g["lon"] for g in sparcs_geo],
            [g["lat"] for g in sparcs_geo],
            s=28,
            alpha=0.85,
            label=f"SPARCS (n={len(sparcs_geo)})",
            c="#F58518",
            marker="x",
        )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title("Geo coverage (where metadata exists)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    """Пишет PNG."""
    arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def color_norm_ablation(photo_rows: list[dict], public_bundles: dict[str, dict], out_dir: Path) -> dict:
    """До/после нормализации на 2–3 публичных кадрах."""
    ensure(out_dir)
    ref_rgbs = [r["rgb_small"] for r in photo_rows[:80]]
    ref = ref_stats_from_images(ref_rgbs)
    # один большой Photos mosaic как ref для hist match
    ref_mosaic = np.concatenate([r.reshape(-1, 3) for r in ref_rgbs], axis=0)
    # синтетический HxW для hist match — берём первые N пикселей в квадрат
    side = int(np.sqrt(min(len(ref_mosaic), 128 * 128)))
    ref_img = ref_mosaic[: side * side].reshape(side, side, 3)

    examples = []
    # берём по одному кадру из hrc / sparcs / 38-cloud
    for ds_key in ("hrc_whu", "sparcs", "38-cloud"):
        rows = public_bundles.get(ds_key, {}).get("color_rows", [])
        if not rows:
            continue
        # средний по яркости — типичный кадр
        rows_sorted = sorted(rows, key=lambda r: abs(r["mean_r"] - 0.4))
        src = rows_sorted[len(rows_sorted) // 2]
        rgb = src["rgb_small"]
        stretched = percentile_stretch(rgb)
        matched_ms = match_mean_std(rgb, ref["mean_rgb"], ref["std_rgb"])
        matched_rh = reinhard_lab_transfer(rgb, ref["mean_lab"], ref["std_lab"])
        matched_hm = histogram_match_rgb(rgb, ref_img)

        stem = f"{ds_key}_{src['stem'][:40]}"
        save_rgb(out_dir / f"{stem}_00_raw.png", rgb)
        save_rgb(out_dir / f"{stem}_01_percentile.png", stretched)
        save_rgb(out_dir / f"{stem}_02_meanstd.png", matched_ms)
        save_rgb(out_dir / f"{stem}_03_reinhard_lab.png", matched_rh)
        save_rgb(out_dir / f"{stem}_04_histmatch.png", matched_hm)

        # панель
        panel = np.concatenate([rgb, matched_ms, matched_rh], axis=1)
        save_rgb(out_dir / f"{stem}_panel.png", panel)
        examples.append(
            {
                "dataset": ds_key,
                "stem": src["stem"],
                "files": [
                    f"{stem}_00_raw.png",
                    f"{stem}_02_meanstd.png",
                    f"{stem}_03_reinhard_lab.png",
                    f"{stem}_panel.png",
                ],
            }
        )

    # референс Photos для глаза
    if photo_rows:
        save_rgb(out_dir / "photos_ref_sample.png", photo_rows[0]["rgb_small"])

    meta = {
        "ref_mean_rgb": ref["mean_rgb"].tolist(),
        "ref_std_rgb": ref["std_rgb"].tolist(),
        "examples": examples,
    }
    (out_dir / "ablation_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def distance_to_photos(color_df: pd.DataFrame) -> pd.DataFrame:
    """L2 между средними RGB датасета и Photos."""
    photo_mean = color_df.loc[color_df["dataset"] == "photos", ["mean_r", "mean_g", "mean_b"]].mean()
    rows = []
    for ds, sub in color_df.groupby("dataset"):
        m = sub[["mean_r", "mean_g", "mean_b"]].mean()
        l2 = float(np.linalg.norm(m.to_numpy() - photo_mean.to_numpy()))
        rows.append(
            {
                "dataset": ds,
                "mean_r": float(m["mean_r"]),
                "mean_g": float(m["mean_g"]),
                "mean_b": float(m["mean_b"]),
                "l2_to_photos": l2,
            }
        )
    return pd.DataFrame(rows).sort_values("l2_to_photos")


def main() -> None:
    """Запускает deep EDA и пишет CSV/PNG/JSON."""
    set_global_seed(SEED)
    rng = random.Random(SEED)
    ensure(OUT)
    ensure(OUT / "color_norm")

    print("[deep] photos ...")
    photos = load_photos_samples(rng)
    print("[deep] 38-cloud ...")
    c38 = load_38cloud(rng)
    print("[deep] sparcs ...")
    sparcs = load_sparcs(rng)
    print("[deep] hrc ...")
    hrc = load_mmseg("hrc_whu", ROOT / "data" / "hrc_whu", rng, ("train", "test"))
    print("[deep] gf1 ...")
    gf1 = load_mmseg("gf1", ROOT / "data" / "gf12ms_whu_gf1", rng, ("train", "val"))
    print("[deep] s2 ...")
    s2 = load_s2(rng)

    bundles = {
        "photos": photos,
        "38-cloud": c38,
        "sparcs": sparcs,
        "hrc_whu": hrc,
        "gf1": gf1,
        "s2": s2,
    }

    # cloud fractions: photos = white_frac proxy
    cloud_by_ds: dict[str, list[float]] = {
        "photos": [r["cloud_frac"] for r in photos["color_rows"]],
        "38-cloud": c38["cloud_fracs"],
        "sparcs": sparcs["cloud_fracs"],
        "hrc_whu": hrc["cloud_fracs"],
        "gf1": gf1["cloud_fracs"],
        "s2": s2["cloud_fracs"],
    }
    cloud_rows = [cloud_summary(v, k) for k, v in cloud_by_ds.items()]
    cloud_df = pd.DataFrame(cloud_rows)
    cloud_df.to_csv(OUT / "cloud_fraction_summary.csv", index=False)
    plot_cloud_hists(cloud_by_ds, OUT / "cloud_fraction_hists.png")

    # color table (без rgb_small в csv)
    all_color = []
    for bundle in bundles.values():
        all_color.extend(bundle.get("color_rows", []))
    color_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "rgb_small"} for r in all_color]
    )
    color_df.to_csv(OUT / "color_per_image.csv", index=False)
    plot_rgb_box(color_df, OUT / "rgb_mean_box.png")
    plot_rgb_kde(color_df, OUT / "rgb_brightness_hist.png")

    # channel mean/std summary
    ch_rows = []
    for ds, sub in color_df.groupby("dataset"):
        ch_rows.append(
            {
                "dataset": ds,
                "n": len(sub),
                "mean_r": float(sub["mean_r"].mean()),
                "mean_g": float(sub["mean_g"].mean()),
                "mean_b": float(sub["mean_b"].mean()),
                "std_r_across": float(sub["mean_r"].std()),
                "std_g_across": float(sub["mean_g"].std()),
                "std_b_across": float(sub["mean_b"].std()),
                "hue_mean": float(sub["hue_mean"].mean()),
            }
        )
    ch_df = pd.DataFrame(ch_rows)
    ch_df.to_csv(OUT / "channel_stats_summary.csv", index=False)

    dist_df = distance_to_photos(color_df)
    dist_df.to_csv(OUT / "distance_to_photos.csv", index=False)

    print("[deep] stat tests ...")
    tests = run_stat_tests(color_df, cloud_by_ds)
    tests.to_csv(OUT / "stat_tests_photos_vs_public.csv", index=False)

    # geo
    plot_geo(photos.get("geo", []), sparcs.get("geo", []), OUT / "geo_latlon.png")
    pd.DataFrame(photos.get("geo", [])).to_csv(OUT / "photos_geo.csv", index=False)
    pd.DataFrame(sparcs.get("geo", [])).to_csv(OUT / "sparcs_geo.csv", index=False)

    # S2 tiles / dates
    if s2.get("tiles"):
        tile_counts = pd.Series(s2["tiles"]).value_counts().head(30)
        tile_counts.to_csv(OUT / "s2_mgrs_tiles_top.csv", header=["count"])
    if s2.get("dates"):
        years = [d[:4] for d in s2["dates"]]
        pd.Series(years).value_counts().sort_index().to_csv(
            OUT / "s2_year_counts.csv", header=["count"]
        )
    if s2.get("tags") is not None:
        tags = s2["tags"]
        tags[["cloud_percent", "clear_percent", "shadow_percent"]].describe().to_csv(
            OUT / "s2_tags_describe.csv"
        )

    # 38-cloud scene ids
    if c38.get("scene_ids"):
        pd.Series(c38["scene_ids"]).value_counts().head(40).to_csv(
            OUT / "38cloud_scene_id_counts.csv", header=["count"]
        )

    print("[deep] color norm ablation ...")
    ablation = color_norm_ablation(photos["color_rows"], bundles, OUT / "color_norm")

    meta_notes = {
        "photos": "KML lat/lon + время в имени файла",
        "sparcs": "MTL CORNER_* + DATE_ACQUIRED",
        "s2": s2.get("geo_note"),
        "38-cloud": "scene/patch id в имени; lat/lon нет",
        "hrc_whu": hrc.get("geo_note"),
        "gf1": gf1.get("geo_note"),
    }

    summary = {
        "seed": SEED,
        "n_color": {k: len(v.get("color_rows", [])) for k, v in bundles.items()},
        "n_cloud_frac": {k: len(v) for k, v in cloud_by_ds.items()},
        "cloud_summary": cloud_rows,
        "distance_to_photos": dist_df.to_dict(orient="records"),
        "meta_notes": meta_notes,
        "color_norm": ablation,
        "stat_tests_n": len(tests),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {OUT}")
    print("L2 to Photos:")
    print(dist_df.to_string(index=False))


if __name__ == "__main__":
    main()
