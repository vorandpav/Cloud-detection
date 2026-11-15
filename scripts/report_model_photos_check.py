"""Визуальный отчёт: best.pt baseline на train/val миксе и Photos с CVAT-масками.

Шаги:
1. Распаковывает CVAT zip → ``labeling/photos_40/cvat_export/``.
2. Растеризует маски (RLE ``<mask>`` / polygon) → ``labeling/photos_40/masks/``.
3. Грузит ``best.pt``, рисует triplet-галереи train/val/photos_test.
4. Пишет ``REPORT.md``, ``metrics.json``, ``README.md``.

Пример::

    set PYTHONPATH=src
    .venv\\Scripts\\python.exe scripts/report_model_photos_check.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from cloudness_detection.data.common import load_rgb_u8, resize_pair
from cloudness_detection.data.mix import (
    build_datasets,
    build_one,
    collate_samples,
    resolve_project_root,
)
from cloudness_detection.metrics import dice_binary, iou_binary, mae_cloud_percent
from cloudness_detection.models.build import build_model
from cloudness_detection.seed import DEFAULT_SEED, set_global_seed

ANN_TAGS = frozenset({"polygon", "polyline", "mask", "box", "ellipse", "skeleton", "points"})
PRED_THRESHOLD = 0.5
OVERLAY_ALPHA = 0.45
OOM_RETRIES = 5
OOM_SLEEP_SEC = 20.0


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
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


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    """CxHxW float → HxWx3 float."""
    return np.transpose(np.clip(image, 0.0, 1.0), (1, 2, 0)).astype(np.float32)


def save_png_hwc(rgb: np.ndarray, path: Path) -> None:
    """Сохраняет HxWx3 float/uint8 в PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(rgb, 0.0, 1.0)
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def make_triplet(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Склеивает rgb | gt-overlay | pred-overlay по ширине.

    Args:
        rgb: HxWx3 float 0..1.
        gt: HxW бинарная.
        pred: HxW бинарная.

    Returns:
        HxW*3x3 float.
    """
    left = np.clip(rgb, 0.0, 1.0)
    mid = overlay_mask(left, gt)
    right = overlay_mask(left, pred)
    return np.concatenate([left, mid, right], axis=1)


def extract_cvat_zip(zip_path: Path, dest: Path) -> Path:
    """Распаковывает CVAT zip (overwrite) и возвращает путь к annotations.xml.

    Args:
        zip_path: Архив CVAT for images 1.1.
        dest: Папка ``cvat_export``.

    Returns:
        Path к ``annotations.xml``.

    Raises:
        FileNotFoundError: Нет zip или xml внутри.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"Нет CVAT zip: {zip_path}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    xml_path = dest / "annotations.xml"
    if not xml_path.is_file():
        found = list(dest.rglob("annotations.xml"))
        if not found:
            raise FileNotFoundError(f"annotations.xml не найден в {dest}")
        xml_path = found[0]
    return xml_path


def decode_cvat_rle(
    rle_str: str,
    left: int,
    top: int,
    mask_w: int,
    mask_h: int,
    image_w: int,
    image_h: int,
) -> np.ndarray:
    """Декодирует CVAT brush ``rle`` в бинарную маску полного кадра.

    RLE: чередование длин фона (0) и облака (1), row-major внутри bbox.

    Args:
        rle_str: Строка чисел через запятую.
        left: X левого края bbox.
        top: Y верхнего края bbox.
        mask_w: Ширина bbox.
        mask_h: Высота bbox.
        image_w: Ширина кадра из XML.
        image_h: Высота кадра из XML.

    Returns:
        HxW uint8 {0,1}.
    """
    counts = [int(x) for x in rle_str.replace(" ", "").split(",") if x]
    flat = np.zeros(mask_w * mask_h, dtype=np.uint8)
    val = 0
    idx = 0
    for count in counts:
        if count:
            end = min(idx + count, flat.size)
            if val == 1:
                flat[idx:end] = 1
            idx = end
        val = 1 - val
        if idx >= flat.size:
            break
    patch = flat.reshape((mask_h, mask_w))
    full = np.zeros((image_h, image_w), dtype=np.uint8)
    y1 = min(top + mask_h, image_h)
    x1 = min(left + mask_w, image_w)
    full[top:y1, left:x1] = patch[: y1 - top, : x1 - left]
    return full


def rasterize_polygon(points_str: str, image_w: int, image_h: int) -> np.ndarray:
    """Рисует polygon ``x1,y1;x2,y2;...`` в бинарную маску.

    Args:
        points_str: Атрибут points из CVAT.
        image_w: Ширина.
        image_h: Высота.

    Returns:
        HxW uint8 {0,1}.
    """
    pairs = []
    for chunk in points_str.strip().split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x_str, y_str = chunk.split(",")
        pairs.append((float(x_str), float(y_str)))
    mask_img = Image.new("L", (image_w, image_h), 0)
    if len(pairs) >= 3:
        ImageDraw.Draw(mask_img).polygon(pairs, outline=1, fill=1)
    return (np.asarray(mask_img) > 0).astype(np.uint8)


def image_has_annotations(image_el: ET.Element) -> bool:
    """True, если у ``<image>`` есть дочерние фигуры разметки."""
    return any(child.tag in ANN_TAGS for child in list(image_el))


def parse_cvat_masks(
    xml_path: Path,
    images_dir: Path,
    masks_dir: Path,
) -> list[dict]:
    """Парсит CVAT XML, сохраняет PNG-маски 0/255, возвращает список сэмплов.

    Args:
        xml_path: ``annotations.xml``.
        images_dir: Превью ``labeling/photos_40/images``.
        masks_dir: Куда писать PNG маски.

    Returns:
        Список словарей с путями, размерами и cloud_frac.
    """
    root = ET.parse(xml_path).getroot()
    masks_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []

    for image_el in root.findall("image"):
        if not image_has_annotations(image_el):
            continue
        name = image_el.get("name") or ""
        image_w = int(image_el.get("width", 0))
        image_h = int(image_el.get("height", 0))
        mask = np.zeros((image_h, image_w), dtype=np.uint8)

        for child in list(image_el):
            if child.tag == "mask":
                mask = np.maximum(
                    mask,
                    decode_cvat_rle(
                        child.get("rle", ""),
                        int(child.get("left", 0)),
                        int(child.get("top", 0)),
                        int(child.get("width", 0)),
                        int(child.get("height", 0)),
                        image_w,
                        image_h,
                    ),
                )
            elif child.tag == "polygon":
                mask = np.maximum(
                    mask,
                    rasterize_polygon(child.get("points", ""), image_w, image_h),
                )
            elif child.tag == "box":
                xtl = int(float(child.get("xtl", 0)))
                ytl = int(float(child.get("ytl", 0)))
                xbr = int(float(child.get("xbr", 0)))
                ybr = int(float(child.get("ybr", 0)))
                mask[ytl:ybr, xtl:xbr] = 1
            elif child.tag == "ellipse":
                cx = float(child.get("cx", 0))
                cy = float(child.get("cy", 0))
                rx = float(child.get("rx", 0))
                ry = float(child.get("ry", 0))
                ell = Image.new("L", (image_w, image_h), 0)
                ImageDraw.Draw(ell).ellipse(
                    [cx - rx, cy - ry, cx + rx, cy + ry],
                    outline=1,
                    fill=1,
                )
                mask = np.maximum(mask, (np.asarray(ell) > 0).astype(np.uint8))

        stem = Path(name).stem
        preview_path = images_dir / Path(name).name
        if not preview_path.is_file():
            matches = list(images_dir.glob(f"*{Path(name).name}"))
            if not matches:
                raise FileNotFoundError(f"Нет превью для {name} в {images_dir}")
            preview_path = matches[0]

        with Image.open(preview_path) as im:
            preview_size = im.size  # (W, H)
        if preview_size != (image_w, image_h):
            raise ValueError(
                f"XML size {(image_w, image_h)} != preview {preview_size} для {name}"
            )

        mask_u8 = (mask * 255).astype(np.uint8)
        mask_path = masks_dir / f"{stem}.png"
        Image.fromarray(mask_u8, mode="L").save(mask_path)

        samples.append(
            {
                "name": Path(name).name,
                "stem": stem,
                "preview_path": str(preview_path),
                "mask_path": str(mask_path),
                "xml_width": image_w,
                "xml_height": image_h,
                "preview_width": preview_size[0],
                "preview_height": preview_size[1],
                "cloud_frac_gt": float(mask.mean()),
                "resolution_used": "preview",
            }
        )
    samples.sort(key=lambda s: s["name"])
    return samples


def load_manifest_source_map(manifest_csv: Path) -> dict[str, str]:
    """stem → source_jpg из manifest.csv."""
    out: dict[str, str] = {}
    if not manifest_csv.is_file():
        return out
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = (row.get("stem") or "").strip()
            src = (row.get("source_jpg") or "").strip()
            preview = (row.get("preview_jpg") or "").strip()
            if stem and src:
                out[stem] = src
            # также по имени превью без расширения
            if preview and src:
                out[Path(preview).stem] = src
    return out


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Грузит best.pt → модель на device.

    Args:
        ckpt_path: Путь к чекпоинту.
        device: cuda/cpu.

    Returns:
        (model.eval(), checkpoint dict).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name") or ckpt.get("config", {}).get("model", "unet")
    enc = None
    cfg = ckpt.get("config") or {}
    if cfg.get("encoder_weights"):
        enc = cfg["encoder_weights"]
    elif "encoder_weights" not in cfg:
        enc = "imagenet"
    model = build_model(model_name, encoder_weights=enc)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    image_chw: np.ndarray,
    image_size: int,
    device: torch.device,
    threshold: float = PRED_THRESHOLD,
) -> np.ndarray:
    """Инференс одной картинки CxHxW → бинарная HxW на исходном размере.

    Ресайз к ``image_size``, sigmoid+thr, затем nearest обратно к HxW.
    При CUDA OOM ждёт и повторяет.

    Args:
        model: Модель.
        image_chw: RGB float CxHxW [0,1].
        image_size: Сторона квадрата как на train.
        device: Устройство.
        threshold: Порог.

    Returns:
        HxW float {0,1} на исходном разрешении.
    """
    _, h0, w0 = image_chw.shape
    dummy_mask = np.zeros((1, h0, w0), dtype=np.float32)
    image_r, _ = resize_pair(image_chw, dummy_mask, image_size)
    tensor = torch.from_numpy(image_r).unsqueeze(0)

    last_err: Exception | None = None
    for attempt in range(1, OOM_RETRIES + 1):
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            logits = model(tensor.to(device))
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            break
        except RuntimeError as exc:
            last_err = exc
            if "out of memory" not in str(exc).lower():
                raise
            print(
                f"CUDA OOM (попытка {attempt}/{OOM_RETRIES}), ждём {OOM_SLEEP_SEC:.0f}s…",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            time.sleep(OOM_SLEEP_SEC)
    else:
        raise RuntimeError(f"Не удалось инференс из-за OOM: {last_err}")

    pred_small = (prob >= threshold).astype(np.uint8) * 255
    pred_img = Image.fromarray(pred_small, mode="L").resize((w0, h0), Image.Resampling.NEAREST)
    return (np.asarray(pred_img) > 127).astype(np.float32)


def rebuild_val_subset(config: dict, project_root: Path) -> Subset:
    """Воспроизводит val=holdout как в ``train_mix.py`` (тот же seed).

    Args:
        config: ``config.json`` рана.
        project_root: Корень репо.

    Returns:
        Subset без аугментаций.
    """
    names = list(config["datasets"])
    image_size = int(config["image_size"])
    seed = int(config.get("seed", DEFAULT_SEED))
    max_items_raw = config.get("max_items", 0) or 0
    max_items = None if int(max_items_raw) <= 0 else int(max_items_raw)
    color_match = config.get("color_match_method") if config.get("color_match_photos") else None
    val_ratio = float(config.get("val_ratio", 0.15))

    mix = build_datasets(
        names,
        image_size=image_size,
        augment=False,
        max_items=max_items,
        seed=seed,
        project_root=project_root,
        color_match=color_match,
    )
    val_name = names[0]
    val_ds = build_one(
        val_name,
        image_size=image_size,
        augment=False,
        max_items=max_items,
        seed=seed + 1,
        project_root=project_root,
        split="test" if val_name == "hrc" else "train",
        color_match=color_match,
    )
    val_len = min(len(val_ds), max(4, int(len(mix) * val_ratio)))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(val_ds), generator=g)[:val_len].tolist()
    return Subset(val_ds, perm)


def pick_diverse_indices(cloud_fracs: list[float], k: int, seed: int) -> list[int]:
    """Берёт k индексов: низкая / средняя / высокая облачность.

    Args:
        cloud_fracs: Доли облаков по сэмплам.
        k: Сколько взять.
        seed: Сид для тай-брейков.

    Returns:
        Отсортированные индексы.
    """
    n = len(cloud_fracs)
    if n == 0:
        return []
    k = min(k, n)
    order = np.argsort(np.asarray(cloud_fracs))
    # равномерно по рангу облачности
    positions = np.linspace(0, n - 1, num=k)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    used: set[int] = set()
    for pos in positions:
        idx = int(order[int(round(pos))])
        if idx in used:
            # соседние свободные
            for delta in range(1, n):
                for cand in (idx - delta, idx + delta):
                    if 0 <= cand < n and int(order[cand]) not in used:
                        idx = int(order[cand])
                        break
                else:
                    continue
                break
        used.add(idx)
        chosen.append(idx)
    # если недобрали — добираем случайно
    while len(chosen) < k:
        extra = int(rng.integers(0, n))
        if extra not in used:
            used.add(extra)
            chosen.append(extra)
    return sorted(chosen)


def metrics_from_masks(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """IoU/Dice/%/reject@75 для одной пары масок."""
    p = pred.astype(bool)
    t = gt.astype(bool)
    pp = float(p.mean() * 100.0)
    tp = float(t.mean() * 100.0)
    return {
        "iou": iou_binary(p, t),
        "dice": dice_binary(p, t),
        "cloud_frac_pred_pct": pp,
        "cloud_frac_gt_pct": tp,
        "mae_pct": abs(pp - tp),
        "reject75_agree": float((pp > 75.0) == (tp > 75.0)),
    }


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Средние по списку per-image метрик."""
    if not rows:
        return {
            "n": 0.0,
            "iou": 0.0,
            "dice": 0.0,
            "mae_pct": 0.0,
            "reject75_acc": 0.0,
            "cloud_frac_gt_pct_mean": 0.0,
            "cloud_frac_pred_pct_mean": 0.0,
        }
    pred_pcts = np.asarray([r["cloud_frac_pred_pct"] for r in rows], dtype=np.float64)
    true_pcts = np.asarray([r["cloud_frac_gt_pct"] for r in rows], dtype=np.float64)
    return {
        "n": float(len(rows)),
        "iou": float(np.mean([r["iou"] for r in rows])),
        "dice": float(np.mean([r["dice"] for r in rows])),
        "mae_pct": mae_cloud_percent(pred_pcts, true_pcts),
        "reject75_acc": float(np.mean([r["reject75_agree"] for r in rows])),
        "cloud_frac_gt_pct_mean": float(true_pcts.mean()),
        "cloud_frac_pred_pct_mean": float(pred_pcts.mean()),
    }


def read_history_best(history_csv: Path) -> dict:
    """Читает history.csv и строку с best=1 и максимальным val_iou."""
    rows = []
    with history_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    best_rows = [r for r in rows if str(r.get("best", "")).strip() in {"1", "1.0"}]
    if not best_rows:
        best_rows = rows
    best = max(best_rows, key=lambda r: float(r["val_iou"]))
    return {
        "epoch": int(float(best["epoch"])),
        "val_iou": float(best["val_iou"]),
        "val_dice": float(best["val_dice"]),
        "val_mae_pct": float(best["val_mae_pct"]),
        "val_reject75_acc": float(best["val_reject75_acc"]),
        "val_loss": float(best.get("val_loss", 0.0) or 0.0),
        "n_epochs_logged": len(rows),
    }


def write_report_md(
    report_path: Path,
    *,
    run_dir: Path,
    config: dict,
    history_best: dict,
    photos_rows: list[dict],
    photos_agg: dict,
    train_names: list[str],
    val_names: list[str],
    photos_meta: list[dict],
    resolution_note: str,
    colornorm_note: str,
) -> None:
    """Пишет REPORT.md на русском."""
    lines: list[str] = []
    lines.append("# Отчёт: baseline U-Net на миксе + грубые маски Photos")
    lines.append("")
    lines.append("## Модель")
    lines.append("")
    lines.append(f"- Веса: `{run_dir.as_posix()}/best.pt`")
    lines.append(f"- Архитектура: **{config.get('model', 'unet')}** (ResNet34, encoder_weights=`{config.get('encoder_weights')}`)")
    lines.append(f"- Датасеты train: `{', '.join(config.get('datasets', []))}`")
    lines.append(f"- `image_size`: **{config.get('image_size')}**, batch_size={config.get('batch_size')}, lr={config.get('lr')}, seed={config.get('seed')}")
    lines.append(f"- Color-match: **нет** (в этом ране `color_match_photos` не включали)")
    lines.append(
        f"- Best по history (ep **{history_best['epoch']}**): "
        f"val IoU **{history_best['val_iou']:.4f}**, "
        f"Dice **{history_best['val_dice']:.4f}**, "
        f"MAE% **{history_best['val_mae_pct']:.2f}**, "
        f"reject@75 **{history_best['val_reject75_acc']:.3f}**"
    )
    lines.append("")
    lines.append("## Важно про разметку Photos")
    lines.append("")
    lines.append(
        "Ручная разметка в CVAT сделана **толстой кистью** — маски приблизительные, "
        "не «золотой стандарт». Цифры IoU/Dice на Photos ниже — ориентир, "
        "главное смотреть картинки (overlay)."
    )
    lines.append("")
    lines.append(f"**Разрешение инференса Photos:** {resolution_note}")
    lines.append("")
    lines.append(f"**Colornorm:** {colornorm_note}")
    lines.append("")
    lines.append("## Метрики val (из history.csv обучения)")
    lines.append("")
    lines.append("| epoch | val IoU | val Dice | MAE % | reject@75 |")
    lines.append("|------:|--------:|---------:|------:|----------:|")
    lines.append(
        f"| {history_best['epoch']} (best) | {history_best['val_iou']:.4f} | "
        f"{history_best['val_dice']:.4f} | {history_best['val_mae_pct']:.2f} | "
        f"{history_best['val_reject75_acc']:.3f} |"
    )
    lines.append("")
    lines.append("## Метрики Photos (грубая GT, только кадры с маской)")
    lines.append("")
    lines.append(
        f"Кадров с аннотациями: **{int(photos_agg['n'])}**. "
        f"Средние: IoU **{photos_agg['iou']:.3f}**, Dice **{photos_agg['dice']:.3f}**, "
        f"MAE% **{photos_agg['mae_pct']:.1f}**, reject@75 acc **{photos_agg['reject75_acc']:.3f}**."
    )
    lines.append("")
    lines.append("| файл | IoU | Dice | GT % | pred % | MAE % | reject@75 ok |")
    lines.append("|------|----:|-----:|-----:|-------:|------:|:------------:|")
    for row in photos_rows:
        lines.append(
            f"| `{row['name']}` | {row['iou']:.3f} | {row['dice']:.3f} | "
            f"{row['cloud_frac_gt_pct']:.1f} | {row['cloud_frac_pred_pct']:.1f} | "
            f"{row['mae_pct']:.1f} | {'yes' if row['reject75_agree'] else 'no'} |"
        )
    lines.append("")
    lines.append("## Галерея train (микс без аугментаций, seed 42)")
    lines.append("")
    lines.append("Каждый кадр: **RGB | GT (красный) | pred (красный)**.")
    lines.append("")
    for name in train_names:
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"![train {name}](train/{name})")
        lines.append("")
    lines.append("## Галерея val (holdout HRC test, разнообразные по % облаков)")
    lines.append("")
    for name in val_names:
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"![val {name}](val/{name})")
        lines.append("")
    lines.append("## Галерея Photos test (все кадры с маской в CVAT)")
    lines.append("")
    for meta in photos_meta:
        stem = meta["stem"]
        lines.append(f"### {meta['name']}")
        lines.append("")
        lines.append(
            f"GT cloud ≈ {meta['cloud_frac_gt'] * 100:.1f}%, "
            f"XML/preview {meta['xml_width']}×{meta['xml_height']}."
        )
        lines.append("")
        lines.append(f"![photos {stem} triplet](photos_test/{stem}_triplet.png)")
        lines.append("")
        lines.append(
            f"![rgb](photos_test/{stem}_rgb.png) "
            f"![gt](photos_test/{stem}_gt.png) "
            f"![pred](photos_test/{stem}_pred.png)"
        )
        lines.append("")
    lines.append("## Как воспроизвести")
    lines.append("")
    lines.append("```text")
    lines.append("set PYTHONPATH=src")
    lines.append(".venv\\Scripts\\python.exe scripts\\report_model_photos_check.py")
    lines.append("```")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description="Visual report: mix checkpoint vs Photos CVAT masks")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/train/unet_mix_hrc_sparcs_38_4060"),
    )
    p.add_argument(
        "--colornorm-run-dir",
        type=Path,
        default=Path("outputs/train/unet_mix_hrc_sparcs_38_4060_colornorm"),
    )
    p.add_argument(
        "--include-colornorm",
        action="store_true",
        help="Попробовать второй столбец colornorm (по умолчанию пропускаем, если идёт train)",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-train", type=int, default=6)
    p.add_argument("--n-val", type=int, default=6)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports/unet_mix_baseline_photos_check"),
    )
    p.add_argument(
        "--cvat-zip",
        type=Path,
        default=Path(
            "labeling/photos_40/job_4729727_annotations_2026_11_15_12_46_42_cvat for images 1.1.zip"
        ),
    )
    return p.parse_args()


def main() -> None:
    """Точка входа отчёта."""
    args = parse_args()
    project_root = resolve_project_root()
    set_global_seed(args.seed)

    run_dir = (project_root / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir
    out_dir = (project_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    cvat_zip = (project_root / args.cvat_zip).resolve() if not args.cvat_zip.is_absolute() else args.cvat_zip

    photos_root = project_root / "labeling" / "photos_40"
    cvat_export = photos_root / "cvat_export"
    images_dir = photos_root / "images"
    masks_dir = photos_root / "masks"

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    history_best = read_history_best(run_dir / "history.csv")
    image_size = int(config["image_size"])

    print("1) CVAT → masks…", flush=True)
    xml_path = extract_cvat_zip(cvat_zip, cvat_export)
    photos_meta = parse_cvat_masks(xml_path, images_dir, masks_dir)
    print(f"   labeled Photos: {len(photos_meta)}", flush=True)
    for m in photos_meta:
        print(f"   - {m['name']} cloud_gt={m['cloud_frac_gt']*100:.1f}%", flush=True)

    # XML совпал с preview → инференс на preview (те же координаты кисти).
    resolution_note = (
        f"XML width/height совпадают с превью ({photos_meta[0]['xml_width']}×{photos_meta[0]['xml_height']} "
        f"для всех размеченных). Инференс и overlay на "
        f"`labeling/photos_40/images/…`, не на полном Photos/ (оригинал ~3280×2464)."
        if photos_meta
        else "Размеченных кадров нет."
    )

    colornorm_dir = (
        (project_root / args.colornorm_run_dir).resolve()
        if not args.colornorm_run_dir.is_absolute()
        else args.colornorm_run_dir
    )
    colornorm_included = False
    if args.include_colornorm and (colornorm_dir / "best.pt").is_file():
        colornorm_included = True
        colornorm_note = f"Включён второй чекпоинт `{colornorm_dir.as_posix()}/best.pt`."
    else:
        colornorm_note = (
            "Пропущен: параллельно ещё идёт обучение colornorm (`train-mix`), "
            "чтобы не конкурировать за GPU. В отчёте только baseline без color-match."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"2) Load baseline → {device}", flush=True)
    model, ckpt = load_checkpoint(run_dir / "best.pt", device)
    ckpt_metrics = ckpt.get("metrics") or {}
    print(
        f"   ckpt epoch={ckpt.get('epoch')} metrics={ckpt_metrics}",
        flush=True,
    )

    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    photos_dir = out_dir / "photos_test"
    for d in (train_dir, val_dir, photos_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- train samples ---
    print("3) Train gallery…", flush=True)
    color_match = config.get("color_match_method") if config.get("color_match_photos") else None
    max_items_raw = config.get("max_items", 0) or 0
    max_items = None if int(max_items_raw) <= 0 else int(max_items_raw)
    mix = build_datasets(
        list(config["datasets"]),
        image_size=image_size,
        augment=False,
        max_items=max_items,
        seed=int(config.get("seed", args.seed)),
        project_root=project_root,
        color_match=color_match,
    )
    rng = np.random.default_rng(args.seed)
    train_idxs = sorted(rng.choice(len(mix), size=min(args.n_train, len(mix)), replace=False).tolist())
    train_file_names: list[str] = []
    for i, idx in enumerate(train_idxs, start=1):
        sample = mix[idx]
        rgb = chw_to_hwc(sample.image)
        gt = sample.mask[0] > 0.5
        pred = predict_mask(model, sample.image, image_size, device) > 0.5
        # pred уже на image_size (sample уже resized)
        triplet = make_triplet(rgb, gt, pred)
        fname = f"{i:02d}_{sample.dataset_id}_triplet.png"
        save_png_hwc(triplet, train_dir / fname)
        train_file_names.append(fname)
        print(f"   train {fname} cloud_gt={sample.cloud_frac*100:.1f}%", flush=True)

    # --- val samples ---
    print("4) Val gallery…", flush=True)
    val_subset = rebuild_val_subset(config, project_root)
    val_fracs: list[float] = []
    val_cache: list[dict] = []
    for i in range(len(val_subset)):
        sample = val_subset[i]
        val_fracs.append(float(sample.cloud_frac))
        val_cache.append({"sample": sample, "frac": float(sample.cloud_frac)})
    val_pick = pick_diverse_indices(val_fracs, args.n_val, args.seed)
    val_file_names: list[str] = []
    for j, idx in enumerate(val_pick, start=1):
        sample = val_cache[idx]["sample"]
        rgb = chw_to_hwc(sample.image)
        gt = sample.mask[0] > 0.5
        pred = predict_mask(model, sample.image, image_size, device) > 0.5
        triplet = make_triplet(rgb, gt, pred)
        fname = f"{j:02d}_cloud{int(round(sample.cloud_frac*100)):02d}_triplet.png"
        save_png_hwc(triplet, val_dir / fname)
        val_file_names.append(fname)
        print(f"   val {fname}", flush=True)

    # --- Photos test ---
    print("5) Photos test…", flush=True)
    source_map = load_manifest_source_map(photos_root / "manifest.csv")
    photos_rows: list[dict] = []
    for meta in photos_meta:
        preview = Path(meta["preview_path"])
        image_chw = load_rgb_u8(preview)
        gt_u8 = np.asarray(Image.open(meta["mask_path"]))
        if gt_u8.ndim == 3:
            gt_u8 = gt_u8[..., 0]
        gt = (gt_u8 >= 127).astype(np.float32)
        pred = predict_mask(model, image_chw, image_size, device)
        # align shapes
        if pred.shape != gt.shape:
            pred = np.asarray(
                Image.fromarray((pred * 255).astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), Image.Resampling.NEAREST
                )
            )
            pred = (pred > 127).astype(np.float32)

        rgb = chw_to_hwc(image_chw)
        stem = meta["stem"]
        save_png_hwc(rgb, photos_dir / f"{stem}_rgb.png")
        save_png_hwc(overlay_mask(rgb, gt), photos_dir / f"{stem}_gt.png")
        save_png_hwc(overlay_mask(rgb, pred), photos_dir / f"{stem}_pred.png")
        save_png_hwc(make_triplet(rgb, gt, pred), photos_dir / f"{stem}_triplet.png")

        m = metrics_from_masks(pred, gt)
        row = {
            "name": meta["name"],
            "stem": stem,
            "source_jpg": source_map.get(stem) or source_map.get(meta["name"].split("_", 1)[-1].rsplit(".", 1)[0], ""),
            "resolution_used": "preview",
            **m,
        }
        photos_rows.append(row)
        print(
            f"   {meta['name']} IoU={m['iou']:.3f} Dice={m['dice']:.3f} "
            f"GT={m['cloud_frac_gt_pct']:.1f}% pred={m['cloud_frac_pred_pct']:.1f}%",
            flush=True,
        )

    photos_agg = aggregate_metrics(photos_rows)

    metrics = {
        "model_run": str(run_dir),
        "checkpoint": str(run_dir / "best.pt"),
        "config": config,
        "history_best": history_best,
        "checkpoint_metrics": ckpt_metrics,
        "photos_resolution": resolution_note,
        "colornorm_included": colornorm_included,
        "colornorm_note": colornorm_note,
        "photos_n_labeled": len(photos_meta),
        "photos_filenames": [m["name"] for m in photos_meta],
        "photos_per_image": photos_rows,
        "photos_aggregate": photos_agg,
        "train_gallery": train_file_names,
        "val_gallery": val_file_names,
        "pred_threshold": PRED_THRESHOLD,
        "seed": args.seed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report_md(
        out_dir / "REPORT.md",
        run_dir=run_dir,
        config=config,
        history_best=history_best,
        photos_rows=photos_rows,
        photos_agg=photos_agg,
        train_names=train_file_names,
        val_names=val_file_names,
        photos_meta=photos_meta,
        resolution_note=resolution_note,
        colornorm_note=colornorm_note,
    )
    (out_dir / "README.md").write_text(
        "# Отчёт baseline U-Net vs Photos (грубая CVAT-разметка)\n\n"
        "Смотри **[REPORT.md](REPORT.md)** — текст, таблицы и галереи.\n\n"
        "- `metrics.json` — те же числа в машиночитаемом виде\n"
        "- `train/`, `val/`, `photos_test/` — PNG triplet’ы\n",
        encoding="utf-8",
    )

    # подчистить тестовый overlay, если остался
    junk = photos_root / "_rle_test.png"
    if junk.is_file():
        junk.unlink()

    print(f"Готово → {out_dir}", flush=True)
    print(f"REPORT → {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
