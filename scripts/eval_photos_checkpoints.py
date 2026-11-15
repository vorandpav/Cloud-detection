"""Одноразовый прогон нескольких чекпоинтов на 8 Photos с CVAT-масками.

На Photos color-match НЕ применяем: в train он матчил публичные → Photos,
сами Photos уже в целевом домене.

Пример::

    set PYTHONPATH=src
    .venv\\Scripts\\python.exe scripts/eval_photos_checkpoints.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from calibrate_photos_threshold import (  # noqa: E402
    aggregate_metrics,
    chw_to_hwc,
    load_checkpoint,
    make_triplet,
    metrics_from_masks,
    predict_prob,
    save_png_hwc,
    split_photos,
)
from cloudness_detection.data.common import load_mask_binary, load_rgb_u8  # noqa: E402
from cloudness_detection.seed import DEFAULT_SEED, set_global_seed  # noqa: E402


def eval_ckpt(
    name: str,
    ckpt: Path,
    labeled: list[Path],
    masks_dir: Path,
    out_dir: Path,
    device: torch.device,
    thr: float = 0.5,
) -> dict:
    """Грузит чекпоинт, считает метрики и пишет triplets.

    Args:
        name: Короткое имя прогона (папка overlays).
        ckpt: Путь к best.pt.
        labeled: JPG превью с масками.
        masks_dir: PNG маски.
        out_dir: Корень отчёта.
        device: cuda/cpu.
        thr: Порог бинаризации.

    Returns:
        Словарь с aggregate + per_image.
    """
    model, raw = load_checkpoint(ckpt, device)
    cfg = raw.get("config") or {}
    image_size = int(cfg.get("image_size") or 384)
    ov = out_dir / name
    ov.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    print(f"=== {name} image_size={image_size} thr={thr} ===", flush=True)
    for jpg in labeled:
        mask_path = masks_dir / f"{jpg.stem}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        image_chw = load_rgb_u8(jpg)
        gt = load_mask_binary(mask_path)
        if gt.ndim == 3:
            gt = gt[0]
        # маска и превью должны совпасть по H×W (CVAT на 768×577)
        if gt.shape != image_chw.shape[-2:]:
            raise ValueError(
                f"{jpg.name}: mask {gt.shape} != image {image_chw.shape[-2:]}"
            )
        prob = predict_prob(model, image_chw, image_size, device)
        pred = (prob >= thr).astype(np.float32)
        gt_bin = (gt > 0.5).astype(np.float32)
        m = metrics_from_masks(pred, gt_bin)
        row = {"name": jpg.name, "stem": jpg.stem, **m}
        rows.append(row)
        rgb = chw_to_hwc(image_chw)
        trip = make_triplet(rgb, gt_bin > 0.5, pred > 0.5)
        save_png_hwc(trip, ov / f"{jpg.stem}_triplet.png")
        print(
            f"  {jpg.name} IoU={m['iou']:.3f} Dice={m['dice']:.3f} "
            f"GT={m['cloud_frac_gt_pct']:.1f}% pred={m['cloud_frac_pred_pct']:.1f}%",
            flush=True,
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    agg = aggregate_metrics(rows)
    return {
        "name": name,
        "checkpoint": str(ckpt),
        "image_size": image_size,
        "threshold": thr,
        "config_snip": {
            "model": cfg.get("model"),
            "color_match_photos": cfg.get("color_match_photos"),
            "color_match_method": cfg.get("color_match_method"),
        },
        "aggregate": agg,
        "per_image": rows,
        "overlays": str(ov),
    }


def main() -> None:
    """Точка входа."""
    set_global_seed(DEFAULT_SEED)
    root = Path(__file__).resolve().parents[1]
    photos = root / "labeling" / "photos_40"
    labeled, _ = split_photos(photos / "images", photos / "masks")
    if len(labeled) != 8:
        raise RuntimeError(f"Ожидали 8 масок, got {len(labeled)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = root / "outputs" / "reports" / "photos_compare_models"
    out.mkdir(parents=True, exist_ok=True)

    jobs = [
        (
            "baseline_unet",
            root / "outputs" / "train" / "unet_mix_hrc_sparcs_38_4060" / "best.pt",
        ),
        (
            "colornorm_unet",
            root / "outputs" / "train" / "unet_mix_hrc_sparcs_38_4060_colornorm" / "best.pt",
        ),
        (
            "segformer_kaggle",
            root / "outputs" / "kaggle" / "kaggle_segformer_mix" / "best.pt",
        ),
    ]
    results = []
    for name, ckpt in jobs:
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        results.append(eval_ckpt(name, ckpt, labeled, photos / "masks", out, device))

    lines = [
        "# Photos-8: baseline vs color-match vs SegFormer",
        "",
        "Одинаковые 8 CVAT-масок, thr=0.5, превью 768×577. "
        "Color-match на инференсе Photos **не** включали "
        "(в train он тянул публичные к домену Photos).",
        "",
        "| модель | val IoU (из train) | Photos IoU | Dice | MAE % | rej@75 |",
        "|--------|-------------------:|-----------:|-----:|------:|-------:|",
    ]
    val_ref = {
        "baseline_unet": 0.881,
        "colornorm_unet": 0.874,
        "segformer_kaggle": 0.885,
    }
    for r in results:
        a = r["aggregate"]
        lines.append(
            f"| `{r['name']}` | {val_ref.get(r['name'], float('nan')):.3f} | "
            f"**{a['iou']:.3f}** | {a['dice']:.3f} | {a['mae_pct']:.1f} | "
            f"{a['reject75_acc']:.3f} |"
        )
    lines.extend(["", "## Per-image IoU", ""])
    lines.append("| файл | baseline | colornorm | segformer |")
    lines.append("|------|---------:|----------:|----------:|")
    by = {r["name"]: {x["stem"]: x for x in r["per_image"]} for r in results}
    for jpg in labeled:
        stem = jpg.stem
        lines.append(
            f"| `{jpg.name}` | "
            f"{by['baseline_unet'][stem]['iou']:.3f} | "
            f"{by['colornorm_unet'][stem]['iou']:.3f} | "
            f"{by['segformer_kaggle'][stem]['iou']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Overlays (RGB | GT | pred)",
            "",
            "- `baseline_unet/`",
            "- `colornorm_unet/`",
            "- `segformer_kaggle/`",
            "",
        ]
    )
    for jpg in labeled:
        stem = jpg.stem
        lines.append(f"### {jpg.name}")
        lines.append("")
        lines.append(f"![baseline](baseline_unet/{stem}_triplet.png)")
        lines.append("")
        lines.append(f"![colornorm](colornorm_unet/{stem}_triplet.png)")
        lines.append("")
        lines.append(f"![segformer](segformer_kaggle/{stem}_triplet.png)")
        lines.append("")

    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Готово → {out / 'REPORT.md'}", flush=True)
    for r in results:
        a = r["aggregate"]
        print(f"  {r['name']}: IoU={a['iou']:.3f} Dice={a['dice']:.3f}", flush=True)


if __name__ == "__main__":
    main()
