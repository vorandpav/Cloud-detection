"""Part C: supervised finetune U-Net на 8 CVAT pixel-масках Photos.

Честная оценка — leave-one-out (LOO): на каждом фолде учимся на 7,
порог (если sweep) выбираем только по этим 7, считаем метрики на 1 held-out.
Финальная модель — один прогон на всех 8 (для демо; overlays = train-set,
оптимистично).

Пример::

    set PYTHONPATH=src
    .venv\\Scripts\\python.exe scripts/finetune_photos_masks.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cloudness_detection.data.common import Sample, load_mask_binary, load_rgb_u8, make_sample
from cloudness_detection.data.mix import (
    WeightedMixedDataset,
    build_one,
    collate_samples,
    resolve_project_root,
)
from cloudness_detection.losses.soft_jaccard import DiceBCELoss
from cloudness_detection.seed import DEFAULT_SEED, set_global_seed

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from calibrate_photos_threshold import (  # noqa: E402
    PRED_THRESHOLDS,
    aggregate_metrics,
    chw_to_hwc,
    load_checkpoint,
    make_triplet,
    metrics_from_masks,
    overlay_mask,
    predict_prob,
    save_png_hwc,
    split_photos,
)
from finetune_photos_pseudo import (  # noqa: E402
    OOM_RETRIES,
    OOM_SLEEP_SEC,
    append_csv,
    wait_for_gpu_headroom,
)

# GT грубая (толстая кисть) — всё равно учимся; в отчёте это явно.
GT_QUALITY_NOTE = (
    "Маски CVAT — грубая разметка толстой кистью (не pixel-perfect). "
    "Метрики и overlays отражают согласование с этой разметкой, не с идеальной границей облака."
)


class HardMaskPhotosDataset(Dataset):
    """Photos JPG + бинарная PNG-маска (CVAT).

    Attributes:
        pairs: Список (jpg, mask_png).
    """

    dataset_id = "photos_masks"

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        *,
        image_size: int,
        augment: bool,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """Args:
            pairs: Пары (preview_jpg, mask_png).
            image_size: Сторона квадрата.
            augment: Flip/rot90.
            seed: Сид (API-совместимость; ауги через global/numpy внутри make_sample).
        """
        if not pairs:
            raise ValueError("Пустой HardMaskPhotosDataset")
        self.pairs = list(pairs)
        self.image_size = image_size
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Sample:
        img_path, mask_path = self.pairs[index]
        if not mask_path.is_file():
            raise FileNotFoundError(f"Нет маски: {mask_path}")
        image = load_rgb_u8(img_path)
        mask = load_mask_binary(mask_path)
        return make_sample(
            image,
            mask,
            self.dataset_id,
            str(img_path),
            self.image_size,
            self.augment,
        )


def plot_curves(history: list[dict[str, float]], out_path: Path, title: str) -> None:
    """Кривая train loss."""
    if not history:
        return
    epochs = [int(r["epoch"]) for r in history]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(epochs, [r["train_loss"] for r in history], label="train")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def build_train_pairs(
    labeled: list[Path],
    masks_dir: Path,
    exclude: Path | None,
) -> list[tuple[Path, Path]]:
    """JPG+mask пары; exclude — held-out (если задан)."""
    pairs: list[tuple[Path, Path]] = []
    for jpg in labeled:
        if exclude is not None and jpg.resolve() == exclude.resolve():
            continue
        mask = masks_dir / f"{jpg.stem}.png"
        if not mask.is_file():
            raise FileNotFoundError(f"Нет маски для {jpg.name}: {mask}")
        pairs.append((jpg, mask))
    if not pairs:
        raise RuntimeError("Пустой train-набор масок")
    return pairs


def make_mix_loader(
    photos_pairs: list[tuple[Path, Path]],
    *,
    image_size: int,
    batch_size: int,
    epoch_size: int,
    seed: int,
    num_workers: int,
    project_root: Path,
    photos_weight: float,
    hrc_weight: float,
    sparcs_weight: float,
) -> DataLoader:
    """Микс: photos masks (высокий вес) + лёгкий hrc/sparcs regularizer."""
    photos_ds = HardMaskPhotosDataset(
        photos_pairs,
        image_size=image_size,
        augment=True,
        seed=seed,
    )
    datasets: list[Dataset] = [photos_ds]
    weights: list[float] = [photos_weight]
    if hrc_weight > 0:
        datasets.append(
            build_one(
                "hrc",
                image_size=image_size,
                augment=True,
                max_items=None,
                seed=seed,
                project_root=project_root,
                split="train",
                color_match=None,
            )
        )
        weights.append(hrc_weight)
    if sparcs_weight > 0:
        datasets.append(
            build_one(
                "sparcs",
                image_size=image_size,
                augment=True,
                max_items=None,
                seed=seed,
                project_root=project_root,
                split="train",
                color_match=None,
            )
        )
        weights.append(sparcs_weight)
    mix = WeightedMixedDataset(
        datasets,
        weights,
        seed=seed,
        epoch_size=epoch_size,
    )
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        mix,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        collate_fn=collate_samples,
    )


def train_epochs(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    history_path: Path | None,
    curves_path: Path | None,
    title: str,
) -> list[dict[str, float]]:
    """Короткий FT; берём последний epoch (held-out не смотрим).

    Args:
        model: Уже на device, веса baseline загружены.
        loader: Train loader.
        device: cuda/cpu.
        epochs: Число эпох.
        lr: AdamW lr.
        history_path: Опционально CSV.
        curves_path: Опционально PNG.
        title: Подпись кривой.

    Returns:
        История по эпохам.
    """
    criterion = DiceBCELoss(jaccard_weight=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    fields = ["epoch", "train_loss", "lr", "epoch_sec"]
    if history_path is not None and history_path.exists():
        history_path.unlink()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses: list[float] = []
        for batch in loader:
            for attempt in range(1, OOM_RETRIES + 1):
                try:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = criterion(logits, masks)
                    loss.backward()
                    optimizer.step()
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    print(
                        f"CUDA OOM train (попытка {attempt}/{OOM_RETRIES}), "
                        f"ждём {OOM_SLEEP_SEC:.0f}s…",
                        flush=True,
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    time.sleep(OOM_SLEEP_SEC)
            else:
                raise RuntimeError("Train step: исчерпаны OOM-ретраи")
            losses.append(float(loss.item()))
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_sec": float(time.time() - t0),
        }
        history.append(row)
        if history_path is not None:
            append_csv(history_path, row, fields)
        print(
            f"  epoch {epoch}/{epochs} loss={row['train_loss']:.4f} "
            f"t={row['epoch_sec']:.0f}s",
            flush=True,
        )
    if curves_path is not None:
        plot_curves(history, curves_path, title)
    return history


@torch.no_grad()
def eval_one_image(
    model: nn.Module,
    jpg: Path,
    mask_path: Path,
    *,
    image_size: int,
    device: torch.device,
    thr: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Pred + метрики для одного кадра.

    Returns:
        (metrics, rgb_hwc, gt, pred).
    """
    image_chw = load_rgb_u8(jpg)
    prob = predict_prob(model, image_chw, image_size, device)
    pred = (prob >= thr).astype(np.float32)
    gt = load_mask_binary(mask_path)[0]
    # load_mask_binary → 1xHxW float; приводим к HxW
    if gt.ndim == 3:
        gt = gt[0]
    rgb = chw_to_hwc(image_chw)
    m = metrics_from_masks(pred, gt)
    return m, rgb, gt, pred


def sweep_thr_on_pairs(
    model: nn.Module,
    pairs: list[tuple[Path, Path]],
    *,
    image_size: int,
    device: torch.device,
) -> tuple[float, list[dict]]:
    """Выбор thr по max mean IoU на fold-train (не на held-out).

    Returns:
        (best_thr, rows_per_thr).
    """
    probs: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    for jpg, mask_path in pairs:
        image_chw = load_rgb_u8(jpg)
        probs.append(predict_prob(model, image_chw, image_size, device))
        gt = load_mask_binary(mask_path)[0]
        gts.append(gt.astype(np.float32))

    rows: list[dict] = []
    best_thr = 0.5
    best_iou = -1.0
    for thr in PRED_THRESHOLDS:
        per: list[dict] = []
        for prob, gt in zip(probs, gts):
            pred = (prob >= thr).astype(np.float32)
            per.append(metrics_from_masks(pred, gt))
        agg = aggregate_metrics(per)
        rows.append({"threshold": float(thr), **agg})
        # tie → меньший thr (чуть агрессивнее)
        if agg["iou"] > best_iou + 1e-9 or (
            abs(agg["iou"] - best_iou) <= 1e-9 and thr < best_thr
        ):
            best_iou = float(agg["iou"])
            best_thr = float(thr)
    return best_thr, rows


def save_overlay_set(
    out_dir: Path,
    stem: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
) -> None:
    """RGB / GT / pred / triplet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    save_png_hwc(rgb, out_dir / f"{stem}_rgb.png")
    save_png_hwc(overlay_mask(rgb, gt), out_dir / f"{stem}_gt.png")
    save_png_hwc(overlay_mask(rgb, pred), out_dir / f"{stem}_pred.png")
    save_png_hwc(make_triplet(rgb, gt, pred), out_dir / f"{stem}_triplet.png")


def load_ab_comparison(project_root: Path) -> dict:
    """Достаёт baseline/A/B из предыдущего отчёта, если есть."""
    metrics_path = (
        project_root / "outputs" / "reports" / "photos_thr_and_pseudo_ft" / "metrics.json"
    )
    if not metrics_path.is_file():
        return {}
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "baseline_0.5": raw.get("baseline_0_5_heldout") or raw.get("comparison_photos8", {}).get(
            "baseline_0.5"
        ),
        "A": (raw.get("part_a") or {}).get("heldout_aggregate")
        or raw.get("comparison_photos8", {}).get("A"),
        "B": (raw.get("part_b") or {}).get("heldout_aggregate")
        or raw.get("comparison_photos8", {}).get("B"),
        "source": str(metrics_path),
    }


def write_report_md(
    report_path: Path,
    *,
    config: dict,
    loo: dict,
    final: dict,
    ab: dict,
    gpu_note: str,
) -> None:
    """REPORT.md на русском."""
    loo_agg = loo["aggregate"]
    final_agg = final["aggregate"]
    lines: list[str] = []
    lines.append("# Photos mask-FT (вариант C)")
    lines.append("")
    lines.append("## Суть")
    lines.append("")
    lines.append(
        "Supervised finetune U-Net на **8 CVAT pixel-масках** "
        "(не cloud_pct CSV и не soft-pseudo с unlabeled)."
    )
    lines.append("")
    lines.append(f"**Качество GT:** {GT_QUALITY_NOTE}")
    lines.append("")
    lines.append("## Протокол")
    lines.append("")
    lines.append(
        "1. **LOO (честно):** 8 фолдов; train = 7 масок + лёгкий mix HRC/SPARCS; "
        "held-out = 1. Порог: "
        f"{loo['thr_rule']}."
    )
    lines.append(
        "2. **Final (оптимистично):** один прогон на всех 8; overlays на этих же 8 — "
        "**train-set**, не test."
    )
    lines.append("")
    lines.append(
        f"- Baseline: `{config['init_checkpoint']}` (image_size={config['image_size']})"
    )
    lines.append(
        f"- Рецепт: epochs={config['epochs']}, lr={config['lr']}, "
        f"bs={config['batch_size']}, epoch_size={config['epoch_size']}, "
        f"seed={config['seed']}, weights={config['weights']}"
    )
    lines.append(f"- GPU: {gpu_note}")
    lines.append("")
    lines.append("## Сравнение (Photos-8)")
    lines.append("")
    lines.append("| вариант | thr | IoU | Dice | MAE % | reject@75 | note |")
    lines.append("|--------|----:|----:|-----:|------:|----------:|------|")

    def _row(name: str, thr: str, agg: dict | None, note: str) -> str:
        if not agg:
            return f"| {name} | — | — | — | — | — | {note} |"
        return (
            f"| {name} | {thr} | {agg['iou']:.3f} | {agg['dice']:.3f} | "
            f"{agg['mae_pct']:.1f} | {agg['reject75_acc']:.3f} | {note} |"
        )

    lines.append(_row("baseline", "0.50", ab.get("baseline_0.5"), "held-out historically"))
    lines.append(_row("A calibrated", "0.50", ab.get("A"), "thr на unlabeled-32"))
    lines.append(_row("B pseudo-FT", "0.50", ab.get("B"), "soft pseudo на 32"))
    thr_loo = loo.get("mean_thr")
    thr_loo_s = f"{thr_loo:.2f}" if thr_loo is not None else "fold"
    lines.append(_row("**C mask-FT LOO**", thr_loo_s, loo_agg, "**честная** оценка"))
    lines.append(
        _row(
            "C mask-FT final@train",
            f"{final['threshold']:.2f}",
            final_agg,
            "⚠ train-set (оптимистично)",
        )
    )
    lines.append("")
    lines.append("## LOO per-image (честно)")
    lines.append("")
    lines.append("| файл | thr_fold | IoU | Dice | GT % | pred % | MAE % |")
    lines.append("|------|---------:|----:|-----:|-----:|-------:|------:|")
    for row in loo["per_image"]:
        lines.append(
            f"| `{row['name']}` | {row['threshold']:.2f} | {row['iou']:.3f} | "
            f"{row['dice']:.3f} | {row['cloud_frac_gt_pct']:.1f} | "
            f"{row['cloud_frac_pred_pct']:.1f} | {row['mae_pct']:.1f} |"
        )
    lines.append("")
    lines.append(
        f"**LOO mean IoU = {loo_agg['iou']:.3f}** "
        f"(baseline был ≈0.492)."
    )
    lines.append("")
    lines.append("## Final @ train-8 (оптимистично)")
    lines.append("")
    lines.append(
        f"Ckpt: `{final['best_ckpt']}` · thr={final['threshold']:.2f} "
        f"({final['thr_rule']})"
    )
    lines.append("")
    lines.append("| файл | IoU | Dice | GT % | pred % | MAE % |")
    lines.append("|------|----:|-----:|-----:|-------:|------:|")
    for row in final["per_image"]:
        lines.append(
            f"| `{row['name']}` | {row['iou']:.3f} | {row['dice']:.3f} | "
            f"{row['cloud_frac_gt_pct']:.1f} | {row['cloud_frac_pred_pct']:.1f} | "
            f"{row['mae_pct']:.1f} |"
        )
    lines.append("")
    lines.append("## Overlays")
    lines.append("")
    lines.append("- `loo/` — pred каждого held-out фолда (честно)")
    lines.append("- `final_train/` — final-модель на тех же 8 (**train-set**)")
    lines.append("")
    for row in loo["per_image"]:
        stem = row["stem"]
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append(f"![LOO {stem}](loo/{stem}_triplet.png)")
        lines.append("")
        lines.append(f"[final@train](final_train/{stem}_triplet.png)")
        lines.append("")
    lines.append("## Как воспроизвести")
    lines.append("")
    lines.append("```text")
    lines.append("set PYTHONPATH=src")
    lines.append(".venv\\Scripts\\python.exe scripts\\finetune_photos_masks.py")
    lines.append("```")
    lines.append("")
    lines.append("## Вывод")
    lines.append("")
    base = ab.get("baseline_0.5") or {}
    base_iou = float(base.get("iou", 0.492))
    delta = float(loo_agg["iou"]) - base_iou
    sign = "+" if delta >= 0 else ""
    lines.append(
        f"LOO IoU {loo_agg['iou']:.3f} vs baseline {base_iou:.3f} ({sign}{delta:.3f}). "
        f"Final@train IoU {final_agg['iou']:.3f} — не обобщение, а fit на 8. "
        "GT грубая; смотреть overlays."
    )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description="Mask-supervised FT on Photos-8 (LOO + final)")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/train/unet_mix_hrc_sparcs_38_4060"),
    )
    p.add_argument(
        "--out-train-dir",
        type=Path,
        default=Path("outputs/train/unet_photos_mask_ft"),
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path("outputs/reports/photos_mask_ft"),
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epoch-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--photos-weight", type=float, default=6.0)
    p.add_argument("--hrc-weight", type=float, default=1.5)
    p.add_argument("--sparcs-weight", type=float, default=1.0)
    p.add_argument(
        "--thr-mode",
        type=str,
        default="fold_sweep",
        choices=["fold_sweep", "fixed_0.5"],
        help="fold_sweep: max IoU на 7 train; fixed_0.5: всегда 0.5",
    )
    p.add_argument("--need-free-mib", type=int, default=2800)
    p.add_argument("--max-wait-sec", type=float, default=7200.0)
    p.add_argument("--skip-loo", action="store_true")
    p.add_argument("--skip-final", action="store_true")
    p.add_argument(
        "--resume-loo",
        action="store_true",
        help="Пропускать фолды, у которых уже есть fold.json (после сбоя ПК).",
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    """Точка входа Part C."""
    args = parse_args()
    project_root = resolve_project_root()
    set_global_seed(args.seed)

    run_dir = (project_root / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir
    out_train = (
        (project_root / args.out_train_dir).resolve()
        if not args.out_train_dir.is_absolute()
        else args.out_train_dir
    )
    report_dir = (
        (project_root / args.report_dir).resolve()
        if not args.report_dir.is_absolute()
        else args.report_dir
    )
    out_train.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    loo_dir = report_dir / "loo"
    final_ov_dir = report_dir / "final_train"
    folds_dir = out_train / "loo_folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    photos_root = project_root / "labeling" / "photos_40"
    images_dir = photos_root / "images"
    masks_dir = photos_root / "masks"
    labeled, _unlabeled = split_photos(images_dir, masks_dir)
    if len(labeled) != 8:
        raise RuntimeError(
            f"Ожидали 8 масок CVAT, получили {len(labeled)}. "
            f"Проверь {masks_dir} (при необходимости перепарси CVAT zip)."
        )

    config_base = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    image_size = int(config_base["image_size"])
    init_ckpt = run_dir / "best.pt"
    if not init_ckpt.is_file():
        raise FileNotFoundError(f"Нет baseline: {init_ckpt}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Запрошен cuda, но CUDA недоступна")
    device = torch.device(args.device)

    ft_config = {
        "model": "unet",
        "part": "C_mask_ft",
        "init_checkpoint": str(init_ckpt),
        "datasets": ["photos_masks", "hrc", "sparcs"],
        "weights": {
            "photos_masks": args.photos_weight,
            "hrc": args.hrc_weight,
            "sparcs": args.sparcs_weight,
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "image_size": image_size,
        "epoch_size": args.epoch_size,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "thr_mode": args.thr_mode,
        "gt_note": GT_QUALITY_NOTE,
        "labeled": [p.name for p in labeled],
        "device": str(device),
    }
    (out_train / "config.json").write_text(
        json.dumps(ft_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    gpu_info: dict = {"notes": ["no wait"], "waited_sec": 0.0, "skipped": True}
    if device.type == "cuda" and not (args.skip_loo and args.skip_final):
        print("Ждём свободную GPU (не убиваем чужой train)…", flush=True)
        gpu_info = wait_for_gpu_headroom(
            need_free_mib=args.need_free_mib,
            poll_sec=60.0,
            max_wait_sec=args.max_wait_sec,
        )

    # ---------- LOO ----------
    loo_rows: list[dict] = []
    loo_fold_meta: list[dict] = []
    if not args.skip_loo:
        print(f"=== LOO на {len(labeled)} кадрах ===", flush=True)
        for fold_i, held in enumerate(labeled, start=1):
            set_global_seed(args.seed + fold_i * 17)
            train_pairs = build_train_pairs(labeled, masks_dir, exclude=held)
            fold_out = folds_dir / f"fold_{held.stem}"
            fold_out.mkdir(parents=True, exist_ok=True)
            fold_json_path = fold_out / "fold.json"
            if args.resume_loo and fold_json_path.is_file():
                fold_meta = json.loads(fold_json_path.read_text(encoding="utf-8"))
                row = fold_meta["heldout_metrics"]
                loo_rows.append(row)
                loo_fold_meta.append(fold_meta)
                # overlays могли пропасть — восстановим из last.pt при наличии
                trip = loo_dir / f"{held.stem}_triplet.png"
                last_pt = fold_out / "last.pt"
                if not trip.is_file() and last_pt.is_file():
                    model, _ = load_checkpoint(last_pt, device)
                    thr = float(fold_meta.get("threshold", 0.5))
                    mask_path = masks_dir / f"{held.stem}.png"
                    _m, rgb, gt, pred = eval_one_image(
                        model,
                        held,
                        mask_path,
                        image_size=image_size,
                        device=device,
                        thr=thr,
                    )
                    save_overlay_set(loo_dir, held.stem, rgb, gt, pred)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                print(
                    f"[fold {fold_i}/{len(labeled)}] SKIP resume "
                    f"held-out={held.name} IoU={row['iou']:.3f}",
                    flush=True,
                )
                continue
            print(
                f"[fold {fold_i}/{len(labeled)}] held-out={held.name} "
                f"train_n={len(train_pairs)}",
                flush=True,
            )
            model, _ = load_checkpoint(init_ckpt, device)
            loader = make_mix_loader(
                train_pairs,
                image_size=image_size,
                batch_size=args.batch_size,
                epoch_size=args.epoch_size,
                seed=args.seed + fold_i,
                num_workers=args.num_workers,
                project_root=project_root,
                photos_weight=args.photos_weight,
                hrc_weight=args.hrc_weight,
                sparcs_weight=args.sparcs_weight,
            )
            history = train_epochs(
                model,
                loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                history_path=fold_out / "history.csv",
                curves_path=fold_out / "curves.png",
                title=f"LOO fold {held.stem}",
            )
            # thr только по train-7
            if args.thr_mode == "fixed_0.5":
                thr = 0.5
                sweep_rows: list[dict] = []
                thr_rule_fold = "fixed 0.5"
            else:
                thr, sweep_rows = sweep_thr_on_pairs(
                    model,
                    train_pairs,
                    image_size=image_size,
                    device=device,
                )
                thr_rule_fold = f"max mean IoU на train-7 → {thr:.2f}"
            mask_path = masks_dir / f"{held.stem}.png"
            m, rgb, gt, pred = eval_one_image(
                model,
                held,
                mask_path,
                image_size=image_size,
                device=device,
                thr=thr,
            )
            save_overlay_set(loo_dir, held.stem, rgb, gt, pred)
            # также @0.5 для справки
            m05, _, _, _ = eval_one_image(
                model,
                held,
                mask_path,
                image_size=image_size,
                device=device,
                thr=0.5,
            )
            row = {
                "name": held.name,
                "stem": held.stem,
                "threshold": thr,
                **m,
                "iou_at_0.5": m05["iou"],
                "dice_at_0.5": m05["dice"],
            }
            loo_rows.append(row)
            fold_meta = {
                "held_out": held.name,
                "train": [p.name for p, _ in train_pairs],
                "threshold": thr,
                "thr_rule": thr_rule_fold,
                "sweep_on_train7": sweep_rows,
                "heldout_metrics": row,
                "history": history,
            }
            loo_fold_meta.append(fold_meta)
            (fold_out / "fold.json").write_text(
                json.dumps(fold_meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # лёгкий ckpt фолда (на всякий)
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": "unet",
                    "image_size": image_size,
                    "seed": args.seed + fold_i,
                    "epoch": args.epochs,
                    "held_out": held.name,
                    "threshold": thr,
                    "config": ft_config,
                },
                fold_out / "last.pt",
            )
            print(
                f"  → held IoU={m['iou']:.3f} Dice={m['dice']:.3f} "
                f"thr={thr:.2f} (IoU@0.5={m05['iou']:.3f})",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    loo_agg = aggregate_metrics(loo_rows) if loo_rows else {}
    mean_thr = float(np.mean([r["threshold"] for r in loo_rows])) if loo_rows else None
    if args.thr_mode == "fixed_0.5":
        loo_thr_rule = "Фиксированный thr=0.5 на всех фолдах (без sweep)."
    else:
        loo_thr_rule = (
            "На каждом фолде: sweep thr∈PRED_THRESHOLDS по max mean IoU на train-7, "
            "затем этот thr на held-out. Held-out в выборе thr не участвует."
        )
    loo_payload = {
        "thr_mode": args.thr_mode,
        "thr_rule": loo_thr_rule,
        "mean_thr": mean_thr,
        "per_image": loo_rows,
        "aggregate": loo_agg,
        "folds": loo_fold_meta,
        "overlays_dir": str(loo_dir),
    }

    # ---------- Final на всех 8 ----------
    final_rows: list[dict] = []
    final_history: list[dict] = []
    best_path = out_train / "best.pt"
    final_thr = 0.5
    final_thr_rule = "fixed 0.5"
    final_sweep: list[dict] = []
    if not args.skip_final:
        print("=== Final FT на всех 8 масках ===", flush=True)
        set_global_seed(args.seed)
        all_pairs = build_train_pairs(labeled, masks_dir, exclude=None)
        model, _ = load_checkpoint(init_ckpt, device)
        loader = make_mix_loader(
            all_pairs,
            image_size=image_size,
            batch_size=args.batch_size,
            epoch_size=args.epoch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            project_root=project_root,
            photos_weight=args.photos_weight,
            hrc_weight=args.hrc_weight,
            sparcs_weight=args.sparcs_weight,
        )
        for name in ("history.csv", "history.json"):
            p = out_train / name
            if p.exists():
                p.unlink()
        final_history = train_epochs(
            model,
            loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            history_path=out_train / "history.csv",
            curves_path=out_train / "curves.png",
            title="photos mask-FT final (all 8)",
        )
        (out_train / "history.json").write_text(
            json.dumps(final_history, indent=2),
            encoding="utf-8",
        )
        if args.thr_mode == "fixed_0.5":
            final_thr = 0.5
            final_thr_rule = "fixed 0.5"
        else:
            final_thr, final_sweep = sweep_thr_on_pairs(
                model,
                all_pairs,
                image_size=image_size,
                device=device,
            )
            final_thr_rule = (
                f"max mean IoU на train-8 → {final_thr:.2f} "
                "(оптимистично: те же кадры, что и eval)"
            )
        torch.save(
            {
                "model": model.state_dict(),
                "model_name": "unet",
                "image_size": image_size,
                "seed": args.seed,
                "epoch": args.epochs,
                "threshold": final_thr,
                "metrics_note": "train-set optimistic; see LOO for honest metrics",
                "config": ft_config,
            },
            best_path,
        )
        print(f"Saved {best_path}", flush=True)
        for jpg in labeled:
            mask_path = masks_dir / f"{jpg.stem}.png"
            m, rgb, gt, pred = eval_one_image(
                model,
                jpg,
                mask_path,
                image_size=image_size,
                device=device,
                thr=final_thr,
            )
            save_overlay_set(final_ov_dir, jpg.stem, rgb, gt, pred)
            final_rows.append({"name": jpg.name, "stem": jpg.stem, **m})
            print(
                f"  final {jpg.name} IoU={m['iou']:.3f} Dice={m['dice']:.3f}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        (out_train / "DONE.txt").write_text(
            f"final_thr={final_thr:.2f}\noutdir={out_train}\n",
            encoding="utf-8",
        )
    elif best_path.is_file():
        model, ckpt = load_checkpoint(best_path, device)
        final_thr = float(ckpt.get("threshold", 0.5))
        for jpg in labeled:
            mask_path = masks_dir / f"{jpg.stem}.png"
            m, rgb, gt, pred = eval_one_image(
                model,
                jpg,
                mask_path,
                image_size=image_size,
                device=device,
                thr=final_thr,
            )
            save_overlay_set(final_ov_dir, jpg.stem, rgb, gt, pred)
            final_rows.append({"name": jpg.name, "stem": jpg.stem, **m})

    final_agg = aggregate_metrics(final_rows) if final_rows else {}
    final_payload = {
        "best_ckpt": str(best_path),
        "threshold": final_thr,
        "thr_rule": final_thr_rule,
        "sweep_on_train8": final_sweep,
        "per_image": final_rows,
        "aggregate": final_agg,
        "history": final_history,
        "overlays_dir": str(final_ov_dir),
        "optimistic_note": "Overlays/metrics на тех же 8, что в train — не test.",
    }

    waited = float(gpu_info.get("waited_sec", 0.0) or 0.0)
    gpu_note = (
        f"ожидание VRAM {waited:.0f}s; notes={gpu_info.get('notes')}"
        if not gpu_info.get("skipped")
        else f"wait skipped/notes={gpu_info.get('notes')}"
    )

    ab = load_ab_comparison(project_root)
    metrics = {
        "seed": args.seed,
        "part": "C_mask_ft",
        "gt_note": GT_QUALITY_NOTE,
        "config": ft_config,
        "gpu_note": gpu_note,
        "gpu_wait": gpu_info,
        "comparison_photos8": {
            "baseline_0.5": ab.get("baseline_0.5"),
            "A": ab.get("A"),
            "B": ab.get("B"),
            "C_mask_ft_LOO": loo_agg,
            "C_mask_ft_final_train": {
                **final_agg,
                "flag": "train_set_optimistic",
                "threshold": final_thr,
            },
        },
        "loo": loo_payload,
        "final": final_payload,
        "ab_source": ab.get("source"),
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / "loo.json").write_text(
        json.dumps(loo_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / "final.json").write_text(
        json.dumps(final_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report_md(
        report_dir / "REPORT.md",
        config=ft_config,
        loo=loo_payload,
        final=final_payload,
        ab=ab,
        gpu_note=gpu_note,
    )
    (report_dir / "README.md").write_text(
        "# Photos mask-FT (C)\n\n"
        "Смотри **[REPORT.md](REPORT.md)**.\n\n"
        "- `loo/` — честные held-out overlays\n"
        "- `final_train/` — final на train-8 (оптимистично)\n"
        "- Train: `outputs/train/unet_photos_mask_ft/`\n",
        encoding="utf-8",
    )
    print(
        f"Готово. LOO IoU={loo_agg.get('iou', float('nan')):.3f} → {report_dir / 'REPORT.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
