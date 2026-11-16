"""Обучение U-Net / SegFormer / Cloud-Net на взвешенном миксе датасетов.

Примеры::

    uv run train-mix --model unet --datasets hrc,sparcs --epochs 1 --max-items 16
    uv run train-mix --model segformer_b2 --device-plan 4060 --datasets hrc,cloud38
    uv run train-mix --model cloudnet --datasets hrc,sparcs,cloud38 --encoder-weights none
    uv run train-mix --model cloudnet_plus --loss fjl1 --encoder-weights none
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cloudness_detection.data.common import enable_cubesat_augs
from cloudness_detection.data.mix import (
    DEFAULT_WEIGHTS,
    DEVICE_PLANS,
    WeightedMixedDataset,
    build_datasets,
    build_one,
    collate_samples,
    resolve_data_root,
    resolve_project_root,
)
from cloudness_detection.losses.filtered_jaccard import FilteredJaccardLoss
from cloudness_detection.losses.soft_jaccard import DiceBCELoss, SoftJaccardLoss
from cloudness_detection.metrics import dice_binary, iou_binary, mae_cloud_percent
from cloudness_detection.models.build import build_model
from cloudness_detection.seed import DEFAULT_SEED, set_global_seed

ROOT = resolve_project_root()


def worker_init_fn(worker_id: int) -> None:
    """Сид каждого DataLoader worker.

    Args:
        worker_id: Номер worker.
    """
    worker_seed = DEFAULT_SEED + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def maybe_clearml(project: str, task_name: str, config: dict) -> object | None:
    """Опциональный ClearML: включается если CLEARML_ENABLE=1.

    Args:
        project: Имя проекта.
        task_name: Имя задачи.
        config: Параметры рана.

    Returns:
        Task или None.
    """
    if os.environ.get("CLEARML_ENABLE", "").strip() not in {"1", "true", "yes"}:
        return None
    try:
        from clearml import Task  # type: ignore
    except ImportError:
        print("CLEARML_ENABLE=1, но clearml не установлен — пропускаем", flush=True)
        return None
    task = Task.init(project_name=project, task_name=task_name)
    task.connect(config)
    return task


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """Dice/IoU/MAE/reject@75 (+ опц. loss).

    Args:
        model: Модель.
        loader: Val loader.
        device: cuda/cpu.
        criterion: Loss.
    """
    model.eval()
    dices: list[float] = []
    ious: list[float] = []
    pred_pcts: list[float] = []
    true_pcts: list[float] = []
    losses: list[float] = []
    hits = 0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        if criterion is not None:
            losses.append(float(criterion(logits, masks).item()))
        preds = (torch.sigmoid(logits) >= 0.5).float()
        for i in range(preds.shape[0]):
            p = preds[i, 0].cpu().numpy()
            t = masks[i, 0].cpu().numpy()
            dices.append(dice_binary(p, t))
            ious.append(iou_binary(p, t))
            pp = float(p.mean() * 100.0)
            tp = float(t.mean() * 100.0)
            pred_pcts.append(pp)
            true_pcts.append(tp)
            hits += int((pp > 75.0) == (tp > 75.0))
            n += 1
    out = {
        "dice": float(np.mean(dices)) if dices else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "mae_pct": mae_cloud_percent(np.asarray(pred_pcts), np.asarray(true_pcts))
        if pred_pcts
        else 0.0,
        "reject75_acc": float(hits / max(n, 1)),
    }
    if losses:
        out["loss"] = float(np.mean(losses))
    return out


def append_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    """Дописывает CSV-строку.

    Args:
        path: Файл.
        row: Значения.
        fieldnames: Колонки.
    """
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_curves(history: list[dict[str, float]], out_dir: Path) -> None:
    """Кривые train/val для слайда.

    Args:
        history: Метрики по эпохам.
        out_dir: Папка.
    """
    epochs = [int(r["epoch"]) for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0, 0].plot(epochs, [r["val_loss"] for r in history], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(epochs, [r["val_iou"] for r in history], label="iou")
    axes[0, 1].plot(epochs, [r["val_dice"] for r in history], label="dice")
    axes[0, 1].set_title("Seg")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(epochs, [r["val_mae_pct"] for r in history], color="#c44e52")
    axes[1, 0].set_title("MAE %")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(epochs, [r["val_reject75_acc"] for r in history], color="#5b8c5a")
    axes[1, 1].set_title("reject@75 acc")
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle("train_mix curves")
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description="Train cloud seg on weighted dataset mix")
    p.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "segformer_b2", "cloudnet", "cloudnet_plus", "cloud_adapter"],
    )
    p.add_argument(
        "--loss",
        type=str,
        default="auto",
        choices=["auto", "dice_bce", "soft_jaccard", "fjl1"],
        help="auto: Cloud-Net → soft Jaccard, Cloud-Net+ → FJL1, иначе Dice+BCE",
    )
    p.add_argument("--device-plan", type=str, default="4060", choices=sorted(DEVICE_PLANS))
    p.add_argument(
        "--datasets",
        type=str,
        default="hrc,sparcs,cloud38,gf1,s2",
        help="Через запятую; первый — val (hrc → test). Кейс: hrc,sparcs,cloud38,gf1,s2",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=0, help="0 = из device-plan")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--image-size", type=int, default=0, help="0 = из device-plan")
    p.add_argument("--max-items", type=int, default=0, help="Лимит на каждый сет (0=все)")
    p.add_argument("--epoch-size", type=int, default=0, help="Сэмплов на эпоху микса (0=auto)")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--log-every-batches", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--jaccard-weight", type=float, default=0.0)
    p.add_argument("--encoder-weights", type=str, default="imagenet")
    p.add_argument(
        "--color-match-photos",
        action="store_true",
        help="Match публичных RGB (HRC/SPARCS/38/…) к μ/σ Photos; Photos сами не трогаем",
    )
    p.add_argument(
        "--color-match-method",
        type=str,
        default="meanstd",
        choices=["meanstd", "reinhard"],
        help="Метод color-match (нужен --color-match-photos)",
    )
    p.add_argument(
        "--dataset-weights",
        type=str,
        default="",
        help="Переопределение весов микса, напр. hrc=5,sparcs=2.5,cloud38=0.15",
    )
    p.add_argument(
        "--cubesat-augs",
        action="store_true",
        help="JPEG/даунскейл/контраст — имитация домена Photos на публичных RGB",
    )
    p.add_argument("--smoke", action="store_true", help="1 эпоха, мало сэмплов")
    return p.parse_args()


def parse_dataset_weights(raw: str) -> dict[str, float] | None:
    """Парсит ``hrc=5,cloud38=0.15`` → dict.

    Args:
        raw: Строка CLI; пустая → None.

    Returns:
        Словарь весов или None.

    Raises:
        ValueError: Битый формат.
    """
    text = (raw or "").strip()
    if not text:
        return None
    out: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Вес без '=': {part!r} (ожидали name=float)")
        name, value = part.split("=", 1)
        name = name.strip()
        out[name] = float(value.strip())
    return out or None


def resolve_loss(name: str, model: str, jaccard_weight: float) -> tuple[str, nn.Module]:
    """Имя loss → модуль. ``auto`` зависит от модели.

    Args:
        name: ``auto`` | ``dice_bce`` | ``soft_jaccard`` | ``fjl1``.
        model: Короткое имя модели.
        jaccard_weight: Добавка soft Jaccard к Dice+BCE.

    Returns:
        (фактическое имя, criterion).
    """
    key = name
    if key == "auto":
        if model == "cloudnet":
            key = "soft_jaccard"
        elif model == "cloudnet_plus":
            key = "fjl1"
        else:
            key = "dice_bce"
    if key == "dice_bce":
        return key, DiceBCELoss(jaccard_weight=jaccard_weight)
    if key == "soft_jaccard":
        return key, SoftJaccardLoss()
    if key == "fjl1":
        return key, FilteredJaccardLoss()
    raise ValueError(f"Неизвестный loss {name!r}")


def main() -> None:
    """Точка входа train-mix."""
    args = parse_args()
    if args.model == "cloud_adapter":
        raise SystemExit(
            "cloud_adapter → docs/KAGGLE_T4_CLOUD_ADAPTER.md (не локальный SMP-путь)"
        )

    plan = DEVICE_PLANS[args.device_plan]
    if args.smoke:
        args.epochs = 1
        args.max_items = args.max_items or 24
        args.epoch_size = args.epoch_size or 48
        args.batch_size = args.batch_size or min(4, plan["batch_size"])

    image_size = args.image_size or int(plan["image_size"])
    batch_size = args.batch_size or int(plan["batch_size"])
    if args.model == "segformer_b2" and batch_size > 4 and args.device_plan == "4060":
        batch_size = 4
        print("segformer_b2 на 4060: batch_size → 4", flush=True)
    if args.model in {"cloudnet", "cloudnet_plus"} and batch_size > 4 and args.device_plan == "4060":
        batch_size = 4
        print(f"{args.model} на 4060: batch_size → 4 (пик ~5.6 GB на 384²)", flush=True)

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = resolve_project_root()
    print(f"device={device} plan={args.device_plan} model={args.model}", flush=True)
    print(f"project_root={project_root}", flush=True)

    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    max_items = None if args.max_items <= 0 else args.max_items
    epoch_size = None if args.epoch_size <= 0 else args.epoch_size
    color_match = args.color_match_method if args.color_match_photos else None
    weight_overrides = parse_dataset_weights(args.dataset_weights)
    effective_weights = {
        name: float((weight_overrides or {}).get(name, DEFAULT_WEIGHTS.get(name, 1.0)))
        for name in names
    }
    print(f"mix weights: {effective_weights}", flush=True)
    for _probe in names:
        print(f"  data[{_probe}]={resolve_data_root(_probe, project_root)}", flush=True)
    if color_match:
        print(
            f"color-match → Photos: method={color_match} "
            "(stats: outputs/eda/deep/photos_ref_stats.json)",
            flush=True,
        )
    if weight_overrides:
        print(f"dataset-weights override: {weight_overrides}", flush=True)
    if args.cubesat_augs:
        enable_cubesat_augs(True)
        print("cubesat-augs: ON (JPEG / downscale / contrast)", flush=True)

    mix = build_datasets(
        names,
        image_size=image_size,
        augment=True,
        max_items=max_items,
        seed=args.seed,
        project_root=project_root,
        epoch_size=epoch_size,
        color_match=color_match,
        weights=effective_weights,
    )
    # Val: holdout первого датасета (для hrc — split=test) без аугов.
    val_name = names[0]
    val_ds = build_one(
        val_name,
        image_size=image_size,
        augment=False,
        max_items=max_items,
        seed=args.seed + 1,
        project_root=project_root,
        split="test" if val_name == "hrc" else "train",
        color_match=color_match,
    )

    if len(val_ds) < 4:
        raise RuntimeError(
            f"Val-сет {val_name!r} слишком мал: len={len(val_ds)}. "
            "Нужен holdout ≥4 (для hrc — split=test)."
        )
    from torch.utils.data import Subset

    val_len = min(len(val_ds), max(4, int(len(mix) * args.val_ratio)))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(val_ds), generator=g)[:val_len].tolist()
    val_subset: DatasetLike = Subset(val_ds, perm)
    train_ds: DatasetLike = mix

    print(
        f"datasets={names} mix_len={len(mix)} train≈{len(train_ds)} val={len(val_subset)} "
        f"img={image_size} bs={batch_size}",
        flush=True,
    )

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        generator=g,
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
    )

    weights = None if args.encoder_weights.lower() in {"none", "", "null"} else args.encoder_weights
    if args.model in {"cloudnet", "cloudnet_plus"}:
        weights = None
    model = build_model(args.model, encoder_weights=weights).to(device)
    loss_name, criterion = resolve_loss(args.loss, args.model, args.jaccard_weight)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loss={loss_name} params={n_params / 1e6:.2f}M", flush=True)
    if args.model in {"cloudnet", "cloudnet_plus"}:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_name = args.run_name or (
        f"{'smoke_' if args.smoke else ''}{args.model}_{'+'.join(names)}_seed{args.seed}"
    )
    out_dir = project_root / "outputs" / "train" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": args.model,
        "device_plan": args.device_plan,
        "datasets": names,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "lr": args.lr,
        "image_size": image_size,
        "max_items": args.max_items,
        "seed": args.seed,
        "loss": loss_name,
        "jaccard_weight": args.jaccard_weight,
        "encoder_weights": weights,
        "params_m": round(n_params / 1e6, 2),
        "init": "kaiming_rgb3" if args.model in {"cloudnet", "cloudnet_plus"} else "smp",
        "device": str(device),
        "smoke": args.smoke,
        "color_match_photos": bool(args.color_match_photos),
        "color_match_method": color_match,
        "cubesat_augs": bool(args.cubesat_augs),
        "dataset_weights": effective_weights,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    maybe_clearml("cloudness-detection", run_name, config)

    history: list[dict[str, float]] = []
    best_iou = -1.0
    bad = 0
    epoch_fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_dice",
        "val_iou",
        "val_mae_pct",
        "val_reject75_acc",
        "lr",
        "epoch_sec",
        "best",
    ]
    batch_fields = ["epoch", "step", "global_step", "loss"]
    for name in ("history.csv", "batch_losses.csv", "history.json"):
        p = out_dir / name
        if p.exists():
            p.unlink()

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            loss_v = float(loss.item())
            losses.append(loss_v)
            global_step += 1
            if step == 1 or step % args.log_every_batches == 0:
                append_csv(
                    out_dir / "batch_losses.csv",
                    {
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "loss": f"{loss_v:.6f}",
                    },
                    batch_fields,
                )

        val = evaluate(model, val_loader, device, criterion=criterion)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "val_loss": float(val.get("loss", 0.0)),
            "val_dice": float(val["dice"]),
            "val_iou": float(val["iou"]),
            "val_mae_pct": float(val["mae_pct"]),
            "val_reject75_acc": float(val["reject75_acc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_sec": float(time.time() - t0),
        }
        is_best = row["val_iou"] > best_iou
        history.append(row)
        append_csv(
            out_dir / "history.csv",
            {**{k: row[k] for k in epoch_fields if k != "best"}, "best": int(is_best)},
            epoch_fields,
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_curves(history, out_dir)
        print(
            f"epoch {epoch}/{args.epochs} loss={row['train_loss']:.4f} "
            f"val_iou={row['val_iou']:.3f} dice={row['val_dice']:.3f} "
            f"mae%={row['val_mae_pct']:.1f} rej={row['val_reject75_acc']:.3f} "
            f"t={row['epoch_sec']:.0f}s" + (" BEST" if is_best else ""),
            flush=True,
        )
        if is_best:
            best_iou = row["val_iou"]
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "image_size": image_size,
                    "seed": args.seed,
                    "epoch": epoch,
                    "metrics": {
                        "val_iou": row["val_iou"],
                        "val_dice": row["val_dice"],
                        "val_mae_pct": row["val_mae_pct"],
                        "val_reject75_acc": row["val_reject75_acc"],
                    },
                    "config": config,
                },
                out_dir / "best.pt",
            )
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop after {args.patience} bad epochs", flush=True)
                break

    (out_dir / "SMOKE_OK.txt" if args.smoke else out_dir / "DONE.txt").write_text(
        f"best_iou={best_iou:.4f}\noutdir={out_dir}\n",
        encoding="utf-8",
    )
    print(f"Готово → {out_dir}", flush=True)


# Для type hints без циклического импорта.
DatasetLike = object


if __name__ == "__main__":
    main()
