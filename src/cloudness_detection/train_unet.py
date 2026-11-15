"""Обучение U-Net (RGB) на 38-Cloud. Сиды зафиксированы, логи под графики."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from cloudness_detection.dataset_38cloud import Cloud38RGBDataset
from cloudness_detection.metrics import dice_binary, iou_binary, mae_cloud_percent
from cloudness_detection.seed import DEFAULT_SEED, set_global_seed

ROOT = Path(__file__).resolve().parents[2]


class TorchCloudDataset(Dataset):
    """Обёртка NumPy-датасета в torch Dataset."""

    def __init__(self, base: Cloud38RGBDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.base[index]
        return {
            "image": torch.from_numpy(item["image"]),
            "mask": torch.from_numpy(item["mask"]),
            "name": item["name"],
            "cloud_frac": torch.tensor(item["cloud_frac"], dtype=torch.float32),
        }


def worker_init_fn(worker_id: int) -> None:
    """Фиксирует сид каждого worker DataLoader.

    Args:
        worker_id: Номер worker.
    """
    worker_seed = DEFAULT_SEED + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class DiceBCELoss(nn.Module):
    """BCE + soft Dice для бинарной сегментации."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: Bx1xHxW без sigmoid.
            target: Bx1xHxW в 0..1.
        """
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)
        return bce + dice.mean()


def build_model(encoder: str = "resnet34", encoder_weights: str | None = None) -> nn.Module:
    """Собирает U-Net из segmentation_models_pytorch.

    Args:
        encoder: Имя энкодера SMP (например ``resnet34``).
        encoder_weights: ``None`` или ``\"imagenet\"``.
    """
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """Считает loss (опц.) + Dice/IoU/MAE/reject@75 на лоадере.

    Args:
        model: Модель в eval.
        loader: DataLoader.
        device: cuda/cpu.
        criterion: Если задан — средний loss по батчам.
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
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

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
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "mae_pct": mae_cloud_percent(np.asarray(pred_pcts), np.asarray(true_pcts)),
        "reject75_acc": float(hits / max(n, 1)),
    }
    if losses:
        out["loss"] = float(np.mean(losses))
    return out


def append_csv(path: Path, row: dict[str, float | int | str], fieldnames: list[str]) -> None:
    """Дописывает строку в CSV (создаёт файл с заголовком при первом вызове).

    Args:
        path: Путь к CSV.
        row: Словарь значений.
        fieldnames: Порядок колонок.
    """
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_curves(history: list[dict[str, float]], out_dir: Path) -> None:
    """Рисует train/val loss, IoU/Dice, MAE, reject@75 для презентации.

    Args:
        history: Список метрик по эпохам.
        out_dir: Куда сохранить PNG.
    """
    epochs = [int(r["epoch"]) for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(epochs, [r["train_loss"] for r in history], label="train_loss")
    axes[0, 0].plot(epochs, [r["val_loss"] for r in history], label="val_loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, [r["val_iou"] for r in history], label="val_iou")
    axes[0, 1].plot(epochs, [r["val_dice"] for r in history], label="val_dice")
    axes[0, 1].set_title("Segmentation")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, [r["val_mae_pct"] for r in history], color="#c44e52")
    axes[1, 0].set_title("MAE cloud %")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, [r["val_reject75_acc"] for r in history], color="#5b8c5a")
    axes[1, 1].set_title("Keep/delete @75% accuracy")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Training curves (для слайда «модель сошлась»)")
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=160)
    plt.close(fig)


def write_run_summary(history: list[dict[str, float]], out_dir: Path, config: dict) -> None:
    """Короткий markdown: лучшая эпоха и признак сходимости.

    Args:
        history: Метрики по эпохам.
        out_dir: Папка рана.
        config: Конфиг обучения.
    """
    best = max(history, key=lambda r: r["val_iou"])
    last = history[-1]
    # Грубая эвристика: за последние 3 эпохи val_iou почти не растёт.
    tail = history[-3:] if len(history) >= 3 else history
    iou_gain = float(tail[-1]["val_iou"] - tail[0]["val_iou"])
    lines = [
        "# Train summary",
        "",
        f"- Config: `{json.dumps(config, ensure_ascii=False)}`",
        f"- Epochs run: **{len(history)}**",
        f"- Best val IoU: **{best['val_iou']:.4f}** (epoch {int(best['epoch'])})",
        f"- Best val Dice: **{best['val_dice']:.4f}**",
        f"- Best reject@75 acc: **{best['val_reject75_acc']:.4f}**",
        f"- Last val IoU: **{last['val_iou']:.4f}**",
        f"- IoU gain over last {len(tail)} epochs: **{iou_gain:+.4f}**",
        "",
        "## Сходимость (для слайда)",
        "",
        (
            "- Похоже на плато: прирост IoU за хвост < 0.005 → дальше учить мало смысла без смены сетапа."
            if abs(iou_gain) < 0.005 and len(history) >= 5
            else "- Ещё растёт или рано судить — смотри `curves.png`."
        ),
        "",
        "Файлы: `history.json`, `history.csv`, `batch_losses.csv`, `curves.png`, `best.pt`",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Учит U-Net на 38-Cloud и пишет чекпоинт + полные логи."""
    parser = argparse.ArgumentParser(description="Train U-Net RGB on 38-Cloud")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-patches", type=int, default=0, help="0 = все nonempty")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5, help="early stop по val_iou")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    data_root = ROOT / "data" / "38-cloud"
    max_patches = None if args.max_patches <= 0 else args.max_patches
    base = Cloud38RGBDataset(
        data_root,
        nonempty_only=True,
        image_size=args.image_size,
        max_patches=max_patches,
        seed=args.seed,
    )
    full = TorchCloudDataset(base)
    val_len = max(1, int(len(full) * args.val_ratio))
    train_len = len(full) - val_len
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full, [train_len, val_len], generator=generator)
    print(f"patches={len(full)} train={train_len} val={val_len}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    weights = None if args.encoder_weights.lower() in {"none", "", "null"} else args.encoder_weights
    print(f"encoder={args.encoder} weights={weights}", flush=True)
    model = build_model(args.encoder, encoder_weights=weights).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_name = args.run_name or f"unet_{args.encoder}_{weights or 'scratch'}_seed{args.seed}"
    out_dir = ROOT / "outputs" / "train" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "encoder": args.encoder,
        "encoder_weights": weights,
        "image_size": args.image_size,
        "max_patches": args.max_patches,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "patience": args.patience,
        "device": str(device),
        "train_len": train_len,
        "val_len": val_len,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history: list[dict[str, float]] = []
    best_iou = -1.0
    bad_epochs = 0
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

    # Чистый старт логов этого рана.
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
            "val_loss": float(val["loss"]),
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
            {
                **{k: row[k] for k in epoch_fields if k != "best"},
                "best": int(is_best),
            },
            epoch_fields,
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_curves(history, out_dir)

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"dice={row['val_dice']:.3f} iou={row['val_iou']:.3f} "
            f"mae%={row['val_mae_pct']:.1f} rej_acc={row['val_reject75_acc']:.3f} "
            f"time={row['epoch_sec']:.0f}s"
            + (" BEST" if is_best else ""),
            flush=True,
        )

        if is_best:
            best_iou = row["val_iou"]
            bad_epochs = 0
            ckpt = {
                "model": model.state_dict(),
                "encoder": args.encoder,
                "encoder_weights": weights,
                "image_size": args.image_size,
                "seed": args.seed,
                "epoch": epoch,
                "metrics": {
                    "val_iou": row["val_iou"],
                    "val_dice": row["val_dice"],
                    "val_mae_pct": row["val_mae_pct"],
                    "val_reject75_acc": row["val_reject75_acc"],
                },
                "config": config,
            }
            torch.save(ckpt, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop: no val_iou improve for {args.patience} epochs", flush=True)
                break

    write_run_summary(history, out_dir, config)
    print(f"Готово → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
