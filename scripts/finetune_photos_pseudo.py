"""Part B: короткий finetune U-Net на public mix + soft pseudo-labels Photos-32.

8 CVAT-кадров не входят в train и не используются для выбора порога.
После FT: лёгкий proxy-sweep на 32, затем один eval на Photos-8.

Пример::

    set PYTHONPATH=src
    .venv\\Scripts\\python.exe scripts/finetune_photos_pseudo.py
"""

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
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cloudness_detection.data.common import Sample, load_rgb_u8, make_sample, resize_pair
from cloudness_detection.data.mix import (
    WeightedMixedDataset,
    build_one,
    collate_samples,
    resolve_project_root,
)
from cloudness_detection.losses.soft_jaccard import DiceBCELoss
from cloudness_detection.metrics import dice_binary, iou_binary, mae_cloud_percent
from cloudness_detection.models.build import build_model
from cloudness_detection.seed import DEFAULT_SEED, set_global_seed

# scripts/ на sys.path, чтобы импортировать calibrate_photos_threshold.
import sys as _sys

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPTS_DIR)

from calibrate_photos_threshold import (  # noqa: E402
    PRED_THRESHOLDS,
    aggregate_metrics,
    chw_to_hwc,
    load_checkpoint,
    load_manifest,
    make_triplet,
    metrics_from_masks,
    overlay_mask,
    pearson_corr,
    pick_threshold,
    predict_prob,
    prefix_of_preview,
    save_png_hwc,
    split_photos,
    stem_from_preview,
)

OOM_RETRIES = 8
OOM_SLEEP_SEC = 45.0


class SoftPseudoPhotosDataset(Dataset):
    """Photos-превью + soft pseudo-mask (.npy), без CVAT held-out.

    Attributes:
        pairs: Список (jpg, npy).
    """

    dataset_id = "photos_pseudo"

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        *,
        image_size: int,
        augment: bool,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """Args:
            pairs: Пары (preview_jpg, soft_prob_npy).
            image_size: Сторона квадрата.
            augment: Flip/rot.
            seed: Сид (для совместимости API; ауги через global seed).
        """
        if not pairs:
            raise ValueError("Пустой SoftPseudoPhotosDataset")
        self.pairs = list(pairs)
        self.image_size = image_size
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Sample:
        img_path, soft_path = self.pairs[index]
        if not soft_path.is_file():
            raise FileNotFoundError(f"Нет soft-маски: {soft_path}")
        image = load_rgb_u8(img_path)
        soft = np.load(soft_path).astype(np.float32)
        if soft.ndim != 2:
            raise ValueError(f"Ожидали HxW soft mask, получили {soft.shape} в {soft_path}")
        _, h, w = image.shape
        if soft.shape != (h, w):
            soft = np.asarray(
                Image.fromarray(soft, mode="F").resize((w, h), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        mask = np.clip(soft, 0.0, 1.0)[None, ...]
        return make_sample(
            image,
            mask,
            self.dataset_id,
            str(img_path),
            self.image_size,
            self.augment,
        )


def append_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    """Дописывает CSV-строку."""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_curves(history: list[dict[str, float]], out_dir: Path) -> None:
    """Кривые train/val."""
    epochs = [int(r["epoch"]) for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, [r["val_iou"] for r in history], label="iou")
    axes[1].plot(epochs, [r["val_dice"] for r in history], label="dice")
    axes[1].set_title("Seg (public val)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("photos pseudo FT")
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=140)
    plt.close(fig)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    """Val на публичном holdout (не Photos-8)."""
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
        losses.append(float(criterion(logits, masks).item()))
        preds = (torch.sigmoid(logits) >= 0.5).float()
        for i in range(preds.shape[0]):
            p = preds[i, 0].cpu().numpy()
            t = (masks[i, 0].cpu().numpy() >= 0.5).astype(np.float32)
            dices.append(dice_binary(p, t))
            ious.append(iou_binary(p, t))
            pp = float(p.mean() * 100.0)
            tp = float(t.mean() * 100.0)
            pred_pcts.append(pp)
            true_pcts.append(tp)
            hits += int((pp > 75.0) == (tp > 75.0))
            n += 1
    return {
        "dice": float(np.mean(dices)) if dices else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "mae_pct": mae_cloud_percent(np.asarray(pred_pcts), np.asarray(true_pcts))
        if pred_pcts
        else 0.0,
        "reject75_acc": float(hits / max(n, 1)),
        "loss": float(np.mean(losses)) if losses else 0.0,
    }


def gpu_mem_used_mib() -> tuple[int, int] | None:
    """(used, total) MiB через nvidia-smi, иначе None."""
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    used_s, total_s = out.splitlines()[0].split(",")
    return int(used_s.strip()), int(total_s.strip())


def wait_for_gpu_headroom(
    *,
    need_free_mib: int,
    poll_sec: float,
    max_wait_sec: float,
) -> dict:
    """Ждёт свободную VRAM, не трогая чужой train.

    Args:
        need_free_mib: Сколько свободных MiB хотим.
        poll_sec: Интервал опроса.
        max_wait_sec: Максимум ожидания.

    Returns:
        Словарь со статусом ожидания.

    Raises:
        RuntimeError: Не дождались.
    """
    t0 = time.time()
    notes: list[str] = []
    while True:
        mem = gpu_mem_used_mib()
        if mem is None:
            notes.append("nvidia-smi недоступен — продолжаем без wait")
            return {"waited_sec": 0.0, "notes": notes, "skipped": True}
        used, total = mem
        free = total - used
        elapsed = time.time() - t0
        print(
            f"GPU mem used={used}/{total} MiB free={free} (need>={need_free_mib}), "
            f"waited={elapsed:.0f}s",
            flush=True,
        )
        if free >= need_free_mib:
            notes.append(f"Свободно {free} MiB после {elapsed:.0f}s")
            return {
                "waited_sec": elapsed,
                "notes": notes,
                "used_mib": used,
                "total_mib": total,
                "free_mib": free,
                "skipped": False,
            }
        if elapsed >= max_wait_sec:
            raise RuntimeError(
                f"GPU не освободился за {max_wait_sec:.0f}s "
                f"(free={free} < {need_free_mib}). Не убиваем чужой train."
            )
        time.sleep(poll_sec)


def write_report_md(
    report_path: Path,
    *,
    part_a: dict,
    part_b: dict,
    metrics: dict,
) -> None:
    """Пишет REPORT.md на русском."""
    a_hold = part_a["heldout_eval"]
    base = a_hold.get("thr_0p50") or next(
        v for k, v in a_hold.items() if abs(float(v["threshold"]) - 0.5) < 1e-9
    )
    a_key = f"thr_{float(part_a['chosen_threshold']):.2f}".replace(".", "p")
    a_res = a_hold[a_key]
    b_agg = part_b["heldout_eval"]["aggregate"]
    lines: list[str] = []
    lines.append("# Photos: калибровка порога + pseudo-label finetune")
    lines.append("")
    lines.append("## Анти-leakage")
    lines.append("")
    lines.append(
        "8 кадров с CVAT-масками (`01,02,03,04,05,06,08,11`) **не** участвовали "
        "в выборе порога и **не** входили в finetune. Они только в одноразовом "
        "финальном сравнении baseline@0.5 / A / B."
    )
    lines.append("")
    lines.append(f"- Resolution: {part_a['resolution_note']}")
    lines.append(f"- Baseline ckpt: `{part_a['checkpoint']}`")
    lines.append("")
    lines.append("## Part A — калибровка порога (без train)")
    lines.append("")
    lines.append(f"**Правило выбора thr:** {part_a['selection_rule']}")
    lines.append("")
    lines.append(f"**Выбранный thr:** **{part_a['chosen_threshold']:.2f}**")
    lines.append("")
    lines.append("| thr | mean pred % | MAE vs proxy | Pearson r |")
    lines.append("|----:|------------:|-------------:|----------:|")
    for row in part_a["proxy_sweep"]:
        mark = " ←" if abs(row["threshold"] - part_a["chosen_threshold"]) < 1e-9 else ""
        lines.append(
            f"| {row['threshold']:.2f} | {row['mean_pred_cloud_pct']:.1f} | "
            f"{row['mae_vs_proxy']:.2f} | {row['pearson_vs_proxy']:.3f}{mark} |"
        )
    lines.append("")
    lines.append("## Part B — soft pseudo-label finetune")
    lines.append("")
    lines.append(
        f"- Pseudo: soft probs baseline на 32 unlabeled (не binary@0.35; "
        f"thr={part_b['pseudo_doc_thr']:.2f} только для справки binary-сравнения)."
    )
    lines.append(
        f"- FT: lr={part_b['config']['lr']}, epochs={part_b['config']['epochs']}, "
        f"bs={part_b['config']['batch_size']}, seed={part_b['config']['seed']}, "
        f"mix={part_b['config']['datasets']}, weights={part_b['config']['weights']}"
    )
    lines.append(f"- Best: `{part_b['best_ckpt']}` (val IoU public={part_b['best_val_iou']:.4f})")
    lines.append(f"- Thr после FT: **{part_b['chosen_threshold']:.2f}** ({part_b['thr_rule']})")
    lines.append(f"- Colornorm/GPU: {part_b['gpu_note']}")
    lines.append("")
    lines.append("## Метрики Photos-8 (одно сравнение)")
    lines.append("")
    lines.append("| вариант | thr | IoU | Dice | MAE % | reject@75 |")
    lines.append("|--------|----:|----:|-----:|------:|----------:|")

    def _row(name: str, thr: float, agg: dict) -> str:
        return (
            f"| {name} | {thr:.2f} | {agg['iou']:.3f} | {agg['dice']:.3f} | "
            f"{agg['mae_pct']:.1f} | {agg['reject75_acc']:.3f} |"
        )

    lines.append(_row("baseline", 0.5, base["aggregate"]))
    lines.append(_row("A calibrated", float(part_a["chosen_threshold"]), a_res["aggregate"]))
    lines.append(_row("B pseudo-FT", float(part_b["chosen_threshold"]), b_agg))
    lines.append("")
    lines.append("### Per-image (B)")
    lines.append("")
    lines.append("| файл | IoU | Dice | GT % | pred % | MAE % |")
    lines.append("|------|----:|-----:|-----:|-------:|------:|")
    for row in part_b["heldout_eval"]["per_image"]:
        lines.append(
            f"| `{row['name']}` | {row['iou']:.3f} | {row['dice']:.3f} | "
            f"{row['cloud_frac_gt_pct']:.1f} | {row['cloud_frac_pred_pct']:.1f} | "
            f"{row['mae_pct']:.1f} |"
        )
    lines.append("")
    lines.append("## Overlays (8 held-out)")
    lines.append("")
    lines.append("Каждый кадр: **RGB | GT | pred**. Папки:")
    lines.append("")
    lines.append("- `baseline_0.5/`")
    lines.append("- `A_calibrated/`")
    lines.append("- `B_pseudo_ft/`")
    lines.append("")
    for row in part_b["heldout_eval"]["per_image"]:
        stem = row["stem"]
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append(f"![B {stem}](B_pseudo_ft/{stem}_triplet.png)")
        lines.append("")
        lines.append(
            f"[baseline](baseline_0.5/{stem}_triplet.png) · "
            f"[A](A_calibrated/{stem}_triplet.png) · "
            f"[B](B_pseudo_ft/{stem}_triplet.png)"
        )
        lines.append("")
    lines.append("## Как воспроизвести")
    lines.append("")
    lines.append("```text")
    lines.append("set PYTHONPATH=src")
    lines.append(".venv\\Scripts\\python.exe scripts\\calibrate_photos_threshold.py")
    lines.append(".venv\\Scripts\\python.exe scripts\\finetune_photos_pseudo.py")
    lines.append("```")
    lines.append("")
    lines.append("## Вывод")
    lines.append("")
    base_iou = float(base["aggregate"]["iou"])
    b_iou = float(b_agg["iou"])
    lines.append(
        f"A thr={part_a['chosen_threshold']:.2f} "
        f"(proxy на 32; часто совпадает с 0.5, если proxy занижает %). "
        f"B vs baseline IoU: {b_iou:.3f} vs {base_iou:.3f}."
    )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description="Pseudo-label finetune on Photos-32")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/train/unet_mix_hrc_sparcs_38_4060"),
    )
    p.add_argument(
        "--out-train-dir",
        type=Path,
        default=Path("outputs/train/unet_photos_pseudo_ft"),
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path("outputs/reports/photos_thr_and_pseudo_ft"),
    )
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epoch-size", type=int, default=640)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pseudo-bin-thr", type=float, default=0.35)
    p.add_argument("--need-free-mib", type=int, default=3500)
    p.add_argument("--max-wait-sec", type=float, default=7200.0)
    p.add_argument("--skip-train", action="store_true", help="Только eval из best.pt")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    """Точка входа Part B + финальный отчёт."""
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

    part_a_path = report_dir / "part_a.json"
    if not part_a_path.is_file():
        raise FileNotFoundError(
            f"Сначала запусти calibrate_photos_threshold.py → нет {part_a_path}"
        )
    part_a = json.loads(part_a_path.read_text(encoding="utf-8"))

    photos_root = project_root / "labeling" / "photos_40"
    images_dir = photos_root / "images"
    masks_dir = photos_root / "masks"
    labeled, unlabeled = split_photos(images_dir, masks_dir)
    if len(unlabeled) != 32:
        raise RuntimeError(f"Ожидали 32 unlabeled, получили {len(unlabeled)}")
    held_prefixes = {prefix_of_preview(p) for p in labeled}
    for jpg in unlabeled:
        if prefix_of_preview(jpg) in held_prefixes:
            raise RuntimeError(f"Leakage: unlabeled пересекается с held-out: {jpg.name}")

    config_base = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    image_size = int(config_base["image_size"])

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Запрошен cuda, но CUDA недоступна")
    device = torch.device(args.device)

    gpu_info = {"notes": ["train skipped"], "waited_sec": 0.0}
    if not args.skip_train and device.type == "cuda":
        print("Ждём свободную GPU (не убиваем colornorm)…", flush=True)
        gpu_info = wait_for_gpu_headroom(
            need_free_mib=args.need_free_mib,
            poll_sec=60.0,
            max_wait_sec=args.max_wait_sec,
        )

    print(f"Load baseline → {device}", flush=True)
    model, ckpt = load_checkpoint(run_dir / "best.pt", device)

    # --- soft pseudo labels for 32 ---
    soft_dir = out_train / "pseudo_soft"
    soft_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = out_train / "pseudo_bin_035"
    bin_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, Path]] = []
    print("Пишем soft pseudo для unlabeled-32…", flush=True)
    for i, jpg in enumerate(unlabeled, start=1):
        soft_path = soft_dir / f"{jpg.stem}.npy"
        bin_path = bin_dir / f"{jpg.stem}.png"
        if soft_path.is_file():
            prob = np.load(soft_path).astype(np.float32)
        else:
            image_chw = load_rgb_u8(jpg)
            prob = predict_prob(model, image_chw, image_size, device)
            np.save(soft_path, prob)
        hard = ((prob >= args.pseudo_bin_thr).astype(np.uint8) * 255)
        Image.fromarray(hard, mode="L").save(bin_path)
        pairs.append((jpg, soft_path))
        print(f"  [{i}/32] {jpg.name} soft_mean={float(prob.mean()):.3f}", flush=True)

    weights = {"hrc": 3.0, "sparcs": 2.5, "cloud38": 0.5, "photos_pseudo": 2.5}
    ft_config = {
        "model": "unet",
        "init_checkpoint": str(run_dir / "best.pt"),
        "datasets": ["hrc", "sparcs", "cloud38", "photos_pseudo"],
        "weights": weights,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "image_size": image_size,
        "epoch_size": args.epoch_size,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "pseudo": "soft_probs",
        "pseudo_bin_thr_for_viz": args.pseudo_bin_thr,
        "held_out_excluded": [p.name for p in labeled],
        "device": str(device),
    }
    (out_train / "config.json").write_text(
        json.dumps(ft_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_path = out_train / "best.pt"
    best_iou = -1.0
    history: list[dict[str, float]] = []

    if not args.skip_train:
        public_sets = []
        public_w = []
        for name in ("hrc", "sparcs", "cloud38"):
            ds = build_one(
                name,
                image_size=image_size,
                augment=True,
                max_items=None,
                seed=args.seed,
                project_root=project_root,
                split="train",
                color_match=None,
            )
            public_sets.append(ds)
            public_w.append(float(weights[name]))
        pseudo_ds = SoftPseudoPhotosDataset(
            pairs,
            image_size=image_size,
            augment=True,
            seed=args.seed,
        )
        train_mix = WeightedMixedDataset(
            public_sets + [pseudo_ds],
            public_w + [float(weights["photos_pseudo"])],
            seed=args.seed,
            epoch_size=args.epoch_size,
        )

        # Val: HRC test holdout — как в train-mix, без Photos-8.
        val_ds = build_one(
            "hrc",
            image_size=image_size,
            augment=False,
            max_items=None,
            seed=args.seed + 1,
            project_root=project_root,
            split="test",
            color_match=None,
        )
        from torch.utils.data import Subset

        val_len = min(len(val_ds), max(16, int(args.epoch_size * 0.1)))
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(val_ds), generator=g)[:val_len].tolist()
        val_subset = Subset(val_ds, perm)

        g_loader = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_mix,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            generator=g_loader,
            collate_fn=collate_samples,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_samples,
        )

        criterion = DiceBCELoss(jaccard_weight=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        model.train()

        for name in ("history.csv", "batch_losses.csv", "history.json"):
            p = out_train / name
            if p.exists():
                p.unlink()

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
        print(
            f"FT start: epochs={args.epochs} lr={args.lr} bs={args.batch_size} "
            f"epoch_size={args.epoch_size} pseudo_n={len(pseudo_ds)}",
            flush=True,
        )
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            losses: list[float] = []
            for step, batch in enumerate(train_loader, start=1):
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
                            f"CUDA OOM train step (попытка {attempt}/{OOM_RETRIES}), "
                            f"ждём {OOM_SLEEP_SEC:.0f}s…",
                            flush=True,
                        )
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        time.sleep(OOM_SLEEP_SEC)
                else:
                    raise RuntimeError("Train step: исчерпаны OOM-ретраи")
                loss_v = float(loss.item())
                losses.append(loss_v)
                global_step += 1
                if step == 1 or step % 20 == 0:
                    append_csv(
                        out_train / "batch_losses.csv",
                        {
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "loss": f"{loss_v:.6f}",
                        },
                        ["epoch", "step", "global_step", "loss"],
                    )
            val = evaluate(model, val_loader, device, criterion)
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
                out_train / "history.csv",
                {**{k: row[k] for k in epoch_fields if k != "best"}, "best": int(is_best)},
                epoch_fields,
            )
            (out_train / "history.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )
            plot_curves(history, out_train)
            print(
                f"epoch {epoch}/{args.epochs} loss={row['train_loss']:.4f} "
                f"val_iou={row['val_iou']:.3f} dice={row['val_dice']:.3f} "
                f"t={row['epoch_sec']:.0f}s" + (" BEST" if is_best else ""),
                flush=True,
            )
            if is_best:
                best_iou = row["val_iou"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_name": "unet",
                        "image_size": image_size,
                        "seed": args.seed,
                        "epoch": epoch,
                        "metrics": {
                            "val_iou": row["val_iou"],
                            "val_dice": row["val_dice"],
                            "val_mae_pct": row["val_mae_pct"],
                            "val_reject75_acc": row["val_reject75_acc"],
                        },
                        "config": ft_config,
                    },
                    best_path,
                )
        (out_train / "DONE.txt").write_text(
            f"best_iou={best_iou:.4f}\noutdir={out_train}\n",
            encoding="utf-8",
        )
    else:
        if not best_path.is_file():
            raise FileNotFoundError(f"--skip-train, но нет {best_path}")
        best_iou = float((torch.load(best_path, map_location="cpu", weights_only=False)
                          .get("metrics") or {}).get("val_iou", -1.0))

    # Reload best for eval
    model, ckpt_ft = load_checkpoint(best_path, device)
    best_iou = float((ckpt_ft.get("metrics") or {}).get("val_iou", best_iou))

    # Proxy thr sweep on unlabeled-32 (no looking at 8 for selection)
    manifest = load_manifest(photos_root / "manifest.csv")
    proxy_arr = np.asarray(
        [float(manifest[stem_from_preview(j)]["white_frac_proxy"]) * 100.0 for j in unlabeled],
        dtype=np.float64,
    )
    probs_list: list[np.ndarray] = []
    print("Proxy sweep после FT на unlabeled-32…", flush=True)
    for jpg in unlabeled:
        probs_list.append(predict_prob(model, load_rgb_u8(jpg), image_size, device))
    sweep_rows: list[dict] = []
    for thr in PRED_THRESHOLDS:
        pred_pcts = np.asarray([(p >= thr).mean() * 100.0 for p in probs_list], dtype=np.float64)
        sweep_rows.append(
            {
                "threshold": float(thr),
                "mean_pred_cloud_pct": float(pred_pcts.mean()),
                "mae_vs_proxy": float(mae_cloud_percent(pred_pcts, proxy_arr)),
                "pearson_vs_proxy": float(pearson_corr(pred_pcts, proxy_arr)),
            }
        )
    chosen = pick_threshold(sweep_rows)
    # Также сравниваем с 0.5 по proxy: если MAE@0.5 не хуже chosen+1.0 — можно оставить 0.5
    row_05 = next(r for r in sweep_rows if abs(r["threshold"] - 0.5) < 1e-9)
    if float(row_05["mae_vs_proxy"]) <= float(chosen["mae_vs_proxy"]) + 1.0:
        # предпочитаем 0.5 если почти так же хорошо (стабильнее)
        if float(row_05["mae_vs_proxy"]) <= float(chosen["mae_vs_proxy"]) + 0.25:
            chosen_thr = 0.5
            thr_rule = (
                "После FT: proxy-sweep на 32; оставили 0.5, т.к. MAE_proxy@0.5 "
                f"={row_05['mae_vs_proxy']:.2f} близко к best "
                f"{chosen['threshold']:.2f}/{chosen['mae_vs_proxy']:.2f} (Δ≤0.25)."
            )
        else:
            chosen_thr = float(chosen["threshold"])
            thr_rule = (
                "После FT: то же proxy-правило что в A (min MAE, tie → max r). "
                f"Выбран {chosen_thr:.2f}."
            )
    else:
        chosen_thr = float(chosen["threshold"])
        thr_rule = (
            "После FT: то же proxy-правило что в A (min MAE, tie → max r). "
            f"Выбран {chosen_thr:.2f}."
        )
    print(f"B thr={chosen_thr:.2f} | {thr_rule}", flush=True)

    # Once eval on 8
    sub = report_dir / "B_pseudo_ft"
    sub.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    print(f"Eval Photos-8 @ thr={chosen_thr:.2f}", flush=True)
    for jpg in labeled:
        mask_path = masks_dir / f"{jpg.stem}.png"
        image_chw = load_rgb_u8(jpg)
        prob = predict_prob(model, image_chw, image_size, device)
        pred = (prob >= chosen_thr).astype(np.float32)
        gt_u8 = np.asarray(Image.open(mask_path))
        if gt_u8.ndim == 3:
            gt_u8 = gt_u8[..., 0]
        gt = (gt_u8 >= 127).astype(np.float32)
        rgb = chw_to_hwc(image_chw)
        stem = jpg.stem
        save_png_hwc(make_triplet(rgb, gt, pred), sub / f"{stem}_triplet.png")
        save_png_hwc(rgb, sub / f"{stem}_rgb.png")
        save_png_hwc(overlay_mask(rgb, gt), sub / f"{stem}_gt.png")
        save_png_hwc(overlay_mask(rgb, pred), sub / f"{stem}_pred.png")
        m = metrics_from_masks(pred, gt)
        rows.append({"name": jpg.name, "stem": stem, **m})
        print(
            f"  {jpg.name} IoU={m['iou']:.3f} Dice={m['dice']:.3f} "
            f"GT={m['cloud_frac_gt_pct']:.1f}% pred={m['cloud_frac_pred_pct']:.1f}%",
            flush=True,
        )
    b_agg = aggregate_metrics(rows)

    waited = float(gpu_info.get("waited_sec", 0.0) or 0.0)
    gpu_note = (
        f"ожидание VRAM {waited:.0f}s; notes={gpu_info.get('notes')}"
        if not gpu_info.get("skipped")
        else f"wait skipped/notes={gpu_info.get('notes')}"
    )

    part_b = {
        "part": "B",
        "best_ckpt": str(best_path),
        "best_val_iou": best_iou,
        "config": ft_config,
        "pseudo_doc_thr": args.pseudo_bin_thr,
        "proxy_sweep": sweep_rows,
        "chosen_threshold": chosen_thr,
        "thr_rule": thr_rule,
        "gpu_note": gpu_note,
        "gpu_wait": gpu_info,
        "history": history,
        "heldout_eval": {
            "threshold": chosen_thr,
            "per_image": rows,
            "aggregate": b_agg,
            "overlays_dir": str(sub),
        },
        "no_leakage_note": (
            "Soft pseudo только на 32 без масок; 8 CVAT не в train и не в выборе thr."
        ),
    }
    (report_dir / "part_b.json").write_text(
        json.dumps(part_b, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Combined metrics.json
    a_hold = part_a["heldout_eval"]
    base = a_hold.get("thr_0p50") or next(
        v for k, v in a_hold.items() if abs(float(v["threshold"]) - 0.5) < 1e-9
    )
    a_key = f"thr_{float(part_a['chosen_threshold']):.2f}".replace(".", "p")
    metrics = {
        "seed": args.seed,
        "baseline_checkpoint": part_a["checkpoint"],
        "resolution_note": part_a["resolution_note"],
        "no_leakage": True,
        "part_a": {
            "chosen_threshold": part_a["chosen_threshold"],
            "selection_rule": part_a["selection_rule"],
            "proxy_sweep": part_a["proxy_sweep"],
            "heldout_aggregate": a_hold[a_key]["aggregate"],
        },
        "baseline_0_5_heldout": base["aggregate"],
        "part_b": {
            "best_ckpt": str(best_path),
            "config": ft_config,
            "chosen_threshold": chosen_thr,
            "thr_rule": thr_rule,
            "proxy_sweep": sweep_rows,
            "heldout_aggregate": b_agg,
            "gpu_note": gpu_note,
        },
        "comparison_photos8": {
            "baseline_0.5": base["aggregate"],
            "A": a_hold[a_key]["aggregate"],
            "B": b_agg,
        },
        "heldout_files": [p.name for p in labeled],
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report_md(report_dir / "REPORT.md", part_a=part_a, part_b=part_b, metrics=metrics)
    (report_dir / "README.md").write_text(
        "# Photos thr + pseudo-FT\n\n"
        "Смотри **[REPORT.md](REPORT.md)**.\n\n"
        "- `metrics.json` — сводные числа\n"
        "- `baseline_0.5/`, `A_calibrated/`, `B_pseudo_ft/` — overlays на 8 held-out\n"
        "- Train FT: `outputs/train/unet_photos_pseudo_ft/`\n",
        encoding="utf-8",
    )
    print(f"Готово → {report_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
