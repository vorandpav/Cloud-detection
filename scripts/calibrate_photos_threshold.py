"""Part A: калибровка порога U-Net на unlabeled Photos-32 по white_frac_proxy.

Держит 8 CVAT-кадров только для одноразового финального eval (не для выбора thr).

Пример::

    set PYTHONPATH=src
    .venv\\Scripts\\python.exe scripts/calibrate_photos_threshold.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cloudness_detection.data.common import load_rgb_u8, resize_pair
from cloudness_detection.data.mix import resolve_project_root
from cloudness_detection.metrics import dice_binary, iou_binary, mae_cloud_percent
from cloudness_detection.models.build import build_model
from cloudness_detection.seed import DEFAULT_SEED, set_global_seed

PRED_THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
OVERLAY_ALPHA = 0.45
OOM_RETRIES = 8
OOM_SLEEP_SEC = 30.0
HELD_OUT_PREFIXES = ("01", "02", "03", "04", "05", "06", "08", "11")


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
    """Склеивает rgb | gt-overlay | pred-overlay по ширине."""
    left = np.clip(rgb, 0.0, 1.0)
    mid = overlay_mask(left, gt)
    right = overlay_mask(left, pred)
    return np.concatenate([left, mid, right], axis=1)


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Грузит best.pt → модель на device.

    Args:
        ckpt_path: Путь к чекпоинту.
        device: cuda/cpu.

    Returns:
        (model.eval(), checkpoint dict).
    """
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Нет чекпоинта: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name") or ckpt.get("config", {}).get("model", "unet")
    cfg = ckpt.get("config") or {}
    enc = cfg.get("encoder_weights")
    if enc is None and "encoder_weights" not in cfg:
        enc = "imagenet"
    model = build_model(model_name, encoder_weights=enc)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_prob(
    model: torch.nn.Module,
    image_chw: np.ndarray,
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    """Инференс → вероятности HxW на исходном размере превью.

    Args:
        model: Модель.
        image_chw: RGB float CxHxW [0,1].
        image_size: Сторона квадрата как на train.
        device: Устройство.

    Returns:
        HxW float32 в [0, 1].
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
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
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

    prob_img = Image.fromarray(prob, mode="F").resize((w0, h0), Image.Resampling.BILINEAR)
    return np.asarray(prob_img, dtype=np.float32)


def load_manifest(manifest_csv: Path) -> dict[str, dict]:
    """stem → строка манифеста.

    Args:
        manifest_csv: labeling/photos_40/manifest.csv.

    Returns:
        Словарь по stem.
    """
    if not manifest_csv.is_file():
        raise FileNotFoundError(f"Нет манифеста: {manifest_csv}")
    out: dict[str, dict] = {}
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = (row.get("stem") or "").strip()
            if not stem:
                raise ValueError(f"Пустой stem в манифесте: {row}")
            out[stem] = row
    return out


def split_photos(
    images_dir: Path,
    masks_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Делит превью на held-out (есть mask) и unlabeled pool.

    Args:
        images_dir: labeling/photos_40/images.
        masks_dir: labeling/photos_40/masks.

    Returns:
        (labeled_jpgs, unlabeled_jpgs), оба отсортированы.
    """
    # На Windows *.jpg и *.JPG дают одни и те же файлы — дедуп по resolve().
    seen: set[Path] = set()
    jpgs: list[Path] = []
    for pattern in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
        for path in images_dir.glob(pattern):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            jpgs.append(path)
    jpgs = sorted(jpgs, key=lambda p: p.name.lower())
    if not jpgs:
        raise FileNotFoundError(f"Нет JPG в {images_dir}")
    labeled: list[Path] = []
    unlabeled: list[Path] = []
    for jpg in jpgs:
        mask_path = masks_dir / f"{jpg.stem}.png"
        if mask_path.is_file():
            labeled.append(jpg)
        else:
            unlabeled.append(jpg)
    return labeled, unlabeled


def stem_from_preview(path: Path) -> str:
    """Имя превью ``NN_STEM.jpg`` → STEM (без числового префикса)."""
    name = path.stem
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return name


def prefix_of_preview(path: Path) -> str:
    """Числовой префикс ``01`` из ``01_….jpg``."""
    return path.stem.split("_", 1)[0]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r; NaN → 0.0 с громким предупреждением не делаем — падаем при const."""
    if x.size < 2:
        raise ValueError("Нужно ≥2 точек для корреляции")
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        raise ValueError("Константный ряд — корреляция не определена")
    return float(np.corrcoef(x, y)[0, 1])


def pick_threshold(rows: list[dict]) -> dict:
    """Выбор thr только по proxy на unlabeled-32.

    Правило:
    1) минимизируем MAE(pred_cloud_pct, white_frac_proxy*100);
    2) при MAE в пределах 0.5 п.п. от лучшего — берём больший Pearson r;
    3) при равном r — меньший thr (чуть агрессивнее по облакам).

    Args:
        rows: Метрики по порогам.

    Returns:
        Строка с выбранным thr.
    """
    if not rows:
        raise ValueError("Пустой sweep порогов")
    best_mae = min(float(r["mae_vs_proxy"]) for r in rows)
    candidates = [r for r in rows if float(r["mae_vs_proxy"]) <= best_mae + 0.5]
    candidates.sort(
        key=lambda r: (-float(r["pearson_vs_proxy"]), float(r["threshold"])),
    )
    return candidates[0]


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
        raise ValueError("Пустой список метрик для aggregate")
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


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description="Calibrate Photos thr on unlabeled-32")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/train/unet_mix_hrc_sparcs_38_4060"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports/photos_thr_and_pseudo_ft"),
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--cache-probs",
        type=Path,
        default=None,
        help="Куда сохранить/загрузить probs unlabeled (npz по умолчанию в out-dir)",
    )
    return p.parse_args()


def main() -> None:
    """Точка входа Part A."""
    args = parse_args()
    project_root = resolve_project_root()
    set_global_seed(args.seed)

    run_dir = (project_root / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir
    out_dir = (project_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    photos_root = project_root / "labeling" / "photos_40"
    images_dir = photos_root / "images"
    masks_dir = photos_root / "masks"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    image_size = int(config["image_size"])

    labeled, unlabeled = split_photos(images_dir, masks_dir)
    labeled_prefixes = {prefix_of_preview(p) for p in labeled}
    expected = set(HELD_OUT_PREFIXES)
    if labeled_prefixes != expected:
        raise RuntimeError(
            f"Held-out префиксы {sorted(labeled_prefixes)} != ожидаемые {sorted(expected)}. "
            "Не калибруем, пока набор 8 не совпадает."
        )
    if len(unlabeled) != 32:
        raise RuntimeError(f"Ожидали 32 unlabeled, получили {len(unlabeled)}")

    with Image.open(unlabeled[0]) as im0:
        preview_wh = im0.size
    resolution_note = (
        f"Инференс на превью labeling/photos_40/images (типично {preview_wh[0]}×{preview_wh[1]}), "
        f"не на полном Photos/. Модель ресайзит к image_size={image_size}, "
        "probs возвращаем bilinear на размер превью."
    )
    print(resolution_note, flush=True)

    manifest = load_manifest(photos_root / "manifest.csv")
    proxies: list[float] = []
    for jpg in unlabeled:
        stem = stem_from_preview(jpg)
        if stem not in manifest:
            raise KeyError(f"Нет stem={stem} в манифесте для {jpg.name}")
        proxies.append(float(manifest[stem]["white_frac_proxy"]) * 100.0)
    proxy_arr = np.asarray(proxies, dtype=np.float64)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Запрошен cuda, но torch.cuda.is_available()=False")
    device = torch.device(args.device)
    print(f"Load baseline → {device}", flush=True)
    model, ckpt = load_checkpoint(run_dir / "best.pt", device)

    cache_path = args.cache_probs
    if cache_path is None:
        cache_path = out_dir / "unlabeled_probs.npz"
    elif not cache_path.is_absolute():
        cache_path = (project_root / cache_path).resolve()

    names: list[str] = []
    stems: list[str] = []
    probs_list: list[np.ndarray] = []

    if cache_path.is_file():
        print(f"Загрузка кэша probs: {cache_path}", flush=True)
        cached = np.load(cache_path, allow_pickle=False)
        names = [str(x) for x in cached["names"].tolist()]
        stems = [str(x) for x in cached["stems"].tolist()]
        for i in range(len(names)):
            probs_list.append(cached[f"prob_{i}"].astype(np.float32))
        if len(names) != len(unlabeled):
            raise RuntimeError(f"Кэш probs n={len(names)} != unlabeled {len(unlabeled)}")
    else:
        print(f"Инференс unlabeled {len(unlabeled)}…", flush=True)
        for i, jpg in enumerate(unlabeled, start=1):
            image_chw = load_rgb_u8(jpg)
            prob = predict_prob(model, image_chw, image_size, device)
            names.append(jpg.name)
            stems.append(stem_from_preview(jpg))
            probs_list.append(prob)
            print(
                f"  [{i}/{len(unlabeled)}] {jpg.name} mean_p={float(prob.mean()):.3f}",
                flush=True,
            )
        save_dict = {
            "names": np.asarray(names),
            "stems": np.asarray(stems),
            "proxy_pct": proxy_arr.astype(np.float64),
        }
        for i, prob in enumerate(probs_list):
            save_dict[f"prob_{i}"] = prob
        np.savez_compressed(cache_path, **save_dict)
        print(f"Кэш probs → {cache_path}", flush=True)

    sweep_rows: list[dict] = []
    for thr in PRED_THRESHOLDS:
        pred_pcts = np.asarray(
            [(p >= thr).mean() * 100.0 for p in probs_list],
            dtype=np.float64,
        )
        mae = mae_cloud_percent(pred_pcts, proxy_arr)
        corr = pearson_corr(pred_pcts, proxy_arr)
        sweep_rows.append(
            {
                "threshold": float(thr),
                "mean_pred_cloud_pct": float(pred_pcts.mean()),
                "mean_proxy_pct": float(proxy_arr.mean()),
                "mae_vs_proxy": float(mae),
                "pearson_vs_proxy": float(corr),
            }
        )
        print(
            f"thr={thr:.2f} mean_pred={pred_pcts.mean():.1f}% "
            f"MAE_proxy={mae:.2f} r={corr:.3f}",
            flush=True,
        )

    chosen = pick_threshold(sweep_rows)
    chosen_thr = float(chosen["threshold"])
    selection_rule = (
        "На unlabeled-32: min MAE(pred%, white_frac_proxy*100); "
        "среди MAE ≤ best+0.5 — max Pearson r; при равном r — меньший thr. "
        "Photos-8 IoU в выборе НЕ участвует."
    )
    print(f"Выбран thr={chosen_thr:.2f} | {selection_rule}", flush=True)

    # --- once eval on held-out 8 ---
    # Всегда считаем baseline@0.5 и chosen (даже если chosen==0.5 — две папки overlays).
    eval_plan = [("baseline_0.5", 0.5), ("A_calibrated", chosen_thr)]
    heldout_results: dict[str, dict] = {}
    # Кэш probs held-out, чтобы не гонять модель дважды при chosen==0.5.
    heldout_probs: dict[str, np.ndarray] = {}
    for folder_name, thr in eval_plan:
        key = f"thr_{thr:.2f}".replace(".", "p")
        rows: list[dict] = []
        sub = out_dir / folder_name
        sub.mkdir(parents=True, exist_ok=True)
        print(f"Eval Photos-8 @ thr={thr:.2f} → {sub.name}", flush=True)
        for jpg in labeled:
            mask_path = masks_dir / f"{jpg.stem}.png"
            if jpg.name not in heldout_probs:
                image_chw = load_rgb_u8(jpg)
                heldout_probs[jpg.name] = (
                    image_chw,
                    predict_prob(model, image_chw, image_size, device),
                )
            image_chw, prob = heldout_probs[jpg.name]
            pred = (prob >= thr).astype(np.float32)
            gt_u8 = np.asarray(Image.open(mask_path))
            if gt_u8.ndim == 3:
                gt_u8 = gt_u8[..., 0]
            gt = (gt_u8 >= 127).astype(np.float32)
            if pred.shape != gt.shape:
                raise ValueError(f"{jpg.name}: pred {pred.shape} != gt {gt.shape}")
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
        heldout_results[key] = {
            "threshold": thr,
            "folder": folder_name,
            "per_image": rows,
            "aggregate": aggregate_metrics(rows),
            "overlays_dir": str(sub),
        }
        # Если chosen==0.5, ключ один — дополнительно помечаем alias для A.
        if folder_name == "A_calibrated" and key in heldout_results:
            heldout_results["A_calibrated_alias"] = heldout_results[key]

    part_a = {
        "part": "A",
        "checkpoint": str(run_dir / "best.pt"),
        "config": config,
        "checkpoint_epoch": ckpt.get("epoch"),
        "resolution_note": resolution_note,
        "preview_size_wh": list(preview_wh),
        "image_size_model": image_size,
        "seed": args.seed,
        "labeled_n": len(labeled),
        "unlabeled_n": len(unlabeled),
        "labeled_files": [p.name for p in labeled],
        "unlabeled_files": [p.name for p in unlabeled],
        "selection_rule": selection_rule,
        "no_leakage_note": (
            "Порог выбран только по white_frac_proxy на 32 кадрах без CVAT-маски. "
            "8 размеченных кадров не участвовали в выборе thr и не в обучении."
        ),
        "proxy_sweep": sweep_rows,
        "chosen_threshold": chosen_thr,
        "chosen_row": chosen,
        "probs_cache": str(cache_path),
        "heldout_eval": heldout_results,
    }
    (out_dir / "part_a.json").write_text(
        json.dumps(part_a, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Part A → {out_dir / 'part_a.json'}", flush=True)


if __name__ == "__main__":
    main()
