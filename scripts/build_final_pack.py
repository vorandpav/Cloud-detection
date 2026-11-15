"""Собирает финальный мега-отчёт + фигуры для презы + PRESENTATION.md.

Читает history/metrics из outputs/, рисует seaborn-графики, пишет:
  - docs/FINAL_CASE_REPORT.md
  - outputs/presentation/PRESENTATION.md
  - outputs/presentation/figures/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "presentation" / "figures"
PREZ = ROOT / "outputs" / "presentation"
REPORT = ROOT / "docs" / "FINAL_CASE_REPORT.md"

# Палитра: тёмный графит + бирюза + терракота (не «AI purple»).
COLORS = {
    "ink": "#1a1f24",
    "muted": "#5c6670",
    "teal": "#0f7a73",
    "coral": "#c45c3e",
    "sand": "#e8e0d5",
    "line": "#d0c8bc",
    "blue": "#2f5d8a",
    "green": "#3d7a4a",
}


def setup_style() -> None:
    """Единый стиль фигур под презу."""
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#faf8f5",
            "figure.facecolor": "white",
            "axes.edgecolor": COLORS["line"],
            "grid.color": "#eee8e0",
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "axes.titleweight": "bold",
        },
    )


def savefig(fig: plt.Figure, name: str) -> Path:
    """Сохраняет PNG 180 dpi."""
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def load_history(csv_path: Path) -> pd.DataFrame:
    """Читает history.csv обучения."""
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)


def fig_val_curves_baseline() -> Path:
    """Кривые baseline U-Net (победитель)."""
    df = load_history(ROOT / "outputs/train/unet_mix_hrc_sparcs_38_4060/history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(df["epoch"], df["train_loss"], color=COLORS["teal"], lw=2.4, label="train")
    axes[0].plot(df["epoch"], df["val_loss"], color=COLORS["coral"], lw=2.4, label="val")
    axes[0].set_title("Loss — baseline U-Net")
    axes[0].set_xlabel("epoch")
    axes[0].legend(frameon=False)
    axes[1].plot(df["epoch"], df["val_iou"], color=COLORS["teal"], lw=2.4, label="IoU")
    axes[1].plot(df["epoch"], df["val_dice"], color=COLORS["blue"], lw=2.4, label="Dice")
    best_ep = int(df.loc[df["val_iou"].idxmax(), "epoch"])
    best_iou = float(df["val_iou"].max())
    axes[1].axvline(best_ep, color=COLORS["coral"], ls="--", lw=1.2, alpha=0.8)
    axes[1].scatter([best_ep], [best_iou], color=COLORS["coral"], s=80, zorder=5)
    axes[1].set_title(f"Val seg — best IoU {best_iou:.3f} @ ep{best_ep}")
    axes[1].set_xlabel("epoch")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return savefig(fig, "01_baseline_unet_curves.png")


def fig_model_zoo_val() -> Path:
    """Сравнение val IoU моделей."""
    rows = [
        ("U-Net baseline", 0.881, COLORS["teal"]),
        ("SegFormer Kaggle", 0.885, COLORS["blue"]),
        ("U-Net color-match", 0.874, COLORS["muted"]),
        ("U-Net cubesat-augs", 0.878, COLORS["muted"]),
    ]
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, height=0.62, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0.82, 0.90)
    ax.set_xlabel("best val IoU (HRC holdout)")
    ax.set_title("Публичный val: все ≈ 0.88 — победителя нет")
    for yi, v in zip(y, vals):
        ax.text(v + 0.0015, yi, f"{v:.3f}", va="center", fontsize=12, color=COLORS["ink"])
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return savefig(fig, "02_val_iou_zoo.png")


def fig_photos_compare() -> Path:
    """Photos-8 IoU по вариантам."""
    rows = [
        ("baseline U-Net", 0.492),
        ("cubesat-augs", 0.340),
        ("color-match", 0.296),
        ("SegFormer", 0.205),
        ("pseudo-FT (B)", 0.461),
        ("mask-FT LOO (C)", 0.619),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [
        COLORS["teal"] if n == "baseline U-Net" else (
            COLORS["coral"] if "mask-FT" in n else COLORS["muted"]
        )
        for n in names
    ]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.65, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("mean IoU на 8 Photos (грубая CVAT GT)")
    ax.set_title("Photos: baseline лучше «умных» трюков; LOO↑ цифрой, глазами хуже")
    for yi, v in zip(y, vals):
        ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=12)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return savefig(fig, "03_photos_iou_compare.png")


def fig_domain_gap() -> Path:
    """Схема domain gap: val vs Photos для baseline."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = ["Val HRC", "Photos-8"]
    vals = [0.881, 0.492]
    bars = ax.bar(labels, vals, color=[COLORS["teal"], COLORS["coral"]], width=0.55, edgecolor="white")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("IoU")
    ax.set_title("Один и тот же baseline: val ≠ кубсат")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=14, fontweight="bold")
    ax.annotate(
        "domain gap",
        xy=(0.5, 0.69),
        xytext=(0.5, 0.85),
        ha="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["ink"]),
        fontsize=12,
        color=COLORS["ink"],
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    return savefig(fig, "04_domain_gap.png")


def fig_segformer_kaggle() -> Path:
    """Кривые SegFormer с Kaggle."""
    path = ROOT / "outputs/kaggle/kaggle_segformer_mix/history.csv"
    if not path.is_file():
        path = ROOT / "outputs/kaggle/history.csv"
    df = load_history(path)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(df["epoch"], df["val_iou"], color=COLORS["blue"], lw=2.5, marker="o", label="val IoU")
    ax.plot(df["epoch"], df["train_loss"], color=COLORS["muted"], lw=1.8, alpha=0.8, label="train loss")
    ax.set_title("Kaggle SegFormer-B2 — best≈0.885 @ ep4, дальше шум val=30")
    ax.set_xlabel("epoch")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return savefig(fig, "05_segformer_kaggle_curves.png")


def fig_underseg_bias() -> Path:
    """GT% vs pred% на Photos для baseline."""
    metrics = ROOT / "outputs/reports/unet_mix_baseline_photos_check/metrics.json"
    if not metrics.is_file():
        # fallback: hardcoded from known report
        gt = [26.8, 94.0, 23.6, 62.4, 93.4, 46.4, 61.6, 14.2]
        pred = [14.5, 79.7, 9.9, 45.8, 59.0, 13.1, 33.1, 24.9]
    else:
        data = json.loads(metrics.read_text(encoding="utf-8"))
        rows = data.get("photos_test") or data.get("per_image") or []
        if not rows and "photos" in data:
            rows = data["photos"].get("per_image", [])
        # try nested
        if not rows:
            for key in data:
                if isinstance(data[key], dict) and "per_image" in data[key]:
                    rows = data[key]["per_image"]
                    break
        if rows:
            gt = [float(r.get("cloud_frac_gt_pct", r.get("gt_pct", 0))) for r in rows]
            pred = [float(r.get("cloud_frac_pred_pct", r.get("pred_pct", 0))) for r in rows]
        else:
            gt = [26.8, 94.0, 23.6, 62.4, 93.4, 46.4, 61.6, 14.2]
            pred = [14.5, 79.7, 9.9, 45.8, 59.0, 13.1, 33.1, 24.9]
    fig, ax = plt.subplots(figsize=(5.8, 5.5))
    ax.scatter(gt, pred, s=90, color=COLORS["teal"], edgecolor="white", zorder=3)
    ax.plot([0, 100], [0, 100], ls="--", color=COLORS["coral"], lw=1.5, label="идеал")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("GT cloud % (грубая кисть)")
    ax.set_ylabel("pred cloud % (baseline)")
    ax.set_title("Систематический under-seg на Photos")
    ax.legend(frameon=False, loc="upper left")
    sns.despine(ax=ax)
    fig.tight_layout()
    return savefig(fig, "06_photos_gt_vs_pred.png")


def write_final_report(fig_names: list[str]) -> None:
    """Пишет docs/FINAL_CASE_REPORT.md."""
    lines = [
        "# Итоговый отчёт: облачность Photos (кейс СПУТНИКС)",
        "",
        "**Дата:** 2026-11-15  ",
        "**Сид:** 42  ",
        "**Победитель:** U-Net ResNet34 ImageNet — "
        "`outputs/train/unet_mix_hrc_sparcs_38_4060/best.pt`",
        "",
        "---",
        "",
        "## 1. Задача",
        "",
        "По RGB-снимкам обзорных камер кубсатов (**Photos**, ~200 м/пкс, JPEG) оценить "
        "**долю облачности** и решение **reject @ 75%**. Готовой разметки Photos не было — "
        "pretrain на публичных RGB-сетах, проверка на ручных масках (8 кадров, грубая кисть).",
        "",
        "## 2. Данные",
        "",
        "| источник | роль | n (примерно) |",
        "|----------|------|-------------:|",
        "| HRC_WHU | ближе к Photos по цвету, holdout val | 120 train / test |",
        "| SPARCS | Landsat holdout, тени/снег | 80 |",
        "| 38-Cloud | много патчей Landsat RGB | ~5155 |",
        "| Photos_40 | целевой домен; 8 с CVAT-масками | 40 / 8 labeled |",
        "",
        "Веса микса baseline: HRC 3.0 · SPARCS 2.5 · 38-Cloud 1.0 "
        "(сэмплирование по весам, не по длине).",
        "",
        "## 3. Модель-победитель",
        "",
        "- **Архитектура:** U-Net, encoder ResNet34, ImageNet",
        "- **Вход:** RGB 384×384, без color-match",
        "- **Лосс:** Dice + BCE",
        "- **Лучший val:** IoU **0.881**, Dice **0.936**, MAE% **2.1**, reject@75 **1.00** (ep4)",
        "- **Early stop:** ep9",
        "",
        f"![baseline curves](../outputs/presentation/figures/01_baseline_unet_curves.png)",
        "",
        "## 4. Сравнение архитектур / абляций (val HRC)",
        "",
        "| модель | best val IoU | комментарий |",
        "|--------|-------------:|-------------|",
        "| **U-Net baseline** | **0.881** | итог кейса |",
        "| SegFormer-B2 (Kaggle T4, 512) | 0.885 | +0.004, шум val=30 |",
        "| U-Net + color-match meanstd | 0.874 | не лучше |",
        "| U-Net HRC↑ + cubesat-augs | 0.878 | val ок, Photos хуже |",
        "",
        f"![val zoo](../outputs/presentation/figures/02_val_iou_zoo.png)",
        "",
        f"![segformer](../outputs/presentation/figures/05_segformer_kaggle_curves.png)",
        "",
        "## 5. Photos: главный разрыв",
        "",
        "На 8 CVAT-масках (грубая кисть — не золотой стандарт) baseline даёт "
        "IoU **0.492**, mean pred% **~35** vs GT% **~53** — **недоразмечает**.",
        "",
        f"![domain gap](../outputs/presentation/figures/04_domain_gap.png)",
        "",
        f"![gt vs pred](../outputs/presentation/figures/06_photos_gt_vs_pred.png)",
        "",
        "### Почему",
        "",
        "1. Domain gap: Landsat/HRC ≠ кубсат JPEG ~200 м.",
        "2. Val = HRC, не Photos — val оптимистичен.",
        "3. Микс насыщен 38-Cloud → prior «плотных» облаков.",
        "4. Dice/BCE при неуверенности режет FP → under-seg.",
        "",
        "### Что пробовали на Photos (и итог)",
        "",
        "| метод | Photos IoU | вердикт |",
        "|-------|-----------:|---------|",
        "| baseline @0.5 | **0.492** | лучший визуально и по цифрам среди честных |",
        "| A: thr по white_frac_proxy на 32 | 0.492 | выбрался thr=0.5 |",
        "| B: soft pseudo-FT на 32 | 0.461 | хуже |",
        "| C: mask-FT LOO на 8 | 0.619 | цифра↑, **глазами хуже** (учит мазню) |",
        "| color-match | 0.296 | сильно under-seg |",
        "| SegFormer | 0.205 | ещё сильнее under-seg |",
        "| cubesat-augs | 0.340 | не помогло |",
        "",
        f"![photos compare](../outputs/presentation/figures/03_photos_iou_compare.png)",
        "",
        "**Вывод:** без аккуратной Photos-разметки (train≠test) потолок низкий. "
        "Кривой finetune в прод не берём — оставляем как негативный эксперимент.",
        "",
        "## 6. Галереи (для глаз / презы)",
        "",
        "| что | путь |",
        "|-----|------|",
        "| train/val baseline triplets | `outputs/reports/unet_mix_baseline_photos_check/` |",
        "| Photos baseline vs color vs SegFormer | `outputs/reports/photos_compare_models/` |",
        "| mask-FT (плохой finetune) | `outputs/reports/photos_mask_ft/` |",
        "| фигуры презы | `outputs/presentation/figures/` |",
        "",
        "## 7. Воспроизведение",
        "",
        "### Локально (победитель)",
        "",
        "```bash",
        "uv sync",
        "uv run train-mix --model unet --device-plan 4060 \\",
        "  --datasets hrc,sparcs,cloud38 --epochs 20 --batch-size 4 --seed 42 \\",
        "  --run-name unet_mix_hrc_sparcs_38_4060",
        "```",
        "",
        "Данные: `data/hrc_whu`, `data/sparcs` (или `sending`), `data/38-cloud`.",
        "",
        "### Kaggle T4",
        "",
        "Один ноутбук: `outputs/kaggle/train_mix_one_cell.ipynb` — "
        "распаковка кода/данных + SegFormer train-mix (сравнение 0.885).",
        "",
        "## 8. Рекомендация для продукта/кейса",
        "",
        "1. В прод/демо — **baseline U-Net**.",
        "2. В презе честно показать domain gap и провал «красивых» абляций.",
        "3. Следующий шаг (если продолжать): 15–20 **аккуратно** размеченных Photos "
        "как train, старые 8 — test; не учить на толстой кисти.",
        "",
        "---",
        "",
        f"*Фигуры: {', '.join(fig_names)}*",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_presentation() -> None:
    """Слайды в MD: текст + пути к картинкам (дизайн делает другая модель)."""
    slides = [
        {
            "n": 1,
            "title": "Титул",
            "on_slide": "Облачность на снимках кубсатов (СПУТНИКС)\n% облаков + reject @ 75%",
            "speaker": "Кейс: Photos без готовой разметки → pretrain на публичных RGB → проверка на 8 ручных масках.",
            "image": None,
        },
        {
            "n": 2,
            "title": "Проблема",
            "on_slide": "Нужен % облачности на Photos\nНет GT кубсатов\nЧужой домен: Landsat/HRC ≠ JPEG ~200 м",
            "speaker": "Готовых весов «под кубсат» нет ни у кого. Берём публичные сеты и переносим.",
            "image": None,
        },
        {
            "n": 3,
            "title": "Данные",
            "on_slide": "HRC · SPARCS · 38-Cloud → pretrain\nPhotos_40 → цель\n8 CVAT-масок → грубый тест",
            "speaker": "HRC ближе по цвету. 38-Cloud даёт объём. Photos — целевой сенсор.",
            "image": "outputs/reports/unet_mix_baseline_photos_check/train/01_hrc_triplet.png",
        },
        {
            "n": 4,
            "title": "Победитель: U-Net",
            "on_slide": "U-Net ResNet34\nval IoU 0.881 · Dice 0.936\nreject@75 = 1.0 на HRC val",
            "speaker": "Простая модель, локальная 4060, воспроизводимо, seed 42.",
            "image": "outputs/presentation/figures/01_baseline_unet_curves.png",
        },
        {
            "n": 5,
            "title": "Val: все рядом",
            "on_slide": "SegFormer 0.885 ≈ U-Net 0.881\nColor-match / cubesat-augs не выиграли",
            "speaker": "На HRC holdout архитектурный разрыв в пределах шума val=30.",
            "image": "outputs/presentation/figures/02_val_iou_zoo.png",
        },
        {
            "n": 6,
            "title": "Kaggle SegFormer",
            "on_slide": "T4 · 512 px · ~10 ep\nbest 0.885 @ ep4\nДальше шум, не рост",
            "speaker": "Современный прогон для нарратива; на Photos позже оказался хуже U-Net.",
            "image": "outputs/presentation/figures/05_segformer_kaggle_curves.png",
        },
        {
            "n": 7,
            "title": "Domain gap",
            "on_slide": "Тот же U-Net:\nval 0.88 → Photos 0.49\nСистематический under-seg",
            "speaker": "Главный слайд кейса: хороший val ≠ готовность к кубсату.",
            "image": "outputs/presentation/figures/04_domain_gap.png",
        },
        {
            "n": 8,
            "title": "Photos глазами",
            "on_slide": "RGB | твоя маска | pred\nМодель скупее толстой кисти",
            "speaker": "Показать 1–2 triplets из baseline photos_test (плотные облака vs сложный кадр).",
            "image": "outputs/reports/photos_compare_models/baseline_unet/02_2025-10-24_21-36-35_SXC3-2331_1_triplet.png",
        },
        {
            "n": 9,
            "title": "Что не сработало",
            "on_slide": "порог / pseudo / color-match\nSegFormer / cubesat-augs\nfinetune на кривой кисти",
            "speaker": "Честный негативный результат: без нормальной Photos-разметки трюки не лечат домен.",
            "image": "outputs/presentation/figures/03_photos_iou_compare.png",
        },
        {
            "n": 10,
            "title": "Плохой finetune (урок)",
            "on_slide": "LOO IoU↑ (0.62), картинка хуже\nМодель выучила мазню",
            "speaker": "Оставить в презе как cautionary tale. В прод не берём.",
            "image": "outputs/reports/photos_mask_ft/loo/11_2026-02-09_21-42-09_SXC3-2342_1_triplet.png",
        },
        {
            "n": 11,
            "title": "Итог",
            "on_slide": "В кейс: baseline U-Net\nЧестно: gap на Photos\nДальше: аккуратная разметка train≠test",
            "speaker": "Закрыть рекомендацией и путём к весам/команде train-mix.",
            "image": "outputs/presentation/figures/06_photos_gt_vs_pred.png",
        },
    ]
    lines = [
        "# Презентация (контент для дизайна)",
        "",
        "Дизайн/вёрстку делает другая модель. Здесь: **номер слайда → текст на слайде → "
        "спикерские → картинка (путь)**.",
        "",
        "Стиль контента: коротко, без жаргона. Палитра фигур: графит / бирюза / терракота.",
        "",
        "Полный отчёт: [`docs/FINAL_CASE_REPORT.md`](../../docs/FINAL_CASE_REPORT.md).",
        "",
        "---",
        "",
    ]
    for s in slides:
        lines.append(f"## Слайд {s['n']}: {s['title']}")
        lines.append("")
        lines.append("**На слайде:**")
        lines.append("")
        lines.append("```")
        lines.append(s["on_slide"])
        lines.append("```")
        lines.append("")
        lines.append(f"**Спикер:** {s['speaker']}")
        lines.append("")
        if s["image"]:
            rel = s["image"]
            lines.append(f"**Картинка:** `{rel}`")
            lines.append("")
            lines.append(f"![{s['title']}](../../{rel})")
        else:
            lines.append("**Картинка:** — (титул / текст)")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend(
        [
            "## Ассеты фигур",
            "",
            "Все сгенерированные графики: `outputs/presentation/figures/`.",
            "",
            "## Галереи triplets (вставить 2–3 лучших)",
            "",
            "- Val OK: `outputs/reports/unet_mix_baseline_photos_check/val/`",
            "- Photos baseline: `outputs/reports/photos_compare_models/baseline_unet/`",
            "- Плохой FT: `outputs/reports/photos_mask_ft/loo/`",
            "",
        ]
    )
    PREZ.mkdir(parents=True, exist_ok=True)
    (PREZ / "PRESENTATION.md").write_text("\n".join(lines), encoding="utf-8")
    (PREZ / "README.md").write_text(
        "# Presentation pack\n\n- `PRESENTATION.md` — слайды (текст + пути)\n"
        "- `figures/` — графики для презы\n"
        "- Полный отчёт: `docs/FINAL_CASE_REPORT.md`\n",
        encoding="utf-8",
    )


def main() -> None:
    """Точка входа."""
    setup_style()
    paths = [
        fig_val_curves_baseline(),
        fig_model_zoo_val(),
        fig_photos_compare(),
        fig_domain_gap(),
        fig_segformer_kaggle(),
        fig_underseg_bias(),
    ]
    write_final_report([p.name for p in paths])
    write_presentation()
    print("figures:", ", ".join(p.name for p in paths))
    print("report:", REPORT)
    print("prez:", PREZ / "PRESENTATION.md")


if __name__ == "__main__":
    main()
