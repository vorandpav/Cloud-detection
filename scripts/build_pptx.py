"""Сборка черновика презентации кейса → outputs/pptx/cloudness_case_draft.pptx."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "pptx" / "cloudness_case_draft.pptx"
EDA = ROOT / "outputs" / "eda" / "datasets"


def _set_run(paragraph, text: str, size: int = 18, bold: bool = False) -> None:
    paragraph.clear()
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = "Calibri"
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)


def add_title(prs: Presentation, title: str, subtitle: str = "") -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    box = slide.shapes.add_textbox(Inches(0.6), Inches(2.2), Inches(12), Inches(1.2))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    _set_run(p, title, size=32, bold=True)
    if subtitle:
        box2 = slide.shapes.add_textbox(Inches(0.6), Inches(3.5), Inches(12), Inches(1.5))
        p2 = box2.text_frame.paragraphs[0]
        _set_run(p2, subtitle, size=18)


def add_bullets(prs: Presentation, title: str, lines: list[str]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tbox = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(12), Inches(0.7))
    _set_run(tbox.text_frame.paragraphs[0], title, size=26, bold=True)
    body = slide.shapes.add_textbox(Inches(0.6), Inches(1.2), Inches(12), Inches(5.5))
    tf = body.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        _set_run(p, "• " + line, size=18)
        p.space_after = Pt(10)


def add_dataset_slide(
    prs: Presentation,
    title: str,
    bullets: list[str],
    image_paths: list[Path],
) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tbox = slide.shapes.add_textbox(Inches(0.4), Inches(0.25), Inches(12), Inches(0.55))
    _set_run(tbox.text_frame.paragraphs[0], title, size=24, bold=True)
    body = slide.shapes.add_textbox(Inches(0.4), Inches(0.9), Inches(5.2), Inches(5.8))
    tf = body.text_frame
    tf.word_wrap = True
    for i, line in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        _set_run(p, "• " + line, size=15)
        p.space_after = Pt(6)
    x = Inches(5.8)
    for path in image_paths[:2]:
        if path.exists():
            slide.shapes.add_picture(str(path), x, Inches(1.0), width=Inches(3.5))
            x = Inches(9.4)


def main() -> None:
    """Собирает PPTX."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    add_title(
        prs,
        "Облачность на кубсатах — кейс СПУТНИКС",
        "RGB обзорные камеры · % облачности · reject@75\nЧерновик презентации · 2026-11-15",
    )

    add_bullets(
        prs,
        "Задача",
        [
            "Вход: RGB JPG с обзорных камер кубсатов (~200 м/пкс, Photos).",
            "Выход: доля облаков на кадре (%) и решение keep/delete при пороге 75%.",
            "Публичные сеты — pretrain и сравнение; финал меряем на Photos.",
            "Железо: RTX 4060 8 GB локально + Kaggle T4 16 GB под SOTA-adapter.",
        ],
    )

    add_dataset_slide(
        prs,
        "Датасет: Photos (целевой)",
        [
            "4272 JPG, ~33 GB; labeling 40 кадров.",
            "Пиксельных масок пока нет — главный gap кейса.",
            "white_frac ≈ 0.07 (прокси, не GT).",
            "Роль: финальная метрика продукта.",
        ],
        [EDA / "photos" / "sample_1.png", EDA / "photos" / "sample_labeling_01.png"],
    )

    add_dataset_slide(
        prs,
        "Датасет: 38-Cloud",
        [
            "Landsat-8, 384², ~8400 train GT, RGB.",
            "Cloud frac ≈ 0.50 на nonempty.",
            "Качество масок: посредственное.",
            "Роль: объём pretrain, меньший вес в миксе.",
        ],
        [EDA / "38-cloud" / "sample_1.png", EDA / "38-cloud" / "sample_1_overlay.png"],
    )

    add_dataset_slide(
        prs,
        "Датасет: SPARCS",
        [
            "80 сцен Landsat, RGB из *_photo.png.",
            "Качество масок: хорошее (validation).",
            "Cloud frac ≈ 0.25; есть shadow/snow.",
            "Роль: высокий вес + holdout/ошибки.",
        ],
        [EDA / "sparcs" / "sample_1.png", EDA / "sparcs" / "sample_1_overlay.png"],
    )

    add_dataset_slide(
        prs,
        "Датасет: HRC_WHU (+ кратко GF1)",
        [
            "HRC: 150 RGB сцен, маски хорошие, ~110 MB.",
            "GF1 pack: объём есть, разметка посредственная.",
            "HRC — якорь качественного pretrain.",
            "GF1 — низкий вес / строка таблицы.",
        ],
        [EDA / "hrc_whu" / "sample_1.png", EDA / "hrc_whu" / "sample_1_overlay.png"],
    )

    add_dataset_slide(
        prs,
        "Датасет: S2 Cloud Mask Catalogue",
        [
            "513 пар masks+subscenes (1022², 13 bands).",
            "RGB = B4,B3,B2; overlay = CLOUD channel.",
            "Качество масок: хорошее (экспертный каталог).",
            "Роль: другой сенсор, mid weight в миксе.",
        ],
        [EDA / "s2_catalogue" / "sample_1.png", EDA / "s2_catalogue" / "sample_1_overlay.png"],
    )

    add_bullets(
        prs,
        "Качество разметки (честно)",
        [
            "Хорошие: SPARCS, HRC, S2 Catalogue (экспертные/валидационные).",
            "Посредственные: 38-Cloud и GF1 pack — грубые границы, шум; не эталон для Photos.",
            "Photos: без pixel GT обучение «в лоб» нельзя — нужна разметка photos_40.",
            "Вывод: веса в миксе и staged finetune важнее слепого concat.",
        ],
    )

    add_bullets(
        prs,
        "Подходы 2019–2026 (обзор)",
        [
            "2019–21: Cloud-Net / Cloud-Net+ на 38-Cloud (история + идея soft Jaccard).",
            "2021–25: SegFormer и CNN++ как pragmatic mid-tier.",
            "2025: Cloud-Adapter — frozen DINOv2-L + лёгкий adapter (~0.6% params).",
            "2026: CloudMamba и др. — опционально; часто 4 bands ≠ Photos RGB.",
        ],
    )

    add_bullets(
        prs,
        "Почему не Cloud-Net как финал",
        [
            "Cloud-Net заточен под Landsat RGBNir / 38-Cloud — не кубсат Photos.",
            "Cloud-Adapter на T4: живой SOTA 2025, RGB-friendly, требует ≥16 GB.",
            "U-Net ResNet34 — локальный baseline на 4060 (уже сходится на 38-Cloud).",
            "SegFormer-B2 — mid-tier сравнение (не «вместо» adapter).",
        ],
    )

    add_bullets(
        prs,
        "План по железу",
        [
            "4060 8 GB: U-Net batch~8 @384; SegFormer-B2 batch≤4; быстрый mix HRC+SPARCS.",
            "Kaggle T4 16 GB: Cloud-Adapter pretrain на HRC (+GF1 строка таблицы).",
            "Микс весов: HRC↑ SPARCS↑ S2 mid 38-Cloud↓ GF1↓; Photos — после масок.",
            "Staged вариант: HRC+SPARCS → soft 38-Cloud → finetune photos_40.",
        ],
    )

    add_bullets(
        prs,
        "Метрики успеха кейса",
        [
            "Главные: MAE по % облачности и accuracy reject@75 на Photos.",
            "Вспомогательные: IoU/Dice на HRC / SPARCS / 38-Cloud (по отдельности).",
            "Не смешивать Jaccard 38-Cloud с mIoU HRC из статей.",
            "Слайд сходимости: curves.png + best.pt из outputs/train/.",
        ],
    )

    prs.save(str(OUT))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
