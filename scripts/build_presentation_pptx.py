"""Сборка product PPTX из outputs/presentation/PRESENTATION.md.

Парсит заголовки, абзацы, списки, таблицы и ![](path), режет на плотные
слайды (не «тонкая» колода), встраивает картинки. Пропущенные файлы —
плейсхолдер-текст, без падения.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "outputs" / "presentation" / "PRESENTATION.md"
OUT_PATH = ROOT / "outputs" / "presentation" / "cloudness_case.pptx"
MD_BASE = MD_PATH.parent

# Тёмная профессиональная тема (графит + бирюза, без purple).
BG = RGBColor(0x1A, 0x1F, 0x24)
CARD = RGBColor(0x24, 0x2B, 0x33)
INK = RGBColor(0xF2, 0xF0, 0xEB)
MUTED = RGBColor(0xA8, 0xB0, 0xB8)
TEAL = RGBColor(0x2A, 0xA8, 0x9E)
CORAL = RGBColor(0xD4, 0x7A, 0x5C)
LINE = RGBColor(0x3A, 0x44, 0x4E)
TABLE_HEADER_BG = RGBColor(0x2F, 0x5D, 0x5A)
TABLE_ALT = RGBColor(0x2A, 0x32, 0x3A)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN_X = Inches(0.45)
CONTENT_W = Inches(12.4)

# Лимиты плотности: цель ~40–70 слайдов, читаемо.
MAX_BULLETS = 14
MAX_PARA_CHARS = 900
MAX_PARA_PER_SLIDE = 5
MAX_TABLE_ROWS = 16  # включая header
MAX_IMAGES = 4
TABLE_ROWS_MERGE = 9  # маленькую таблицу можно клеить к предыдущему слайду
LEAD_IN_CHARS = 220  # короткий ввод к таблице/картинкам → subtitle, не отдельный слайд
MERGE_MAX_BULLETS = 16
MERGE_MAX_PARAS = 6
MERGE_MAX_IMAGES = 4
TINY_SLIDE_CHARS = 90  # «пустые» слайды вклеиваем в предыдущий

# Мета про процесс сборки / агента — не на слайды (научные оговорки оставляем).
_META_FRAGMENTS = (
    "markdown-презентация",
    "для переноса в pptx",
    "переноса в слайды",
    "source of truth",
    "не живой отчёт",
    "не живой отчет",
    "черновик агента",
    "для вставки в pptx",
    "самодостаточный текст",
    "конец презентационного",
    "прочитай другой md",
    "stub",
    "заглушк",
)


@dataclass
class Block:
    """Один блок контента из markdown."""

    kind: str  # heading1|heading2|heading3|para|list|table|image|hr|code
    text: str = ""
    level: int = 0
    items: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    path: str = ""
    alt: str = ""


@dataclass
class SlideSpec:
    """Спецификация одного слайда перед рендером."""

    title: str
    kind: str  # title|section|bullets|table|images|mixed|code
    subtitle: str = ""
    bullets: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    images: list[tuple[str, Path | None]] = field(default_factory=list)
    footer: str = ""


def _rgb_fill(shape, color: RGBColor) -> None:
    """Заливка фигуры сплошным RGB."""
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def _set_run(
    paragraph,
    text: str,
    *,
    size: int = 16,
    bold: bool = False,
    color: RGBColor = INK,
    font: str = "Calibri",
) -> None:
    """Пишет один run в абзац."""
    paragraph.clear()
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = font
    run.font.color.rgb = color


def _paint_slide_bg(slide, prs: Presentation) -> None:
    """Тёмный фон на весь слайд."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height
    )
    shape.line.fill.background()
    _rgb_fill(shape, BG)
    # Уводим фон за текст: в python-pptx порядок = порядок добавления,
    # поэтому фон добавляем первым на каждом слайде.


def _accent_bar(slide) -> None:
    """Тонкая бирюзовая полоска сверху."""
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(0.08))
    bar.line.fill.background()
    _rgb_fill(bar, TEAL)


def _title_box(slide, title: str, size: int = 24, subtitle: str = "") -> float:
    """Заголовок (+ опциональный subtitle). Возвращает Y начала контента."""
    box = slide.shapes.add_textbox(MARGIN_X, Inches(0.18), CONTENT_W, Inches(0.48))
    tf = box.text_frame
    tf.word_wrap = True
    _set_run(tf.paragraphs[0], title, size=size, bold=True, color=INK)
    y = 0.70
    if subtitle:
        sub = slide.shapes.add_textbox(MARGIN_X, Inches(0.62), CONTENT_W, Inches(0.35))
        _set_run(sub.text_frame.paragraphs[0], subtitle, size=12, color=TEAL)
        y = 1.00
    return y


def _footer(slide, text: str) -> None:
    """Мелкий футер."""
    if not text:
        return
    box = slide.shapes.add_textbox(MARGIN_X, Inches(7.15), CONTENT_W, Inches(0.28))
    _set_run(box.text_frame.paragraphs[0], text, size=10, color=MUTED)


def _strip_md_inline(text: str) -> str:
    """Убирает простой markdown-инлайн, оставляет читаемый текст."""
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_markdown(md: str) -> list[Block]:
    """Разбирает PRESENTATION.md на блоки."""
    lines = md.splitlines()
    blocks: list[Block] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            blocks.append(Block(kind="hr"))
            i += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # closing fence
            blocks.append(Block(kind="code", text="\n".join(code_lines)))
            continue

        m_img = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if m_img:
            blocks.append(Block(kind="image", alt=m_img.group(1), path=m_img.group(2)))
            i += 1
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[level:].strip()
            kind = {1: "heading1", 2: "heading2", 3: "heading3"}.get(level, "heading3")
            blocks.append(Block(kind=kind, text=_strip_md_inline(title), level=level))
            i += 1
            continue

        if "|" in stripped and stripped.startswith("|"):
            table_lines: list[str] = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = _parse_table(table_lines)
            if rows:
                blocks.append(Block(kind="table", rows=rows))
            continue

        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            items: list[str] = []
            while i < n:
                s = lines[i].strip()
                if re.match(r"^[-*]\s+", s):
                    items.append(_strip_md_inline(re.sub(r"^[-*]\s+", "", s)))
                    i += 1
                elif re.match(r"^\d+\.\s+", s):
                    items.append(_strip_md_inline(re.sub(r"^\d+\.\s+", "", s)))
                    i += 1
                elif s.startswith("  ") and items:
                    items[-1] += " " + _strip_md_inline(s)
                    i += 1
                else:
                    break
            blocks.append(Block(kind="list", items=items))
            continue

        # Обычный абзац (возможно несколько строк до пустой).
        para_parts: list[str] = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt.startswith("|")
                or nxt.startswith("```")
                or nxt == "---"
                or re.match(r"!\[", nxt)
                or re.match(r"^[-*]\s+", nxt)
                or re.match(r"^\d+\.\s+", nxt)
            ):
                break
            para_parts.append(nxt)
            i += 1
        text = _strip_md_inline(" ".join(para_parts))
        if text:
            blocks.append(Block(kind="para", text=text))

    return blocks


def _parse_table(table_lines: list[str]) -> list[list[str]]:
    """Парсит markdown-таблицу в список строк."""
    rows: list[list[str]] = []
    for line in table_lines:
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append([_strip_md_inline(c) for c in cells])
    return rows


def _resolve_image(rel: str) -> Path | None:
    """Резолвит путь картинки относительно outputs/presentation/."""
    path = (MD_BASE / rel).resolve()
    return path if path.is_file() else None


def _is_lead_in(paras: list[str], bullets: list[str]) -> bool:
    """Короткий ввод к таблице/картинкам — subtitle, не отдельный слайд."""
    if bullets:
        return False
    if not paras or len(paras) > 3:
        return False
    total = sum(len(p) for p in paras)
    return total <= LEAD_IN_CHARS * 2 and all(len(p) <= LEAD_IN_CHARS for p in paras)


def _spec_body_chars(spec: SlideSpec) -> int:
    """Сколько текста на слайде (без заголовка)."""
    return sum(len(p) for p in spec.paragraphs) + sum(len(b) for b in spec.bullets)


def _spec_kind_bucket(spec: SlideSpec) -> str:
    if spec.kind in {"title", "section"}:
        return spec.kind
    if spec.rows:
        return "table"
    if spec.images and (spec.bullets or spec.paragraphs):
        return "mixed"
    if spec.images:
        return "images"
    if spec.kind == "code":
        return "code"
    return "text"


def _can_merge_specs(a: SlideSpec, b: SlideSpec) -> bool:
    """Склеиваем соседние слайды одной подтемы, если влезает по лимитам."""
    if a.kind == "title" or b.kind == "title":
        return False
    if a.kind == "section" or b.kind == "section":
        return True
    if _spec_kind_bucket(a) == "table" and _spec_kind_bucket(b) == "table":
        return False
    if b.rows and len(b.rows) > TABLE_ROWS_MERGE:
        return False
    if a.rows and len(a.rows) > TABLE_ROWS_MERGE:
        return False
    if (a.rows and b.images) or (b.rows and a.images):
        return False
    if _spec_kind_bucket(a) == "code" or _spec_kind_bucket(b) == "code":
        return False
    if a.title.split(" (")[0] != b.title.split(" (")[0]:
        return False
    if (
        len(a.bullets) + len(b.bullets) > MERGE_MAX_BULLETS
        or len(a.paragraphs) + len(b.paragraphs) > MERGE_MAX_PARAS
        or len(a.images) + len(b.images) > MERGE_MAX_IMAGES
    ):
        return False
    return _spec_body_chars(a) + _spec_body_chars(b) <= MAX_PARA_CHARS * 2


def _merge_two_specs(a: SlideSpec, b: SlideSpec) -> SlideSpec:
    if a.kind == "section" and b.kind != "section":
        prefix = a.title
        if b.subtitle:
            b = SlideSpec(
                title=b.title,
                kind=b.kind,
                subtitle=f"{prefix} · {b.subtitle}",
                bullets=b.bullets,
                paragraphs=b.paragraphs,
                rows=b.rows,
                images=b.images,
                footer=b.footer,
            )
        else:
            b = SlideSpec(
                title=b.title,
                kind=b.kind,
                subtitle=prefix,
                bullets=b.bullets,
                paragraphs=b.paragraphs,
                rows=b.rows,
                images=b.images,
                footer=b.footer,
            )
        return b
    merged = SlideSpec(
        title=a.title,
        kind="mixed",
        subtitle=a.subtitle or b.subtitle,
        bullets=a.bullets + b.bullets,
        paragraphs=a.paragraphs + b.paragraphs,
        rows=a.rows if a.rows else b.rows,
        images=a.images + b.images,
        footer=b.footer or a.footer,
    )
    if merged.images and not (merged.bullets or merged.paragraphs):
        merged.kind = "images"
    elif merged.images:
        merged.kind = "mixed"
    elif merged.bullets and not merged.paragraphs:
        merged.kind = "bullets"
    else:
        merged.kind = "mixed"
    return merged


def compress_slides(slides: list[SlideSpec]) -> list[SlideSpec]:
    """Вторая проходка: раздел без контента, картинки+текст, микро-слайды."""
    out: list[SlideSpec] = []
    for spec in slides:
        if not spec.bullets and not spec.paragraphs and not spec.images and not spec.rows:
            if spec.kind == "section":
                out.append(spec)
                continue
        if out and _can_merge_specs(out[-1], spec):
            out[-1] = _merge_two_specs(out[-1], spec)
            continue
        out.append(spec)

    merged: list[SlideSpec] = []
    for spec in out:
        if (
            merged
            and _spec_body_chars(spec) < TINY_SLIDE_CHARS
            and spec.kind not in {"title", "section"}
            and _can_merge_specs(merged[-1], spec)
        ):
            merged[-1] = _merge_two_specs(merged[-1], spec)
        else:
            merged.append(spec)

    # Текст сразу перед чисто-картинковым с тем же заголовком.
    again: list[SlideSpec] = []
    i = 0
    while i < len(merged):
        spec = merged[i]
        if (
            i + 1 < len(merged)
            and spec.images == []
            and merged[i + 1].images
            and spec.title.split(" (")[0] == merged[i + 1].title.split(" (")[0]
            and _can_merge_specs(spec, merged[i + 1])
        ):
            again.append(_merge_two_specs(spec, merged[i + 1]))
            i += 2
            continue
        again.append(spec)
        i += 1
    return again


def _skip_closing_note(text: str) -> bool:
    """Хвост markdown «Конец презентационного…» на слайды не нужен."""
    t = text.lstrip("* ").lower()
    return t.startswith("конец презентационного") or _is_meta_content(text)


def _is_meta_content(text: str) -> bool:
    """Самореференция про сборку/черновик — выкидываем; domain-оговорки — нет."""
    low = re.sub(r"\s+", " ", text.lower().replace("**", "")).strip()
    if low.startswith("формат:") and "pptx" in low:
        return True
    if low.startswith("дата сборки:"):
        return True
    return any(frag in low for frag in _META_FRAGMENTS)


def _filter_meta_strings(items: list[str]) -> list[str]:
    return [x for x in items if x and not _is_meta_content(x) and not _skip_closing_note(x)]


def blocks_to_slides(blocks: list[Block]) -> list[SlideSpec]:
    """Группирует блоки в плотные, но читаемые слайды."""
    slides: list[SlideSpec] = []
    section = ""
    subsection = ""
    buffer_paras: list[str] = []
    buffer_bullets: list[str] = []
    buffer_images: list[tuple[str, Path | None]] = []
    pending_lead_in = ""
    current_title = "Облачность"

    def title_now() -> str:
        if subsection:
            return subsection
        if section:
            return section
        return current_title

    def take_lead_in() -> str:
        nonlocal pending_lead_in
        text = pending_lead_in
        pending_lead_in = ""
        return text

    def flush_buffers(*, force_title: str | None = None) -> None:
        """Сбрасывает текст и картинки; по возможности — один mixed-слайд."""
        nonlocal buffer_paras, buffer_bullets, buffer_images
        if not buffer_paras and not buffer_bullets and not buffer_images:
            return
        t = force_title or title_now()
        paras = _filter_meta_strings(list(buffer_paras))
        bullets = _filter_meta_strings(list(buffer_bullets))
        pending_imgs = list(buffer_images)
        buffer_paras = []
        buffer_bullets = []
        buffer_images = []
        if not paras and not bullets and not pending_imgs:
            return

        lead = take_lead_in()
        lead_used = False

        def subtitle_once() -> str:
            nonlocal lead_used
            if lead and not lead_used:
                lead_used = True
                return lead
            return ""

        pending_bullets = bullets
        chunk_paras: list[str] = []
        imgs_attached = False

        def take_images() -> list[tuple[str, Path | None]]:
            nonlocal imgs_attached
            if imgs_attached or not pending_imgs:
                return []
            chunk = pending_imgs[:MAX_IMAGES]
            del pending_imgs[: len(chunk)]
            imgs_attached = True
            return chunk

        def emit(ps: list[str], bs: list[str], imgs: list[tuple[str, Path | None]]) -> None:
            if not ps and not bs and not imgs:
                return
            if imgs and (ps or bs):
                kind = "mixed"
            elif imgs:
                kind = "images"
            elif bs and not ps:
                kind = "bullets"
            else:
                kind = "mixed"
            slides.append(
                SlideSpec(
                    title=t,
                    kind=kind,
                    subtitle=subtitle_once(),
                    paragraphs=ps,
                    bullets=bs,
                    images=imgs,
                )
            )

        for p in paras:
            if len(p) > MAX_PARA_CHARS:
                if chunk_paras or pending_bullets:
                    take_b = pending_bullets[:MAX_BULLETS]
                    pending_bullets = pending_bullets[MAX_BULLETS:]
                    emit(chunk_paras, take_b, take_images())
                    chunk_paras = []
                for part in _split_long_text(p, MAX_PARA_CHARS):
                    emit([part], [], take_images())
            else:
                chunk_paras.append(p)
                too_many = len(chunk_paras) >= MAX_PARA_PER_SLIDE
                too_long = sum(len(x) for x in chunk_paras) > MAX_PARA_CHARS
                if too_many or too_long:
                    take_b = pending_bullets[:MAX_BULLETS]
                    pending_bullets = pending_bullets[MAX_BULLETS:]
                    emit(chunk_paras, take_b, take_images())
                    chunk_paras = []

        while pending_bullets or chunk_paras or pending_imgs:
            if chunk_paras or pending_bullets:
                take_b = pending_bullets[:MAX_BULLETS]
                pending_bullets = pending_bullets[MAX_BULLETS:]
                emit(chunk_paras, take_b, take_images())
                chunk_paras = []
                continue
            if pending_imgs:
                emit([], [], pending_imgs[:MAX_IMAGES])
                del pending_imgs[:MAX_IMAGES]
                continue
            break

    # Титульный разбор: первый H1 + следующие para до следующего heading.
    i = 0
    if blocks and blocks[0].kind == "heading1":
        main_title = blocks[0].text
        current_title = main_title
        i = 1
        meta: list[str] = []
        while i < len(blocks) and blocks[i].kind in {"para", "list", "hr"}:
            b = blocks[i]
            if b.kind == "para":
                meta.append(b.text)
            elif b.kind == "list":
                meta.extend(b.items)
            i += 1
            if b.kind == "hr":
                break
        meta = _filter_meta_strings(meta)
        slides.append(
            SlideSpec(
                title=main_title,
                kind="title",
                subtitle=meta[0] if meta else "",
                paragraphs=meta[1:] if len(meta) > 1 else [],
            )
        )

    while i < len(blocks):
        b = blocks[i]

        if b.kind == "hr":
            flush_buffers()
            i += 1
            continue

        if b.kind == "heading1":
            flush_buffers()
            section = b.text
            subsection = ""
            current_title = b.text
            i += 1
            continue

        if b.kind == "heading2":
            flush_buffers()
            subsection = b.text
            current_title = b.text
            i += 1
            continue

        if b.kind == "heading3":
            subsection = b.text
            current_title = b.text
            i += 1
            continue

        if b.kind == "para":
            if _skip_closing_note(b.text) or _is_meta_content(b.text):
                i += 1
                continue
            buffer_paras.append(b.text)
            i += 1
            continue

        if b.kind == "list":
            buffer_bullets.extend(_filter_meta_strings(b.items))
            if len(buffer_bullets) >= MAX_BULLETS:
                flush_buffers()
            i += 1
            continue

        if b.kind == "table":
            flush_buffers()
            # Короткий ввод → subtitle таблицы; остальной текст — обычные слайды.
            if _is_lead_in(buffer_paras, buffer_bullets):
                pending_lead_in = " ".join(_filter_meta_strings(buffer_paras))
                buffer_paras = []
            else:
                flush_buffers()
            lead = take_lead_in()
            rows = b.rows
            header = rows[0] if rows else []
            body = rows[1:] if len(rows) > 1 else []
            max_body = MAX_TABLE_ROWS - 1
            for start in range(0, max(len(body), 1), max_body):
                chunk_body = body[start : start + max_body]
                chunk = [header] + chunk_body if header else chunk_body
                suffix = ""
                if len(body) > max_body:
                    page = start // max_body + 1
                    total = (len(body) + max_body - 1) // max_body
                    suffix = f" ({page}/{total})"
                slides.append(
                    SlideSpec(
                        title=title_now() + suffix,
                        kind="table",
                        subtitle=lead if start == 0 else "",
                        rows=chunk,
                    )
                )
            i += 1
            continue

        if b.kind == "image":
            if buffer_paras and _is_lead_in(buffer_paras, buffer_bullets):
                pending_lead_in = " ".join(_filter_meta_strings(buffer_paras))
                buffer_paras = []
            resolved = _resolve_image(b.path)
            caption = b.alt or Path(b.path).name
            buffer_images.append((caption, resolved))
            i += 1
            continue

        if b.kind == "code":
            if len(b.text) < 520:
                buffer_paras.append(b.text)
            else:
                flush_buffers()
                slides.append(
                    SlideSpec(
                        title=title_now(),
                        kind="code",
                        paragraphs=[b.text],
                    )
                )
            i += 1
            continue

        i += 1

    flush_buffers()
    return compress_slides(slides)


def _split_long_text(text: str, limit: int) -> list[str]:
    """Режет длинный абзац по предложениям / запятым."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    rest = text
    while rest:
        if len(rest) <= limit:
            parts.append(rest)
            break
        cut = rest.rfind(". ", 0, limit)
        if cut < limit // 3:
            cut = rest.rfind("; ", 0, limit)
        if cut < limit // 3:
            cut = rest.rfind(", ", 0, limit)
        if cut < limit // 3:
            cut = limit
        else:
            cut = cut + 1
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    return parts


def render_slides(prs: Presentation, specs: list[SlideSpec]) -> None:
    """Рисует слайды по спецификациям."""
    for spec in specs:
        if spec.kind == "title":
            _render_title(prs, spec)
        elif spec.kind == "section":
            _render_section(prs, spec)
        elif spec.kind == "table":
            _render_table(prs, spec)
        elif spec.kind == "mixed" and spec.rows:
            _render_mixed_with_table(prs, spec)
        elif spec.kind == "images":
            _render_images(prs, spec)
        elif spec.kind == "mixed":
            _render_mixed(prs, spec)
        elif spec.kind == "code":
            _render_code(prs, spec)
        else:
            _render_text(prs, spec)


def _blank(prs: Presentation):
    """Пустой слайд с фоном и accent bar."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _paint_slide_bg(slide, prs)
    _accent_bar(slide)
    return slide


def _render_title(prs: Presentation, spec: SlideSpec) -> None:
    slide = _blank(prs)
    box = slide.shapes.add_textbox(Inches(0.7), Inches(2.0), Inches(11.8), Inches(1.4))
    tf = box.text_frame
    tf.word_wrap = True
    _set_run(tf.paragraphs[0], spec.title, size=34, bold=True, color=INK)

    y = Inches(3.5)
    if spec.subtitle:
        box2 = slide.shapes.add_textbox(Inches(0.7), y, Inches(11.8), Inches(1.0))
        _set_run(box2.text_frame.paragraphs[0], spec.subtitle, size=16, color=TEAL)
        y = Inches(4.5)

    if spec.paragraphs:
        box3 = slide.shapes.add_textbox(Inches(0.7), y, Inches(11.8), Inches(2.2))
        tf3 = box3.text_frame
        tf3.word_wrap = True
        for j, para in enumerate(spec.paragraphs[:4]):
            p = tf3.paragraphs[0] if j == 0 else tf3.add_paragraph()
            _set_run(p, para, size=14, color=MUTED)
            p.space_after = Pt(8)

    _footer(slide, "Кейс СПУТНИКС · 2026-11-15 · сид 42")


def _render_section(prs: Presentation, spec: SlideSpec) -> None:
    slide = _blank(prs)
    box = slide.shapes.add_textbox(Inches(0.7), Inches(2.8), Inches(11.8), Inches(1.5))
    tf = box.text_frame
    tf.word_wrap = True
    _set_run(tf.paragraphs[0], spec.title, size=32, bold=True, color=TEAL)
    _footer(slide, "раздел")


def _render_text(prs: Presentation, spec: SlideSpec) -> None:
    slide = _blank(prs)
    top = _title_box(slide, spec.title, size=22, subtitle=spec.subtitle)
    height = 7.05 - top

    body = slide.shapes.add_textbox(MARGIN_X, Inches(top), CONTENT_W, Inches(height))
    tf = body.text_frame
    tf.word_wrap = True
    first = True

    for para in spec.paragraphs:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        _set_run(p, para, size=15, color=INK)
        p.space_after = Pt(10)

    for bullet in spec.bullets:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        size = 14 if len(bullet) > 160 else 15
        _set_run(p, "•  " + bullet, size=size, color=INK)
        p.space_after = Pt(7)

    _footer(slide, spec.footer)


def _render_code(prs: Presentation, spec: SlideSpec) -> None:
    slide = _blank(prs)
    top = _title_box(slide, spec.title, size=22, subtitle=spec.subtitle)
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        MARGIN_X,
        Inches(top),
        CONTENT_W,
        Inches(6.9 - top),
    )
    card.line.fill.background()
    _rgb_fill(card, CARD)
    box = slide.shapes.add_textbox(
        Inches(0.65), Inches(top + 0.2), Inches(12.0), Inches(6.5 - top)
    )
    tf = box.text_frame
    tf.word_wrap = True
    code = spec.paragraphs[0] if spec.paragraphs else ""
    _set_run(tf.paragraphs[0], code, size=13, color=INK, font="Consolas")
    _footer(slide, "воспроизведение")


def _set_cell(cell, text: str, *, bold: bool = False, size: int = 11, header: bool = False) -> None:
    """Заполняет ячейку таблицы."""
    cell.text = ""
    p = cell.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = "Calibri"
    run.font.color.rgb = INK
    cell.text_frame.word_wrap = True
    # Фон
    fill = TABLE_HEADER_BG if header else CARD
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill


def _render_table(prs: Presentation, spec: SlideSpec) -> None:
    slide = _blank(prs)
    top_y = _title_box(slide, spec.title, size=20, subtitle=spec.subtitle)

    rows = spec.rows
    if not rows:
        return
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    # Выравниваем ширину строк.
    norm = [r + [""] * (n_cols - len(r)) for r in rows]

    # Высота таблицы под число строк.
    top = Inches(top_y)
    height = min(Inches(7.05 - top_y), Inches(0.42) * n_rows + Inches(0.2))
    table_shape = slide.shapes.add_table(n_rows, n_cols, MARGIN_X, top, CONTENT_W, height)
    table = table_shape.table

    # Ширина колонок: первая чуть шире для имён.
    total = CONTENT_W
    if n_cols == 2:
        widths = [int(total * 0.35), int(total * 0.65)]
    elif n_cols >= 6:
        widths = [int(total / n_cols)] * n_cols
    else:
        first = int(total * 0.22)
        rest = int((total - first) / (n_cols - 1))
        widths = [first] + [rest] * (n_cols - 1)
    for idx, w in enumerate(widths):
        table.columns[idx].width = w

    for r_i, row in enumerate(norm):
        for c_i, cell_text in enumerate(row):
            is_header = r_i == 0
            # Чуть мельче для широких таблиц.
            size = 10 if n_cols >= 7 else (11 if n_cols >= 5 else 12)
            _set_cell(
                table.cell(r_i, c_i),
                cell_text,
                bold=is_header,
                size=size,
                header=is_header,
            )
            if not is_header and r_i % 2 == 0:
                table.cell(r_i, c_i).fill.solid()
                table.cell(r_i, c_i).fill.fore_color.rgb = TABLE_ALT

    _footer(slide, "таблица")


def _place_images_on_slide(
    slide,
    spec: SlideSpec,
    *,
    area_left: float,
    area_top: float,
    area_w: float,
    area_h: float,
) -> list[str]:
    """Вставляет картинки в прямоугольник; возвращает подписи пропущенных файлов."""
    images = spec.images
    n = len(images)
    missing: list[str] = []
    if n == 0:
        return missing

    if n == 1:
        slots = [(area_left + area_w * 0.08, area_top, area_w * 0.84, area_h * 0.92)]
    elif n == 2:
        w = area_w * 0.48
        slots = [
            (area_left, area_top, w, area_h * 0.92),
            (area_left + area_w * 0.52, area_top, w, area_h * 0.92),
        ]
    elif n == 3:
        w = area_w * 0.31
        slots = [
            (area_left, area_top, w, area_h * 0.92),
            (area_left + area_w * 0.345, area_top, w, area_h * 0.92),
            (area_left + area_w * 0.69, area_top, w, area_h * 0.92),
        ]
    else:
        w = area_w * 0.48
        h = area_h * 0.46
        slots = [
            (area_left, area_top, w, h),
            (area_left + area_w * 0.52, area_top, w, h),
            (area_left, area_top + area_h * 0.5, w, h),
            (area_left + area_w * 0.52, area_top + area_h * 0.5, w, h),
        ]

    for (caption, path), (x, y, w, h) in zip(images, slots):
        if path is None or not path.is_file():
            missing.append(caption)
            ph = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                int(x),
                int(y),
                int(w),
                int(h),
            )
            ph.line.color.rgb = LINE
            _rgb_fill(ph, CARD)
            tb = slide.shapes.add_textbox(
                int(x) + Inches(0.1),
                int(y) + Inches(h / 2 - 0.2),
                int(w) - Inches(0.2),
                Inches(0.6),
            )
            tf = tb.text_frame
            tf.word_wrap = True
            _set_run(tf.paragraphs[0], f"[нет файла] {caption}", size=11, color=CORAL)
            continue
        try:
            from PIL import Image

            with Image.open(path) as im:
                iw, ih = im.size
            scale = min(w / iw, h / ih)
            pw, ph = iw * scale, ih * scale
            px = x + (w - pw) / 2
            py = y + (h - ph) / 2
            slide.shapes.add_picture(str(path), int(px), int(py), width=int(pw), height=int(ph))
        except Exception:
            slide.shapes.add_picture(str(path), int(x), int(y), width=int(w))
    return missing


def _render_mixed_with_table(prs: Presentation, spec: SlideSpec) -> None:
    """Текст/буллеты + компактная таблица на одном слайде."""
    slide = _blank(prs)
    top_y = _title_box(slide, spec.title, size=19, subtitle=spec.subtitle)
    text_h = 2.15 if spec.images else 2.6
    body = slide.shapes.add_textbox(
        MARGIN_X, Inches(top_y), CONTENT_W, Inches(text_h)
    )
    tf = body.text_frame
    tf.word_wrap = True
    first = True
    for para in spec.paragraphs[:3]:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        _set_run(p, para, size=12, color=INK)
        p.space_after = Pt(6)
    for bullet in spec.bullets[:8]:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        _set_run(p, "•  " + bullet, size=12, color=INK)
        p.space_after = Pt(4)

    tbl_top = top_y + text_h + 0.08
    rows = spec.rows
    if rows:
        n_rows = len(rows)
        n_cols = max(len(r) for r in rows)
        norm = [r + [""] * (n_cols - len(r)) for r in rows]
        height = min(Inches(7.0 - tbl_top), Inches(0.36) * n_rows + Inches(0.15))
        table_shape = slide.shapes.add_table(
            n_rows, n_cols, MARGIN_X, Inches(tbl_top), CONTENT_W, height
        )
        table = table_shape.table
        for r_i, row in enumerate(norm):
            for c_i, cell_text in enumerate(row):
                is_header = r_i == 0
                size = 10 if n_cols >= 6 else 11
                _set_cell(
                    table.cell(r_i, c_i),
                    cell_text,
                    bold=is_header,
                    size=size,
                    header=is_header,
                )

    if spec.images:
        mini = SlideSpec(
            title=spec.title,
            kind="images",
            images=spec.images[:2],
        )
        _place_images_on_slide(
            slide,
            mini,
            area_left=float(MARGIN_X) + float(CONTENT_W) * 0.62,
            area_top=float(Inches(top_y)),
            area_w=float(CONTENT_W) * 0.35,
            area_h=float(Inches(text_h)),
        )
    _footer(slide, spec.footer)


def _render_mixed(prs: Presentation, spec: SlideSpec) -> None:
    """Текст слева, картинки справа (или снизу, если текста мало)."""
    slide = _blank(prs)
    top_y = _title_box(slide, spec.title, size=20, subtitle=spec.subtitle)
    has_text = bool(spec.paragraphs or spec.bullets)

    if has_text:
        text_w = float(CONTENT_W * 0.46)
        body = slide.shapes.add_textbox(
            MARGIN_X,
            Inches(top_y),
            int(text_w),
            Inches(7.05 - top_y),
        )
        tf = body.text_frame
        tf.word_wrap = True
        first = True
        for para in spec.paragraphs:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            _set_run(p, para, size=13, color=INK)
            p.space_after = Pt(8)
        for bullet in spec.bullets:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            size = 12 if len(bullet) > 160 else 13
            _set_run(p, "•  " + bullet, size=size, color=INK)
            p.space_after = Pt(5)
        img_left = float(MARGIN_X) + text_w + float(Inches(0.15))
        img_w = float(CONTENT_W) - text_w - float(Inches(0.15))
        missing = _place_images_on_slide(
            slide,
            spec,
            area_left=img_left,
            area_top=float(Inches(top_y)),
            area_w=img_w,
            area_h=float(Inches(7.05 - top_y)),
        )
    else:
        missing = _place_images_on_slide(
            slide,
            spec,
            area_left=float(MARGIN_X),
            area_top=float(Inches(top_y)),
            area_w=float(CONTENT_W),
            area_h=float(Inches(7.05 - top_y)),
        )

    note = ""
    if missing:
        note = "пропущены: " + "; ".join(missing[:2])
    _footer(slide, note[:120])


def _render_images(prs: Presentation, spec: SlideSpec) -> None:
    slide = _blank(prs)
    top_y = _title_box(slide, spec.title, size=20, subtitle=spec.subtitle)

    if not spec.images:
        return
    missing = _place_images_on_slide(
        slide,
        spec,
        area_left=float(MARGIN_X),
        area_top=float(Inches(top_y)),
        area_w=float(CONTENT_W),
        area_h=float(Inches(7.05 - top_y)),
    )

    note = ""
    if missing:
        note = "пропущены: " + "; ".join(missing[:3])
    elif spec.images:
        note = " · ".join(c for c, _ in spec.images[:4])
    _footer(slide, note[:120])


def main() -> None:
    """Читает PRESENTATION.md и пишет cloudness_case.pptx."""
    if not MD_PATH.is_file():
        raise FileNotFoundError(MD_PATH)

    md = MD_PATH.read_text(encoding="utf-8")
    blocks = parse_markdown(md)
    specs = blocks_to_slides(blocks)

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    render_slides(prs, specs)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PATH))

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"path={OUT_PATH}")
    print(f"slides={len(prs.slides)}")
    print(f"size_mb={size_mb:.2f}")
    print(f"blocks={len(blocks)} specs={len(specs)}")

    # Spot-check: первые и последние слайды имеют текст.
    def slide_text(slide) -> str:
        parts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
        return "\n".join(parts).strip()

    n = len(prs.slides)
    for idx in list(range(min(3, n))) + list(range(max(0, n - 3), n)):
        text = slide_text(prs.slides[idx])
        preview = text.replace("\n", " | ")[:120]
        print(f"slide[{idx}] chars={len(text)} :: {preview}")


if __name__ == "__main__":
    main()
