"""Стратифицированный отбор снимков Photos для ручной разметки.

По умолчанию собирает ``labeling/photos_40/``.
``--n 100`` собирает ``labeling/photos_100/`` тем же сидом:
первые 40 отобранных кадров совпадают с ``photos_40``, остальные новые.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from cloudness_detection.kml_meta import PhotoMeta, iter_photo_meta, lat_band
from cloudness_detection.seed import DEFAULT_SEED, set_global_seed

ROOT = Path(__file__).resolve().parents[2]
PHOTOS_DIR = ROOT / "Photos"
OUT_DIR = ROOT / "labeling" / "photos_40"
N_SELECT = 40
THUMB_MAX_SIDE = 768


def white_fraction(jpg_path: Path, threshold: int = 200) -> float:
    """Доля ярких пикселей как грубый прокси «белизны» кадра.

    Это не разметка облаков. Нужна только чтобы в выборку попали
    и тёмные/ясные, и очень светлые кадры.

    Args:
        jpg_path: Путь к JPEG.
        threshold: Порог яркости (0..255) после перевода в grayscale.

    Returns:
        Число от 0 до 1.
    """
    with Image.open(jpg_path) as img:
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
    # Считаем по уменьшенной копии — быстрее на 3280×2464.
    small = np.asarray(
        Image.fromarray(gray).resize((320, 240), Image.Resampling.BILINEAR)
    )
    return float((small >= threshold).mean())


def _white_fraction_stem(jpg_path_str: str) -> tuple[str, float]:
    """Считает прокси-белизну одного кадра. Нужна для пула потоков.

    Args:
        jpg_path_str: Путь к JPEG строкой (так проще отдать в executor).

    Returns:
        Пара (stem, доля ярких пикселей).
    """
    path = Path(jpg_path_str)
    return path.stem, white_fraction(path)


def compute_white_fracs(metas: list[PhotoMeta], workers: int) -> dict[str, float]:
    """Считает прокси-белизну для всех кадров.

    Args:
        metas: Кадры с валидным KML.
        workers: Сколько потоков. 1 — последовательно, как раньше.

    Returns:
        stem -> доля ярких пикселей.
    """
    white_fracs: dict[str, float] = {}
    total = len(metas)
    if workers <= 1:
        for i, meta in enumerate(metas, start=1):
            white_fracs[meta.stem] = white_fraction(meta.jpg_path)
            if i % 500 == 0:
                print(f"  {i}/{total}")
        return white_fracs

    paths = [str(meta.jpg_path) for meta in metas]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_white_fraction_stem, path) for path in paths]
        done = 0
        for fut in as_completed(futures):
            stem, frac = fut.result()
            white_fracs[stem] = frac
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{total}")
    return white_fracs


def load_photos_40_stems(root: Path) -> tuple[set[str], set[str]]:
    """Читает, какие кадры уже лежат в photos_40 и у каких есть маска.

    Args:
        root: Корень проекта.

    Returns:
        (stems из manifest.csv, stems, для которых есть PNG в masks/).
    """
    manifest = root / "labeling" / "photos_40" / "manifest.csv"
    stems: set[str] = set()
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            stems = {row["stem"] for row in csv.DictReader(handle)}

    masked: set[str] = set()
    mask_dir = root / "labeling" / "photos_40" / "masks"
    if mask_dir.is_dir():
        for path in mask_dir.glob("*.png"):
            # Имена вида 01_<stem>.png
            masked.add(path.stem.split("_", 1)[1])
    return stems, masked


def white_bin(frac: float) -> str:
    """Корзина по прокси-белизне.

    Args:
        frac: Доля ярких пикселей.

    Returns:
        low / mid / high.
    """
    if frac < 0.15:
        return "low"
    if frac < 0.45:
        return "mid"
    return "high"


def stratum_key(meta: PhotoMeta, white_frac: float) -> str:
    """Ключ страты: широта × сезон × белизна.

    Args:
        meta: Метаданные кадра.
        white_frac: Результат ``white_fraction``.

    Returns:
        Строковый ключ для группировки.
    """
    season = f"m{meta.captured_at.month:02d}"
    return f"{lat_band(meta.center_lat)}|{season}|{white_bin(white_frac)}"


def stratified_sample(
    metas: list[PhotoMeta],
    white_fracs: dict[str, float],
    n: int,
    rng: np.random.Generator,
) -> list[PhotoMeta]:
    """Берёт примерно поровну из каждой непустой страты.

    Args:
        metas: Все кадры с валидным KML.
        white_fracs: stem -> доля ярких пикселей.
        n: Сколько кадров нужно (обычно 40).
        rng: NumPy Generator с зафиксированным сидом.

    Returns:
        Список из ``n`` ``PhotoMeta`` (или меньше, если данных мало).
    """
    buckets: dict[str, list[PhotoMeta]] = defaultdict(list)
    for meta in metas:
        buckets[stratum_key(meta, white_fracs[meta.stem])].append(meta)

    keys = sorted(buckets.keys())
    for key in keys:
        rng.shuffle(buckets[key])

    chosen: list[PhotoMeta] = []
    chosen_stems: set[str] = set()
    while len(chosen) < n:
        progressed = False
        for key in keys:
            if len(chosen) >= n:
                break
            bucket = buckets[key]
            while bucket:
                candidate = bucket.pop()
                if candidate.stem in chosen_stems:
                    continue
                chosen.append(candidate)
                chosen_stems.add(candidate.stem)
                progressed = True
                break
        if not progressed:
            break

    # Добиваем случайными остатками, если страт мало.
    if len(chosen) < n:
        rest = [m for m in metas if m.stem not in chosen_stems]
        rng.shuffle(rest)
        for meta in rest:
            if len(chosen) >= n:
                break
            chosen.append(meta)
            chosen_stems.add(meta.stem)

    return chosen[:n]


def write_thumbnail(src: Path, dst: Path) -> None:
    """Копирует уменьшенный JPG для удобной разметки глазами.

    Args:
        src: Исходный полный кадр.
        dst: Куда сохранить превью.
    """
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), Image.Resampling.LANCZOS)
        img.save(dst, quality=90)


def parse_args() -> argparse.Namespace:
    """Разбирает размер выборки и папку выхода.

    Returns:
        Аргументы командной строки. Без флагов это прежний набор на 40.
    """
    parser = argparse.ArgumentParser(
        description="Стратифицированный отбор кадров Photos для ручной разметки."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N_SELECT,
        help="Сколько кадров отобрать (40 → labeling/photos_40, 100 → labeling/photos_100).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Папка набора. По умолчанию labeling/photos_<n>.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Потоки для прокси-белизны. На результат выборки не влияет.",
    )
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n должен быть >= 1")
    if args.workers < 1:
        parser.error("--workers должен быть >= 1")
    return args


def readme_text(n: int, n_old: int, n_masked: int) -> str:
    """Текст README для папки разметки.

    Args:
        n: Размер нового набора.
        n_old: Сколько кадров уже были в photos_40.
        n_masked: Сколько из них уже имеют маску CVAT.

    Returns:
        Markdown.
    """
    script = "select-photos-40" if n == N_SELECT else f"select-photos-40 --n {n}"
    overlap = ""
    if n != N_SELECT:
        overlap = f"""
## Связь с photos_40

- Сид тот же (**42**), поэтому выборка на {n} — расширение выборки на 40, а не другая случайная пачка.
- В этом наборе **{n_old}** кадров уже были в `labeling/photos_40/`.
- У **{n_masked}** из них уже есть маска CVAT. В `labels_template.csv` у них в `notes` стоит `already_masked_in_photos_40` — эти можно не размечать заново.
- Остальные кадры новые. Колонка `in_photos_40` в `manifest.csv`: `1` если кадр из старого набора.
"""
    return f"""# Ручная разметка: {n} кадров Photos

## Где смотреть картинки

Папка `images/` — уменьшенные превью (удобно листать).
Полные оригиналы лежат в `Photos/` (путь в `manifest.csv` → `source_jpg`).

## Что заполнить

Открой `labels_template.csv` и для каждой строки:

| Поле | Что писать |
|------|------------|
| `cloud_pct` | Оценка доли облаков на кадре, целое **0…100** |
| `reject_75` | `1` если cloud_pct > 75, иначе `0` (можно заполнить формулой потом) |
| `notes` | Опционально: `snow`, `haze`, `desert`, `ocean`, `night`, `hard` |

Сохрани файл как `labels.csv` рядом (или просто допиши `labels_template.csv`).
{overlap}
## Важно

- `white_frac_proxy` в манифесте — **не** твоя метка, а грубый автоматический прокси для стратификации выборки.
- Сид выборки: **42** (скрипт `{script}`).
"""


def main() -> None:
    """Собирает ``labeling/photos_<n>/``: превью, манифест, шаблон меток."""
    args = parse_args()
    n_select = args.n
    out_dir = args.out if args.out is not None else ROOT / "labeling" / f"photos_{n_select}"
    track_old = n_select != N_SELECT
    old_stems, masked_stems = load_photos_40_stems(ROOT) if track_old else (set(), set())

    set_global_seed(DEFAULT_SEED)
    rng = np.random.default_rng(DEFAULT_SEED)

    if not PHOTOS_DIR.is_dir():
        raise FileNotFoundError(f"Нет папки Photos: {PHOTOS_DIR}")

    print("Читаю KML…")
    metas = iter_photo_meta(PHOTOS_DIR)
    print(f"Кадров с KML: {len(metas)}")

    print("Считаю прокси-белизну (это не метки)…")
    white_fracs = compute_white_fracs(metas, args.workers)

    selected = stratified_sample(metas, white_fracs, n_select, rng)
    selected = sorted(selected, key=lambda m: m.stem)

    images_dir = out_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    labels_path = out_dir / "labels_template.csv"
    id_width = 3 if n_select >= 100 else 2
    manifest_fields = [
        "id",
        "stem",
        "source_jpg",
        "board_id",
        "captured_at",
        "center_lon",
        "center_lat",
        "lat_band",
        "white_frac_proxy",
        "stratum",
        "preview_jpg",
    ]
    if track_old:
        manifest_fields.append("in_photos_40")

    n_old = 0
    n_masked = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as mf, labels_path.open(
        "w", newline="", encoding="utf-8"
    ) as lf:
        manifest_writer = csv.DictWriter(mf, fieldnames=manifest_fields)
        labels_writer = csv.DictWriter(
            lf,
            fieldnames=["id", "stem", "cloud_pct", "reject_75", "notes"],
        )
        manifest_writer.writeheader()
        labels_writer.writeheader()

        for idx, meta in enumerate(selected, start=1):
            preview_name = f"{idx:0{id_width}d}_{meta.stem}.jpg"
            preview_path = images_dir / preview_name
            write_thumbnail(meta.jpg_path, preview_path)
            wf = white_fracs[meta.stem]
            in_old = meta.stem in old_stems
            has_mask = meta.stem in masked_stems
            if in_old:
                n_old += 1
            if has_mask:
                n_masked += 1
            row = {
                "id": idx,
                "stem": meta.stem,
                "source_jpg": str(meta.jpg_path),
                "board_id": meta.board_id,
                "captured_at": meta.captured_at.isoformat(sep=" "),
                "center_lon": f"{meta.center_lon:.6f}",
                "center_lat": f"{meta.center_lat:.6f}",
                "lat_band": lat_band(meta.center_lat),
                "white_frac_proxy": f"{wf:.4f}",
                "stratum": stratum_key(meta, wf),
                "preview_jpg": str(preview_path.relative_to(ROOT)),
            }
            if track_old:
                row["in_photos_40"] = "1" if in_old else "0"
            manifest_writer.writerow(row)
            notes = "already_masked_in_photos_40" if has_mask else ""
            labels_writer.writerow(
                {
                    "id": idx,
                    "stem": meta.stem,
                    "cloud_pct": "",
                    "reject_75": "",
                    "notes": notes,
                }
            )

    readme = out_dir / "README.md"
    readme.write_text(
        readme_text(n_select, n_old, n_masked),
        encoding="utf-8",
    )

    print(f"Готово: {len(selected)} кадров → {out_dir}")
    if track_old:
        print(f"Из них уже в photos_40: {n_old}, с маской CVAT: {n_masked}")
        print(f"Новых кадров: {len(selected) - n_old}")
    print(f"Смотри превью: {images_dir}")
    print(f"Пиши метки в: {labels_path}")


if __name__ == "__main__":
    main()
