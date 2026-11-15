"""Стратифицированный отбор 40 снимков Photos для ручной разметки."""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
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


def main() -> None:
    """Собирает ``labeling/photos_40/``: превью, манифест, шаблон меток."""
    set_global_seed(DEFAULT_SEED)
    rng = np.random.default_rng(DEFAULT_SEED)

    if not PHOTOS_DIR.is_dir():
        raise FileNotFoundError(f"Нет папки Photos: {PHOTOS_DIR}")

    print("Читаю KML…")
    metas = iter_photo_meta(PHOTOS_DIR)
    print(f"Кадров с KML: {len(metas)}")

    print("Считаю прокси-белизну (это не метки)…")
    white_fracs: dict[str, float] = {}
    for i, meta in enumerate(metas, start=1):
        white_fracs[meta.stem] = white_fraction(meta.jpg_path)
        if i % 500 == 0:
            print(f"  {i}/{len(metas)}")

    selected = stratified_sample(metas, white_fracs, N_SELECT, rng)
    selected = sorted(selected, key=lambda m: m.stem)

    images_dir = OUT_DIR / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = OUT_DIR / "manifest.csv"
    labels_path = OUT_DIR / "labels_template.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as mf, labels_path.open(
        "w", newline="", encoding="utf-8"
    ) as lf:
        manifest_writer = csv.DictWriter(
            mf,
            fieldnames=[
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
            ],
        )
        labels_writer = csv.DictWriter(
            lf,
            fieldnames=["id", "stem", "cloud_pct", "reject_75", "notes"],
        )
        manifest_writer.writeheader()
        labels_writer.writeheader()

        for idx, meta in enumerate(selected, start=1):
            preview_name = f"{idx:02d}_{meta.stem}.jpg"
            preview_path = images_dir / preview_name
            write_thumbnail(meta.jpg_path, preview_path)
            wf = white_fracs[meta.stem]
            manifest_writer.writerow(
                {
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
            )
            labels_writer.writerow(
                {
                    "id": idx,
                    "stem": meta.stem,
                    "cloud_pct": "",
                    "reject_75": "",
                    "notes": "",
                }
            )

    readme = OUT_DIR / "README.md"
    readme.write_text(
        """# Ручная разметка: 40 кадров Photos

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

## Важно

- `white_frac_proxy` в манифесте — **не** твоя метка, а грубый автоматический прокси для стратификации выборки.
- Сид выборки: **42** (скрипт `select-photos-40`).
""",
        encoding="utf-8",
    )

    print(f"Готово: {len(selected)} кадров → {OUT_DIR}")
    print(f"Смотри превью: {images_dir}")
    print(f"Пиши метки в: {labels_path}")


if __name__ == "__main__":
    main()
