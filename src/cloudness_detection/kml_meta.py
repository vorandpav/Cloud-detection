"""Разбор KML-метаданных снимков СПУТНИКС."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_COORD_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
)
_NAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_(?P<board>SXC3-[^_]+)_"
)


@dataclass(frozen=True)
class PhotoMeta:
    """Метаданные одной пары JPG+KML."""

    stem: str
    jpg_path: Path
    kml_path: Path
    board_id: str
    captured_at: datetime
    center_lon: float
    center_lat: float


def parse_latlon_quad_center(kml_path: Path) -> tuple[float, float]:
    """Читает углы gx:LatLonQuad и возвращает центр кадра.

    Args:
        kml_path: Путь к KML рядом с JPG.

    Returns:
        Пара (longitude, latitude) центра четырёхугольника.

    Raises:
        ValueError: Если в KML нет четырёх координат углов.
    """
    tree = ET.parse(kml_path)
    text_chunks: list[str] = []
    for elem in tree.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "coordinates" and elem.text:
            text_chunks.append(elem.text)

    if not text_chunks:
        raise ValueError(f"Нет coordinates в {kml_path}")

    pairs = _COORD_RE.findall(text_chunks[0])
    if len(pairs) < 4:
        raise ValueError(f"Ожидали 4 угла, получили {len(pairs)} в {kml_path}")

    lons = [float(p[0]) for p in pairs[:4]]
    lats = [float(p[1]) for p in pairs[:4]]
    return sum(lons) / 4.0, sum(lats) / 4.0


def parse_stem(stem: str) -> tuple[str, datetime]:
    """Достаёт board id и время съёмки из имени файла.

    Args:
        stem: Имя без расширения, например
            ``2026-11-14_07-37-27_SXC3-2352_1``.

    Returns:
        (board_id, datetime в UTC-наивном виде из имени файла).

    Raises:
        ValueError: Если имя не совпало с ожидаемым шаблоном.
    """
    match = _NAME_RE.match(stem)
    if not match:
        raise ValueError(f"Не разобрали имя файла: {stem}")
    captured_at = datetime.strptime(
        f"{match.group('date')} {match.group('time')}",
        "%Y-%m-%d %H-%M-%S",
    )
    return match.group("board"), captured_at


def iter_photo_meta(photos_dir: Path) -> list[PhotoMeta]:
    """Собирает метаданные по всем JPG, у которых есть парный KML.

    Args:
        photos_dir: Папка ``Photos/`` со снимками СПУТНИКС.

    Returns:
        Список ``PhotoMeta``. Битые пары пропускаются с печатью в stdout.
    """
    items: list[PhotoMeta] = []
    for jpg_path in sorted(photos_dir.glob("*.jpg")):
        kml_path = jpg_path.with_suffix(".kml")
        if not kml_path.exists():
            print(f"skip: нет KML для {jpg_path.name}")
            continue
        try:
            board_id, captured_at = parse_stem(jpg_path.stem)
            lon, lat = parse_latlon_quad_center(kml_path)
        except ValueError as exc:
            print(f"skip: {exc}")
            continue
        items.append(
            PhotoMeta(
                stem=jpg_path.stem,
                jpg_path=jpg_path,
                kml_path=kml_path,
                board_id=board_id,
                captured_at=captured_at,
                center_lon=lon,
                center_lat=lat,
            )
        )
    return items


def lat_band(lat: float) -> str:
    """Грубая широтная корзина для стратификации.

    Args:
        lat: Широта центра кадра в градусах.

    Returns:
        Имя корзины: polar / mid / tropics.
    """
    abs_lat = abs(lat)
    if abs_lat >= 60:
        return "polar"
    if abs_lat >= 30:
        return "mid"
    return "tropics"
