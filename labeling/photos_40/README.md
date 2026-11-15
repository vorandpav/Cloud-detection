# Ручная разметка: 40 кадров Photos

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
- Отчёт baseline thr + pseudo-FT (held-out 8 не в train): `outputs/reports/photos_thr_and_pseudo_ft/REPORT.md`.
