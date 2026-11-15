# Датасеты — итоговый разбор (canonical)

Дата: **2026-11-15**.  
Точка входа по данным. Подробный EDA-черновик: [`DATASETS_REPORT.md`](DATASETS_REPORT.md).  
Скачивание: [`DOWNLOAD_NOW.md`](DOWNLOAD_NOW.md) · модели: [`MODEL_SELECTION_REPORT.md`](MODEL_SELECTION_REPORT.md) · пайплайн: [`PIPELINE.md`](PIPELINE.md).

Цифры EDA: [`../outputs/eda/datasets/summary.json`](../outputs/eda/datasets/summary.json) (seed **42**).  
Глубокий сравнительный EDA: [`EDA_DEEP.md`](EDA_DEEP.md) · [`../outputs/eda/deep/`](../outputs/eda/deep/).

---

## Короткий вердикт

| Датасет | На диске | Качество масок | Роль |
|---------|----------|----------------|------|
| **Photos** | да (`Photos/`, `labeling/photos_40/`) | пока нет pixel GT | **целевой** продукт |
| **38-Cloud** | да | **посредственное** | объём pretrain, низкий вес |
| **SPARCS** | да (`data/sending/` = `data/sparcs/`) | **хорошее** | pretrain / holdout |
| **HRC_WHU** | да | **хорошее** | pretrain, высокий вес |
| **GF1 pack** | да | **посредственное** (как 38-Cloud) | низкий вес / eval |
| **S2 Catalogue** | **на диске** `data/s2_catalogue/` (masks+subscenes, 513) | хорошее (экспертное) | другой сенсор, mid weight |
| 95-Cloud / CloudSEN12 full / Levir | нет | — | не берём сейчас |

---

## 1. Photos (целевой)

**Сводка.** RGB JPG обзорных камер кубсатов СПУТНИКС, GSD ~200 м/пкс. Продукт: **% облачности** + **reject@75**.

| Поле | Значение |
|------|----------|
| Объём | **4272** JPG; labeling **40** кадров |
| Диск | ~**33.3 GB** |
| Маски | нет (нужна ручная разметка `photos_40`) |
| EDA | white_frac mean ≈ **0.069** (прокси, не GT) |

![Photos](../outputs/eda/datasets/photos/sample_1.png)

**Почему взяли:** это домен сдачи. Публичные сеты — только мост к нему.

---

## 2. 38-Cloud

**Сводка.** Landsat-8 патчи 384², классический бенчмарк Cloud-Net.

| Поле | Значение |
|------|----------|
| GSD | ~30 м |
| Train GT | **8400** (у нас RGB без NIR) |
| Диск | ~**8.6 GB** |
| Cloud frac | mean ≈ **0.50** (n=398 nonempty) |
| **Качество разметки** | **посредственное** — грубые границы, артефакты краёв сцен |

![38-Cloud](../outputs/eda/datasets/38-cloud/sample_1.png)
![38-Cloud overlay](../outputs/eda/datasets/38-cloud/sample_1_overlay.png)

**Роль:** большой labeled RGB-объём; в миксе **меньший вес**, не «учитель истины».

---

## 3. SPARCS

**Сводка.** USGS validation, 80 сцен Landsat, multi-class маски. Файлы в `data/sending/`; `data/sparcs/` — junction туда же.

| Поле | Значение |
|------|----------|
| GSD | ~30 м |
| Сцены | **80** |
| RGB | `*_photo.png` |
| Cloud frac | mean ≈ **0.25**, median ≈ **0.17** |
| **Качество разметки** | **хорошее** (валидационный сет) |

![SPARCS](../outputs/eda/datasets/sparcs/sample_1.png)
![SPARCS overlay](../outputs/eda/datasets/sparcs/sample_1_overlay.png)

**Роль:** высокий вес в pretrain + анализ ошибок (shadow/snow).

---

## 4. HRC_WHU

**Сводка.** RGB 1280×720, 150 сцен, mmseg-раскладка. На нём меряют Cloud-Adapter / NFCNet.

| Поле | Значение |
|------|----------|
| GSD | 0.5–15 м |
| Пары | train **120** + test **30** |
| Диск | ~**110 MB** |
| Cloud frac | mean ≈ **0.34** |
| **Качество разметки** | **хорошее** |

![HRC](../outputs/eda/datasets/hrc_whu/sample_1.png)
![HRC overlay](../outputs/eda/datasets/hrc_whu/sample_1_overlay.png)

**Роль:** якорь качественных масок, **высокий вес**.

---

## 5. GF1 (`gf12ms_whu_gf1`)

**Сводка.** Pack Cloud-Adapter, PNG 256², train/val.

| Поле | Значение |
|------|----------|
| Пары | **10428** |
| Диск | ~**0.9 GB** |
| Cloud frac | mean ≈ **0.09**, median ≈ **0.001** (много clear) |
| **Качество разметки** | **посредственное** — по уровню близко к 38-Cloud: шумные/грубые границы, не эталон для Photos |

![GF1](../outputs/eda/datasets/gf12ms_whu_gf1/sample_1.png)
![GF1 overlay](../outputs/eda/datasets/gf12ms_whu_gf1/sample_1_overlay.png)

**Роль:** строка «как у Cloud-Adapter» в таблице; в train — **низкий вес** или только eval. Не путать «есть в paper» с «хорошие labels».

---

## 6. Sentinel-2 Cloud Mask Catalogue

**Сводка.** Francis et al., Zenodo 4172871. Thumbs уже были; **masks.zip + subscenes.zip** положены в `data/s2_catalogue/`.

| Поле | Значение |
|------|----------|
| Размер кадра | **1022²**, 13 bands `.npy` |
| Маски | CLEAR / CLOUD / SHADOW (bool HxWx3) |
| Сцен | **513** (masks + subscenes) |
| Thumbs | `data/s2_catalogue_thumbs/` + junction |
| **Качество разметки** | **хорошее** (экспертный каталог) |

RGB из subscenes = **B4,B3,B2** (percentile 2–98) + overlay канала CLOUD:

![S2 RGB](../outputs/eda/datasets/s2_catalogue/sample_1.png)
![S2 overlay](../outputs/eda/datasets/s2_catalogue/sample_1_overlay.png)

![S2 cloud frac](../outputs/eda/datasets/s2_catalogue/cloud_fraction_hist.png)

**Роль:** другой сенсор в миксе; mid weight. Полные subscenes тяжёлые (~15 GB) — после сдачи можно удалить.

---

## 7. Зоопарк — что делаем

Не склеиваем всё в один DataLoader с равными долями.

1. Общий формат `Sample` (RGB, mask, dataset_id).
2. **Weighted pretrain:** HRC ↑ · SPARCS ↑ · S2 mid · 38-Cloud ↓ · GF1 ↓↓.
3. Или **staged:** HRC+SPARCS → soft 38-Cloud → Photos_40.
4. Eval **по источникам отдельно** + финал на Photos.
5. Код: `src/cloudness_detection/data/mix.py` · текст: [`PIPELINE.md`](PIPELINE.md).

---

## 8. Сводка EDA

| Датасет | N | Disk | Cloud/proxy |
|---------|---|------|-------------|
| Photos | 4272 | 33.3 GB | white_frac ≈ 0.07 |
| 38-Cloud | 8400 GT | 8.6 GB | ≈ 0.50 |
| SPARCS | 80 | 1.45 GB | ≈ 0.25 |
| HRC | 150 | 0.11 GB | ≈ 0.34 |
| GF1 | 10428 | 0.90 GB | ≈ 0.09 |
| S2 | 513 | masks+subscenes | cloud ≈ 0.53 (много 0/1) |

**Главный gap продукта** — разметка Photos, не ещё один Landsat-zip.
