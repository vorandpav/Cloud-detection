# Исследование: модели детекции облаков (пруфы и ссылки)

Дата обзора: **2026-11-15**.  
Цель: не цепляться за Cloud-Net / 38-Cloud, а понять, что реально свежее и что переносимо на **RGB Photos** (кубсаты, ~200 м/пкс).

**Важно про 2026:** да, новые модели уже есть. Тренд — foundation models + adapters, Mamba, semi-supervised, кросс-сенсор (Swin+UPerNet). Cloud-Net (2019) / Cloud-Net+ (2020–21) — исторический ориентир на узком бенчмарке, не актуальный «единственный SOTA».

---

## 1. Лидерборды и сводные таблицы (пруфы цифр)

| Источник | Что показывает | Ссылка |
|----------|----------------|--------|
| Papers With Code — task Cloud Detection | список papers with code; лидерборды по датасетам часто пустые/устаревшие | https://paperswithcode.com/task/cloud-detection |
| 38-Cloud semantic segmentation archive | Cloud-Net+ **88.90** Jaccard, Cloud-Net **87.32** (снимок ~2025-07; новых строк почти нет) | https://codewithpapers.app/sota/semantic-segmentation-on-38-cloud |
| Wizwand — 95-Cloud | SOTA Jaccard **91.57** (часто Cloud-Net+ линия) | https://www.wizwand.com/dataset/95-cloud |
| Wizwand — SPARCS 5CV | Cloud-Net+ FJL2 Jaccard **83.14** | https://www.wizwand.com/sota/cloud-detection-on-sparcs-5cv |
| **cloudseg / Cloud-Adapter leaderboard** | единый mIoU на HRC / GF1 / GF2 / Sentinel L1C / L2A / L8 Biome — **лучший живой кросс-сет** | https://github.com/XavierJiezou/cloudseg · https://xavierjiezou.github.io/Cloud-Adapter/ |
| Awesome Cloud Detection and Removal | каталог датасетов и papers | https://github.com/PeeBar/Awesome-Cloud-Detection-and-Removal |

### Таблица cloudseg (mIoU %, ↑) — снимок с GitHub Cloud-Adapter

| Method | HRC | GF1 | GF2 | L1C | L2A | L8B |
|--------|-----|-----|-----|-----|-----|-----|
| CDNetv2 | 76.75 | 84.93 | 78.84 | 65.60 | 66.05 | 43.63 |
| UNetMobv2 | 79.91 | 91.71 | **80.44** | 71.65 | 70.36 | 47.76 |
| DBNet | 77.78 | 91.36 | 78.68 | 65.52 | 65.65 | **51.41** |
| HRCloudNet | **83.44*** | 91.86 | 75.57 | 68.26 | 68.35 | 43.51 |
| KappaMask | 67.48 | **92.42*** | 72.00 | 41.27 | 45.28 | 42.12 |
| **Cloud-Adapter** | **89.05** | **92.55** | **83.02** | **74.18** | **73.38** | **57.53** |

\* без Cloud-Adapter; с ним Cloud-Adapter лучший почти везде.  
Источник: https://xavierjiezou.github.io/Cloud-Adapter/

**Вывод по лидербордам:** на 38-Cloud «висит» Cloud-Net+, потому что новые работы редко пересдают этот протокол. На современных мультисенсорных сетах лидируют **Cloud-Adapter** и соседние VFM/adapter / high-res сети, не Cloud-Net.

---

## 2. Обзоры (каркас поля)

| Работа | Год | Ссылка | Заметка |
|--------|-----|--------|---------|
| Deep Learning-Based Cloud Detection for Optical Remote Sensing Images: A Survey | 2024 | https://doi.org/10.3390/rs16234583 | ~169 journal papers; после 2022 — рост attention/Transformer |
| Cloud Detection Methods for Optical Satellite Imagery: A Comprehensive Review | 2025 | https://www.mdpi.com/2673-7418/5/3/27 | датасеты + методы |
| An Introduction to Machine and Deep Learning Methods for Cloud Masking | 2024 | https://www.mdpi.com/2076-3417/14/7/2887 | masking overview |

---

## 3. Хронология моделей (с пруфами)

### 2019–2021 — «классика Landsat / 38-Cloud»

| Модель | Год | Ссылка | Метрики / идея | Нам полезно? |
|--------|-----|--------|----------------|--------------|
| **Cloud-Net** | 2019 | https://arxiv.org/abs/1901.10077 | FCN на RGBNir, 38-Cloud | Исторический baseline; 4 канала ≠ Photos |
| **Cloud-Net+** | 2020/21 | https://arxiv.org/abs/2001.08768 | FJL loss, чуть легче Cloud-Net; Jaccard ~**88.9** на 38-Cloud | **Loss (FJL)** — да; архитектура — не цель |
| **Cloud-AttU** | 2020 | https://doi.org/10.3390/sym12061056 | U-Net + attention | Идея attention gates |
| **CDNet / CDNetV2** | 2019/21 | https://doi.org/10.1109/tgrs.2020.2991398 | thumbnail RGB, cloud–snow, AFFM | Близко к «грубым» кадрам кубсатов |
| **nnU-Net** (On Cloud N) | 2022 | https://arxiv.org/abs/2210.13659 | Jaccard **0.882** на Sentinel challenge (top ~7%) | «Сильный U-Net без ручного дизайна» |

### 2022–2024 — attention, dual-branch, high-res

| Модель | Год | Ссылка | Метрики / идея | Нам полезно? |
|--------|-----|--------|----------------|--------------|
| **SEUNet++** | 2022 | https://doi.org/10.1016/j.eswa.2022.118493 | U-Net++ + SE; IoU **91.8%** на 95-Cloud | Простой сильный RGB/MSI путь + ImageNet |
| **CLiSA / C-LiSA** | 2023–24 | https://arxiv.org/abs/2311.17475 · https://doi.org/10.1109/tgrs.2024.3394929 | hybrid transformer; Biome/SPARCS/Sentinel/Cartosat; абляции RGB | **Кросс-сенсор** — главный narrative кейса |
| **DB-Net** | 2024 | https://doi.org/10.1117/1.JRS.18.034524 | CNN + Swin dual-branch | Thin/small clouds |
| **HR-cloud-Net** | 2024 | https://arxiv.org/html/2407.07365 · code https://github.com/kunzhan/hr-cloud-net | high-res branches; HRC mIoU **83.44** | Границы/текстура |
| **SegFormer** vs U-Net (Sentinel) | 2025 | https://doi.org/10.5753/wcama.2025.7754 | SegFormer IoU **0.85** > U-Net / DeepLab | Практичный SMP-кандидат |

### 2025 — foundation adapters и сильные CNN

| Модель | Год | Ссылка | Метрики / идея | Нам полезно? |
|--------|-----|--------|----------------|--------------|
| **Cloud-Adapter** (TGRS) | 2025 | paper https://arxiv.org/abs/2411.13127 · site https://xavierjiezou.github.io/Cloud-Adapter/ · code https://github.com/XavierJiezou/cloudseg | frozen VFM + adapter (~0.6% params); **SOTA** на HRC/GF/Sentinel/L8B | Свежий SOTA-слайд; внедрение тяжелее |
| **NFCNet** | 2025 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11861746/ · https://doi.org/10.3390/s25041245 | HCAM/SPFA/DCA; бьёт UNet/DeepLab/Cloud-AttU на HRC, CHLandsat8, **95-Cloud (RGB)** | Хорошие идеи под RGB |
| **TransCNet** | 2025 | https://doi.org/10.3390/app15189870 | ResNet-34 + PVT, RGB 352²; vs Cloud-Net, CDNetV2, SegFormer | Dual-branch RGB story |
| **MSCloudCAM** | 2025–26 | https://arxiv.org/abs/2510.10802 | multi-scale CAM; CloudSEN12 + L8Biome | MSI; для Photos вторично |

### 2026 — да, уже есть новое (на момент 2026-11)

| Модель / работа | Когда | Ссылка | Суть | Нам полезно? |
|-----------------|-------|--------|------|--------------|
| **STUPmask** (Swin + UPerNet) | RSE, online ~2026-01 (DOI 2025.115206) | https://doi.org/10.1016/j.rse.2025.115206 · PDF https://weijing-rs.github.io/publications/Pang_et_al-RSE-2026.pdf | OA **97.51%** L8, **96.27%** S2; IoU ~**92–93%**; transfer GF-2 / MODIS / Himawari | **Сильный кросс-сенсор** — близко к задаче Photos |
| **CloudMamba** | TGRS 2026 | https://arxiv.org/html/2604.06844 · code https://github.com/jayoungo/CloudMamba | CNN–Mamba + uncertainty 2-stage; GF1 mIoU **89.27**, Levir_CS **91.24** | Thin clouds / boundaries; свежий SOTA на GF1/Levir |
| **DinoCloudNet** | JSTARS 2026 | https://doi.org/10.1109/jstars.2026.3690021 | frozen DINOv3 + ECA-SPM + MGFF adapter | Эволюция Cloud-Adapter линии |
| **Lightweight dual-path + gated fusion** | Sensors 2026 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12987299/ | HRC OA **96.31%**, mIoU **92.82%**, **12 GFLOPs** | Лёгкая модель — дружит с 4060 / onboard story |
| **F1-weighted ensemble** | Sci Rep 2026 | https://www.nature.com/articles/s41598-026-58959-0 | ансамбль DL; F1 ~**90.25%**, OA ~**95.47%** | Скорее идея fusion, не одна сеть |
| **CloudMatch** | 2026 | https://arxiv.org/html/2601.03528 | semi-supervised weak→strong | Мало разметки Photos — релевантно |
| **Low-data supervised adaptation > prompting** | 2026 | https://arxiv.org/pdf/2604.08956 | CLIPSeg prompts проигрывают; **0.1–10% labels** уже сильно лучше | Пруф: на Photos нужны метки / finetune, не «промпт» |
| **TIRAuxCloud** | 2026 | https://arxiv.org/html/2602.21905v2 | day/night TIR датасет | Нам не целевой (у Photos RGB day) |

**CloudMamba vs baselines (из статьи, Table I):**

| Method | GF1_WHU mIoU | Levir_CS mIoU |
|--------|--------------|---------------|
| U-Net | 84.72 | 89.43 |
| DeepLabV3+ | 86.66 | 81.10 |
| U-Mamba | 86.94 | 90.78 |
| **CloudMamba** | **89.27** | **91.24** |

Источник: https://arxiv.org/html/2604.06844

---

## 4. Датасеты (почему 38-Cloud ≠ «единственный мир»)

| Датасет | Ссылка / заметка | Зачем в обзоре |
|---------|------------------|----------------|
| 38-Cloud | Mohajerani; RGBNir Landsat-8 | У нас есть; удобный train; **не** похож 1:1 на Photos |
| 95-Cloud | расширение 38-Cloud | Больше данных того же семейства |
| SPARCS | Landsat patches | Holdout / тени |
| HRC_WHU | high-res 0.5–15 m | Границы, разнообразие фона |
| GF1-WHU / Levir_CS | китайские HR | CloudMamba / CDNet бенчмарки |
| CloudSEN12 / L8 Biome | Sentinel multi-class | Cloud-Adapter / MSCloudCAM |
| Sentinel-2 Cloud Mask Catalogue | Francis et al. | RGB(+bands) сравнения SegFormer |

Мы используем 38-Cloud, потому что **скачали**, а не потому что он лучший прокси Photos.

---

## 5. Что это значит для кейса СПУТНИКС

Критерий успеха кейса: **% облачности + reject@75 на Photos**, не Jaccard на 38-Cloud.

**Железо:** локально **4060 8 GB** (baseline) + **Kaggle T4 16 GB** (современный прогон). Подробный выбор: [`MODEL_SELECTION_REPORT.md`](MODEL_SELECTION_REPORT.md) · скачать данные: [`DOWNLOAD_NOW.md`](DOWNLOAD_NOW.md).

| Приоритет | Выбор | Почему |
|-----------|-------|--------|
| 1. Тренд 2025 на T4 | **Cloud-Adapter** (DINOv2-L + adapter, ≥16 GB) | SOTA-слайд + реальный train; код https://github.com/XavierJiezou/cloud-adapter |
| 2. Baseline на 4060 | U-Net ResNet34 + Photos калибровка | Уже обучен; быстрые итерации |
| 3. Mid-tier сравнение | SegFormer-B2 + soft Jaccard/FJL | Efficient transformer; не «вместо» Cloud-Adapter |
| 4. Опц. 2026 | CloudMamba | Если успеют; риск 4ch→RGB для Photos |
| 5. Домен | finetune / калибровка на `photos_40` | Готовых весов под кубсаты нет; arXiv:2604.08956 |
| 6. Не цель | гнаться за 0.89 на official 38-Cloud с RGBNir Cloud-Net+ | Не наш сенсор |

---

## 6. Код / репозитории (для воспроизведения)

| Репо | Ссылка |
|------|--------|
| Cloud-Adapter (код) | https://github.com/XavierJiezou/cloud-adapter |
| cloudseg + weights / leaderboard | https://github.com/XavierJiezou/cloudseg |
| Cloud-Adapter project page | https://xavierjiezou.github.io/Cloud-Adapter/ |
| Cloud-Adapter datasets (HF) | https://huggingface.co/datasets/XavierJiezou/cloud-adapter-datasets |
| CloudMamba | https://github.com/jayoungo/CloudMamba |
| HR-cloud-Net | https://github.com/kunzhan/hr-cloud-net |
| Awesome list | https://github.com/PeeBar/Awesome-Cloud-Detection-and-Removal |

---

## 7. Как обновлять этот файл

1. Раз в этап кейса — прогнать поиск: `cloud detection remote sensing 2026`, `cloud segmentation foundation model`, CloudSEN12 leaderboard.  
2. Новые цифры класть **только со ссылкой** (DOI / arXiv / GitHub table).  
3. Отдельно помечаить: *метрика на каком датасете* — нельзя смешивать 38-Cloud Jaccard и HRC mIoU.

Связанный обзор датасетов (что качать, размеры, на чём учили SOTA): [`CLOUD_DATASETS.md`](CLOUD_DATASETS.md).  
Канонический отчёт по данным + EDA: [`DATASETS_REPORT.md`](DATASETS_REPORT.md).  
Выбор моделей для кейса: [`MODEL_SELECTION_REPORT.md`](MODEL_SELECTION_REPORT.md).  
Старый черновик про железо/протокол: [`SOTA_GAP.md`](SOTA_GAP.md).
