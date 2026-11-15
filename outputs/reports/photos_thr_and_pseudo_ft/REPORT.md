# Photos: калибровка порога + pseudo-label finetune

## Анти-leakage

8 кадров с CVAT-масками (`01,02,03,04,05,06,08,11`) **не** участвовали в выборе порога и **не** входили в finetune. Они только в одноразовом финальном сравнении baseline@0.5 / A / B.

- Resolution: Инференс на превью labeling/photos_40/images (типично 768×577), не на полном Photos/. Модель ресайзит к image_size=384, probs возвращаем bilinear на размер превью.
- Baseline ckpt: `E:\Cloudness detection\outputs\train\unet_mix_hrc_sparcs_38_4060\best.pt`

## Part A — калибровка порога (без train)

**Правило выбора thr:** На unlabeled-32: min MAE(pred%, white_frac_proxy*100); среди MAE ≤ best+0.5 — max Pearson r; при равном r — меньший thr. Photos-8 IoU в выборе НЕ участвует.

**Выбранный thr:** **0.50**

| thr | mean pred % | MAE vs proxy | Pearson r |
|----:|------------:|-------------:|----------:|
| 0.20 | 58.7 | 44.07 | 0.248 |
| 0.25 | 56.9 | 42.46 | 0.268 |
| 0.30 | 55.2 | 40.82 | 0.287 |
| 0.35 | 53.5 | 39.31 | 0.307 |
| 0.40 | 52.0 | 37.91 | 0.325 |
| 0.45 | 50.6 | 36.56 | 0.341 |
| 0.50 | 49.2 | 35.26 | 0.356 ← |

## Part B — soft pseudo-label finetune

- Pseudo: soft probs baseline на 32 unlabeled (не binary@0.35; thr=0.35 только для справки binary-сравнения).
- FT: lr=0.0001, epochs=4, bs=2, seed=42, mix=['hrc', 'sparcs', 'cloud38', 'photos_pseudo'], weights={'hrc': 3.0, 'sparcs': 2.5, 'cloud38': 0.5, 'photos_pseudo': 2.5}
- Best: `E:\Cloudness detection\outputs\train\unet_photos_pseudo_ft\best.pt` (val IoU public=0.8569)
- Thr после FT: **0.50** (После FT: proxy-sweep на 32; оставили 0.5, т.к. MAE_proxy@0.5 =38.68 близко к best 0.50/38.68 (Δ≤0.25).)
- Colornorm/GPU: ожидание VRAM 0s; notes=['Свободно 5393 MiB после 0s']

## Метрики Photos-8 (одно сравнение)

| вариант | thr | IoU | Dice | MAE % | reject@75 |
|--------|----:|----:|-----:|------:|----------:|
| baseline | 0.50 | 0.492 | 0.621 | 20.5 | 0.875 |
| A calibrated | 0.50 | 0.492 | 0.621 | 20.5 | 0.875 |
| B pseudo-FT | 0.50 | 0.461 | 0.595 | 23.9 | 0.875 |

**Вывод:** A совпал с baseline@0.5 — на unlabeled-32 `white_frac_proxy` (~15%) сильно ниже pred cloud%, и среди thr∈[0.20…0.50] лучший MAE/r даёт **0.50**. Короткий soft-pseudo FT (B) на Photos-8 **не улучшил** IoU/Dice/MAE относительно baseline (чуть хуже); reject@75 тот же 0.875. Смотреть overlays важнее цифр (маски CVAT грубые).

### Per-image (B)

| файл | IoU | Dice | GT % | pred % | MAE % |
|------|----:|-----:|-----:|-------:|------:|
| `01_2025-10-06_13-11-44_SXC3-2325_1.jpg` | 0.378 | 0.549 | 26.8 | 12.9 | 13.9 |
| `02_2025-10-24_21-36-35_SXC3-2331_1.jpg` | 0.836 | 0.911 | 94.0 | 79.9 | 14.1 |
| `03_2025-11-03_13-31-48_SXC3-2342_1.jpg` | 0.422 | 0.593 | 23.6 | 10.2 | 13.4 |
| `04_2025-11-28_17-30-29_SXC3-2341_1.jpg` | 0.594 | 0.745 | 62.4 | 37.1 | 25.3 |
| `05_2025-12-27_14-10-10_SXC3-2325_1.jpg` | 0.649 | 0.787 | 93.4 | 60.6 | 32.8 |
| `06_2025-12-28_08-32-19_SXC3-2360_1.jpg` | 0.304 | 0.466 | 46.4 | 14.1 | 32.3 |
| `08_2026-01-04_13-45-50_SXC3-2313_1.jpg` | 0.470 | 0.639 | 61.6 | 29.0 | 32.7 |
| `11_2026-02-09_21-42-09_SXC3-2342_1.jpg` | 0.034 | 0.065 | 14.2 | 41.3 | 27.1 |

## Overlays (8 held-out)

Каждый кадр: **RGB | GT | pred**. Папки:

- `baseline_0.5/`
- `A_calibrated/`
- `B_pseudo_ft/`

### 01_2025-10-06_13-11-44_SXC3-2325_1.jpg

![B 01_2025-10-06_13-11-44_SXC3-2325_1](B_pseudo_ft/01_2025-10-06_13-11-44_SXC3-2325_1_triplet.png)

[baseline](baseline_0.5/01_2025-10-06_13-11-44_SXC3-2325_1_triplet.png) · [A](A_calibrated/01_2025-10-06_13-11-44_SXC3-2325_1_triplet.png) · [B](B_pseudo_ft/01_2025-10-06_13-11-44_SXC3-2325_1_triplet.png)

### 02_2025-10-24_21-36-35_SXC3-2331_1.jpg

![B 02_2025-10-24_21-36-35_SXC3-2331_1](B_pseudo_ft/02_2025-10-24_21-36-35_SXC3-2331_1_triplet.png)

[baseline](baseline_0.5/02_2025-10-24_21-36-35_SXC3-2331_1_triplet.png) · [A](A_calibrated/02_2025-10-24_21-36-35_SXC3-2331_1_triplet.png) · [B](B_pseudo_ft/02_2025-10-24_21-36-35_SXC3-2331_1_triplet.png)

### 03_2025-11-03_13-31-48_SXC3-2342_1.jpg

![B 03_2025-11-03_13-31-48_SXC3-2342_1](B_pseudo_ft/03_2025-11-03_13-31-48_SXC3-2342_1_triplet.png)

[baseline](baseline_0.5/03_2025-11-03_13-31-48_SXC3-2342_1_triplet.png) · [A](A_calibrated/03_2025-11-03_13-31-48_SXC3-2342_1_triplet.png) · [B](B_pseudo_ft/03_2025-11-03_13-31-48_SXC3-2342_1_triplet.png)

### 04_2025-11-28_17-30-29_SXC3-2341_1.jpg

![B 04_2025-11-28_17-30-29_SXC3-2341_1](B_pseudo_ft/04_2025-11-28_17-30-29_SXC3-2341_1_triplet.png)

[baseline](baseline_0.5/04_2025-11-28_17-30-29_SXC3-2341_1_triplet.png) · [A](A_calibrated/04_2025-11-28_17-30-29_SXC3-2341_1_triplet.png) · [B](B_pseudo_ft/04_2025-11-28_17-30-29_SXC3-2341_1_triplet.png)

### 05_2025-12-27_14-10-10_SXC3-2325_1.jpg

![B 05_2025-12-27_14-10-10_SXC3-2325_1](B_pseudo_ft/05_2025-12-27_14-10-10_SXC3-2325_1_triplet.png)

[baseline](baseline_0.5/05_2025-12-27_14-10-10_SXC3-2325_1_triplet.png) · [A](A_calibrated/05_2025-12-27_14-10-10_SXC3-2325_1_triplet.png) · [B](B_pseudo_ft/05_2025-12-27_14-10-10_SXC3-2325_1_triplet.png)

### 06_2025-12-28_08-32-19_SXC3-2360_1.jpg

![B 06_2025-12-28_08-32-19_SXC3-2360_1](B_pseudo_ft/06_2025-12-28_08-32-19_SXC3-2360_1_triplet.png)

[baseline](baseline_0.5/06_2025-12-28_08-32-19_SXC3-2360_1_triplet.png) · [A](A_calibrated/06_2025-12-28_08-32-19_SXC3-2360_1_triplet.png) · [B](B_pseudo_ft/06_2025-12-28_08-32-19_SXC3-2360_1_triplet.png)

### 08_2026-01-04_13-45-50_SXC3-2313_1.jpg

![B 08_2026-01-04_13-45-50_SXC3-2313_1](B_pseudo_ft/08_2026-01-04_13-45-50_SXC3-2313_1_triplet.png)

[baseline](baseline_0.5/08_2026-01-04_13-45-50_SXC3-2313_1_triplet.png) · [A](A_calibrated/08_2026-01-04_13-45-50_SXC3-2313_1_triplet.png) · [B](B_pseudo_ft/08_2026-01-04_13-45-50_SXC3-2313_1_triplet.png)

### 11_2026-02-09_21-42-09_SXC3-2342_1.jpg

![B 11_2026-02-09_21-42-09_SXC3-2342_1](B_pseudo_ft/11_2026-02-09_21-42-09_SXC3-2342_1_triplet.png)

[baseline](baseline_0.5/11_2026-02-09_21-42-09_SXC3-2342_1_triplet.png) · [A](A_calibrated/11_2026-02-09_21-42-09_SXC3-2342_1_triplet.png) · [B](B_pseudo_ft/11_2026-02-09_21-42-09_SXC3-2342_1_triplet.png)

## Как воспроизвести

```text
set PYTHONPATH=src
.venv\Scripts\python.exe scripts\calibrate_photos_threshold.py
.venv\Scripts\python.exe scripts\finetune_photos_pseudo.py
```
