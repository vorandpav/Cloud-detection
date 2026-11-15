# Photos-8: baseline vs color-match vs SegFormer

Одинаковые 8 CVAT-масок, thr=0.5, превью 768×577. Color-match на инференсе Photos **не** включали (в train он тянул публичные к домену Photos).

| модель | val IoU (из train) | Photos IoU | Dice | MAE % | rej@75 |
|--------|-------------------:|-----------:|-----:|------:|-------:|
| `baseline_unet` | 0.881 | **0.492** | 0.621 | 20.5 | 0.875 |
| `colornorm_unet` | 0.874 | **0.296** | 0.401 | 34.7 | 0.750 |
| `segformer_kaggle` | 0.885 | **0.205** | 0.296 | 39.0 | 0.750 |

## Per-image IoU

| файл | baseline | colornorm | segformer |
|------|---------:|----------:|----------:|
| `01_2025-10-06_13-11-44_SXC3-2325_1.jpg` | 0.456 | 0.428 | 0.000 |
| `02_2025-10-24_21-36-35_SXC3-2331_1.jpg` | 0.841 | 0.306 | 0.464 |
| `03_2025-11-03_13-31-48_SXC3-2342_1.jpg` | 0.400 | 0.560 | 0.460 |
| `04_2025-11-28_17-30-29_SXC3-2341_1.jpg` | 0.732 | 0.014 | 0.051 |
| `05_2025-12-27_14-10-10_SXC3-2325_1.jpg` | 0.632 | 0.695 | 0.466 |
| `06_2025-12-28_08-32-19_SXC3-2360_1.jpg` | 0.281 | 0.050 | 0.013 |
| `08_2026-01-04_13-45-50_SXC3-2313_1.jpg` | 0.536 | 0.313 | 0.099 |
| `11_2026-02-09_21-42-09_SXC3-2342_1.jpg` | 0.054 | 0.000 | 0.089 |

## Overlays (RGB | GT | pred)

- `baseline_unet/`
- `colornorm_unet/`
- `segformer_kaggle/`

### 01_2025-10-06_13-11-44_SXC3-2325_1.jpg

![baseline](baseline_unet/01_2025-10-06_13-11-44_SXC3-2325_1_triplet.png)

![colornorm](colornorm_unet/01_2025-10-06_13-11-44_SXC3-2325_1_triplet.png)

![segformer](segformer_kaggle/01_2025-10-06_13-11-44_SXC3-2325_1_triplet.png)

### 02_2025-10-24_21-36-35_SXC3-2331_1.jpg

![baseline](baseline_unet/02_2025-10-24_21-36-35_SXC3-2331_1_triplet.png)

![colornorm](colornorm_unet/02_2025-10-24_21-36-35_SXC3-2331_1_triplet.png)

![segformer](segformer_kaggle/02_2025-10-24_21-36-35_SXC3-2331_1_triplet.png)

### 03_2025-11-03_13-31-48_SXC3-2342_1.jpg

![baseline](baseline_unet/03_2025-11-03_13-31-48_SXC3-2342_1_triplet.png)

![colornorm](colornorm_unet/03_2025-11-03_13-31-48_SXC3-2342_1_triplet.png)

![segformer](segformer_kaggle/03_2025-11-03_13-31-48_SXC3-2342_1_triplet.png)

### 04_2025-11-28_17-30-29_SXC3-2341_1.jpg

![baseline](baseline_unet/04_2025-11-28_17-30-29_SXC3-2341_1_triplet.png)

![colornorm](colornorm_unet/04_2025-11-28_17-30-29_SXC3-2341_1_triplet.png)

![segformer](segformer_kaggle/04_2025-11-28_17-30-29_SXC3-2341_1_triplet.png)

### 05_2025-12-27_14-10-10_SXC3-2325_1.jpg

![baseline](baseline_unet/05_2025-12-27_14-10-10_SXC3-2325_1_triplet.png)

![colornorm](colornorm_unet/05_2025-12-27_14-10-10_SXC3-2325_1_triplet.png)

![segformer](segformer_kaggle/05_2025-12-27_14-10-10_SXC3-2325_1_triplet.png)

### 06_2025-12-28_08-32-19_SXC3-2360_1.jpg

![baseline](baseline_unet/06_2025-12-28_08-32-19_SXC3-2360_1_triplet.png)

![colornorm](colornorm_unet/06_2025-12-28_08-32-19_SXC3-2360_1_triplet.png)

![segformer](segformer_kaggle/06_2025-12-28_08-32-19_SXC3-2360_1_triplet.png)

### 08_2026-01-04_13-45-50_SXC3-2313_1.jpg

![baseline](baseline_unet/08_2026-01-04_13-45-50_SXC3-2313_1_triplet.png)

![colornorm](colornorm_unet/08_2026-01-04_13-45-50_SXC3-2313_1_triplet.png)

![segformer](segformer_kaggle/08_2026-01-04_13-45-50_SXC3-2313_1_triplet.png)

### 11_2026-02-09_21-42-09_SXC3-2342_1.jpg

![baseline](baseline_unet/11_2026-02-09_21-42-09_SXC3-2342_1_triplet.png)

![colornorm](colornorm_unet/11_2026-02-09_21-42-09_SXC3-2342_1_triplet.png)

![segformer](segformer_kaggle/11_2026-02-09_21-42-09_SXC3-2342_1_triplet.png)

