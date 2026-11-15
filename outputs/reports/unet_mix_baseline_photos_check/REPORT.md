# Отчёт: baseline U-Net на миксе + грубые маски Photos

## Модель

- Веса: `E:/Cloudness detection/outputs/train/unet_mix_hrc_sparcs_38_4060/best.pt`
- Архитектура: **unet** (ResNet34, encoder_weights=`imagenet`)
- Датасеты train: `hrc, sparcs, cloud38`
- `image_size`: **384**, batch_size=4, lr=0.0003, seed=42
- Color-match: **нет** (в этом ране `color_match_photos` не включали)
- Best по history (ep **4**): val IoU **0.8809**, Dice **0.9358**, MAE% **2.05**, reject@75 **1.000**

## Важно про разметку Photos

Ручная разметка в CVAT сделана **толстой кистью** — маски приблизительные, не «золотой стандарт». Цифры IoU/Dice на Photos ниже — ориентир, главное смотреть картинки (overlay).

**Разрешение инференса Photos:** XML width/height совпадают с превью (768×577 для всех размеченных). Инференс и overlay на `labeling/photos_40/images/…`, не на полном Photos/ (оригинал ~3280×2464).

**Colornorm:** Пропущен: параллельно ещё идёт обучение colornorm (`train-mix`), чтобы не конкурировать за GPU. В отчёте только baseline без color-match.

## Метрики val (из history.csv обучения)

| epoch | val IoU | val Dice | MAE % | reject@75 |
|------:|--------:|---------:|------:|----------:|
| 4 (best) | 0.8809 | 0.9358 | 2.05 | 1.000 |

## Метрики Photos (грубая GT, только кадры с маской)

Кадров с аннотациями: **8**. Средние: IoU **0.492**, Dice **0.622**, MAE% **20.5**, reject@75 acc **0.875**.

| файл | IoU | Dice | GT % | pred % | MAE % | reject@75 ok |
|------|----:|-----:|-----:|-------:|------:|:------------:|
| `01_2025-10-06_13-11-44_SXC3-2325_1.jpg` | 0.456 | 0.626 | 26.8 | 14.5 | 12.3 | yes |
| `02_2025-10-24_21-36-35_SXC3-2331_1.jpg` | 0.841 | 0.914 | 94.0 | 79.7 | 14.2 | yes |
| `03_2025-11-03_13-31-48_SXC3-2342_1.jpg` | 0.400 | 0.572 | 23.6 | 9.9 | 13.7 | yes |
| `04_2025-11-28_17-30-29_SXC3-2341_1.jpg` | 0.732 | 0.845 | 62.4 | 45.8 | 16.6 | yes |
| `05_2025-12-27_14-10-10_SXC3-2325_1.jpg` | 0.632 | 0.775 | 93.4 | 59.0 | 34.4 | no |
| `06_2025-12-28_08-32-19_SXC3-2360_1.jpg` | 0.282 | 0.440 | 46.4 | 13.1 | 33.3 | yes |
| `08_2026-01-04_13-45-50_SXC3-2313_1.jpg` | 0.536 | 0.698 | 61.6 | 33.1 | 28.6 | yes |
| `11_2026-02-09_21-42-09_SXC3-2342_1.jpg` | 0.055 | 0.103 | 14.2 | 24.9 | 10.7 | yes |

## Галерея train (микс без аугментаций, seed 42)

Каждый кадр: **RGB | GT (красный) | pred (красный)**.

### 01_hrc_triplet.png

![train 01_hrc_triplet.png](train/01_hrc_triplet.png)

### 02_sparcs_triplet.png

![train 02_sparcs_triplet.png](train/02_sparcs_triplet.png)

### 03_hrc_triplet.png

![train 03_hrc_triplet.png](train/03_hrc_triplet.png)

### 04_hrc_triplet.png

![train 04_hrc_triplet.png](train/04_hrc_triplet.png)

### 05_sparcs_triplet.png

![train 05_sparcs_triplet.png](train/05_sparcs_triplet.png)

### 06_hrc_triplet.png

![train 06_hrc_triplet.png](train/06_hrc_triplet.png)

## Галерея val (holdout HRC test, разнообразные по % облаков)

### 01_cloud54_triplet.png

![val 01_cloud54_triplet.png](val/01_cloud54_triplet.png)

### 02_cloud41_triplet.png

![val 02_cloud41_triplet.png](val/02_cloud41_triplet.png)

### 03_cloud09_triplet.png

![val 03_cloud09_triplet.png](val/03_cloud09_triplet.png)

### 04_cloud29_triplet.png

![val 04_cloud29_triplet.png](val/04_cloud29_triplet.png)

### 05_cloud36_triplet.png

![val 05_cloud36_triplet.png](val/05_cloud36_triplet.png)

### 06_cloud62_triplet.png

![val 06_cloud62_triplet.png](val/06_cloud62_triplet.png)

## Галерея Photos test (все кадры с маской в CVAT)

### 01_2025-10-06_13-11-44_SXC3-2325_1.jpg

GT cloud ≈ 26.8%, XML/preview 768×577.

![photos 01_2025-10-06_13-11-44_SXC3-2325_1 triplet](photos_test/01_2025-10-06_13-11-44_SXC3-2325_1_triplet.png)

![rgb](photos_test/01_2025-10-06_13-11-44_SXC3-2325_1_rgb.png) ![gt](photos_test/01_2025-10-06_13-11-44_SXC3-2325_1_gt.png) ![pred](photos_test/01_2025-10-06_13-11-44_SXC3-2325_1_pred.png)

### 02_2025-10-24_21-36-35_SXC3-2331_1.jpg

GT cloud ≈ 94.0%, XML/preview 768×577.

![photos 02_2025-10-24_21-36-35_SXC3-2331_1 triplet](photos_test/02_2025-10-24_21-36-35_SXC3-2331_1_triplet.png)

![rgb](photos_test/02_2025-10-24_21-36-35_SXC3-2331_1_rgb.png) ![gt](photos_test/02_2025-10-24_21-36-35_SXC3-2331_1_gt.png) ![pred](photos_test/02_2025-10-24_21-36-35_SXC3-2331_1_pred.png)

### 03_2025-11-03_13-31-48_SXC3-2342_1.jpg

GT cloud ≈ 23.6%, XML/preview 768×577.

![photos 03_2025-11-03_13-31-48_SXC3-2342_1 triplet](photos_test/03_2025-11-03_13-31-48_SXC3-2342_1_triplet.png)

![rgb](photos_test/03_2025-11-03_13-31-48_SXC3-2342_1_rgb.png) ![gt](photos_test/03_2025-11-03_13-31-48_SXC3-2342_1_gt.png) ![pred](photos_test/03_2025-11-03_13-31-48_SXC3-2342_1_pred.png)

### 04_2025-11-28_17-30-29_SXC3-2341_1.jpg

GT cloud ≈ 62.4%, XML/preview 768×577.

![photos 04_2025-11-28_17-30-29_SXC3-2341_1 triplet](photos_test/04_2025-11-28_17-30-29_SXC3-2341_1_triplet.png)

![rgb](photos_test/04_2025-11-28_17-30-29_SXC3-2341_1_rgb.png) ![gt](photos_test/04_2025-11-28_17-30-29_SXC3-2341_1_gt.png) ![pred](photos_test/04_2025-11-28_17-30-29_SXC3-2341_1_pred.png)

### 05_2025-12-27_14-10-10_SXC3-2325_1.jpg

GT cloud ≈ 93.4%, XML/preview 768×577.

![photos 05_2025-12-27_14-10-10_SXC3-2325_1 triplet](photos_test/05_2025-12-27_14-10-10_SXC3-2325_1_triplet.png)

![rgb](photos_test/05_2025-12-27_14-10-10_SXC3-2325_1_rgb.png) ![gt](photos_test/05_2025-12-27_14-10-10_SXC3-2325_1_gt.png) ![pred](photos_test/05_2025-12-27_14-10-10_SXC3-2325_1_pred.png)

### 06_2025-12-28_08-32-19_SXC3-2360_1.jpg

GT cloud ≈ 46.4%, XML/preview 768×577.

![photos 06_2025-12-28_08-32-19_SXC3-2360_1 triplet](photos_test/06_2025-12-28_08-32-19_SXC3-2360_1_triplet.png)

![rgb](photos_test/06_2025-12-28_08-32-19_SXC3-2360_1_rgb.png) ![gt](photos_test/06_2025-12-28_08-32-19_SXC3-2360_1_gt.png) ![pred](photos_test/06_2025-12-28_08-32-19_SXC3-2360_1_pred.png)

### 08_2026-01-04_13-45-50_SXC3-2313_1.jpg

GT cloud ≈ 61.6%, XML/preview 768×577.

![photos 08_2026-01-04_13-45-50_SXC3-2313_1 triplet](photos_test/08_2026-01-04_13-45-50_SXC3-2313_1_triplet.png)

![rgb](photos_test/08_2026-01-04_13-45-50_SXC3-2313_1_rgb.png) ![gt](photos_test/08_2026-01-04_13-45-50_SXC3-2313_1_gt.png) ![pred](photos_test/08_2026-01-04_13-45-50_SXC3-2313_1_pred.png)

### 11_2026-02-09_21-42-09_SXC3-2342_1.jpg

GT cloud ≈ 14.2%, XML/preview 768×577.

![photos 11_2026-02-09_21-42-09_SXC3-2342_1 triplet](photos_test/11_2026-02-09_21-42-09_SXC3-2342_1_triplet.png)

![rgb](photos_test/11_2026-02-09_21-42-09_SXC3-2342_1_rgb.png) ![gt](photos_test/11_2026-02-09_21-42-09_SXC3-2342_1_gt.png) ![pred](photos_test/11_2026-02-09_21-42-09_SXC3-2342_1_pred.png)

## Как воспроизвести

```text
set PYTHONPATH=src
.venv\Scripts\python.exe scripts\report_model_photos_check.py
```
