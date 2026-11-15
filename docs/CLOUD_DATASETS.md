# Исследование: датасеты облаков (что качать и на чём учили SOTA)

Дата обзора: **2026-11-15**.  
**Канонический итоговый разбор:** [`DATASETS_FINAL.md`](DATASETS_FINAL.md) · EDA-детали: [`DATASETS_REPORT.md`](DATASETS_REPORT.md).  
Связано с моделями: [`CLOUD_DETECTION_RESEARCH.md`](CLOUD_DETECTION_RESEARCH.md) · выбор моделей: [`MODEL_SELECTION_REPORT.md`](MODEL_SELECTION_REPORT.md).

Локально сейчас (обновлено **2026-11-15**, ночной прогон — **на диске**):
- `data/38-cloud/` ≈ **8.6 GB** (есть; NIR удалён ранее — для Photos ок RGB);
- `data/sending/` ≈ **1.45 GB** — **SPARCS**; `data/sparcs/` → junction на `sending/`;
- `data/hrc_whu/` — **распакован** (~110 MB, `img_dir`/`ann_dir`); zip рядом;
- `data/gf12ms_whu_gf1/` — **распакован** (~0.9 GB); zip рядом; **разметка mediocre**;
- `data/s2_catalogue_thumbs/` — thumbs; `data/s2_catalogue/` — **masks + subscenes** (из загруженных zip);
- Сырые zip S2/HRC/GF1 лежат в `data/` — после проверки можно удалить.

---

## 1. Короткий вердикт: что ещё качать

| Приоритет | Датасет | Вес (архив) | Зачем | Решение |
|-----------|---------|-------------|-------|---------|
| **A** | **HRC_WHU** | **~110 MB** | RGB high-res; NFCNet / Cloud-Adapter / HRCloudNet | **Скачать обязательно** |
| **A** | **SPARCS** | ~1.4 GB (у тебя уже в `sending`) | Holdout Landsat; Cloud-Net+, TransCNet, CLiSA | **Дособрать/переложить**, не качать заново |
| **B** | **GF1MS-WHU (GF1)** из Cloud-Adapter HF | **~0.9 GB** | Тот же бенчмарк, что у Cloud-Adapter / близко к CloudMamba GF1 | Полезно, если место есть |
| **A (сейчас)** | S2 Cloud Mask Catalogue — **`masks.zip` + `subscenes.zip`** | **~15.2 GB** | Разметка + сцены для train (не только thumbs) | См. [`DOWNLOAD_NOW.md`](DOWNLOAD_NOW.md) |
| **B** | S2 thumbs (уже есть) | **~146 MB** | Быстрый RGB look / слайды | Не качать заново |
| **C** | GF2MS-WHU | **~1.7 GB** | Второй сет Cloud-Adapter | Опционально |
| **D / не сейчас** | CloudSEN12-high L1C/L2A (HF адаптер) | **~3.1 + 3.4 GB** каждый | SOTA multi-class Sentinel | Тяжеловато под 57 GB + Photos |
| **D / не сейчас** | L8 Biome (полный) | **~2.6 GB** zip в HF | Cloud-Adapter L8B | Можно позже |
| **Не надо** | полный CloudSEN12+ | **~100–250 GB** | Гигант | Нет |
| **Не надо** | Levir_CS full | **34 GB+** / full-res терабайты | CloudMamba второй сет | Нет |
| **Низкий ROI** | 95-Cloud | ещё ~столько же семейства 38-Cloud | Тот же Landsat RGBNir | Сначала Photos + HRC |

**Минимальный пакет «как у свежих статей, но лёгкий»:**  
уже есть 38-Cloud + SPARCS → добрать **HRC_WHU (~110 MB)**. Итого +0.1 GB.

**Если хочешь один «SOTA-совместимый» доп. сет среднего веса:**  
HRC + **gf12ms_whu_gf1.zip (~0.9 GB)** с Hugging Face Cloud-Adapter.

---

## 2. На чём учили / валидировали свежие SOTA

| Модель (год) | Train / eval датасеты | Ссылка на пруф |
|--------------|----------------------|----------------|
| **Cloud-Adapter** (2025) | HRC, GF1, GF2, CloudSEN12-high L1C/L2A, L8 Biome | https://xavierjiezou.github.io/Cloud-Adapter/ · HF https://huggingface.co/datasets/XavierJiezou/cloud-adapter-datasets |
| **CloudMamba** (2026) | **GF1_WHU**, **Levir_CS** | https://arxiv.org/html/2604.06844 |
| **STUPmask** (RSE 2026) | Landsat-8 + Sentinel-2 manual masks; transfer GF-2 / MODIS / Himawari | https://doi.org/10.1016/j.rse.2025.115206 |
| **NFCNet** (2025) | **HRC_WHU**, CHLandsat8, **95-Cloud (RGB)** | https://pmc.ncbi.nlm.nih.gov/articles/PMC11861746/ |
| **TransCNet** (2025) | CHLandsat-8, **38-Cloud**, **SPARCS** (вход RGB) | https://doi.org/10.3390/app15189870 |
| **CLiSA** (2024) | L8 Biome train; transfer **SPARCS**, Sentinel-2, Cartosat | https://arxiv.org/abs/2311.17475 |
| **HR-cloud-Net** (2024) | HRC / GF семейство (см. cloudseg) | https://arxiv.org/html/2407.07365 |
| **SEUNet++** (2022) | **95-Cloud** | https://doi.org/10.1016/j.eswa.2022.118493 |
| **Cloud-Net / +** (2019–21) | **38-Cloud / 95-Cloud**, SPARCS, Biome | https://arxiv.org/abs/2001.08768 |
| **SegFormer сравнение** (2025) | Sentinel-2 Cloud Mask Catalogue | https://doi.org/10.5753/wcama.2025.7754 |

Паттерн: свежие лидерборды крутятся вокруг **HRC + GF + CloudSEN12 + L8 Biome**, а не вокруг 38-Cloud.

---

## 3. Карточки датасетов (пруфы размеров и ссылок)

### 3.1 Уже у нас

#### 38-Cloud
- Сенсор: Landsat-8, патчи 384², каналы R/G/B/NIR (+ GT).
- Объём: у нас ≈ **8.6 GB** (без части NIR).
- Скачать: https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images  
- README: https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset  
- Роль: основной train; **не** лучший прокси Photos.

#### SPARCS (L8)
- 80 патчей 1000×1000, классы cloud / shadow / snow / water / land.
- Архив ~**1.4 GB**: https://www.usgs.gov/landsat-missions/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation-data  
- Зеркало OSU: https://emapr.ceoas.oregonstate.edu/sparcs/ (`sparcs_data_L8.zip` 1.43 GB)  
- У нас: файлы уже лежат в `data/sending/` — имеет смысл переименовать/переложить в `data/sparcs/` и добить, если чего-то не хватает (полный zip = 80 сцен).

### 3.2 Лёгкие / приоритетные

#### HRC_WHU ⭐
- **RGB**, 150 сцен 1280×720, GSD 0.5–15 м, 5 типов поверхности (water / veg / urban / snow / barren).
- Оригинал: https://github.com/dr-lizhiwei/HRC_WHU (Baidu / Google Drive).
- Удобный zip для Cloud-Adapter: **`hrc_whu.zip` ≈ 110 MB**  
  https://huggingface.co/datasets/XavierJiezou/cloud-adapter-datasets  
  Команда:
  ```bash
  huggingface-cli download --repo-type dataset XavierJiezou/cloud-adapter-datasets --local-dir data --include hrc_whu.zip
  ```
- Почему нам: тот же RGB-мир, что Photos; на нём меряют NFCNet / Cloud-Adapter / lightweight Sensors 2026.

#### Sentinel-2 Cloud Mask Catalogue (Francis)
- 513 subscenes 1022² @ 20 m; CLEAR / CLOUD / SHADOW + теги поверхности.
- Zenodo: https://zenodo.org/records/4172871  
  - `masks.zip` ~7 MB  
  - `thumbnails.zip` ~**146 MB** ← лёгкий вариант  
  - `subscenes.zip` ~**15.2 GB** ← полный  
- Роль: другой сенсор (хорошо для «не только Landsat» на слайде).

### 3.3 Средний вес (Cloud-Adapter pack)

Источник всех zip: https://huggingface.co/datasets/XavierJiezou/cloud-adapter-datasets  
(суммарно репо ~12.5 GB)

| Файл | Размер (байт → ≈) | Источник оригинал |
|------|-------------------|-------------------|
| `hrc_whu.zip` | 115 524 658 ≈ **0.11 GB** | https://github.com/dr-lizhiwei/HRC_WHU |
| `gf12ms_whu_gf1.zip` | 942 211 975 ≈ **0.88 GB** | https://github.com/whu-ZSC/GF1-GF2MS-WHU |
| `gf12ms_whu_gf2.zip` | 1 744 108 191 ≈ **1.6 GB** | то же |
| `cloudsen12_high_l1c.zip` | 3 360 887 512 ≈ **3.1 GB** | https://huggingface.co/datasets/csaybar/CloudSEN12-high |
| `cloudsen12_high_l2a.zip` | 3 571 709 018 ≈ **3.3 GB** | то же |
| `l8_biome.zip` | 2 771 988 653 ≈ **2.6 GB** | USGS L8 CCA / Biome |

Оригиналы GF1/GF2 MS: OpenSICDR список — https://github.com/dr-lizhiwei/OpenSICDR

### 3.4 Тяжёлые / неудобные

| Датасет | Размер | Ссылка | Комментарий |
|---------|--------|--------|-------------|
| **95-Cloud** | больше 38-Cloud (ещё 57 сцен train) | https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset | То же семейство; ROI низкий, пока нет Photos |
| **CloudSEN12 / +** | ~1 TB legacy / **~248 GB** HF taco | https://cloudsen12.github.io/ · https://huggingface.co/datasets/tacofoundation/cloudsen12 | Не для нашего диска |
| **GF1_WHU** (полные сцены CloudMamba) | Google Drive / Baidu | https://github.com/dr-lizhiwei/GF1_WHU | Тяжелее pack’а GF1MS; для CloudMamba-цифр |
| **Levir_CS** | **34.42 GB** 7z (full-res огромный) | https://github.com/permanentCH5/GeoInfoNet | Пропустить |
| **L7 Irish / полный Biome scenes** | большие сцены USGS | через OpenSICDR | Позже, если нужен multi-class snow |

Каталог открытых cloud datasets: https://github.com/dr-lizhiwei/OpenSICDR · https://zhiweili.net/resources/

---

## 4. Сопоставление с Photos (кубсат RGB)

| Датасет | RGB? | GSD близко к ~200 м? | Похожесть на Photos | Комментарий |
|---------|------|----------------------|---------------------|-------------|
| Photos (целевой) | да | да | 100% | Разметка `labeling/photos_40` |
| HRC_WHU | да | нет (0.5–15 м, мельче) | средняя (RGB, сложный фон) | Лучший лёгкий доп. для границ/фона |
| 38-Cloud RGB | да (без NIR) | ~30 м | низкая–средняя | Train-объём |
| SPARCS | можно RGB из bands | ~30 м | низкая–средняя | Holdout + тени/снег |
| GF1 MS | 4 bands VIS–NIR | 8–16 м | средняя | SOTA-совместимый бенчмарк |
| CloudSEN12 | MSI Sentinel | 10 м | низкая (много каналов) | Только если гоняешь adapter-story |
| S2 Catalogue | MSI / thumbs RGB | 20 м | средняя как «другой сенсор» | thumbs дёшево |

**Главный gap закрывается не 95-Cloud, а разметкой Photos.** Публичные сеты = pretrain / ablations / слайд «сравнивали с литературой».

---

## 5. Практический план скачивания (под ~57 GB free)

1. **Сейчас (~110 MB):**
   ```bash
   huggingface-cli download --repo-type dataset XavierJiezou/cloud-adapter-datasets --local-dir "e:/Cloudness detection/data" --include hrc_whu.zip
   ```
   Распаковать в `data/hrc_whu/`.

2. **Навести порядок:** `data/sending/` → `data/sparcs/` (это уже SPARCS).

3. **Если останется >5 GB и хочется SOTA-таблицу как у Cloud-Adapter:**  
   `gf12ms_whu_gf1.zip` (~0.9 GB).

4. **Не качать**, пока не почистишь диск / не закончишь Photos: полный CloudSEN12, Levir_CS, второй кусок 95-Cloud «на всякий».

5. Для презентации без train: `thumbnails.zip` S2 Catalogue (~146 MB) с Zenodo.

---

## 6. Как обновлять

- Новые сеты из статей → строка в §2 + карточка в §3 с **размером и URL**.  
- Перед скачиванием сверять свободное место (`Get-PSDrive E`).  
- Сырые `Photos/` в git не класть (см. `AGENTS.md`).
