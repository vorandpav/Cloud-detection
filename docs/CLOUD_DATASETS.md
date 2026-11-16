# Датасеты облаков для кейса Photos (кубсат RGB → %)

Задача известная - сегментация облаков на оптике - оценка % облачности и reject@75%. Проблема именно данного датасета Photos:

- неизвестный для моделей датасет
- отсутсвие GT
- ограничение RGB

---

## 1. Короткий вердикт

| Датасет | Использовали | Почему | Как |
|---------|--------------|--------|-----|
| **Photos** | **Да** (продукт) | Домен сдачи | Eval продукта: % / reject@75; 8 CVAT — ориентир; в train победителя не входили |
| **HRC_WHU** | **Да** | Хорошие RGB-маски, ближе к Photos по цвету | Train вес **3.0**; **val = HRC test** (30 сцен) |
| **SPARCS** | **Да** | Хорошие multi-class маски, shadow/snow | Train вес **2.5**; binary cloud из маски |
| **38-Cloud** | **Да** | Большой labeled RGB-объём; маски посредственные и не лечат under-seg | Train вес **0.5** |
| **GF1 pack** | **Да** | Маски **крупнее** истинного облака (over-mask) — контр-смещение к under-seg | Train вес **0.75** |
| **S2 Catalogue** | **Да** | Экспертные маски высокого качества | Train вес **1.5**; другой cloud-fraction — не повод выкидывать |
| 95-Cloud / CloudSEN12 / Levir / полные GF1_WHU | **Нет** | Другой Landsat-мир, другой MSI-протокол или другой масштаб | Не в case-mix |

**Case-mix (train):** HRC + SPARCS + S2 + GF1 + 38-Cloud, веса **3.0 / 2.5 / 1.5 / 0.75 / 0.5**. Сэмплирование **по весам**, не по числу файлов. Val = **HRC test** (`hrc` первый в `--datasets`). Чекпоинт `unet_mix_hrc_sparcs_38_4060` снят со старого микса без GF1 и S2 — его нужно переобучить на этих весах.

---

## 2. Photos (целевой)

| Поле | Значение |
|------|----------|
| Сенсор / тип | RGB JPG обзорных камер кубсатов |
| GSD | порядка **~200 м/пкс** |
| Объём | **4272** JPG (сводка датасетов); **4270** с валидным KML (EDA photos) |
| Размер кадра | mean ≈ **3280×2464**; labeling-превью **768×577** |
| Метаданные | центр, борт, время (KML / каталог) |
| Борта / даты | **38** бортов; **2024-10-30 → 2026-11-14** |
| География | lat **−80.9…82.8**, lon **−179.7…179.0** |
| Маски | нет на старте; 8 CVAT — грубая кисть |
| Прокси | `white_frac`: datasets n=120 mean **0.0685**; deep n=240 mean **0.0507** |

![Photos sample 1](results_figures/photos_sample_1.png)

![Photos sample 2](results_figures/photos_sample_2.png)

![Photos sample 3](results_figures/photos_sample_3.png)

![Photos labeling preview](results_figures/photos_labeling_01.png)

---

## 3. HRC_WHU

**Использовали:** **Да**.  
**Почему:** RGB, хорошие маски, ближе всех публичных к Photos по среднему RGB (L2 **0.248**); на нём меряют Cloud-Adapter / NFCNet / HRCloudNet — удобный якорь val.  
**Как:** train **120** / test **30**; в mix вес **3.0**; **val кейса = HRC test**.

| Поле | Значение |
|------|----------|
| Формат | RGB **1280×720** |
| Объём / split | **150** сцен: train **120** / test **30** |
| GSD | **0.5–15 м**; 5 типов поверхности (water / veg / urban / snow / barren) |
| Cloud frac | mean **0.342**, median **0.320** (n=150) |
| Качество масок | **хорошее** |
| Источник | [HRC_WHU](https://github.com/dr-lizhiwei/HRC_WHU) · pack Cloud-Adapter datasets |

**Пределы:** GSD мельче кубсата — текстуры краёв другие.

![HRC RGB](results_figures/hrc_sample_1.png)

![HRC overlay](results_figures/hrc_sample_1_overlay.png)

---

## 4. SPARCS

**Использовали:** **Да**.  
**Почему:** валидационный Landsat с честными multi-class масками и fail-modes (shadow/snow), которых мало у 38-Cloud; в литературе — holdout Cloud-Net+ / TransCNet / CLiSA.  
**Как:** **80** сцен; в mix вес **2.5**; для train — binary cloud из маски; отдельно — анализ классов / snow.

| Поле | Значение |
|------|----------|
| Источник | USGS SPARCS validation |
| Сцены | **80** × типично **1000×1000** |
| Вход | RGB (`*_photo.png`) |
| Классы маски | shadow / shadow_water / water / snow / land / cloud / flooded |
| Cloud (класс 5) | mean **19.4%**, snow **5.3%**, сцен >75% — **2/80** |
| Качество масок | **хорошее** |
| L2 RGB → Photos | **0.577** (дальше всех среди рассмотренных публичных) |

**Пределы:** по цвету далеко от Photos — trade-off в пользу качества масок.

![SPARCS RGB](results_figures/sparcs_sample_1.png)

![SPARCS overlay](results_figures/sparcs_sample_1_overlay.png)

---

## 5. 38-Cloud

**Использовали:** **Да**.  
**Почему:** большой labeled RGB-объём, ручные маски (не QA60), классический бенчмарк Cloud-Net; удобный размер патчей под обучение.  
**Как:** ~**8400** train GT, RGB (без NIR — под Photos); в mix вес **0.5**. Раньше стояло **1.0** — и так ниже HRC/SPARCS. Снизили ещё: маски посредственные и **не** шире облака, а продукт систематически under-seg; контр-смещение даёт GF1. Доля шагов ~**6%**, хотя патчей тысячи.

| Поле | Значение |
|------|----------|
| Сенсор | Landsat-8, патчи **384²** |
| Каналы | **RGB** (NIR не использовали) |
| Train GT | **8400** |
| GSD | ~**30 м** |
| Cloud frac (n=398 nonempty) | mean **0.500**, median **0.461** |
| Cloud frac (EDA n=800) | mean **49.6%**, median **43.8%**, доля >75% **36.8%** |
| Качество масок | **посредственное** — грубые границы, артефакты краёв |
| Источник | [38-Cloud](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset) |

**Пределы:** mediocre masks; prior облачности гуще, чем у Photos; GSD/оптика далеко от кубсата — не «учитель истины».

![38-Cloud RGB](results_figures/cloud38_sample_1.png)

![38-Cloud overlay](results_figures/cloud38_sample_1_overlay.png)

---

## 6. GF1 pack (`gf12ms_whu_gf1`)

**Использовали:** **Да**.  
**Почему:** в отличие от 38-Cloud, маски нарисованы **крупнее истинного облака** (over-mask). Тестируемые модели страдали **under-segmentation**, поэтому GF1 — контр-смещение: учит не зажимать маску. Границы сами по себе посредственные, cloud-fraction низкий (mean ~**0.09–0.10**, near-empty ~**66%**) — вес не на уровне HRC/SPARCS.  
**Как:** train-split pack'а; в загрузчике отсечка `min_cloud_frac=0.05` (почти ясные кадры не учим); вес **0.75** — чуть выше нового веса 38-Cloud (**0.5**). На диске train **6343** / val **4085** PNG ~256²; в микс входит train. Доля шагов ~**9%**: её задаёт вес, не число PNG.

| Поле | Значение |
|------|----------|
| Формат | PNG ~**256²**, pack Cloud-Adapter (не полные сцены GF1_WHU CloudMamba) |
| Пары / split | **10428**; train **6343** / val **4085** |
| Cloud frac | datasets mean **0.086**, median **0.001**; deep n=400 mean **0.101** |
| Качество масок | **посредственное**, контур **шире** облака (over-mask) |

![GF1 RGB](results_figures/gf1_sample_1.png)

![GF1 overlay](results_figures/gf1_sample_1_overlay.png)

---

## 7. Sentinel-2 Cloud Mask Catalogue (Francis)

**Использовали:** **Да**.  
**Почему:** экспертные маски CLEAR/CLOUD/SHADOW — высокое качество разметки. Бимодальная облачность (много почти 0 и почти 1) **не** причина выкидывать сет: другой prior cloud-fraction не отменяет метки.  
**Как:** **513** subscenes; RGB = B4, B3, B2; в train бинарное облако = канал CLOUD; вес **1.5** (ниже HRC **3.0** и SPARCS **2.5**, заметно выше GF1/38). Доля шагов ~**18%** при 513 кадрах.

| Поле | Значение |
|------|----------|
| Кадры | **513** subscenes **1022²** @ **20 m** |
| Маски | CLEAR / CLOUD / SHADOW (экспертные) |
| RGB для EDA | B4,B3,B2 + percentile 2–98 |
| Cloud frac (n=513) | mean **0.526**, median **0.558**; almost_empty **0.326**, almost_full **0.398** |
| Источник | Zenodo [4172871](https://zenodo.org/records/4172871) |

![S2 RGB](results_figures/s2_sample_1.png)

![S2 overlay](results_figures/s2_sample_1_overlay.png)

---

## 8. Что сознательно не брали

GF1 pack и S2 Catalogue **в** case-mix (§6–7). Ниже — другие сеты, их по-прежнему нет.

| Датасет | Использовали | Почему отбросили |
|---------|--------------|------------------|
| **95-Cloud** | Нет | Расширение того же Landsat RGBNir-мира, что 38-Cloud; мало новой информации для Photos % |
| **CloudSEN12** (full / high) | Нет | Другой MSI-протокол, не каталог Francis (§7) |
| **L8 Biome / GF2** | Нет | Имеет смысл под Cloud-Adapter-таблицу, не под % на кубсате в этом кейсе |
| **Levir_CS / полные сцены GF1_WHU** | Нет | Протоколы CloudMamba / другой масштаб; в mix входит pack `gf12ms_whu_gf1`, не эти сцены |

Паттерн литературы (для контекста, не «наш лидерборд»): свежие работы крутятся вокруг HRC + GF + CloudSEN12 + L8 Biome. Это объясняет, **какие** мосты мы рассматривали; критерий кейса всё равно Photos.

| Модель (год) | Их train / eval | Ссылка |
|--------------|-----------------|--------|
| Cloud-Adapter (2025) | HRC, GF1, GF2, CloudSEN12-high, L8 Biome | https://xavierjiezou.github.io/Cloud-Adapter/ |
| CloudMamba (2026) | GF1_WHU, Levir_CS | https://arxiv.org/html/2604.06844 |
| NFCNet (2025) | HRC_WHU, CHLandsat8, 95-Cloud RGB | https://pmc.ncbi.nlm.nih.gov/articles/PMC11861746/ |
| TransCNet (2025) | CHLandsat-8, 38-Cloud, SPARCS | https://doi.org/10.3390/app15189870 |
| Cloud-Net / + | 38-Cloud / 95-Cloud, SPARCS, Biome | https://arxiv.org/abs/2001.08768 |
| SegFormer vs U-Net (2025) | S2 Cloud Mask Catalogue | https://doi.org/10.5753/wcama.2025.7754 |

---

## 9. EDA: распределения, баланс, domain gap

Таблицы — из `outputs/eda/**` и пересчёта тех же CSV (seed **42**).

### 9.1. Cloud / proxy (deep)

![Гистограммы cloud fraction (deep)](results_figures/deep_cloud_fraction_hists.png)

| Датасет | n | mean | median | std | near-empty | near-full | p25 | p75 |
|---------|--:|-----:|-------:|----:|-----------:|----------:|----:|----:|
| Photos (`white_frac`) | 240 | 0.0507 | 0.0317 | 0.0696 | 0.617 | 0.000 | 0.0101 | 0.0697 |
| 38-Cloud | 397 | 0.4853 | 0.4252 | 0.4062 | 0.249 | 0.285 | 0.0508 | 0.9871 |
| SPARCS (класс 5) | 80 | 0.1937 | 0.1388 | 0.1867 | 0.263 | 0.000 | 0.0447 | 0.2494 |
| HRC_WHU | 150 | 0.3423 | 0.3195 | 0.1837 | 0.007 | 0.000 | 0.1875 | 0.4583 |
| GF1 | 400 | 0.1011 | 0.0024 | 0.1972 | 0.663 | 0.010 | 0.0000 | 0.1107 |
| S2 | 513 | 0.5258 | 0.5584 | 0.4458 | 0.326 | 0.398 | 0.0006 | 1.0000 |

Photos по прокси «яснее»: **~62%** near-empty; у 38-Cloud **~28%** near-full. Равный микс плотных Landsat-облаков учит prior, которого у типичного Photos нет.

**20-bin counts** (0..1), где списки значений полные:

| bin | Photos n=240 | HRC n=150 | SPARCS deep n=80 | S2 n=513 |
|-----|-------------:|----------:|-----------------:|---------:|
| 0.00–0.05 | 148 | 1 | 21 | 167 |
| 0.05–0.10 | 65 | 11 | 10 | 13 |
| 0.10–0.15 | 18 | 13 | 15 | 9 |
| 0.15–0.20 | 3 | 18 | 6 | 14 |
| 0.20–0.25 | 2 | 9 | 8 | 7 |
| 0.25–0.30 | 0 | 19 | 4 | 5 |
| 0.30–0.35 | 1 | 10 | 3 | 4 |
| 0.35–0.40 | 2 | 16 | 1 | 9 |
| 0.40–0.45 | 0 | 12 | 2 | 7 |
| 0.45–0.50 | 0 | 9 | 2 | 11 |
| 0.50–0.55 | 0 | 10 | 3 | 9 |
| 0.55–0.60 | 0 | 6 | 1 | 8 |
| 0.60–0.65 | 0 | 5 | 1 | 6 |
| 0.65–0.70 | 0 | 7 | 1 | 8 |
| 0.70–0.75 | 1 | 2 | 0 | 6 |
| 0.75–0.80 | 0 | 0 | 2 | 7 |
| 0.80–0.85 | 0 | 1 | 0 | 5 |
| 0.85–0.90 | 0 | 1 | 0 | 3 |
| 0.90–0.95 | 0 | 0 | 0 | 11 |
| 0.95–1.00 | 0 | 0 | 0 | 204 |

Для deep-панелей **38-Cloud n=397** и **GF1 n=400** полные per-image списки в CSV не сохранены — bin-таблицы этих двух панелей не восстановлены; есть summary выше.

### 9.2. Photos: покрытие

![Карта центров Photos](results_figures/photos_map_centers.png)

| Показатель | Значение |
|------------|----------|
| Кадров с KML | 4270 |
| Бортов | 38 |
| Даты | 2024-10-30 → 2026-11-14 |
| Lat / lon | −80.9…82.8 / −179.7…179.0 |

![Месяцы и борта](results_figures/photos_month_and_boards.png)

| lat_band | n |
|----------|--:|
| tropics | 1843 |
| mid | 1842 |
| polar | 585 |

| month | n |
|------:|--:|
| 1 | 249 |
| 2 | 196 |
| 3 | 181 |
| 4 | 239 |
| 5 | 260 |
| 6 | 275 |
| 7 | 252 |
| 8 | 319 |
| 9 | 568 |
| 10 | 962 |
| 11 | 415 |
| 12 | 354 |

| board_id (top-10) | n |
|-------------------|--:|
| SXC3-2316 | 280 |
| SXC3-2313 | 275 |
| SXC3-235 | 227 |
| SXC3-233 | 202 |
| SXC3-234 | 201 |
| SXC3-2310 | 183 |
| SXC3-2342 | 177 |
| SXC3-232 | 174 |
| SXC3-2344 | 170 |
| SXC3-2311 | 167 |

![Гео lat/lon deep](results_figures/deep_geo_latlon.png)

Полный каталог центров: [`photos_catalog.csv`](../outputs/eda/photos/photos_catalog.csv).

### 9.3. Photos: white_frac

![White frac Photos](results_figures/photos_white_frac_hist.png)

n=**120**, seed 42; mean **0.0685**.

| bin | count |
|-----|------:|
| 0.00–0.05 | 48 |
| 0.05–0.10 | 42 |
| 0.10–0.15 | 23 |
| 0.15–0.20 | 5 |
| 0.20–0.25 | 1 |
| 0.25–0.30 | 0 |
| 0.30–0.35 | 0 |
| 0.35–0.40 | 1 |
| 0.40–1.00 | 0 |

![Цвет и white proxy](results_figures/photos_color_and_white.png)

![Широта vs white](results_figures/photos_lat_vs_white.png)

Точки scatter целиком в md не выгружены; цвет — §9.4 (deep n=240, каталог n=400).

### 9.4. Цвет RGB и gap к Photos

![RGB brightness hist](results_figures/deep_rgb_brightness_hist.png)

![RGB mean box](results_figures/deep_rgb_mean_box.png)

| Датасет | n | mean R | mean G | mean B | hue_mean | L2 → Photos |
|---------|--:|-------:|-------:|-------:|---------:|------------:|
| Photos | 240 | 0.418 | 0.518 | **0.708** | **0.613** | 0 |
| HRC_WHU | 150 | 0.491 | 0.494 | 0.473 | 0.314 | **0.248** |
| S2 | 180 | 0.429 | 0.429 | 0.425 | 0.392 | 0.297 |
| 38-Cloud | 180 | 0.374 | 0.374 | 0.369 | 0.329 | 0.371 |
| GF1 | 180 | 0.213 | 0.391 | 0.390 | 0.481 | 0.399 |
| SPARCS | 80 | 0.243 | 0.287 | 0.209 | 0.256 | 0.577 |

Photos «синее» (B≈0.71). HRC ближе всех по L2; SPARCS дальше всех по цвету, но ценен масками.

![Photos vs 38-Cloud color](results_figures/compare_color_hist_overlay.png)

Ранний compare (Photos **400** vs 38-Cloud **800**): mean RGB Photos (0.419, 0.521, 0.712), 38-Cloud (0.370, 0.368, 0.362), L2 **0.385** — тот же порядок, что deep ≈0.371.

KS / Mann–Whitney Photos vs публичные по каналам и cloud_frac: p крайне малы — распределения **не совпадают**. Пример (vs 38-Cloud, `mean_b`): KS=0.926, p≈3e−96. Полная таблица: [`stat_tests_photos_vs_public.csv`](../outputs/eda/deep/stat_tests_photos_vs_public.csv).

### 9.5. 38-Cloud: hist и reject@75

![38-Cloud cloud hist](results_figures/eda38_cloud_fraction_hist.png)

`cloud_frac×100`, 25 bins, n=**800**:

| bin % | count |
|------:|------:|
| 0–4 | 163 |
| 4–8 | 23 |
| 8–12 | 23 |
| 12–16 | 22 |
| 16–20 | 31 |
| 20–24 | 32 |
| 24–28 | 28 |
| 28–32 | 22 |
| 32–36 | 24 |
| 36–40 | 17 |
| 40–44 | 16 |
| 44–48 | 20 |
| 48–52 | 17 |
| 52–56 | 14 |
| 56–60 | 18 |
| 60–64 | 15 |
| 64–68 | 5 |
| 68–72 | 9 |
| 72–76 | 10 |
| 76–80 | 20 |
| 80–84 | 17 |
| 84–88 | 12 |
| 88–92 | 11 |
| 92–96 | 20 |
| 96–100 | 211 |

mean **49.6%**, median **43.8%**, доля >75% **36.8%**.

![38-Cloud reject@75](results_figures/eda38_reject75_balance.png)

| Класс | n | доля |
|-------|--:|-----:|
| keep ≤75% | 506 | 0.6325 |
| delete >75% | 294 | 0.3675 |

![38-Cloud RGB vs cloud](results_figures/eda38_rgb_vs_cloud.png)

![38-Cloud datasets hist](results_figures/datasets38_cloud_fraction_hist.png)

| n_stats | mean | median |
|--------:|-----:|-------:|
| 398 | 0.5003 | 0.4608 |

Bin-counts для n=398 не сохранены.

### 9.6. SPARCS: классы и cloud/snow

![SPARCS class balance](results_figures/sparcs_class_balance.png)

| code | class | pixels | frac |
|-----:|-------|-------:|-----:|
| 0 | shadow | 5 691 893 | 0.0711 |
| 1 | shadow_water | 207 571 | 0.0026 |
| 2 | water | 4 521 053 | 0.0565 |
| 3 | snow | 4 241 560 | 0.0530 |
| 4 | land | 48 661 507 | 0.6083 |
| 5 | cloud | 15 497 471 | 0.1937 |
| 6 | flooded | 1 178 945 | 0.0147 |

![SPARCS cloud and snow](results_figures/sparcs_cloud_and_snow.png)

| Метрика (класс 5 / 3) | Значение |
|-----------------------|----------|
| mean cloud_frac | 0.1937 |
| mean snow_frac | 0.0530 |
| reject75 scenes | 2 / 80 |

Hist cloud% (16 bins, 0..100):

| bin % | count |
|------:|------:|
| 0.00–6.25 | 21 |
| 6.25–12.50 | 13 |
| 12.50–18.75 | 16 |
| 18.75–25.00 | 10 |
| 25.00–31.25 | 5 |
| 31.25–37.50 | 2 |
| 37.50–43.75 | 1 |
| 43.75–50.00 | 4 |
| 50.00–56.25 | 3 |
| 56.25–62.50 | 1 |
| 62.50–68.75 | 2 |
| 68.75–75.00 | 0 |
| 75.00–81.25 | 2 |
| 81.25–100 | 0 |

![SPARCS datasets hist](results_figures/sparcs_cloud_fraction_hist.png)

| n=80 (класс 5) | mean | median |
|----------------|-----:|-------:|
| cloud_frac | 0.1937 | 0.1388 |

![SPARCS examples](results_figures/sparcs_examples_grid.png)

### 9.7. HRC / GF1 / S2 hists

![HRC cloud hist](results_figures/hrc_cloud_fraction_hist.png)

| bin | count (n=150) |
|-----|--------------:|
| 0.00–0.05 | 1 |
| 0.05–0.10 | 11 |
| 0.10–0.15 | 13 |
| 0.15–0.20 | 18 |
| 0.20–0.25 | 9 |
| 0.25–0.30 | 19 |
| 0.30–0.35 | 10 |
| 0.35–0.40 | 16 |
| 0.40–0.45 | 12 |
| 0.45–0.50 | 9 |
| 0.50–0.55 | 10 |
| 0.55–0.60 | 6 |
| 0.60–0.65 | 5 |
| 0.65–0.70 | 7 |
| 0.70–0.75 | 2 |
| 0.75–0.80 | 0 |
| 0.80–0.85 | 1 |
| 0.85–0.90 | 1 |
| 0.90–1.00 | 0 |

mean **0.342**, median **0.320**.

![GF1 cloud hist](results_figures/gf1_cloud_fraction_hist.png)

| n_pairs | mean | median |
|--------:|-----:|-------:|
| 10428 | 0.0860 | 0.0011 |

Bin-counts сэмпла гистограммы не сохранены.

![S2 cloud hist](results_figures/s2_cloud_fraction_hist.png)

| bin | count (n=513) |
|-----|--------------:|
| 0.00–0.05 | 167 |
| 0.05–0.10 | 13 |
| 0.10–0.15 | 9 |
| 0.15–0.20 | 14 |
| 0.20–0.25 | 7 |
| 0.25–0.30 | 5 |
| 0.30–0.35 | 4 |
| 0.35–0.40 | 9 |
| 0.40–0.45 | 7 |
| 0.45–0.50 | 11 |
| 0.50–0.55 | 9 |
| 0.55–0.60 | 8 |
| 0.60–0.65 | 6 |
| 0.65–0.70 | 8 |
| 0.70–0.75 | 6 |
| 0.75–0.80 | 7 |
| 0.80–0.85 | 5 |
| 0.85–0.90 | 3 |
| 0.90–0.95 | 11 |
| 0.95–1.00 | 204 |

mean **0.526**, median **0.558**.

### 9.8. Эвристический baseline (stage-1)

На **300** патчах 38-Cloud (seed 42): яркость + низкая насыщенность → MAE% ≈ **42%**, accuracy keep/delete@75 ≈ **0.68**. Пол для нейросети на публичном протоколе; на Photos эвристика ещё опаснее.

### 9.9. Выводы EDA → выбор данных

1. С нуля на Photos нельзя — нужен transfer на публичных RGB-масках.
2. Веса неравные: HRC↑, SPARCS↑, 38-Cloud↓ — из качества масок, цвета и prior облачности.
3. GF1 в mix как over-mask против under-seg (вес **0.75**, выше 38-Cloud **0.5**). S2 в mix из‑за качества меток (вес **1.5**); другой cloud-fraction — не повод его снимать.
4. Gap продукта закрывает **аккуратная разметка Photos**, не ещё один публичный Landsat/MSI.
5. Продукт: **MAE%** и **reject@75 на Photos**; IoU на HRC — сходимость, не финал.

---

## 10. Веса микса

Доля шага = вес / **8.25**. Длины (train, файлы): HRC **120**, SPARCS **80**, S2 **513**, GF1 **6343** (после `min_cloud_frac=0.05` меньше), 38-Cloud ~**8400**. Длина эпохи по умолчанию — сумма длин, состав батча — веса.

| Источник | Вес | Доля шагов | Роль |
|----------|----:|----------:|------|
| HRC | **3.0** | 36% | train + **val** |
| SPARCS | **2.5** | 30% | train; binary cloud = класс **5** |
| S2 Catalogue | **1.5** | 18% | train; качественные экспертные маски |
| GF1 pack | **0.75** | 9% | train; over-mask против under-seg |
| 38-Cloud | **0.5** | 6% | train; объём, самый низкий вес |
| Photos | **0** | — | не в этом mix; product eval / после нормальных масок |

---

## 11. Сопоставление с Photos

| Датасет | RGB? | GSD ~200 м? | Близость к Photos | Роль в кейсе |
|---------|------|-------------|-------------------|--------------|
| Photos | да | да | целевой | product eval |
| HRC | да | нет (0.5–15 м) | выше (цвет) | train↑ + val |
| SPARCS | да | ~30 м | ниже по цвету | train↑ (маски) |
| 38-Cloud | да | ~30 м | ниже–средняя | train, вес **0.5** |
| GF1 pack | PNG из MS | 8–16 м класс | ниже по цвету | train, вес **0.75** (over-mask) |
| S2 Catalogue | RGB из MSI | 20 м | другой сенсор, метки сильные | train, вес **1.5** |
