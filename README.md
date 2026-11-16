# Cloudness detection (кейс СПУТНИКС)

Оценка доли облаков на RGB-снимках обзорных камер кубсатов. Модель сегментирует облако по пикселям, доля считается как среднее по маске. Кадр отклоняется, если облачность больше 75%.

Обучение идёт на публичных наборах (HRC, SPARCS, 38-Cloud, GF1, Sentinel-2 Cloud Mask Catalogue). Photos в train этих прогонов не входят: это целевой домен, на нём смотрят перенос. Модель, которую оставляем для кейса, — U-Net `outputs/train/unet_mix_hrc_sparcs_38_4060/best.pt`. Каталог `outputs/` в `.gitignore`, этого чекпоинта в git нет: он должен лежать локально после обучения.

## Установка

Нужен Python 3.12 (в `pyproject.toml`: `>=3.12,<3.14`) и [uv](https://docs.astral.sh/uv/). Зависимости и lock — в корне репозитория. Torch и torchvision берутся с индекса CUDA 12.4, он уже прописан в `pyproject.toml` (`tool.uv.sources`), отдельно ставить PyTorch не нужно.

Из корня репозитория:

```powershell
uv sync
```

Точка входа обучения: `train-mix` → `cloudness_detection.train.train_mix:main`.

```powershell
uv run train-mix --help
```

То же самое без entry point: `uv run python -m cloudness_detection.train.train_mix`.

## Где лежат данные

Корень публичных наборов — `<репозиторий>/data`. Его можно подменить переменной `CLOUDNESS_DATA_ROOT` или `DATA_ROOT` (берётся первая непустая). Корень репозитория — `CLOUDNESS_PROJECT_ROOT` или `PROJECT_ROOT`, иначе каталог на три уровня выше пакета `cloudness_detection.data`.

| Имя в `--datasets` | Путь | Что должен увидеть лоадер |
|---|---|---|
| `hrc` | `data/hrc_whu` | `img_dir/train`, `img_dir/test`, `ann_dir/train`, `ann_dir/test`, пары PNG с одинаковыми именами |
| `sparcs` | `data/sparcs` | `*_photo.png` и рядом `*_mask.png` или `*_labels.tif`. Если в `sparcs` нет `*_photo.png`, берётся `data/sending` |
| `cloud38` | `data/38-cloud` | патчи 38-Cloud (RGB; NIR для этого пайплайна не нужен) |
| `gf1` | `data/gf12ms_whu_gf1` | `img_dir/{train,val}` и `ann_dir/{train,val}` |
| `s2` | `data/s2_catalogue` | `subscenes/` и `masks/` (`.npy`) |
| `photos` | `labeling/photos_40` | JPG и, для train с масками, PNG в `masks/`. Сырые снимки кубсатов лежат в `Photos/` |

`Photos/` и `data/` в git не входят. Размеченная подвыборка для проверки на кубсате: `labeling/photos_40/` (превью в `images/`, маски CVAT в `masks/`). Команды ниже `photos` в микс не кладут.

Первое имя в `--datasets` — источник валидации. Для `hrc` это split `test`, без аугментаций. Поэтому `hrc` в командах стоит первым: val остаётся HRC test.

Веса по умолчанию в `src/cloudness_detection/data/mix.py`: `hrc=3`, `sparcs=2.5`, `s2=1.5`, `gf1=0.75`, `cloud38=0.5`. Это case-mix Cloud-Net, не микс победившего U-Net. Доля шага — вес / сумма весов, не число файлов. План `4060` сам ставит картинку 384 и batch 8; в командах ниже batch переопределён на 4. Устройство — CUDA, если она есть, иначе CPU. Отдельного флага устройства нет.

Одинаковый `--run-name` пишет в ту же папку и затирает `best.pt`. Для нового прогона имя лучше сменить. Папку лидера не переиспользуйте, если чекпоинт ещё нужен.

## Обучение

### U-Net, прогон лидера

Сохранённый `outputs/train/unet_mix_hrc_sparcs_38_4060/config.json`: U-Net, план `4060`, датасеты `hrc,sparcs,cloud38`, 20 эпох, batch 4, lr `3e-4`, картинка 384, энкодер ImageNet, seed 42. GF1 и S2 в этот микс не входили. Вес 38-Cloud в том прогоне был **1.0** (сейчас в коде по умолчанию 0.5), поэтому его надо передать явно. Loss для U-Net при `--loss auto` — Dice+BCE, оптимизатор AdamW. `--patience` в том `config.json` не записан; в CLI значение по умолчанию 5, и по истории лучшая эпоха — 4-я, после неё ещё пять эпох без нового лучшего val IoU.

```powershell
uv run train-mix --model unet --device-plan 4060 --datasets hrc,sparcs,cloud38 --dataset-weights "hrc=3,sparcs=2.5,cloud38=1" --epochs 20 --batch-size 4 --lr 3e-4 --image-size 384 --encoder-weights imagenet --seed 42 --patience 5 --run-name unet_mix_hrc_sparcs_38_4060
```

Короткий прогон на один шаг проверки лоадера: добавьте `--smoke` (1 эпоха, мало сэмплов) и другой `--run-name`.

### Cloud-Net и Cloud-Net+

Те же флаги, что в `outputs/train/cloudnet_case_mix_4060/config.json` и `outputs/train/cloudnet_plus_case_mix_4060/config.json`: пять наборов, веса case-mix, 6 эпох, batch 4, lr `1e-4`, картинка 384, seed 42, энкодер не используется (`--encoder-weights none` код приводит к `null` и для этих моделей всё равно игнорирует). `--epoch-size 2048` и `--patience 3` в эти `config.json` не сериализуются; они совпадают с описанием прогона в `docs/CLOUD_DETECTION_RESULTS.md`. При `--loss auto` Cloud-Net получает soft Jaccard, Cloud-Net+ — Filtered Jaccard (FJL1). Оптимизатор для обеих — Adam.

```powershell
uv run train-mix --model cloudnet --device-plan 4060 --datasets hrc,sparcs,s2,gf1,cloud38 --dataset-weights "hrc=3,sparcs=2.5,s2=1.5,gf1=0.75,cloud38=0.5" --epochs 6 --batch-size 4 --lr 1e-4 --image-size 384 --epoch-size 2048 --encoder-weights none --seed 42 --patience 3 --run-name cloudnet_case_mix_4060
```

```powershell
uv run train-mix --model cloudnet_plus --device-plan 4060 --datasets hrc,sparcs,s2,gf1,cloud38 --dataset-weights "hrc=3,sparcs=2.5,s2=1.5,gf1=0.75,cloud38=0.5" --epochs 6 --batch-size 4 --lr 1e-4 --image-size 384 --epoch-size 2048 --encoder-weights none --seed 42 --patience 3 --run-name cloudnet_plus_case_mix_4060
```

`cloud_adapter` этим скриптом не обучается: в каркасе нет порта mmseg + DINOv2.

## Оценка на Photos

Отдельной CLI с флагами нет. Скрипт `scripts/eval_photos_checkpoints.py` жёстко сравнивает три чекпоинта на 8 кадрах из `labeling/photos_40` (ровно 8 масок, порог 0.5) и падает с `FileNotFoundError`, если любого файла нет:

- `outputs/train/unet_mix_hrc_sparcs_38_4060/best.pt`
- `outputs/train/unet_mix_hrc_sparcs_38_4060_colornorm/best.pt`
- `outputs/kaggle/kaggle_segformer_mix/best.pt`

```powershell
uv run python scripts/eval_photos_checkpoints.py
```

Результат: `outputs/reports/photos_compare_models/` (`REPORT.md`, `metrics.json`, оверлеи). Color-match на самих Photos скрипт не включает.

## Куда пишется обучение

`outputs/train/<run-name>/`:

- `best.pt` — лучший val IoU
- `config.json` — фактические параметры
- `history.csv`, `history.json`, `curves.png`
- `DONE.txt` (для `--smoke` — `SMOKE_OK.txt`)

## Отчёты для проверяющих

В `docs/` только эти три текста. Картинки к ним — `docs/results_figures/`.

- [`docs/CLOUD_DATASETS.md`](docs/CLOUD_DATASETS.md) — данные
- [`docs/CLOUD_DETECTION_RESEARCH.md`](docs/CLOUD_DETECTION_RESEARCH.md) — идеи и модели
- [`docs/CLOUD_DETECTION_RESULTS.md`](docs/CLOUD_DETECTION_RESULTS.md) — что учили и что победило

Модель кейса: `outputs/train/unet_mix_hrc_sparcs_38_4060/best.pt`.
