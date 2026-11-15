# Cloudness detection (кейс СПУТНИКС)

Оценка **% облачности** на RGB Photos кубсатов и **reject @ 75%**.

## Итог

- **Модель:** U-Net ResNet34 — `outputs/train/unet_mix_hrc_sparcs_38_4060/best.pt`
- **Val IoU ≈ 0.881** (HRC holdout). На Photos — domain gap (under-seg).
- **Отчёт:** [`docs/FINAL_CASE_REPORT.md`](docs/FINAL_CASE_REPORT.md)
- **Преза (кейс / case deck):** [`Presentation.pptx`](Presentation.pptx) — презентация кейса cloud detection (СПУТНИКС)
- **Преза (контент / markdown):** [`outputs/presentation/PRESENTATION.md`](outputs/presentation/PRESENTATION.md)

## Быстрый старт (локально)

```bash
uv sync
# данные: data/hrc_whu, data/sparcs (или sending), data/38-cloud, Photos/
uv run train-mix --model unet --device-plan 4060 \
  --datasets hrc,sparcs,cloud38 --epochs 20 --batch-size 4 --seed 42 \
  --run-name unet_mix_hrc_sparcs_38_4060
```

Сид по умолчанию: **42**. Правила кода: [`AGENTS.md`](AGENTS.md).

## Kaggle T4

Одна клетка: [`outputs/kaggle/train_mix_one_cell.ipynb`](outputs/kaggle/train_mix_one_cell.ipynb) · кратко [`docs/KAGGLE.md`](docs/KAGGLE.md).

## Данные

См. [`docs/DATASETS_FINAL.md`](docs/DATASETS_FINAL.md) и [`docs/CLOUD_DATASETS.md`](docs/CLOUD_DATASETS.md).  
Разметка Photos: `labeling/photos_40/` (8 масок CVAT в `masks/`).

## Литература / выбор моделей

[`docs/CLOUD_DETECTION_RESEARCH.md`](docs/CLOUD_DETECTION_RESEARCH.md)
