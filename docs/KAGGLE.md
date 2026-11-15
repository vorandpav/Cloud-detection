# Kaggle T4 — один ноутбук

Файл: [`../outputs/kaggle/train_mix_one_cell.ipynb`](../outputs/kaggle/train_mix_one_cell.ipynb)

## Перед запуском

1. Add Input: dataset с `cloudness_code.zip` + `sorour/38cloud-cloud-segmentation-in-satellite-images`
2. Accelerator **GPU T4**, **Internet ON**
3. Run **единственную code-ячейку**

Она ставит код, HRC (HF), SPARCS (OSU zip), symlink на 38-Cloud и запускает:

```text
train-mix --model segformer_b2 --device-plan t4 --datasets hrc,sparcs,cloud38 --epochs 10 --seed 42
```

Ожидаемый ориентир: best val IoU **≈ 0.885** (см. `docs/FINAL_CASE_REPORT.md`).

Локальный победитель кейса — **U-Net** (`train-mix --model unet --device-plan 4060`), не SegFormer.
