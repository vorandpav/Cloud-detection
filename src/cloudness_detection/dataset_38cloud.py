"""Датасет 38-Cloud: собираем RGB из отдельных каналов + бинарная маска.

Пока без PyTorch: отдаём NumPy. В train-скрипте обернём в Tensor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cloudness_detection.image_io import load_raster01
from cloudness_detection.seed import DEFAULT_SEED


class Cloud38RGBDataset:
    """Патчи 38-Cloud в формате RGB + маска облаков.

    Attributes:
        root: Папка ``data/38-cloud``.
        patch_names: Имена патчей без префикса канала.
        image_size: Если задано, ресайз к квадрату.
    """

    def __init__(
        self,
        root: Path,
        nonempty_only: bool = True,
        image_size: int | None = 384,
        max_patches: int | None = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """Args:
            root: Корень 38-Cloud.
            nonempty_only: Брать CSV без чёрных краёв, если файл есть.
            image_size: Сторона квадрата после resize; ``None`` = как есть.
            max_patches: Лимит для быстрых прогонов.
            seed: Сид для обрезки списка при ``max_patches``.
        """
        self.root = Path(root)
        self.image_size = image_size
        # Допускаем два корня: ``…/38-cloud`` (с подпапкой 38-Cloud_training)
        # или сразу ``…/38-Cloud_training`` (как symlink на Kaggle).
        self.train_dir = self._resolve_train_dir(self.root)

        nonempty_candidates = [
            self.root / "training_patches_38-cloud_nonempty.csv",
            self.train_dir / "training_patches_38-cloud_nonempty.csv",
            self.root.parent / "training_patches_38-cloud_nonempty.csv",
        ]
        full_candidates = [
            self.train_dir / "training_patches_38-Cloud.csv",
            self.root / "training_patches_38-Cloud.csv",
        ]
        nonempty = next((p for p in nonempty_candidates if p.is_file()), None)
        full = next((p for p in full_candidates if p.is_file()), None)
        if nonempty_only and nonempty is not None:
            csv_path = nonempty
        elif full is not None:
            csv_path = full
        elif nonempty is not None:
            csv_path = nonempty
        else:
            raise FileNotFoundError(
                f"Не найден CSV патчей 38-Cloud рядом с {self.root} "
                f"(train_dir={self.train_dir})"
            )
        names = pd.read_csv(csv_path)["name"].tolist()

        if max_patches is not None and max_patches < len(names):
            rng = np.random.default_rng(seed)
            names = list(rng.choice(names, size=max_patches, replace=False))

        self.patch_names = names

    @staticmethod
    def _resolve_train_dir(root: Path) -> Path:
        """Находит папку с ``train_red`` / CSV.

        Args:
            root: ``data/38-cloud`` или уже ``38-Cloud_training``.

        Returns:
            Путь к training-директории.
        """
        nested = root / "38-Cloud_training"
        if (nested / "train_red").is_dir():
            return nested
        if (root / "train_red").is_dir():
            return root
        if nested.is_dir():
            return nested
        raise FileNotFoundError(
            f"Не нашли 38-Cloud training (train_red) в {root} или {nested}"
        )

    def __len__(self) -> int:
        return len(self.patch_names)

    def _read_band(self, band: str, name: str) -> np.ndarray:
        """Читает один канал как float32 HxW в 0..1.

        Args:
            band: ``red`` / ``green`` / ``blue`` / ``gt``.
            name: Имя патча из CSV.
        """
        prefix = "gt" if band == "gt" else band
        folder = "train_gt" if band == "gt" else f"train_{band}"
        path = self.train_dir / folder / f"{prefix}_{name}.TIF"
        stretch = band != "gt"
        return load_raster01(path, percentile_stretch=stretch)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str | float]:
        """Возвращает ``image`` CxHxW, ``mask`` 1xHxW, имя и cloud_frac.

        Args:
            index: Индекс патча.
        """
        name = self.patch_names[index]
        red = self._read_band("red", name)
        green = self._read_band("green", name)
        blue = self._read_band("blue", name)
        gt = self._read_band("gt", name)

        rgb = np.stack([red, green, blue], axis=0)
        mask = (gt >= 0.5).astype(np.float32)[None, ...]

        if self.image_size is not None:
            rgb_img = Image.fromarray(
                (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
            )
            mask_img = Image.fromarray((mask[0] * 255).astype(np.uint8))
            rgb_img = rgb_img.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            mask_img = mask_img.resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST
            )
            rgb = np.transpose(np.asarray(rgb_img, dtype=np.float32) / 255.0, (2, 0, 1))
            mask = (np.asarray(mask_img, dtype=np.float32) / 255.0)[None, ...]

        return {
            "image": rgb,
            "mask": mask,
            "name": name,
            "cloud_frac": float(mask.mean()),
        }
