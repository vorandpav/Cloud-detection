"""Фиксация всех источников случайности в одном месте."""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Фиксирует сиды Python и (если установлены) NumPy / PyTorch.

    Args:
        seed: Целое число, которое должно давать один и тот же порядок
            выборок и аугментаций при повторном запуске.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return
    if not hasattr(torch, "manual_seed"):
        raise RuntimeError(
            "В .venv битый torch (нет manual_seed). "
            "Удали его и поставь заново через uv."
        )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
