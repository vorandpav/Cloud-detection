"""Factory: unet | segformer_b2 | cloudnet | cloudnet_plus | cloud_adapter."""

from __future__ import annotations

import torch.nn as nn

MODEL_NAMES = ("unet", "segformer_b2", "cloudnet", "cloudnet_plus", "cloud_adapter")


def list_models() -> tuple[str, ...]:
    """Известные имена моделей для CLI."""
    return MODEL_NAMES


def build_model(
    name: str,
    *,
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
) -> nn.Module:
    """Собирает модель по короткому имени.

    Args:
        name: ``unet`` | ``segformer_b2`` | ``cloudnet`` | ``cloudnet_plus`` | ``cloud_adapter``.
        encoder_weights: Веса энкодера SMP (``imagenet`` / None). Cloud-Net их не использует.
        in_channels: Число входных каналов (Photos/публичные = 3).

    Returns:
        nn.Module, логиты Bx1xHxW.

    Raises:
        ValueError: Неизвестное имя или cloud_adapter без локального порта.
    """
    key = name.lower().strip()
    if key in {"unet", "unet_resnet34"}:
        import segmentation_models_pytorch as smp

        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
    if key in {"segformer_b2", "segformer"}:
        import segmentation_models_pytorch as smp

        return smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
    if key in {"cloudnet", "cloud_net"}:
        from cloudness_detection.models.cloud_net import CloudNet

        return CloudNet(in_channels=in_channels, plus=False)
    if key in {"cloudnet_plus", "cloud_net_plus", "cloudnet+"}:
        from cloudness_detection.models.cloud_net import CloudNet

        return CloudNet(in_channels=in_channels, plus=True)
    if key == "cloud_adapter":
        raise ValueError(
            "Cloud-Adapter не встроен в этот каркас (нужен mmseg + DINOv2). "
            "Контекст: docs/CLOUD_DETECTION_RESEARCH.md."
        )
    raise ValueError(f"Неизвестная модель {name!r}. Доступны: {MODEL_NAMES}")
