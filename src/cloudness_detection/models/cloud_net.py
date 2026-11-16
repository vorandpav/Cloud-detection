"""Cloud-Net и Cloud-Net+ на RGB (порт Keras-модели Mohajerani & Saeedi).

Оригинал: https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection
Вход статьи — 192×192×4 (RGBNir). Здесь stem на 3 канала: в кейсе нет NIR.
Выход — логиты Bx1xHxW (sigmoid снаружи, как у U-Net в этом репозитории).
Сторона входа должна делиться на 32.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNReLU(nn.Module):
    """BatchNorm → ReLU, как ``bn_relu`` в Keras-коде."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(x))


def _conv(in_ch: int, out_ch: int, kernel: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=True)


class ContrArm(nn.Module):
    """Две свёртки 3×3 + shortcut 1×1, сложение с входом.

    Cloud-Net+: между двумя 3×3 вставлена 1×1.
    """

    def __init__(self, in_ch: int, filters: int, *, plus: bool) -> None:
        super().__init__()
        self.conv1 = _conv(in_ch, filters, 3)
        self.bn1 = BNReLU(filters)
        self.mid_1x1 = _conv(filters, filters, 1) if plus else None
        self.bn_mid = BNReLU(filters) if plus else None
        self.conv2 = _conv(filters, filters, 3)
        self.bn2 = BNReLU(filters)
        self.skip = _conv(in_ch, filters // 2, 1)
        self.bn_skip = BNReLU(filters // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn1(self.conv1(x))
        if self.mid_1x1 is not None and self.bn_mid is not None:
            y = self.bn_mid(self.mid_1x1(y))
        y = self.bn2(self.conv2(y))
        shortcut = torch.cat([x, self.bn_skip(self.skip(x))], dim=1)
        return F.relu(y + shortcut, inplace=False)


class ImprvContrArm(nn.Module):
    """Глубокий contracting-блок (3×3 ×3 в Cloud-Net).

    Cloud-Net+ убирает среднюю 3×3 и ставит вместо неё 1×1.
    """

    def __init__(self, in_ch: int, filters: int, *, plus: bool) -> None:
        super().__init__()
        self.plus = plus
        self.conv1 = _conv(in_ch, filters, 3)
        self.bn1 = BNReLU(filters)
        self.mid = _conv(filters, filters, 1 if plus else 3)
        self.bn_mid = BNReLU(filters)
        self.conv3 = _conv(filters, filters, 3)
        self.bn3 = BNReLU(filters)
        self.skip = _conv(in_ch, filters // 2, 1)
        self.bn_skip = BNReLU(filters // 2)
        self.side = _conv(filters, filters, 1)
        self.bn_side = BNReLU(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.bn1(self.conv1(x))
        mid = self.bn_mid(self.mid(first))
        y = self.bn3(self.conv3(mid))
        shortcut = torch.cat([x, self.bn_skip(self.skip(x))], dim=1)
        side_src = first if self.plus else mid
        side = self.bn_side(self.side(side_src))
        return F.relu(y + shortcut + side, inplace=False)


class Bridge(nn.Module):
    """Дно сети: две 3×3, dropout 0.15, shortcut. В Plus между ними 1×1."""

    def __init__(self, in_ch: int, filters: int, *, plus: bool) -> None:
        super().__init__()
        self.conv1 = _conv(in_ch, filters, 3)
        self.bn1 = BNReLU(filters)
        self.mid_1x1 = _conv(filters, filters, 1) if plus else None
        self.bn_mid = BNReLU(filters) if plus else None
        self.conv2 = _conv(filters, filters, 3)
        self.drop = nn.Dropout(0.15)
        self.bn2 = BNReLU(filters)
        self.skip = _conv(in_ch, filters // 2, 1)
        self.bn_skip = BNReLU(filters // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn1(self.conv1(x))
        if self.mid_1x1 is not None and self.bn_mid is not None:
            y = self.bn_mid(self.mid_1x1(y))
        y = self.bn2(self.drop(self.conv2(y)))
        shortcut = torch.cat([x, self.bn_skip(self.skip(x))], dim=1)
        return F.relu(y + shortcut, inplace=False)


class ExpBlock(nn.Module):
    """2 или 3 свёртки 3×3 в expanding-блоке. 1×1 сюда Plus не добавляет."""

    def __init__(self, in_ch: int, filters: int, n_convs: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_ch
        for _ in range(n_convs):
            layers.append(_conv(c_in, filters, 3))
            layers.append(BNReLU(filters))
            c_in = filters
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _stack_pool(x: torch.Tensor, copies: int, pool: int) -> torch.Tensor:
    """Повтор каналов и max-pool. Так Keras-цикл ``concatenate`` догоняет глубину skip."""
    if copies != 1:
        x = torch.cat([x] * copies, dim=1)
    return F.max_pool2d(x, kernel_size=pool, stride=pool)


class AggBranch(nn.Module):
    """Aggregation branch Cloud-Net+: каждая карта → 1 канал, upsample, сумма.

    Эквивалентно одному 1×1 после склейки upsampled-карт, но без гигантского тензора.
    """

    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(c, 1, kernel_size=1) for c in channels])

    def forward(self, feats: list[torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
        acc: torch.Tensor | None = None
        for proj, feat in zip(self.projs, feats, strict=True):
            y = proj(feat)
            if y.shape[-2:] != size:
                y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            acc = y if acc is None else acc + y
        if acc is None:
            raise RuntimeError("AggBranch получил пустой список карт")
        return acc


class CloudNet(nn.Module):
    """Cloud-Net (``plus=False``) или Cloud-Net+ (``plus=True``)."""

    def __init__(self, in_channels: int = 3, *, plus: bool = False) -> None:
        """Args:
            in_channels: 3 для RGB кейса. В статье было 4.
            plus: Cloud-Net+ (1×1 в contracting, без средней 3×3, aggregation branch).
        """
        super().__init__()
        self.plus = plus
        self.stem = _conv(in_channels, 16, 3)
        self.arm1 = ContrArm(16, 32, plus=plus)
        self.arm2 = ContrArm(32, 64, plus=plus)
        self.arm3 = ContrArm(64, 128, plus=plus)
        self.arm4 = ContrArm(128, 256, plus=plus)
        self.arm5 = ImprvContrArm(256, 512, plus=plus)
        self.bridge = Bridge(512, 1024, plus=plus)

        self.up7 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.exp7 = ExpBlock(1024, 512, n_convs=2 if plus else 3)
        self.up8 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.exp8 = ExpBlock(512, 256, n_convs=2)
        self.up9 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.exp9 = ExpBlock(256, 128, n_convs=2)
        self.up10 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.exp10 = ExpBlock(128, 64, n_convs=2)
        self.up11 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.exp11 = ExpBlock(64, 32, n_convs=2)
        self.head = nn.Conv2d(32, 1, kernel_size=1)
        self.agg = AggBranch((1024, 512, 256, 128, 64, 32)) if plus else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 32 or x.shape[-2] % 32:
            raise ValueError(f"Cloud-Net ждёт сторону кратную 32, получили {tuple(x.shape[-2:])}")

        c1 = F.relu(self.stem(x), inplace=False)
        c1 = self.arm1(c1)
        c2 = self.arm2(F.max_pool2d(c1, 2))
        c3 = self.arm3(F.max_pool2d(c2, 2))
        c4 = self.arm4(F.max_pool2d(c3, 2))
        c5 = self.arm5(F.max_pool2d(c4, 2))
        c6 = self.bridge(F.max_pool2d(c5, 2))

        skip7 = _ff4(c4, c3, c2, c1, c5)
        t7 = self.up7(c6)
        d7 = F.relu(self.exp7(torch.cat([t7, skip7], dim=1)) + c5 + t7, inplace=False)

        skip8 = _ff3(c3, c2, c1, c4)
        t8 = self.up8(d7)
        d8 = F.relu(self.exp8(torch.cat([t8, skip8], dim=1)) + c4 + t8, inplace=False)

        skip9 = _ff2(c2, c1, c3)
        t9 = self.up9(d8)
        d9 = F.relu(self.exp9(torch.cat([t9, skip9], dim=1)) + c3 + t9, inplace=False)

        skip10 = _ff1(c1, c2)
        t10 = self.up10(d9)
        d10 = F.relu(self.exp10(torch.cat([t10, skip10], dim=1)) + c2 + t10, inplace=False)

        t11 = self.up11(d10)
        d11 = F.relu(self.exp11(torch.cat([t11, c1], dim=1)) + c1 + t11, inplace=False)
        logits = self.head(d11)
        if self.agg is not None:
            logits = logits + self.agg([c6, d7, d8, d9, d10, d11], size=x.shape[-2:])
        return logits


def _ff4(
    c4: torch.Tensor,
    c3: torch.Tensor,
    c2: torch.Tensor,
    c1: torch.Tensor,
    pure: torch.Tensor,
) -> torch.Tensor:
    """improve_ff_block4: все ветки к глубине ``pure`` (512)."""
    y = (
        _stack_pool(c4, 2, 2)
        + _stack_pool(c3, 4, 4)
        + _stack_pool(c2, 8, 8)
        + _stack_pool(c1, 16, 16)
        + pure
    )
    return F.relu(y, inplace=False)


def _ff3(c3: torch.Tensor, c2: torch.Tensor, c1: torch.Tensor, pure: torch.Tensor) -> torch.Tensor:
    y = _stack_pool(c3, 2, 2) + _stack_pool(c2, 4, 4) + _stack_pool(c1, 8, 8) + pure
    return F.relu(y, inplace=False)


def _ff2(c2: torch.Tensor, c1: torch.Tensor, pure: torch.Tensor) -> torch.Tensor:
    y = _stack_pool(c2, 2, 2) + _stack_pool(c1, 4, 4) + pure
    return F.relu(y, inplace=False)


def _ff1(c1: torch.Tensor, pure: torch.Tensor) -> torch.Tensor:
    return F.relu(_stack_pool(c1, 2, 2) + pure, inplace=False)
