"""Testes de src/models/unet.py (arquitetura U-Net binária)."""

from __future__ import annotations

import pytest
import torch

from src.models.unet import UNet, build_unet


def test_build_unet_from_config() -> None:
    """build_unet deve construir um modelo a partir de config.yaml."""
    model = build_unet()
    assert isinstance(model, UNet)
    assert len(model.encoders) == 4  # unet_channels = [32, 64, 128, 256]


def test_unet_forward_shape() -> None:
    """A saída deve preservar a resolução espacial com 1 canal de logit."""
    model = UNet(in_channels=4, channels=[8, 16], out_channels=1)
    image = torch.randn(2, 4, 32, 32)
    logits = model(image)
    assert logits.shape == (2, 1, 32, 32)


def test_unet_in_channels_from_bands() -> None:
    """O número de canais de entrada deve ser o das bandas do Sentinel-2."""
    model = build_unet()
    first_weight = next(model.parameters())
    assert tuple(first_weight.shape)[1] == 4  # canais de entrada = bandas B2/B3/B4/B8


def test_unet_requires_non_empty_channels() -> None:
    """Canais vazios devem falhar com mensagem clara."""
    with pytest.raises(ValueError, match="unet_channels"):
        UNet(in_channels=4, channels=[], out_channels=1)


def test_unet_output_logits_binary() -> None:
    """Com 1 canal de saída, os logits são escalares por pixel (segmentação binária)."""
    model = UNet(in_channels=4, channels=[4, 8], out_channels=1)
    image = torch.randn(1, 4, 16, 16)
    logits = model(image)
    assert logits.shape[1] == 1
    assert torch.is_tensor(logits)
