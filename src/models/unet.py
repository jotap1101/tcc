"""U-Net para segmentação semântica binária de lavouras cafeeiras (estágio 09).

Arquitetura codificador-decoder com conexões de skip, batch normalization e
ativações ReLU. O número de canais por nível vem de config.yaml
(`model.unet_channels`); a entrada tem `len(data.bands)` canais (4 bandas do
Sentinel-2) e a saída é 1 canal de logit binário (fundo vs café).
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import get_config


class DoubleConv(nn.Module):
    """Bloco duplo de convolução 3x3 com BatchNorm e ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net binária com canais por nível configuráveis.

    O codificador aplica `DoubleConv` e reduz pela metade a resolução a cada
    nível; o bottleneck dobra o último canal; o decodificador restaura a
    resolução com `ConvTranspose2d`, concatenando os skips correspondentes.
    """

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("model.unet_channels não pode ser vazio.")
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for channel_count in channels:
            self.encoders.append(DoubleConv(prev, channel_count))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = channel_count
        self.bottleneck = DoubleConv(prev, prev * 2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_input = prev * 2  # canais do bottleneck (nível mais profundo)
        for channel_count in reversed(channels):
            self.upconvs.append(
                nn.ConvTranspose2d(decoder_input, channel_count, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv(channel_count * 2, channel_count))
            decoder_input = channel_count
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder, pool in zip(self.encoders, self.pools, strict=True):
            x = encoder(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips), strict=True):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        return self.out_conv(x)


def build_unet() -> nn.Module:
    """Constrói o U-Net a partir de src/config.yaml (uso nos estágios 09/10)."""
    config = get_config()
    return UNet(
        in_channels=len(config["data"]["bands"]),
        channels=[int(channel) for channel in config["model"]["unet_channels"]],
        out_channels=1,
    )
