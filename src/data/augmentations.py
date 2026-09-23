"""Aumentações geométricas determinísticas para o treino (estágios 09/10).

Aplicadas ao par (imagem, máscara) durante o treino com o mesmo protocolo nos
dois modelos. As decisões aleatórias usam um ``torch.Generator`` próprio,
reseedado por época no trainer (``set_seed``), o que torna o treino
bit-reproduzível quando ``training.num_workers = 0`` (as aumentações dependem do
gerador local do processo). Transformações: rotação de 90°, flip horizontal e
flip vertical, com probabilidades de config.yaml (`augmentation.*`).
"""

from __future__ import annotations

from typing import Any

import torch

FLIP_DIMS = (1, 2)  # dimensões espaciais (H, W) de tensores (C, H, W)


class SegmentAugmentation:
    """Transformação geométrica determinística aplicada a imagem e máscara juntas."""

    def __init__(self, config: dict[str, Any]) -> None:
        aug = config.get("augmentation") or {}
        self.enabled = bool(aug.get("enabled", False))
        self.hflip_prob = float(aug.get("hflip_prob", 0.5))
        self.vflip_prob = float(aug.get("vflip_prob", 0.5))
        self.rot90_prob = float(aug.get("rot90_prob", 0.25))
        self.generator = torch.Generator()
        self.generator.manual_seed(0)

    def set_seed(self, seed: int) -> None:
        """Fixa a semente do gerador (chamada pelo trainer a cada época)."""
        self.generator.manual_seed(seed)

    def _draw(self, probability: float) -> bool:
        """Sorteia uma decisão binária com o gerador próprio do transform."""
        return torch.rand(1, generator=self.generator).item() < probability

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aplica rotação e flips à imagem e à máscara com as mesmas decisões."""
        if not self.enabled:
            return image, mask
        if self._draw(self.rot90_prob):
            image = torch.rot90(image, k=1, dims=FLIP_DIMS)
            mask = torch.rot90(mask, k=1, dims=FLIP_DIMS)
        if self._draw(self.hflip_prob):
            image = torch.flip(image, dims=[FLIP_DIMS[-1]])
            mask = torch.flip(mask, dims=[FLIP_DIMS[-1]])
        if self._draw(self.vflip_prob):
            image = torch.flip(image, dims=[FLIP_DIMS[0]])
            mask = torch.flip(mask, dims=[FLIP_DIMS[0]])
        return image, mask
