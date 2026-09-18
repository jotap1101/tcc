"""Utilidades de reprodutibilidade: sementes fixas e flags determinísticas."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Fixa as sementes de python, numpy e torch (cpu e cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic_flags() -> None:
    """Habilita flags determinísticas do PyTorch (quando suportadas)."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)