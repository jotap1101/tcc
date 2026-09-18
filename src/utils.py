"""Utilidades de reprodutibilidade e autenticação."""

from __future__ import annotations

import os
import random
from pathlib import Path

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


def authenticate_gee() -> None:
    """Autentica e inicializa o Earth Engine com a conta principal.

    Se a variável de ambiente GEE_CREDENTIALS estiver definida (ex.: Secret no
    Kaggle), escreve a credencial em ~/.config/earthengine/credentials e
    inicializa de forma headless; caso contrário, executa o fluxo interativo
    ee.Authenticate() (padrão do Colab).
    """
    import ee

    credentials_env = os.getenv("GEE_CREDENTIALS")
    if credentials_env:
        config_dir = Path.home() / ".config" / "earthengine"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "credentials").write_text(credentials_env, encoding="utf-8")
        ee.Initialize()
        return
    ee.Authenticate()
    ee.Initialize()
