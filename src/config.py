"""Configuração única do projeto, carregada de src/config.yaml (fonte de verdade)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

REQUIRED_TOP_LEVEL = ("storage",)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Carrega e valida o YAML de configuração; lança erro se estiver incompleto."""
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Configuração inválida: {path} não contém um mapeamento.")
    missing = [key for key in REQUIRED_TOP_LEVEL if key not in config]
    if missing:
        raise ValueError(f"Configuração incompleta; faltam chaves: {missing}")
    return config


def get_config() -> dict[str, Any]:
    """Carrega a configuração (sem cache, para refletir atualizações do arquivo)."""
    return load_config()
