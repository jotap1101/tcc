"""Inicialização comum do runtime para todos os notebooks (prólogo padronizado).

Centraliza o bloco repetido entre os notebooks 00-08: detecção de plataforma,
resolução do armazenamento canônico (MyDrive/tcc/) e carregamento da
configuração, expondo um único ``initialize()`` que retorna um ``RuntimeContext``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import io
from src.config import get_config


@dataclass(frozen=True)
class RuntimeContext:
    """Contexto de runtime resolvido para a execução de um notebook."""

    platform: str
    workspace: Path
    storage_paths: dict[str, Path]
    config: dict[str, Any]

    @property
    def seed(self) -> int:
        """Semente global de reprodutibilidade, definida em config.yaml."""
        return int(self.config["reproducibility"]["seed"])


def initialize(workspace: Path | None = None) -> RuntimeContext:
    """Resolve plataforma, armazenamento canônico e configuração (idempotente)."""
    platform = io.detect_platform()
    storage_paths = io.resolve_storage_paths()
    config = get_config()
    return RuntimeContext(
        platform=platform,
        workspace=Path.cwd() if workspace is None else workspace,
        storage_paths=storage_paths,
        config=config,
    )
