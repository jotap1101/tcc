"""Carregamento centralizado de configuração e utilidades de reprodutibilidade.

A fonte única de verdade é ``src/config.yaml``. Este módulo resolve todos os
caminhos a partir da raiz do repositório (ou da variável de ambiente
``TCC_ROOT``), fixa sementes de execução e registra o ambiente por run, sem
nunca expor credenciais.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform as _platform
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

# Raiz do repositório: permite sobrescrever via ambiente para Kaggle/Colab.
PROJECT_ROOT: Path = Path(
    os.environ.get("TCC_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
CONFIG_PATH: Path = PROJECT_ROOT / "src" / "config.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    """Lê o arquivo YAML e valida que o conteúdo é um mapeamento."""
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data: Any = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuração inválida (esperado mapeamento): {path}")
    return data


@dataclass(frozen=True)
class Paths:
    """Caminhos absolutos derivados da configuração."""

    root: Path
    data: Path
    raw: Path
    interim: Path
    processed: Path
    external: Path
    models: Path
    artifacts: Path
    manifest: Path

    def ensure(self) -> None:
        """Cria a árvore de diretórios de trabalho, se ainda não existir."""
        for directory in (
            self.data,
            self.raw,
            self.interim,
            self.processed,
            self.external,
            self.models,
            self.artifacts,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.manifest.parent.mkdir(parents=True, exist_ok=True)


class Config:
    """Acesso tipado e somente-leitura à configuração do projeto."""

    def __init__(self, raw: dict[str, Any], root: Path = PROJECT_ROOT) -> None:
        self._raw = raw
        self.root = root
        self.paths = self._build_paths(raw)

    def _build_paths(self, raw: dict[str, Any]) -> Paths:
        declared = raw.get("paths", {})
        if not isinstance(declared, dict):
            declared = {}

        def resolve(key: str, default: str) -> Path:
            relative = str(declared.get(key, default))
            return (self.root / relative).resolve()

        return Paths(
            root=self.root,
            data=resolve("data", "data"),
            raw=resolve("raw", "data/raw"),
            interim=resolve("interim", "data/interim"),
            processed=resolve("processed", "data/processed"),
            external=resolve("external", "data/external"),
            models=resolve("models", "models"),
            artifacts=resolve("artifacts", "artifacts"),
            manifest=resolve("manifest", "data/processed/manifest.parquet"),
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Recupera um valor por caminho pontuado (ex.: ``gee.bands``)."""
        node: Any = self._raw
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def raw(self) -> dict[str, Any]:
        """Cópia defensiva do mapeamento bruto da configuração."""
        return cast("dict[str, Any]", json.loads(json.dumps(self._raw)))

    @property
    def seed(self) -> int:
        return int(self.get("project.seed", 42))

    @property
    def bands(self) -> list[str]:
        return list(self.get("gee.bands", []))

    @property
    def patch_size(self) -> int:
        return int(self.get("dataset.patch_size", 512))

    @property
    def fold_count(self) -> int:
        return int(self.get("dataset.fold_count", 5))

    @property
    def coffee_min_ratio(self) -> float:
        return float(self.get("dataset.coffee_min_ratio", 0.01))

    @property
    def model_variants(self) -> dict[str, Any]:
        return dict(self.get("models", {}))

    @property
    def config_hash(self) -> str:
        """Hash estável da configuração, usado para rastrear cada run."""
        payload = json.dumps(self._raw, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


# Instância global carregada na importação do módulo.
CONFIG = Config(_read_yaml(CONFIG_PATH))


def seed_everything(seed: int | None = None) -> int:
    """Fixa as sementes de python/numpy/torch/cuda e retorna a semente usada."""
    resolved = CONFIG.seed if seed is None else int(seed)
    random.seed(resolved)
    os.environ["PYTHONHASHSEED"] = str(resolved)

    # Importações opcionais: não exigem as dependências pesadas no CI.
    with contextlib.suppress(ImportError):
        import numpy as np

        np.random.seed(resolved)

    with contextlib.suppress(ImportError):
        import torch

        torch.manual_seed(resolved)
        torch.cuda.manual_seed_all(resolved)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return resolved


def _package_versions() -> dict[str, str]:
    """Coleta versões das dependências-chave instaladas (sem segredos)."""
    from importlib.metadata import PackageNotFoundError, version

    names = [
        "numpy",
        "pandas",
        "pyarrow",
        "pyyaml",
        "scikit-learn",
        "rasterio",
        "geopandas",
        "earthengine-api",
        "torch",
        "torchvision",
        "transformers",
        "huggingface-hub",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_sha() -> str | None:
    """SHA curto do commit atual, quando disponível."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def log_environment() -> dict[str, Any]:
    """Monta um retrato do ambiente do run (versões, semente, hash, commit)."""
    return {
        "python": sys.version.split()[0],
        "platform": _platform.platform(),
        "machine": _platform.machine(),
        "git_sha": _git_sha(),
        "config_hash": CONFIG.config_hash,
        "seed": CONFIG.seed,
        "packages": _package_versions(),
    }


def save_environment_log(destination: Path | None = None) -> Path:
    """Persiste o retrato do ambiente em JSON dentro de ``artifacts/``."""
    target = destination or (CONFIG.paths.artifacts / "environment.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(log_environment(), indent=2, ensure_ascii=False)
    target.write_text(payload, encoding="utf-8")
    return target
