"""Inicialização do Google Earth Engine a partir de variáveis de ambiente.

Nenhuma credencial é lida de arquivos versionados, impressa em logs ou
persistida: os valores vêm exclusivamente de variáveis de ambiente, conforme
``.env.example``. Este módulo é reutilizado pelos notebooks das fases 0 e 1.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

# Variáveis obrigatórias para a autenticação por conta de serviço.
REQUIRED_ENV_VARS: tuple[str, ...] = (
    "GEE_SERVICE_ACCOUNT_EMAIL",
    "GEE_PROJECT",
)

# Formatos aceitos para a chave privada (um dos dois é obrigatório).
KEY_PATH_ENV = "GEE_SERVICE_ACCOUNT_KEY_PATH"
KEY_JSON_ENV = "GEE_SERVICE_ACCOUNT_KEY_JSON"


class GEECredentialsError(RuntimeError):
    """Erro de configuração das credenciais do Earth Engine."""


def _write_key_from_json(key_json: str) -> Path:
    """Materializa a chave JSON embutida em um arquivo temporário."""
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="gee-key-",
        delete=False,
        encoding="utf-8",
    )
    with handle:
        handle.write(key_json)
    return Path(handle.name)


def _resolve_key_path() -> Path:
    """Resolve o caminho da chave privada a partir do ambiente."""
    key_path = os.environ.get(KEY_PATH_ENV)
    if key_path:
        path = Path(key_path).expanduser()
        if not path.is_file():
            raise GEECredentialsError(
                f"{KEY_PATH_ENV} aponta para arquivo inexistente: {path}"
            )
        return path

    key_json = os.environ.get(KEY_JSON_ENV)
    if key_json:
        return _write_key_from_json(key_json)

    raise GEECredentialsError(
        f"Defina {KEY_PATH_ENV} ou {KEY_JSON_ENV} com a chave da conta de serviço."
    )


def init_ee() -> Any:
    """Inicializa o Earth Engine com a conta de serviço e retorna o módulo ``ee``."""
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise GEECredentialsError(
            "Variáveis de ambiente ausentes: " + ", ".join(missing)
        )

    import ee  # importação tardia: não exige a dependência no CI

    email = os.environ["GEE_SERVICE_ACCOUNT_EMAIL"]
    project = os.environ["GEE_PROJECT"]
    key_path = _resolve_key_path()

    credentials = ee.ServiceAccountCredentials(email, str(key_path))
    ee.Initialize(credentials, project=project)
    return ee
