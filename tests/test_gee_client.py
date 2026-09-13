"""Testes da autenticação do Earth Engine por variáveis de ambiente."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import gee_client


def _clear_gee_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GEE_SERVICE_ACCOUNT_EMAIL",
        "GEE_PROJECT",
        gee_client.KEY_PATH_ENV,
        gee_client.KEY_JSON_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def test_init_ee_requires_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_gee_env(monkeypatch)
    with pytest.raises(gee_client.GEECredentialsError):
        gee_client.init_ee()


def test_resolve_key_path_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_gee_env(monkeypatch)
    with pytest.raises(gee_client.GEECredentialsError):
        gee_client._resolve_key_path()


def test_resolve_key_path_rejects_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_gee_env(monkeypatch)
    monkeypatch.setenv(gee_client.KEY_PATH_ENV, str(tmp_path / "nope.json"))
    with pytest.raises(gee_client.GEECredentialsError):
        gee_client._resolve_key_path()


def test_resolve_key_path_materializes_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_gee_env(monkeypatch)
    monkeypatch.setenv(gee_client.KEY_JSON_ENV, '{"type": "service_account"}')
    path = gee_client._resolve_key_path()
    assert path.is_file()
    assert "service_account" in path.read_text(encoding="utf-8")
    path.unlink()


def test_resolve_key_path_prefers_existing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_gee_env(monkeypatch)
    key = tmp_path / "key.json"
    key.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(gee_client.KEY_PATH_ENV, str(key))
    assert gee_client._resolve_key_path() == key
