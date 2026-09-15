"""Testes da autenticação OAuth do Earth Engine por variáveis de ambiente."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.data import gee_client


def _clear_gee_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GEE_PROJECT",
        gee_client.OAUTH_CREDENTIALS_PATH_ENV,
        gee_client.OAUTH_CREDENTIALS_JSON_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def test_init_ee_requires_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_gee_env(monkeypatch)
    with pytest.raises(gee_client.GEECredentialsError):
        gee_client.init_ee()


def test_init_ee_requires_oauth_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gee_env(monkeypatch)
    monkeypatch.setenv("GEE_PROJECT", "projeto-teste")
    with pytest.raises(gee_client.GEECredentialsError):
        gee_client.init_ee()


def test_make_export_description_is_deterministic() -> None:
    description = gee_client.make_export_description(
        "guaxupe", "310044", "2023-06-01", "2023-09-30"
    )
    assert description == "guaxupe_310044_20230601_20230930"


def test_materialize_oauth_credentials_from_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_gee_env(monkeypatch)
    credentials_dir = tmp_path / "earthengine"
    monkeypatch.setattr(gee_client, "EE_CREDENTIALS_DIR", credentials_dir)
    monkeypatch.setattr(
        gee_client, "EE_CREDENTIALS_FILE", credentials_dir / "credentials"
    )
    monkeypatch.setenv(
        gee_client.OAUTH_CREDENTIALS_JSON_ENV, '{"type": "authorized_user"}'
    )
    path = gee_client._materialize_oauth_credentials()
    assert path.is_file()
    assert "authorized_user" in path.read_text(encoding="utf-8")


def test_materialize_oauth_credentials_from_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_gee_env(monkeypatch)
    credentials_dir = tmp_path / "earthengine"
    monkeypatch.setattr(gee_client, "EE_CREDENTIALS_DIR", credentials_dir)
    monkeypatch.setattr(
        gee_client, "EE_CREDENTIALS_FILE", credentials_dir / "credentials"
    )
    source = tmp_path / "oauth.json"
    source.write_text('{"refresh_token": "abc"}', encoding="utf-8")
    monkeypatch.setenv(gee_client.OAUTH_CREDENTIALS_PATH_ENV, str(source))
    path = gee_client._materialize_oauth_credentials()
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == '{"refresh_token": "abc"}'


def test_materialize_oauth_credentials_requires_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gee_env(monkeypatch)
    with pytest.raises(gee_client.GEECredentialsError):
        gee_client._materialize_oauth_credentials()


def test_materialize_oauth_credentials_rejects_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_gee_env(monkeypatch)
    monkeypatch.setenv(
        gee_client.OAUTH_CREDENTIALS_PATH_ENV, str(tmp_path / "nope.json")
    )
    with pytest.raises(gee_client.GEECredentialsError):
        gee_client._materialize_oauth_credentials()