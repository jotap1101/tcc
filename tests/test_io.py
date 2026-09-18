"""Testes de src/io.py (abstração de plataforma e de armazenamento tcc/)."""

from src import io
from src.config import get_config


def test_detect_platform_returns_known_value() -> None:
    """A detecção de plataforma deve retornar colab, kaggle ou local."""
    assert io.detect_platform() in {"colab", "kaggle", "local"}


def test_ensure_storage_root_idempotent_local(tmp_path, monkeypatch) -> None:
    """No modo local, a raiz tcc/ e todas as subpastas são criadas de forma idempotente."""
    monkeypatch.setattr(io, "detect_platform", lambda: "local")
    monkeypatch.setattr(io, "mount_drive", lambda: tmp_path)

    root = io.ensure_storage_root()
    assert root == tmp_path / "tcc"
    assert root.is_dir()

    root_again = io.ensure_storage_root()
    assert root_again == root

    for sub in get_config()["storage"]["subfolders"].values():
        assert (root / sub).is_dir()


def test_resolve_storage_paths_local(tmp_path, monkeypatch) -> None:
    """Os caminhos resolvidos devem apontar para dentro da raiz tcc/ no Drive."""
    monkeypatch.setattr(io, "detect_platform", lambda: "local")
    monkeypatch.setattr(io, "mount_drive", lambda: tmp_path)

    paths = io.resolve_storage_paths()
    assert paths["data_raw"] == tmp_path / "tcc" / "data" / "raw"
    assert paths["models"] == tmp_path / "tcc" / "models"
