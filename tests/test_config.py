"""Testes de src/config.py (carregamento e validação da configuração única)."""

from src.config import get_config


def test_config_has_storage_block() -> None:
    """A configuração deve expor o bloco de armazenamento com a raiz canônica."""
    config = get_config()
    assert config["storage"]["drive_root"] == "tcc"
    assert "subfolders" in config["storage"]


def test_config_has_ground_truth_sources() -> None:
    """A configuração deve listar as fontes de ground truth (uma subpasta por fonte)."""
    sources = set(get_config()["ground_truth"]["sources"])
    assert {"mapbiomas", "alphaearth"} <= sources
