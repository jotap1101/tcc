"""Testes de src/data/mask_utils.py (máscaras de referência por fonte de ground truth)."""

import pytest

from src.config import get_config
from src.data import mask_utils


def test_reference_year_derives_from_data_dates() -> None:
    """O ano de referência deve derivar das datas de início do mosaico."""
    config = get_config()
    expected = int(config["data"]["dates"]["start"][:4])
    assert mask_utils.reference_year() == expected


def test_mask_file_name_stable() -> None:
    """O nome da máscara deve derivar da fonte, região e ano."""
    assert mask_utils.mask_file_name("mapbiomas", "310044", 2023) == "mask_mapbiomas_310044_2023"


def test_ground_truth_sources_have_required_config() -> None:
    """Cada fonte habilitada deve expor coleção e parâmetro de classe/limiar."""
    sources = get_config()["ground_truth"]["sources"]

    mapbiomas = sources["mapbiomas"]
    assert mapbiomas["collection_id"]
    assert mapbiomas["coffee_class"] == 46

    alphaearth = sources["alphaearth"]
    assert alphaearth["collection_id"]
    assert 0 < alphaearth["probability_threshold"] <= 1


def test_mask_builders_registered_per_source() -> None:
    """Deve existir um construtor dedicado para cada fonte habilitada."""
    sources = get_config()["ground_truth"]["sources"]
    for name, cfg in sources.items():
        if cfg["enabled"]:
            assert name in mask_utils.MASK_BUILDERS


def test_build_mask_raises_for_unknown_source() -> None:
    """Uma fonte desconhecida deve levantar ValueError."""
    with pytest.raises(ValueError):
        mask_utils.build_mask("fonte_inexistente", None)


def test_ensure_source_mask_disabled_returns_none(monkeypatch) -> None:
    """Uma fonte desabilitada não deve exportar nada e retornar None."""
    config = get_config()
    disabled = {"enabled": False}
    config["ground_truth"]["sources"]["s2dr"] = disabled
    monkeypatch.setattr(mask_utils, "get_config", lambda: config)
    assert mask_utils.ensure_source_mask("s2dr", None, {}) is None
