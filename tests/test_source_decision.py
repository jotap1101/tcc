"""Testes de src/data/source_decision.py (registro da decisão manual da fonte)."""

import json

import pytest

from src.config import get_config
from src.data import source_decision
from src.data.source_decision import (
    chosen_source,
    comparison_pair_key,
    load_comparison_report,
    save_source_decision_report,
    source_decision_report_file_name,
)


def _comparison_payload() -> dict:
    """Payload sintético do relatório comparativo do estágio 03."""
    return {
        "sources": ["mapbiomas", "alphaearth", "emater"],
        "years": {"mapbiomas": 2025, "alphaearth": 2024, "emater": 2018},
        "area_km2": {"mapbiomas": 455.0, "alphaearth": 856.0, "emater": 604.0},
        "coffee_share": {"mapbiomas": 0.08, "alphaearth": 0.15, "emater": 0.10},
        "pairs": {
            "mapbiomas_vs_alphaearth": {"metrics": {"overall_agreement": 0.9, "iou": 0.5}},
            "mapbiomas_vs_emater": {"metrics": {"overall_agreement": 0.7, "iou": 0.3}},
            "alphaearth_vs_emater": {"metrics": {"overall_agreement": 0.6, "iou": 0.2}},
        },
    }


def test_artifact_file_names_derive_from_config() -> None:
    """O nome do relatório deve derivar da região e ser independente de ano único."""
    config = get_config()
    assert source_decision_report_file_name().endswith(".json")
    assert config["aoi"]["region_code"] in source_decision_report_file_name()


def test_chosen_source_reads_config() -> None:
    """A fonte escolhida deve ser a registrada em ground_truth.chosen_source."""
    chosen = chosen_source()
    assert chosen == get_config()["ground_truth"]["chosen_source"]


def test_chosen_source_requires_definition(monkeypatch) -> None:
    """Sem chosen_source o registro deve falhar com mensagem clara."""
    config = get_config()
    config["ground_truth"].pop("chosen_source", None)
    monkeypatch.setattr(source_decision, "get_config", lambda: config)
    with pytest.raises(ValueError, match="chosen_source"):
        chosen_source()


def test_chosen_source_rejects_disabled_source(monkeypatch) -> None:
    """Uma fonte desabilitada não pode ser escolhida para treino."""
    config = get_config()
    config["ground_truth"]["sources"]["emater"]["enabled"] = False
    config["ground_truth"]["chosen_source"] = "emater"
    monkeypatch.setattr(source_decision, "get_config", lambda: config)
    with pytest.raises(ValueError, match="desabilitada"):
        chosen_source()


def test_comparison_pair_key_selects_chosen_source_pair() -> None:
    """A chave do par deve ser a que envolve a fonte escolhida."""
    assert comparison_pair_key(_comparison_payload(), "alphaearth") == "mapbiomas_vs_alphaearth"


def test_load_comparison_report_requires_stage03(tmp_path) -> None:
    """Sem o relatório do estágio 03, a carga deve falhar."""
    storage_paths = {"artifacts_metrics_ground_truth": tmp_path / "ground_truth"}
    with pytest.raises(FileNotFoundError, match="estágio 03"):
        load_comparison_report(storage_paths)


def test_save_source_decision_report_idempotent(tmp_path, monkeypatch) -> None:
    """O relatório da decisão deve ser persistido uma única vez e reutilizado."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    storage_paths = {"artifacts_metrics_ground_truth": tmp_path / "ground_truth"}
    comparison = _comparison_payload()

    first = save_source_decision_report("alphaearth", comparison, storage_paths)
    second = save_source_decision_report("alphaearth", comparison, storage_paths)

    assert first == second
    assert first.is_file()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["chosen_source"] == "alphaearth"
    assert payload["decision"] == "manual"
    assert payload["year"] == 2024
    assert payload["sources_year"] == comparison["years"]
    assert "decision_basis" in payload
    assert "mapbiomas_vs_alphaearth" in payload["decision_basis"]["pairs"]
