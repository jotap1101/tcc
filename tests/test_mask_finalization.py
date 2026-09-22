"""Testes de src/data/mask_finalization.py (finalização da máscara de café)."""

import json

import numpy as np
import pytest

from src.config import get_config
from src.data import mask_finalization
from src.data.mask_finalization import (
    chosen_source,
    ensure_final_mask,
    final_mask_file_name,
    final_mask_path,
    final_mask_preview_file_name,
    finalization_report_file_name,
    load_comparison_report,
    save_final_mask_preview,
    save_finalization_report,
    verify_final_mask,
)
from src.data.mask_utils import source_mask_path


def _write_geotiff(path, array, transform, crs="EPSG:31983") -> None:
    """Escreve um GeoTIFF uint8 de uma banda para uso nos testes."""
    import rasterio

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array.astype("uint8"), 1)


def _storage_paths(tmp_path) -> dict:
    """Caminhos de armazenamento mínimos para os testes do estágio 04."""
    return {
        "data_interim": tmp_path / "data" / "interim",
        "data_processed_ground_truth": tmp_path / "data" / "processed" / "ground_truth",
        "artifacts_metrics_ground_truth": tmp_path / "artifacts" / "metrics" / "ground_truth",
        "artifacts_figures": tmp_path / "artifacts" / "figures",
    }


def test_artifact_file_names_derive_from_config() -> None:
    """Os nomes dos artefatos devem derivar da região e do ano de referência."""
    config = get_config()
    assert final_mask_file_name().startswith("mask_final_")
    assert finalization_report_file_name().endswith(".json")
    assert final_mask_preview_file_name().endswith(".png")
    assert config["aoi"]["region_code"] in final_mask_file_name()


def test_chosen_source_reads_config() -> None:
    """A fonte escolhida deve ser a registrada em ground_truth.chosen_source."""
    config = get_config()
    chosen = chosen_source()
    assert chosen == config["ground_truth"]["chosen_source"]
    assert config["ground_truth"]["sources"][chosen]["enabled"]


def test_chosen_source_requires_definition(monkeypatch) -> None:
    """Sem chosen_source a finalização deve falhar com mensagem clara."""
    config = get_config()
    config["ground_truth"].pop("chosen_source", None)
    monkeypatch.setattr(mask_finalization, "get_config", lambda: config)
    with pytest.raises(ValueError, match="chosen_source"):
        chosen_source()


def test_chosen_source_rejects_disabled_source(monkeypatch) -> None:
    """Uma fonte desabilitada não pode ser a escolhida."""
    config = get_config()
    config["ground_truth"]["chosen_source"] = "s2dr"
    monkeypatch.setattr(mask_finalization, "get_config", lambda: config)
    with pytest.raises(ValueError, match="inválida ou desabilitada"):
        chosen_source()


def test_final_mask_path_points_to_processed_ground_truth(tmp_path) -> None:
    """A máscara final deve residir em data/processed/ground_truth/."""
    paths = _storage_paths(tmp_path)
    target = final_mask_path(paths)
    assert target.parent == paths["data_processed_ground_truth"]
    assert target.suffix == ".tif"


def test_load_comparison_report_missing(tmp_path, monkeypatch) -> None:
    """Sem o relatório do estágio 03, a carga deve falhar com FileNotFoundError."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_comparison_report(paths)


def test_ensure_final_mask_idempotent(tmp_path, monkeypatch) -> None:
    """A cópia final deve ser criada uma vez e reutilizada em reexecuções."""
    from rasterio.transform import from_origin

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)

    source = np.zeros((4, 4), dtype=np.uint8)
    source[0, 0] = 1
    source_path = source_mask_path("alphaearth", paths)
    source_path.parent.mkdir(parents=True)
    _write_geotiff(source_path, source, from_origin(0, 3, 10, 10))

    first = ensure_final_mask("alphaearth", paths)
    second = ensure_final_mask("alphaearth", paths)

    assert first == second
    assert first.is_file()
    assert first.read_bytes() == source_path.read_bytes()


def test_ensure_final_mask_requires_source_mask(tmp_path, monkeypatch) -> None:
    """Sem a máscara da fonte (estágio 02), a finalização deve falhar."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="estágio 02"):
        ensure_final_mask("alphaearth", paths)


def test_verify_final_mask_matches_source(tmp_path, monkeypatch) -> None:
    """A verificação deve confirmar grid, CRS e valores da fonte."""
    from rasterio.transform import from_origin

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)

    source = np.zeros((4, 4), dtype=np.uint8)
    source[0, 0] = 1
    source_path = source_mask_path("alphaearth", paths)
    source_path.parent.mkdir(parents=True)
    _write_geotiff(source_path, source, from_origin(0, 3, 10, 10))
    ensure_final_mask("alphaearth", paths)

    stats = verify_final_mask("alphaearth", paths)
    assert stats["shape"] == [4, 4]
    assert stats["crs"] == "EPSG:31983"
    assert stats["coffee_pixels"] == 1
    assert stats["area_km2"] == pytest.approx(1e-4)


def test_verify_final_mask_detects_divergence(tmp_path, monkeypatch) -> None:
    """Uma máscara final divergente da fonte deve falhar na verificação."""
    from rasterio.transform import from_origin

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)

    source = np.zeros((4, 4), dtype=np.uint8)
    source[0, 0] = 1
    source_path = source_mask_path("alphaearth", paths)
    source_path.parent.mkdir(parents=True)
    _write_geotiff(source_path, source, from_origin(0, 3, 10, 10))
    ensure_final_mask("alphaearth", paths)

    divergent = np.zeros((4, 4), dtype=np.uint8)
    divergent[3, 3] = 1
    _write_geotiff(final_mask_path(paths), divergent, from_origin(0, 3, 10, 10))

    with pytest.raises(ValueError, match="divergentes"):
        verify_final_mask("alphaearth", paths)


def test_save_finalization_report_idempotent(tmp_path, monkeypatch) -> None:
    """O relatório de finalização deve ser persistido uma única vez."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)

    comparison = {
        "sources": ["mapbiomas", "alphaearth"],
        "area_km2": {"mapbiomas": 1.0, "alphaearth": 2.0},
        "coffee_share": {"mapbiomas": 0.1, "alphaearth": 0.2},
        "pairs": {
            "mapbiomas_vs_alphaearth": {
                "confusion_pixels": {
                    "both_coffee": 1,
                    "only_reference": 1,
                    "only_other": 1,
                    "both_non_coffee": 13,
                },
                "metrics": {"overall_agreement": 0.8},
            }
        },
    }
    final_path = final_mask_path(paths)

    first = save_finalization_report("alphaearth", comparison, final_path, paths)
    second = save_finalization_report("alphaearth", comparison, final_path, paths)

    assert first == second
    assert first.is_file()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["chosen_source"] == "alphaearth"
    assert payload["final_mask_path"] == str(final_path)


def test_save_final_mask_preview_idempotent(tmp_path, monkeypatch) -> None:
    """A figura da máscara final deve ser persistida uma única vez."""
    from rasterio.transform import from_origin

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)

    source = np.zeros((4, 4), dtype=np.uint8)
    source[0, 0] = 1
    source_path = source_mask_path("alphaearth", paths)
    source_path.parent.mkdir(parents=True)
    _write_geotiff(source_path, source, from_origin(0, 3, 10, 10))
    ensure_final_mask("alphaearth", paths)

    first = save_final_mask_preview(paths)
    second = save_final_mask_preview(paths)

    assert first == second
    assert first.is_file()
    assert first.read_bytes().startswith(b"\x89PNG")
