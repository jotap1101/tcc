"""Testes de src/data/mask_comparison.py (diagnóstico comparativo de fontes)."""

import json

import numpy as np
import pytest

from src.config import get_config
from src.data.mask_comparison import (
    MaskSet,
    agreement_labels,
    compute_comparison,
    figure_file_name,
    load_masks,
    report_file_name,
    save_figure,
    save_report,
)


def _sample_mask_set() -> MaskSet:
    """Conjunto sintético: 2 fontes com 1 pixel em comum e 1 exclusivo cada."""
    reference = np.zeros((4, 4), dtype=bool)
    reference[0, 0] = True
    reference[0, 1] = True
    other = np.zeros((4, 4), dtype=bool)
    other[0, 1] = True
    other[1, 0] = True
    return MaskSet(
        masks={"mapbiomas": reference, "alphaearth": other},
        crs="EPSG:31983",
        shape=(4, 4),
        pixel_size_m=10.0,
    )


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


def test_artifact_file_names_derive_from_config() -> None:
    """Os nomes dos artefatos devem derivar da região e do ano de referência."""
    config = get_config()
    assert report_file_name().startswith("mask_sources_comparison_")
    assert report_file_name().endswith(".json")
    assert figure_file_name().endswith(".png")
    assert config["aoi"]["region_code"] in report_file_name()


def test_agreement_labels_mapping() -> None:
    """O mapa de concordância deve rotular nenhum/ambos/só referência/só outra."""
    reference = np.zeros((2, 2), dtype=bool)
    other = np.zeros((2, 2), dtype=bool)
    reference[0, 0] = True  # só referência
    other[0, 1] = True  # só outra
    reference[1, 0] = True
    other[1, 0] = True  # ambos
    labels = agreement_labels(reference, other)
    assert labels[0, 0] == 2
    assert labels[0, 1] == 3
    assert labels[1, 0] == 1
    assert labels[1, 1] == 0


def test_compute_comparison_metrics() -> None:
    """As métricas de concordância devem refletir a tabela 2x2 do conjunto sintético."""
    comparison = compute_comparison(_sample_mask_set(), "mapbiomas")
    pair = comparison["pairs"]["mapbiomas_vs_alphaearth"]
    assert pair["confusion_pixels"] == {
        "both_coffee": 1,
        "only_reference": 1,
        "only_other": 1,
        "both_non_coffee": 13,
    }
    metrics = pair["metrics"]
    assert metrics["overall_agreement"] == pytest.approx(14 / 16)
    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["kappa"] == pytest.approx(0.428571428, rel=1e-6)


def test_compute_comparison_areas() -> None:
    """A área de café por fonte deve derivar da contagem de pixels e do tamanho do pixel."""
    comparison = compute_comparison(_sample_mask_set(), "mapbiomas")
    assert comparison["area_km2"]["mapbiomas"] == pytest.approx(2e-4)
    assert comparison["area_km2"]["alphaearth"] == pytest.approx(2e-4)
    assert comparison["coffee_share"]["mapbiomas"] == pytest.approx(2 / 16)


def test_compute_comparison_invalid_reference() -> None:
    """Uma fonte de referência inexistente deve levantar ValueError."""
    with pytest.raises(ValueError):
        compute_comparison(_sample_mask_set(), "fonte_inexistente")


def test_load_masks_reads_on_common_grid(tmp_path) -> None:
    """As máscaras devem ser lidas no grid comum, mesmo com grids divergentes."""
    from rasterio.transform import from_origin

    reference_array = np.zeros((3, 3), dtype=bool)
    reference_array[0, 0] = True
    other_array = np.zeros((2, 2), dtype=bool)
    other_array[0, 0] = True

    reference_path = tmp_path / "ref.tif"
    other_path = tmp_path / "other.tif"
    _write_geotiff(reference_path, reference_array, from_origin(0, 3, 10, 10))
    _write_geotiff(other_path, other_array, from_origin(0, 3, 10, 10))

    mask_set = load_masks({"mapbiomas": reference_path, "alphaearth": other_path})
    assert mask_set.shape == (3, 3)
    assert mask_set.pixel_size_m == pytest.approx(10.0)
    assert mask_set.crs == "EPSG:31983"
    assert mask_set.masks["mapbiomas"][0, 0]
    assert mask_set.masks["alphaearth"][0, 0]
    assert not mask_set.masks["alphaearth"][1, 1]


def test_save_report_idempotent(tmp_path, monkeypatch) -> None:
    """O relatório deve ser persistido uma única vez e reutilizado em reexecuções."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    storage_paths = {"artifacts_metrics_ground_truth": tmp_path / "ground_truth"}

    comparison = compute_comparison(_sample_mask_set(), "mapbiomas")
    first = save_report(comparison, storage_paths)
    second = save_report(comparison, storage_paths)

    assert first == second
    assert first.is_file()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["reference_source"] == "mapbiomas"


def test_save_figure_idempotent(tmp_path, monkeypatch) -> None:
    """A figura deve ser persistida uma única vez e reutilizada em reexecuções."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    storage_paths = {"artifacts_figures": tmp_path / "figures"}

    mask_set = _sample_mask_set()
    first = save_figure(mask_set, "mapbiomas", storage_paths)
    second = save_figure(mask_set, "mapbiomas", storage_paths)

    assert first == second
    assert first.is_file()
    assert first.read_bytes().startswith(b"\x89PNG")


def test_save_figure_requires_two_sources(tmp_path, monkeypatch) -> None:
    """A figura do diagnóstico exige pelo menos duas fontes habilitadas."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    storage_paths = {"artifacts_figures": tmp_path / "figures"}
    single = MaskSet(
        masks={"mapbiomas": np.zeros((4, 4), dtype=bool)},
        crs="EPSG:31983",
        shape=(4, 4),
        pixel_size_m=10.0,
    )
    with pytest.raises(ValueError):
        save_figure(single, "mapbiomas", storage_paths)
