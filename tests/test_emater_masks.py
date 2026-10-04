"""Testes de src/data/emater_masks.py (máscara vetorial da Emater)."""

import json

import geopandas as gpd
import pytest
from shapely.geometry import box

from src.data.emater_masks import (
    ensure_emater_mask,
    load_emater_features,
    save_emater_mask_preview,
    verify_emater_mask,
)
from src.data.mask_utils import source_mask_path

CRS = "EPSG:31983"
AOI = box(200000, 200000, 201000, 201000)  # quadrado de 1 km x 1 km em UTM 23S
COFFEE = box(200100, 200100, 200900, 200900)  # lavoura quadrada de 800 m x 800 m


def _write_municipio_geojson(path, geometry_utm, municipio: str) -> None:
    """Persiste um GeoJSON de um município com uma lavoura (EPSG:4326)."""
    gdf = gpd.GeoDataFrame({"variedade": ["Café"]}, geometry=[geometry_utm], crs=CRS)
    gdf = gdf.to_crs("EPSG:4326")
    feature = {
        "type": "Feature",
        "properties": {"variedade": "Café", "area": 64.0},
        "geometry": gdf.geometry.iloc[0].__geo_interface__,
    }
    payload = {"type": "FeatureCollection", "features": [feature]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _storage_paths(tmp_path) -> dict:
    """Caminhos de armazenamento mínimos para os testes da Emater."""
    return {
        "data_interim": tmp_path / "data" / "interim",
        "artifacts_figures": tmp_path / "artifacts" / "figures",
    }


def test_load_emater_features_concatena_municipios(tmp_path) -> None:
    """As feições devem ser carregadas de todos os municípios, com o nome do município."""
    _write_municipio_geojson(tmp_path / "GetGlebasGeoJson - Um.json", COFFEE, "Um")
    _write_municipio_geojson(tmp_path / "GetGlebasGeoJson - Dois.json", COFFEE, "Dois")
    features = load_emater_features(tmp_path)
    assert len(features) == 2
    assert set(features["municipio"]) == {"Um", "Dois"}
    assert features.crs == "EPSG:4326"


def test_ensure_emater_mask_rasterizes(tmp_path, monkeypatch) -> None:
    """A máscara deve ser rasterizada no grid de 10 m do AOI, em EPSG:31983."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    data_dir = tmp_path / "dados"
    _write_municipio_geojson(data_dir / "GetGlebasGeoJson - Um.json", COFFEE, "Um")

    mask_path = ensure_emater_mask("emater", AOI, paths, data_dir)

    assert mask_path == source_mask_path("emater", paths)
    assert mask_path.is_file()
    assert mask_path.name == "mask_emater_310044_2018.tif"


def test_ensure_emater_mask_idempotent(tmp_path, monkeypatch) -> None:
    """Execuções repetidas devem reutilizar a máscara já gerada."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    data_dir = tmp_path / "dados"
    _write_municipio_geojson(data_dir / "GetGlebasGeoJson - Um.json", COFFEE, "Um")

    first = ensure_emater_mask("emater", AOI, paths, data_dir)
    second = ensure_emater_mask("emater", AOI, paths, data_dir)
    assert first == second
    assert first.read_bytes() == second.read_bytes()


def test_verify_emater_mask_reports_grid_and_area(tmp_path, monkeypatch) -> None:
    """A verificação deve reportar grid, CRS, pixels e área de café."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    data_dir = tmp_path / "dados"
    _write_municipio_geojson(data_dir / "GetGlebasGeoJson - Um.json", COFFEE, "Um")

    ensure_emater_mask("emater", AOI, paths, data_dir)
    stats = verify_emater_mask("emater", paths)

    assert stats["shape"] == [100, 100]
    assert stats["crs"] == "EPSG:31983"
    # Lavoura de 800 m x 800 m no grid de 10 m: ~80 x 80 pixels.
    assert 6000 < stats["coffee_pixels"] <= 6400
    assert stats["area_km2"] == pytest.approx(stats["coffee_pixels"] * 1e-4)


def test_ensure_emater_mask_requires_data(tmp_path, monkeypatch) -> None:
    """Sem GeoJSONs a integração deve falhar com FileNotFoundError."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    with pytest.raises(FileNotFoundError):
        ensure_emater_mask("emater", AOI, paths, tmp_path / "vazio")


def test_save_emater_mask_preview_png(tmp_path, monkeypatch) -> None:
    """A miniatura da máscara deve ser persistida em PNG."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    data_dir = tmp_path / "dados"
    _write_municipio_geojson(data_dir / "GetGlebasGeoJson - Um.json", COFFEE, "Um")
    ensure_emater_mask("emater", AOI, paths, data_dir)

    figure = save_emater_mask_preview("emater", paths)
    assert figure.is_file()
    assert figure.read_bytes().startswith(b"\x89PNG")
