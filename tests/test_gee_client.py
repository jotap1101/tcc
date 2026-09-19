"""Testes de src/data/gee_client.py (aquisição Sentinel-2 via Earth Engine)."""

from pathlib import Path

import pytest

from src import io
from src.config import get_config
from src.data import gee_client


def test_mosaic_file_name() -> None:
    """O nome do mosaico deve derivar da região e das datas da configuração."""
    config = get_config()
    name = gee_client.mosaic_file_name(
        config["aoi"]["region_code"],
        config["data"]["dates"]["start"],
        config["data"]["dates"]["end"],
    )
    assert name == "sentinel2_310044_2023-01-01_2023-12-31"


def test_load_aoi_gdf_matches_guaxupe() -> None:
    """A malha IBGE versionada deve conter exatamente a RGI 310044 (Guaxupé)."""
    pytest.importorskip("geopandas")
    config = get_config()
    mesh_path = Path(config["aoi"]["mesh_path"])
    aoi = gee_client.load_aoi_gdf(mesh_path, config["aoi"]["region_code"])
    assert len(aoi) == 1
    assert aoi.iloc[0]["CD_RGI"] == "310044"
    assert aoi.iloc[0]["NM_RGI"] == "Guaxupé"


def test_load_aoi_gdf_rejects_unknown_region() -> None:
    """Uma região inexistente na malha deve levantar ValueError."""
    pytest.importorskip("geopandas")
    config = get_config()
    mesh_path = Path(config["aoi"]["mesh_path"])
    with pytest.raises(ValueError):
        gee_client.load_aoi_gdf(mesh_path, "999999")


def test_drive_relative_path(tmp_path, monkeypatch) -> None:
    """O caminho relativo ao Drive deve ignorar a raiz MyDrive montada."""
    monkeypatch.setattr(io, "mount_drive", lambda: tmp_path / "MyDrive")
    storage = tmp_path / "MyDrive" / "tcc" / "data" / "raw" / "sentinel2"
    assert gee_client.drive_relative_path(storage) == "tcc/data/raw/sentinel2"
