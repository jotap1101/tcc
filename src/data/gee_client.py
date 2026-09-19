"""Cliente do Google Earth Engine para aquisição da imagem Sentinel-2 do AOI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.config import get_config

S2_CLOUD_BITMASK = (1 << 10) | (1 << 11)


def load_aoi_gdf(mesh_path: str | Path, region_code: str) -> Any:
    """Carrega a malha IBGE e filtra a Região Geográfica Imediata do AOI.

    A malha original (SIRGAS 2000 geográfico) é reprojetada para EPSG:4326,
    o formato esperado por geometrias do Earth Engine.
    """
    import geopandas as gpd

    mesh = gpd.read_file(mesh_path).to_crs("EPSG:4326")
    aoi = mesh[mesh["CD_RGI"] == region_code]
    if len(aoi) != 1:
        raise ValueError(
            f"Esperada exatamente 1 feição com CD_RGI={region_code}; encontradas {len(aoi)}."
        )
    return aoi


def aoi_to_ee_geometry(aoi: Any) -> Any:
    """Converte a geometria do AOI (GeoDataFrame) em geometria do Earth Engine."""
    import ee

    return ee.Geometry(aoi.geometry.iloc[0].__geo_interface__)


def load_aoi_geometry(mesh_path: str | Path, region_code: str) -> tuple[Any, Any]:
    """Carrega o AOI e retorna (geometria GEE, GeoDataFrame filtrado)."""
    aoi = load_aoi_gdf(mesh_path, region_code)
    return aoi_to_ee_geometry(aoi), aoi


def _mask_s2_clouds(image: Any) -> Any:
    """Mascara nuvens e sombras pela banda QA60 (bits 10 e 11)."""
    qa = image.select("QA60")
    cloud_free = qa.bitwiseAnd(S2_CLOUD_BITMASK).eq(0)
    return image.updateMask(cloud_free)


def build_s2_collection(
    aoi: Any,
    bands: list[str],
    start: str,
    end: str,
    cloud_threshold: int,
) -> Any:
    """Constrói a coleção Sentinel-2 filtrada por AOI, datas e cobertura de nuvem."""
    import ee

    config = get_config()
    return (
        ee.ImageCollection(config["data"]["collection"])
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .map(_mask_s2_clouds)
        .select(bands)
    )


def build_annual_mosaic(collection: Any, aoi: Any) -> Any:
    """Compõe o mosaico mediano do período e recorta ao polígono do AOI."""
    return collection.median().clip(aoi)


def export_mosaic_to_drive(
    mosaic: Any,
    description: str,
    folder: str,
    file_name_prefix: str,
) -> Any:
    """Dispara a exportação do mosaico em GeoTIFF para o Drive (tarefa assíncrona).

    CRS, escala e limite de pixels vêm de src/config.yaml (fonte única de verdade).
    """
    import ee

    config = get_config()
    task = ee.batch.Export.image.toDrive(
        image=mosaic,
        description=description,
        folder=folder,
        fileNamePrefix=file_name_prefix,
        crs=config["data"]["crs"],
        scale=int(config["data"]["export"]["scale"]),
        region=mosaic.geometry(),
        maxPixels=int(config["data"]["export"]["max_pixels"]),
    )
    task.start()
    return task


def wait_for_task(
    task: Any,
    timeout_seconds: int = 7200,
    poll_interval: int = 30,
) -> str:
    """Aguarda a conclusão da tarefa GEE, imprimindo o progresso percentual.

    O progresso (0.0 a 1.0) é lido de ee.data.getOperation, que o GEE expõe
    nas operações de exportação; estados retornados (SUCCEEDED/FAILED) são
    normalizados para o mesmo vocabulário de task.status() (COMPLETED).
    """
    import ee

    deadline = time.monotonic() + timeout_seconds
    state = "READY"
    while state in ("READY", "PENDING", "RUNNING") and time.monotonic() < deadline:
        time.sleep(poll_interval)
        operation = ee.data.getOperation(task.name)
        metadata = operation.get("metadata", {})
        state = str(metadata.get("state", "UNKNOWN"))
        if state == "SUCCEEDED":
            state = "COMPLETED"
        progress = metadata.get("progress")
        if isinstance(progress, (int, float)):
            print(f"  Tarefa GEE: {state} ({progress * 100:.1f}%)")
        else:
            print(f"  Tarefa GEE: {state}")
    if state != "COMPLETED":
        raise RuntimeError(f"Tarefa GEE finalizou com estado: {state}")
    return state


def mosaic_file_name(region_code: str, start: str, end: str) -> str:
    """Gera o nome estável do mosaico exportado, derivado da configuração."""
    return f"sentinel2_{region_code}_{start}_{end}"


def drive_relative_path(path: Path) -> str:
    """Retorna o caminho relativo à raiz MyDrive (pasta de exportação no GEE)."""
    from src import io

    return str(path.relative_to(io.mount_drive()))
