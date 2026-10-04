"""Máscara binária de café da Emater — fonte vetorial local (estágio 02).

A fonte Emater (Geoportal do Café — mapeamento do parque cafeeiro de Minas
Gerais concluído em 2018, com validação em campo pelos extensionistas) é um
vetor local: os GeoJSONs por município ficam em `data/external/emater/`
(versionados e entregues ao runtime pelo bootstrap), não vêm do GEE.

Esta etapa concatena os municípios, reprojeta para o CRS do projeto
(EPSG:31983), repara geometrias inválidas (buffer(0)), clipa ao AOI e
rasteriza a máscara binária sobre o grid de 10 m do AOI, persistindo o
GeoTIFF em MyDrive/tcc/data/interim/emater/. A persistência é idempotente.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_config
from src.data.mask_utils import reference_year

CRS = "EPSG:31983"  # SIRGAS 2000 / UTM 23S (mesmo grid dos rasters do projeto)


def _load_geojson_paths(data_dir: Path) -> list[Path]:
    """Retorna os GeoJSONs da fonte (um por município), ordenados pelo nome."""
    paths = sorted(set(data_dir.glob("*.geojson")) | set(data_dir.glob("*.json")))
    if not paths:
        raise FileNotFoundError(f"Nenhum GeoJSON da Emater encontrado em: {data_dir}")
    return paths


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    """Resolve o diretório de GeoJSONs (relativo ao workspace quando necessário)."""
    if data_dir is None:
        source = get_config()["ground_truth"]["sources"]["emater"]
        data_dir = source["data_dir"]
    path = Path(data_dir)
    if path.is_absolute() or path.is_dir():
        return path
    # Workspaces padrão das plataformas (o bootstrap entrega data/external/).
    for root in (Path.cwd(), Path("/content"), Path("/kaggle/working")):
        candidate = root / path
        if candidate.is_dir():
            return candidate
    return path


def load_emater_features(data_dir: Path) -> Any:
    """Carrega e concatena as feições de café de todos os municípios (EPSG:4326)."""
    import geopandas as gpd
    import pandas as pd

    paths = _load_geojson_paths(data_dir)
    frames = []
    for path in paths:
        features = gpd.read_file(path)
        features["municipio"] = path.stem.replace("GetGlebasGeoJson - ", "")
        frames.append(features)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")


def _aoi_grid(aoi_utm: Any, scale: int) -> tuple[Any, int, int]:
    """Grid de 10 m cobrindo o AOI em EPSG:31983 (origem alinhada à escala)."""
    from rasterio.transform import from_origin

    minx, miny, maxx, maxy = aoi_utm.bounds
    x0 = math.floor(minx / scale) * scale
    y0 = math.floor(miny / scale) * scale
    x1 = math.ceil(maxx / scale) * scale
    y1 = math.ceil(maxy / scale) * scale
    transform = from_origin(x0, y1, scale, scale)
    height = int(round((y1 - y0) / scale))
    width = int(round((x1 - x0) / scale))
    return transform, height, width


def rasterize_emater_mask(
    aoi_utm: Any,
    features_utm: Any,
    transform: Any,
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasteriza as glebas de café no grid do AOI (0/1), clipando ao polígono."""
    from rasterio.features import rasterize

    coffee = rasterize(
        [(geometry, 1) for geometry in features_utm.geometry if not geometry.is_empty],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    aoi_clip = rasterize(
        [(aoi_utm, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    return (coffee * aoi_clip).astype("uint8")


def _write_mask_geotiff(path: Path, array: np.ndarray, transform: Any) -> None:
    """Escreve a máscara binária em GeoTIFF uint8 no CRS do projeto."""
    import rasterio

    profile = {
        "driver": "GTiff",
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "count": 1,
        "dtype": "uint8",
        "crs": CRS,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def ensure_emater_mask(
    source_name: str,
    aoi_utm: Any,
    storage_paths: dict[str, Path],
    data_dir: str | Path | None = None,
) -> Path:
    """Garante a máscara binária da Emater no caminho canônico (idempotente)."""
    from src import io
    from src.data.mask_utils import source_mask_path

    config = get_config()
    target_path = source_mask_path(source_name, storage_paths)
    if io.path_exists(target_path):
        print(f"Máscara {source_name} já existente (reutilizada): {target_path}")
        return target_path

    scale = int(config["data"]["export"]["scale"])
    features = load_emater_features(_resolve_data_dir(data_dir))
    features_utm = features.to_crs(CRS)
    # Repara geometrias inválidas (ex.: anéis autointersectantes) sem alterar área.
    features_utm = features_utm.copy()
    features_utm["geometry"] = features_utm.geometry.buffer(0)

    transform, height, width = _aoi_grid(aoi_utm, scale)
    array = rasterize_emater_mask(aoi_utm, features_utm, transform, (height, width))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    _write_mask_geotiff(target_path, array, transform)
    io.persist_file(target_path, target_path)
    coffee = int(np.count_nonzero(array > 0))
    print(
        f"Máscara {source_name} gerada em: {target_path} "
        f"({height}x{width} px, {coffee:,} pixels de café)"
    )
    return target_path


def verify_emater_mask(source_name: str, storage_paths: dict[str, Path]) -> dict[str, Any]:
    """Verifica a máscara da Emater: grid, CRS, valores binários e área."""
    import rasterio

    from src import io
    from src.data.mask_utils import source_mask_path

    path = io.ensure_local_copy(source_mask_path(source_name, storage_paths))
    with rasterio.open(path) as src:
        array = src.read(1)
        if not set(np.unique(array)).issubset({0, 1}):
            raise ValueError("Máscara da Emater com valores não binários.")
        pixel_size_m = abs(float(src.transform.a))
        shape = (int(src.height), int(src.width))
        crs = str(src.crs)
    coffee = int(np.count_nonzero(array > 0))
    return {
        "shape": list(shape),
        "crs": crs,
        "coffee_pixels": coffee,
        "area_km2": coffee * pixel_size_m**2 / 1e6,
    }


def emater_mask_preview_file_name(source_name: str) -> str:
    """Nome estável da miniatura da máscara da Emater, derivado da configuração."""
    config = get_config()
    prefix = f"mask_{source_name}_{config['aoi']['region_code']}_{reference_year(source_name)}"
    return f"{prefix}_preview.png"


def save_emater_mask_preview(source_name: str, storage_paths: dict[str, Path]) -> Path:
    """Persiste a miniatura binária da máscara da Emater (idempotente)."""
    from src import io
    from src.data.mask_utils import source_mask_path

    figure_path = storage_paths["artifacts_figures"] / emater_mask_preview_file_name(source_name)
    if io.path_exists(figure_path):
        print(f"Figura já existente: {figure_path}")
        return figure_path
    mask_path = io.ensure_local_copy(source_mask_path(source_name, storage_paths))
    io.persist_bytes(figure_path, render_emater_mask_preview(mask_path))
    print(f"Figura salva em: {figure_path}")
    return figure_path


def render_emater_mask_preview(mask_path: Path) -> bytes:
    """Renderiza a miniatura binária da máscara da Emater (café em roxo).

    Reproduz o padrão das miniaturas getThumbURL das fontes GEE
    (MapBiomas/AlphaEarth): imagem pura (sem eixos/moldura) de 1024 px no maior
    lado, transparente fora do polígono do AOI, branco (#ffffff) para não-café
    e roxo (#800080) para café. O raster UTM é reprojetado para o grid 4326 do
    AOI (mesma projeção/extensão do getThumbURL), garantindo as mesmas
    dimensões e cores das demais fontes.
    """
    import io as stdlib_io

    import matplotlib
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    matplotlib.use("Agg")
    import matplotlib.image as mpl_image

    from src.config import get_config
    from src.data.gee_client import load_aoi_gdf

    config = get_config()
    aoi = load_aoi_gdf(config["aoi"]["mesh_path"], config["aoi"]["region_code"])
    minx, miny, maxx, maxy = aoi.geometry.iloc[0].bounds
    # Grid 4326 do AOI com 1024 px no maior lado (mesma convenção do getThumbURL;
    # o GEE arredonda o menor lado para cima — ceil).
    width = 1024
    height = max(1, math.ceil((maxy - miny) / (maxx - minx) * width))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    with rasterio.open(mask_path) as src:
        mask_utm = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

    # Reprojeta a máscara UTM para o grid 4326 do AOI (vizinho mais próximo
    # preserva os valores binários 0/1).
    mask = np.zeros((height, width), dtype=np.uint8)
    rasterio.warp.reproject(
        source=mask_utm,
        destination=mask,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
    )
    # Footprint do AOI no grid 4326: fora do polígono fica transparente.
    footprint = rasterize(
        [(aoi.geometry.iloc[0], 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    # Paleta igual ao getThumbURL: branco (#ffffff) não-café, roxo (#800080) café.
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = np.where(mask > 0, 128, 255)
    rgba[..., 1] = np.where(mask > 0, 0, 255)
    rgba[..., 2] = np.where(mask > 0, 128, 255)
    rgba[..., 3] = np.where(footprint > 0, 255, 0)

    buffer = stdlib_io.BytesIO()
    mpl_image.imsave(buffer, rgba, format="png")
    return buffer.getvalue()
