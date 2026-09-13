"""Carregamento da malha vetorial do IBGE para definir a área de estudo.

A área de estudo é a Região Geográfica Imediata de Guaxupé - MG, recortada da
malha vetorial oficial do IBGE. ``geopandas`` é importado tardiamente para que
o pacote possa ser validado no CI (sem o extra ``geo``). Todo caminho é
resolvido a partir de ``src/config.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.config import CONFIG

# Variável de ambiente que permite sobrescrever o caminho da malha (ex.: Kaggle).
MESH_PATH_ENV = "TCC_MESH_PATH"


class AOIError(RuntimeError):
    """Erro ao resolver ou carregar a malha vetorial da área de estudo."""


def resolve_mesh_path(mesh_path: str | Path | None = None) -> Path:
    """Resolve o caminho absoluto da malha vetorial do IBGE.

    A precedência é: argumento explícito, variável de ambiente ``TCC_MESH_PATH``
    e, por fim, ``aoi.mesh_path`` de ``config.yaml``. Caminhos relativos são
    ancorados na raiz do repositório.
    """
    candidate = (
        mesh_path or os.environ.get(MESH_PATH_ENV) or CONFIG.get("aoi.mesh_path")
    )
    if not candidate:
        raise AOIError("Caminho da malha vetorial não definido em aoi.mesh_path.")

    path = Path(str(candidate)).expanduser()
    if not path.is_absolute():
        path = CONFIG.paths.root / path
    return path


def select_region_frame(
    frame: Any,
    region_code: str,
    region_code_column: str = "CD_RGI",
) -> Any:
    """Filtra o registro da região imediata pelo código IBGE.

    Recebe qualquer objeto tabular compatível com ``pandas`` (o GeoDataFrame
    retornado por ``geopandas.read_file``), o que mantém o helper testável sem
    dependência geoespacial.
    """
    if region_code_column not in frame.columns:
        raise AOIError(
            f"Coluna '{region_code_column}' ausente na malha vetorial: "
            f"{list(frame.columns)}"
        )

    subset = frame[frame[region_code_column].astype(str) == str(region_code)]
    if subset.empty:
        raise AOIError(
            f"Região de código {region_code} não encontrada na coluna "
            f"'{region_code_column}'."
        )
    return subset


def load_region_mesh(
    mesh_path: str | Path | None = None,
    region_code: str | None = None,
    region_code_column: str | None = None,
) -> Any:
    """Carrega a malha do IBGE e retorna apenas o registro da região alvo."""
    import geopandas as gpd  # importação tardia: requer o extra ``geo``

    path = resolve_mesh_path(mesh_path)
    if not path.is_file():
        raise AOIError(f"Arquivo da malha vetorial não encontrado: {path}")

    frame = gpd.read_file(path)
    code = str(region_code or CONFIG.get("aoi.region_code"))
    column = str(region_code_column or CONFIG.get("aoi.region_code_column", "CD_RGI"))
    return select_region_frame(frame, code, column)


def get_region_geometry(
    mesh_path: str | Path | None = None,
    region_code: str | None = None,
) -> Any:
    """Retorna a geometria unificada da região como um único polígono/multipolígono."""
    subset = load_region_mesh(mesh_path=mesh_path, region_code=region_code)
    geometries = subset.geometry
    if hasattr(geometries, "union_all"):
        return geometries.union_all()
    return geometries.unary_union


def geometry_bounds(geometry: Any) -> tuple[float, float, float, float]:
    """Retorna os limites (minx, miny, maxx, maxy) de uma geometria."""
    minx, miny, maxx, maxy = (float(value) for value in geometry.bounds)
    return (minx, miny, maxx, maxy)


def geometry_to_ee(ee: Any, geometry: Any) -> Any:
    """Converte uma geometria (shapely/GeoJSON) em ``ee.Geometry``."""
    return ee.Geometry(geometry.__geo_interface__)
