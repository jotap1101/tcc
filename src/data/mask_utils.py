"""Máscaras de referência por fonte de ground truth (uma função por fonte).

Cada fonte habilitada (MapBiomas, AlphaEarth, ...) tem um construtor dedicado
de máscara binária de café e uma subpasta própria em MyDrive/tcc/data/interim.
Coleções, classes e limiares vêm de src/config.yaml (fonte única de verdade);
as exportações são idempotentes — reexecuções reutilizam o GeoTIFF já existente.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config import get_config

MASK_BAND = "mask"


def reference_year() -> int:
    """Ano de referência das máscaras, derivado das datas do mosaico Sentinel-2."""
    config = get_config()
    return int(config["data"]["dates"]["start"][:4])


def mask_file_name(source: str, region_code: str, year: int) -> str:
    """Gera o nome estável do arquivo de máscara, derivado da configuração."""
    return f"mask_{source}_{region_code}_{year}"


def source_mask_path(source_name: str, storage_paths: dict[str, Path]) -> Path:
    """Caminho canônico do GeoTIFF da máscara da fonte (data/interim/<fonte>/)."""
    config = get_config()
    file_prefix = mask_file_name(source_name, config["aoi"]["region_code"], reference_year())
    return storage_paths["data_interim"] / source_name / f"{file_prefix}.tif"


def build_mapbiomas_mask(aoi: Any) -> Any:
    """Máscara binária de café da MapBiomas (classe 46) para o ano de referência.

    A coleção integrada possui uma banda `classification_<ano>` por ano;
    a máscara equivale aos pixels da classe café (3.2.2.1) recortados ao AOI.
    """
    import ee

    source = get_config()["ground_truth"]["sources"]["mapbiomas"]
    image = ee.Image(source["collection_id"]).select(f"classification_{reference_year()}")
    return image.eq(source["coffee_class"]).clip(aoi).rename(MASK_BAND)


def build_alphaearth_mask(aoi: Any) -> Any:
    """Máscara binária de café do modelo de probabilidade da AlphaEarth (FDaP).

    O modelo do Forest Data Partnership (2025a), derivado dos embeddings do
    AlphaEarth Foundations, fornece a banda `probability` por ano; a máscara
    aplica o limiar configurado em config.yaml para o ano de referência.
    """
    import ee

    source = get_config()["ground_truth"]["sources"]["alphaearth"]
    year = reference_year()
    collection = ee.ImageCollection(source["collection_id"]).filterDate(
        f"{year}-01-01", f"{year}-12-31"
    )
    probability = collection.mosaic().select("probability")
    return probability.gte(source["probability_threshold"]).clip(aoi).rename(MASK_BAND)


MASK_BUILDERS: dict[str, Callable[[Any], Any]] = {
    "mapbiomas": build_mapbiomas_mask,
    "alphaearth": build_alphaearth_mask,
}


def build_mask(source_name: str, aoi: Any) -> Any:
    """Constrói a máscara binária da fonte usando o construtor dedicado."""
    builder = MASK_BUILDERS.get(source_name)
    if builder is None:
        raise ValueError(f"Fonte de ground truth desconhecida: {source_name}")
    return builder(aoi)


def export_mask_to_drive(
    mask: Any,
    description: str,
    folder: str,
    file_name_prefix: str,
) -> Any:
    """Dispara a exportação da máscara binária em GeoTIFF para o Drive.

    Usa a escala de processamento (10 m) e o CRS do config.yaml, garantindo o
    mesmo grid do mosaico Sentinel-2; a reamostragem padrão do GEE (near)
    preserva os valores categóricos (0/1) da máscara.
    """
    from src.data.gee_client import export_image_to_drive

    config = get_config()
    return export_image_to_drive(
        image=mask,
        description=description,
        folder=folder,
        file_name_prefix=file_name_prefix,
        region=mask.geometry(),
        scale=int(config["data"]["export"]["scale"]),
    )


def save_mask_preview(source_name: str, aoi: Any, preview_path: Path) -> None:
    """Salva a miniatura binária (0/1) da máscara da fonte (idempotente)."""
    from src import io

    # Reutiliza a miniatura já gerada (execuções repetidas não reprocessam).
    if io.path_exists(preview_path):
        print(f"Figura já existente: {preview_path}")
        return
    mask = build_mask(source_name, aoi)
    thumb_url = mask.getThumbURL(
        {
            "min": 0,
            "max": 1,
            "bands": [MASK_BAND],
            "palette": ["white", "purple"],
            "dimensions": 1024,
        }
    )
    io.persist_bytes(preview_path, urllib.request.urlopen(thumb_url).read())
    print(f"Figura salva em: {preview_path}")


def ensure_source_mask(
    source_name: str,
    aoi: Any,
    storage_paths: dict[str, Path],
) -> Path | None:
    """Garante a máscara binária da fonte no caminho canônico (idempotente).

    Retorna o caminho do GeoTIFF quando a fonte está habilitada (existente ou
    exportado nesta execução); None se a fonte está desabilitada no config.yaml.
    """
    from src import io
    from src.data.gee_client import wait_for_task

    config = get_config()
    source = config["ground_truth"]["sources"][source_name]
    if not source["enabled"]:
        print(f"Fonte {source_name} desabilitada em config.yaml; nada a exportar.")
        return None

    region_code = config["aoi"]["region_code"]
    year = reference_year()
    file_prefix = mask_file_name(source_name, region_code, year)
    target_path = source_mask_path(source_name, storage_paths)
    staging_folder = config["storage"]["drive_root"]

    # Reutiliza o GeoTIFF já exportado (execuções repetidas não reprocessam).
    if io.path_exists(target_path):
        print(f"Máscara {source_name} já exportada (reutilizada): {target_path}")
        return target_path

    mask = build_mask(source_name, aoi)
    task = export_mask_to_drive(
        mask=mask,
        description=f"mask_{source_name}_{region_code}",
        folder=staging_folder,
        file_name_prefix=file_prefix,
    )
    print(f"Tarefa de exportação iniciada: {task.id}")
    wait_for_task(task)
    io.relocate_exported_file(
        file_name=f"{file_prefix}.tif",
        staging_folder=staging_folder,
        target_path=target_path,
    )
    print(f"Máscara {source_name} exportada em: {target_path}")
    return target_path
