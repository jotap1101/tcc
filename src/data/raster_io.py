"""Entrada e saída de rasters (GeoTIFF) com ``rasterio``.

``rasterio`` é importado tardiamente para que o módulo permaneça importável no
CI (extra ``geo`` não instalado). O alinhamento das máscaras ao grid da
composição Sentinel-2 garante o casamento pixel a pixel na comparação.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_band(path: str | Path, band: int = 1) -> np.ndarray:
    """Lê uma banda de um GeoTIFF como arranjo numpy."""
    import rasterio  # importação tardia: requer o extra ``geo``

    with rasterio.open(path) as source:
        return np.asarray(source.read(band))


def write_single_band(
    array: np.ndarray,
    out_path: str | Path,
    reference_path: str | Path,
) -> Path:
    """Grava um arranjo como GeoTIFF de banda única no perfil do referência."""
    import rasterio  # importação tardia: requer o extra ``geo``

    with rasterio.open(reference_path) as reference:
        profile = reference.profile.copy()
        profile.update(dtype=array.dtype.name, count=1)
        with rasterio.open(out_path, "w", **profile) as destination:
            destination.write(array, 1)
    return Path(out_path)


def align_to_reference(
    src_path: str | Path,
    ref_path: str | Path,
    out_path: str | Path,
    resampling: str = "nearest",
) -> Path:
    """Reprojeta/recorta uma máscara para o grid exato do raster de referência."""
    import rasterio  # importação tardia: requer o extra ``geo``
    from rasterio.enums import Resampling

    resampling_method = getattr(Resampling, resampling)
    with rasterio.open(ref_path) as reference, rasterio.open(src_path) as source:
        profile = source.profile.copy()
        # Alinha CRS, transform e dimensões ao raster de referência.
        profile.update(
            crs=reference.crs,
            transform=reference.transform,
            width=reference.width,
            height=reference.height,
        )
        with rasterio.open(out_path, "w", **profile) as destination:
            for band_index in range(1, source.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(source, band_index),
                    destination=rasterio.band(destination, band_index),
                    src_transform=source.transform,
                    src_crs=source.crs,
                    dst_transform=reference.transform,
                    dst_crs=reference.crs,
                    resampling=resampling_method,
                )
    return Path(out_path)
