"""Pré-processamento do mosaico Sentinel-2: composite alinhado e normalizado (estágio 05).

Constrói o composite multiespectral a partir do mosaico livre de nuvens do
estágio 01, alinhado ao grid da máscara binária final do estágio 04 (garantindo
sobreposição pixel a pixel para a geração de patches) e com a reflectância
normalizada para o intervalo [0, 1] (escala de reflectância do Sentinel-2, 1e-4).
A ordem das bandas do composite segue `data.bands` do config.yaml; o resultado é
persistido em MyDrive/tcc/data/processed/composites/ e a figura de registro em
MyDrive/tcc/artifacts/figures/. As persistências são idempotentes — reexecuções
reutilizam o composite e a figura existentes.
"""

from __future__ import annotations

import io as stdlib_io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_config
from src.data.mask_comparison import display_array
from src.data.mask_utils import reference_year

REFLECTANCE_SCALE = 1e-4  # fator de escala das bandas de reflectância do Sentinel-2


@dataclass(frozen=True)
class AlignedComposite:
    """Mosaico alinhado ao grid da máscara, com metadados de escrita."""

    array: np.ndarray  # [bandas, altura, largura] em valores digitais (sem normalizar)
    profile: dict[str, Any]  # perfil de escrita do GeoTIFF (grid da máscara)
    aligned: bool  # True se o grid do mosaico já coincidia com o da máscara


def mosaic_path(storage_paths: dict[str, Path]) -> Path:
    """Caminho canônico do mosaico Sentinel-2 do estágio 01."""
    from src.data.gee_client import mosaic_file_name

    config = get_config()
    prefix = mosaic_file_name(
        config["aoi"]["region_code"],
        config["data"]["dates"]["start"],
        config["data"]["dates"]["end"],
    )
    return storage_paths["data_raw_sentinel2"] / f"{prefix}.tif"


def composite_file_name() -> str:
    """Nome estável do composite normalizado, derivado da configuração."""
    config = get_config()
    return f"composite_{config['aoi']['region_code']}_{reference_year()}"


def composite_path(storage_paths: dict[str, Path]) -> Path:
    """Caminho canônico do composite normalizado (data/processed/composites/)."""
    return storage_paths["data_processed_composites"] / f"{composite_file_name()}.tif"


def load_aligned_mosaic(mosaic_path: Path, mask_path: Path) -> AlignedComposite:
    """Carrega o mosaico no grid da máscara de referência, reamostrando se preciso.

    Quando os grids coincidem (mesma CRS, transform e dimensões) os valores são
    lidos diretamente; caso contrário, cada banda é reamostrada (bilinear) para o
    grid exato da máscara, garantindo o alinhamento pixel a pixel exigido pelo
    estágio 06 (geração de patches).
    """
    import rasterio

    with rasterio.open(mosaic_path) as mosaic, rasterio.open(mask_path) as mask:
        bands = int(mosaic.count)
        if _same_grid(mosaic, mask):
            array = mosaic.read().astype(np.float32)
            aligned = True
        else:
            array = _reproject_to_grid(mosaic, mask)
            aligned = False
        profile = _composite_profile(mask, bands)
    return AlignedComposite(array=array, profile=profile, aligned=aligned)


def _same_grid(mosaic: Any, mask: Any) -> bool:
    """Indica se o mosaico e a máscara compartilham CRS, transform e dimensões."""
    return (
        mosaic.height == mask.height
        and mosaic.width == mask.width
        and mosaic.transform == mask.transform
        and mosaic.crs == mask.crs
    )


def _reproject_to_grid(mosaic: Any, mask: Any) -> np.ndarray:
    """Reamostra todas as bandas do mosaico para o grid da máscara."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    shape = (int(mask.height), int(mask.width))
    destination = np.full((int(mosaic.count), *shape), np.nan, dtype=np.float32)
    for band in range(1, int(mosaic.count) + 1):
        out_band = np.full(shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(mosaic, band),
            destination=out_band,
            src_transform=mosaic.transform,
            src_crs=mosaic.crs,
            dst_transform=mask.transform,
            dst_crs=mask.crs,
            resampling=Resampling.bilinear,
            src_nodata=None,
            dst_nodata=np.nan,
        )
        destination[band - 1] = out_band
    return destination


def _composite_profile(mask: Any, bands: int) -> dict[str, Any]:
    """Perfil de escrita do composite com base no grid da máscara de referência."""
    return {
        "driver": "GTiff",
        "height": int(mask.height),
        "width": int(mask.width),
        "count": bands,
        "dtype": "float32",
        "crs": mask.crs,
        "transform": mask.transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }


def normalize_reflectance(array: np.ndarray) -> np.ndarray:
    """Normaliza valores digitais (SR) para reflectância em [0, 1] (float32).

    O fator de escala 1e-4 do Sentinel-2 converte o valor digital em reflectância;
    valores fora do intervalo físico (ruído) são saturados em [0, 1] e NaN é
    preservado (propaga pelo clip e pela multiplicação).
    """
    scaled = array.astype(np.float32) * REFLECTANCE_SCALE
    return np.clip(scaled, 0.0, 1.0)


def build_preprocessed_composite(storage_paths: dict[str, Path]) -> Path:
    """Garante o composite alinhado e normalizado no caminho canônico (idempotente).

    Se o composite já existir, é reutilizado; caso contrário, o mosaico do estágio
    01 é alinhado ao grid da máscara final do estágio 04 e normalizado, com o
    GeoTIFF persistido em data/processed/composites/ (streaming no Kaggle).
    """
    from src import io
    from src.data.mask_finalization import final_mask_path

    target_path = composite_path(storage_paths)
    if io.path_exists(target_path):
        print(f"Composite já existente (reutilizado): {target_path}")
        return target_path

    mosaic_local = io.ensure_local_copy(mosaic_path(storage_paths))
    mask_local = io.ensure_local_copy(final_mask_path(storage_paths))
    if not mosaic_local.is_file():
        raise FileNotFoundError(f"Mosaico do estágio 01 não encontrado: {mosaic_local}")
    if not mask_local.is_file():
        raise FileNotFoundError(f"Máscara final do estágio 04 não encontrada: {mask_local}")

    composite = load_aligned_mosaic(mosaic_local, mask_local)
    normalized = normalize_reflectance(composite.array)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    _write_composite(target_path, normalized, composite.profile)
    # No Kaggle o arquivo local é enviado ao Drive canônico; no Colab/local é no-op.
    io.persist_file(target_path, target_path)
    status = "já alinhado" if composite.aligned else "reamostrado para o grid da máscara"
    print(f"Composite salvo em: {target_path} ({status})")
    return target_path


def _write_composite(path: Path, array: np.ndarray, profile: dict[str, Any]) -> None:
    """Escreve o composite normalizado em GeoTIFF com o perfil da máscara."""
    import rasterio

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def verify_composite(storage_paths: dict[str, Path]) -> dict[str, Any]:
    """Verifica o composite: grid, CRS, bandas, alinhamento e intervalo de valores."""
    import rasterio

    from src import io
    from src.data.mask_finalization import final_mask_path

    composite_local = io.ensure_local_copy(composite_path(storage_paths))
    mask_local = io.ensure_local_copy(final_mask_path(storage_paths))
    with rasterio.open(composite_local) as composite, rasterio.open(mask_local) as mask:
        shape = (int(composite.height), int(composite.width))
        crs = str(composite.crs)
        bands = int(composite.count)
        nodata = composite.nodata
        aligned = (
            composite.height == mask.height
            and composite.width == mask.width
            and composite.transform == mask.transform
            and composite.crs == mask.crs
        )
        array = composite.read()
        finite = array[np.isfinite(array)]
        value_range: list[float | None] = (
            [float(finite.min()), float(finite.max())] if finite.size else [None, None]
        )
    return {
        "shape": list(shape),
        "crs": crs,
        "bands": bands,
        "nodata": nodata,
        "aligned_to_mask": aligned,
        "value_range": value_range,
    }


def composite_preview_file_name() -> str:
    """Nome estável da figura do composite, derivado da configuração."""
    config = get_config()
    return f"composite_{config['aoi']['region_code']}_{reference_year()}.png"


def save_composite_preview(storage_paths: dict[str, Path]) -> Path:
    """Persiste a figura do composite no Drive canônico (idempotente)."""
    from src import io

    figure_path = storage_paths["artifacts_figures"] / composite_preview_file_name()
    if io.path_exists(figure_path):
        print(f"Figura já existente (reutilizada): {figure_path}")
        return figure_path

    composite_local = io.ensure_local_copy(composite_path(storage_paths))
    io.persist_bytes(figure_path, render_composite_preview(composite_local))
    print(f"Figura salva em: {figure_path}")
    return figure_path


def render_composite_preview(composite_path: Path) -> bytes:
    """Renderiza a miniatura RGB do composite (B4, B3, B2) com realce por percentil."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    with rasterio.open(composite_path) as src:
        data = src.read()
    band_index = {name: i for i, name in enumerate(get_config()["data"]["bands"])}
    red = _percentile_stretch(data[band_index["B4"]])
    green = _percentile_stretch(data[band_index["B3"]])
    blue = _percentile_stretch(data[band_index["B2"]])
    # Reduz cada banda (2D) antes de compor o RGB (display_array exige arrays 2D).
    rgb = np.stack([display_array(red), display_array(green), display_array(blue)], axis=-1)

    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(rgb)
    axis.set_title("Composite normalizado (RGB: B4, B3, B2)")
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()

    buffer = stdlib_io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    return buffer.getvalue()


def _percentile_stretch(band: np.ndarray, low: int = 2, high: int = 98) -> np.ndarray:
    """Realça uma banda contínua pelo estiramento de percentis para [0, 255]."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    p_low, p_high = np.percentile(finite, [low, high])
    if p_high <= p_low:
        p_high = p_low + 1e-6
    stretched = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)
    return (stretched * 255).astype(np.uint8)
