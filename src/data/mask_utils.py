"""Construção de máscaras de referência para a classe café.

Duas fontes candidatas são preparadas na fase de aquisição: a classificação
MapBiomas (classe café = 46) e um agrupamento não supervisionado sobre os
embeddings AlphaEarth. ``numpy`` é dependência base; ``ee`` é recebido como
argumento para manter o módulo importável no CI.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import CONFIG


def mapbiomas_band_candidates(band: str, year: int) -> list[str]:
    """Lista os nomes de banda aceitos para o ano de referência.

    Cobre tanto assets multi-banda (``classification_YYYY``) quanto assets
    anuais com banda única (``classification``).
    """
    return [f"{band}_{year}", band]


def binarize_array(array: Any, target_class: int) -> Any:
    """Binariza um arranjo de classes, retornando uint8 com 1 na classe alvo."""
    return (np.asarray(array) == int(target_class)).astype(np.uint8)


def get_mapbiomas_image(
    ee: Any,
    asset: str,
    band: str,
    year: int,
    from_collection: bool = False,
) -> Any:
    """Obtém a imagem de classificação MapBiomas do ano de referência.

    ``from_collection=True`` trata o asset como ``ImageCollection`` anual;
    caso contrário, como uma imagem multi-banda.
    """
    if from_collection:
        collection = ee.ImageCollection(asset).filterDate(
            f"{year}-01-01", f"{year}-12-31"
        )
        return collection.mosaic().select(band)
    return ee.Image(asset).select(band)


def build_mapbiomas_coffee_mask(
    ee: Any,
    aoi: Any,
    asset: str | None = None,
    band: str | None = None,
    year: int | None = None,
    coffee_class: int | None = None,
    from_collection: bool = False,
) -> Any:
    """Máscara binária de café a partir da classificação MapBiomas."""
    asset_id = str(asset or CONFIG.get("masks.mapbiomas_10m_asset"))
    band_name = str(band or CONFIG.get("masks.mapbiomas_band", "classification"))
    ref_year = int(year or CONFIG.get("masks.year"))
    target = int(coffee_class or CONFIG.get("masks.coffee_class"))

    classification = get_mapbiomas_image(
        ee, asset_id, band_name, ref_year, from_collection=from_collection
    )
    # Pixels iguais à classe de café recebem 1; os demais, 0.
    mask = classification.eq(target).rename("coffee")
    return mask.toUint8().clip(aoi)


def get_alphaearth_embedding(
    ee: Any,
    aoi: Any,
    collection: str | None = None,
    year: int | None = None,
) -> Any:
    """Mosaico anual dos embeddings AlphaEarth (64 bandas) recortado pela AOI."""
    collection_id = str(collection or CONFIG.get("masks.alphaearth_collection"))
    ref_year = int(year or CONFIG.get("masks.alphaearth_year"))

    return (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(f"{ref_year}-01-01", f"{ref_year}-12-31")
        .mosaic()
        .clip(aoi)
    )


def cluster_alphaearth(
    ee: Any,
    image: Any,
    aoi: Any,
    clusters: int | None = None,
    samples: int | None = None,
    seed: int | None = None,
) -> Any:
    """Agrupa os embeddings com k-means como máscara candidata não supervisionada."""
    n_clusters = int(clusters or CONFIG.get("masks.alphaearth_clusters", 8))
    n_samples = int(samples or CONFIG.get("masks.alphaearth_samples", 5000))
    fixed_seed = int(seed if seed is not None else CONFIG.seed)
    scale = int(CONFIG.get("gee.scale_m", 10))

    training = image.sample(
        region=aoi,
        scale=scale,
        numPixels=n_samples,
        seed=fixed_seed,
    )
    clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(training)
    return image.cluster(clusterer)
