"""Inicialização do Google Earth Engine e utilitários de aquisição.

Nenhuma credencial é lida de arquivos versionados, impressa em logs ou
persistida: os valores vêm exclusivamente de variáveis de ambiente, conforme
``.env.example``. Este módulo é reutilizado pelos notebooks das fases 0 e 1.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from src.config import CONFIG

# Variáveis obrigatórias para a autenticação por conta de serviço.
REQUIRED_ENV_VARS: tuple[str, ...] = (
    "GEE_SERVICE_ACCOUNT_EMAIL",
    "GEE_PROJECT",
)

# Formatos aceitos para a chave privada (um dos dois é obrigatório).
KEY_PATH_ENV = "GEE_SERVICE_ACCOUNT_KEY_PATH"
KEY_JSON_ENV = "GEE_SERVICE_ACCOUNT_KEY_JSON"


class GEECredentialsError(RuntimeError):
    """Erro de configuração das credenciais do Earth Engine."""


def _write_key_from_json(key_json: str) -> Path:
    """Materializa a chave JSON embutida em um arquivo temporário."""
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="gee-key-",
        delete=False,
        encoding="utf-8",
    )
    with handle:
        handle.write(key_json)
    return Path(handle.name)


def _resolve_key_path() -> Path:
    """Resolve o caminho da chave privada a partir do ambiente."""
    key_path = os.environ.get(KEY_PATH_ENV)
    if key_path:
        path = Path(key_path).expanduser()
        if not path.is_file():
            raise GEECredentialsError(
                f"{KEY_PATH_ENV} aponta para arquivo inexistente: {path}"
            )
        return path

    key_json = os.environ.get(KEY_JSON_ENV)
    if key_json:
        return _write_key_from_json(key_json)

    raise GEECredentialsError(
        f"Defina {KEY_PATH_ENV} ou {KEY_JSON_ENV} com a chave da conta de serviço."
    )


def init_ee() -> Any:
    """Inicializa o Earth Engine com a conta de serviço e retorna o módulo ``ee``."""
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise GEECredentialsError(
            "Variáveis de ambiente ausentes: " + ", ".join(missing)
        )

    import ee  # importação tardia: não exige a dependência no CI

    email = os.environ["GEE_SERVICE_ACCOUNT_EMAIL"]
    project = os.environ["GEE_PROJECT"]
    key_path = _resolve_key_path()

    credentials = ee.ServiceAccountCredentials(email, str(key_path))
    ee.Initialize(credentials, project=project)
    return ee


def get_s2_sr_collection(
    ee: Any,
    aoi: Any,
    start_date: str | None = None,
    end_date: str | None = None,
    cloud_filter_pct: float | None = None,
    collection_id: str | None = None,
) -> Any:
    """Monta a coleção Sentinel-2 Nível-2A filtrada por área, data e nuvem.

    Os parâmetros ausentes são resolvidos a partir de ``config.yaml``.
    """
    start = str(start_date or CONFIG.get("gee.start_date"))
    end = str(end_date or CONFIG.get("gee.end_date"))
    cloud_pct = float(
        cloud_filter_pct
        if cloud_filter_pct is not None
        else CONFIG.get("gee.cloud_filter_pct")
    )
    asset = str(collection_id or CONFIG.get("gee.image_collection"))

    return (
        ee.ImageCollection(asset)
        .filterBounds(aoi)
        .filterDate(start, end)
        # Descarta cenas globalmente muito nubladas antes da correção fina.
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )


def mask_clouds_cloud_score_plus(
    ee: Any,
    collection: Any,
    qa_band: str | None = None,
    threshold: float | None = None,
    collection_id: str | None = None,
) -> Any:
    """Aplica a máscara Cloud Score+ via ``linkCollection``.

    Cada imagem Sentinel-2 é vinculada à sua cena Cloud Score+ pelo
    ``system:index``; pixels com probabilidade de céu claro abaixo do limiar são
    mascarados.
    """
    band = str(qa_band or CONFIG.get("gee.cloud_score_band", "cs_cdf"))
    limit = float(
        threshold
        if threshold is not None
        else CONFIG.get("gee.cloud_score_threshold", 0.60)
    )
    cs_id = str(collection_id or CONFIG.get("gee.cloud_score_plus_collection"))

    cloud_score = ee.ImageCollection(cs_id)
    linked = collection.linkCollection(cloud_score, [band])

    def _apply_mask(image: Any) -> Any:
        clear = image.select(band).gte(limit)
        return image.updateMask(clear)

    return linked.map(_apply_mask)


def mask_clouds_qa60(ee: Any, image: Any, qa_band: str | None = None) -> Any:
    """Máscara de nuvem pela banda nativa QA60 (bits 10 e 11).

    Usada como fallback quando a cena não possui correspondente no Cloud Score+.
    """
    band = str(qa_band or CONFIG.get("gee.qa_band", "QA60"))
    qa = image.select(band)
    # Bit 10 = nuvem opaca; bit 11 = cirro.
    cloud_free = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_free)


def mask_clouds(
    ee: Any,
    collection: Any,
    strategy: str = "cloud_score_plus",
) -> Any:
    """Seleciona a estratégia de máscara de nuvem da coleção."""
    if strategy == "cloud_score_plus":
        return mask_clouds_cloud_score_plus(ee, collection)
    if strategy == "qa60":
        return collection.map(lambda image: mask_clouds_qa60(ee, image))
    raise ValueError(f"Estratégia de máscara de nuvem desconhecida: {strategy}")


def build_composite(
    ee: Any,
    collection: Any,
    bands: list[str] | None = None,
    composite: str | None = None,
) -> Any:
    """Reduz a série temporal em uma composição livre de nuvens."""
    selected_bands = list(bands or CONFIG.bands)
    reducer = str(composite or CONFIG.get("gee.composite", "median"))
    selected = collection.select(selected_bands)

    if reducer == "median":
        return selected.median()
    if reducer == "mean":
        return selected.mean()
    if reducer == "mosaic":
        return selected.mosaic()
    raise ValueError(f"Estatística de composição desconhecida: {reducer}")


def build_cloud_free_composite(
    ee: Any,
    aoi: Any,
    strategy: str = "cloud_score_plus",
    bands: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Any:
    """Pipeline completo: coleção S2 → máscara de nuvem → composição recortada."""
    collection = get_s2_sr_collection(ee, aoi, start_date=start_date, end_date=end_date)
    masked = mask_clouds(ee, collection, strategy=strategy)
    composite = build_composite(ee, masked, bands=bands)
    return composite.clip(aoi)


def make_export_description(
    prefix: str,
    region_code: str,
    start_date: str,
    end_date: str,
) -> str:
    """Gera a descrição determinística de uma tarefa de exportação."""
    compact_start = start_date.replace("-", "")
    compact_end = end_date.replace("-", "")
    return f"{prefix}_{region_code}_{compact_start}_{compact_end}"


def export_image_to_drive(
    ee: Any,
    image: Any,
    description: str,
    region: Any,
    folder: str | None = None,
    subfolder: str | None = None,
    file_name_prefix: str | None = None,
    scale: int | None = None,
    crs: str | None = None,
    max_pixels: float | None = None,
) -> Any:
    """Dispara uma tarefa de exportação de imagem para o Google Drive."""
    # Pasta raiz configurada, opcionalmente complementada por uma subpasta.
    drive_folder = str(folder or CONFIG.get("gee.export_folder", "tcc"))
    if subfolder:
        drive_folder = f"{drive_folder}/{subfolder}"
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=drive_folder,
        fileNamePrefix=str(
            file_name_prefix or CONFIG.get("gee.export_prefix", "guaxupe")
        ),
        region=region,
        scale=int(scale or CONFIG.get("gee.scale_m", 10)),
        crs=str(crs or CONFIG.get("gee.crs")),
        maxPixels=float(max_pixels or CONFIG.get("gee.max_pixels")),
    )
    task.start()
    return task
