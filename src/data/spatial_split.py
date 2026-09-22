"""Divisão espacial k-fold dos patches (estágio 07).

Atribui cada patch registrado no manifesto do estágio 06 a uma das k dobras
(`splits.fold_count`) por proximidade geográfica: os centroides das bboxes são
agrupados com KMeans (semente fixa, algoritmo Lloyd) e os grupos são rotulados
de forma determinística pela posição dos centros. A contiguidade espacial dos
grupos reduz o vazamento por autocorrelação espacial entre treino e validação
nos estágios 09/10. A divisão é gravada na coluna `fold` do próprio manifesto e
versionada em split.meta.json (fingerprint do manifesto + dobras + semente);
execuções repetidas reutilizam a divisão vigente sem reprocessamento.
"""

from __future__ import annotations

import io as stdlib_io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config
from src.data.mask_utils import reference_year
from src.data.patch_generation import manifest_path, patch_fingerprint_hash

SPLIT_SCHEMA_VERSION = 1


def split_meta_path(storage_paths: dict[str, Path]) -> Path:
    """Caminho do metadata da divisão (fingerprint para idempotência)."""
    return storage_paths["data_processed"] / "split.meta.json"


def fold_count() -> int:
    """Número de dobras da validação cruzada, definido em config.yaml."""
    return int(get_config()["splits"]["fold_count"])


def split_seed() -> int:
    """Semente da divisão espacial, herdada da reprodutibilidade global."""
    return int(get_config()["reproducibility"]["seed"])


def bbox_centroid(bbox: str) -> tuple[float, float]:
    """Centroide (x, y) de uma bbox registrada como 'minx,miny,maxx,maxy'."""
    min_x, min_y, max_x, max_y = (float(value) for value in bbox.split(","))
    return (min_x + max_x) / 2.0, (min_y + max_y) / 2.0


def spatial_fold_centroids(
    centroids: np.ndarray, k: int, seed: int
) -> np.ndarray:
    """Agrupa os centroides em k regiões espaciais e rotula de forma determinística.

    Aplica KMeans (semente fixa, Lloyd) sobre as coordenadas dos centroides e
    reordena os rótulos pela posição dos centros (x, depois y), tornando o
    número da dobra estável e independente da ordem dos patches no manifesto.
    """
    from sklearn.cluster import KMeans

    if len(centroids) < k:
        raise ValueError(
            f"Patches insuficientes para {k} dobras ({len(centroids)}); "
            "reduza splits.fold_count."
        )
    model = KMeans(n_clusters=k, random_state=seed, n_init=10, algorithm="lloyd")
    labels = model.fit_predict(centroids)
    centers = model.cluster_centers_
    # Ordena os grupos por posição (x, depois y) para um rótulo estável.
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    relabel = {old: new for new, old in enumerate(order)}
    return np.array([relabel[int(label)] for label in labels])


def split_meta_payload(storage_paths: dict[str, Path]) -> dict[str, Any]:
    """Payload do metadata da divisão (fingerprint + dobras + semente)."""
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "manifest_fingerprint_hash": patch_fingerprint_hash(storage_paths),
        "fold_count": fold_count(),
        "seed": split_seed(),
    }


def _load_manifest(storage_paths: dict[str, Path]) -> pd.DataFrame:
    """Carrega o manifesto persistido e valida que não está vazio."""
    from src import io

    local_manifest = io.ensure_local_copy(manifest_path(storage_paths))
    manifest = pd.read_parquet(local_manifest)
    if manifest.empty:
        raise ValueError("Manifesto vazio; execute o estágio 06 antes.")
    return manifest


def _require_manifest(storage_paths: dict[str, Path]) -> None:
    """Falha com mensagem clara quando o manifesto do estágio 06 está ausente."""
    from src import io

    manifest = manifest_path(storage_paths)
    if not io.path_exists(manifest):
        raise FileNotFoundError(f"Manifesto do estágio 06 não encontrado: {manifest}")


def split_is_current(storage_paths: dict[str, Path]) -> bool:
    """Indica se a divisão persistida está vigente frente às entradas atuais."""
    from src import io

    if not io.path_exists(manifest_path(storage_paths)):
        return False
    if not io.path_exists(split_meta_path(storage_paths)):
        return False
    local_meta = io.ensure_local_copy(split_meta_path(storage_paths))
    try:
        stored = json.loads(local_meta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    current = split_meta_payload(storage_paths)
    if any(stored.get(key) != value for key, value in current.items()):
        return False
    return bool(_load_manifest(storage_paths)["fold"].notna().all())


def _write_manifest(
    path: Path, manifest: pd.DataFrame, storage_paths: dict[str, Path]
) -> None:
    """Persiste o manifesto (com a coluna fold) no Drive canônico."""
    from src import io

    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(path, index=False)
    io.persist_file(path, path)


def _write_meta(
    path: Path, payload: dict[str, Any], storage_paths: dict[str, Path]
) -> None:
    """Persiste o metadata da divisão no Drive canônico."""
    from src import io

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    io.persist_file(path, path)


def assign_spatial_folds(storage_paths: dict[str, Path]) -> Path:
    """Garante a divisão espacial k-fold no manifesto (idempotente).

    Reutiliza a divisão quando o metadata persistido coincide com o atual
    (fingerprint do manifesto, dobras e semente) e a coluna `fold` está preenchida;
    caso contrário, agrupa os patches por proximidade geográfica e grava a coluna
    `fold` no manifesto do estágio 06.
    """
    manifest_file = manifest_path(storage_paths)
    if split_is_current(storage_paths):
        print(f"Divisão já existente e atual (reutilizada): {manifest_file}")
        return manifest_file

    _require_manifest(storage_paths)
    manifest = _load_manifest(storage_paths)
    k = fold_count()
    seed = split_seed()
    centroids = np.array([bbox_centroid(value) for value in manifest["bbox"]])
    manifest["fold"] = spatial_fold_centroids(centroids, k, seed)
    _write_manifest(manifest_file, manifest, storage_paths)
    _write_meta(
        split_meta_path(storage_paths), split_meta_payload(storage_paths), storage_paths
    )
    print(f"Divisão espacial k-fold salva em: {manifest_file}")
    return manifest_file


def verify_split(storage_paths: dict[str, Path]) -> dict[str, Any]:
    """Verifica a divisão: contagens, café e localização média por dobra."""
    manifest = _load_manifest(storage_paths)
    k = fold_count()
    per_fold = []
    for fold in range(k):
        group = manifest[manifest["fold"] == fold]
        centroids = np.array([bbox_centroid(value) for value in group["bbox"]])
        per_fold.append(
            {
                "fold": fold,
                "n_patches": int(len(group)),
                "coffee_ratio_mean": float(group["coffee_ratio"].mean()),
                "centroid_x": float(centroids[:, 0].mean()),
                "centroid_y": float(centroids[:, 1].mean()),
            }
        )
    return {
        "fold_count": k,
        "n_patches": int(len(manifest)),
        "n_tiles": int(manifest["tile_id"].nunique()),
        "per_fold": per_fold,
    }


def split_figure_file_name() -> str:
    """Nome estável da figura da divisão espacial, derivado da configuração."""
    config = get_config()
    return f"split_kfold_{config['aoi']['region_code']}_{reference_year()}.png"


def save_split_figure(storage_paths: dict[str, Path]) -> Path:
    """Persiste a figura da divisão espacial no Drive canônico (idempotente)."""
    from src import io

    figure_path = storage_paths["artifacts_figures"] / split_figure_file_name()
    if io.path_exists(figure_path):
        print(f"Figura já existente (reutilizada): {figure_path}")
        return figure_path

    manifest = _load_manifest(storage_paths)
    io.persist_bytes(figure_path, render_split_figure(manifest))
    print(f"Figura salva em: {figure_path}")
    return figure_path


def render_split_figure(manifest: pd.DataFrame) -> bytes:
    """Renderiza o mapa de centroides dos patches coloridos por dobra."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centroids = np.array([bbox_centroid(value) for value in manifest["bbox"]])
    folds = manifest["fold"].astype(int)

    figure, axis = plt.subplots(figsize=(8, 8))
    scatter = axis.scatter(
        centroids[:, 0], centroids[:, 1], c=folds, cmap="tab10", s=12, alpha=0.8
    )
    axis.set_title(f"Divisão espacial k-fold (k={int(folds.max()) + 1})")
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    figure.colorbar(scatter, ax=axis, label="Dobra")
    figure.tight_layout()

    buffer = stdlib_io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    return buffer.getvalue()