"""Análise exploratória dos patches e estatísticas de normalização (estágio 08).

Calcula as estatísticas de normalização por banda (média, desvio padrão e fração
de NaN) sobre todos os pixels finitos dos patches registrados no manifesto do
estágio 06/07, persistindo-as em data/processed/normalization_stats.json com
fingerprint de versão — idempotente, pois reexecuções reutilizam o arquivo
vigente sem reprocessar. Executa também verificações de sanidade do dataset
(formas, tipos, máscaras binárias, distribuição de café, dobras e amostra de
arquivos) e gera a figura de resumo em artifacts/figures/.
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
from src.data.patch_generation import (
    manifest_path,
    patch_fingerprint_hash,
    resolve_from_root,
)

EDA_SCHEMA_VERSION = 1
SANITY_SAMPLE_SIZE = 10  # tamanho da amostra determinística das verificações
HIST_BINS = 32  # número de bins do histograma de reflectância da figura


def normalization_stats_path(storage_paths: dict[str, Path]) -> Path:
    """Caminho canônico das estatísticas de normalização (data/processed/)."""
    return storage_paths["data_processed"] / "normalization_stats.json"


def eda_figure_file_name() -> str:
    """Nome estável da figura de resumo da EDA, derivado da configuração."""
    config = get_config()
    return f"eda_summary_{config['aoi']['region_code']}_{reference_year()}.png"


def _stats_fingerprint(storage_paths: dict[str, Path]) -> dict[str, Any]:
    """Fingerprint das estatísticas (schema, bandas e hash dos patches)."""
    config = get_config()
    return {
        "schema_version": EDA_SCHEMA_VERSION,
        "bands": list(config["data"]["bands"]),
        "fingerprint_hash": patch_fingerprint_hash(storage_paths),
    }


def _require_manifest(storage_paths: dict[str, Path]) -> None:
    """Falha com mensagem clara quando o manifesto do estágio 06 está ausente."""
    from src import io

    manifest = manifest_path(storage_paths)
    if not io.path_exists(manifest):
        raise FileNotFoundError(f"Manifesto do estágio 06 não encontrado: {manifest}")


def _load_manifest(storage_paths: dict[str, Path]) -> pd.DataFrame:
    """Carrega o manifesto persistido e valida que não está vazio."""
    from src import io

    local_manifest = io.ensure_local_copy(manifest_path(storage_paths))
    manifest = pd.read_parquet(local_manifest)
    if manifest.empty:
        raise ValueError("Manifesto vazio; execute o estágio 06 antes.")
    return manifest


def _deterministic_sample(manifest: pd.DataFrame, size: int) -> pd.DataFrame:
    """Amostra determinística de patches para as verificações e a figura."""
    return manifest.sample(n=min(size, len(manifest)), random_state=0)


def _finite_or_none(value: float) -> float | None:
    """Converte o valor em float, ou None quando não é finito (ex.: banda toda NaN)."""
    return float(value) if np.isfinite(value) else None


def normalization_stats_is_current(storage_paths: dict[str, Path]) -> bool:
    """Indica se as estatísticas persistidas estão vigentes frente aos patches."""
    from src import io

    stats_path = normalization_stats_path(storage_paths)
    if not io.path_exists(stats_path):
        return False
    local_stats = io.ensure_local_copy(stats_path)
    try:
        stored = json.loads(local_stats.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return stored.get("fingerprint") == _stats_fingerprint(storage_paths)


def compute_normalization_stats(
    storage_paths: dict[str, Path],
) -> dict[str, Any]:
    """Garante as estatísticas de normalização por banda (idempotente).

    Reutiliza o arquivo persistido quando o fingerprint vigente coincide com o
    armazenado; caso contrário, percorre todos os patches do manifesto e acumula
    soma, soma dos quadrados, contagem de pixels finitos e de NaN por banda,
    derivando média, desvio padrão e fração de NaN.
    """
    from src import io

    stats_path = normalization_stats_path(storage_paths)
    if normalization_stats_is_current(storage_paths):
        print(f"Estatísticas já existentes e atuais (reutilizadas): {stats_path}")
        local_stats = io.ensure_local_copy(stats_path)
        return json.loads(local_stats.read_text(encoding="utf-8"))

    _require_manifest(storage_paths)
    manifest = _load_manifest(storage_paths)
    config = get_config()
    bands = list(config["data"]["bands"])
    n_bands = len(bands)

    sums = np.zeros(n_bands, dtype=np.float64)
    sums_sq = np.zeros(n_bands, dtype=np.float64)
    counts = np.zeros(n_bands, dtype=np.float64)
    nan_counts = np.zeros(n_bands, dtype=np.float64)
    total_counts = np.zeros(n_bands, dtype=np.float64)

    for record in manifest.itertuples():
        image = np.load(
            io.ensure_local_copy(resolve_from_root(str(record.image_path), storage_paths))
        ).reshape(n_bands, -1)
        finite = np.isfinite(image)
        total_counts += image.shape[1]
        nan_counts += (~finite).sum(axis=1)
        values = np.where(finite, image, 0.0)
        sums += values.sum(axis=1)
        sums_sq += (values * values).sum(axis=1)
        counts += finite.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
        variances = sums_sq / counts - means**2
    stds = np.sqrt(np.maximum(variances, 0.0))

    per_band: dict[str, dict[str, float | None]] = {}
    for index, band in enumerate(bands):
        per_band[band] = {
            "mean": _finite_or_none(float(means[index])),
            "std": _finite_or_none(float(stds[index])),
            "nan_fraction": float(nan_counts[index] / total_counts[index]),
        }

    payload = {
        "schema_version": EDA_SCHEMA_VERSION,
        "fingerprint": _stats_fingerprint(storage_paths),
        "bands": bands,
        "n_patches": int(len(manifest)),
        "per_band": per_band,
    }
    io.persist_bytes(
        stats_path,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    print(f"Estatísticas de normalização salvas em: {stats_path}")
    return payload


def run_sanity_checks(
    storage_paths: dict[str, Path],
) -> dict[str, Any]:
    """Executa verificações de sanidade do dataset a partir do manifesto.

    Valida o manifesto (contagens, café, dobras, unicidade) e uma amostra
    determinística de patches (forma esperada, dtype, máscara binária, intervalo
    de valores e fração de NaN), acumulando uma lista de problemas encontrados.
    """
    from src import io

    _require_manifest(storage_paths)
    manifest = _load_manifest(storage_paths)
    config = get_config()
    bands = list(config["data"]["bands"])
    patch_size = int(config["data"]["patch_size"])
    expected_shape = (len(bands), patch_size, patch_size)

    sample = _deterministic_sample(manifest, SANITY_SAMPLE_SIZE)
    shape_issues: list[str] = []
    dtypes: set[str] = set()
    mask_values: set[int] = set()
    value_min = float("inf")
    value_max = float("-inf")
    nan_pixels = 0
    for record in sample.itertuples():
        image = np.load(
            io.ensure_local_copy(resolve_from_root(str(record.image_path), storage_paths))
        )
        mask = np.load(
            io.ensure_local_copy(resolve_from_root(str(record.mask_path), storage_paths))
        )
        if image.shape != expected_shape:
            shape_issues.append(f"{record.patch_id}: {image.shape}")
        dtypes.add(str(image.dtype))
        mask_values.update(np.unique(mask).astype(int).tolist())
        finite = image[np.isfinite(image)]
        if finite.size:
            value_min = min(value_min, float(finite.min()))
            value_max = max(value_max, float(finite.max()))
        nan_pixels += int(np.isnan(image).sum())

    coffee = manifest["coffee_ratio"]
    fold_present = "fold" in manifest.columns
    issues: list[str] = []
    fold_distribution: dict[str, int] = {}
    if fold_present:
        if manifest["fold"].isna().any():
            issues.append("coluna fold com valores ausentes")
        for fold, count in manifest["fold"].value_counts().sort_index().items():
            fold_distribution[str(int(fold))] = int(count)
    if shape_issues:
        issues.append(f"formas divergentes do esperado: {shape_issues}")
    if not mask_values.issubset({0, 1}):
        issues.append(f"máscara com valores não binários: {sorted(mask_values)}")

    sample_size = int(len(sample))
    sample_total_pixels = sample_size * len(bands) * patch_size**2

    return {
        "n_patches": int(len(manifest)),
        "n_tiles": int(manifest["tile_id"].nunique()),
        "unique_patch_ids": int(manifest["patch_id"].nunique()),
        "mask_sources": sorted(manifest["mask_source"].dropna().unique().tolist()),
        "tile_ids": sorted(manifest["tile_id"].dropna().unique().tolist()),
        "coffee_ratio": {
            "min": float(coffee.min()),
            "mean": float(coffee.mean()),
            "max": float(coffee.max()),
        },
        "fold": {
            "present": fold_present,
            "distribution": fold_distribution,
        },
        "sample": {
            "size": sample_size,
            "expected_shape": list(expected_shape),
            "shape_issues": shape_issues,
            "dtypes": sorted(dtypes),
            "mask_values": sorted(mask_values),
            "value_range": [
                _finite_or_none(value_min),
                _finite_or_none(value_max),
            ],
            "nan_fraction": float(nan_pixels / sample_total_pixels),
        },
        "issues": issues,
    }


def save_eda_figure(storage_paths: dict[str, Path]) -> Path:
    """Persiste a figura de resumo da EDA no Drive canônico (idempotente)."""
    from src import io

    figure_path = storage_paths["artifacts_figures"] / eda_figure_file_name()
    if io.path_exists(figure_path):
        print(f"Figura já existente (reutilizada): {figure_path}")
        return figure_path

    manifest = _load_manifest(storage_paths)
    stats = compute_normalization_stats(storage_paths)
    io.persist_bytes(figure_path, render_eda_figure(manifest, stats, storage_paths))
    print(f"Figura salva em: {figure_path}")
    return figure_path


def render_eda_figure(
    manifest: pd.DataFrame,
    stats: dict[str, Any],
    storage_paths: dict[str, Path],
) -> bytes:
    """Renderiza a figura de resumo: café, dobras, média±desvio e histogramas."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src import io

    bands = list(get_config()["data"]["bands"])
    sample = _deterministic_sample(manifest, SANITY_SAMPLE_SIZE)
    per_band = stats["per_band"]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(manifest["coffee_ratio"], bins=20, color="#7b1fa2", alpha=0.8)
    axes[0, 0].set_title("Distribuição da proporção de café por patch")
    axes[0, 0].set_xlabel("coffee_ratio")
    axes[0, 0].set_ylabel("patches")

    if "fold" in manifest.columns and manifest["fold"].notna().any():
        counts = manifest["fold"].value_counts().sort_index()
        axes[0, 1].bar(
            [str(int(fold)) for fold in counts.index],
            counts.values,
            color="#2e7d32",
        )
        axes[0, 1].set_title("Balanceamento por dobra (k-fold)")
        axes[0, 1].set_xlabel("dobra")
        axes[0, 1].set_ylabel("patches")
    else:
        axes[0, 1].text(0.5, 0.5, "coluna fold ausente", ha="center", va="center")
        axes[0, 1].set_axis_off()

    band_names = list(per_band)
    x = np.arange(len(band_names))
    means = [per_band[band]["mean"] or 0.0 for band in band_names]
    stds = [per_band[band]["std"] or 0.0 for band in band_names]
    axes[1, 0].bar(x, means, yerr=stds, capsize=4, color="#1565c0", alpha=0.85)
    axes[1, 0].set_xticks(x, band_names)
    axes[1, 0].set_title("Média ± desvio padrão por banda")
    axes[1, 0].set_ylabel("reflectância")

    for index, band in enumerate(bands):
        values: list[np.ndarray] = []
        for record in sample.itertuples():
            image = np.load(
                io.ensure_local_copy(resolve_from_root(str(record.image_path), storage_paths))
            )
            band_values = image[index]
            values.append(band_values[np.isfinite(band_values)])
        if values:
            axes[1, 1].hist(
                np.concatenate(values),
                bins=HIST_BINS,
                alpha=0.5,
                label=band,
            )
    axes[1, 1].legend()
    axes[1, 1].set_title("Histograma de reflectância por banda (amostra)")

    figure.suptitle("Análise exploratória do dataset de patches", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    buffer = stdlib_io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    return buffer.getvalue()
