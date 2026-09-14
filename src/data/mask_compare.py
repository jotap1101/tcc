"""Diagnóstico comparativo de fontes de máscaras de referência.

Funções puras (apenas ``numpy`` e ``pandas``, ambos dependências base) que
quantificam a concordância entre máscaras candidatas e a referência, além de
auxiliares para binarizar o mapa de clusters do k-means.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.gee_client import make_export_description

# Nomes de métricas avaliadas em nível de pixel (alinhadas a ``evaluation``).
METRIC_COLUMNS: tuple[str, ...] = ("iou", "f1", "precision", "recall", "kappa")


def mask_candidate_filename(
    source: str,
    prefix: str,
    region_code: str,
    start_date: str,
    end_date: str,
) -> str:
    """Nome esperado do GeoTIFF exportado para uma fonte candidata."""
    description = make_export_description(
        f"{prefix}_{source}", region_code, start_date, end_date
    )
    return f"{description}.tif"


def binary_confusion(reference: Any, candidate: Any) -> dict[str, int]:
    """Matriz de confusão 2x2 (tp, fp, fn, tn) entre dois arranjos binários."""
    ref = np.asarray(reference).astype(bool)
    cand = np.asarray(candidate).astype(bool)
    if ref.shape != cand.shape:
        raise ValueError(
            f"Formas incompatíveis: referência {ref.shape} vs candidato {cand.shape}"
        )
    return {
        "tp": int(np.logical_and(cand, ref).sum()),
        "fp": int(np.logical_and(cand, ~ref).sum()),
        "fn": int(np.logical_and(~cand, ref).sum()),
        "tn": int(np.logical_and(~cand, ~ref).sum()),
    }


def _denominator(c: dict[str, int], positives: int, negatives: int) -> int:
    return c[positives] + c[negatives]


def iou_from_confusion(c: dict[str, int]) -> float:
    """IoU = tp / (tp + fp + fn)."""
    denom = _denominator(c, "tp", "fp") + c["fn"]
    return float(c["tp"] / denom) if denom else 0.0


def f1_from_confusion(c: dict[str, int]) -> float:
    """F1 = 2*tp / (2*tp + fp + fn)."""
    denom = 2 * c["tp"] + c["fp"] + c["fn"]
    return float(2 * c["tp"] / denom) if denom else 0.0


def precision_from_confusion(c: dict[str, int]) -> float:
    """Precision = tp / (tp + fp)."""
    denom = _denominator(c, "tp", "fp")
    return float(c["tp"] / denom) if denom else 0.0


def recall_from_confusion(c: dict[str, int]) -> float:
    """Recall = tp / (tp + fn)."""
    denom = _denominator(c, "tp", "fn")
    return float(c["tp"] / denom) if denom else 0.0


def cohen_kappa_from_confusion(c: dict[str, int]) -> float:
    """Coeficiente Kappa de Cohen, corrigindo a concordância ao acaso."""
    total = c["tp"] + c["fp"] + c["fn"] + c["tn"]
    if total == 0:
        return 0.0
    observed = (c["tp"] + c["tn"]) / total
    expected = (
        (c["tp"] + c["fp"]) * (c["tp"] + c["fn"])
        + (c["tn"] + c["fn"]) * (c["tn"] + c["fp"])
    ) / (total * total)
    if expected == 1.0:
        return 0.0
    return float((observed - expected) / (1 - expected))


def metrics_from_confusion(c: dict[str, int]) -> dict[str, float]:
    """Dicionário com todas as métricas derivadas da confusão."""
    return {
        "iou": iou_from_confusion(c),
        "f1": f1_from_confusion(c),
        "precision": precision_from_confusion(c),
        "recall": recall_from_confusion(c),
        "kappa": cohen_kappa_from_confusion(c),
    }


def compare_sources(arrays: dict[str, Any], reference_name: str) -> pd.DataFrame:
    """Tabela de métricas de cada fonte candidata em relação à referência."""
    if reference_name not in arrays:
        raise ValueError(f"Referência ausente em arrays: {reference_name}")
    reference = arrays[reference_name]

    rows: list[dict[str, Any]] = []
    for name, candidate in arrays.items():
        if name == reference_name:
            continue
        confusion = binary_confusion(reference, candidate)
        row = {"source": name}
        row.update(metrics_from_confusion(confusion))
        rows.append(row)
    return pd.DataFrame(rows, columns=["source", *METRIC_COLUMNS])


def cluster_prevalence(clusters: Any, reference: Any) -> dict[int, float]:
    """Prevalência de café por cluster, com base na referência binária."""
    cl = np.asarray(clusters)
    ref = np.asarray(reference).astype(bool)
    prevalence: dict[int, float] = {}
    for label in np.unique(cl):
        mask = cl == label
        prevalence[int(label)] = float(ref[mask].mean()) if mask.any() else 0.0
    return prevalence


def threshold_clusters(
    clusters: Any,
    prevalence: dict[int, float],
    threshold: float = 0.5,
) -> np.ndarray:
    """Binariza o mapa de clusters: clusters com prevalência >= limiar viram café."""
    cl = np.asarray(clusters)
    binary = np.zeros_like(cl, dtype=np.uint8)
    for label, ratio in prevalence.items():
        if ratio >= threshold:
            binary[cl == label] = 1
    return binary


def coffee_ratio(mask: Any) -> float:
    """Proporção de pixels de café (1) em uma máscara binária."""
    arr = np.asarray(mask)
    return float(arr.mean()) if arr.size else 0.0
