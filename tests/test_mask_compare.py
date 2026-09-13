"""Testes do diagnóstico comparativo de fontes de máscaras."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.mask_compare import (
    binary_confusion,
    cluster_prevalence,
    coffee_ratio,
    compare_sources,
    cohen_kappa_from_confusion,
    f1_from_confusion,
    iou_from_confusion,
    mask_candidate_filename,
    precision_from_confusion,
    recall_from_confusion,
    threshold_clusters,
)


def test_binary_confusion_counts() -> None:
    reference = np.array([[1, 0], [1, 1]])
    candidate = np.array([[1, 1], [0, 1]])
    assert binary_confusion(reference, candidate) == {
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "tn": 0,
    }


def test_binary_confusion_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        binary_confusion(np.zeros((2, 2)), np.zeros((3, 3)))


def test_metrics_on_known_confusion() -> None:
    confusion = {"tp": 2, "fp": 1, "fn": 1, "tn": 0}
    assert iou_from_confusion(confusion) == pytest.approx(0.5)
    assert f1_from_confusion(confusion) == pytest.approx(2 / 3)
    assert precision_from_confusion(confusion) == pytest.approx(2 / 3)
    assert recall_from_confusion(confusion) == pytest.approx(2 / 3)
    assert cohen_kappa_from_confusion(confusion) == pytest.approx(-1 / 3)


def test_kappa_perfect_agreement_is_one() -> None:
    confusion = {"tp": 3, "fp": 0, "fn": 0, "tn": 1}
    assert cohen_kappa_from_confusion(confusion) == pytest.approx(1.0)


def test_compare_sources_returns_metrics_table() -> None:
    arrays = {
        "mapbiomas_coffee": np.array([[1, 0], [1, 1]]),
        "alphaearth_clusters": np.array([[1, 1], [0, 1]]),
    }
    frame = compare_sources(arrays, reference_name="mapbiomas_coffee")
    assert list(frame["source"]) == ["alphaearth_clusters"]
    assert frame.loc[0, "iou"] == pytest.approx(0.5)
    assert set(frame.columns) >= {"source", "iou", "f1", "kappa"}


def test_compare_sources_requires_reference() -> None:
    with pytest.raises(ValueError):
        compare_sources({"cand": np.zeros((2, 2))}, reference_name="ref")


def test_cluster_prevalence_by_label() -> None:
    clusters = np.array([[0, 1], [1, 2]])
    reference = np.array([[1, 0], [0, 1]])
    assert cluster_prevalence(clusters, reference) == {0: 1.0, 1: 0.0, 2: 1.0}


def test_threshold_clusters_binarizes() -> None:
    clusters = np.array([[0, 1], [1, 2]])
    prevalence = {0: 1.0, 1: 0.0, 2: 1.0}
    binary = threshold_clusters(clusters, prevalence, threshold=0.5)
    assert binary.tolist() == [[1, 0], [0, 1]]
    assert binary.dtype == np.uint8


def test_coffee_ratio() -> None:
    assert coffee_ratio(np.array([[1, 0], [1, 1]])) == pytest.approx(0.75)
    assert coffee_ratio(np.zeros((2, 2))) == 0.0


def test_mask_candidate_filename() -> None:
    assert (
        mask_candidate_filename(
            "mapbiomas_coffee", "guaxupe", "310044", "2023-06-01", "2023-09-30"
        )
        == "guaxupe_mapbiomas_coffee_310044_20230601_20230930.tif"
    )