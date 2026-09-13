"""Testes dos helpers puros de construção de máscaras de referência."""

from __future__ import annotations

import numpy as np
from src.data.mask_utils import binarize_array, mapbiomas_band_candidates


def test_mapbiomas_band_candidates_order() -> None:
    assert mapbiomas_band_candidates("classification", 2023) == [
        "classification_2023",
        "classification",
    ]


def test_binarize_array_marks_only_target_class() -> None:
    array = np.array([[46, 0, 46], [3, 46, 1]], dtype=np.int16)
    result = binarize_array(array, 46)
    assert result.dtype == np.uint8
    assert result.tolist() == [[1, 0, 1], [0, 1, 0]]


def test_binarize_array_without_target_returns_zeros() -> None:
    array = np.array([[1, 2], [3, 4]], dtype=np.int16)
    assert binarize_array(array, 46).sum() == 0
