"""Helpers compartilhados de raster e de renderização de figuras.

Reúne operações usadas por vários módulos de `src/data`: comparação de grids
(sobreposição pixel a pixel), estiramento de percentis para visualização e o
boilerplate de renderização matplotlib (backend Agg + buffer PNG), evitando a
duplicação entre os estágios.
"""

from __future__ import annotations

import io as stdlib_io
from collections.abc import Callable
from typing import Any

import numpy as np


def same_grid(first: Any, second: Any) -> bool:
    """Indica se dois rasters compartilham CRS, transform e dimensões."""
    return (
        first.height == second.height
        and first.width == second.width
        and first.transform == second.transform
        and first.crs == second.crs
    )


def percentile_stretch(band: np.ndarray, low: int = 2, high: int = 98) -> np.ndarray:
    """Realça uma banda contínua pelo estiramento de percentis para [0, 255]."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    p_low, p_high = np.percentile(finite, [low, high])
    if p_high <= p_low:
        p_high = p_low + 1e-6
    stretched = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)
    return (stretched * 255).astype(np.uint8)


def render_figure(build_fn: Callable[[Any], Any]) -> bytes:
    """Renderiza uma figura matplotlib como bytes PNG (backend Agg, sem display).

    ``build_fn`` recebe o módulo ``matplotlib.pyplot`` e retorna a figura já
    montada (títulos, layout e legendas); a persistência em buffer e o
    fechamento da figura são centralizados aqui.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = build_fn(plt)
    buffer = stdlib_io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    return buffer.getvalue()
