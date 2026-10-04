"""Diagnóstico comparativo das máscaras de café por fonte de ground truth (estágio 03).

Quantifica a concordância entre as fontes habilitadas (MapBiomas, AlphaEarth)
sobre o grid comum de 10 m: área de café por fonte, sobreposição, discordância e
métricas de concordância (acordo global, IoU, F1, precisão, recall e kappa).
Produz o relatório JSON e a figura de apoio à escolha da fonte no estágio 04.
O processamento é determinístico e as persistências são idempotentes.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_config
from src.data.mask_utils import reference_year


@dataclass(frozen=True)
class MaskSet:
    """Máscaras binárias carregadas em um grid comum, com metadados do grid."""

    masks: dict[str, np.ndarray]
    crs: str
    shape: tuple[int, int]
    pixel_size_m: float


def load_masks(mask_paths: dict[str, Path]) -> MaskSet:
    """Carrega as máscaras binárias em um grid comum (o da primeira fonte).

    A primeira fonte define o grid de referência; as demais são reamostradas
    (vizinho mais próximo) para esse grid quando necessário, garantindo arrays
    sobrepostos pixel a pixel para o cálculo de concordância.
    """
    import rasterio

    names = list(mask_paths)
    reference = names[0]
    with rasterio.open(mask_paths[reference]) as src:
        transform = src.transform
        crs = src.crs
        shape = (int(src.height), int(src.width))
        masks = {reference: src.read(1) > 0}
    for name in names[1:]:
        masks[name] = _read_on_reference_grid(mask_paths[name], shape, transform, crs)
    return MaskSet(
        masks=masks,
        crs=str(crs),
        shape=shape,
        pixel_size_m=abs(float(transform.a)),
    )


def _read_on_reference_grid(
    path: Path,
    shape: tuple[int, int],
    transform: Any,
    crs: Any,
) -> np.ndarray:
    """Lê uma máscara no grid de referência, reamostrando quando for necessário."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(path) as src:
        if (src.height, src.width) == shape and src.transform == transform and src.crs == crs:
            return src.read(1) > 0
        destination = np.zeros(shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
    return destination > 0


def native_resolution_m(source_name: str) -> int:
    """Resolução nativa (m) da fonte de ground truth, declarada em config.yaml.

    Fontes com resolução diferente da escala de processamento (10 m) são
    reamostradas na exportação do GEE (nearest) para o grid do Sentinel-2.
    """
    source_cfg = get_config()["ground_truth"]["sources"].get(source_name, {})
    return int(source_cfg.get("native_resolution_m", 10))


def compute_comparison(mask_set: MaskSet) -> dict[str, Any]:
    """Calcula o diagnóstico comparativo entre as fontes (áreas e concordância).

    Cruza todas as fontes habilitadas entre si (todos os pares, sem fonte de
    referência fixa): a matriz de confusão de cada par e as métricas simétricas
    (acordo global, IoU, F1, kappa) são calculadas para cada combinação. Nas
    métricas direcionais (precisão e recall), o ponto de vista é a primeira
    fonte do par (ordem do config.yaml); o diagnóstico não elege fonte alguma.
    """
    import itertools

    names = list(mask_set.masks)
    if len(names) < 2:
        raise ValueError("O diagnóstico comparativo exige pelo menos duas fontes.")
    total = int(np.prod(mask_set.shape))

    result: dict[str, Any] = {
        "sources": names,
        "years": {name: reference_year(name) for name in names},
        "native_resolution_m": {name: native_resolution_m(name) for name in names},
        "grid": {
            "shape": list(mask_set.shape),
            "crs": mask_set.crs,
            "pixel_size_m": mask_set.pixel_size_m,
            "total_pixels": total,
            "total_area_km2": total * mask_set.pixel_size_m**2 / 1e6,
        },
        "area_km2": {},
        "coffee_share": {},
        "pairs": {},
    }

    for name in names:
        count = int(np.count_nonzero(mask_set.masks[name]))
        result["area_km2"][name] = count * mask_set.pixel_size_m**2 / 1e6
        result["coffee_share"][name] = count / total

    for first, other in itertools.combinations(names, 2):
        a = mask_set.masks[first]
        b = mask_set.masks[other]
        both = int(np.count_nonzero(a & b))
        only_first = int(np.count_nonzero(a & ~b))
        only_other = int(np.count_nonzero(~a & b))
        neither = total - both - only_first - only_other
        result["pairs"][f"{first}_vs_{other}"] = {
            "confusion_pixels": {
                "both_coffee": both,
                "only_reference": only_first,
                "only_other": only_other,
                "both_non_coffee": neither,
            },
            "metrics": {
                "overall_agreement": _safe_ratio(both + neither, total),
                "iou": _safe_ratio(both, both + only_first + only_other),
                "f1": _safe_ratio(2 * both, 2 * both + only_first + only_other),
                "precision": _safe_ratio(both, both + only_first),
                "recall": _safe_ratio(both, both + only_other),
                "kappa": _cohen_kappa(both, only_first, only_other, neither, total),
            },
        }
    return result


def report_file_name() -> str:
    """Nome estável do relatório JSON do diagnóstico (multianual).

    As fontes comparadas podem ter anos de referência distintos (ex.: MapBiomas
    2025, AlphaEarth 2024, Emater 2018); o nome do relatório é, portanto,
    independente de um ano único.
    """
    region_code = get_config()["aoi"]["region_code"]
    return f"mask_sources_comparison_{region_code}.json"


def figure_file_name() -> str:
    """Nome estável da figura do diagnóstico (multianual)."""
    region_code = get_config()["aoi"]["region_code"]
    return f"mask_sources_comparison_{region_code}.png"


def save_report(result: dict[str, Any], storage_paths: dict[str, Path]) -> Path:
    """Persiste o relatório JSON do diagnóstico no Drive canônico (idempotente)."""
    from src import io

    report_path = storage_paths["artifacts_metrics_ground_truth"] / report_file_name()
    if io.path_exists(report_path):
        print(f"Relatório já existente (reutilizado): {report_path}")
        return report_path
    payload = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
    io.persist_bytes(report_path, payload)
    print(f"Relatório salvo em: {report_path}")
    return report_path


def save_figure(
    mask_set: MaskSet,
    storage_paths: dict[str, Path],
) -> Path:
    """Persiste a figura do diagnóstico no Drive canônico (idempotente)."""
    from src import io

    figure_path = storage_paths["artifacts_figures"] / figure_file_name()
    if io.path_exists(figure_path):
        print(f"Figura já existente (reutilizada): {figure_path}")
        return figure_path
    io.persist_bytes(figure_path, render_comparison_figure(mask_set))
    print(f"Figura salva em: {figure_path}")
    return figure_path


def render_comparison_figure(mask_set: MaskSet) -> bytes:
    """Renderiza a figura do diagnóstico: todas as fontes + todos os pares.

    A primeira linha exibe a máscara de cada fonte (com o ano de referência);
    as linhas seguintes mostram o mapa de concordância de cada par de fontes,
    permitindo comparar todas as combinações, não apenas um par.
    """
    from src.data.raster_utils import render_figure

    names = list(mask_set.masks)
    if len(names) < 2:
        raise ValueError("O diagnóstico comparativo exige pelo menos duas fontes.")
    pairs = list(itertools.combinations(names, 2))
    n_cols = len(names)
    n_rows = 1 + len(pairs)

    def _build(plt: Any) -> Any:
        figure, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.4 * n_rows))
        axes = np.atleast_2d(axes)
        for col, name in enumerate(names):
            _plot_mask(
                axes[0, col],
                display_array(mask_set.masks[name]),
                f"{name} ({reference_year(name)})",
            )
        for row, (first, other) in enumerate(pairs, start=1):
            _plot_agreement(
                axes[row, 0],
                display_array(agreement_labels(mask_set.masks[first], mask_set.masks[other])),
                first,
                other,
            )
            for col in range(1, n_cols):
                axes[row, col].set_axis_off()
        figure.suptitle("Comparação das máscaras de café por fonte de ground truth", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        return figure

    return render_figure(_build)


def agreement_labels(reference_mask: np.ndarray, other_mask: np.ndarray) -> np.ndarray:
    """Mapa de concordância rotulado: 0 nenhum, 1 ambos, 2 só referência, 3 só outra."""
    combined = 2 * reference_mask.astype(np.uint8) + other_mask.astype(np.uint8)
    labels = np.zeros(combined.shape, dtype=np.uint8)
    labels[combined == 3] = 1
    labels[combined == 2] = 2
    labels[combined == 1] = 3
    return labels


COFFEE_COLORS = ["#f2f2f2", "#7b1fa2"]


def display_array(mask: np.ndarray, max_dim: int = 4096) -> np.ndarray:
    """Reduz o array para a figura quando a maior dimensão excede o limite."""
    height, width = mask.shape
    stride = max(1, max(height, width) // max_dim)
    if stride == 1:
        return mask
    return mask[::stride, ::stride]


def _plot_mask(axis: Any, mask: np.ndarray, title: str) -> None:
    """Exibe uma máscara binária (café em roxo sobre fundo claro)."""
    from matplotlib.colors import ListedColormap

    axis.imshow(mask, cmap=ListedColormap(COFFEE_COLORS), vmin=0, vmax=1)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def _plot_agreement(axis: Any, labels: np.ndarray, reference: str, other: str) -> None:
    """Exibe o mapa de concordância com as quatro classes e a legenda."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    colors = ["#f2f2f2", "#2e7d32", "#f9a825", "#1565c0"]
    axis.imshow(labels, cmap=ListedColormap(colors), vmin=0, vmax=3)
    axis.set_title("Concordância")
    axis.set_xticks([])
    axis.set_yticks([])
    legend = [
        Patch(facecolor=colors[1], label="Café (ambas)"),
        Patch(facecolor=colors[2], label=f"Somente {reference}"),
        Patch(facecolor=colors[3], label=f"Somente {other}"),
    ]
    axis.legend(handles=legend, loc="lower center", fontsize=8, ncol=3)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """Retorna a razão, ou None quando o denominador é nulo ou o valor não é finito."""
    if denominator == 0:
        return None
    value = numerator / denominator
    return float(value) if np.isfinite(value) else None


def _cohen_kappa(
    both: int,
    only_reference: int,
    only_other: int,
    neither: int,
    total: int,
) -> float | None:
    """Coeficiente kappa de Cohen para a tabela 2x2 das duas fontes."""
    agreement = (both + neither) / total
    reference_total = both + only_reference
    other_total = both + only_other
    chance = (
        reference_total * other_total + (only_other + neither) * (only_reference + neither)
    ) / total**2
    return _safe_ratio(agreement - chance, 1 - chance)
