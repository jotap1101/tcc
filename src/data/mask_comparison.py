"""Diagnóstico comparativo das máscaras de café por fonte de ground truth (estágio 03).

Quantifica a concordância entre as fontes habilitadas (MapBiomas, AlphaEarth)
sobre o grid comum de 10 m: área de café por fonte, sobreposição, discordância e
métricas de concordância (acordo global, IoU, F1, precisão, recall e kappa).
Produz o relatório JSON e a figura de apoio à escolha da fonte no estágio 04.
O processamento é determinístico e as persistências são idempotentes.
"""

from __future__ import annotations

import io as stdlib_io
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


def compute_comparison(mask_set: MaskSet, reference: str) -> dict[str, Any]:
    """Calcula o diagnóstico comparativo entre as fontes (áreas e concordância).

    A fonte de referência é o ponto de vista das métricas direcionais (precisão
    e recall); acordo global, IoU, F1 e kappa são simétricos entre as fontes.
    """
    names = list(mask_set.masks)
    if reference not in names:
        raise ValueError(f"Fonte de referência inválida: {reference}")
    total = int(np.prod(mask_set.shape))

    result: dict[str, Any] = {
        "reference_source": reference,
        "sources": names,
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

    for other in names:
        if other == reference:
            continue
        a = mask_set.masks[reference]
        b = mask_set.masks[other]
        both = int(np.count_nonzero(a & b))
        only_reference = int(np.count_nonzero(a & ~b))
        only_other = int(np.count_nonzero(~a & b))
        neither = total - both - only_reference - only_other
        result["pairs"][f"{reference}_vs_{other}"] = {
            "confusion_pixels": {
                "both_coffee": both,
                "only_reference": only_reference,
                "only_other": only_other,
                "both_non_coffee": neither,
            },
            "metrics": {
                "overall_agreement": _safe_ratio(both + neither, total),
                "iou": _safe_ratio(both, both + only_reference + only_other),
                "f1": _safe_ratio(2 * both, 2 * both + only_reference + only_other),
                "precision": _safe_ratio(both, both + only_reference),
                "recall": _safe_ratio(both, both + only_other),
                "kappa": _cohen_kappa(both, only_reference, only_other, neither, total),
            },
        }
    return result


def report_file_name() -> str:
    """Nome estável do relatório JSON do diagnóstico, derivado da configuração."""
    region_code = get_config()["aoi"]["region_code"]
    return f"mask_sources_comparison_{region_code}_{reference_year()}.json"


def figure_file_name() -> str:
    """Nome estável da figura do diagnóstico, derivado da configuração."""
    region_code = get_config()["aoi"]["region_code"]
    return f"mask_sources_comparison_{region_code}_{reference_year()}.png"


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
    reference: str,
    storage_paths: dict[str, Path],
) -> Path:
    """Persiste a figura do diagnóstico no Drive canônico (idempotente)."""
    from src import io

    figure_path = storage_paths["artifacts_figures"] / figure_file_name()
    if io.path_exists(figure_path):
        print(f"Figura já existente (reutilizada): {figure_path}")
        return figure_path
    io.persist_bytes(figure_path, render_comparison_figure(mask_set, reference))
    print(f"Figura salva em: {figure_path}")
    return figure_path


def render_comparison_figure(mask_set: MaskSet, reference: str) -> bytes:
    """Renderiza a figura do diagnóstico (fontes individuais + concordância)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(mask_set.masks)
    others = [name for name in names if name != reference]
    if not others:
        raise ValueError("O diagnóstico comparativo exige pelo menos duas fontes.")
    other = others[0]

    reference_mask = display_array(mask_set.masks[reference])
    other_mask = display_array(mask_set.masks[other])
    agreement = display_array(agreement_labels(mask_set.masks[reference], mask_set.masks[other]))

    figure, axes = plt.subplots(1, 3, figsize=(16, 6))
    _plot_mask(axes[0], reference_mask, reference)
    _plot_mask(axes[1], other_mask, other)
    _plot_agreement(axes[2], agreement, reference, other)
    figure.suptitle("Comparação das máscaras de café por fonte de ground truth", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    buffer = stdlib_io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    return buffer.getvalue()


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
