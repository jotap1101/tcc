"""Finalização da máscara binária de café a partir da fonte escolhida (estágio 04).

Seleciona a fonte definitiva registrada em `ground_truth.chosen_source` do
config.yaml, valida a escolha contra o relatório comparativo do estágio 03 e
persiste a máscara binária final em MyDrive/tcc/data/processed/ground_truth/,
além do relatório de finalização e da figura de registro. As persistências são
idempotentes — reexecuções reutilizam a máscara, o relatório e a figura existentes.
"""

from __future__ import annotations

import io as stdlib_io
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_config
from src.data.mask_comparison import COFFEE_COLORS, display_array, report_file_name
from src.data.mask_utils import reference_year, source_mask_path


def chosen_source() -> str:
    """Fonte definitiva de ground truth, registrada em config.yaml."""
    config = get_config()
    chosen = config["ground_truth"].get("chosen_source")
    if not chosen:
        raise ValueError("Defina ground_truth.chosen_source em src/config.yaml.")
    enabled = [name for name, cfg in config["ground_truth"]["sources"].items() if cfg["enabled"]]
    if chosen not in enabled:
        raise ValueError(f"Fonte escolhida inválida ou desabilitada: {chosen}")
    return chosen


def final_mask_file_name() -> str:
    """Nome estável da máscara binária final, derivado da configuração."""
    config = get_config()
    return f"mask_final_{config['aoi']['region_code']}_{reference_year()}"


def final_mask_path(storage_paths: dict[str, Path]) -> Path:
    """Caminho canônico da máscara binária final (data/processed/ground_truth/)."""
    return storage_paths["data_processed_ground_truth"] / f"{final_mask_file_name()}.tif"


def finalization_report_file_name() -> str:
    """Nome estável do relatório de finalização, derivado da configuração."""
    config = get_config()
    return f"mask_finalization_{config['aoi']['region_code']}_{reference_year()}.json"


def load_comparison_report(storage_paths: dict[str, Path]) -> dict[str, Any]:
    """Carrega o relatório comparativo do estágio 03 (reutilizado, idempotente)."""
    from src import io

    report_path = storage_paths["artifacts_metrics_ground_truth"] / report_file_name()
    if not io.path_exists(report_path):
        raise FileNotFoundError(f"Relatório do estágio 03 não encontrado: {report_path}")
    local_path = io.ensure_local_copy(report_path)
    return json.loads(local_path.read_text(encoding="utf-8"))


def ensure_final_mask(chosen: str, storage_paths: dict[str, Path]) -> Path:
    """Garante a máscara binária final no caminho canônico (cópia idempotente).

    Se a máscara final já existir, é reutilizada; caso contrário, a máscara da
    fonte escolhida (estágio 02) é copiada para data/processed/ground_truth/.
    """
    from src import io

    target_path = final_mask_path(storage_paths)
    if io.path_exists(target_path):
        print(f"Máscara final já existente (reutilizada): {target_path}")
        return target_path

    source_path = source_mask_path(chosen, storage_paths)
    if not io.path_exists(source_path):
        raise FileNotFoundError(
            f"Máscara da fonte {chosen} (estágio 02) não encontrada: {source_path}"
        )
    local_source = io.ensure_local_copy(source_path)
    io.persist_bytes(target_path, local_source.read_bytes())
    print(f"Máscara final salva em: {target_path}")
    return target_path


def verify_final_mask(chosen: str, storage_paths: dict[str, Path]) -> dict[str, Any]:
    """Verifica a máscara final contra a fonte: mesmo grid, CRS e valores binários."""
    import rasterio

    from src import io

    final_path = io.ensure_local_copy(final_mask_path(storage_paths))
    source_path = io.ensure_local_copy(source_mask_path(chosen, storage_paths))

    with rasterio.open(final_path) as dst, rasterio.open(source_path) as src:
        if (dst.height, dst.width) != (src.height, src.width) or dst.crs != src.crs:
            raise ValueError("Máscara final com grid/CRS divergente da fonte.")
        final_array = dst.read(1)
        source_array = src.read(1)
        if not np.array_equal(final_array, source_array):
            raise ValueError("Máscara final com valores divergentes da fonte.")
        coffee_pixels = int(np.count_nonzero(final_array > 0))
        pixel_size_m = abs(float(dst.transform.a))
        shape = (int(dst.height), int(dst.width))
        crs = str(dst.crs)

    return {
        "shape": list(shape),
        "crs": crs,
        "coffee_pixels": coffee_pixels,
        "area_km2": coffee_pixels * pixel_size_m**2 / 1e6,
    }


def save_finalization_report(
    chosen: str,
    comparison: dict[str, Any],
    final_path: Path,
    storage_paths: dict[str, Path],
) -> Path:
    """Persiste o relatório JSON de finalização (idempotente)."""
    from src import io

    report_path = storage_paths["artifacts_metrics_ground_truth"] / finalization_report_file_name()
    if io.path_exists(report_path):
        print(f"Relatório já existente (reutilizado): {report_path}")
        return report_path

    config = get_config()
    payload = {
        "chosen_source": chosen,
        "region_code": config["aoi"]["region_code"],
        "year": reference_year(),
        "comparison_report": report_file_name(),
        "sources": comparison["sources"],
        "area_km2": comparison["area_km2"],
        "coffee_share": comparison["coffee_share"],
        "pairs": comparison["pairs"],
        "final_mask_path": str(final_path),
    }
    io.persist_bytes(
        report_path,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    print(f"Relatório salvo em: {report_path}")
    return report_path


def final_mask_preview_file_name() -> str:
    """Nome estável da figura da máscara final, derivado da configuração."""
    config = get_config()
    return f"mask_final_{config['aoi']['region_code']}_{reference_year()}.png"


def save_final_mask_preview(storage_paths: dict[str, Path]) -> Path:
    """Persiste a figura da máscara final no Drive canônico (idempotente)."""
    from src import io

    figure_path = storage_paths["artifacts_figures"] / final_mask_preview_file_name()
    if io.path_exists(figure_path):
        print(f"Figura já existente (reutilizada): {figure_path}")
        return figure_path

    final_path = io.ensure_local_copy(final_mask_path(storage_paths))
    io.persist_bytes(figure_path, render_final_mask_preview(final_path))
    print(f"Figura salva em: {figure_path}")
    return figure_path


def render_final_mask_preview(mask_path: Path) -> bytes:
    """Renderiza a miniatura binária da máscara final (café em roxo)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import ListedColormap

    with rasterio.open(mask_path) as src:
        mask = src.read(1) > 0

    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(
        display_array(mask),
        cmap=ListedColormap(COFFEE_COLORS),
        vmin=0,
        vmax=1,
    )
    axis.set_title("Máscara binária final de café")
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()

    buffer = stdlib_io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    return buffer.getvalue()
