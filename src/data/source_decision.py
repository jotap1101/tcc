"""Registro da decisão manual da fonte de ground truth para treino (estágio 04).

O sistema não elege fonte alguma: `ground_truth.chosen_source` é uma decisão do
pesquisador, editada em src/config.yaml e apoiada pelo diagnóstico comparativo
do estágio 03 (todas as fontes geram máscaras e patches próprios). Este módulo
apenas lê e valida a escolha, carrega o relatório comparativo como base métrica
da decisão e persiste um relatório de registro (auditoria) em
MyDrive/tcc/artifacts/metrics/ground_truth/. As persistências são idempotentes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import get_config
from src.data.mask_comparison import report_file_name
from src.data.mask_utils import reference_year


def chosen_source() -> str:
    """Fonte de ground truth para treinamento, decidida manualmente no config.yaml."""
    config = get_config()
    chosen = config["ground_truth"].get("chosen_source")
    if not chosen:
        raise ValueError("Defina ground_truth.chosen_source em src/config.yaml.")
    enabled = [name for name, cfg in config["ground_truth"]["sources"].items() if cfg["enabled"]]
    if chosen not in enabled:
        raise ValueError(f"Fonte escolhida inválida ou desabilitada: {chosen}")
    return chosen


def source_decision_report_file_name() -> str:
    """Nome estável do relatório da decisão (multianual), derivado da configuração."""
    config = get_config()
    return f"source_decision_{config['aoi']['region_code']}.json"


def comparison_pair_key(comparison: dict[str, Any], chosen: str) -> str:
    """Chave do par de concordância do relatório que envolve a fonte escolhida."""
    return next(key for key in comparison["pairs"] if chosen in key)


def load_comparison_report(storage_paths: dict[str, Path]) -> dict[str, Any]:
    """Carrega o relatório comparativo do estágio 03 (reutilizado, idempotente)."""
    from src import io

    report_path = storage_paths["artifacts_metrics_ground_truth"] / report_file_name()
    if not io.path_exists(report_path):
        raise FileNotFoundError(f"Relatório do estágio 03 não encontrado: {report_path}")
    local_path = io.ensure_local_copy(report_path)
    return json.loads(local_path.read_text(encoding="utf-8"))


def save_source_decision_report(
    chosen: str,
    comparison: dict[str, Any],
    storage_paths: dict[str, Path],
) -> Path:
    """Persiste o relatório JSON da decisão manual (idempotente).

    O relatório registra a fonte escolhida (decisão do pesquisador), o ano de
    referência dela e a base métrica do diagnóstico — sem qualquer seleção
    automática: todas as fontes do comparativo possuem máscara e patches.
    """
    from src import io

    report_path = storage_paths["artifacts_metrics_ground_truth"] / (
        source_decision_report_file_name()
    )
    if io.path_exists(report_path):
        print(f"Relatório já existente (reutilizado): {report_path}")
        return report_path

    config = get_config()
    payload = {
        "chosen_source": chosen,
        "region_code": config["aoi"]["region_code"],
        "year": reference_year(chosen),
        "decision": "manual",
        "sources_year": {name: reference_year(name) for name in comparison["sources"]},
        "comparison_report": report_file_name(),
        "sources": comparison["sources"],
        "area_km2": comparison["area_km2"],
        "coffee_share": comparison["coffee_share"],
        "pairs": comparison["pairs"],
        # Base concreta para a decisão manual: métricas dos pares que envolvem a
        # fonte escolhida (não é uma eleição do sistema).
        "decision_basis": {
            "chosen_source": chosen,
            "year": reference_year(chosen),
            "area_km2": comparison["area_km2"][chosen],
            "coffee_share": comparison["coffee_share"][chosen],
            "pairs": {
                key: {"metrics": value["metrics"]}
                for key, value in comparison["pairs"].items()
                if chosen in key
            },
        },
    }
    io.persist_bytes(
        report_path,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    print(f"Relatório salvo em: {report_path}")
    return report_path
