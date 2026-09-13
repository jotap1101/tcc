"""Testes do carregamento de configuração e das utilidades de reprodutibilidade."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from src.config import (
    CONFIG,
    CONFIG_PATH,
    PROJECT_ROOT,
    Config,
    _read_yaml,
    log_environment,
    save_environment_log,
    seed_everything,
)


def test_config_loads_expected_values() -> None:
    assert CONFIG.seed == 42
    assert CONFIG.patch_size == 512
    assert CONFIG.fold_count == 5
    assert CONFIG.bands == ["B2", "B3", "B4", "B8"]
    assert CONFIG.coffee_min_ratio == pytest.approx(0.01)


def test_get_supports_dotted_paths_and_defaults() -> None:
    assert CONFIG.get("gee.scale_m") == 10
    assert CONFIG.get("does.not.exist", "fallback") == "fallback"


def test_paths_resolve_under_project_root() -> None:
    assert CONFIG.paths.root == PROJECT_ROOT
    assert CONFIG.paths.data == (PROJECT_ROOT / "data").resolve()
    assert CONFIG.paths.manifest.name == "manifest.parquet"


def test_config_hash_is_stable_and_short() -> None:
    assert CONFIG.config_hash == CONFIG.config_hash
    assert len(CONFIG.config_hash) == 12


def test_raw_is_a_defensive_copy() -> None:
    raw = CONFIG.raw
    raw["project"]["seed"] = 999
    assert CONFIG.seed == 42


def test_model_variants_exposes_both_architectures() -> None:
    variants = CONFIG.model_variants
    assert "unet" in variants
    assert "segformer" in variants


def test_read_yaml_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _read_yaml(tmp_path / "missing.yaml")


def test_read_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    target.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _read_yaml(target)


def test_config_accepts_custom_root(tmp_path: Path) -> None:
    cfg = Config({"paths": {"data": "dados"}, "project": {"seed": 7}}, root=tmp_path)
    assert cfg.paths.data == (tmp_path / "dados").resolve()
    assert cfg.seed == 7


def test_seed_everything_is_deterministic() -> None:
    seed_everything(123)
    first = [random.random() for _ in range(3)]
    seed_everything(123)
    second = [random.random() for _ in range(3)]
    assert first == second


def test_log_environment_contains_traceability_fields() -> None:
    report = log_environment()
    assert report["config_hash"] == CONFIG.config_hash
    assert report["seed"] == CONFIG.seed
    assert "packages" in report


def test_save_environment_log_writes_json(tmp_path: Path) -> None:
    target = save_environment_log(tmp_path / "env.json")
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8"))["seed"] == CONFIG.seed


def test_config_path_points_to_repo_config() -> None:
    assert CONFIG_PATH == PROJECT_ROOT / "src" / "config.yaml"
