"""Testes de src/utils.py (sementes fixas e flags determinísticas)."""

from src.utils import set_all_seeds, set_deterministic_flags


def test_set_all_seeds_and_flags_run() -> None:
    """Sementes e flags determinísticas devem executar sem erro."""
    set_all_seeds(42)
    set_deterministic_flags()
