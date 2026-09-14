"""Testes dos helpers puros de resolução e recorte da área de estudo."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from src.config import CONFIG
from src.data.aoi import (
    MESH_PATH_ENV,
    AOIError,
    geometry_bounds,
    geometry_to_ee,
    resolve_mesh_path,
    select_region_frame,
)


def _frame() -> pd.DataFrame:
    """DataFrame mínimo que imita o DBF da malha do IBGE."""
    return cast(
        "pd.DataFrame",
        pd.DataFrame(
            {
                "CD_RGI": ["310044", "310027"],
                "NM_RGI": ["Guaxupé", "Juiz de Fora"],
            }
        ),
    )


def test_select_region_frame_returns_target() -> None:
    subset = select_region_frame(_frame(), "310044")
    assert len(subset) == 1
    assert subset.iloc[0]["NM_RGI"] == "Guaxupé"


def test_select_region_frame_missing_code_raises() -> None:
    with pytest.raises(AOIError):
        select_region_frame(_frame(), "999999")


def test_select_region_frame_missing_column_raises() -> None:
    with pytest.raises(AOIError):
        select_region_frame(_frame(), "310044", region_code_column="CD_UF")


def test_resolve_mesh_path_explicit_argument() -> None:
    resolved = resolve_mesh_path("/tmp/custom.shp")
    assert resolved == Path("/tmp/custom.shp")


def test_resolve_mesh_path_relative_is_anchored_on_root() -> None:
    resolved = resolve_mesh_path("data/external/mesh.shp")
    assert resolved == CONFIG.paths.root / "data/external/mesh.shp"


def test_resolve_mesh_path_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MESH_PATH_ENV, "/tmp/from-env.shp")
    assert resolve_mesh_path() == Path("/tmp/from-env.shp")


def test_resolve_mesh_path_defaults_to_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MESH_PATH_ENV, raising=False)
    resolved = resolve_mesh_path()
    assert resolved == CONFIG.paths.root / str(CONFIG.get("aoi.mesh_path"))


def test_geometry_bounds_returns_floats() -> None:
    class _Geometry:
        bounds = (1, 2, 3, 4)

    assert geometry_bounds(_Geometry()) == (1.0, 2.0, 3.0, 4.0)


def test_geometry_to_ee_uses_geo_interface() -> None:
    class _Geometry:
        def __init__(self) -> None:
            self.__geo_interface__ = {"type": "Point", "coordinates": (0.0, 0.0)}

    class _EE:
        def __init__(self) -> None:
            self.received: dict[str, object] | None = None

        def Geometry(self, payload: dict) -> dict:  # noqa: N802
            self.received = payload
            return payload

    fake_ee = _EE()
    geometry_to_ee(fake_ee, _Geometry())
    assert fake_ee.received == {"type": "Point", "coordinates": (0.0, 0.0)}
