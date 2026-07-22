# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import _vane_duckdb
import pytest

import vane
from vane._ray_cxx import require_ray_cxx_attr


def test_require_ray_cxx_attr_returns_registered_binding():
    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")

    assert require_ray_cxx_attr("PyLogicalPlan") is ray_cxx.PyLogicalPlan


def test_require_ray_cxx_attr_missing_binding_raises_clear_importerror(monkeypatch):
    monkeypatch.setattr(_vane_duckdb, "ray_cxx", object())

    with pytest.raises(ImportError, match=r"vane\.ray_cxx\.MissingBinding"):
        require_ray_cxx_attr("MissingBinding")
