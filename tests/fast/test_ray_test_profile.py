# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from ray_test_profile import (
    RAY_OBJECT_STORE_BYTES_ENV,
    ray_test_object_store_options,
)


def test_ray_test_object_store_follows_ray_by_default(monkeypatch):
    monkeypatch.delenv(RAY_OBJECT_STORE_BYTES_ENV, raising=False)

    assert ray_test_object_store_options() == {}


def test_ray_test_object_store_allows_override(monkeypatch):
    monkeypatch.setenv(RAY_OBJECT_STORE_BYTES_ENV, str(3 * 1024**3))

    assert ray_test_object_store_options() == {"object_store_memory": 3 * 1024**3}


@pytest.mark.parametrize("value", ["", "0", "-1", "not-an-integer"])
def test_ray_test_object_store_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv(RAY_OBJECT_STORE_BYTES_ENV, value)

    with pytest.raises(ValueError, match=RAY_OBJECT_STORE_BYTES_ENV):
        ray_test_object_store_options()
