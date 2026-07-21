# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from ray_test_profile import (
    DEFAULT_RAY_OBJECT_STORE_BYTES,
    RAY_OBJECT_STORE_BYTES_ENV,
    ray_test_object_store_bytes,
)


def test_ray_test_object_store_defaults_to_two_gib(monkeypatch):
    monkeypatch.delenv(RAY_OBJECT_STORE_BYTES_ENV, raising=False)

    assert ray_test_object_store_bytes() == 2 * 1024**3
    assert ray_test_object_store_bytes() == DEFAULT_RAY_OBJECT_STORE_BYTES


def test_ray_test_object_store_allows_override(monkeypatch):
    monkeypatch.setenv(RAY_OBJECT_STORE_BYTES_ENV, str(3 * 1024**3))

    assert ray_test_object_store_bytes() == 3 * 1024**3


@pytest.mark.parametrize("value", ["", "0", "-1", "not-an-integer"])
def test_ray_test_object_store_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv(RAY_OBJECT_STORE_BYTES_ENV, value)

    with pytest.raises(ValueError, match=RAY_OBJECT_STORE_BYTES_ENV):
        ray_test_object_store_bytes()
