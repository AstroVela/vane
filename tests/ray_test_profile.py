# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

RAY_OBJECT_STORE_BYTES_ENV = "VANE_TEST_RAY_OBJECT_STORE_BYTES"


def ray_test_object_store_options() -> dict[str, int]:
    """Return an explicit object-store override for a test-owned Ray cluster."""
    raw_value = os.environ.get(RAY_OBJECT_STORE_BYTES_ENV)
    if raw_value is None:
        return {}
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{RAY_OBJECT_STORE_BYTES_ENV} must be a positive integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{RAY_OBJECT_STORE_BYTES_ENV} must be a positive integer, got {raw_value!r}")
    return {"object_store_memory": value}
