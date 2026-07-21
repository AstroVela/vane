# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

DEFAULT_RAY_OBJECT_STORE_BYTES = 2 * 1024**3
RAY_OBJECT_STORE_BYTES_ENV = "VANE_TEST_RAY_OBJECT_STORE_BYTES"


def ray_test_object_store_bytes() -> int:
    """Return the object-store capacity for a test-owned Ray cluster."""
    raw_value = os.environ.get(RAY_OBJECT_STORE_BYTES_ENV, str(DEFAULT_RAY_OBJECT_STORE_BYTES))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{RAY_OBJECT_STORE_BYTES_ENV} must be a positive integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{RAY_OBJECT_STORE_BYTES_ENV} must be a positive integer, got {raw_value!r}")
    return value
