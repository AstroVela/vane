# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from vane._vane_session import ensure_vane_session_dir

_OPTIONAL_ENV_KEYS = (
    "S3FS_ANON",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
    "RAY_ADDRESS",
)


def collect_vane_env_overrides() -> dict[str, str]:
    ensure_vane_session_dir()

    overrides = {key: value for key, value in os.environ.items() if key.startswith("VANE_")}
    for key in ("PYTHONPATH", "PYTHONWARNINGS"):
        value = os.environ.get(key)
        if value:
            overrides[key] = value
    for key in _OPTIONAL_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            overrides[key] = value
    for key, value in os.environ.items():
        if key in overrides:
            continue
        if key.startswith(("AWS_", "DUCKDB_", "S3FS_", "VANE_")):
            overrides[key] = value
    return overrides
