# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from typing import Any

MAX_ACTOR_RESTARTS = 4
MAX_ACTOR_TASK_RETRIES = 4
DEFAULT_RAY_UDF_CPUS = 1.0
SUBMIT_RESULT_MARKER = "__vane_submit_result__"
REF_BUNDLE_RESULT_MARKER = "__vane_ref_bundle_result__"

TRUTHY_FALSE_VALUES = ("", "0", "false", "no", "off")


def required_positive_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"UDF payload is missing required positive integer field '{key}'")
    if type(value) is not int or value <= 0:
        raise ValueError(f"UDF payload field '{key}' must be a positive integer")
    return value


def actor_pool_size(payload: dict[str, Any]) -> int:
    return required_positive_int(payload, "actor_pool_size")


def payload_num_cpus(payload: dict[str, Any]) -> float:
    value = payload.get("cpus")
    if value is None:
        return DEFAULT_RAY_UDF_CPUS
    if type(value) not in (int, float):
        raise ValueError("UDF payload field 'cpus' must be a non-negative number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("UDF payload field 'cpus' must be a finite non-negative number")
    return parsed


def payload_num_gpus(payload: dict[str, Any]) -> float:
    value = payload.get("gpus")
    if value is None:
        return 0.0
    if type(value) not in (int, float):
        raise ValueError("UDF payload field 'gpus' must be a non-negative number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("UDF payload field 'gpus' must be a finite non-negative number")
    return parsed


def eager_actor_warm_up_enabled(payload: dict[str, Any] | None = None) -> bool:
    value = os.getenv("VANE_UDF_EAGER_WARM_UP", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value not in TRUTHY_FALSE_VALUES:
        raise ValueError("VANE_UDF_EAGER_WARM_UP must be a boolean value")
    if payload is not None:
        configured = payload.get("eager_warm_up", False)
        if type(configured) is not bool:
            raise ValueError("UDF payload field 'eager_warm_up' must be boolean")
        return configured
    return False


def stream_output_enabled(payload: dict[str, Any]) -> bool:
    value = payload.get("stream_output", False)
    if type(value) is not bool:
        raise ValueError("UDF payload field 'stream_output' must be boolean")
    return value


def has_tabular_output_schema(payload: dict[str, Any]) -> bool:
    return bool(payload.get("output_schema"))
