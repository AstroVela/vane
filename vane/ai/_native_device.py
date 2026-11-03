# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""GPU admission for worker-owned vLLM and SGLang engines.

Planning validates configuration without importing SDKs or probing the driver's
hardware. Only the executing worker asks its SDK which platform it selected.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import Decimal
from numbers import Real
from typing import Any


def validate_native_device(family: str, args: Mapping[str, Any]) -> None:
    if family == "vllm":
        if "device" in args:
            raise ValueError(
                "vLLM 0.31 does not accept engine_args.device; its SDK selects the platform. "
                "Vane native vLLM execution requires CUDA."
            )
    elif args.get("device") is not None and args["device"] not in ("cuda", "cuda:0"):
        raise ValueError(
            "Native SGLang engine_args.device must be 'cuda', 'cuda:0' or omitted; "
            "CPU and other accelerator execution are not supported by this Vane path."
        )


def normalize_native_gpu_options(family: str, options: dict[str, Any], *, fractional: bool = False) -> None:
    args = options.get("engine_args") or {}
    validate_native_device(family, args)
    parallel = ("tensor_parallel_size", "pipeline_parallel_size") if family == "vllm" else ("tp_size", "pp_size")
    required = 1
    for name in parallel:
        value = args.get(name, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"engine_args.{name} must be a positive integer")
        required *= value
    gpus = options.get("gpus_per_actor", required)
    if fractional:
        if isinstance(gpus, bool) or not isinstance(gpus, (Real, Decimal)):
            raise ValueError("gpus_per_actor must be a finite positive number")
        gpus = float(gpus)
        if not math.isfinite(gpus) or gpus <= 0 or (gpus >= 1 and not gpus.is_integer()):
            raise ValueError("gpus_per_actor must be positive and must be an integer when >= 1")
        if gpus >= 1:
            gpus = int(gpus)
    elif type(gpus) is not int or gpus < 1:
        raise ValueError("gpus_per_actor must be a positive integer for native GPU execution")
    if required > 1 and gpus < required:
        raise ValueError("gpus_per_actor must cover engine parallelism")
    options["gpus_per_actor"] = gpus


def require_native_cuda(family: str, args: Mapping[str, Any]) -> None:
    """Check the worker SDK's platform before processor I/O or engine creation."""
    validate_native_device(family, args)
    if family == "vllm":
        from vllm.platforms import current_platform

        device = current_platform.device_type
    else:
        from sglang.srt.utils import get_device  # type: ignore[import-not-found, import-untyped, unused-ignore]

        device = get_device()
    if device != "cuda":
        raise ValueError(
            f"Vane native {family} execution requires CUDA, but the worker SDK selected {device!r}; "
            "CPU and other accelerator execution are not supported by this Vane path."
        )
