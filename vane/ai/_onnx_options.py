# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Closed, SDK-free declarations for EmbeddingGemma 2 ONNX execution."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from vane.ai._embedding_inputs import EmbeddingConfigurationError

ONNX_MODEL = "onnx-community/embeddinggemma-2-ONNX"
ONNX_DIMENSIONS = (128, 256, 512, 768)
ONNX_DTYPES = {"fp32": "", "fp16": "_fp16", "q8": "_quantized", "q4": "_q4", "q4f16": "_q4f16"}
ONNX_OPTIONS = frozenset(
    {
        "runtime",
        "model_path",
        "model_sha256",
        "dtype",
        "backend",
        "num_threads",
        "max_length",
        "max_input_bytes",
    }
)
ONNX_SERVICE_OPTIONS = frozenset(
    {
        "runtime",
        "service_path",
        "service_id",
        "model_sha256",
        "max_length",
        "max_input_bytes",
        "timeout",
        "max_concurrency_per_actor",
    }
)


def onnx_option_names(*, text: bool, service: bool) -> frozenset[str]:
    return (ONNX_SERVICE_OPTIONS if service else ONNX_OPTIONS) | ({"input_type"} if text else set())


def validate_onnx_model(model: str | None, dimensions: int | None) -> tuple[str, int]:
    model = ONNX_MODEL if model is None else model
    dimensions = 768 if dimensions is None else dimensions
    if model != ONNX_MODEL:
        raise EmbeddingConfigurationError("ONNX requires the declared EmbeddingGemma 2 ONNX export")
    if type(dimensions) is not int or dimensions not in ONNX_DIMENSIONS:
        raise EmbeddingConfigurationError("ONNX dimensions must be 128, 256, 512 or 768")
    return model, dimensions


def validate_onnx_options(options: Any, *, text: bool, service: bool = False) -> dict[str, Any]:
    copied = dict(options)
    if copied.keys() - onnx_option_names(text=text, service=service):
        raise TypeError("ONNX options accept only the declared runtime settings")
    prepared: dict[str, Any] = {
        "runtime": "service" if service else "native",
        "max_length": 2048,
        "max_input_bytes": 16 * 1024**2,
    } | copied
    if prepared["runtime"] != ("service" if service else "native"):
        raise EmbeddingConfigurationError("ONNX runtime must be native or service")
    path_name = "service_path" if service else "model_path"
    path = prepared.get(path_name)
    if (
        not isinstance(path, str)
        or not path
        or not Path(path).is_absolute()
        or "://" in path
        or any(ord(c) < 32 or ord(c) == 127 for c in path)
    ):
        raise EmbeddingConfigurationError(f"ONNX {path_name} requires an absolute local path")
    digest = prepared.get("model_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise EmbeddingConfigurationError("ONNX model_sha256 requires the model bundle's 64-character SHA-256")
    prepared["model_sha256"] = digest.lower()
    for name, default, maximum in (("max_length", 2048, 8192), ("max_input_bytes", 16 * 1024**2, 64 * 1024**2)):
        value = prepared.setdefault(name, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise EmbeddingConfigurationError(f"ONNX {name} must be an integer from 1 to {maximum}")
    if service:
        identity = prepared.get("service_id")
        if not isinstance(identity, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", identity) is None:
            raise EmbeddingConfigurationError(
                "ONNX service_id requires 1 to 128 ASCII letters, digits, '-', '_' or '.'"
            )
        for name, default, maximum in (("max_concurrency_per_actor", 8, 1024),):
            value = prepared.setdefault(name, default)
            if type(value) is not int or not 1 <= value <= maximum:
                raise EmbeddingConfigurationError(f"ONNX {name} must be an integer from 1 to {maximum}")
        timeout = prepared.setdefault("timeout", 60.0)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise EmbeddingConfigurationError("ONNX timeout must be finite and positive")
    else:
        for setting, initial in (("dtype", "fp32"), ("backend", "cpu"), ("num_threads", 2)):
            prepared.setdefault(setting, initial)
        if not isinstance(prepared["dtype"], str) or prepared["dtype"] not in ONNX_DTYPES:
            raise EmbeddingConfigurationError("ONNX dtype must be fp32, fp16, q8, q4 or q4f16")
        if prepared["backend"] not in ("cpu", "cuda", "coreml"):
            raise EmbeddingConfigurationError("ONNX backend must be cpu, cuda or coreml")
        for name, minimum, maximum in (("num_threads", 1, 256),):
            if type(prepared[name]) is not int or not minimum <= prepared[name] <= maximum:
                raise EmbeddingConfigurationError(f"ONNX {name} must be an integer from {minimum} to {maximum}")
    if text:
        role = prepared.setdefault("input_type", "document")
        if role not in ("query", "document"):
            raise EmbeddingConfigurationError("ONNX input_type must be query or document")
    return prepared
