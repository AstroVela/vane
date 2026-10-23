# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Model declarations and encoding options for the fixed LiteRT-LM API."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from vane.ai._embedding_inputs import EmbeddingConfigurationError

LITERT_MODELS: dict[str, tuple[Literal["text", "image", "audio"], ...]] = {
    "litert-community/embeddinggemma-2-text-270m-litert-lm": ("text",),
    "litert-community/embeddinggemma-2-text-vision-440m-litert-lm": ("text", "image"),
    "litert-community/embeddinggemma-2-740m-litert-lm": ("text", "image", "audio"),
}
LITERT_DEFAULT_MODEL = "litert-community/embeddinggemma-2-740m-litert-lm"
LITERT_DIMENSIONS = (128, 256, 512, 768)
LITERT_COMMON_OPTIONS = frozenset(
    {
        "model_path",
        "model_sha256",
        "backend",
        "cache_dir",
        "num_threads",
        "max_length",
        "max_input_bytes",
        "vision_tokens_per_image",
    }
)
LITERT_TEXT_OPTIONS = LITERT_COMMON_OPTIONS | {"input_type"}
LITERT_SERVICE_OPTIONS = frozenset(
    {
        "runtime",
        "service_path",
        "service_id",
        "model_sha256",
        "max_length",
        "vision_tokens_per_image",
        "max_input_bytes",
        "timeout",
        "max_concurrency_per_actor",
    }
)


def litert_option_names(*, text: bool, service: bool) -> frozenset[str]:
    names = LITERT_SERVICE_OPTIONS if service else LITERT_COMMON_OPTIONS
    return names | {"input_type"} if text else names


def validate_litert_model(model: str | None, dimensions: int | None, kind: str) -> tuple[str, int]:
    selected = LITERT_DEFAULT_MODEL if model is None else model
    width = 768 if dimensions is None else dimensions
    if selected not in LITERT_MODELS:
        raise EmbeddingConfigurationError("LiteRT requires one of the declared EmbeddingGemma 2 LiteRT-LM models")
    if type(width) is not int or width not in LITERT_DIMENSIONS:
        raise EmbeddingConfigurationError("LiteRT dimensions must be 128, 256, 512 or 768")
    if kind != "multimodal" and kind not in LITERT_MODELS[selected]:
        raise EmbeddingConfigurationError("Selected LiteRT model does not support this input modality")
    return selected, width


def validate_litert_options(options: Any, *, text: bool, service: bool = False) -> dict[str, Any]:
    copied = dict(options)
    allowed = litert_option_names(text=text, service=service)
    if copied.keys() - allowed:
        raise TypeError("LiteRT options accept only the declared local model encoding settings")
    defaults = {"runtime": "service"} if service else {"backend": "cpu", "cache_dir": None, "num_threads": 2}
    prepared = (
        defaults
        | {
            "max_length": 2048,
            "max_input_bytes": 16 * 1024**2,
            "vision_tokens_per_image": 70,
        }
        | copied
    )
    for name in ("service_path",) if service else ("model_path", "cache_dir"):
        value = prepared.get(name)
        if name == "cache_dir" and value is None:
            continue
        if (
            not isinstance(value, str)
            or not value
            or not Path(value).is_absolute()
            or "://" in value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or name == "model_path"
            and not value.endswith(".litertlm")
        ):
            raise EmbeddingConfigurationError(f"LiteRT {name} requires an absolute local path")
    digest = prepared.get("model_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise EmbeddingConfigurationError("LiteRT model_sha256 requires the model file's 64-character SHA-256")
    prepared["model_sha256"] = digest.lower()
    if service:
        if prepared["runtime"] != "service":
            raise EmbeddingConfigurationError("LiteRT service requires runtime='service'")
        identity = prepared.get("service_id")
        if (
            not isinstance(identity, str)
            or not 1 <= len(identity) <= 128
            or re.fullmatch(r"[A-Za-z0-9_.-]+", identity) is None
        ):
            raise EmbeddingConfigurationError(
                "LiteRT service_id requires 1 to 128 ASCII letters, digits, '-', '_' or '.'"
            )
    elif not isinstance(prepared["backend"], str) or prepared["backend"] not in {"cpu", "gpu"}:
        raise EmbeddingConfigurationError("LiteRT backend must be cpu or gpu")
    bounds = (
        (("max_input_bytes", 1, 64 * 1024**2),)
        if service
        else (("num_threads", 1, 256), ("max_input_bytes", 1, 64 * 1024**2))
    )
    for name, minimum, maximum in bounds:
        value = prepared[name]
        if type(value) is not int or not minimum <= value <= maximum:
            raise EmbeddingConfigurationError(f"LiteRT {name} must be an integer from {minimum} to {maximum}")
    if type(prepared["max_length"]) is not int or prepared["max_length"] not in (128, 256, 512, 1024, 2048, 8192):
        raise EmbeddingConfigurationError(
            "LiteRT max_length must select a supported signature: 128, 256, 512, 1024, 2048, 8192"
        )
    if type(prepared["vision_tokens_per_image"]) is not int or prepared["vision_tokens_per_image"] not in (70, 140):
        raise EmbeddingConfigurationError("LiteRT vision_tokens_per_image must be 70 or 140")
    if text:
        role = prepared.setdefault("input_type", "document")
        if not isinstance(role, str) or role not in {"query", "document"}:
            raise EmbeddingConfigurationError("LiteRT input_type must be query or document")
    return prepared
