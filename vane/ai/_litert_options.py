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


def validate_litert_options(options: Any, *, text: bool) -> dict[str, Any]:
    copied = dict(options)
    allowed = LITERT_TEXT_OPTIONS if text else LITERT_COMMON_OPTIONS
    if copied.keys() - allowed:
        raise TypeError("LiteRT options accept only the declared local model encoding settings")
    prepared = {
        "backend": "cpu",
        "cache_dir": None,
        "num_threads": 2,
        "max_length": 2048,
        "max_input_bytes": 16 * 1024**2,
        "vision_tokens_per_image": 70,
    } | copied
    for name in ("model_path", "cache_dir"):
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
    if not isinstance(prepared["backend"], str) or prepared["backend"] not in {"cpu", "gpu"}:
        raise EmbeddingConfigurationError("LiteRT backend must be cpu or gpu")
    for name, minimum, maximum in (("num_threads", 1, 256), ("max_input_bytes", 1, 64 * 1024**2)):
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
