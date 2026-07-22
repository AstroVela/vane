# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import pyarrow as pa

from vane.ai._redaction import is_sensitive_option_key
from vane.ai.functions import (
    _actor_number_or_one,
    _adapt_batch_wrapper_for_backend,
    _EmbedTextBatch,
    _gpus_or_zero,
    _PromptBatch,
    _resolve_ai_batch_size,
    _resolve_provider,
)


def _drop_none(options: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in options.items() if value is not None}


_INT_OPTION_NAMES = {
    "actor_number",
    "batch_size",
    "concurrency",
    "dimensions",
    "max_api_concurrency",
    "max_output_tokens",
    "max_retries",
    "max_tokens",
    "top_k",
}

_FLOAT_OPTION_NAMES = {
    "frequency_penalty",
    "gpus_per_actor",
    "gpu_memory_utilization",
    "presence_penalty",
    "temperature",
    "timeout",
    "top_p",
}


def _reject_inline_credentials(value: Any, path: str = "options") -> None:
    """Reject sensitive-keyed options at any nesting depth (defense layer 1).

    SQL options must never carry credentials inline; environment variables are
    the supported path. Key matching uses the shared sensitive-key table in
    ``vane.ai._redaction``, which also drives the Python-layer ``Secret``
    sealing (defense layer 2) — see that module's docstring for the full
    two-layer design.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if is_sensitive_option_key(key_text):
                raise ValueError(
                    f"AI SQL options cannot include inline credential field {path}.{key_text}; "
                    "configure provider credentials through environment variables"
                )
            _reject_inline_credentials(item, f"{path}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_inline_credentials(item, f"{path}[{index}]")


def _decimal_to_number(value: Decimal, name: str | None) -> int | float:
    if not value.is_finite():
        option_name = name or "SQL option"
        raise ValueError(f"{option_name} must be finite")

    key = name or ""
    if key in _FLOAT_OPTION_NAMES:
        return float(value)
    if key in _INT_OPTION_NAMES:
        if value != value.to_integral_value():
            raise ValueError(f"{key} must be an integer")
        return int(value)
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _normalize_option_value(value: Any, name: str | None = None) -> Any:
    if isinstance(value, Decimal):
        return _decimal_to_number(value, name)
    if isinstance(value, dict):
        return {key: _normalize_option_value(item, str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_option_value(item, name) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_option_value(item, name) for item in value)
    return value


def _int_or_none(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _pop_execution_options(opts: dict[str, Any]) -> tuple[int | None, int | None]:
    """Remove UDF-execution options that must not reach provider API kwargs."""
    batch_size = _int_or_none(opts.pop("batch_size", None), "batch_size")
    raw_max_retries = opts.pop("max_retries", None)
    max_retries: int | None = None
    if raw_max_retries is not None:
        max_retries = int(raw_max_retries)
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
    return batch_size, max_retries


def _embedding_output_type(dimensions: int | None) -> str:
    return "FLOAT[]" if dimensions is None else f"FLOAT[{dimensions}]"


def _embedding_arrow_type(dimensions: int | None) -> pa.DataType:
    return pa.list_(pa.float32()) if dimensions is None else pa.list_(pa.float32(), dimensions)


def _normalize_sql_options(options: dict[str, Any] | None) -> dict[str, Any]:
    _reject_inline_credentials(options or {})
    opts = {key: _normalize_option_value(value, key) for key, value in _drop_none(dict(options or {})).items()}
    concurrency = opts.pop("concurrency", None)
    if concurrency is not None and "actor_number" not in opts:
        opts["actor_number"] = _int_or_none(concurrency, "concurrency")

    for source, target in (("engine_args_json", "engine_args"), ("generate_args_json", "generate_args")):
        raw = opts.pop(source, None)
        if raw is not None and target not in opts:
            opts[target] = _normalize_option_value(json.loads(str(raw)), target)
    _reject_inline_credentials(opts)
    return opts


def build_ai_prompt_sql_spec(options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = _normalize_sql_options(options)
    provider = opts.pop("provider", "openai")
    model = opts.pop("model", None)
    system_message = opts.pop("system_message", None)
    batch_size, max_retries = _pop_execution_options(opts)

    prov = _resolve_provider(provider, "openai")
    try:
        descriptor = prov.get_prompter(model=model, system_message=system_message, **opts)
    except NotImplementedError as exc:
        raise ValueError(f"Provider {provider!r} is not a prompt provider") from exc

    # vLLM already has a native, relation-scoped physical operator. Keep its
    # executor alive across every input batch and let PhysicalVLLM send the
    # single terminal signal when the relation is exhausted.
    from vane.ai.providers.vllm import VLLMPrompterDescriptor

    if isinstance(descriptor, VLLMPrompterDescriptor):
        if max_retries not in (None, 0):
            raise ValueError("native vLLM ai_prompt does not support max_retries")

        native_options = descriptor.build_physical_vllm_options()
        if batch_size is not None:
            native_options["batch_size"] = batch_size
        try:
            options_json = json.dumps(native_options, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise TypeError("vLLM native options must be JSON-serializable") from exc

        return {
            "execution_kind": "native_vllm",
            "name": "ai_prompt",
            "provider": descriptor.get_provider(),
            "model": descriptor.get_model(),
            "return_type": "VARCHAR",
            "system_message": descriptor.system_message,
            "options_json": options_json,
        }

    udf_opts = descriptor.get_udf_options()
    resolved_max_retries = udf_opts.max_retries if max_retries is None else max_retries
    wrapper = _PromptBatch(
        descriptor,
        "messages",
        "response",
        udf_opts.max_api_concurrency,
        max_retries=resolved_max_retries,
        on_error=udf_opts.on_error,
    )
    actor_callable = _adapt_batch_wrapper_for_backend(wrapper, "subprocess_actor", force_actor=True)
    return {
        "function": actor_callable,
        "name": "ai_prompt",
        "provider": descriptor.get_provider(),
        "model": descriptor.get_model(),
        "return_type": "VARCHAR",
        "input_names": ["messages"],
        "schema": {"response": "VARCHAR"},
        "batch_size": batch_size if batch_size is not None else _resolve_ai_batch_size(udf_opts),
        "row_preserving": True,
        "actor_number": _actor_number_or_one(udf_opts),
        "gpus": _gpus_or_zero(udf_opts),
    }


def build_ai_embed_sql_spec(options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = _normalize_sql_options(options)
    provider = opts.pop("provider", "openai")
    model = opts.pop("model", None)
    dimensions = _int_or_none(opts.pop("dimensions", None), "dimensions")
    normalize = bool(opts.pop("normalize", False))
    batch_size, max_retries = _pop_execution_options(opts)

    prov = _resolve_provider(provider, "openai")
    try:
        descriptor = prov.get_text_embedder(model=model, dimensions=dimensions, **opts)
    except NotImplementedError as exc:
        raise ValueError(f"Provider {provider!r} is not an embedding provider") from exc

    udf_opts = descriptor.get_udf_options()
    resolved_max_retries = udf_opts.max_retries if max_retries is None else max_retries
    wrapper = _EmbedTextBatch(
        descriptor,
        "text",
        "embedding",
        max_retries=resolved_max_retries,
        on_error=udf_opts.on_error,
        normalize=normalize,
        arrow_type=_embedding_arrow_type(dimensions),
    )
    actor_callable = _adapt_batch_wrapper_for_backend(wrapper, "subprocess_actor", force_actor=True)
    return {
        "function": actor_callable,
        "name": "ai_embed",
        "provider": descriptor.get_provider(),
        "model": descriptor.get_model(),
        "dimensions": dimensions,
        "return_type": _embedding_output_type(dimensions),
        "input_names": ["text"],
        "schema": {"embedding": _embedding_output_type(dimensions)},
        "batch_size": batch_size if batch_size is not None else _resolve_ai_batch_size(udf_opts),
        "row_preserving": True,
        "actor_number": _actor_number_or_one(udf_opts),
        "gpus": _gpus_or_zero(udf_opts),
    }
