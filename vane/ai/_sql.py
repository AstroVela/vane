# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from vane.ai._redaction import is_sensitive_option_key
from vane.ai.functions import (
    _actor_number_or_one,
    _adapt_batch_wrapper_for_backend,
    _EmbedTextBatch,
    _gpus_or_zero,
    _prepare_embed_call,
    _prepare_prompt_call,
    _PromptBatch,
    _resolve_ai_batch_size,
    _ValidateStructuredOutputBatch,
)
from vane.ai.protocols import NativePrompterPlan

_INT_OPTION_NAMES = {
    "actor_number",
    "batch_size",
    "batch_token_limit",
    "dimensions",
    "inflight_limit",
    "input_text_token_limit",
    "load_balance_threshold",
    "max_buffer_size",
    "max_concurrency_per_actor",
    "max_output_tokens",
    "max_retries",
    "max_tokens",
    "min_bucket_size",
    "top_k",
}

_FLOAT_OPTION_NAMES = {
    "engine_init_timeout_s",
    "gpus_per_actor",
    "prefix_match_threshold",
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


def _embedding_output_type(dimensions: int) -> str:
    return f"FLOAT[{dimensions}]"


def _normalize_sql_options(options: dict[str, Any] | None) -> dict[str, Any]:
    _reject_inline_credentials(options or {})
    # Preserve explicit NULL fields until the closed API validators see them.
    # Dropping them here would let removed/unknown names and non-nullable
    # values bypass planning-time validation.
    opts = {key: _normalize_option_value(value, key) for key, value in dict(options or {}).items()}
    _reject_inline_credentials(opts)
    return opts


def build_ai_prompt_sql_spec(
    provider: str = "openai",
    model: str | None = None,
    system_message: str | None = None,
    on_error: str = "raise",
    options: dict[str, Any] | None = None,
    image_input: bool = False,
    return_format: str | dict[str, Any] | None = None,
    return_raw_response: bool = False,
) -> dict[str, Any]:
    """Build the row-preserving SQL prompt UDF specification."""
    opts = _normalize_sql_options(options)
    if isinstance(return_format, str):
        try:
            return_format = json.loads(return_format)
        except json.JSONDecodeError as exc:
            raise ValueError("AI_PROMPT return_format must be valid JSON") from exc
    descriptor, udf_opts, _, structured_output = _prepare_prompt_call(
        provider,
        model,
        return_format,
        system_message,
        return_raw_response,
        on_error,
        opts,
        relation=False,
    )

    # vLLM already has a native, relation-scoped physical operator. Keep its
    # executor alive across every input batch and let PhysicalVLLM send the
    # single terminal signal when the relation is exhausted.
    from vane.ai.providers.vllm import NativeVLLMPromptPlan, _build_native_vllm_options_argument

    if isinstance(descriptor, NativeVLLMPromptPlan):
        if image_input:
            raise ValueError("native vLLM ai_prompt does not support image inputs")

        native_options = descriptor.build_physical_vllm_options()
        options_argument = _build_native_vllm_options_argument(native_options)

        spec = {
            "execution_kind": "native_vllm",
            "name": "ai_prompt",
            "provider": descriptor.get_provider(),
            "model": descriptor.get_model(),
            "return_type": structured_output.duckdb_type if structured_output is not None else "VARCHAR",
            "system_message": descriptor.system_message,
            "options": options_argument,
        }
        if structured_output is not None:
            validator = _ValidateStructuredOutputBatch(
                structured_output,
                "response",
                "response",
                udf_opts.on_error,
            )
            actor_callable = _adapt_batch_wrapper_for_backend(
                validator,
                "subprocess_actor",
                force_actor=True,
            )
            spec["validation_spec"] = {
                "function": actor_callable,
                "name": "validate_vllm_structured_output",
                "provider": descriptor.get_provider(),
                "model": descriptor.get_model(),
                "return_type": "VARCHAR",
                "input_names": ["response"],
                "schema": {"response": "VARCHAR"},
                "batch_size": None,
                "row_preserving": True,
                "actor_number": 1,
                "gpus": 0,
            }
        return spec
    if isinstance(descriptor, NativePrompterPlan):
        raise ValueError(f"Unsupported native prompt plan {type(descriptor).__name__}")
    if image_input and not descriptor.supports_image_inputs():
        raise ValueError(
            f"Provider {descriptor.get_provider()!r} model {descriptor.get_model()!r} "
            "does not support Prompt image inputs"
        )

    input_names = ["message_0", "message_1"] if image_input else ["message_0"]
    wrapper = _PromptBatch(
        descriptor,
        input_names,
        "response",
        udf_opts.max_concurrency_per_actor,
        single_message=True,
        return_format=None if return_raw_response else structured_output,
        return_raw_response=return_raw_response,
        max_retries=udf_opts.max_retries,
        on_error=udf_opts.on_error,
    )
    actor_callable = _adapt_batch_wrapper_for_backend(wrapper, "subprocess_actor", force_actor=True)
    return {
        "function": actor_callable,
        "name": "ai_prompt",
        "provider": descriptor.get_provider(),
        "model": descriptor.get_model(),
        "return_type": (
            structured_output.duckdb_type if structured_output is not None and not return_raw_response else "VARCHAR"
        ),
        "input_names": input_names,
        "schema": {"response": "VARCHAR"},
        "batch_size": _resolve_ai_batch_size(udf_opts),
        "row_preserving": True,
        "actor_number": _actor_number_or_one(udf_opts),
        "gpus": _gpus_or_zero(udf_opts),
    }


def build_ai_embed_sql_spec(
    provider: str = "openai",
    model: str | None = None,
    dimensions: int | None = None,
    on_error: str = "raise",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    opts = _normalize_sql_options(options)
    descriptor, resolved_dimensions, udf_opts, normalize, _, _, _ = _prepare_embed_call(
        provider,
        model,
        dimensions,
        on_error,
        opts,
        relation=False,
    )
    wrapper = _EmbedTextBatch(
        descriptor,
        "text",
        "embedding",
        resolved_dimensions,
        max_retries=udf_opts.max_retries,
        on_error=udf_opts.on_error,
        normalize=normalize,
    )
    actor_callable = _adapt_batch_wrapper_for_backend(wrapper, "subprocess_actor", force_actor=True)
    return {
        "function": actor_callable,
        "name": "ai_embed",
        "provider": descriptor.get_provider(),
        "model": descriptor.get_model(),
        "dimensions": resolved_dimensions,
        "return_type": _embedding_output_type(resolved_dimensions),
        "input_names": ["text"],
        "schema": {"embedding": _embedding_output_type(resolved_dimensions)},
        "batch_size": _resolve_ai_batch_size(udf_opts),
        "row_preserving": True,
        "actor_number": _actor_number_or_one(udf_opts),
        "gpus": _gpus_or_zero(udf_opts),
    }
