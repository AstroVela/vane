# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Typed option objects for high-level AI helper functions."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, TypedDict
from urllib.parse import parse_qsl, urlsplit

from vane.ai._redaction import is_sensitive_option_key

VLLMJSONPrimitive: TypeAlias = str | int | float | bool | None
VLLMJSONValue: TypeAlias = VLLMJSONPrimitive | list["VLLMJSONValue"] | dict[str, "VLLMJSONValue"]


class PromptOptions(TypedDict, total=False):
    """Closed keyword surface shared by the Python Prompt entry points."""

    # Common generation and execution options.
    temperature: float | None
    batch_size: int
    actor_number: int
    max_concurrency_per_actor: int
    execution_backend: Literal["subprocess_task", "subprocess_actor", "ray_task", "ray_actor"] | None
    max_retries: int

    # OpenAI / OpenAI-compatible prompt options.
    use_chat_completions: bool
    max_output_tokens: int | None
    top_p: float | None
    stop_sequences: list[str] | None
    base_url: str | None
    timeout: float | None

    # Anthropic keeps its native output-token name.
    max_tokens: int | None
    top_k: int | None

    # Native vLLM execution options.
    gpus_per_actor: float
    engine_args: Mapping[str, VLLMJSONValue]
    generate_args: Mapping[str, VLLMJSONValue]
    do_prefix_routing: bool
    max_buffer_size: int
    min_bucket_size: int
    prefix_match_threshold: float
    load_balance_threshold: int
    inflight_limit: int
    engine_init_timeout_s: float | None


class EmbedOptions(TypedDict, total=False):
    """Closed keyword surface shared by the Python Embed entry points."""

    normalize: bool
    batch_size: int
    actor_number: int
    execution_backend: Literal["subprocess_task", "subprocess_actor", "ray_task", "ray_actor"] | None
    max_retries: int
    max_chunk_chars: int | None
    chunk_overlap_chars: int

    # OpenAI / OpenAI-compatible embedding options.
    encoding_format: Literal["float", "base64"]
    base_url: str | None
    timeout: float | None
    batch_token_limit: int
    input_text_token_limit: int | None

    # Google native embedding options.
    task_type: (
        Literal[
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION",
            "CODE_RETRIEVAL_QUERY",
        ]
        | None
    )
    title: str | None

    # SentenceTransformers / Hugging Face model-loading options.
    cache_folder: str | None
    device: str | None
    local_files_only: bool
    revision: str | None
    trust_remote_code: bool


_EMBED_COMMON_OPTIONS = frozenset({"normalize", "batch_size", "actor_number", "max_retries"})
_EMBED_RELATION_OPTIONS = frozenset({"execution_backend", "max_chunk_chars", "chunk_overlap_chars"})
_EMBED_PROVIDER_OPTIONS = {
    "openai": frozenset({"encoding_format", "base_url", "timeout", "batch_token_limit", "input_text_token_limit"}),
    "google": frozenset({"task_type", "title"}),
    "transformers": frozenset({"cache_folder", "device", "local_files_only", "revision", "trust_remote_code"}),
}
_GOOGLE_EMBED_TASK_TYPES = frozenset(
    {
        "RETRIEVAL_QUERY",
        "RETRIEVAL_DOCUMENT",
        "SEMANTIC_SIMILARITY",
        "CLASSIFICATION",
        "CLUSTERING",
        "QUESTION_ANSWERING",
        "FACT_VERIFICATION",
        "CODE_RETRIEVAL_QUERY",
    }
)
_EXECUTION_BACKENDS = frozenset({"subprocess_task", "subprocess_actor", "ray_task", "ray_actor"})
_OPENAI_EXTRA_SENSITIVE_KEYS = frozenset({"organization"})
_HUGGING_FACE_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")
_VLLM_REMOTE_CODE_SPECULATIVE_METHODS = frozenset(
    {
        "mtp",
        "deepseek_mtp",
        "mimo_mtp",
        "glm4_moe_mtp",
        "ernie_mtp",
        "qwen3_next_mtp",
        "longcat_flash_mtp",
    }
)


def _is_hugging_face_commit_sha(value: Any) -> bool:
    return isinstance(value, str) and _HUGGING_FACE_COMMIT_SHA.fullmatch(value) is not None


def _validate_base_url_option(options: Mapping[str, Any], *, api: Literal["Embed", "Prompt"]) -> None:
    if "base_url" not in options or options["base_url"] is None:
        return
    value = options["base_url"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{api} option 'base_url' must be a non-empty HTTP(S) URL or None")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{api} option 'base_url' must be a non-empty HTTP(S) URL or None")
    sensitive_query_keys = [
        key
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        if is_sensitive_option_key(key, _OPENAI_EXTRA_SENSITIVE_KEYS)
    ]
    if parsed.username or parsed.password or sensitive_query_keys:
        raise ValueError(f"{api} option 'base_url' cannot contain credentials")
    if parsed.fragment:
        raise ValueError(f"{api} option 'base_url' cannot contain a URL fragment")


def _reject_sensitive_embed_options(value: Any, path: str = "options") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if is_sensitive_option_key(key):
                raise ValueError(
                    f"Embed options cannot include sensitive field {path}.{key}; "
                    "configure credentials through the environment or runtime secret management"
                )
            _reject_sensitive_embed_options(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_sensitive_embed_options(item, f"{path}[{index}]")


def _require_embed_int(options: Mapping[str, Any], name: str, *, minimum: int) -> None:
    if name not in options:
        return
    value = options[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "a positive integer" if minimum == 1 else f"an integer >= {minimum}"
        raise ValueError(f"Embed option {name!r} must be {qualifier}")


def _require_optional_nonempty_string(options: Mapping[str, Any], name: str) -> None:
    if name not in options or options[name] is None:
        return
    value = options[name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Embed option {name!r} must be a non-empty string or None")


def validate_embed_options(
    provider_family: str | None,
    options: Mapping[str, Any],
    *,
    relation: bool,
) -> dict[str, Any]:
    """Validate and copy the closed Embed options for one entry point."""

    copied = dict(options)
    _reject_sensitive_embed_options(copied)
    family = (provider_family or "").casefold()
    allowed = _EMBED_COMMON_OPTIONS | _EMBED_PROVIDER_OPTIONS.get(family, frozenset())
    if relation:
        allowed |= _EMBED_RELATION_OPTIONS
    unknown = sorted(set(copied) - allowed)
    if unknown:
        raise TypeError(
            f"Unsupported Embed option(s) for provider {provider_family or 'custom'!r}: " + ", ".join(unknown)
        )

    if "normalize" in copied and not isinstance(copied["normalize"], bool):
        raise ValueError("Embed option 'normalize' must be a bool")
    for name in ("batch_size", "actor_number", "batch_token_limit"):
        _require_embed_int(copied, name, minimum=1)
    if copied.get("input_text_token_limit") is not None:
        _require_embed_int(copied, "input_text_token_limit", minimum=1)
    _require_embed_int(copied, "max_retries", minimum=0)

    backend = copied.get("execution_backend")
    if backend is not None:
        if not isinstance(backend, str) or backend not in _EXECUTION_BACKENDS:
            raise ValueError(
                "Embed option 'execution_backend' must be one of: "
                "subprocess_task, subprocess_actor, ray_task, ray_actor"
            )
        if "actor_number" in copied and backend in {"subprocess_task", "ray_task"}:
            raise ValueError("Embed option 'actor_number' requires an actor execution backend")

    max_chunk_chars = copied.get("max_chunk_chars")
    if max_chunk_chars is not None:
        if family == "openai" and copied.get("input_text_token_limit") is not None:
            raise ValueError("Embed options 'max_chunk_chars' and 'input_text_token_limit' cannot be used together")
        _require_embed_int(copied, "max_chunk_chars", minimum=1)
        overlap = copied.get("chunk_overlap_chars", 200)
        if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
            raise ValueError("Embed option 'chunk_overlap_chars' must be an integer >= 0")
        if overlap >= max_chunk_chars:
            raise ValueError("Embed option 'chunk_overlap_chars' must be smaller than max_chunk_chars")
    elif "chunk_overlap_chars" in copied:
        raise ValueError("Embed option 'chunk_overlap_chars' requires max_chunk_chars")

    if family == "openai":
        encoding_format = copied.get("encoding_format", "float")
        if encoding_format not in {"float", "base64"}:
            raise ValueError("Embed option 'encoding_format' must be 'float' or 'base64'")
        _validate_base_url_option(copied, api="Embed")
        timeout = copied.get("timeout")
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise ValueError("Embed option 'timeout' must be a finite positive number or None")
            if not math.isfinite(float(timeout)) or float(timeout) <= 0:
                raise ValueError("Embed option 'timeout' must be a finite positive number or None")

    if family == "google":
        task_type = copied.get("task_type")
        if task_type is not None and task_type not in _GOOGLE_EMBED_TASK_TYPES:
            raise ValueError(f"Embed option 'task_type' must be one of {sorted(_GOOGLE_EMBED_TASK_TYPES)} or None")
        _require_optional_nonempty_string(copied, "title")
        if copied.get("title") is not None and task_type != "RETRIEVAL_DOCUMENT":
            raise ValueError("Embed option 'title' is only valid with task_type='RETRIEVAL_DOCUMENT'")

    if family == "transformers":
        for name in ("cache_folder", "device", "revision"):
            _require_optional_nonempty_string(copied, name)
        for name in ("local_files_only", "trust_remote_code"):
            if name in copied and not isinstance(copied[name], bool):
                raise ValueError(f"Embed option {name!r} must be a bool")
        if copied.get("trust_remote_code") is True:
            revision = copied.get("revision")
            if not _is_hugging_face_commit_sha(revision):
                raise ValueError(
                    "Embed option 'trust_remote_code=True' requires a pinned revision as a full 40-character commit SHA"
                )

    return copied


_PROMPT_COMMON_OPTIONS = frozenset({"temperature", "batch_size", "actor_number", "max_retries"})
_PROMPT_RELATION_OPTIONS = frozenset({"execution_backend"})
_PROMPT_REMOTE_OPTIONS = frozenset({"max_concurrency_per_actor"})
_PROMPT_PROVIDER_OPTIONS = {
    "openai": frozenset(
        {"use_chat_completions", "max_output_tokens", "top_p", "stop_sequences", "base_url", "timeout"}
    ),
    "anthropic": frozenset({"max_tokens", "top_p", "top_k", "stop_sequences", "base_url", "timeout"}),
    "google": frozenset({"max_output_tokens", "top_p", "top_k", "stop_sequences"}),
    "vllm": frozenset(
        {
            "max_tokens",
            "gpus_per_actor",
            "engine_args",
            "generate_args",
            "do_prefix_routing",
            "max_buffer_size",
            "min_bucket_size",
            "prefix_match_threshold",
            "load_balance_threshold",
            "inflight_limit",
            "engine_init_timeout_s",
        }
    ),
}


def _reject_sensitive_prompt_options(value: Any, path: str = "options") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Prompt option {path} must use string keys; got {type(key).__name__}")
            if is_sensitive_option_key(key, _OPENAI_EXTRA_SENSITIVE_KEYS):
                raise ValueError(
                    f"Prompt options cannot include sensitive field {path}.{key}; "
                    "configure credentials through the environment or runtime secret management"
                )
            _reject_sensitive_prompt_options(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_sensitive_prompt_options(item, f"{path}[{index}]")


def _require_prompt_int(options: Mapping[str, Any], name: str, *, minimum: int, nullable: bool = False) -> None:
    if name not in options:
        return
    value = options[name]
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "a positive integer" if minimum == 1 else f"an integer >= {minimum}"
        if nullable:
            qualifier += " or None"
        raise ValueError(f"Prompt option {name!r} must be {qualifier}")


def _require_prompt_number(
    options: Mapping[str, Any],
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    nullable: bool = False,
) -> None:
    if name not in options:
        return
    value = options[name]
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Prompt option {name!r} must be a finite number" + (" or None" if nullable else ""))
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"Prompt option {name!r} must be >= {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"Prompt option {name!r} must be <= {maximum:g}")


def _validate_prompt_stop_sequences(options: Mapping[str, Any]) -> None:
    if "stop_sequences" not in options or options["stop_sequences"] is None:
        return
    value = options["stop_sequences"]
    if not isinstance(value, list) or not value:
        raise ValueError("Prompt option 'stop_sequences' must be a non-empty string list or None")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError("Prompt option 'stop_sequences' must contain only non-empty strings")


def _validate_vllm_json(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Prompt option {path} must contain only finite numbers")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Prompt option {path} must use string keys; got {type(key).__name__}")
            if key in {"structured_outputs", "guided_decoding"}:
                raise ValueError(f"Prompt option generate_args cannot configure {key} directly; use return_format")
            _validate_vllm_json(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_vllm_json(item, f"{path}[{index}]")
        return
    raise TypeError(f"Prompt option {path} must be JSON-compatible; got {type(value).__name__}")


def validate_prompt_options(
    provider_family: str | None,
    options: Mapping[str, Any],
    *,
    relation: bool,
) -> dict[str, Any]:
    """Validate and copy the closed Prompt options for one entry point."""

    copied = dict(options)
    _reject_sensitive_prompt_options(copied)
    family = (provider_family or "").casefold()
    allowed = _PROMPT_COMMON_OPTIONS | _PROMPT_PROVIDER_OPTIONS.get(family, frozenset())
    if family != "vllm":
        allowed |= _PROMPT_REMOTE_OPTIONS
    if relation and family != "vllm":
        allowed |= _PROMPT_RELATION_OPTIONS
    unknown = sorted(set(copied) - allowed)
    if unknown:
        raise TypeError(
            f"Unsupported Prompt option(s) for provider {provider_family or 'custom'!r}: " + ", ".join(unknown)
        )

    _require_prompt_number(copied, "temperature", nullable=True)
    for name in ("batch_size", "actor_number", "max_concurrency_per_actor"):
        _require_prompt_int(copied, name, minimum=1)
    _require_prompt_int(copied, "max_retries", minimum=0)

    backend = copied.get("execution_backend")
    if backend is not None:
        if not isinstance(backend, str) or backend not in _EXECUTION_BACKENDS:
            raise ValueError(
                "Prompt option 'execution_backend' must be one of: "
                "subprocess_task, subprocess_actor, ray_task, ray_actor"
            )
        if "actor_number" in copied and backend in {"subprocess_task", "ray_task"}:
            raise ValueError("Prompt option 'actor_number' requires an actor execution backend")

    if family == "openai":
        if "use_chat_completions" in copied and not isinstance(copied["use_chat_completions"], bool):
            raise ValueError("Prompt option 'use_chat_completions' must be a bool")
        _require_prompt_int(copied, "max_output_tokens", minimum=1, nullable=True)
    elif family == "anthropic":
        _require_prompt_int(copied, "max_tokens", minimum=0, nullable=True)
        _require_prompt_int(copied, "top_k", minimum=0, nullable=True)
    elif family == "google":
        _require_prompt_int(copied, "max_output_tokens", minimum=1, nullable=True)
        _require_prompt_int(copied, "top_k", minimum=0, nullable=True)
    elif family == "vllm":
        _require_prompt_int(copied, "max_tokens", minimum=1, nullable=True)
        for name in ("max_buffer_size", "min_bucket_size", "load_balance_threshold", "inflight_limit"):
            _require_prompt_int(copied, name, minimum=0)
        _require_prompt_number(copied, "prefix_match_threshold", minimum=0, maximum=1)
        _require_prompt_number(copied, "engine_init_timeout_s", minimum=0, nullable=True)
        if "gpus_per_actor" in copied:
            _require_prompt_number(copied, "gpus_per_actor", minimum=0)
            value = float(copied["gpus_per_actor"])
            if value <= 0 or (value >= 1 and not value.is_integer()):
                raise ValueError("Prompt option 'gpus_per_actor' must be positive and must be an integer when >= 1")
        if "do_prefix_routing" in copied and not isinstance(copied["do_prefix_routing"], bool):
            raise ValueError("Prompt option 'do_prefix_routing' must be a bool")
        for name in ("engine_args", "generate_args"):
            if name in copied:
                value = copied[name]
                if not isinstance(value, Mapping):
                    raise TypeError(f"Prompt option {name!r} must be a mapping")
                _validate_vllm_json(value, name)
        generate_args = copied.get("generate_args", {})
        sampling_params = generate_args.get("sampling_params")
        if isinstance(sampling_params, str):
            try:
                decoded_sampling_params = json.loads(sampling_params)
            except json.JSONDecodeError as exc:
                raise ValueError("Prompt option 'generate_args.sampling_params' must be valid JSON") from exc
            if not isinstance(decoded_sampling_params, Mapping):
                raise ValueError("Prompt option 'generate_args.sampling_params' JSON must decode to a mapping")
            _reject_sensitive_prompt_options(
                decoded_sampling_params,
                path="options.generate_args.sampling_params",
            )
            _validate_vllm_json(decoded_sampling_params, "generate_args.sampling_params")
        engine_args = copied.get("engine_args", {})
        if "trust_remote_code" in engine_args and not isinstance(engine_args["trust_remote_code"], bool):
            raise ValueError("Prompt option 'engine_args.trust_remote_code' must be a bool")
        if engine_args.get("trust_remote_code") is True:
            for revision_name in ("code_revision", "tokenizer_revision"):
                if not _is_hugging_face_commit_sha(engine_args.get(revision_name)):
                    raise ValueError(
                        "Prompt option 'engine_args.trust_remote_code=True' requires "
                        f"engine_args.{revision_name} as a full 40-character commit SHA"
                    )
            speculative_config = engine_args.get("speculative_config")
            speculative_model = speculative_config.get("model") if isinstance(speculative_config, Mapping) else None
            speculative_method = speculative_config.get("method") if isinstance(speculative_config, Mapping) else None
            if (
                isinstance(speculative_config, Mapping)
                and (
                    (speculative_model is not None and speculative_model not in {"ngram", "[ngram]"})
                    or speculative_method in _VLLM_REMOTE_CODE_SPECULATIVE_METHODS
                )
                and not _is_hugging_face_commit_sha(speculative_config.get("code_revision"))
            ):
                raise ValueError(
                    "Prompt option 'engine_args.trust_remote_code=True' with a speculative model requires "
                    "engine_args.speculative_config.code_revision as a full 40-character commit SHA"
                )
        if copied.get("max_retries", 0) != 0:
            raise ValueError("native vLLM prompting only accepts max_retries=0")

    if family in {"openai", "anthropic", "google"}:
        _require_prompt_number(copied, "top_p", minimum=0, maximum=1, nullable=True)
        _validate_prompt_stop_sequences(copied)
        _validate_base_url_option(copied, api="Prompt")
        _require_prompt_number(copied, "timeout", minimum=0, nullable=True)
        if copied.get("timeout") == 0:
            raise ValueError("Prompt option 'timeout' must be a finite positive number or None")

    return copied
