# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Typed option objects for high-level AI helper functions."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, TypedDict
from urllib.parse import parse_qsl, urlsplit

from vane.ai._litert_options import LITERT_TEXT_OPTIONS, litert_option_names, validate_litert_options
from vane.ai._redaction import is_sensitive_option_key

VLLMJSONPrimitive: TypeAlias = str | int | float | bool | None
VLLMJSONValue: TypeAlias = VLLMJSONPrimitive | list["VLLMJSONValue"] | dict[str, "VLLMJSONValue"]


class JevOptions(TypedDict, total=False):
    """TypeSafe SDK request options and Vane Jev execution limits."""

    batch_size: int
    actor_number: int
    max_concurrency_per_actor: int
    execution_backend: Literal["subprocess_task", "subprocess_actor", "ray_task", "ray_actor"] | None
    max_retries: int
    base_url: str | None
    timeout: float | None


class MultimodalEmbedOptions(TypedDict, total=False):
    model_path: str
    model_sha256: str
    backend: Literal["cpu", "gpu"]
    cache_dir: str | None
    num_threads: int
    vision_tokens_per_image: int
    input_modalities: list[str]
    sample_rate: int
    max_samples: int
    max_concurrency_per_actor: int
    instruction: str
    max_frames: int
    max_input_bytes: int
    paired_image_queries: bool
    max_length: int
    max_pixels: int
    processor_kwargs: Mapping[str, VLLMJSONValue]
    pooling_args: Mapping[str, VLLMJSONValue]
    chat_template: str
    chat_template_kwargs: Mapping[str, VLLMJSONValue]
    engine_args: Mapping[str, VLLMJSONValue]
    gpus_per_actor: int
    supports_overriding_dimensions: bool


class PromptOptions(TypedDict, total=False):
    """Closed keyword surface shared by the Python Prompt entry points."""

    # Provider request option shared by every built-in Prompt adapter.
    temperature: float | None

    # Vane Prompt execution options.
    batch_size: int
    actor_number: int
    max_concurrency_per_actor: int
    execution_backend: Literal["subprocess_task", "subprocess_actor", "ray_task", "ray_actor"] | None
    max_retries: int

    # OpenAI / OpenAI-compatible prompt options.
    use_chat_completions: bool
    base_url: str | None
    timeout: float | None
    max_output_tokens: int | None
    top_p: float | None
    stop_sequences: list[str] | None

    # Anthropic keeps its native output-token name.
    max_tokens: int | None
    # SGLang keeps its native output-token name.
    max_new_tokens: int | None
    top_k: int | None

    # Native model capabilities, validated before loading any weights.
    media_mime_types: list[str]
    chat_template: str
    chat_template_kwargs: Mapping[str, VLLMJSONValue]
    max_input_bytes: int
    max_frames: int

    # Native vLLM provider options.
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


class EmbedOptions(MultimodalEmbedOptions, total=False):
    """Closed keyword surface shared by the Python Embed entry points."""

    normalize: bool
    batch_size: int
    actor_number: int
    execution_backend: Literal["subprocess_task", "subprocess_actor", "ray_task", "ray_actor"] | None
    max_retries: int
    max_chunk_chars: int | None
    chunk_overlap_chars: int

    # Inference providers: choose a remote service explicitly instead of an engine.
    runtime: Literal["native", "http", "service"]
    service_path: str
    service_id: str
    http_input_format: Literal["text", "multimodal", "encode"]
    http_media_tokens: Mapping[str, str]
    http_add_special_tokens: bool
    media_mime_types: list[str]
    max_request_bytes: int

    # Remote embedding request batching (OpenAI and Google).
    request_batch_size: int

    # Retrieval encoding and explicit long-input handling.
    input_type: Literal["query", "document"]
    prompt_name: str
    prompt: str
    overlength: Literal["error", "truncate", "chunk_mean"]

    # OpenAI / OpenAI-compatible embedding options.
    base_url: str | None
    timeout: float | None
    encoding_format: Literal["float", "base64"]
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
    dtype: Literal["float32", "float16", "bfloat16"]


_MLX_COMMON_DEFAULTS: dict[str, Any] = {
    "revision": "1a4ffddb7905d3f63486748deabe091a01fb6201",
    "dtype": "bfloat16",
    "device": "metal",
    "local_files_only": True,
    "trust_remote_code": False,
    "max_length": 2048,
    "overlength": "error",
    "max_concurrency_per_actor": 1,
}
_MLX_TEXT_DEFAULTS = _MLX_COMMON_DEFAULTS | {"input_type": "document"}


_EMBED_COMMON_OPTIONS = frozenset({"normalize", "batch_size", "actor_number", "max_retries"})
_EMBED_RELATION_OPTIONS = frozenset({"execution_backend", "max_chunk_chars", "chunk_overlap_chars"})
_EMBED_REMOTE_OPTIONS = frozenset({"request_batch_size", "max_concurrency_per_actor"})
_HTTP_EMBED_OPTIONS = _EMBED_REMOTE_OPTIONS | frozenset(
    {
        "runtime",
        "base_url",
        "timeout",
        "http_input_format",
        "http_media_tokens",
        "http_add_special_tokens",
        "media_mime_types",
        "max_input_bytes",
        "max_request_bytes",
        "supports_overriding_dimensions",
    }
)


_SERVICE_EMBED_OPTIONS = frozenset(
    {
        "runtime",
        "service_path",
        "service_id",
        "timeout",
        "max_concurrency_per_actor",
        "input_modalities",
        "max_frames",
        "max_input_bytes",
        "sample_rate",
        "max_samples",
    }
)

_MLX_SERVICE_OPTIONS = frozenset(
    {
        "runtime",
        "service_path",
        "service_id",
        "timeout",
        "max_concurrency_per_actor",
        "revision",
        "dtype",
        "max_length",
        "overlength",
    }
)


def _mlx_embed_option_names(options: Mapping[str, Any], *, text: bool) -> frozenset[str]:
    from vane.ai._embedding_inputs import EmbeddingConfigurationError

    runtime = options.get("runtime", "native")
    if runtime == "service":
        return _MLX_SERVICE_OPTIONS | ({"input_type"} if text else set())
    if runtime != "native":
        raise EmbeddingConfigurationError("MLX runtime must be native or service")
    return frozenset(_MLX_TEXT_DEFAULTS if text else _MLX_COMMON_DEFAULTS) | {"runtime"}


def _validate_mlx_service_options(options: Mapping[str, Any], *, text: bool) -> dict[str, Any]:
    from vane.ai._embedding_inputs import EmbeddingConfigurationError

    copied = dict(options)
    _reject_sensitive_embed_options(copied)
    if copied.get("runtime") != "service" or copied.keys() - _mlx_embed_option_names(copied, text=text):
        raise TypeError(
            "MLX service accepts connection and encoding settings; place loading settings in the service config"
        )
    names = {"revision", "dtype", "max_length", "overlength"} | ({"input_type"} if text else set())
    validator = _validate_mlx_text_options if text else _validate_mlx_multimodal_options
    native = validator({key: value for key, value in copied.items() if key in names})
    if native["revision"] != _MLX_COMMON_DEFAULTS["revision"]:
        raise EmbeddingConfigurationError("MLX service requires the fixed checkpoint revision")
    return copied | {key: native[key] for key in names}


def _inference_embed_options(options: Mapping[str, Any]) -> frozenset[str]:
    runtime = options.get("runtime", "native")
    if runtime not in ("native", "http", "service"):
        raise ValueError("Embed option 'runtime' must be 'native', 'http' or 'service'")
    if runtime == "service":
        return _SERVICE_EMBED_OPTIONS
    return _HTTP_EMBED_OPTIONS if runtime == "http" else _NATIVE_EMBED_OPTIONS | {"runtime"}


_NATIVE_EMBED_OPTIONS = frozenset(
    {
        "input_modalities",
        "sample_rate",
        "max_samples",
        "max_concurrency_per_actor",
        "instruction",
        "max_frames",
        "max_input_bytes",
        "paired_image_queries",
        "processor_kwargs",
        "pooling_args",
        "chat_template",
        "chat_template_kwargs",
        "engine_args",
        "gpus_per_actor",
        "supports_overriding_dimensions",
    }
)
_NATIVE_MEDIA_PROMPT_OPTIONS = frozenset(
    {
        "media_mime_types",
        "temperature",
        "max_tokens",
        "top_p",
        "stop_sequences",
        "max_frames",
        "max_input_bytes",
        "engine_args",
        "gpus_per_actor",
        "generate_args",
        "chat_template",
        "chat_template_kwargs",
    }
)
_EMBED_PROVIDER_OPTIONS = {
    "litert": LITERT_TEXT_OPTIONS,
    "mlx": frozenset(_MLX_TEXT_DEFAULTS),
    "vllm": _NATIVE_EMBED_OPTIONS,
    "sglang": _NATIVE_EMBED_OPTIONS,
    "openai": _EMBED_REMOTE_OPTIONS
    | frozenset(
        {
            "encoding_format",
            "base_url",
            "timeout",
            "batch_token_limit",
            "input_text_token_limit",
            "supports_overriding_dimensions",
            "overlength",
        }
    ),
    "google": _EMBED_REMOTE_OPTIONS | frozenset({"task_type", "title", "input_type"}),
    "transformers": frozenset(
        {
            "cache_folder",
            "device",
            "local_files_only",
            "revision",
            "trust_remote_code",
            "dtype",
            "instruction",
            "max_frames",
            "max_input_bytes",
            "max_length",
            "max_pixels",
            "input_type",
            "prompt_name",
            "prompt",
            "overlength",
            "max_concurrency_per_actor",
        }
    ),
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


def _is_hugging_face_commit_sha(value: Any) -> bool:
    return isinstance(value, str) and _HUGGING_FACE_COMMIT_SHA.fullmatch(value) is not None


def _validate_base_url_option(options: Mapping[str, Any], *, api: Literal["Embed", "Prompt", "Jev"]) -> None:
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


def _validate_mlx_common_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Shared encoding settings, without a text retrieval role."""
    from vane.ai._embedding_inputs import EmbeddingConfigurationError

    copied = dict(options)
    if copied.pop("runtime", "native") != "native":
        raise EmbeddingConfigurationError("MLX native encoders require runtime=native")
    _reject_sensitive_embed_options(copied)
    if copied.keys() - _MLX_COMMON_DEFAULTS.keys():
        raise TypeError("MLX media options accept only the declared common model encoding settings")
    prepared = _MLX_COMMON_DEFAULTS | copied
    if not _is_hugging_face_commit_sha(prepared["revision"]):
        raise EmbeddingConfigurationError("MLX revision must be a full 40-character commit SHA")
    prepared["revision"] = prepared["revision"].lower()
    for key, choices in {
        "dtype": ("bfloat16",),
        "device": ("metal",),
        "overlength": ("error",),
    }.items():
        if not isinstance(prepared[key], str) or prepared[key] not in choices:
            raise EmbeddingConfigurationError(f"MLX {key} must be one of: {', '.join(choices)}")
    if type(prepared["local_files_only"]) is not bool:
        raise EmbeddingConfigurationError("MLX local_files_only must be a bool")
    if prepared["trust_remote_code"] is not False:
        raise EmbeddingConfigurationError("MLX requires trust_remote_code=False")
    if type(prepared["max_length"]) is not int or not 1 <= prepared["max_length"] <= 2048:
        raise EmbeddingConfigurationError("MLX max_length must be an integer from 1 to 2048")
    if type(prepared["max_concurrency_per_actor"]) is not int or prepared["max_concurrency_per_actor"] != 1:
        raise EmbeddingConfigurationError("MLX requires max_concurrency_per_actor=1")
    return prepared


def _validate_mlx_text_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """The unchanged text encoding contract for factories and public Embed."""
    from vane.ai._embedding_inputs import EmbeddingConfigurationError

    copied = dict(options)
    if copied.pop("runtime", "native") != "native":
        raise EmbeddingConfigurationError("MLX native encoders require runtime=native")
    _reject_sensitive_embed_options(copied)
    if copied.keys() - _MLX_TEXT_DEFAULTS.keys():
        raise TypeError("MLX text options accept only the declared model encoding settings")
    role = copied.pop("input_type", "document")
    if not isinstance(role, str) or role not in ("document", "query"):
        raise EmbeddingConfigurationError("MLX input_type must be one of: document, query")
    return _validate_mlx_common_options(copied) | {"input_type": role}


def _validate_mlx_image_options(options: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_mlx_common_options(options)


def _validate_mlx_audio_options(options: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_mlx_common_options(options)


def _validate_mlx_video_options(options: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_mlx_common_options(options)


def _validate_mlx_multimodal_options(options: Mapping[str, Any]) -> dict[str, Any]:
    prepared = _validate_mlx_common_options(options)
    if prepared["revision"] != _MLX_COMMON_DEFAULTS["revision"]:
        from vane.ai._embedding_inputs import EmbeddingConfigurationError

        raise EmbeddingConfigurationError("MLX joint encoding requires the fixed checkpoint revision")
    return prepared


def validate_embed_options(
    provider_family: str | None,
    options: Mapping[str, Any],
    *,
    relation: bool,
    _input_kind: Literal["text", "image", "audio", "video", "multimodal"] = "text",
) -> dict[str, Any]:
    """Validate and copy the closed Embed options for one entry point."""

    copied = dict(options)
    _reject_sensitive_embed_options(copied)
    family = (provider_family or "").casefold()
    allowed = _EMBED_COMMON_OPTIONS | _EMBED_PROVIDER_OPTIONS.get(family, frozenset())
    if family in {"sglang", "vllm"}:
        allowed = _EMBED_COMMON_OPTIONS | _inference_embed_options(copied)
    if family == "mlx":
        allowed = _EMBED_COMMON_OPTIONS | _mlx_embed_option_names(copied, text=_input_kind == "text")
    if family == "litert":
        allowed = _EMBED_COMMON_OPTIONS | litert_option_names(
            text=_input_kind == "text", service=copied.get("runtime") == "service", http=copied.get("runtime") == "http"
        )
    if relation:
        allowed |= _EMBED_RELATION_OPTIONS
    unknown = sorted(set(copied) - allowed)
    if unknown:
        raise TypeError(
            f"Unsupported Embed option(s) for provider {provider_family or 'custom'!r}: " + ", ".join(unknown)
        )

    if family == "litert":
        from vane.ai._embedding_inputs import EmbeddingConfigurationError

        actor_number = copied.get("actor_number")
        service = copied.get("runtime") == "service"
        http = copied.get("runtime") == "http"
        if not service and not http:
            if actor_number is not None and (type(actor_number) is not int or actor_number != 1):
                raise EmbeddingConfigurationError("LiteRT requires actor_number=1")
            if copied.get("execution_backend") not in (None, "subprocess_actor"):
                raise EmbeddingConfigurationError("LiteRT requires execution_backend='subprocess_actor'")
        if copied.keys() & {"max_chunk_chars", "chunk_overlap_chars"}:
            raise EmbeddingConfigurationError("LiteRT does not support character chunk averaging")
        names = litert_option_names(text=_input_kind == "text", service=service, http=http)
        copied.update(
            validate_litert_options(
                {key: value for key, value in copied.items() if key in names},
                text=_input_kind == "text",
                service=service,
                http=http,
            )
        )
        if actor_number is None:
            copied.pop("actor_number", None)

    if family == "mlx":
        from vane.ai._embedding_inputs import EmbeddingConfigurationError

        if copied.keys() & {"max_chunk_chars", "chunk_overlap_chars"}:
            raise EmbeddingConfigurationError("MLX does not support character chunk averaging")
        if copied.get("runtime", "native") == "service":
            names = _mlx_embed_option_names(copied, text=_input_kind == "text")
            copied.update(
                _validate_mlx_service_options(
                    {k: v for k, v in copied.items() if k in names}, text=_input_kind == "text"
                )
            )
        else:
            actor_number = copied.get("actor_number")
            if actor_number is not None and (type(actor_number) is not int or actor_number != 1):
                raise EmbeddingConfigurationError("MLX requires actor_number=1")
            if copied.get("execution_backend") not in (None, "subprocess_actor"):
                raise EmbeddingConfigurationError("MLX requires execution_backend='subprocess_actor'")
            defaults = _MLX_TEXT_DEFAULTS if _input_kind == "text" else _MLX_COMMON_DEFAULTS
            validator = (
                _validate_mlx_text_options
                if _input_kind == "text"
                else _validate_mlx_multimodal_options
                if _input_kind == "multimodal"
                else _validate_mlx_common_options
            )
            copied.update(validator({k: v for k, v in copied.items() if k in defaults}))
            copied.pop("runtime", None)
            if actor_number is None:
                copied.pop("actor_number", None)

    if "normalize" in copied and not isinstance(copied["normalize"], bool):
        raise ValueError("Embed option 'normalize' must be a bool")
    for name in ("batch_size", "actor_number", "batch_token_limit", "request_batch_size", "max_concurrency_per_actor"):
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

    if "supports_overriding_dimensions" in copied and not isinstance(copied["supports_overriding_dimensions"], bool):
        raise ValueError("Embed option 'supports_overriding_dimensions' must be a bool")
    if family in {"sglang", "vllm"} and copied.get("runtime") == "http":
        _validate_base_url_option(copied, api="Embed")
        if not copied.get("base_url") or urlsplit(copied["base_url"]).query:
            raise ValueError("HTTP embedding requires an explicit base_url without query parameters")
        wire_format = copied.get("http_input_format", "text")
        if wire_format not in ("text", "multimodal", "encode"):
            raise ValueError("http_input_format must be 'text', 'multimodal' or 'encode'")
        if wire_format == "encode" and family != "sglang":
            raise ValueError("http_input_format='encode' requires the SGLang provider")
        if "http_add_special_tokens" in copied:
            if family != "vllm" or not isinstance(copied["http_add_special_tokens"], bool):
                raise ValueError("http_add_special_tokens requires the vLLM HTTP protocol and a bool")
        if "http_media_tokens" in copied:
            tokens = copied["http_media_tokens"]
            if wire_format != "encode" or not isinstance(tokens, Mapping):
                raise ValueError("http_media_tokens requires the SGLang encode format and a mapping")
            if any(
                key not in {"image", "video", "audio"}
                or not isinstance(value, str)
                or not value.strip()
                or len(value) > 1024
                or "\x00" in value
                for key, value in tokens.items()
            ) or len(set(tokens.values())) != len(tokens):
                raise ValueError("http_media_tokens requires distinct nonempty image/video/audio markers")
            copied["http_media_tokens"] = dict(tokens)
        timeout = copied.get("timeout", 60.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("HTTP embedding timeout must be finite and positive")
        for name in ("max_input_bytes", "max_request_bytes"):
            _require_embed_int(copied, name, minimum=1)
            if copied.get(name, 1) > 256 * 1024**2:
                raise ValueError(f"HTTP embedding {name} must not exceed 256 MiB")
        if copied.get("max_concurrency_per_actor", 4) > 1024:
            raise ValueError("HTTP embedding max_concurrency_per_actor must not exceed 1024")
        mime_types = copied.get("media_mime_types", [])
        if not isinstance(mime_types, list) or any(not isinstance(value, str) for value in mime_types):
            raise TypeError("media_mime_types must be a list of MIME type strings")
        if mime_types and wire_format == "text":
            raise ValueError("media_mime_types requires a multimodal HTTP format")
        if (
            wire_format == "encode"
            and {mime.split("/", 1)[0] for mime in mime_types} - copied.get("http_media_tokens", {}).keys()
        ):
            raise ValueError("SGLang encode requires http_media_tokens for each declared media modality")
    if "input_type" in copied and copied["input_type"] not in ("query", "document"):
        raise ValueError("Embed option 'input_type' must be 'query' or 'document'")
    if "overlength" in copied and copied["overlength"] not in ("error", "truncate", "chunk_mean"):
        raise ValueError("Embed option 'overlength' must be 'error', 'truncate', or 'chunk_mean'")
    if copied.get("overlength") is not None and copied.get("max_chunk_chars") is not None:
        raise ValueError("Embed options 'overlength' and 'max_chunk_chars' cannot be used together")
    if family == "google" and "input_type" in copied:
        if copied.get("task_type") is not None:
            raise ValueError("Embed options 'input_type' and 'task_type' cannot be used together")
        copied["task_type"] = "RETRIEVAL_QUERY" if copied.pop("input_type") == "query" else "RETRIEVAL_DOCUMENT"
    if family == "transformers":
        if "dtype" in copied and copied["dtype"] not in ("float32", "float16"):
            raise ValueError("Embed option 'dtype' must be 'float32' or 'float16'")
        if copied.get("max_concurrency_per_actor", 1) != 1:
            raise ValueError("Transformers Embed requires max_concurrency_per_actor=1")
        if sum(name in copied for name in ("input_type", "prompt_name", "prompt")) > 1:
            raise ValueError("Embed options 'input_type', 'prompt_name', and 'prompt' are mutually exclusive")
        for name in ("prompt_name", "prompt"):
            if name in copied and (not isinstance(copied[name], str) or not copied[name]):
                raise ValueError(f"Embed option {name!r} must be a non-empty string")

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


def validate_embed_image_options(
    provider_family: str | None, options: Mapping[str, Any], *, relation: bool
) -> dict[str, Any]:
    """Share execution/loading validation without accepting text-only options."""
    allowed = _EMBED_COMMON_OPTIONS
    if provider_family == "litert":
        allowed |= litert_option_names(
            text=False, service=options.get("runtime") == "service", http=options.get("runtime") == "http"
        )
    if provider_family == "mlx":
        allowed |= _mlx_embed_option_names(options, text=False)
    if provider_family == "google":
        allowed |= {"max_concurrency_per_actor"}
    if provider_family == "transformers":
        allowed |= frozenset(
            {
                "cache_folder",
                "device",
                "local_files_only",
                "revision",
                "trust_remote_code",
                "dtype",
                "instruction",
                "max_frames",
                "max_input_bytes",
                "max_length",
                "max_pixels",
            }
        )
    if provider_family in {"vllm", "sglang"}:
        allowed |= _inference_embed_options(options)
    if relation:
        allowed |= {"execution_backend"}
    _reject_sensitive_embed_options(options)
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise TypeError("Unsupported EmbedImage option(s): " + ", ".join(unknown))
    return validate_embed_options(provider_family, options, relation=relation, _input_kind="image")


def validate_embed_audio_options(
    provider_family: str | None, options: Mapping[str, Any], *, relation: bool
) -> dict[str, Any]:
    """Share execution/loading validation without accepting text-only options."""
    allowed = _EMBED_COMMON_OPTIONS
    if provider_family == "litert":
        return validate_embed_options(provider_family, options, relation=relation, _input_kind="audio")
    if provider_family == "mlx":
        allowed |= _mlx_embed_option_names(options, text=False)
    if provider_family == "google":
        allowed |= {"max_concurrency_per_actor"}
    if provider_family == "transformers":
        allowed |= frozenset({"cache_folder", "device", "local_files_only", "revision", "trust_remote_code"})
    if provider_family in {"vllm", "sglang"}:
        allowed |= _inference_embed_options(options)
    if relation:
        allowed |= {"execution_backend"}
    _reject_sensitive_embed_options(options)
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise TypeError("Unsupported EmbedAudio option(s): " + ", ".join(unknown))
    return validate_embed_options(
        provider_family, options, relation=relation, _input_kind="audio" if provider_family == "mlx" else "text"
    )


def validate_embed_video_options(
    provider_family: str | None, options: Mapping[str, Any], *, relation: bool
) -> dict[str, Any]:
    """Video providers cannot inherit text chunking or image loading policies."""
    allowed = _EMBED_COMMON_OPTIONS
    if provider_family == "litert":
        allowed |= litert_option_names(
            text=False, service=options.get("runtime") == "service", http=options.get("runtime") == "http"
        )
    if provider_family == "mlx":
        allowed |= _mlx_embed_option_names(options, text=False)
    if provider_family == "google":
        allowed |= {"max_concurrency_per_actor"}
    if provider_family == "transformers":
        allowed |= frozenset(
            {
                "cache_folder",
                "device",
                "local_files_only",
                "revision",
                "trust_remote_code",
                "dtype",
                "instruction",
                "max_frames",
                "max_input_bytes",
                "max_length",
                "max_pixels",
            }
        )
    if provider_family in {"vllm", "sglang"}:
        allowed |= _inference_embed_options(options)
    if relation:
        allowed |= {"execution_backend"}
    _reject_sensitive_embed_options(options)
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise TypeError("Unsupported EmbedVideo option(s): " + ", ".join(unknown))
    return validate_embed_options(
        provider_family,
        options,
        relation=relation,
        _input_kind="video" if provider_family in {"mlx", "litert"} else "text",
    )


def validate_embed_multimodal_options(
    provider_family: str | None, options: Mapping[str, Any], *, relation: bool
) -> dict[str, Any]:
    if provider_family == "mlx":
        return validate_embed_options(provider_family, options, relation=relation, _input_kind="multimodal")
    return validate_embed_image_options(provider_family, options, relation=relation)


_PROMPT_SHARED_PROVIDER_OPTIONS = frozenset({"temperature"})
_PROMPT_BASE_EXECUTION_OPTIONS = frozenset({"batch_size", "actor_number", "max_retries"})
_PROMPT_RELATION_EXECUTION_OPTIONS = frozenset({"execution_backend"})
_PROMPT_REMOTE_EXECUTION_OPTIONS = frozenset({"max_concurrency_per_actor"})
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
    "sglang": frozenset(
        {
            "max_tokens",
            "max_new_tokens",
            "gpus_per_actor",
            "engine_args",
            "generate_args",
            "max_buffer_size",
            "inflight_limit",
            "load_balance_threshold",
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


def normalize_prompt_options(
    provider_family: str | None,
    options: Mapping[str, Any],
    *,
    relation: bool,
) -> dict[str, Any]:
    """Copy and validate the outer Prompt option envelope.

    Provider adapters own the semantics of every option they consume. This
    outer normalization boundary only applies Vane-wide safety and closed-key
    checks, then validates the execution fields consumed by Prompt planning.
    """

    copied = dict(options)
    _reject_sensitive_prompt_options(copied)
    family = (provider_family or "").casefold()
    media_inference = family in {"vllm", "sglang"} and "media_mime_types" in copied
    provider_options = (
        _NATIVE_MEDIA_PROMPT_OPTIONS if media_inference else _PROMPT_PROVIDER_OPTIONS.get(family, frozenset())
    )
    allowed = _PROMPT_SHARED_PROVIDER_OPTIONS | _PROMPT_BASE_EXECUTION_OPTIONS | provider_options
    if family not in {"vllm", "sglang"} or media_inference:
        allowed |= _PROMPT_REMOTE_EXECUTION_OPTIONS
    if relation and (family not in {"vllm", "sglang"} or media_inference):
        allowed |= _PROMPT_RELATION_EXECUTION_OPTIONS
    unknown = sorted(set(copied) - allowed)
    if unknown:
        raise TypeError(
            f"Unsupported Prompt option(s) for provider {provider_family or 'custom'!r}: " + ", ".join(unknown)
        )

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

    if family in {"vllm", "sglang"} and not media_inference and copied.get("max_retries", 0) != 0:
        raise ValueError("native prompting only accepts max_retries=0")

    return copied
