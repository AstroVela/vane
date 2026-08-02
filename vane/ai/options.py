# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Typed option objects for high-level AI helper functions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Literal, TypeAlias, TypedDict
from urllib.parse import parse_qsl, urlsplit

from vane.ai._redaction import REDACTED_PLACEHOLDER, is_sensitive_option_key, wrap_sensitive_options


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
_EMBED_EXECUTION_BACKENDS = frozenset({"subprocess_task", "subprocess_actor", "ray_task", "ray_actor"})
_OPENAI_EMBED_EXTRA_SENSITIVE_KEYS = frozenset({"organization"})


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
        if not isinstance(backend, str) or backend not in _EMBED_EXECUTION_BACKENDS:
            raise ValueError(
                "Embed option 'execution_backend' must be one of: "
                "subprocess_task, subprocess_actor, ray_task, ray_actor"
            )
        if "actor_number" in copied and backend in {"subprocess_task", "ray_task"}:
            raise ValueError("Embed option 'actor_number' requires an actor execution backend")

    max_chunk_chars = copied.get("max_chunk_chars")
    if max_chunk_chars is not None:
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
        base_url = copied.get("base_url")
        if base_url is not None:
            if not isinstance(base_url, str) or not base_url.strip():
                raise ValueError("Embed option 'base_url' must be a non-empty HTTP(S) URL or None")
            parsed = urlsplit(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("Embed option 'base_url' must be a non-empty HTTP(S) URL or None")
            sensitive_query_keys = [
                key
                for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
                if is_sensitive_option_key(key, _OPENAI_EMBED_EXTRA_SENSITIVE_KEYS)
            ]
            if parsed.username or parsed.password or sensitive_query_keys:
                raise ValueError("Embed option 'base_url' cannot contain credentials")
            if parsed.fragment:
                raise ValueError("Embed option 'base_url' cannot contain a URL fragment")
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
        if copied.get("trust_remote_code") is True and copied.get("revision") is None:
            raise ValueError("Embed option 'trust_remote_code=True' requires a pinned revision")

    return copied


VLLMJSONPrimitive: TypeAlias = str | int | float | bool | None
VLLMJSONValue: TypeAlias = VLLMJSONPrimitive | list["VLLMJSONValue"] | dict[str, "VLLMJSONValue"]


def _set_if_not_none(target: dict[str, Any], key: str, value: object) -> None:
    if value is not None:
        target[key] = value


class _RedactedOptionsRepr:
    """Mixin rendering credential-bearing fields redacted in the dataclass-style repr.

    Scalar fields named in ``_REDACTED_FIELDS`` render as a fixed placeholder
    when set (``None`` still renders as ``None`` — masking an absent key would
    be misleading). Mapping-valued fields render with sensitive keys sealed at
    any nesting depth, so nested credentials (e.g. an HF hub token inside
    ``engine_args``) never reach the repr. Dataclasses opting in must be
    declared with ``repr=False`` so the generated repr does not shadow this one.
    """

    _REDACTED_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key"})

    def __repr__(self) -> str:
        parts = []
        for field in fields(self):  # type: ignore[arg-type]
            value = getattr(self, field.name)
            if field.name in self._REDACTED_FIELDS and value is not None:
                parts.append(f"{field.name}={REDACTED_PLACEHOLDER}")
            elif isinstance(value, Mapping):
                parts.append(f"{field.name}={wrap_sensitive_options(value)!r}")
            else:
                parts.append(f"{field.name}={value!r}")
        return f"{type(self).__qualname__}({', '.join(parts)})"


@dataclass(frozen=True, repr=False)
class OpenAIProviderOptions(_RedactedOptionsRepr):
    """OpenAI-compatible provider options shared by prompt and embedding calls."""

    # ``organization`` identifies the paying account; the OpenAI provider seals
    # it at the descriptor layer, so the public repr must not leak it either.
    _REDACTED_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key", "organization"})

    base_url: str | None = None
    api_key: str | None = None
    organization: str | None = None
    timeout: float | None = None
    concurrency: int | None = None
    max_api_concurrency: int | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        _set_if_not_none(options, "base_url", self.base_url)
        _set_if_not_none(options, "api_key", self.api_key)
        _set_if_not_none(options, "organization", self.organization)
        _set_if_not_none(options, "timeout", self.timeout)
        _set_if_not_none(options, "actor_number", self.concurrency)
        _set_if_not_none(options, "max_api_concurrency", self.max_api_concurrency)
        return options


@dataclass(frozen=True, repr=False)
class VLLMProviderOptions(_RedactedOptionsRepr):
    """vLLM provider options for actor count, GPU allocation, and engine args.

    ``engine_args`` crosses the native operator boundary as JSON. Values must
    therefore contain only JSON primitives, string-keyed mappings, and lists.
    """

    engine_args: Mapping[str, VLLMJSONValue] | None = None
    concurrency: int | None = None
    gpus_per_actor: float | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        if self.engine_args is not None:
            options["engine_args"] = dict(self.engine_args)
        _set_if_not_none(options, "actor_number", self.concurrency)
        _set_if_not_none(options, "gpus_per_actor", self.gpus_per_actor)
        return options


@dataclass(frozen=True)
class OpenAIPromptOptions:
    """OpenAI-compatible prompt request options."""

    use_chat_completions: bool | None = None
    max_output_tokens: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    on_error: Literal["raise", "log", "ignore"] | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        _set_if_not_none(options, "use_chat_completions", self.use_chat_completions)
        _set_if_not_none(options, "max_output_tokens", self.max_output_tokens)
        _set_if_not_none(options, "max_tokens", self.max_tokens)
        _set_if_not_none(options, "temperature", self.temperature)
        _set_if_not_none(options, "on_error", self.on_error)
        return options


@dataclass(frozen=True, repr=False)
class AnthropicProviderOptions(_RedactedOptionsRepr):
    """Anthropic provider options for client configuration and execution limits."""

    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    max_retries: int | None = None
    concurrency: int | None = None
    max_api_concurrency: int | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        _set_if_not_none(options, "api_key", self.api_key)
        _set_if_not_none(options, "base_url", self.base_url)
        _set_if_not_none(options, "timeout", self.timeout)
        _set_if_not_none(options, "max_retries", self.max_retries)
        _set_if_not_none(options, "actor_number", self.concurrency)
        _set_if_not_none(options, "max_api_concurrency", self.max_api_concurrency)
        return options


@dataclass(frozen=True)
class AnthropicPromptOptions:
    """Anthropic prompt request options."""

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    on_error: Literal["raise", "log", "ignore"] | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        _set_if_not_none(options, "max_tokens", self.max_tokens)
        _set_if_not_none(options, "temperature", self.temperature)
        _set_if_not_none(options, "top_p", self.top_p)
        _set_if_not_none(options, "top_k", self.top_k)
        if self.stop_sequences is not None:
            options["stop_sequences"] = list(self.stop_sequences)
        _set_if_not_none(options, "on_error", self.on_error)
        return options


@dataclass(frozen=True, repr=False)
class GoogleProviderOptions(_RedactedOptionsRepr):
    """Google provider options for client configuration and execution limits."""

    api_key: str | None = None
    concurrency: int | None = None
    max_api_concurrency: int | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        _set_if_not_none(options, "api_key", self.api_key)
        _set_if_not_none(options, "actor_number", self.concurrency)
        _set_if_not_none(options, "max_api_concurrency", self.max_api_concurrency)
        return options


@dataclass(frozen=True)
class GooglePromptOptions:
    """Google Gemini prompt request options."""

    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    on_error: Literal["raise", "log", "ignore"] | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        _set_if_not_none(options, "max_output_tokens", self.max_output_tokens)
        _set_if_not_none(options, "temperature", self.temperature)
        _set_if_not_none(options, "top_p", self.top_p)
        _set_if_not_none(options, "top_k", self.top_k)
        _set_if_not_none(options, "on_error", self.on_error)
        return options


@dataclass(frozen=True, repr=False)
class VLLMPromptOptions(_RedactedOptionsRepr):
    """vLLM prompt generation options.

    ``max_tokens`` and ``temperature`` are convenience fields: the vLLM
    plan folds them into ``generate_args["sampling_params"]`` when it
    builds the native operator options, so they reach the engine's
    ``SamplingParams``. On conflict, an entry set explicitly in
    ``generate_args["sampling_params"]`` wins over the convenience field
    of the same name.

    ``generate_args`` crosses the native operator boundary as JSON. Values
    must therefore contain only JSON primitives, string-keyed mappings, and
    lists.
    """

    generate_args: Mapping[str, VLLMJSONValue] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    on_error: Literal["raise", "log", "ignore"] | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        if self.generate_args is not None:
            options["generate_args"] = dict(self.generate_args)
        _set_if_not_none(options, "max_tokens", self.max_tokens)
        _set_if_not_none(options, "temperature", self.temperature)
        _set_if_not_none(options, "on_error", self.on_error)
        return options
