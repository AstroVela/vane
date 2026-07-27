# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Anthropic provider for Vane AI.

Supports prompting via the Anthropic Messages API with multimodal input
(text + images) and structured output via tool_use. Anthropic does not
offer an embedding API, so only ``get_prompter`` is implemented.

Vane does not ship a default Anthropic model: the model must be supplied
per call (``model=...``) or configured on the provider
(``AnthropicProvider(prompt_model=...)``); the per-call model wins.

Prompt options follow a strict contract enforced at descriptor
construction, before any request is dispatched:

- Every known Messages API request option is forwarded to the API
  (``thinking``, ``metadata``, ``tool_choice``, ``service_tier``, ...).
- Unknown option keys raise :class:`ValueError` instead of being
  silently dropped.
- Option/model combinations the API is documented to reject (for
  example sampling parameters on Claude Sonnet 5) raise
  :class:`ValueError` before dispatch.
- ``max_tokens`` has no library default: it must be supplied per call
  (``max_tokens=...``) or configured on the provider
  (``AnthropicProvider(max_tokens=...)``); the per-call value wins.
- A truncated response (``stop_reason == "max_tokens"``) raises instead
  of silently returning partial text or ``None``.

Requires::

    pip install 'vane-ai[anthropic]'
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai.protocols import PrompterDescriptor
from vane.ai.provider import Provider, ProviderImportError
from vane.ai.typing import UDFOptions

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vane.ai.protocols import Prompter
    from vane.ai.typing import Options


# Messages API request options Vane forwards verbatim. Verified against the
# ``anthropic`` SDK (v0.117.0) ``Messages.create`` signature. Keys Vane owns
# are deliberately absent: ``model``, ``messages``, ``system``, ``tools``, and
# ``stream`` are managed by the prompter, and client configuration
# (``api_key``, ``base_url``, ``timeout``, ``max_retries``) travels through
# provider options.
_KNOWN_PROMPT_OPTIONS: frozenset[str] = frozenset(
    {
        "cache_control",
        "container",
        "extra_body",
        "extra_headers",
        "extra_query",
        "inference_geo",
        "max_tokens",
        "metadata",
        "output_config",
        "service_tier",
        "stop_sequences",
        "temperature",
        "thinking",
        "tool_choice",
        "top_k",
        "top_p",
        "user_profile_id",
    }
)

# Vane execution options that ride in ``prompt_options`` but are consumed by
# the UDF machinery and never forwarded to the Messages API.
_VANE_EXECUTION_OPTIONS: frozenset[str] = frozenset(
    {
        "actor_number",
        "batch_size",
        "concurrency",
        "max_api_concurrency",
        "num_gpus",
        "on_error",
    }
)

# Request options the API is documented to reject per model family. A model
# matches an entry by family prefix: the exact id or ``<family>-<suffix>``
# (dated snapshots and successors within the family). Documented sources:
# https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5
# (Claude Sonnet 5 returns a 400 error for non-default ``temperature``,
# ``top_p``, or ``top_k``) and
# https://platform.claude.com/docs/en/about-claude/models/migration-guide
# (the same constraint on Claude Opus 4.7 and later Opus models).
_MODEL_UNSUPPORTED_OPTIONS: dict[str, frozenset[str]] = {
    "claude-opus-4-7": frozenset({"temperature", "top_k", "top_p"}),
    "claude-opus-4-8": frozenset({"temperature", "top_k", "top_p"}),
    "claude-opus-5": frozenset({"temperature", "top_k", "top_p"}),
    "claude-sonnet-5": frozenset({"temperature", "top_k", "top_p"}),
}


def _validate_prompt_options(model_name: str, options: Mapping[str, Any]) -> None:
    """Reject unknown or model-unsupported prompt options before dispatch.

    Raises:
        ValueError: If ``options`` contains keys that are neither known
            Messages API request options nor Vane execution options, or
            contains request options the selected model is documented to
            reject.
    """
    unknown = sorted(key for key in options if key not in _KNOWN_PROMPT_OPTIONS and key not in _VANE_EXECUTION_OPTIONS)
    if unknown:
        raise ValueError(
            f"Unknown Anthropic prompt option(s) {unknown} for model {model_name!r}. "
            f"Supported Messages API request options: {sorted(_KNOWN_PROMPT_OPTIONS)}."
        )
    for family, unsupported in _MODEL_UNSUPPORTED_OPTIONS.items():
        if model_name == family or model_name.startswith(f"{family}-"):
            rejected = sorted(key for key in options if key in unsupported)
            if rejected:
                raise ValueError(
                    f"Anthropic model {model_name!r} does not support option(s) {rejected}: "
                    "the API rejects non-default sampling parameters for this model "
                    "family. Remove the option(s) or select a model that supports them."
                )


def _guess_media_type(data: bytes) -> str:
    """Guess image MIME type from magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


class AnthropicProvider(Provider):
    """Provider backed by the Anthropic Messages API.

    Vane does not choose an Anthropic model for you. Configure an
    application-owned model via ``prompt_model=...`` here, or pass
    ``model=...`` on each call; the per-call model takes precedence.
    The same applies to ``max_tokens``: configure it here or pass it per
    call; the per-call value wins.

    Args:
        name: Optional display-name override (default ``"anthropic"``).
        api_key: Anthropic API key. Prefer the ``ANTHROPIC_API_KEY``
            environment variable over passing keys in code.
        base_url: Optional API endpoint override.
        timeout: Optional client timeout forwarded to the Anthropic client.
        max_retries: Optional client retry count forwarded to the Anthropic
            client.
        prompt_model: Model used by ``get_prompter`` when the call does not
            pass ``model=...``.
        max_tokens: Response token cap used by ``get_prompter`` when the
            call does not pass ``max_tokens``.

    All parameters are named; a mistyped keyword raises :class:`TypeError`
    instead of silently leaking into API request options.
    """

    _CLIENT_KEYS: ClassVar[frozenset[str]] = frozenset({"api_key", "base_url", "timeout", "max_retries"})

    def __init__(
        self,
        name: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: Any = None,
        max_retries: int | None = None,
        prompt_model: str | None = None,
        max_tokens: int | None = None,
    ):
        self._name = name or "anthropic"
        self._prompt_model = prompt_model
        self._max_tokens = max_tokens
        client_options = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        self._options: dict[str, Any] = {k: v for k, v in client_options.items() if v is not None}

    @property
    def name(self) -> str:
        return self._name

    def _split_options(self, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        merged = {**self._options, **options}
        provider_options = {k: v for k, v in merged.items() if k in self._CLIENT_KEYS}
        prompt_options = {k: v for k, v in merged.items() if k not in self._CLIENT_KEYS}
        return provider_options, prompt_options

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        return_format: Any | None = None,
        **options: Any,
    ) -> PrompterDescriptor:
        """Build a prompter descriptor for an explicitly selected model.

        The model resolves as: call-site ``model`` > provider
        ``prompt_model`` configuration; the selected value must be a
        non-empty string. ``max_tokens`` resolves as: call-site option >
        provider ``max_tokens`` configuration.

        Raises:
            ValueError: If no model or no ``max_tokens`` is configured
                through either path, if the selected model is not a
                non-empty string, or if ``options`` contains unknown keys
                or options the selected model does not support.
        """
        provider_options, prompt_options = self._split_options(options)
        model_name = model if model is not None else self._prompt_model
        if model_name is None:
            raise ValueError(
                "No prompt model configured for the Anthropic provider. "
                "Pass model=... or configure AnthropicProvider(prompt_model=...)."
            )
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                f"Anthropic prompt model must be a non-empty string, got {model_name!r}. "
                "Pass model=... or configure AnthropicProvider(prompt_model=...)."
            )
        if "max_tokens" not in prompt_options and self._max_tokens is not None:
            prompt_options["max_tokens"] = self._max_tokens
        return AnthropicPrompterDescriptor(
            provider_name=self._name,
            provider_options=provider_options,
            model_name=model_name,
            system_message=system_message,
            return_format=return_format,
            prompt_options=prompt_options,
        )


@dataclass
class AnthropicPrompterDescriptor(PrompterDescriptor):
    """Serializable factory for an Anthropic Messages API prompter.

    ``model_name`` is required — Vane does not ship a default Anthropic
    model — and ``prompt_options`` must carry ``max_tokens``: there is no
    library default. Construction validates the option contract: unknown
    option keys and options the selected model is documented to reject
    raise :class:`ValueError` before anything is dispatched. Supports
    structured output via Anthropic's ``tool_use`` mechanism and
    multimodal input (text + images).
    """

    model_name: str
    provider_name: str = "anthropic"
    provider_options: dict[str, Any] = field(default_factory=dict)
    system_message: str | None = None
    return_format: Any | None = None
    prompt_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.provider_options = wrap_sensitive_options(self.provider_options)
        self.prompt_options = wrap_sensitive_options(self.prompt_options)
        _validate_prompt_options(self.model_name, self.prompt_options)
        if "max_tokens" not in self.prompt_options:
            raise ValueError(
                "No max_tokens configured for the Anthropic provider. "
                "Pass max_tokens=... or configure AnthropicProvider(max_tokens=...)."
            )

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.prompt_options)

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(
            max_retries=0,  # anthropic client handles retries internally
            on_error=self.prompt_options.get("on_error", "raise"),
            actor_number=self.prompt_options.get("actor_number"),
            num_gpus=self.prompt_options.get("num_gpus"),
            max_api_concurrency=self.prompt_options.get("max_api_concurrency", 16),
        )

    def instantiate(self) -> Prompter:
        return AnthropicPrompter(
            provider_options=self.provider_options,
            model=self.model_name,
            system_message=self.system_message,
            return_format=self.return_format,
            **self.prompt_options,
        )


class AnthropicPrompter:
    """Async prompter using the Anthropic Messages API.

    Features:
    - Multimodal: str, bytes (images), numpy arrays → content parts
    - Structured Output: ``return_format`` (Pydantic BaseModel) uses
      Anthropic's ``tool_use`` with forced tool choice to extract
      structured data.
    """

    def __init__(
        self,
        provider_options: dict[str, Any],
        model: str,
        system_message: str | None = None,
        return_format: Any | None = None,
        **options: Any,
    ):
        from anthropic import AsyncAnthropic

        # Restore plaintext credentials sealed by the descriptor; plain dicts
        # from direct callers pass through unchanged.
        provider_options = unwrap_sensitive_options(provider_options)
        options = unwrap_sensitive_options(options)
        self._model = model
        self._system_message = system_message
        self._return_format = return_format
        # Forward only known Messages API request options; Vane execution
        # options (and anything else) never reach the API. The descriptor
        # rejects unknown keys before the prompter is ever built.
        self._options = {k: v for k, v in options.items() if k in _KNOWN_PROMPT_OPTIONS}
        client_opts = {
            k: v for k, v in provider_options.items() if k in {"api_key", "base_url", "timeout", "max_retries"}
        }
        self._client = AsyncAnthropic(**client_opts)

    # --- Multimodal message processing -----------------------------------

    def _process_message(self, msg: Any) -> dict[str, Any]:
        """Convert a message part into an Anthropic content block."""
        if isinstance(msg, str):
            return {"type": "text", "text": msg}
        if isinstance(msg, bytes):
            return self._process_bytes(msg)
        if isinstance(msg, dict):
            return msg
        # numpy array
        type_name = type(msg).__name__
        mod = getattr(type(msg), "__module__", "")
        if type_name == "ndarray" and "numpy" in mod:
            return self._process_ndarray(msg)
        raise ValueError(f"Unsupported multimodal content type: {type(msg)}")

    def _process_bytes(self, msg: bytes) -> dict[str, Any]:
        media_type = _guess_media_type(msg)
        b64 = base64.b64encode(msg).decode("utf-8")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64,
            },
        }

    def _process_ndarray(self, arr: Any) -> dict[str, Any]:
        import io

        try:
            from PIL import Image
        except ImportError as exc:
            raise ProviderImportError("image", function="ndarray image input") from exc

        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        }

    # --- Structured output via tool_use ----------------------------------

    def _build_tool_schema(self) -> dict[str, Any]:
        """Build an Anthropic tool definition from a Pydantic model."""
        rf = self._return_format
        if hasattr(rf, "model_json_schema"):
            schema = rf.model_json_schema()
        elif isinstance(rf, dict):
            schema = rf
        else:
            schema = {"type": "object"}
        return {
            "name": "extract_data",
            "description": "Extract structured data from the response.",
            "input_schema": schema,
        }

    # --- API call --------------------------------------------------------

    async def prompt(self, messages: tuple[Any, ...]) -> Any:
        """Generate a response for the given message(s).

        Supports multimodal content (str, bytes, numpy arrays) and
        structured output via ``tool_use``.
        """
        chat_messages: list[dict[str, Any]] = []

        # Build content parts
        content_parts: list[dict[str, Any]] = []

        def _flush_content() -> None:
            if not content_parts:
                return
            chat_messages.append({"role": "user", "content": list(content_parts)})
            content_parts.clear()

        for msg in messages:
            if isinstance(msg, dict) and "role" in msg:
                _flush_content()
                chat_messages.append(msg)
            else:
                content_parts.append(self._process_message(msg))

        _flush_content()

        # Forward every known request option; there is no implicit
        # max_tokens default (the descriptor enforces its presence).
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            **self._options,
        }
        if self._system_message:
            kwargs["system"] = self._system_message

        # Structured output: use tool_use with forced tool choice. This
        # overrides any caller-supplied tool_choice for the extraction turn.
        if self._return_format is not None:
            tool_def = self._build_tool_schema()
            kwargs["tools"] = [tool_def]
            kwargs["tool_choice"] = {"type": "tool", "name": "extract_data"}

        response = await self._client.messages.create(**kwargs)

        # Record token usage metrics
        usage = getattr(response, "usage", None)
        if usage is not None:
            from vane.ai.metrics import record_token_metrics

            record_token_metrics(
                protocol="prompt",
                model=self._model,
                provider="anthropic",
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
            )

        # Truncation is an error, never a silent partial answer. The raise
        # routes through the UDF on_error policy like any other failure.
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise ValueError(
                f"Anthropic response from model {self._model!r} was truncated at "
                f"max_tokens={self._options.get('max_tokens')} (stop_reason='max_tokens'). "
                "Raise max_tokens on the call or via AnthropicProvider(max_tokens=...) "
                "to fit the full response."
            )

        if self._return_format is not None:
            # Extract structured data from tool_use block
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    result = block.input
                    # If return_format is a Pydantic model, instantiate it
                    if hasattr(self._return_format, "model_validate"):
                        return self._return_format.model_validate(result)
                    return result
            return None

        if response.content:
            text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
            return "".join(text_blocks) if text_blocks else None
        return None
