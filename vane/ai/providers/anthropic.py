"""Anthropic provider for Vane AI.

Supports prompting via the Anthropic Messages API with multimodal input
(text + images) and structured output via tool_use. Anthropic does not
offer an embedding API, so only ``get_prompter`` is implemented.

Requires::

    pip install 'vane-ai[anthropic]'
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from vane.ai.protocols import PrompterDescriptor
from vane.ai.provider import Provider
from vane.ai.typing import UDFOptions

if TYPE_CHECKING:
    from vane.ai.protocols import Prompter
    from vane.ai.typing import Options


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
    """Provider backed by the Anthropic Messages API."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    _CLIENT_KEYS: ClassVar[frozenset[str]] = frozenset({"api_key", "base_url", "timeout", "max_retries"})

    def __init__(self, name: str | None = None, **options: Any):
        self._name = name or "anthropic"
        self._options: dict[str, Any] = options

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
        provider_options, prompt_options = self._split_options(options)
        return AnthropicPrompterDescriptor(
            provider_name=self._name,
            provider_options=provider_options,
            model_name=model or prompt_options.pop("model", self.DEFAULT_MODEL),
            system_message=system_message,
            return_format=return_format,
            prompt_options=prompt_options,
        )


@dataclass
class AnthropicPrompterDescriptor(PrompterDescriptor):
    """Serializable factory for an Anthropic Messages API prompter.

    Supports structured output via Anthropic's ``tool_use`` mechanism
    and multimodal input (text + images).
    """

    provider_name: str = "anthropic"
    provider_options: dict[str, Any] = field(default_factory=dict)
    model_name: str = "claude-sonnet-4-20250514"
    system_message: str | None = None
    return_format: Any | None = None
    prompt_options: dict[str, Any] = field(default_factory=dict)

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

        self._model = model
        self._system_message = system_message
        self._return_format = return_format
        self._options = {
            k: v
            for k, v in options.items()
            if k
            not in {
                "api_key",
                "base_url",
                "timeout",
                "max_retries",
                "on_error",
                "actor_number",
                "num_gpus",
                "concurrency",
                "max_api_concurrency",
                "model",
                "batch_size",
            }
        }
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

        from PIL import Image

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

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            "max_tokens": self._options.get("max_tokens", 1024),
        }
        if self._system_message:
            kwargs["system"] = self._system_message

        # Forward extra options
        for k in ("temperature", "top_p", "top_k", "stop_sequences"):
            if k in self._options:
                kwargs[k] = self._options[k]

        # Structured output: use tool_use with forced tool choice
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
            return response.content[0].text
        return None
