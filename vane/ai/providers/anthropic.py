# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Anthropic provider for basic text/image Prompt calls."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai.options import validate_prompt_options
from vane.ai.protocols import PrompterDescriptor
from vane.ai.provider import Provider, ProviderCapabilityError
from vane.ai.typing import UDFOptions

if TYPE_CHECKING:
    from vane.ai.protocols import Prompter
    from vane.ai.typing import Options


_REQUEST_OPTIONS = frozenset({"max_tokens", "temperature", "top_p", "top_k", "stop_sequences"})
_EXECUTION_OPTIONS = frozenset({"batch_size", "actor_number", "max_concurrency_per_actor", "max_retries"})


def _guess_media_type(data: bytes) -> str | None:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _is_prompt_capability_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {404, 405, 501}:
        return True
    if status_code not in {400, 422}:
        return False
    code = getattr(exc, "code", None)
    param = getattr(exc, "param", None)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        details = body.get("error", body)
        if isinstance(details, dict):
            code = code or details.get("code") or details.get("type")
            param = param or details.get("param")
    normalized_code = str(code or "").casefold()
    normalized_param = str(param or "").casefold()
    return "model" in normalized_code or "model" in normalized_param or "endpoint" in normalized_code


class AnthropicProvider(Provider):
    """Provider backed by the Anthropic Messages API.

    Anthropic has no Vane default model or ``max_tokens`` value. Configure
    them on the provider or supply them on each Prompt call.
    """

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
    ) -> None:
        self._name = name or "anthropic"
        self._prompt_model = prompt_model
        self._client_options = {
            key: value
            for key, value in {"api_key": api_key, "base_url": base_url, "timeout": timeout}.items()
            if value is not None
        }
        self._options: dict[str, Any] = {}
        if max_retries is not None:
            self._options["max_retries"] = max_retries
        if max_tokens is not None:
            self._options["max_tokens"] = max_tokens

    @property
    def name(self) -> str:
        return self._name

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        **options: Any,
    ) -> PrompterDescriptor:
        model_name = model if model is not None else self._prompt_model
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "No prompt model configured for the Anthropic provider. "
                "Pass model=... or configure AnthropicProvider(prompt_model=...)."
            )

        prompt_options = {**self._options, **options}
        validation_options = {
            key: value for key, value in self._client_options.items() if key in {"base_url", "timeout"}
        }
        validation_options.update(prompt_options)
        validate_prompt_options("anthropic", validation_options, relation=False)
        if prompt_options.get("max_tokens") is None:
            raise ValueError(
                "No max_tokens configured for the Anthropic provider. "
                "Pass max_tokens=... or configure AnthropicProvider(max_tokens=...)."
            )

        provider_options = dict(self._client_options)
        for key in ("base_url", "timeout"):
            if key in prompt_options:
                value = prompt_options.pop(key)
                if value is None:
                    provider_options.pop(key, None)
                else:
                    provider_options[key] = value

        return AnthropicPrompterDescriptor(
            provider_name=self._name,
            provider_options=provider_options,
            model_name=model_name,
            system_message=system_message,
            prompt_options=prompt_options,
        )


@dataclass
class AnthropicPrompterDescriptor(PrompterDescriptor):
    """Serializable factory for the Anthropic Messages API."""

    model_name: str
    provider_name: str = "anthropic"
    provider_options: dict[str, Any] = field(default_factory=dict)
    system_message: str | None = None
    prompt_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Anthropic prompt model must be a non-empty string")
        validation_options = {
            key: value for key, value in self.provider_options.items() if key in {"base_url", "timeout"}
        }
        validation_options.update(self.prompt_options)
        validate_prompt_options("anthropic", validation_options, relation=False)
        if self.prompt_options.get("max_tokens") is None:
            raise ValueError(
                "No max_tokens configured for the Anthropic provider. "
                "Pass max_tokens=... or configure AnthropicProvider(max_tokens=...)."
            )
        self.provider_options = wrap_sensitive_options(self.provider_options)
        self.prompt_options = wrap_sensitive_options(self.prompt_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.prompt_options)

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(
            max_retries=self.prompt_options.get("max_retries", 3),
            actor_number=self.prompt_options.get("actor_number", 1),
            num_gpus=0,
            batch_size=self.prompt_options.get("batch_size", 32),
            max_concurrency_per_actor=self.prompt_options.get("max_concurrency_per_actor", 16),
        )

    def instantiate(self) -> Prompter:
        return AnthropicPrompter(
            provider_options=self.provider_options,
            provider_name=self.provider_name,
            model=self.model_name,
            system_message=self.system_message,
            **self.prompt_options,
        )


class AnthropicPrompter:
    """Async basic text/image prompter using Anthropic Messages."""

    def __init__(
        self,
        provider_options: dict[str, Any],
        model: str,
        system_message: str | None = None,
        provider_name: str = "anthropic",
        **options: Any,
    ) -> None:
        from anthropic import AsyncAnthropic

        provider_options = unwrap_sensitive_options(provider_options)
        options = unwrap_sensitive_options(options)
        self._provider_name = provider_name
        self._model = model
        self._system_message = system_message
        self._options = {key: value for key, value in options.items() if key in _REQUEST_OPTIONS and value is not None}
        client_options = {
            key: value for key, value in provider_options.items() if key in {"api_key", "base_url", "timeout"}
        }
        client_options["max_retries"] = 0
        self._client = AsyncAnthropic(**client_options)

    async def aclose(self) -> None:
        await self._client.close()

    @staticmethod
    def _process_message(message: Any) -> dict[str, Any]:
        if isinstance(message, str):
            return {"type": "text", "text": message}
        if isinstance(message, bytes):
            media_type = _guess_media_type(message)
            if media_type is None:
                raise ValueError("Prompt image BLOB has an unsupported or unrecognized image format")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(message).decode("ascii"),
                },
            }
        raise TypeError(f"Unsupported Prompt content type: {type(message).__name__}")

    async def prompt(self, messages: tuple[Any, ...]) -> str | None:
        content = [self._process_message(message) for message in messages]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            **self._options,
        }
        if self._system_message is not None:
            kwargs["system"] = self._system_message

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as exc:
            if _is_prompt_capability_error(exc):
                raise ProviderCapabilityError(
                    self._provider_name,
                    self._model,
                    "basic Prompt text/image generation",
                    original_error=exc,
                ) from exc
            raise

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

        if getattr(response, "stop_reason", None) == "max_tokens":
            if self._options.get("max_tokens") == 0:
                return None
            raise ValueError(
                f"Anthropic response from model {self._model!r} was truncated at "
                f"max_tokens={self._options.get('max_tokens')}"
            )

        blocks = getattr(response, "content", None) or []
        text_blocks = [block.text for block in blocks if getattr(block, "type", None) == "text"]
        return "".join(text_blocks) if text_blocks else None
