# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Anthropic provider for basic text/image Prompt calls."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai._schema import OutputValidationError, serialize_raw_response
from vane.ai.options import (
    _require_prompt_int,
    _require_prompt_number,
    _validate_base_url_option,
    _validate_prompt_stop_sequences,
)
from vane.ai.protocols import PrompterDescriptor
from vane.ai.provider import (
    Provider,
    ProviderCapabilityError,
    _ProviderResultError,
    _translate_missing_provider_dependency,
)
from vane.ai.providers._mime import ImageMimePolicy
from vane.ai.typing import UDFOptions

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vane.ai.protocols import Prompter
    from vane.ai.typing import Options


_REQUEST_OPTIONS = frozenset({"max_tokens", "temperature", "top_p", "top_k", "stop_sequences"})
_PROMPT_OPTIONS = _REQUEST_OPTIONS | frozenset({"base_url", "timeout"})
_STRUCTURED_OUTPUT_TOOL = "vane_structured_output"
# https://platform.claude.com/docs/en/build-with-claude/vision#supported-formats
_IMAGE_MIME_POLICY = ImageMimePolicy(
    provider_name="Anthropic",
    supported_mime_types=frozenset(
        {
            "image/gif",
            "image/jpeg",
            "image/png",
            "image/webp",
        }
    ),
)


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

    Anthropic has no Vane default model or ``max_tokens`` value. Configure a
    model as provider metadata or pass one per call; ``max_tokens`` is always
    a call-level option.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        prompt_model: str | None = None,
    ) -> None:
        self._name = name or "anthropic"
        self._prompt_model = prompt_model

    @property
    def name(self) -> str:
        return self._name

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        return_format: dict[str, Any] | None = None,
        return_raw_response: bool = False,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> PrompterDescriptor:
        model_name = model if model is not None else self._prompt_model
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "No prompt model configured for the Anthropic provider. "
                "Pass model=... or configure AnthropicProvider(prompt_model=...)."
            )

        resolved_options = dict(options or {})

        return AnthropicPrompterDescriptor(
            provider_name=self._name,
            model_name=model_name,
            system_message=system_message,
            return_format=return_format,
            return_raw_response=return_raw_response,
            options=resolved_options,
        )


def _validate_anthropic_prompt_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Validate options owned by the Anthropic Prompt adapter."""

    copied = dict(options)
    unknown = sorted(set(copied) - _PROMPT_OPTIONS)
    if unknown:
        raise TypeError(f"Unsupported Anthropic Prompt option(s): {', '.join(unknown)}")
    _require_prompt_number(copied, "temperature", minimum=0, nullable=True)
    _require_prompt_int(copied, "max_tokens", minimum=0, nullable=True)
    _require_prompt_int(copied, "top_k", minimum=0, nullable=True)
    _require_prompt_number(copied, "top_p", minimum=0, maximum=1, nullable=True)
    _validate_prompt_stop_sequences(copied)
    _validate_base_url_option(copied, api="Prompt")
    _require_prompt_number(copied, "timeout", minimum=0, nullable=True)
    if copied.get("timeout") == 0:
        raise ValueError("Prompt option 'timeout' must be a finite positive number or None")
    return copied


@dataclass
class AnthropicPrompterDescriptor(PrompterDescriptor):
    """Serializable factory for the Anthropic Messages API."""

    model_name: str
    provider_name: str = "anthropic"
    system_message: str | None = None
    return_format: dict[str, Any] | None = None
    return_raw_response: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Anthropic prompt model must be a non-empty string")
        validated_options = _validate_anthropic_prompt_options(self.options)
        if validated_options.get("max_tokens") is None:
            raise ValueError(
                "No max_tokens configured for the Anthropic provider. Pass max_tokens=... on the Prompt call."
            )
        if validated_options["max_tokens"] == 0 and self.return_format is not None:
            raise ValueError("Anthropic max_tokens=0 cannot be used with structured Prompt output")
        self.options = wrap_sensitive_options(validated_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.options)

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=0)

    def instantiate(self) -> Prompter:
        return AnthropicPrompter(
            options=self.options,
            provider_name=self.provider_name,
            model=self.model_name,
            system_message=self.system_message,
            return_format=self.return_format,
            return_raw_response=self.return_raw_response,
        )


class AnthropicPrompter:
    """Async basic text/image prompter using Anthropic Messages."""

    def __init__(
        self,
        options: dict[str, Any],
        model: str,
        system_message: str | None = None,
        return_format: dict[str, Any] | None = None,
        return_raw_response: bool = False,
        provider_name: str = "anthropic",
    ) -> None:
        with _translate_missing_provider_dependency("anthropic", "anthropic"):
            from anthropic import AsyncAnthropic  # type: ignore[import-not-found, import-untyped, unused-ignore]

        options = unwrap_sensitive_options(options)
        self._provider_name = provider_name
        self._model = model
        self._system_message = system_message
        self._return_format = return_format
        self._return_raw_response = return_raw_response
        self._options = {key: value for key, value in options.items() if key in _REQUEST_OPTIONS and value is not None}
        client_options = {key: options[key] for key in ("base_url", "timeout") if options.get(key) is not None}
        client_options["max_retries"] = 0
        self._client = AsyncAnthropic(**client_options)

    async def aclose(self) -> None:
        await self._client.close()

    def _requested_capability(self) -> str:
        structured = getattr(self, "_return_format", None) is not None
        raw = getattr(self, "_return_raw_response", False)
        if structured and raw:
            return "structured Prompt generation with raw response body"
        if structured:
            return "structured Prompt generation"
        if raw:
            return "Prompt raw response body"
        return "basic Prompt text/image generation"

    @staticmethod
    def _process_message(message: Any) -> dict[str, Any]:
        if isinstance(message, str):
            return {"type": "text", "text": message}
        if isinstance(message, bytes):
            media_type = _IMAGE_MIME_POLICY.require_supported(message)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(message).decode("ascii"),
                },
            }
        raise TypeError(f"Unsupported Prompt content type: {type(message).__name__}")

    async def prompt(self, messages: tuple[Any, ...]) -> Any:
        content = [self._process_message(message) for message in messages]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            **self._options,
        }
        if self._system_message is not None:
            kwargs["system"] = self._system_message
        return_format = getattr(self, "_return_format", None)
        complete_stop_reasons: tuple[str, ...]
        if return_format is not None:
            kwargs["tools"] = [
                {
                    "name": _STRUCTURED_OUTPUT_TOOL,
                    "description": "Return the response in the requested structured format.",
                    "input_schema": return_format,
                    "strict": True,
                }
            ]
            kwargs["tool_choice"] = {
                "type": "tool",
                "name": _STRUCTURED_OUTPUT_TOOL,
                "disable_parallel_tool_use": True,
            }

        capability_error: ProviderCapabilityError | None = None
        retry_error: Exception | None = None
        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as exc:
            if _is_prompt_capability_error(exc):
                capability_error = ProviderCapabilityError(
                    self._provider_name,
                    self._model,
                    self._requested_capability(),
                    original_error=exc,
                )
            else:
                from vane.ai.functions import _retry_after_error

                retry_error = _retry_after_error(exc)
                if retry_error is None:
                    raise
        if retry_error is not None:
            # Raised outside the handler so the raw SDK error is not retained
            # as __context__ (mirrors the Google provider's raise shape).
            raise retry_error from None
        if capability_error is not None:
            raise capability_error from None

        if getattr(self, "_return_raw_response", False):
            return serialize_raw_response(response)

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "max_tokens":
            if self._options.get("max_tokens") == 0:
                return None
            raise _ProviderResultError(
                f"Anthropic response from model {self._model!r} was truncated at "
                f"max_tokens={self._options.get('max_tokens')}"
            )
        if stop_reason == "model_context_window_exceeded":
            raise _ProviderResultError(
                f"Anthropic response from model {self._model!r} was truncated at the model context window"
            )
        if stop_reason == "refusal":
            raise _ProviderResultError(f"Anthropic model {self._model!r} refused the Prompt request")
        if stop_reason == "pause_turn":
            raise _ProviderResultError(
                f"Anthropic response from model {self._model!r} paused before completion; "
                "Vane Prompt does not support continuing paused turns"
            )

        if return_format is not None:
            complete_stop_reasons = ("tool_use",)
            output_mode = "structured"
        else:
            complete_stop_reasons = ("end_turn", "stop_sequence")
            output_mode = "plain"
        if stop_reason not in complete_stop_reasons:
            if stop_reason is None:
                received_stop_reason = "a missing stop_reason"
            elif stop_reason in ("end_turn", "stop_sequence", "tool_use"):
                received_stop_reason = f"stop_reason {stop_reason!r}"
            else:
                received_stop_reason = "an unsupported stop_reason"
            expected_stop_reasons = " or ".join(repr(reason) for reason in complete_stop_reasons)
            raise _ProviderResultError(
                f"Anthropic response from model {self._model!r} returned {received_stop_reason} "
                f"for {output_mode} Prompt output; expected {expected_stop_reasons}"
            )

        blocks = getattr(response, "content", None) or []
        if return_format is not None:
            tool_uses = [
                block.input
                for block in blocks
                if getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == _STRUCTURED_OUTPUT_TOOL
            ]
            if len(tool_uses) != 1:
                raise OutputValidationError(
                    "Anthropic structured output must contain exactly one vane_structured_output tool call"
                )
            return tool_uses[0]
        text_blocks = [block.text for block in blocks if getattr(block, "type", None) == "text"]
        return "".join(text_blocks) if text_blocks else None
