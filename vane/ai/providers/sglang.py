# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""SGLang prompt provider for the native SGLang engine."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from vane.ai.options import (
    _reject_sensitive_prompt_options,
    _require_prompt_int,
    _require_prompt_number,
    normalize_prompt_options,
)
from vane.ai.protocols import NativeInferencePlan
from vane.ai.provider import Provider

if TYPE_CHECKING:
    from vane.ai.typing import Options


def _validate_sglang_json(value: Any, path: str) -> None:
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
            _validate_sglang_json(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_sglang_json(item, f"{path}[{index}]")
        return
    raise TypeError(f"Prompt option {path} must be JSON-compatible; got {type(value).__name__}")


def _validate_sglang_prompt_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Validate native SGLang controls before query or actor creation."""

    copied = dict(options)
    _require_prompt_number(copied, "temperature", minimum=0, nullable=True)
    for name in ("max_tokens", "max_new_tokens"):
        _require_prompt_int(copied, name, minimum=1, nullable=True)
    for name in ("max_buffer_size", "inflight_limit", "load_balance_threshold"):
        _require_prompt_int(copied, name, minimum=0)
    _require_prompt_number(copied, "engine_init_timeout_s", minimum=0, nullable=True)

    if "gpus_per_actor" in copied:
        _require_prompt_number(copied, "gpus_per_actor", minimum=0)
        value = float(copied["gpus_per_actor"])
        if value <= 0 or (value >= 1 and not value.is_integer()):
            raise ValueError("Prompt option 'gpus_per_actor' must be positive and must be an integer when >= 1")

    for name in ("engine_args", "generate_args"):
        if name not in copied:
            continue
        value = copied[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"Prompt option {name!r} must be a mapping")
        _validate_sglang_json(value, name)

    generate_args = copied.get("generate_args", {})
    sampling_params = generate_args.get("sampling_params")
    if isinstance(sampling_params, str):
        try:
            sampling_params = json.loads(sampling_params)
        except json.JSONDecodeError as exc:
            raise ValueError("Prompt option 'generate_args.sampling_params' must be valid JSON") from exc
    if sampling_params is not None and not isinstance(sampling_params, Mapping):
        raise TypeError("Prompt option 'generate_args.sampling_params' must be a mapping or JSON object")
    if isinstance(sampling_params, Mapping):
        _reject_sensitive_prompt_options(sampling_params, path="options.generate_args.sampling_params")
        _validate_sglang_json(sampling_params, "generate_args.sampling_params")
        for name in ("max_tokens", "max_new_tokens"):
            if name in sampling_params:
                nested_name = f"generate_args.sampling_params.{name}"
                _require_prompt_int({nested_name: sampling_params[name]}, nested_name, minimum=1, nullable=True)
        if "temperature" in sampling_params:
            nested_name = "generate_args.sampling_params.temperature"
            _require_prompt_number({nested_name: sampling_params["temperature"]}, nested_name, minimum=0, nullable=True)

    return copied


@dataclass
class NativeSGLangPromptPlan(NativeInferencePlan):
    """Serializable configuration for native SGLang query planning."""

    provider_name: str = "sglang"
    model_name: str = "Qwen/Qwen3-1.7B"
    system_message: str | None = None
    on_error: str = "raise"
    return_format: dict[str, Any] | None = None
    sglang_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("SGLang prompt model must be a non-empty string")

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.sglang_options)

    def get_engine(self) -> str:
        return "sglang"

    def build_physical_vllm_options(self) -> dict[str, Any]:
        options = dict(self.sglang_options)

        actor_number = options.pop("actor_number", None)
        if actor_number is not None:
            options.setdefault("concurrency", actor_number)

        max_retries = options.pop("max_retries", None)
        if max_retries not in (None, 0):
            raise ValueError("native SGLang prompting does not support max_retries")

        # PhysicalVLLM invokes the executor through a synchronous C++ bridge.
        # Inside a generic Ray actor that bridge still needs a dedicated
        # event-loop thread rather than Ray's async actor loop.
        options["use_threading"] = True
        options["_force_background_thread"] = True
        options["on_error"] = "null" if self.on_error == "ignore" else "raise"

        # Lower accepted prompt controls into SGLang sampling params so they are
        # not silently dropped. SGLang names the output-length field
        # max_new_tokens; accept vLLM's max_tokens as an alias (max_new_tokens
        # wins when both are supplied).
        sampling_overrides: dict[str, Any] = {}
        max_tokens = options.pop("max_tokens", None)
        if max_tokens is not None:
            sampling_overrides["max_new_tokens"] = max_tokens
        for name in ("max_new_tokens", "temperature"):
            value = options.pop(name, None)
            if value is not None:
                sampling_overrides[name] = value

        generate_args = options.get("generate_args")
        has_sampling_params = isinstance(generate_args, Mapping) and generate_args.get("sampling_params") is not None
        if self.return_format is None and not sampling_overrides and not has_sampling_params:
            return options

        if generate_args is None:
            generate_args = {}
        elif isinstance(generate_args, Mapping):
            generate_args = dict(generate_args)
        else:
            raise TypeError("SGLang generate_args must be a mapping when sampling parameters are configured")
        options["generate_args"] = generate_args

        sampling_params = generate_args.get("sampling_params")
        if sampling_params is None:
            sampling_params = {}
        elif isinstance(sampling_params, str):
            try:
                sampling_params = json.loads(sampling_params)
            except json.JSONDecodeError as exc:
                raise ValueError("SGLang sampling_params JSON could not be parsed") from exc
            if not isinstance(sampling_params, dict):
                raise TypeError("SGLang sampling_params JSON must decode to an object")
        elif isinstance(sampling_params, Mapping):
            sampling_params = dict(sampling_params)
        else:
            raise TypeError("SGLang sampling_params must be a mapping or JSON string")
        generate_args["sampling_params"] = sampling_params

        for name, value in sampling_overrides.items():
            sampling_params.setdefault(name, value)
        if self.return_format is not None:
            # SGLang's SamplingParams carries JSON-mode decoding via json_schema
            # (a JSON-schema string), unlike vLLM's structured_outputs object.
            sampling_params["json_schema"] = json.dumps(self.return_format)

        return options


class SGLangProvider(Provider):
    """Provider backed by a local or remote SGLang engine."""

    DEFAULT_MODEL = "Qwen/Qwen3-1.7B"

    def __init__(self, name: str | None = None):
        self._name = name or "sglang"

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
    ) -> NativeSGLangPromptPlan:
        if return_raw_response:
            raise ValueError("Provider 'sglang' does not support return_raw_response")
        prepared = normalize_prompt_options("sglang", options or {}, relation=False)
        prepared = _validate_sglang_prompt_options(prepared)
        model_name = model or self.DEFAULT_MODEL
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("SGLang prompt model must be a non-empty string")
        return NativeSGLangPromptPlan(
            provider_name=self._name,
            model_name=model_name,
            system_message=system_message,
            return_format=return_format,
            sglang_options=prepared,
        )
