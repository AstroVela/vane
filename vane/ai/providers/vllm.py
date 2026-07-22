# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""vLLM provider — wraps the existing ``duckdb.execution.vllm`` engine.

The vLLM executor already manages its own ``AsyncLLMEngine`` event loop,
request queuing, prefix routing, and Ray actor pool. This provider wraps that
machinery in a planner-only Vane AI provider so users can write::

    from vane.ai import prompt

    result = prompt(
        rel,
        "text",
        provider="vllm",
        model="Qwen/Qwen3-1.7B",
        engine_args={"max_model_len": 2048},
        generate_args={"sampling_params": {"max_tokens": 256}},
    )

Structured Output is supported via vLLM structured decoding.  Pass a
Pydantic ``BaseModel`` as ``return_format``::

    class Person(BaseModel):
        name: str
        age: int


    result = prompt(
        rel,
        "text",
        provider="vllm",
        model="Qwen/Qwen3-1.7B",
        return_format=Person,
    )

Under the hood the model's JSON schema is injected into
``SamplingParams.structured_outputs``.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai.options import VLLMJSONValue
from vane.ai.protocols import NativePrompterPlan
from vane.ai.provider import Provider

if TYPE_CHECKING:
    from vane.ai.typing import Options


def _json_schema_from_return_format(return_format: Any) -> dict[str, Any]:
    """Extract a JSON schema dict from a return_format value.

    Accepts:
    - Pydantic BaseModel *class*  → ``model_json_schema()``
    - ``dict``                    → used as-is (assumed to be a valid JSON schema)
    """
    if return_format is None:
        return {}
    if isinstance(return_format, dict):
        return return_format
    if hasattr(return_format, "model_json_schema"):
        return return_format.model_json_schema()
    raise TypeError(
        f"return_format must be a Pydantic BaseModel class or a JSON schema dict, got {type(return_format).__name__}"
    )


def _canonicalize_native_json(value: Any, path: str = "options") -> Any:
    """Return a strict JSON-compatible copy or fail with an option path."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"vLLM native option {path} must be finite")
        return value
    if isinstance(value, Mapping):
        canonical: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"vLLM native option {path} must use string keys; got {type(key).__name__}")
            canonical[key] = _canonicalize_native_json(item, f"{path}.{key}")
        return canonical
    if isinstance(value, (list, tuple)):
        return [_canonicalize_native_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"vLLM native option {path} must be JSON-compatible; got {type(value).__name__}")


def _serialize_native_vllm_options(options: Mapping[str, Any]) -> str:
    """Serialize native options after enforcing the public JSON boundary."""
    canonical = _canonicalize_native_json(options)
    return json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


class VLLMProvider(Provider):
    """Provider backed by a local or remote vLLM engine."""

    DEFAULT_MODEL = "Qwen/Qwen3-1.7B"

    def __init__(self, name: str | None = None, **options: Any):
        self._name = name or "vllm"
        self._options: dict[str, Any] = options

    @property
    def name(self) -> str:
        return self._name

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        return_format: Any | None = None,
        **options: Any,
    ) -> NativeVLLMPromptPlan:
        merged = {**self._options, **options}
        return NativeVLLMPromptPlan(
            provider_name=self._name,
            model_name=model or merged.pop("model", self.DEFAULT_MODEL),
            system_message=system_message,
            return_format=return_format,
            vllm_options=merged,
        )


@dataclass
class NativeVLLMPromptPlan(NativePrompterPlan):
    """Serializable configuration for native vLLM query planning.

    High-level prompt APIs consume this plan while binding the native
    ``vllm()`` expression. The resulting ``PhysicalVLLM`` operator owns one
    executor for the relation and sends its terminal signal only after every
    input batch has been submitted.
    """

    provider_name: str = "vllm"
    model_name: str = "Qwen/Qwen3-1.7B"
    system_message: str | None = None
    return_format: Any | None = None
    vllm_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vllm_options = wrap_sensitive_options(self.vllm_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.vllm_options)

    def build_physical_vllm_options(self) -> dict[str, VLLMJSONValue]:
        """Build JSON-ready options for the native ``PhysicalVLLM`` operator.

        The Python UDF path used ``actor_number`` to control the number of
        outer UDF actors. The native operator owns one executor instead, so
        that capacity becomes the executor's ``concurrency``. Structured
        output configuration is copied into vLLM sampling parameters without
        mutating the descriptor or caller-owned nested dictionaries.
        """
        options = _canonicalize_native_json(unwrap_sensitive_options(self.vllm_options))
        assert isinstance(options, dict)

        actor_number = options.pop("actor_number", None)
        if actor_number is not None:
            options.setdefault("concurrency", actor_number)

        max_retries = options.pop("max_retries", None)
        if max_retries not in (None, 0):
            raise ValueError("native vLLM prompting does not support max_retries")

        # PhysicalVLLM invokes the executor through a synchronous C++ bridge.
        # If that bridge runs inside a generic Ray actor, it still needs a
        # dedicated event-loop thread rather than Ray's async actor loop.
        options["use_threading"] = True
        options["_force_background_thread"] = True

        # The public AI API calls the null-producing policy ``ignore`` while
        # the native vLLM executor calls the same policy ``null``.
        if options.get("on_error") == "ignore":
            options["on_error"] = "null"

        sampling_overrides: dict[str, Any] = {}
        for name in ("max_tokens", "temperature"):
            value = options.pop(name, None)
            if value is not None:
                sampling_overrides[name] = value
        if self.return_format is None and not sampling_overrides:
            return options

        generate_args = options.get("generate_args")
        if generate_args is None:
            generate_args = {}
        elif isinstance(generate_args, Mapping):
            generate_args = dict(generate_args)
        else:
            raise TypeError("vLLM generate_args must be a mapping when sampling parameters are configured")
        options["generate_args"] = generate_args

        sampling_params = generate_args.get("sampling_params")
        if sampling_params is None:
            sampling_params = {}
        elif isinstance(sampling_params, str):
            try:
                sampling_params = json.loads(sampling_params)
            except json.JSONDecodeError as exc:
                raise ValueError("vLLM sampling_params JSON could not be parsed") from exc
            if not isinstance(sampling_params, dict):
                raise TypeError("vLLM sampling_params JSON must decode to an object")
        elif isinstance(sampling_params, Mapping):
            sampling_params = dict(sampling_params)
        else:
            raise TypeError("vLLM sampling_params must be a mapping or JSON string")
        generate_args["sampling_params"] = sampling_params

        for name, value in sampling_overrides.items():
            sampling_params.setdefault(name, value)

        if self.return_format is not None:
            schema = copy.deepcopy(_json_schema_from_return_format(self.return_format))
            sampling_params["structured_outputs"] = {"type": "json", "value": schema}
        canonical = _canonicalize_native_json(options)
        assert isinstance(canonical, dict)
        return canonical


# Compatibility import for callers that used the original name. The object is
# now explicitly planner-only and no longer inherits PrompterDescriptor or
# exposes an instantiate() method.
VLLMPrompterDescriptor = NativeVLLMPromptPlan
