# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Basic text-only Prompt provider for the native vLLM engine.

The vLLM executor already manages its own ``AsyncLLMEngine`` event loop,
request queuing, prefix routing, and Ray actor pool. This provider wraps that
machinery in a planner-only Vane AI provider so users can write::

    from vane.ai import prompt

    result = prompt(
        rel,
        vane.col("text"),
        provider="vllm",
        model="Qwen/Qwen3-1.7B",
        engine_args={"max_model_len": 2048},
        generate_args={"sampling_params": {"max_tokens": 256}},
    )

"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from duckdb.execution._vllm_options_protocol import (
    _NATIVE_OPTIONS_PAYLOAD_VERSION,
    _NATIVE_OPTIONS_PUBLIC_KEY,
    _NATIVE_OPTIONS_RESERVED_KEYS,
    _NATIVE_OPTIONS_SECRET_KEY,
    _NATIVE_OPTIONS_VERSION_KEY,
    _NATIVE_SECRET_PAYLOAD_VALUES_KEY,
    _NATIVE_SECRET_PAYLOAD_VERSION_KEY,
    _NATIVE_SECRET_REF_KEY,
    _dump_vllm_protocol_json,
)
from vane.ai._redaction import Secret, wrap_sensitive_options
from vane.ai.options import validate_prompt_options
from vane.ai.protocols import NativePrompterPlan
from vane.ai.provider import Provider

if TYPE_CHECKING:
    from vane.ai.typing import Options


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


def _canonicalize_native_plan_json(value: Any, path: str = "options") -> Any:
    """Validate native options while preserving sealed values for planning."""
    if isinstance(value, Secret):
        return Secret(_canonicalize_native_json(value.reveal(), path))
    if isinstance(value, Mapping):
        canonical: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"vLLM native option {path} must use string keys; got {type(key).__name__}")
            canonical[key] = _canonicalize_native_plan_json(item, f"{path}.{key}")
        return canonical
    if isinstance(value, (list, tuple)):
        return [_canonicalize_native_plan_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    return _canonicalize_native_json(value, path)


def _serialize_native_vllm_options(options: Mapping[str, Any]) -> str:
    """Serialize native options after enforcing the public JSON boundary."""
    canonical = _canonicalize_native_json(options)
    return _dump_vllm_protocol_json(canonical)


def _split_native_vllm_secrets(value: Any, secrets: list[Any], path: str = "options") -> Any:
    if isinstance(value, Secret):
        secret_index = len(secrets)
        secrets.append(value.reveal())
        return {_NATIVE_SECRET_REF_KEY: secret_index}
    if isinstance(value, Mapping):
        if _NATIVE_SECRET_REF_KEY in value:
            raise ValueError(f"vLLM native option {path} uses reserved field {_NATIVE_SECRET_REF_KEY!r}")
        return {key: _split_native_vllm_secrets(item, secrets, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [_split_native_vllm_secrets(item, secrets, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


def _build_native_vllm_options_argument(options: Mapping[str, Any]) -> dict[str, Any]:
    """Encode native options in the single versioned STRUCT wire format.

    Public options remain JSON while sealed plaintext values are confined to an
    opaque BLOB. Opaque here means excluded from structured plan rendering, not
    encrypted. The runtime decodes the BLOB with strict JSON rather than
    executable pickle data. Plans without credentials use the same envelope
    with an empty values payload.
    """
    canonical = _canonicalize_native_plan_json(options)
    assert isinstance(canonical, dict)
    reserved_protocol_keys = _NATIVE_OPTIONS_RESERVED_KEYS.intersection(canonical)
    if reserved_protocol_keys:
        raise ValueError(
            "vLLM native options use reserved protocol fields: " + ", ".join(sorted(reserved_protocol_keys))
        )
    secrets: list[Any] = []
    public_options = _split_native_vllm_secrets(canonical, secrets)
    public_options_json = _serialize_native_vllm_options(public_options)
    secret_payload = _dump_vllm_protocol_json(
        {
            _NATIVE_SECRET_PAYLOAD_VERSION_KEY: _NATIVE_OPTIONS_PAYLOAD_VERSION,
            _NATIVE_SECRET_PAYLOAD_VALUES_KEY: secrets,
        }
    ).encode("utf-8")
    return {
        _NATIVE_OPTIONS_VERSION_KEY: _NATIVE_OPTIONS_PAYLOAD_VERSION,
        _NATIVE_OPTIONS_PUBLIC_KEY: public_options_json,
        _NATIVE_OPTIONS_SECRET_KEY: secret_payload,
    }


class VLLMProvider(Provider):
    """Provider backed by a local or remote vLLM engine."""

    DEFAULT_MODEL = "Qwen/Qwen3-1.7B"

    def __init__(self, name: str | None = None):
        self._name = name or "vllm"

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
    ) -> NativeVLLMPromptPlan:
        if return_raw_response:
            raise ValueError("Provider 'vllm' does not support return_raw_response")
        prepared = validate_prompt_options("vllm", options or {}, relation=False)
        model_name = model or self.DEFAULT_MODEL
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("vLLM prompt model must be a non-empty string")
        return NativeVLLMPromptPlan(
            provider_name=self._name,
            model_name=model_name,
            system_message=system_message,
            return_format=return_format,
            vllm_options=prepared,
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
    on_error: str = "raise"
    return_format: dict[str, Any] | None = None
    vllm_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("vLLM prompt model must be a non-empty string")
        self.vllm_options = wrap_sensitive_options(self.vllm_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.vllm_options)

    def build_physical_vllm_options(self) -> dict[str, Any]:
        """Build options for the native ``PhysicalVLLM`` operator.

        The Python UDF path used ``actor_number`` to control the number of
        outer UDF actors. The native operator owns one executor instead, so
        that capacity becomes the executor's internal actor-pool size.
        Caller-owned nested dictionaries are never mutated.
        """
        options = _canonicalize_native_plan_json(self.vllm_options)
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
        options["on_error"] = "null" if self.on_error == "ignore" else "raise"

        sampling_overrides: dict[str, Any] = {}
        for name in ("max_tokens", "temperature"):
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
        sampling_params = wrap_sensitive_options(sampling_params)
        generate_args["sampling_params"] = sampling_params

        for name, value in sampling_overrides.items():
            sampling_params.setdefault(name, value)
        if self.return_format is not None:
            sampling_params["structured_outputs"] = {"json": self.return_format}

        canonical = _canonicalize_native_plan_json(options)
        assert isinstance(canonical, dict)
        return canonical
