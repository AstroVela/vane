# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""vLLM provider — wraps the existing ``duckdb.execution.vllm`` engine.

The vLLM executor already manages its own ``AsyncLLMEngine`` event loop,
request queuing, prefix routing, and Ray actor pool.  This provider wraps
that machinery into the Vane AI Provider/Descriptor pattern so users can
write::

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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai.protocols import PrompterDescriptor
from vane.ai.provider import Provider
from vane.ai.typing import UDFOptions

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
    ) -> PrompterDescriptor:
        merged = {**self._options, **options}
        return VLLMPrompterDescriptor(
            provider_name=self._name,
            model_name=model or merged.pop("model", self.DEFAULT_MODEL),
            system_message=system_message,
            return_format=return_format,
            vllm_options=merged,
        )


@dataclass
class VLLMPrompterDescriptor(PrompterDescriptor):
    """Serializable configuration for native vLLM query planning.

    High-level prompt APIs consume this descriptor while binding the native
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

    def build_physical_vllm_options(self) -> dict[str, Any]:
        """Build JSON-ready options for the native ``PhysicalVLLM`` operator.

        The Python UDF path used ``actor_number`` to control the number of
        outer UDF actors. The native operator owns one executor instead, so
        that capacity becomes the executor's ``concurrency``. Structured
        output configuration is copied into vLLM sampling parameters without
        mutating the descriptor or caller-owned nested dictionaries.
        """
        options = copy.deepcopy(self.vllm_options)

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
        return options

    def get_udf_options(self) -> UDFOptions:
        opts = self.vllm_options
        return UDFOptions(
            batch_size=opts.get("batch_size"),
            num_gpus=opts.get("gpus_per_actor", 1),
            actor_number=opts.get("actor_number"),
            max_retries=0,  # vLLM engine handles retries
            on_error=opts.get("on_error", "raise"),
        )

    def instantiate(self) -> VLLMPrompter:
        return VLLMPrompter(
            model=self.model_name,
            system_message=self.system_message,
            return_format=self.return_format,
            vllm_options=self.vllm_options,
        )


class VLLMPrompter:
    """Compatibility object that rejects non-native vLLM execution.

    A prompter call cannot see relation boundaries, so it cannot know when the
    shared executor should receive ``finished_submitting()``. All supported
    vLLM entry points are therefore lowered to ``PhysicalVLLM`` before this
    object could be invoked.
    """

    _NATIVE_ONLY_ERROR = (
        "VLLMPrompter cannot execute prompts directly; use vane.ai.prompt(..., provider='vllm') "
        "or SQL ai_prompt(..., provider := 'vllm') so the query planner can use PhysicalVLLM"
    )

    def __init__(
        self,
        model: str,
        system_message: str | None = None,
        return_format: Any | None = None,
        vllm_options: dict[str, Any] | None = None,
    ):
        pass

    async def prompt(self, messages: tuple[Any, ...]) -> Any:
        raise NotImplementedError(self._NATIVE_ONLY_ERROR)

    def prompt_batch(self, texts: list[str]) -> list[Any]:
        raise NotImplementedError(self._NATIVE_ONLY_ERROR)
