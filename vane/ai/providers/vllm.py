# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""vLLM provider — wraps the existing ``vane.execution.vllm`` engine.

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

import json
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


def _parse_structured_output(raw_text: str | None, return_format: Any) -> Any:
    """Parse raw JSON text into a structured object.

    If *return_format* is a Pydantic model class the parsed dict is validated
    through ``model_validate``.  Otherwise returns a plain ``dict``.
    """
    if raw_text is None:
        return None
    data = json.loads(raw_text)
    if hasattr(return_format, "model_validate"):
        return return_format.model_validate(data)
    return data


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
    """Serializable factory for a vLLM-backed prompter.

    Stores model name and vLLM configuration.  On ``instantiate()`` it
    creates a ``LocalVLLMExecutor`` or ``RemoteVLLMExecutor`` via the
    existing ``vane.execution.vllm.build_executor()`` factory.

    When ``return_format`` is set (Pydantic model or JSON schema dict),
    the JSON schema is injected as ``structured_outputs`` in the executor's
    ``SamplingParams``.
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
    """Prompter that uses a vLLM ``LocalVLLMExecutor`` / ``RemoteVLLMExecutor``.

    The executor is created lazily on the first call. Structured output
    uses ``SamplingParams.structured_outputs`` and the raw JSON output is
    parsed back into the requested schema/model.
    """

    def __init__(
        self,
        model: str,
        system_message: str | None = None,
        return_format: Any | None = None,
        vllm_options: dict[str, Any] | None = None,
    ):
        self._model = model
        self._system_message = system_message
        self._return_format = return_format
        # Deep-unwrap restores plaintext sealed by the descriptor (including
        # nested engine_args/generate_args) before the engine sees the options;
        # plain dicts from direct callers pass through unchanged.
        options = unwrap_sensitive_options(vllm_options or {})
        self._options = {k: v for k, v in options.items() if k not in {"actor_number"}}
        self._executor = None

        # Pre-compute JSON schema if return_format is set, so the executor
        # receives the current vLLM structured output config.
        if self._return_format is not None:
            schema = _json_schema_from_return_format(self._return_format)
            gen_args = self._options.setdefault("generate_args", {})
            sp = gen_args.setdefault("sampling_params", {})
            if isinstance(sp, dict):
                sp["structured_outputs"] = {"type": "json", "value": schema}

    def _ensure_executor(self) -> Any:
        if self._executor is None:
            from vane.execution.vllm import build_executor

            options = dict(self._options)
            options["use_threading"] = True
            options["_force_background_thread"] = True
            self._executor = build_executor(self._model, options)
        return self._executor

    def _format_prompt(self, text: str) -> str:
        if self._system_message:
            return f"{self._system_message}\n\n{text}"
        return text

    def _maybe_parse(self, raw: str | None) -> Any:
        """Parse structured output if return_format is set."""
        if self._return_format is None or raw is None:
            return raw
        return _parse_structured_output(raw, self._return_format)

    @staticmethod
    async def _wait_for_result_async(executor: Any) -> None:
        import asyncio
        import inspect

        wait_for_result = executor.wait_for_result
        if inspect.iscoroutinefunction(wait_for_result):
            await wait_for_result()
            return
        await asyncio.to_thread(wait_for_result)

    @staticmethod
    def _wait_for_result(executor: Any) -> None:
        executor.wait_for_result()

    async def prompt(self, messages: tuple[Any, ...]) -> Any:
        """Single-row prompt — required by the Prompter protocol.

        For vLLM this is less efficient than batch submission, but
        allows the wrapper classes in ``functions.py`` to use the
        standard ``asyncio.gather`` pattern.
        """
        text = str(messages[0]) if messages else ""
        formatted = self._format_prompt(text)

        executor = self._ensure_executor()

        import pyarrow as pa

        dummy_row = pa.table({"_": [""]})
        executor.submit(None, [formatted], dummy_row)
        executor.finished_submitting()

        result = executor.take_ready_result()
        if result is None:
            await self._wait_for_result_async(executor)
            result = executor.take_ready_result()
        if result is None:
            raise RuntimeError("vllm executor finished without returning a prompt result")
        output_texts, _row = result
        return self._maybe_parse(output_texts[0])

    def prompt_batch(self, texts: list[str]) -> list[Any]:
        """Batch prompt — more efficient for vLLM's continuous batching."""
        import pyarrow as pa

        executor = self._ensure_executor()
        prompts = [self._format_prompt(t) for t in texts]
        rows = pa.table({"_idx": list(range(len(texts)))})
        executor.submit(None, prompts, rows)
        executor.finished_submitting()

        results: list[tuple[Any, int]] = []
        while len(results) < len(texts):
            result = executor.take_ready_result()
            if result is None:
                if executor.all_tasks_finished():
                    break
                self._wait_for_result(executor)
                result = executor.take_ready_result()
                if result is None:
                    if executor.all_tasks_finished():
                        break
                    raise RuntimeError("vllm executor wait_for_result returned without a ready result")
            output_texts, row_table = result
            indices = row_table.column("_idx").to_pylist()
            for text, idx in zip(output_texts, indices, strict=False):
                results.append((self._maybe_parse(text), idx))

        if len(results) != len(texts):
            raise RuntimeError(f"vllm executor returned {len(results)} results for {len(texts)} prompts")

        results.sort(key=lambda x: x[1])
        return [r[0] for r in results]
