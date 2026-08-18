# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""SGLang inference backend executor.

This module is the SGLang sibling of :mod:`vane.execution.vllm`. It implements
the engine-agnostic :class:`LLMExecutor` contract using SGLang's offline
``Engine`` to run generations. All submission/batching/wakeup/distributed
machinery lives in :class:`LocalEngineExecutor`; this module supplies only the
SGLang engine hooks.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from vane.ai.provider import (
    _SafeProviderError,
    _translate_missing_provider_dependency,
)
from vane.execution._llm_executor import LLMExecutor, LocalEngineExecutor, RayActorExecutorMixin
from vane.execution._vllm_options_protocol import _unpack_native_options_envelope
from vane.runners.ray.ray_env import install_explicit_session_runtime_env


class SGLangExecutor(LLMExecutor):
    """Base class for SGLang-backed executors.

    The engine-agnostic contract lives in :class:`LLMExecutor`; this class is
    the SGLang backend seam for concrete SGLang executors.
    """


class SGLangLocalExecutor(SGLangExecutor, LocalEngineExecutor):
    """SGLang backend executor: local async engine.

    Submission, batching, wakeup, and distributed routing live in
    :class:`LocalEngineExecutor`; this class supplies only the SGLang engine
    hooks.
    """

    _engine_name = "sglang"

    def _materialize_sampling_params(self, generate_args: dict[str, Any]) -> Any:
        with _translate_missing_provider_dependency("sglang", "sglang"):
            from sglang.srt.sampling.sampling_params import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                SamplingParams,
            )

        sampling_params = generate_args.pop("sampling_params", None)
        if sampling_params is None:
            return SamplingParams()
        if isinstance(sampling_params, SamplingParams):
            return sampling_params
        if isinstance(sampling_params, str):
            try:
                sampling_params = json.loads(sampling_params)
            except json.JSONDecodeError as exc:
                raise ValueError("sglang sampling_params JSON could not be parsed") from exc
        if isinstance(sampling_params, dict):
            return SamplingParams(**sampling_params)
        raise TypeError("sglang sampling_params must be a dict, JSON string, or SamplingParams instance")

    def _create_engine(self) -> None:
        with _translate_missing_provider_dependency("sglang", "sglang"):
            from sglang import Engine  # type: ignore[import-not-found, import-untyped, unused-ignore]

        self.llm = Engine(model_path=self.model, **self.engine_args)
        # SGLang's synchronous Engine drives a single internal event loop, so
        # generation calls must be serialized across the executor's threads.
        self._generate_lock = threading.Lock()

    async def _run_generate(self, prompt: str, request_id: str) -> str:
        del request_id  # SGLang's offline Engine is synchronous; no per-request id.
        output = await asyncio.to_thread(self._generate_sync, prompt)
        if output is None:
            raise _SafeProviderError("sglang returned no outputs")
        return self._extract_output_text(output)

    def _generate_sync(self, prompt: str) -> Any:
        # SGLang's sync Engine is not thread-safe: overlapping generate() calls
        # on its internal event loop fail with an already-running loop. Serialize
        # them and forward the remaining generate_args (return_logprob, etc.).
        with self._generate_lock:
            return self.llm.generate(prompt, self.sampling_params, **self.generate_args)

    def _shutdown_engine(self) -> None:
        # SGLang's offline Engine owns scheduler/detokenizer subprocesses and a
        # GPU context that must be released explicitly.
        llm = self.llm
        if llm is None:
            return
        shutdown = getattr(llm, "shutdown", None)
        if shutdown is not None:
            shutdown()

    @staticmethod
    def _extract_output_text(output: Any) -> str:
        if isinstance(output, dict):
            return str(output["text"])
        return str(getattr(output, "text"))


class SGLangRayLocalExecutor(RayActorExecutorMixin, SGLangLocalExecutor):  # type: ignore[misc]
    """SGLang backend executor for Ray actor pools.

    Reuses the engine hooks from :class:`SGLangLocalExecutor` and the async
    Ray-actor machinery from :class:`RayActorExecutorMixin`.
    """

    def __init__(
        self,
        model: str,
        engine_args: dict[str, Any],
        generate_args: dict[str, Any],
        on_error: str = "raise",
        use_threading: bool = True,
        engine_init_timeout_s: float | None = None,
        force_background_thread: bool = False,
        *,
        _install_session_runtime_env: bool = False,
    ):
        if _install_session_runtime_env:
            install_explicit_session_runtime_env()
        super().__init__(
            model,
            engine_args,
            generate_args,
            on_error=on_error,
            use_threading=use_threading,
            engine_init_timeout_s=engine_init_timeout_s,
            force_background_thread=force_background_thread,
        )


class _NormalizedSGLangOptions(dict[str, Any]):
    """Marker subclass so :func:`normalize_options` is idempotent."""


_SGLANG_DEFAULTS: dict[str, Any] = {
    "concurrency": 1,
    "gpus_per_actor": 1,
    "do_prefix_routing": False,  # SGLang's RadixAttention needs no explicit bucketing.
    "max_buffer_size": 5000,
    "min_bucket_size": 16,
    "prefix_match_threshold": 0.0,
    "load_balance_threshold": 32,
    "batch_size": 128,
    "on_error": "raise",
    "engine_args": {},
    "generate_args": {},
    "use_ray": False,
    "inflight_limit": 128,
    "engine_init_timeout_s": None,
}


def normalize_options(options: Any | None) -> dict[str, Any]:
    """Normalize a native SGLang options envelope into backend options.

    Returns the same shape the C++ executor factory reads (batch/backpressure
    controls plus engine/generate args); ``do_prefix_routing`` is forced off.
    """
    if isinstance(options, _NormalizedSGLangOptions):
        return options
    if options is None or isinstance(options, str):
        raise ValueError("sglang options must use the versioned envelope; bare JSON is not supported")
    if not isinstance(options, dict):
        try:
            options = dict(options)
        except Exception as exc:
            raise TypeError("sglang options must use the versioned envelope") from exc
    options = _unpack_native_options_envelope(options)
    merged = _NormalizedSGLangOptions(_SGLANG_DEFAULTS)
    merged.update(options)
    merged["engine_args"] = dict(merged.get("engine_args") or {})
    merged["generate_args"] = dict(merged.get("generate_args") or {})
    merged["on_error"] = str(merged.get("on_error") or "raise").lower()
    if merged["on_error"] not in ("raise", "log", "null"):
        raise ValueError("sglang on_error must be one of: raise, log, null")
    return merged


def build_executor(model: str, options: Any | None) -> LLMExecutor:
    from vane.execution.vllm import _restore_native_vllm_secrets

    opts = normalize_options(options)
    if opts.get("use_ray"):
        import ray

        if not ray.is_initialized():
            raise RuntimeError("Ray SGLang execution requires an initialized RayRunner runtime")
        pool_name = opts.get("ray_actor_pool_name")
        if pool_name:
            from vane.execution.vllm import LLMActors

            llm_actors = LLMActors.lookup_named(concurrency=opts["concurrency"], name_prefix=pool_name)
        else:
            from vane.execution.vllm import LLMActors

            opts = _restore_native_vllm_secrets(opts)
            llm_actors = LLMActors(
                model=model,
                engine_args=opts["engine_args"],
                generate_args=opts["generate_args"],
                on_error=opts["on_error"],
                gpus_per_actor=opts["gpus_per_actor"],
                concurrency=opts["concurrency"],
                load_balance_threshold=opts["load_balance_threshold"],
                engine_init_timeout_s=opts.get("engine_init_timeout_s"),
                actor_cls=SGLangRayLocalExecutor,
            )
        from vane.execution.vllm import RemoteVLLMExecutor

        return RemoteVLLMExecutor(llm_actors, pool_name=pool_name)

    opts = _restore_native_vllm_secrets(opts)
    return SGLangLocalExecutor(
        model,
        opts["engine_args"],
        opts["generate_args"],
        on_error=opts["on_error"],
        engine_init_timeout_s=opts.get("engine_init_timeout_s"),
    )
