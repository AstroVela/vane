# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""SGLang inference backend executor.

This module is the SGLang sibling of :mod:`vane.execution.vllm`. It implements
the same engine-agnostic :class:`LLMExecutor` contract using SGLang's offline
``Engine`` to run generations.
"""

from __future__ import annotations

import json
import queue
import threading
from collections import deque
from typing import Any

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

from vane.ai.provider import _safe_provider_execution_error
from vane.execution._llm_executor import LLMExecutor
from vane.execution._vllm_options_protocol import _unpack_native_options_envelope


class SGLangExecutor(LLMExecutor):
    """Base class for SGLang-backed executors.

    The engine-agnostic contract lives in :class:`LLMExecutor`; this class is
    the SGLang backend seam for concrete SGLang executors.
    """


def _ensure_table(rows: Any) -> pa.Table:
    if isinstance(rows, pa.Table):
        return rows
    if isinstance(rows, pa.RecordBatch):
        return pa.Table.from_batches([rows])
    if isinstance(rows, pa.RecordBatchReader):
        return pa.Table.from_batches(list(rows))
    raise TypeError("rows must be a pyarrow Table, RecordBatch, or RecordBatchReader")


class SGLangLocalExecutor(SGLangExecutor):
    """Local SGLang executor running the offline engine on a background thread."""

    def __init__(
        self,
        model: str,
        engine_args: dict[str, Any],
        generate_args: dict[str, Any],
        on_error: str = "raise",
        use_threading: bool = True,
        engine_init_timeout_s: float | None = None,
        force_background_thread: bool = False,
    ):
        del use_threading, force_background_thread  # local SGLang always uses a thread
        self.model = model
        self.engine_args = dict(engine_args or {})
        self.generate_args = dict(generate_args or {})
        self.on_error = on_error
        self.engine_init_timeout_s = engine_init_timeout_s

        self.engine: Any = None
        self.engine_error_message: str | None = None
        self.error_message: str | None = None
        self.error_lock = threading.Lock()
        self.engine_ready = threading.Event()

        self.sampling_params = self._materialize_sampling_params(self.generate_args)

        self.counter = 0
        self.counter_lock = threading.Lock()
        self.running_task_count = 0
        self.task_count_lock = threading.Lock()
        self.completed_tasks: deque[tuple[str | None, pa.Table]] = deque()
        self._finished_submitting = False
        self._shutdown_called = False
        self._result_cv = threading.Condition(threading.RLock())
        self._ensure_wakeup_state()

        self._submit_queue: queue.Queue[tuple[str, str, pa.Table] | None] = queue.Queue()
        self._engine_thread = threading.Thread(target=self._engine_loop, daemon=True)
        self._engine_thread.start()

    def _materialize_sampling_params(self, generate_args: dict[str, Any]) -> Any:
        from sglang import SamplingParams  # type: ignore[import-not-found, import-untyped, unused-ignore]

        sampling_params = generate_args.get("sampling_params", {})
        if isinstance(sampling_params, str):
            sampling_params = json.loads(sampling_params)
        if not isinstance(sampling_params, dict):
            raise TypeError("sglang sampling_params must be a dict or JSON string")
        return SamplingParams(**sampling_params)

    def _engine_loop(self) -> None:
        try:
            from sglang import Engine  # type: ignore[import-not-found, import-untyped, unused-ignore]

            self.engine = Engine(model_path=self.model, **self.engine_args)
        except Exception as exc:
            error_message = str(_safe_provider_execution_error("sglang", self.model, "engine initialization", exc))
            if self.on_error == "raise":
                with self.error_lock:
                    if self.error_message is None:
                        self.error_message = error_message
            self.engine_error_message = error_message
            self.engine_ready.set()
            return
        self.engine_ready.set()

        while True:
            item = self._submit_queue.get()
            if item is None:
                break
            _request_id, prompt, row = item
            try:
                output = self.engine.generate(prompt, self.sampling_params)
                self.completed_tasks.append((self._extract_output_text(output), row))
            except Exception as exc:
                if self.on_error == "raise":
                    error_message = str(_safe_provider_execution_error("sglang", self.model, "generation", exc))
                    with self.error_lock:
                        if self.error_message is None:
                            self.error_message = error_message
                    self._notify_state_change(force=True)
                else:
                    self.completed_tasks.append((None, row))
            finally:
                with self.task_count_lock:
                    self.running_task_count -= 1
                self._notify_state_change()

    def _extract_output_text(self, output: Any) -> str:
        if isinstance(output, dict):
            return str(output["text"])
        return str(getattr(output, "text"))

    def submit(self, _prefix: str | None, prompts: list[str], rows: pa.Table) -> None:
        rows = _ensure_table(rows)
        if len(prompts) != rows.num_rows:
            raise ValueError("Number of prompts and rows must match")
        if not self.engine_ready.is_set():
            self._wait_for_engine_ready_blocking()
        if self.engine_error_message is not None:
            if self.on_error == "raise":
                raise RuntimeError(f"sglang engine init failed: {self.engine_error_message}")
            for i in range(rows.num_rows):
                self.completed_tasks.append((None, rows.slice(i, 1)))
            self._notify_state_change(force=True)
            return
        with self.task_count_lock:
            self.running_task_count += len(prompts)
        for i, prompt in enumerate(prompts):
            with self.counter_lock:
                request_id = str(self.counter)
                self.counter += 1
            self._submit_queue.put((request_id, prompt, rows.slice(i, 1)))
        self._notify_state_change(force=True)

    def take_ready_result(self) -> tuple[list[str | None], pa.Table] | None:
        if self.error_message is not None:
            raise RuntimeError(f"sglang task failed: {self.error_message}")
        if not self.completed_tasks:
            return None
        output, row = self.completed_tasks.popleft()
        self._notify_state_change()
        return [output], row

    def finished_submitting(self) -> None:
        self._finished_submitting = True
        self._notify_state_change(force=True)

    def all_tasks_finished(self) -> bool:
        with self.task_count_lock:
            return self._finished_submitting and self.running_task_count == 0 and len(self.completed_tasks) == 0

    def _wakeup_ready(self) -> bool:
        if self._shutdown_called or self.error_message is not None:
            return True
        if self.completed_tasks:
            return True
        return self._finished_submitting and self.running_task_count == 0

    def wait_for_result(self) -> bool:
        with self._result_cv:
            self._result_cv.wait_for(self._wakeup_ready)
            return self._wakeup_ready()

    def _wait_for_engine_ready_blocking(self) -> None:
        if self.engine_ready.is_set():
            return
        timeout_s = self.engine_init_timeout_s
        if timeout_s is None:
            self.engine_ready.wait()
            return
        if not self.engine_ready.wait(timeout_s):
            raise RuntimeError(f"sglang engine init did not finish before deadline ({timeout_s:.3f}s)")

    def shutdown(self) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self._finished_submitting = True
        self._notify_state_change(force=True)
        self._submit_queue.put(None)


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


def build_executor(model: str, options: Any | None) -> SGLangExecutor:
    opts = normalize_options(options)
    if opts.get("use_ray"):
        raise NotImplementedError("SGLang distributed (Ray) execution is not yet implemented")
    return SGLangLocalExecutor(
        model,
        opts["engine_args"],
        opts["generate_args"],
        on_error=opts["on_error"],
        engine_init_timeout_s=opts.get("engine_init_timeout_s"),
    )
