# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from numbers import Integral, Real
from typing import Any

import pyarrow as pa

from vane.runners.ray.safe_get import configured_ray_get_timeout_s, resolve_object_refs_blocking

# ---------------------------------------------------------------------------
# Shared inflight tracking across all RemoteVLLMExecutors on the same worker.
# Keyed by pool_name so all executors sharing the same vLLM actor pool see the
# TRUE global inflight per actor (not per-executor approximation).
# ---------------------------------------------------------------------------
_shared_inflight_lock = threading.Lock()
_shared_inflight: dict[str, list[int]] = {}


def _positive_float_env(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _query_deadline_remaining_s() -> float | None:
    deadline = _positive_float_env("VANE_QUERY_DEADLINE_EPOCH_S")
    if deadline is None:
        return None
    remaining = deadline - time.time()
    if remaining <= 0.0:
        raise TimeoutError("query deadline expired before vLLM wait")
    return remaining


def _bounded_query_timeout_s(timeout_s: float | None) -> float | None:
    deadline_remaining = _query_deadline_remaining_s()
    if timeout_s is None:
        return deadline_remaining
    timeout_s = max(0.0, float(timeout_s))
    if deadline_remaining is None:
        return timeout_s
    return min(timeout_s, deadline_remaining)


def _vllm_engine_init_timeout_s(value: Any | None = None) -> float | None:
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("vllm engine_init_timeout_s must be a finite non-negative number")
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError("vllm engine_init_timeout_s must be a finite non-negative number")
        return result
    return _positive_float_env("VANE_VLLM_ENGINE_INIT_TIMEOUT_S")


def _get_shared_inflight(pool_key: str, num_actors: int) -> list[int]:
    """Return the shared inflight list for a pool, creating if needed."""
    with _shared_inflight_lock:
        if pool_key not in _shared_inflight:
            _shared_inflight[pool_key] = [0] * num_actors
        return _shared_inflight[pool_key]


class VLLMExecutor(ABC):
    @abstractmethod
    def submit(self, _prefix: str | None, prompts: list[str], rows: pa.Table) -> None:
        pass

    @abstractmethod
    def take_ready_result(self) -> tuple[list[str | None], pa.Table] | None:
        pass

    @abstractmethod
    def finished_submitting(self) -> None:
        pass

    @abstractmethod
    def all_tasks_finished(self) -> bool:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass


def _ensure_table(rows: Any) -> pa.Table:
    if isinstance(rows, pa.Table):
        return rows
    if isinstance(rows, pa.RecordBatch):
        return pa.Table.from_batches([rows])
    if isinstance(rows, pa.RecordBatchReader):
        return pa.Table.from_batches(list(rows))
    raise TypeError("rows must be a pyarrow Table, RecordBatch, or RecordBatchReader")


def _concat_tables(tables: list[pa.Table]) -> pa.Table:
    if not tables:
        return pa.table({})
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables)


class LocalVLLMExecutor(VLLMExecutor):
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
        from vllm import SamplingParams

        self.model = model
        self.engine_args = dict(engine_args)
        self.llm: Any = None
        self.engine_ready = threading.Event()
        self.engine_error_message: str | None = None
        self.engine_init_timeout_s = _vllm_engine_init_timeout_s(engine_init_timeout_s)

        sampling_params = generate_args.pop("sampling_params", None)
        if sampling_params is None:
            self.sampling_params = SamplingParams()
        elif isinstance(sampling_params, SamplingParams):
            self.sampling_params = sampling_params
        else:
            if isinstance(sampling_params, str):
                try:
                    sampling_params = json.loads(sampling_params)
                except json.JSONDecodeError as exc:
                    raise ValueError("vllm sampling_params JSON could not be parsed") from exc
            if isinstance(sampling_params, dict):
                self.sampling_params = SamplingParams(**sampling_params)
            else:
                raise TypeError("vllm sampling_params must be a dict, JSON string, or SamplingParams instance")
        self.generate_args = generate_args

        self.counter = 0
        self.counter_lock = threading.Lock()

        self.running_task_count = 0
        self.task_count_lock = threading.Lock()

        self.completed_tasks: deque[tuple[str | None, pa.Table]] = deque()
        self.error_message: str | None = None
        self.error_lock = threading.Lock()
        self.on_error = on_error

        # Condition variable for WaitForResult(): C++ blocks here until a
        # result is available.
        self._result_cv = threading.Condition(threading.Lock())

        # Dedicated async Ray actors use their actor loop. Synchronous wrappers
        # hosted inside a generic Ray actor need their own background loop.
        self._ray_actor_mode = self._detect_ray_actor() and not force_background_thread

        self.use_threading = use_threading
        if self._ray_actor_mode:
            self._init_engine_sync()
        elif self.use_threading:
            self.loop_ready = threading.Event()
            self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self.loop_thread.start()
            if not self.loop_ready.wait(_bounded_query_timeout_s(self.engine_init_timeout_s)):
                raise RuntimeError(f"vllm event loop did not start before {self._engine_init_deadline_description()}")

        self._finished_submitting = False
        self._shutdown_called = False
        # Per-executor result deques: distributed executors submit/read with
        # a unique executor_id so results are never stolen across tasks.
        self._per_executor_deques: dict[str, deque[tuple[str | None, pa.Table]]] = {}
        self._per_executor_running_task_count: dict[str, int] = {}
        self._per_executor_finished: set[str] = set()

    @staticmethod
    def _detect_ray_actor() -> bool:
        try:
            import ray

            ctx = ray.get_runtime_context()
            return ctx.get_actor_id() is not None
        except Exception:
            return False

    def _init_engine_sync(self) -> None:
        """Synchronous engine init — blocks until engine is ready.

        Used in Ray actor mode so that the actor's __init__ doesn't return
        until the engine is fully initialized.
        """
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine

            args = AsyncEngineArgs(model=self.model, **self.engine_args)
            self.llm = AsyncLLMEngine.from_engine_args(args)
        except Exception as exc:
            if self.on_error == "raise":
                with self.error_lock:
                    if self.error_message is None:
                        self.error_message = f"{type(exc).__name__}: {exc}"
            self.engine_error_message = f"{type(exc).__name__}: {exc}"
        finally:
            self.engine_ready.set()

    async def _init_engine(self) -> None:
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine

            args = AsyncEngineArgs(model=self.model, **self.engine_args)
            self.llm = AsyncLLMEngine.from_engine_args(args)
        except Exception as exc:
            if self.on_error == "raise":
                with self.error_lock:
                    if self.error_message is None:
                        self.error_message = f"{type(exc).__name__}: {exc}"
            self.engine_error_message = f"{type(exc).__name__}: {exc}"
        finally:
            self.engine_ready.set()

    def _run_event_loop(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self._init_engine())
        self.loop_ready.set()
        try:
            self.loop.run_forever()
        finally:
            pending = [task for task in asyncio.all_tasks(self.loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.close()
            asyncio.set_event_loop(None)

    async def _generate(self, prompt: str, row: pa.Table, executor_id: str | None = None) -> None:
        # Route results to per-executor deque when executor_id is given,
        # otherwise use the shared completed_tasks deque.
        target_deque = (
            self._per_executor_deques.get(executor_id, self.completed_tasks) if executor_id else self.completed_tasks
        )
        try:
            if not self._ray_actor_mode and not self.engine_ready.is_set():
                await self._wait_for_engine_ready_async()
            if self.engine_error_message is not None:
                raise RuntimeError(f"vllm engine init failed: {self.engine_error_message}")
            if self.llm is None:
                raise RuntimeError("vllm engine not initialized")
            with self.counter_lock:
                request_id = self.counter
                self.counter += 1

            final_output = None
            async for output in self.llm.generate(prompt, self.sampling_params, str(request_id), **self.generate_args):
                final_output = output

            if final_output is None or not final_output.outputs:
                raise RuntimeError("vllm returned no outputs")

            output_text: str = final_output.outputs[0].text
            target_deque.append((output_text, row))
            with self._result_cv:
                self._result_cv.notify_all()
        except Exception as exc:
            if self.on_error == "raise":
                with self.error_lock:
                    if self.error_message is None:
                        self.error_message = f"{type(exc).__name__}: {exc}"
                with self._result_cv:
                    self._result_cv.notify_all()
            else:
                target_deque.append((None, row))
                with self._result_cv:
                    self._result_cv.notify_all()
        finally:
            with self.task_count_lock:
                self.running_task_count -= 1
                if executor_id:
                    remaining = self._per_executor_running_task_count.get(executor_id, 0) - 1
                    self._per_executor_running_task_count[executor_id] = max(0, remaining)
                if (self._finished_submitting and self.running_task_count == 0) or (
                    executor_id and executor_id in self._per_executor_finished
                ):
                    with self._result_cv:
                        self._result_cv.notify_all()

    def _append_error_rows(self, rows: pa.Table, executor_id: str | None = None) -> None:
        rows = _ensure_table(rows)
        target_deque = (
            self._per_executor_deques.get(executor_id, self.completed_tasks) if executor_id else self.completed_tasks
        )
        for i in range(rows.num_rows):
            target_deque.append((None, rows.slice(i, 1)))
        with self._result_cv:
            self._result_cv.notify_all()

    def submit(self, _prefix: str | None, prompts: list[str], rows: pa.Table) -> None:
        rows = _ensure_table(rows)
        if len(prompts) != rows.num_rows:
            raise ValueError("Number of prompts and rows must match")

        if not self.use_threading:
            raise ValueError("Synchronous mode not supported when use_threading is False")

        self._wait_for_engine_ready_blocking()
        if self.engine_error_message is not None:
            if self.on_error == "raise":
                raise RuntimeError(f"vllm engine init failed: {self.engine_error_message}")
            self._append_error_rows(rows)
            return
        with self.task_count_lock:
            self.running_task_count += len(prompts)

        for i, prompt in enumerate(prompts):
            row = rows.slice(i, 1)
            asyncio.run_coroutine_threadsafe(self._generate(prompt, row), self.loop)

    async def submit_async(self, prompts: list[str], rows: pa.Table, executor_id: str | None = None) -> None:
        rows = _ensure_table(rows)
        if len(prompts) != rows.num_rows:
            raise ValueError("Number of prompts and rows must match")

        # Create per-executor deque on first submit from this executor.
        if executor_id and executor_id not in self._per_executor_deques:
            self._per_executor_deques[executor_id] = deque()
        if executor_id and executor_id in self._per_executor_finished:
            raise RuntimeError(f"vllm executor {executor_id} is already finished")

        if self._ray_actor_mode:
            # Engine is already ready from sync __init__; skip wait.
            if self.engine_error_message is not None:
                if self.on_error == "raise":
                    raise RuntimeError(f"vllm engine init failed: {self.engine_error_message}")
                self._append_error_rows(rows, executor_id)
                return

            with self.task_count_lock:
                self.running_task_count += len(prompts)
                if executor_id:
                    self._per_executor_running_task_count[executor_id] = self._per_executor_running_task_count.get(
                        executor_id, 0
                    ) + len(prompts)

            for i, prompt in enumerate(prompts):
                row = rows.slice(i, 1)
                # Run _generate on Ray's actor event loop (same loop as
                # vLLM engine's async IPC — avoids cross-thread scheduling).
                asyncio.create_task(self._generate(prompt, row, executor_id))
        else:
            # Background-thread mode for non-Ray use.
            if not self.engine_ready.is_set():
                await self._wait_for_engine_ready_async()
            if self.engine_error_message is not None:
                if self.on_error == "raise":
                    raise RuntimeError(f"vllm engine init failed: {self.engine_error_message}")
                self._append_error_rows(rows, executor_id)
                return

            with self.task_count_lock:
                self.running_task_count += len(prompts)
                if executor_id:
                    self._per_executor_running_task_count[executor_id] = self._per_executor_running_task_count.get(
                        executor_id, 0
                    ) + len(prompts)

            for i, prompt in enumerate(prompts):
                row = rows.slice(i, 1)
                # Must schedule in self.loop (where vLLM engine lives), NOT the
                # current event loop (Ray's).  asyncio.Event is not thread-safe;
                # _generate() awaits vLLM's internal Events that are set() by the
                # output_handler running in self.loop.
                asyncio.run_coroutine_threadsafe(self._generate(prompt, row, executor_id), self.loop)

    def take_ready_result(self, executor_id: str | None = None) -> tuple[list[str | None], pa.Table] | None:
        if self.error_message is not None and self.on_error == "raise":
            raise RuntimeError(f"vllm task failed: {self.error_message}")

        source_deque = (
            self._per_executor_deques.get(executor_id, self.completed_tasks) if executor_id else self.completed_tasks
        )
        try:
            output, row = source_deque.popleft()
        except IndexError:
            return None

        return [output], row

    def finished_submitting(self) -> None:
        self._finished_submitting = True
        with self._result_cv:
            self._result_cv.notify_all()

    def _engine_ready_wait_timeout_s(self) -> float | None:
        return _bounded_query_timeout_s(self.engine_init_timeout_s)

    def _engine_init_deadline_message(self) -> str:
        return f"vllm engine init did not finish before {self._engine_init_deadline_description()}"

    def _engine_init_deadline_description(self) -> str:
        timeout_s = self.engine_init_timeout_s
        if timeout_s is None:
            return "query deadline"
        return f"deadline ({timeout_s:.3f}s)"

    def _wait_for_engine_ready_blocking(self) -> None:
        if self.engine_ready.is_set():
            return
        timeout_s = self._engine_ready_wait_timeout_s()
        if timeout_s is None:
            self.engine_ready.wait()
            return
        if not self.engine_ready.wait(timeout_s):
            raise RuntimeError(self._engine_init_deadline_message())

    async def _wait_for_engine_ready_async(self) -> None:
        if self.engine_ready.is_set():
            return
        timeout_s = self._engine_ready_wait_timeout_s()
        if timeout_s is None:
            await asyncio.to_thread(self.engine_ready.wait)
            return
        ready = await asyncio.to_thread(self.engine_ready.wait, timeout_s)
        if not ready:
            raise RuntimeError(self._engine_init_deadline_message())

    def finished_executor(self, executor_id: str) -> None:
        self._per_executor_finished.add(executor_id)
        with self._result_cv:
            self._result_cv.notify_all()

    def all_tasks_finished(self) -> bool:
        with self.task_count_lock:
            return self._finished_submitting and self.running_task_count == 0 and len(self.completed_tasks) == 0

    def _wait_for_result_blocking(self, executor_id: str | None = None) -> bool:
        source_deque = (
            self._per_executor_deques.get(executor_id, self.completed_tasks) if executor_id else self.completed_tasks
        )

        def done() -> bool:
            if executor_id:
                return (
                    executor_id in self._per_executor_finished
                    and self._per_executor_running_task_count.get(executor_id, 0) == 0
                )
            return self._finished_submitting and self.running_task_count == 0

        with self._result_cv:
            self._result_cv.wait_for(lambda: len(source_deque) > 0 or self.error_message is not None or done())
            return len(source_deque) > 0

    def wait_for_result(self, executor_id: str | None = None) -> bool:
        """Block until at least one result is available or all tasks are done."""
        return self._wait_for_result_blocking(executor_id)

    def shutdown(self) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self._finished_submitting = True
        with self._result_cv:
            self._result_cv.notify_all()
        loop = getattr(self, "loop", None)
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)


class RayLocalVLLMExecutor(LocalVLLMExecutor):
    # Ray actor calls are awaitable, while the in-process executor exposes a blocking method.
    async def wait_for_result(self, executor_id: str | None = None) -> bool:  # type: ignore[override]
        return await asyncio.to_thread(self._wait_for_result_blocking, executor_id)


class RemoteVLLMExecutor(VLLMExecutor):
    def __init__(self, llm_actors: LLMActors, pool_name: str | None = None):
        self.router_actor = llm_actors.router_actor
        resolve_object_refs_blocking(self.router_actor.report_start.remote())

        self.llm_actors = llm_actors.llm_actors
        self._result_cv = threading.Condition(threading.Lock())
        self._finished = False
        self._finished_submitting_flag = False
        self._submit_per_actor = [0] * len(self.llm_actors)
        # Shared inflight tracking: all executors for the same pool see the
        # true global inflight per actor, enabling accurate load-aware routing.
        self._pool_key = pool_name or f"_anon_{id(llm_actors)}"
        self._inflight_per_actor = _get_shared_inflight(self._pool_key, len(self.llm_actors))
        self._inflight_lock = threading.Lock()

        # Unique ID for per-executor result isolation on actors
        import uuid

        self._executor_id = str(uuid.uuid4())

        self._result_buffer: deque[tuple[list[str | None], pa.Table]] = deque()
        self._error_message: str | None = None
        self._shutdown_called = False
        self._results_per_actor = [0] * len(self.llm_actors)
        self._wait_refs_by_actor: list[Any | None] = [None] * len(self.llm_actors)
        self._submit_refs: dict[Any, tuple[int, int]] = {}
        self._released_outstanding_inflight = False

    def _actor_has_pending_result(self, actor_idx: int) -> bool:
        return self._results_per_actor[actor_idx] < self._submit_per_actor[actor_idx]

    def _ensure_wait_ref(self, actor_idx: int) -> None:
        if self._wait_refs_by_actor[actor_idx] is not None:
            return
        if not self._actor_has_pending_result(actor_idx):
            return
        actor = self.llm_actors[actor_idx]
        wait_ref = actor.wait_for_result.remote(self._executor_id)
        self._wait_refs_by_actor[actor_idx] = wait_ref
        try:
            wait_ref.future().add_done_callback(lambda _future, _ref=wait_ref: self._handle_wait_ref_ready(_ref))
        except Exception as exc:
            self._wait_refs_by_actor[actor_idx] = None
            self._record_error(TypeError(f"vllm wait ObjectRef does not support completion callbacks: {exc}"))

    def _actor_index_for_wait_ref(self, ready_ref: Any) -> int:
        for actor_idx, ref in enumerate(self._wait_refs_by_actor):
            if ref == ready_ref:
                return actor_idx
        raise RuntimeError("vllm remote wait returned an unknown actor ref")

    def _take_ready_wait_ref(self, ready_ref: Any) -> int | None:
        with self._result_cv:
            if self._shutdown_called or self._finished:
                return None
            actor_idx = self._actor_index_for_wait_ref(ready_ref)
            return actor_idx

    def _handle_wait_ref_ready(self, ready_ref: Any) -> None:
        try:
            actor_idx = self._take_ready_wait_ref(ready_ref)
            if actor_idx is None:
                return
            ready = resolve_object_refs_blocking(ready_ref)
            self._drain_ready_actor(actor_idx, ready, ready_ref)
        except Exception as exc:
            self._record_error(exc)

    def _drain_ready_actor(self, actor_idx: int, ready: bool, ready_ref: Any) -> None:
        actor = self.llm_actors[actor_idx]
        if not ready:
            if self._finished_submitting_flag:
                if sum(self._results_per_actor) >= sum(self._submit_per_actor):
                    self._mark_finished()
                else:
                    self._record_error(
                        RuntimeError(
                            "vllm actor finished without returning all submitted results: "
                            f"actor_idx={actor_idx} submitted={self._submit_per_actor[actor_idx]} "
                            f"received={self._results_per_actor[actor_idx]}"
                        )
                    )
                return
            raise RuntimeError("vllm actor wait completed without a result")

        result = resolve_object_refs_blocking(actor.take_ready_result.remote(self._executor_id))
        if result is None:
            raise RuntimeError("vllm actor reported readiness but returned no result")

        results_text, _ = result
        n_results = len(results_text) if results_text else 0
        with self._inflight_lock:
            self._inflight_per_actor[actor_idx] -= n_results
        self._results_per_actor[actor_idx] += n_results
        with self._result_cv:
            if self._wait_refs_by_actor[actor_idx] == ready_ref:
                self._wait_refs_by_actor[actor_idx] = None
            self._result_buffer.append(result)
            self._result_cv.notify_all()

    def _ensure_remote_wait_refs(self) -> None:
        for actor_idx in range(len(self.llm_actors)):
            self._ensure_wait_ref(actor_idx)
        if not any(ref is not None for ref in self._wait_refs_by_actor) and self._finished_submitting_flag:
            if sum(self._results_per_actor) >= sum(self._submit_per_actor):
                self._mark_finished()

    def _has_pending_wait_ref(self) -> bool:
        return any(ref is not None for ref in self._wait_refs_by_actor)

    def _rollback_submitted_actor_batch(self, actor_idx: int, prompt_count: int) -> None:
        with self._inflight_lock:
            self._inflight_per_actor[actor_idx] = max(0, self._inflight_per_actor[actor_idx] - prompt_count)
        with self._result_cv:
            self._submit_per_actor[actor_idx] = max(0, self._submit_per_actor[actor_idx] - prompt_count)
            self._result_cv.notify_all()

    def _release_outstanding_inflight(self) -> None:
        with self._inflight_lock:
            if getattr(self, "_released_outstanding_inflight", False):
                return
            for actor_idx in range(len(self._inflight_per_actor)):
                outstanding = max(0, self._submit_per_actor[actor_idx] - self._results_per_actor[actor_idx])
                if outstanding:
                    self._inflight_per_actor[actor_idx] = max(0, self._inflight_per_actor[actor_idx] - outstanding)
            self._released_outstanding_inflight = True

    def _take_submit_ref(self, ready_ref: Any) -> tuple[int, int] | None:
        with self._result_cv:
            if self._shutdown_called or self._finished:
                return None
            return self._submit_refs.pop(ready_ref, None)

    def _handle_submit_ref_ready(self, ready_ref: Any) -> None:
        submit_meta = self._take_submit_ref(ready_ref)
        if submit_meta is None:
            return
        actor_idx, prompt_count = submit_meta
        try:
            resolve_object_refs_blocking(ready_ref)
        except Exception as exc:
            self._rollback_submitted_actor_batch(actor_idx, prompt_count)
            self._record_error(exc)

    def _track_submit_ref(self, submit_ref: Any, actor_idx: int, prompt_count: int) -> None:
        with self._result_cv:
            if self._shutdown_called or self._finished:
                return
            self._submit_refs[submit_ref] = (actor_idx, prompt_count)
        try:
            submit_ref.future().add_done_callback(lambda _future, _ref=submit_ref: self._handle_submit_ref_ready(_ref))
        except Exception as exc:
            submit_meta = self._take_submit_ref(submit_ref)
            if submit_meta is not None:
                rollback_actor_idx, rollback_prompt_count = submit_meta
                self._rollback_submitted_actor_batch(rollback_actor_idx, rollback_prompt_count)
            self._record_error(TypeError(f"vllm submit ObjectRef does not support completion callbacks: {exc}"))

    def _record_error(self, exc: Exception) -> None:
        self._release_outstanding_inflight()
        self._cancel_remote_refs()
        with self._result_cv:
            if self._error_message is None:
                self._error_message = f"{type(exc).__name__}: {exc}"
            self._finished = True
            self._wait_refs_by_actor = [None] * len(self.llm_actors)
            self._submit_refs.clear()
            self._result_cv.notify_all()

    def _mark_finished(self) -> None:
        self._release_outstanding_inflight()
        with self._result_cv:
            self._finished = True
            self._wait_refs_by_actor = [None] * len(self.llm_actors)
            self._submit_refs.clear()
            self._result_cv.notify_all()

    def _cancel_remote_refs(self) -> None:
        refs = list(self._submit_refs)
        refs.extend(ref for ref in self._wait_refs_by_actor if ref is not None)
        if not refs:
            return
        try:
            import ray
        except Exception:
            return
        for ref in refs:
            try:
                ray.cancel(ref)
            except Exception:
                pass

    def submit(self, _prefix: str | None, prompts: list[str], rows: pa.Table) -> None:
        if self._shutdown_called:
            raise RuntimeError("vllm remote executor is shut down")
        prompt_count = len(prompts)
        # Route to actor with lowest actual inflight (adapts to processing speed)
        with self._inflight_lock:
            route_to = min(range(len(self.llm_actors)), key=lambda i: self._inflight_per_actor[i])
            self._inflight_per_actor[route_to] += prompt_count

        try:
            submit_ref = self.llm_actors[route_to].submit_async.remote(prompts, rows, self._executor_id)
        except Exception:
            with self._inflight_lock:
                self._inflight_per_actor[route_to] = max(0, self._inflight_per_actor[route_to] - prompt_count)
            raise

        self._submit_per_actor[route_to] += prompt_count
        self._track_submit_ref(submit_ref, route_to, prompt_count)

    def take_ready_result(self) -> tuple[list[str | None], pa.Table] | None:
        if self._error_message is not None:
            raise RuntimeError(f"vllm remote task failed: {self._error_message}")
        try:
            return self._result_buffer.popleft()
        except IndexError:
            return None

    def finished_submitting(self) -> None:
        if self._finished_submitting_flag:
            return
        self._finished_submitting_flag = True
        errors: list[Exception] = []
        for actor in self.llm_actors:
            try:
                resolve_object_refs_blocking(actor.finished_executor.remote(self._executor_id))
            except Exception as exc:
                errors.append(exc)
        try:
            resolve_object_refs_blocking(self.router_actor.report_completion.remote())
        except Exception as exc:
            errors.append(exc)
        if errors:
            if len(errors) == 1:
                raise RuntimeError(f"vllm remote finished_submitting failed: {errors[0]}") from errors[0]
            message = "; ".join(str(error) for error in errors)
            raise RuntimeError(f"vllm remote finished_submitting failed: {message}") from errors[0]

    def all_tasks_finished(self) -> bool:
        if self._result_buffer:
            return False
        if self._error_message is not None:
            raise RuntimeError(f"vllm remote task failed: {self._error_message}")
        if not self._finished_submitting_flag:
            return False
        # Per-task completion: received at least as many results as submitted
        total_submitted = sum(self._submit_per_actor)
        total_received = sum(self._results_per_actor)
        if total_submitted == 0:
            self._mark_finished()
            return True
        if total_submitted > 0 and total_received >= total_submitted:
            self._mark_finished()
            return True
        return False

    def wait_for_result(self) -> None:
        """Block until a result is available in the buffer."""
        if self._error_message is not None:
            raise RuntimeError(f"vllm remote task failed: {self._error_message}")
        if self._result_buffer or self._finished:
            return
        try:
            self._ensure_remote_wait_refs()
            with self._result_cv:
                if (
                    not self._result_buffer
                    and self._error_message is None
                    and not self._finished
                    and not self._has_pending_wait_ref()
                ):
                    raise RuntimeError("vllm remote wait has no pending actor wait refs before completion")
                timeout_s = configured_ray_get_timeout_s()
                ready = self._result_cv.wait_for(
                    lambda: bool(self._result_buffer) or self._error_message is not None or self._finished,
                    timeout=timeout_s,
                )
                if not ready:
                    raise RuntimeError("vllm remote wait exceeded query deadline")
        except Exception as exc:
            self._record_error(exc)
        if self._error_message is not None:
            raise RuntimeError(f"vllm remote task failed: {self._error_message}")

    def shutdown(self) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True
        completion_error: Exception | None = None
        try:
            if not self._finished_submitting_flag:
                self.finished_submitting()
        except Exception as exc:
            completion_error = exc
        finally:
            self._mark_finished()
        if completion_error is not None:
            raise completion_error


class PrefixRouter:
    def __init__(self, llm_actors: list[Any], _load_balance_threshold: int, _max_recent_prefixes: int = 8):
        self.llm_actors = llm_actors
        self.unfinished_actors = 0

    def report_start(self) -> None:
        self.unfinished_actors += 1

    def report_completion(self) -> None:
        if self.unfinished_actors <= 0:
            raise RuntimeError("vllm router received completion without a matching start")
        self.unfinished_actors -= 1
        if self.unfinished_actors == 0:
            for actor in self.llm_actors:
                resolve_object_refs_blocking(actor.finished_submitting.remote())


class LLMActors:
    def __init__(
        self,
        model: str,
        engine_args: dict[str, Any],
        generate_args: dict[str, Any],
        on_error: str,
        gpus_per_actor: int | float,
        concurrency: int,
        load_balance_threshold: int,
        name_prefix: str | None = None,
        engine_init_timeout_s: float | None = None,
    ):
        import ray

        LocalVLLMExecutorActor = ray.remote(num_gpus=gpus_per_actor, max_restarts=4)(RayLocalVLLMExecutor)
        PrefixRouterActor = ray.remote(PrefixRouter)

        if name_prefix:
            llm_names = [f"{name_prefix}-llm-{i}" for i in range(concurrency)]
            self.llm_actors = [
                LocalVLLMExecutorActor.options(name=llm_name).remote(
                    model,
                    engine_args,
                    generate_args,
                    on_error,
                    engine_init_timeout_s=engine_init_timeout_s,
                )
                for llm_name in llm_names
            ]
            self.router_actor = PrefixRouterActor.options(name=f"{name_prefix}-router").remote(
                self.llm_actors, load_balance_threshold
            )
        else:
            self.llm_actors = [
                LocalVLLMExecutorActor.remote(
                    model,
                    engine_args,
                    generate_args,
                    on_error,
                    engine_init_timeout_s=engine_init_timeout_s,
                )
                for _ in range(concurrency)
            ]
            self.router_actor = PrefixRouterActor.remote(self.llm_actors, load_balance_threshold)

    @classmethod
    def _from_handles(cls, llm_actors: list[Any], router_actor: Any) -> LLMActors:
        instance = cls.__new__(cls)
        instance.llm_actors = llm_actors
        instance.router_actor = router_actor
        return instance

    @classmethod
    def get_or_create_named(
        cls,
        *,
        model: str,
        engine_args: dict[str, Any],
        generate_args: dict[str, Any],
        on_error: str,
        gpus_per_actor: int | float,
        concurrency: int,
        load_balance_threshold: int,
        name_prefix: str,
        engine_init_timeout_s: float | None = None,
    ) -> LLMActors:
        import ray

        router_name = f"{name_prefix}-router"
        llm_names = [f"{name_prefix}-llm-{i}" for i in range(concurrency)]

        missing: list[str] = []
        try:
            router_actor = ray.get_actor(router_name)
        except ValueError:
            router_actor = None
            missing.append(router_name)

        llm_actors: list[Any] = []
        for llm_name in llm_names:
            try:
                llm_actors.append(ray.get_actor(llm_name))
            except ValueError:
                missing.append(llm_name)

        found = (1 if router_actor is not None else 0) + len(llm_actors)
        expected = 1 + concurrency
        if found == expected:
            return cls._from_handles(llm_actors, router_actor)
        if found:
            raise RuntimeError(
                f"Named vLLM actor pool '{name_prefix}' partially available: "
                f"found={found} missing={len(missing)} expected={expected} "
                f"missing_names={', '.join(missing)}"
            )

        return cls(
            model=model,
            engine_args=engine_args,
            generate_args=generate_args,
            on_error=on_error,
            gpus_per_actor=gpus_per_actor,
            concurrency=concurrency,
            load_balance_threshold=load_balance_threshold,
            name_prefix=name_prefix,
            engine_init_timeout_s=engine_init_timeout_s,
        )

    @classmethod
    def lookup_named(
        cls,
        *,
        concurrency: int,
        name_prefix: str,
    ) -> LLMActors:
        import ray

        router_name = f"{name_prefix}-router"
        llm_names = [f"{name_prefix}-llm-{i}" for i in range(concurrency)]
        try:
            router_actor = ray.get_actor(router_name)
        except ValueError as exc:
            raise RuntimeError(f"Named vLLM actor pool '{name_prefix}' router was not found") from exc
        missing: list[str] = []
        llm_actors = []
        for llm_name in llm_names:
            try:
                llm_actors.append(ray.get_actor(llm_name))
            except ValueError:
                missing.append(llm_name)
        if missing:
            raise RuntimeError(
                f"Named vLLM actor pool '{name_prefix}' is incomplete; missing actors: {', '.join(missing)}"
            )
        return cls._from_handles(llm_actors, router_actor)


_DEFAULTS: dict[str, Any] = {
    "concurrency": 1,
    "gpus_per_actor": 1,
    "do_prefix_routing": True,
    "max_buffer_size": 5000,
    "min_bucket_size": 16,
    "prefix_match_threshold": 0.33,
    "load_balance_threshold": 32,
    "batch_size": 128,
    "on_error": "raise",
    "engine_args": {},
    "generate_args": {},
    "use_ray": False,
    "use_threading": True,
    "inflight_limit": 128,
    "engine_init_timeout_s": None,
}


def _integer_option(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"vllm {name} must be an integer")
    result = int(value)
    if result < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"vllm {name} must be {qualifier}")
    return result


def _fractional_gpu_option(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("vllm gpus_per_actor must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("vllm gpus_per_actor must be a finite positive number")
    if result >= 1:
        if not result.is_integer():
            raise ValueError("vllm gpus_per_actor values >= 1 must be integers")
        return int(result)
    return result


def _unit_interval_option(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"vllm {name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"vllm {name} must be a finite number in [0, 1]")
    return result


def normalize_options(options: Any | None) -> dict[str, Any]:
    merged = dict(_DEFAULTS)
    if isinstance(options, str):
        try:
            parsed = json.loads(options)
        except json.JSONDecodeError as exc:
            raise ValueError("vllm options JSON could not be parsed") from exc
        if not isinstance(parsed, dict):
            raise ValueError("vllm options JSON must decode to a dict")
        options = parsed
    elif options is not None and not isinstance(options, dict):
        try:
            options = dict(options)
        except Exception as exc:
            raise TypeError("vllm options must be a dict or JSON string") from exc

    if options is not None and options.get("ray_address") is not None:
        raise ValueError("vLLM ray_address has been removed; configure RayRunner instead")

    if options is not None:
        merged.update(options)
    merged["concurrency"] = _integer_option("concurrency", merged["concurrency"], minimum=1)
    merged["gpus_per_actor"] = _fractional_gpu_option(merged["gpus_per_actor"])
    if not isinstance(merged["do_prefix_routing"], bool):
        raise ValueError("vllm do_prefix_routing must be a boolean")
    merged["max_buffer_size"] = _integer_option("max_buffer_size", merged["max_buffer_size"], minimum=0)
    merged["min_bucket_size"] = _integer_option("min_bucket_size", merged["min_bucket_size"], minimum=0)
    merged["prefix_match_threshold"] = _unit_interval_option("prefix_match_threshold", merged["prefix_match_threshold"])
    merged["load_balance_threshold"] = _integer_option(
        "load_balance_threshold", merged["load_balance_threshold"], minimum=0
    )
    batch_size = merged["batch_size"]
    merged["batch_size"] = None if batch_size is None else _integer_option("batch_size", batch_size, minimum=1)
    merged["inflight_limit"] = _integer_option("inflight_limit", merged["inflight_limit"], minimum=0)
    merged["engine_init_timeout_s"] = _vllm_engine_init_timeout_s(merged.get("engine_init_timeout_s"))
    on_error = merged.get("on_error")
    if on_error is None:
        on_error = "raise"
    on_error = str(on_error).lower()
    if on_error not in ("raise", "log", "null"):
        raise ValueError("vllm on_error must be one of: raise, log, null")
    merged["on_error"] = on_error
    merged["engine_args"] = dict(merged["engine_args"] or {})
    merged["generate_args"] = dict(merged["generate_args"] or {})
    if "ray_actor_pool_name" in merged and merged["ray_actor_pool_name"] is not None:
        merged["ray_actor_pool_name"] = str(merged["ray_actor_pool_name"])
        # Shared actor pool: low inflight_limit (default 128) enforces streaming
        # submission — each plan submits small batches and blocks, spreading data
        # over the full task lifetime instead of submitting everything upfront.
        # Combined with shared inflight tracking across executors, routing
        # decisions reflect the true global load on each actor.
    return merged


def _is_ray_worker() -> bool:
    try:
        import ray
        from ray._private import worker as ray_worker
    except Exception:
        return False
    try:
        return ray.is_initialized() and ray_worker.global_worker.mode == ray_worker.WORKER_MODE
    except Exception:
        return False


def ensure_named_vllm_pools_for_plan(plan: Any, conn: Any = None) -> tuple[list[LLMActors], dict[str, Any]]:
    """Pre-create named Ray actor pools for vLLM nodes in a distributed plan.

    Called on the driver before task dispatch so that workers find actors
    already running instead of waiting for the first worker to initialise them.

    Returns ``(created_list, {})``.
    """
    # Skip on Vane worker processes (they discover actors by name).
    # RayQueryDriverActor is a Ray actor but NOT a Vane worker — it must
    # run pre-creation.
    if os.environ.get("VANE_WORKER") is not None:
        return [], {}

    vllm_nodes = plan.collect_vllm_nodes(conn=conn)

    if not vllm_nodes:
        return [], {}

    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray vLLM actor creation requires an initialized RayRunner runtime")

    created: list[LLMActors] = []
    for node in vllm_nodes:
        pool_name = str(node["pool_name"])
        model = str(node.get("model", ""))

        # Parse options through normalize_options to get clean defaults.
        raw_opts = node.get("options")
        opts = normalize_options(raw_opts)

        engine_args = _apply_engine_defaults(dict(opts.get("engine_args") or {}))
        generate_args = dict(opts.get("generate_args") or {})
        on_error = str(opts.get("on_error", "raise"))
        gpus_per_actor = opts["gpus_per_actor"]
        concurrency = max(1, int(opts.get("concurrency", 1)))
        load_balance_threshold = max(0, int(opts.get("load_balance_threshold", 32)))
        engine_init_timeout_s = _vllm_engine_init_timeout_s(opts.get("engine_init_timeout_s"))

        actors_obj = LLMActors.get_or_create_named(
            model=model,
            engine_args=engine_args,
            generate_args=generate_args,
            on_error=on_error,
            gpus_per_actor=gpus_per_actor,
            concurrency=concurrency,
            load_balance_threshold=load_balance_threshold,
            name_prefix=pool_name,
            engine_init_timeout_s=engine_init_timeout_s,
        )
        created.append(actors_obj)

    return created, {}


def _apply_engine_defaults(engine_args: dict[str, Any]) -> dict[str, Any]:
    """Inject throughput-oriented vLLM defaults.

    AsyncLLMEngine uses UsageContext.ENGINE_CONTEXT which falls through to
    conservative scheduler defaults (max_num_batched_tokens=2048,
    max_num_seqs=128).  For batch inference we want the throughput defaults
    that vLLM's LLM class uses (UsageContext.LLM_CLASS):
      max_num_batched_tokens=8192, max_num_seqs=256.
    """
    engine_args.setdefault("max_num_batched_tokens", 8192)
    engine_args.setdefault("max_num_seqs", 256)
    return engine_args


def build_executor(model: str, options: Any | None) -> VLLMExecutor:
    opts = normalize_options(options)
    engine_args = _apply_engine_defaults(dict(opts["engine_args"]))
    generate_args = dict(opts["generate_args"])
    pool_name = opts.get("ray_actor_pool_name")
    on_error = opts.get("on_error", "raise")
    require_ray_worker = bool(opts.get("require_ray_worker") or opts.get("ray_worker_only"))
    if require_ray_worker and not _is_ray_worker():
        raise RuntimeError("vllm executor must be constructed on a Ray worker when require_ray_worker is set")

    # `use_ray` is the only routing switch. Pool/address metadata only
    # configures Ray-backed execution after it has been explicitly selected.
    use_ray = bool(opts.get("use_ray"))
    if use_ray:
        import ray

        if not ray.is_initialized():
            raise RuntimeError("Ray vLLM execution requires an initialized RayRunner runtime")
        if pool_name:
            llm_actors = LLMActors.lookup_named(
                concurrency=opts["concurrency"],
                name_prefix=pool_name,
            )
        else:
            llm_actors = LLMActors(
                model=model,
                engine_args=engine_args,
                generate_args=generate_args,
                on_error=on_error,
                gpus_per_actor=opts["gpus_per_actor"],
                concurrency=opts["concurrency"],
                load_balance_threshold=opts["load_balance_threshold"],
                engine_init_timeout_s=opts["engine_init_timeout_s"],
            )
        return RemoteVLLMExecutor(llm_actors, pool_name=pool_name)

    return LocalVLLMExecutor(
        model,
        engine_args,
        generate_args,
        on_error=on_error,
        use_threading=bool(opts.get("use_threading", True)),
        engine_init_timeout_s=opts["engine_init_timeout_s"],
        force_background_thread=bool(opts.get("_force_background_thread", False)),
    )
