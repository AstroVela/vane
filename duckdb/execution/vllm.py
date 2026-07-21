# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import threading
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from decimal import Decimal
from numbers import Integral, Real
from typing import Any, Callable, overload

import pyarrow as pa

from duckdb.runners.ray.safe_get import configured_ray_get_timeout_s, resolve_object_refs_blocking


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
        if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
            raise ValueError("vllm engine_init_timeout_s must be a finite non-negative number")
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError("vllm engine_init_timeout_s must be a finite non-negative number")
        return result
    return _positive_float_env("VANE_VLLM_ENGINE_INIT_TIMEOUT_S")


def _vllm_control_rpc_timeout_s() -> float:
    default_timeout_s = 30.0
    raw_value = os.getenv("VANE_VLLM_CONTROL_RPC_TIMEOUT_S")
    if raw_value is None:
        return default_timeout_s
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default_timeout_s
    if not math.isfinite(value) or value <= 0.0:
        return default_timeout_s
    return value


def _resolve_vllm_control_ref(object_refs: Any) -> Any:
    """Resolve terminal/control RPCs with a budget independent of the query."""
    return resolve_object_refs_blocking(
        object_refs,
        timeout=_vllm_control_rpc_timeout_s(),
        honor_query_deadline=False,
    )


class VLLMExecutor(ABC):
    """Common execution contract shared by local and Ray-backed vLLM executors.

    Besides the submit/result lifecycle, the base class owns a one-shot wakeup
    protocol used by DuckDB's native scheduler.  A scheduler callback is armed
    only while no result or terminal state is ready, then consumed by the next
    relevant state change so a blocked pipeline task can be scheduled again.
    """

    def _ensure_wakeup_state(self) -> None:
        """Lazily initialize callback state for subclasses and test doubles."""
        if not hasattr(self, "_wakeup_lock"):
            self._wakeup_lock = threading.Lock()
        if not hasattr(self, "_wakeup_callbacks"):
            self._wakeup_callbacks: list[Callable[[], None]] = []

    def _wakeup_ready(self) -> bool:
        """Return whether the native scheduler should resume without arming."""
        return False

    def register_wakeup_callback(self, callback: Callable[[], None]) -> bool:
        """Arm a one-shot native wakeup unless work is already actionable.

        True means the callback is stored and the scheduler may safely block;
        False means it must immediately recheck results or terminal state.
        """
        if not callable(callback):
            raise TypeError("vllm wakeup callback must be callable")
        self._ensure_wakeup_state()
        with self._wakeup_lock:
            if self._wakeup_ready():
                return False
            self._wakeup_callbacks.append(callback)
            return True

    def _notify_state_change(self, *, force: bool = False) -> None:
        """Wake condition waiters and consume actionable native callbacks.

        Condition waiters are always notified.  Native callbacks are one-shot
        and run only when `_wakeup_ready()` is true, unless `force` requests an
        unconditional scheduler recheck after a state transition.
        """
        self._ensure_wakeup_state()
        callbacks: list[Callable[[], None]] = []
        with self._wakeup_lock:
            if force or self._wakeup_ready():
                callbacks = self._wakeup_callbacks
                self._wakeup_callbacks = []
        result_cv = getattr(self, "_result_cv", None)
        if result_cv is not None:
            with result_cv:
                result_cv.notify_all()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass

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
        self._result_cv = threading.Condition(threading.RLock())
        self._ensure_wakeup_state()

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
        self._per_executor_deques: dict[str, deque[tuple[Any, ...]]] = {}
        self._per_executor_running_task_count: dict[str, int] = {}
        self._per_executor_finished: set[str] = set()
        self._per_executor_request_ids: dict[str, set[str]] = {}
        self._per_executor_tasks: dict[str, set[Any]] = {}
        self._per_executor_errors: dict[str, str] = {}
        self._per_executor_aborted: set[str] = set()
        self._per_executor_waiters: dict[str, int] = {}
        self._per_executor_abort_wait_required: set[str] = set()
        self._per_executor_terminal_wait_observed: set[str] = set()
        self._async_waiter_lock = threading.Lock()
        self._async_waiters: dict[str, list[tuple[Any, asyncio.Event]]] = {}

    @staticmethod
    def _detect_ray_actor() -> bool:
        try:
            import ray

            if not ray.is_initialized():
                return False
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

    async def _generate(
        self,
        prompt: str,
        row: pa.Table,
        executor_id: str | None = None,
        reservation_id: str | None = None,
    ) -> None:
        request_id: str | None = None
        try:
            if not self._ray_actor_mode and not self.engine_ready.is_set():
                await self._wait_for_engine_ready_async()
            if self.engine_error_message is not None:
                raise RuntimeError(f"vllm engine init failed: {self.engine_error_message}")
            if self.llm is None:
                raise RuntimeError("vllm engine not initialized")
            with self.counter_lock:
                request_id = str(self.counter)
                self.counter += 1
            if executor_id:
                with self.task_count_lock:
                    self._per_executor_request_ids.setdefault(executor_id, set()).add(request_id)

            final_output = None
            async for output in self.llm.generate(prompt, self.sampling_params, request_id, **self.generate_args):
                final_output = output

            if final_output is None or not final_output.outputs:
                raise RuntimeError("vllm returned no outputs")

            output_text: str = final_output.outputs[0].text
            if executor_id:
                self._per_executor_deques.setdefault(executor_id, deque()).append((output_text, row, reservation_id))
            else:
                self.completed_tasks.append((output_text, row))
            self._notify_state_change()
        except Exception as exc:
            if self.on_error == "raise":
                error_message = f"{type(exc).__name__}: {exc}"
                if executor_id:
                    with self.task_count_lock:
                        self._per_executor_errors.setdefault(executor_id, error_message)
                else:
                    with self.error_lock:
                        if self.error_message is None:
                            self.error_message = error_message
                self._notify_state_change(force=True)
            else:
                if executor_id:
                    self._per_executor_deques.setdefault(executor_id, deque()).append((None, row, reservation_id))
                else:
                    self.completed_tasks.append((None, row))
                self._notify_state_change()
        finally:
            with self.task_count_lock:
                self.running_task_count -= 1
                if executor_id:
                    if request_id is not None:
                        self._per_executor_request_ids.get(executor_id, set()).discard(request_id)
                    remaining = self._per_executor_running_task_count.get(executor_id, 0) - 1
                    self._per_executor_running_task_count[executor_id] = max(0, remaining)
            self._notify_state_change()

    def _append_error_rows(
        self,
        rows: pa.Table,
        executor_id: str | None = None,
        reservation_id: str | None = None,
    ) -> None:
        rows = _ensure_table(rows)
        for i in range(rows.num_rows):
            row = rows.slice(i, 1)
            if executor_id:
                self._per_executor_deques.setdefault(executor_id, deque()).append((None, row, reservation_id))
            else:
                self.completed_tasks.append((None, row))
        self._notify_state_change()

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
        self._notify_state_change(force=True)

    async def submit_async(
        self,
        prompts: list[str],
        rows: pa.Table,
        executor_id: str | None = None,
        reservation_id: str | None = None,
    ) -> None:
        rows = _ensure_table(rows)
        if len(prompts) != rows.num_rows:
            raise ValueError("Number of prompts and rows must match")

        # Create per-executor deque on first submit from this executor.
        if executor_id and executor_id not in self._per_executor_deques:
            self._per_executor_deques[executor_id] = deque()
            self._per_executor_request_ids[executor_id] = set()
            self._per_executor_tasks[executor_id] = set()
        if executor_id and (
            executor_id in self._per_executor_finished
            or executor_id in self._per_executor_aborted
            or executor_id in self._per_executor_errors
        ):
            raise RuntimeError(f"vllm executor {executor_id} is already finished")

        if self._ray_actor_mode:
            # Engine is already ready from sync __init__; skip wait.
            if self.engine_error_message is not None:
                if self.on_error == "raise":
                    raise RuntimeError(f"vllm engine init failed: {self.engine_error_message}")
                self._append_error_rows(rows, executor_id, reservation_id)
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
                asyncio_task = asyncio.create_task(self._generate(prompt, row, executor_id, reservation_id))
                self._track_executor_task(executor_id, asyncio_task)
        else:
            # Background-thread mode for non-Ray use.
            if not self.engine_ready.is_set():
                await self._wait_for_engine_ready_async()
            if self.engine_error_message is not None:
                if self.on_error == "raise":
                    raise RuntimeError(f"vllm engine init failed: {self.engine_error_message}")
                self._append_error_rows(rows, executor_id, reservation_id)
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
                thread_future = asyncio.run_coroutine_threadsafe(
                    self._generate(prompt, row, executor_id, reservation_id), self.loop
                )
                self._track_executor_task(executor_id, thread_future)
        self._notify_state_change(force=True)

    def _track_executor_task(self, executor_id: str | None, task: Any) -> None:
        """Retain an executor task until its completion callback runs."""
        if not executor_id:
            return
        tasks = self._per_executor_tasks.setdefault(executor_id, set())
        tasks.add(task)

        def discard(done: Any) -> None:
            tasks.discard(done)

        task.add_done_callback(discard)

    @overload
    def take_ready_result(self, executor_id: None = None) -> tuple[list[str | None], pa.Table] | None: ...

    @overload
    def take_ready_result(self, executor_id: str) -> tuple[list[str | None], pa.Table, str] | None: ...

    def take_ready_result(self, executor_id: str | None = None) -> tuple[Any, ...] | None:
        if self.on_error == "raise":
            error_message = self._per_executor_errors.get(executor_id) if executor_id else self.error_message
            if error_message is not None:
                raise RuntimeError(f"vllm task failed: {error_message}")

        source_deque = (
            self._per_executor_deques.setdefault(executor_id, deque()) if executor_id else self.completed_tasks
        )
        try:
            item = source_deque.popleft()
        except IndexError:
            return None
        output, row, *extra = item
        self._notify_state_change()
        if executor_id:
            if len(extra) != 1 or not isinstance(extra[0], str) or not extra[0]:
                raise RuntimeError("vllm per-executor result must include a non-empty reservation_id")
            return [output], row, extra[0]
        return [output], row

    def finished_submitting(self) -> None:
        self._finished_submitting = True
        self._notify_state_change(force=True)

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
        self._notify_state_change(force=True)

    def release_executor(self, executor_id: str) -> bool:
        """Drop terminal state after an executor drained or aborted."""
        with self.task_count_lock:
            if self._per_executor_waiters.get(executor_id, 0) > 0:
                return False
            if (
                executor_id in self._per_executor_abort_wait_required
                and executor_id not in self._per_executor_terminal_wait_observed
            ):
                return False
            if self._per_executor_deques.get(executor_id):
                return False
            if self._per_executor_running_task_count.get(executor_id, 0) > 0:
                return False
            if self._per_executor_request_ids.get(executor_id):
                return False
            if self._per_executor_tasks.get(executor_id):
                return False
            self._per_executor_deques.pop(executor_id, None)
            self._per_executor_running_task_count.pop(executor_id, None)
            self._per_executor_request_ids.pop(executor_id, None)
            self._per_executor_tasks.pop(executor_id, None)
            self._per_executor_errors.pop(executor_id, None)
            self._per_executor_aborted.discard(executor_id)
            self._per_executor_waiters.pop(executor_id, None)
            self._per_executor_abort_wait_required.discard(executor_id)
            self._per_executor_terminal_wait_observed.discard(executor_id)
            self._per_executor_finished.discard(executor_id)
        self._notify_state_change(force=True)
        return True

    async def abort_executor(self, executor_id: str, wait_expected: bool = False) -> None:
        """Abort requests and discard state owned by one remote executor."""
        request_ids = set(self._per_executor_request_ids.get(executor_id, ()))
        tasks = set(self._per_executor_tasks.get(executor_id, ()))
        abort = getattr(getattr(self, "llm", None), "abort", None) or getattr(
            getattr(self, "llm", None), "abort_request", None
        )
        errors: list[BaseException] = []
        if abort is not None:
            for request_id in request_ids:
                try:
                    result = abort(request_id)
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    errors.append(exc)
        for task in tasks:
            try:
                task.cancel()
            except Exception as exc:
                errors.append(exc)
        async_tasks = [task for task in tasks if isinstance(task, asyncio.Future)]
        if async_tasks:
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            errors.extend(
                result
                for result in results
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError)
            )
        if errors:
            raise RuntimeError(
                f"vllm executor {executor_id} abort failed: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            ) from errors[0]
        with self.task_count_lock:
            remaining = self._per_executor_running_task_count.pop(executor_id, 0)
            if remaining:
                self.running_task_count = max(0, self.running_task_count - remaining)
            self._per_executor_deques.pop(executor_id, None)
            self._per_executor_request_ids.pop(executor_id, None)
            self._per_executor_tasks.pop(executor_id, None)
            self._per_executor_errors.pop(executor_id, None)
            self._per_executor_aborted.add(executor_id)
            self._per_executor_finished.discard(executor_id)
            if wait_expected:
                self._per_executor_abort_wait_required.add(executor_id)
                if self._per_executor_waiters.get(executor_id, 0) == 0:
                    self._per_executor_terminal_wait_observed.add(executor_id)
        self._notify_state_change(force=True)
        if wait_expected:
            deadline = time.monotonic() + _vllm_control_rpc_timeout_s()
            while True:
                with self.task_count_lock:
                    acknowledged = (
                        self._per_executor_waiters.get(executor_id, 0) == 0
                        and executor_id in self._per_executor_terminal_wait_observed
                    )
                if acknowledged:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"vllm executor {executor_id} abort waiter did not acknowledge termination")
                await asyncio.sleep(0.01)

    def all_tasks_finished(self) -> bool:
        with self.task_count_lock:
            return self._finished_submitting and self.running_task_count == 0 and len(self.completed_tasks) == 0

    def _wakeup_ready(self) -> bool:
        if self._shutdown_called or self.error_message is not None:
            return True
        if self.completed_tasks or any(bool(results) for results in self._per_executor_deques.values()):
            return True
        if self._per_executor_errors or self._per_executor_aborted:
            return True
        return self._finished_submitting and self.running_task_count == 0

    def _wait_for_result_blocking(self, executor_id: str | None = None) -> bool:
        with self._result_cv:
            self._result_cv.wait_for(lambda: any(self._wait_for_result_state(executor_id)))
            return self._wait_for_result_state(executor_id)[0]

    def _wait_for_result_state(self, executor_id: str | None) -> tuple[bool, bool]:
        source_deque = (
            self._per_executor_deques.setdefault(executor_id, deque()) if executor_id else self.completed_tasks
        )
        has_result = bool(source_deque)
        if executor_id:
            terminal = (
                executor_id in self._per_executor_errors
                or executor_id in self._per_executor_aborted
                or (
                    executor_id in self._per_executor_finished
                    and self._per_executor_running_task_count.get(executor_id, 0) == 0
                )
            )
        else:
            terminal = self.error_message is not None or (self._finished_submitting and self.running_task_count == 0)
        return has_result, terminal

    def wait_for_result(self, executor_id: str | None = None) -> bool:
        """Block until at least one result is available or all tasks are done."""
        return self._wait_for_result_blocking(executor_id)

    def shutdown(self) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self._finished_submitting = True
        self._notify_state_change(force=True)
        loop = getattr(self, "loop", None)
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)


class RayLocalVLLMExecutor(LocalVLLMExecutor):
    def _ensure_async_waiter_state(self) -> None:
        if not hasattr(self, "_async_waiter_lock"):
            self._async_waiter_lock = threading.Lock()
        if not hasattr(self, "_async_waiters"):
            self._async_waiters: dict[str, list[tuple[Any, asyncio.Event]]] = {}

    def _notify_state_change(self, *, force: bool = False) -> None:
        super()._notify_state_change(force=force)
        self._ensure_async_waiter_state()
        with self._async_waiter_lock:
            waiters = [waiter for executor_waiters in self._async_waiters.values() for waiter in executor_waiters]
        for loop, event in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # The waiter removes itself during normal loop shutdown.
                continue

    # Ray actor calls are awaitable, while the in-process executor exposes a blocking method.
    async def wait_for_result(self, executor_id: str | None = None) -> bool:  # type: ignore[override]
        if executor_id is None:
            return await asyncio.to_thread(self._wait_for_result_blocking, None)
        self._ensure_async_waiter_state()
        loop = asyncio.get_running_loop()
        state_changed = asyncio.Event()
        waiter = (loop, state_changed)
        with self._async_waiter_lock:
            self._async_waiters.setdefault(executor_id, []).append(waiter)
        with self.task_count_lock:
            self._per_executor_waiters[executor_id] = self._per_executor_waiters.get(executor_id, 0) + 1
        try:
            while True:
                has_result, terminal = self._wait_for_result_state(executor_id)
                if has_result or terminal:
                    return has_result
                state_changed.clear()
                has_result, terminal = self._wait_for_result_state(executor_id)
                if has_result or terminal:
                    continue
                await state_changed.wait()
        finally:
            with self._async_waiter_lock:
                executor_waiters = self._async_waiters.get(executor_id, [])
                try:
                    executor_waiters.remove(waiter)
                except ValueError:
                    pass
                if not executor_waiters:
                    self._async_waiters.pop(executor_id, None)
            with self.task_count_lock:
                remaining = self._per_executor_waiters.get(executor_id, 0) - 1
                if remaining > 0:
                    self._per_executor_waiters[executor_id] = remaining
                else:
                    self._per_executor_waiters.pop(executor_id, None)
                if executor_id in self._per_executor_aborted:
                    self._per_executor_terminal_wait_observed.add(executor_id)
            self._notify_state_change(force=True)


class RemoteVLLMExecutor(VLLMExecutor):
    # Marks the short interval after reserving an actor's wait slot and before
    # its Ray ObjectRef is installed. This sentinel is shared by all instances.
    _WAIT_REF_INSTALLING = object()

    def __init__(self, llm_actors: LLMActors, pool_name: str | None = None):
        # Keep the owner, rather than only its handles, so anonymous pools can
        # release the actors they created while borrowed named pools remain alive.
        self._actors_owner = llm_actors
        router_actor = llm_actors.router_actor
        if router_actor is None:
            raise ValueError("vllm remote executor requires a PrefixRouter actor")
        self.router_actor = router_actor
        self.llm_actors = list(llm_actors.llm_actors)
        if not self.llm_actors:
            raise ValueError("vllm remote executor requires at least one actor")

        # The ID is the isolation boundary for router reservations and all
        # per-executor state stored inside the shared model actors.
        self._executor_id = str(uuid.uuid4())
        self._pool_key = pool_name or f"_anon_{id(llm_actors)}"
        self._result_cv = threading.Condition(threading.RLock())
        self._ensure_wakeup_state()
        # Lifecycle operations may acquire the result condition, so callers
        # must preserve the lifecycle-lock -> result-condition lock order.
        self._lifecycle_lock = threading.RLock()
        self._reservation_lock = threading.Lock()
        # Reservation-mutating router RPCs remain serialized for exact local/
        # remote accounting, without holding the local state lock over Ray I/O.
        self._reservation_rpc_lock = threading.Lock()
        self._inflight_lock = threading.Lock()
        self._finished = False
        self._finished_submitting_flag = False
        self._shutdown_called = False
        self._shutdown_complete = False
        self._error_message: str | None = None
        # Successful queries remain successful when best-effort terminal cleanup
        # fails. Shutdown retries unfinished cleanup and surfaces persistent errors.
        self._terminal_cleanup_errors: list[str] = []
        self._terminal_cleanup_warning_emitted = False
        # Actor-indexed counters distinguish dispatched prompts, consumed
        # results, and the executor's local mirror of router reservations.
        self._submit_per_actor = [0] * len(self.llm_actors)
        self._results_per_actor = [0] * len(self.llm_actors)
        self._inflight_per_actor = [0] * len(self.llm_actors)
        self._result_buffer: deque[tuple[list[str | None], pa.Table]] = deque()
        # At most one wait ref is armed per actor. Completion callbacks only
        # enqueue refs here; the executor thread resolves and re-arms them.
        self._wait_refs_by_actor: list[Any | None] = [None] * len(self.llm_actors)
        self._ready_wait_refs: deque[Any] = deque()
        # Submit refs form the acknowledgement barrier before terminal RPCs;
        # their metadata is also sufficient to roll back a failed submission.
        self._submit_refs: dict[Any, tuple[int, int, str]] = {}
        self._ready_submit_refs: deque[Any] = deque()
        # Reservations track the remaining prompt count for exact release.
        self._reservations: OrderedDict[str, dict[str, int]] = OrderedDict()
        # Successful terminal RPCs are recorded so shutdown/error retries do
        # not repeat actor or router transitions that already completed.
        self._released_outstanding_inflight = False
        self._aborted_actor_indices: set[int] = set()
        self._released_actor_indices: set[int] = set()
        self._finished_actor_indices: set[int] = set()
        self._router_completion_reported = False

        try:
            resolve_object_refs_blocking(self.router_actor.report_start.remote(self._executor_id))
        except Exception:
            cancel_start = getattr(self.router_actor, "cancel_executor_start", None)
            if cancel_start is not None:
                try:
                    _resolve_vllm_control_ref(cancel_start.remote(self._executor_id))
                except Exception:
                    pass
            raise

    def _router_call(self, method_name: str, *args: Any, control_rpc: bool = False) -> Any:
        method = getattr(self.router_actor, method_name, None)
        if method is None:
            raise TypeError(f"vllm PrefixRouter does not implement {method_name}")
        resolve = _resolve_vllm_control_ref if control_rpc else resolve_object_refs_blocking
        return resolve(method.remote(*args))

    def _router_release(
        self,
        method_name: str,
        expected: int,
        *args: Any,
        operation_id: str,
        allow_reconcile: bool = False,
        control_rpc: bool = False,
    ) -> int:
        once_name = f"{method_name}_once"
        method = getattr(self.router_actor, once_name, None)
        resolve = _resolve_vllm_control_ref if control_rpc else resolve_object_refs_blocking
        if method is not None:
            call_args = (*args, operation_id)
            try:
                result = resolve(method.remote(*call_args))
            except Exception:
                result = resolve(method.remote(*call_args))
            if not isinstance(result, dict) or result.get("operation_id") != operation_id:
                raise RuntimeError(f"vllm PrefixRouter.{once_name} returned an invalid operation result")
            result = result.get("released")
        else:
            result = self._router_call(method_name, *args, control_rpc=control_rpc)
        if isinstance(result, bool) or not isinstance(result, int):
            raise RuntimeError(f"vllm PrefixRouter.{method_name} must return an integer release count")
        if (allow_reconcile and result < 0) or (not allow_reconcile and result != expected):
            raise RuntimeError(f"vllm PrefixRouter.{method_name} released {result} prompts; expected {expected}")
        return result

    def _actor_has_pending_result(self, actor_idx: int) -> bool:
        return self._results_per_actor[actor_idx] < self._submit_per_actor[actor_idx]

    def _queue_wait_ref_ready(self, ready_ref: Any) -> None:
        with self._result_cv:
            if self._shutdown_called or self._finished or ready_ref not in self._wait_refs_by_actor:
                return
            if ready_ref not in self._ready_wait_refs:
                self._ready_wait_refs.append(ready_ref)
        self._notify_state_change(force=True)

    def _ensure_wait_ref(self, actor_idx: int) -> None:
        with self._result_cv:
            if self._wait_refs_by_actor[actor_idx] is not None or not self._actor_has_pending_result(actor_idx):
                return
            self._wait_refs_by_actor[actor_idx] = self._WAIT_REF_INSTALLING
        try:
            wait_ref = self.llm_actors[actor_idx].wait_for_result.remote(self._executor_id)
        except Exception as exc:
            with self._result_cv:
                self._wait_refs_by_actor[actor_idx] = None
            self._record_error(exc)
            return
        with self._result_cv:
            if self._shutdown_called or self._finished:
                install = False
            else:
                self._wait_refs_by_actor[actor_idx] = wait_ref
                install = True
        if not install:
            self._cancel_refs([wait_ref])
            return
        try:
            wait_ref.future().add_done_callback(lambda _future, _ref=wait_ref: self._queue_wait_ref_ready(_ref))
        except Exception as exc:
            with self._result_cv:
                if self._wait_refs_by_actor[actor_idx] == wait_ref:
                    self._wait_refs_by_actor[actor_idx] = None
            self._record_error(TypeError(f"vllm wait ObjectRef does not support completion callbacks: {exc}"))

    def _ensure_remote_wait_refs(self) -> None:
        for actor_idx in range(len(self.llm_actors)):
            self._ensure_wait_ref(actor_idx)

    def register_wakeup_callback(self, callback: Callable[[], None]) -> bool:
        self._ensure_remote_wait_refs()
        return super().register_wakeup_callback(callback)

    def _actor_index_for_wait_ref(self, ready_ref: Any) -> int:
        for actor_idx, ref in enumerate(self._wait_refs_by_actor):
            if ref == ready_ref:
                return actor_idx
        raise RuntimeError("vllm remote wait returned an unknown actor ref")

    def _drain_queued_wait_refs(self) -> None:
        while True:
            with self._result_cv:
                if not self._ready_wait_refs:
                    return
                ready_ref = self._ready_wait_refs.popleft()
            try:
                actor_idx = self._actor_index_for_wait_ref(ready_ref)
                ready = resolve_object_refs_blocking(ready_ref)
                self._drain_ready_actor(actor_idx, bool(ready), ready_ref)
            except Exception as exc:
                self._record_error(exc)
                return

    def _drain_ready_actor(self, actor_idx: int, ready: bool, ready_ref: Any) -> None:
        if not ready:
            if (
                self._finished_submitting_flag
                and self._results_per_actor[actor_idx] >= self._submit_per_actor[actor_idx]
            ):
                with self._result_cv:
                    if self._wait_refs_by_actor[actor_idx] == ready_ref:
                        self._wait_refs_by_actor[actor_idx] = None
                return
            raise RuntimeError(
                "vllm actor finished without returning all submitted results: "
                f"actor_idx={actor_idx} submitted={self._submit_per_actor[actor_idx]} "
                f"received={self._results_per_actor[actor_idx]}"
            )
        result = resolve_object_refs_blocking(self.llm_actors[actor_idx].take_ready_result.remote(self._executor_id))
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError("vllm actor result must be a 3-item tuple including reservation_id")
        results_text, rows, reservation_id = result
        if not isinstance(reservation_id, str) or not reservation_id:
            raise RuntimeError("vllm actor result must include a non-empty reservation_id")
        count = len(results_text) if results_text else 0
        self._complete_reservation(reservation_id, count)
        with self._result_cv:
            if self._wait_refs_by_actor[actor_idx] == ready_ref:
                self._wait_refs_by_actor[actor_idx] = None
            self._results_per_actor[actor_idx] += count
            self._result_buffer.append((results_text, rows))
        # Actor waits are one-shot. Re-arm immediately after consuming a
        # result so an already-buffered successor cannot be stranded without
        # a completion callback while native backpressure is waiting.
        self._ensure_wait_ref(actor_idx)
        self._notify_state_change()

    def _complete_reservation(self, reservation_id: str | None, count: int) -> None:
        if reservation_id is None or count <= 0:
            return
        reservation_id = str(reservation_id)
        with self._reservation_rpc_lock:
            with self._reservation_lock:
                reservation = self._reservations.get(reservation_id)
                if reservation is None:
                    raise RuntimeError(f"vllm result references unknown reservation {reservation_id}")
                released = min(count, reservation["remaining"])
                remaining_before = reservation["remaining"]
                actor_idx = reservation["actor_idx"]
            self._router_release(
                "complete",
                released,
                reservation_id,
                released,
                operation_id=f"{self._executor_id}:complete:{reservation_id}:{remaining_before}:{released}",
            )
            with self._reservation_lock:
                current = self._reservations.get(reservation_id)
                if current is not reservation or current["remaining"] != remaining_before:
                    raise RuntimeError(f"vllm reservation {reservation_id} changed during completion")
                reservation["remaining"] -= released
                if reservation["remaining"] == 0:
                    self._reservations.pop(reservation_id, None)
            with self._inflight_lock:
                self._inflight_per_actor[actor_idx] = max(0, self._inflight_per_actor[actor_idx] - released)

    def _rollback_reservation(self, reservation_id: str, count: int) -> int:
        with self._reservation_rpc_lock:
            with self._reservation_lock:
                reservation = self._reservations.get(reservation_id)
                if reservation is None:
                    return 0
                released = min(count, reservation["remaining"])
                remaining_before = reservation["remaining"]
                actor_idx = reservation["actor_idx"]
            self._router_release(
                "rollback",
                released,
                reservation_id,
                released,
                operation_id=f"{self._executor_id}:rollback:{reservation_id}:{remaining_before}:{released}",
            )
            with self._reservation_lock:
                current = self._reservations.get(reservation_id)
                if current is not reservation or current["remaining"] != remaining_before:
                    raise RuntimeError(f"vllm reservation {reservation_id} changed during rollback")
                reservation["remaining"] -= released
                if reservation["remaining"] == 0:
                    self._reservations.pop(reservation_id, None)
            with self._inflight_lock:
                self._inflight_per_actor[actor_idx] = max(0, self._inflight_per_actor[actor_idx] - released)
        return released

    def _queue_submit_ref_ready(self, ready_ref: Any) -> None:
        with self._result_cv:
            if self._shutdown_called or self._finished or ready_ref not in self._submit_refs:
                return
            if ready_ref not in self._ready_submit_refs:
                self._ready_submit_refs.append(ready_ref)
        self._notify_state_change(force=True)

    def _track_submit_ref(self, submit_ref: Any, actor_idx: int, count: int, reservation_id: str) -> None:
        with self._result_cv:
            self._submit_refs[submit_ref] = (actor_idx, count, reservation_id)
        try:
            submit_ref.future().add_done_callback(lambda _future, _ref=submit_ref: self._queue_submit_ref_ready(_ref))
        except Exception as exc:
            self._submit_refs.pop(submit_ref, None)
            self._rollback_submitted_batch(actor_idx, count, reservation_id)
            self._record_error(TypeError(f"vllm submit ObjectRef does not support completion callbacks: {exc}"))

    def _resolve_submit_ref(self, submit_ref: Any, *, control_rpc: bool = False) -> bool:
        with self._result_cv:
            metadata = self._submit_refs.pop(submit_ref, None)
            try:
                self._ready_submit_refs.remove(submit_ref)
            except ValueError:
                pass
        if metadata is None:
            return True
        actor_idx, count, reservation_id = metadata
        resolve = _resolve_vllm_control_ref if control_rpc else resolve_object_refs_blocking
        try:
            resolve(submit_ref)
        except Exception as exc:
            try:
                self._rollback_submitted_batch(actor_idx, count, reservation_id)
            except Exception as rollback_error:
                exc = RuntimeError(f"{exc}; reservation rollback failed: {rollback_error}")
            self._record_error(exc)
            return False
        return True

    def _drain_queued_submit_refs(self) -> None:
        while True:
            with self._result_cv:
                if not self._ready_submit_refs:
                    return
                submit_ref = self._ready_submit_refs.popleft()
            if not self._resolve_submit_ref(submit_ref):
                return

    def _await_pending_submit_refs(self) -> None:
        """Establish actor-side submission order before sending completion."""
        while True:
            with self._result_cv:
                pending = list(self._submit_refs)
            if not pending:
                return
            for submit_ref in pending:
                if not self._resolve_submit_ref(submit_ref, control_rpc=True):
                    raise RuntimeError(f"vllm remote task failed: {self._error_message}")

    def _rollback_submitted_batch(self, actor_idx: int, count: int, reservation_id: str) -> None:
        released = self._rollback_reservation(reservation_id, count)
        with self._result_cv:
            self._submit_per_actor[actor_idx] = max(0, self._submit_per_actor[actor_idx] - released)
        self._notify_state_change(force=True)

    def _release_outstanding_inflight(self) -> None:
        with self._reservation_rpc_lock:
            with self._reservation_lock:
                if self._released_outstanding_inflight:
                    return
                expected = sum(reservation["remaining"] for reservation in self._reservations.values())
            self._router_release(
                "release_executor",
                expected,
                self._executor_id,
                operation_id=f"{self._executor_id}:release:{expected}",
                allow_reconcile=True,
                control_rpc=True,
            )
            with self._reservation_lock:
                self._reservations.clear()
                self._released_outstanding_inflight = True
            with self._inflight_lock:
                self._inflight_per_actor = [0] * len(self.llm_actors)

    def _cancel_refs(self, refs: list[Any]) -> None:
        if not refs:
            return
        try:
            import ray
        except Exception:
            return
        try:
            if not ray.is_initialized():
                return
        except Exception:
            return
        for ref in refs:
            try:
                ray.cancel(ref)
            except Exception:
                pass

    def _cancel_remote_refs(self) -> None:
        refs = list(self._submit_refs)
        refs.extend(ref for ref in self._wait_refs_by_actor if ref is not None and ref is not self._WAIT_REF_INSTALLING)
        self._cancel_refs(refs)

    def _abort_actor_state(self) -> None:
        errors: list[Exception] = []
        for actor_idx, actor in enumerate(self.llm_actors):
            if actor_idx in self._aborted_actor_indices:
                continue
            method = getattr(actor, "abort_executor", None)
            if method is None:
                errors.append(RuntimeError(f"vllm actor {actor_idx} does not implement abort_executor"))
                continue
            try:
                wait_ref = self._wait_refs_by_actor[actor_idx]
                wait_expected = wait_ref is not None and wait_ref is not self._WAIT_REF_INSTALLING
                _resolve_vllm_control_ref(method.remote(self._executor_id, wait_expected))
                self._aborted_actor_indices.add(actor_idx)
            except Exception as exc:
                errors.append(RuntimeError(f"actor {actor_idx}: {exc}"))
        if errors:
            raise RuntimeError("vllm remote abort failed: " + "; ".join(str(error) for error in errors))

    def _release_actor_state(self) -> None:
        errors: list[Exception] = []
        for actor_idx, actor in enumerate(self.llm_actors):
            if actor_idx in self._released_actor_indices:
                continue
            method = getattr(actor, "release_executor", None)
            if method is None:
                errors.append(RuntimeError(f"vllm actor {actor_idx} does not implement release_executor"))
                continue
            try:
                if _resolve_vllm_control_ref(method.remote(self._executor_id)) is not True:
                    raise RuntimeError("release_executor did not confirm release")
                self._released_actor_indices.add(actor_idx)
            except Exception as exc:
                errors.append(RuntimeError(f"actor {actor_idx}: {exc}"))
        if errors:
            raise RuntimeError("vllm remote actor-state release failed: " + "; ".join(str(error) for error in errors))

    def _report_router_completion(self) -> None:
        if self._router_completion_reported:
            return
        _resolve_vllm_control_ref(self.router_actor.report_completion.remote(self._executor_id))
        self._router_completion_reported = True

    def _defer_terminal_cleanup(self, errors: list[str]) -> None:
        if not errors:
            return
        for error in errors:
            if error not in self._terminal_cleanup_errors:
                self._terminal_cleanup_errors.append(error)
        if self._terminal_cleanup_warning_emitted:
            return
        self._terminal_cleanup_warning_emitted = True
        try:
            warnings.warn(
                "vllm query completed successfully but terminal cleanup remains incomplete and will be retried: "
                + "; ".join(errors),
                RuntimeWarning,
                stacklevel=3,
            )
        except Exception:
            # Warning filters may promote RuntimeWarning to an exception; cleanup
            # diagnostics must never turn an otherwise successful query into failure.
            pass

    def _record_error(self, exc: Exception) -> None:
        with self._lifecycle_lock:
            if self._finished and self._error_message is not None:
                return
            cleanup_errors: list[Exception] = []
            try:
                self._abort_actor_state()
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                self._report_router_completion()
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                self._release_outstanding_inflight()
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
            self._cancel_remote_refs()
            with self._result_cv:
                self._error_message = self._error_message or f"{type(exc).__name__}: {exc}"
                if cleanup_errors:
                    self._error_message += "; cleanup failed: " + "; ".join(str(error) for error in cleanup_errors)
                self._finished = True
                self._wait_refs_by_actor = [None] * len(self.llm_actors)
                self._submit_refs.clear()
                self._ready_wait_refs.clear()
                self._ready_submit_refs.clear()
            self._notify_state_change(force=True)

    def _mark_finished(self) -> None:
        with self._lifecycle_lock:
            if self._finished:
                return
            cleanup_errors: list[str] = []
            try:
                self._release_outstanding_inflight()
            except Exception as exc:
                cleanup_errors.append(f"router reservation release: {exc}")
            try:
                self._release_actor_state()
            except Exception as exc:
                cleanup_errors.append(f"actor state release: {exc}")
            self._defer_terminal_cleanup(cleanup_errors)
            with self._result_cv:
                self._finished = True
                self._wait_refs_by_actor = [None] * len(self.llm_actors)
                self._submit_refs.clear()
                self._ready_wait_refs.clear()
                self._ready_submit_refs.clear()
            self._notify_state_change(force=True)

    def _wakeup_ready(self) -> bool:
        return bool(
            self._result_buffer
            or self._ready_wait_refs
            or self._ready_submit_refs
            or self._error_message is not None
            or self._finished
            or self._shutdown_called
        )

    def submit(self, prefix: str | None, prompts: list[str], rows: pa.Table) -> None:
        with self._lifecycle_lock:
            if self._shutdown_called or self._finished or self._finished_submitting_flag:
                raise RuntimeError("vllm remote executor no longer accepts submissions")
            if self._error_message is not None:
                raise RuntimeError(f"vllm remote task failed: {self._error_message}")
            rows = _ensure_table(rows)
            count = len(prompts)
            if count != rows.num_rows:
                raise ValueError("Number of prompts and rows must match")
            if count == 0:
                return
            operation_id = f"{self._executor_id}:route:{uuid.uuid4()}"
            route_once = getattr(self.router_actor, "route_and_reserve_once", None)
            if route_once is not None:
                args = (prefix, count, self._executor_id, operation_id)
                try:
                    decision = resolve_object_refs_blocking(route_once.remote(*args))
                except Exception:
                    decision = resolve_object_refs_blocking(route_once.remote(*args))
            else:
                decision = self._router_call("route_and_reserve", prefix, count, self._executor_id)
            if not isinstance(decision, dict):
                raise RuntimeError("vllm PrefixRouter.route_and_reserve must return a dict")
            actor_idx = int(decision.get("actor_idx", -1))
            reservation_id = str(decision.get("reservation_id") or "")
            if actor_idx < 0 or actor_idx >= len(self.llm_actors) or not reservation_id:
                raise RuntimeError("vllm PrefixRouter returned an invalid reservation")
            with self._reservation_lock:
                self._reservations[reservation_id] = {"actor_idx": actor_idx, "remaining": count}
            with self._inflight_lock:
                self._inflight_per_actor[actor_idx] += count
            try:
                submit_ref = self.llm_actors[actor_idx].submit_async.remote(
                    prompts, rows, self._executor_id, reservation_id
                )
            except Exception:
                self._rollback_reservation(reservation_id, count)
                raise
            with self._result_cv:
                self._submit_per_actor[actor_idx] += count
            self._track_submit_ref(submit_ref, actor_idx, count, reservation_id)
            self._ensure_remote_wait_refs()
            self._notify_state_change(force=True)

    def take_ready_result(self) -> tuple[list[str | None], pa.Table] | None:
        self._drain_queued_submit_refs()
        self._drain_queued_wait_refs()
        if self._error_message is not None:
            raise RuntimeError(f"vllm remote task failed: {self._error_message}")
        try:
            result = self._result_buffer.popleft()
        except IndexError:
            return None
        self._notify_state_change()
        return result

    def finished_submitting(self) -> None:
        with self._lifecycle_lock:
            if self._finished_submitting_flag and self._router_completion_reported:
                return
            # Ray async actors do not guarantee actor-call execution order. A
            # terminal call can therefore overtake an earlier submit. Resolving
            # every submit acknowledgement is the causal barrier that guarantees
            # each actor has registered all work before it sees finished_executor.
            self._await_pending_submit_refs()
            errors: list[Exception] = []
            for actor_idx, actor in enumerate(self.llm_actors):
                if actor_idx in self._finished_actor_indices:
                    continue
                try:
                    _resolve_vllm_control_ref(actor.finished_executor.remote(self._executor_id))
                    self._finished_actor_indices.add(actor_idx)
                except Exception as exc:
                    errors.append(RuntimeError(f"actor {actor_idx}: {exc}"))
            if not self._router_completion_reported:
                try:
                    self._report_router_completion()
                except Exception as exc:
                    errors.append(RuntimeError(f"router: {exc}"))
            if errors:
                raise RuntimeError(
                    "vllm remote finished_submitting failed: " + "; ".join(str(error) for error in errors)
                )
            self._finished_submitting_flag = True
            self._notify_state_change(force=True)

    def all_tasks_finished(self) -> bool:
        self._drain_queued_submit_refs()
        self._drain_queued_wait_refs()
        if self._result_buffer:
            return False
        if self._error_message is not None:
            raise RuntimeError(f"vllm remote task failed: {self._error_message}")
        if not self._finished_submitting_flag:
            return False
        if sum(self._results_per_actor) >= sum(self._submit_per_actor):
            self._mark_finished()
            return True
        return False

    def wait_for_result(self) -> None:
        if self._error_message is not None:
            raise RuntimeError(f"vllm remote task failed: {self._error_message}")
        try:
            while not self._result_buffer and self._error_message is None and not self._finished:
                self._drain_queued_submit_refs()
                self._drain_queued_wait_refs()
                if self._result_buffer or self._error_message is not None or self._finished:
                    break
                self._ensure_remote_wait_refs()
                should_mark_finished = False
                with self._result_cv:
                    pending = any(ref is not None for ref in self._wait_refs_by_actor)
                    if not pending and self._finished_submitting_flag:
                        if sum(self._results_per_actor) >= sum(self._submit_per_actor):
                            should_mark_finished = True
                        else:
                            raise RuntimeError("vllm remote wait has no pending actor wait refs before completion")
                    if not should_mark_finished:
                        ready = self._result_cv.wait_for(
                            lambda: (
                                bool(self._result_buffer)
                                or bool(self._ready_submit_refs)
                                or bool(self._ready_wait_refs)
                                or self._error_message is not None
                                or self._finished
                            ),
                            timeout=configured_ray_get_timeout_s(),
                        )
                        if not ready:
                            raise RuntimeError("vllm remote wait exceeded query deadline")
                if should_mark_finished:
                    self._mark_finished()
                    break
        except Exception as exc:
            self._record_error(exc)
        if self._error_message is not None:
            raise RuntimeError(f"vllm remote task failed: {self._error_message}")

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            if self._shutdown_complete:
                return
            completed_successfully = (
                self._finished
                and self._error_message is None
                and self._finished_submitting_flag
                and sum(self._results_per_actor) >= sum(self._submit_per_actor)
            )
            self._shutdown_called = True
            errors: list[Exception] = []
            try:
                if not self._finished_submitting_flag:
                    self.finished_submitting()
            except Exception as exc:
                errors.append(exc)
            if sum(self._results_per_actor) < sum(self._submit_per_actor):
                try:
                    self._abort_actor_state()
                except Exception as exc:
                    errors.append(exc)
            try:
                self._report_router_completion()
            except Exception as exc:
                errors.append(exc)
            try:
                self._release_outstanding_inflight()
            except Exception as exc:
                errors.append(exc)
            try:
                self._release_actor_state()
            except Exception as exc:
                errors.append(exc)
            self._cancel_remote_refs()
            with self._result_cv:
                self._finished = True
                self._wait_refs_by_actor = [None] * len(self.llm_actors)
                self._submit_refs.clear()
                self._ready_wait_refs.clear()
                self._ready_submit_refs.clear()
            try:
                self._actors_owner.shutdown()
            except Exception as exc:
                errors.append(exc)
            self._shutdown_complete = not errors
            if errors and completed_successfully:
                self._defer_terminal_cleanup([f"shutdown cleanup: {error}" for error in errors])
            self._notify_state_change(force=True)
            if errors and not completed_successfully:
                raise RuntimeError("vllm remote shutdown incomplete: " + "; ".join(str(error) for error in errors))


class PrefixRouter:
    def __init__(
        self,
        llm_actors: list[Any],
        load_balance_threshold: int,
        max_recent_prefixes: int = 8,
    ):
        if not llm_actors:
            raise ValueError("vllm PrefixRouter requires at least one actor")
        self.llm_actors = llm_actors
        self.load_balance_threshold = int(load_balance_threshold)
        self.max_recent_prefixes = max(1, int(max_recent_prefixes))
        # This is the authoritative cross-executor reserved-prompt count used
        # for routing; worker-local mirrors must not influence actor selection.
        self.inflight = [0] * len(llm_actors)
        # Prefix affinity is a bounded LRU map so sticky routing cannot retain
        # an unbounded set of prompt prefixes.
        self._prefix_affinity: OrderedDict[str, int] = OrderedDict()
        # Each reservation belongs to one executor and actor, and records the
        # count still requiring completion or rollback.
        self._reservations: dict[str, dict[str, Any]] = {}
        self._active_executors: set[str] = set()
        # Completed IDs are retained as bounded tombstones, making duplicate
        # completion idempotent while rejecting accidental ID reuse.
        self._completed_executors: OrderedDict[str, None] = OrderedDict()
        # Operation-ID caches replay the original result for retried route and
        # release RPCs, providing exact-once state mutation over at-least-once calls.
        self._route_operations: OrderedDict[str, tuple[tuple[Any, ...], dict[str, Any]]] = OrderedDict()
        self._release_operations: OrderedDict[str, tuple[tuple[Any, ...], int]] = OrderedDict()
        # All routing, reservation, lifecycle, and idempotency state above is
        # mutated under one lock so decisions observe one coherent snapshot.
        self._lock = threading.Lock()

    @staticmethod
    def _trim(mapping: OrderedDict[Any, Any], limit: int) -> None:
        while len(mapping) > limit:
            mapping.popitem(last=False)

    def report_start(self, executor_id: str) -> bool:
        executor_id = str(executor_id)
        with self._lock:
            if executor_id in self._active_executors:
                return False
            if executor_id in self._completed_executors:
                raise RuntimeError("vllm router executor was already completed")
            self._active_executors.add(executor_id)
            return True

    def cancel_executor_start(self, executor_id: str) -> bool:
        with self._lock:
            executor_id = str(executor_id)
            existed = executor_id in self._active_executors
            self._active_executors.discard(executor_id)
            return existed

    def report_completion(self, executor_id: str) -> bool:
        executor_id = str(executor_id)
        with self._lock:
            if executor_id in self._completed_executors:
                self._completed_executors.move_to_end(executor_id)
                return False
            if executor_id not in self._active_executors:
                raise RuntimeError("vllm router received completion without a matching start")
            self._active_executors.remove(executor_id)
            self._completed_executors[executor_id] = None
            self._trim(self._completed_executors, 4096)
            return True

    def _route_and_reserve_locked(self, prefix: str | None, prompt_count: Any, executor_id: str) -> dict[str, Any]:
        executor_id = str(executor_id)
        if executor_id not in self._active_executors:
            raise RuntimeError("vllm router executor is not active")
        if isinstance(prompt_count, bool) or not isinstance(prompt_count, Integral) or prompt_count <= 0:
            raise ValueError("vllm prompt_count must be a positive integer")
        prefix = None if prefix is None else str(prefix)
        min_actor = min(range(len(self.inflight)), key=self.inflight.__getitem__)
        if prefix is None:
            actor_idx = min_actor
            reason = "no_prefix"
        elif prefix not in self._prefix_affinity:
            actor_idx = min_actor
            reason = "initial"
            self._prefix_affinity[prefix] = actor_idx
        else:
            actor_idx = self._prefix_affinity[prefix]
            if self.inflight[actor_idx] <= self.inflight[min_actor] + self.load_balance_threshold:
                reason = "sticky"
            else:
                actor_idx = min_actor
                reason = "load_balance"
                self._prefix_affinity[prefix] = actor_idx
            self._prefix_affinity.move_to_end(prefix)
        self._trim(self._prefix_affinity, self.max_recent_prefixes)
        reservation_id = str(uuid.uuid4())
        count = int(prompt_count)
        self.inflight[actor_idx] += count
        self._reservations[reservation_id] = {
            "actor_idx": actor_idx,
            "remaining": count,
            "executor_id": executor_id,
        }
        return {
            "reservation_id": reservation_id,
            "actor_idx": actor_idx,
            "route_reason": reason,
            "prompt_count": count,
        }

    def route_and_reserve(self, prefix: str | None, prompt_count: int, executor_id: str) -> dict[str, Any]:
        with self._lock:
            return self._route_and_reserve_locked(prefix, prompt_count, executor_id)

    def route_and_reserve_once(
        self,
        prefix: str | None,
        prompt_count: int,
        executor_id: str,
        operation_id: str,
    ) -> dict[str, Any]:
        token = str(operation_id)
        if not token:
            raise ValueError("vllm route operation_id must not be empty")
        signature = (None if prefix is None else str(prefix), int(prompt_count), str(executor_id))
        with self._lock:
            previous = self._route_operations.get(token)
            if previous is not None:
                if previous[0] != signature:
                    raise RuntimeError("vllm route operation_id was reused with different arguments")
                self._route_operations.move_to_end(token)
                return {**previous[1], "operation_id": token, "replayed": True}
            decision = self._route_and_reserve_locked(prefix, prompt_count, executor_id)
            self._route_operations[token] = (signature, dict(decision))
            self._trim(self._route_operations, 8192)
            return {**decision, "operation_id": token, "replayed": False}

    def _release_locked(self, reservation_id: str, count: int | None) -> int:
        reservation = self._reservations.get(str(reservation_id))
        if reservation is None:
            return 0
        remaining = int(reservation["remaining"])
        release_count = remaining if count is None else min(remaining, max(0, int(count)))
        if release_count <= 0:
            return 0
        actor_idx = int(reservation["actor_idx"])
        reservation["remaining"] = remaining - release_count
        self.inflight[actor_idx] = max(0, self.inflight[actor_idx] - release_count)
        if reservation["remaining"] == 0:
            self._reservations.pop(str(reservation_id), None)
        return release_count

    def complete(self, reservation_id: str, count: int) -> int:
        with self._lock:
            return self._release_locked(reservation_id, count)

    def rollback(self, reservation_id: str, count: int | None = None) -> int:
        with self._lock:
            return self._release_locked(reservation_id, count)

    def _release_once(
        self,
        operation_id: str,
        signature: tuple[Any, ...],
        release: Callable[[], int],
    ) -> dict[str, Any]:
        token = str(operation_id)
        if not token:
            raise ValueError("vllm terminal operation_id must not be empty")
        previous = self._release_operations.get(token)
        if previous is not None:
            if previous[0] != signature:
                raise RuntimeError("vllm terminal operation_id was reused with different arguments")
            self._release_operations.move_to_end(token)
            return {"operation_id": token, "released": previous[1], "replayed": True}
        released = release()
        self._release_operations[token] = (signature, released)
        self._trim(self._release_operations, 8192)
        return {"operation_id": token, "released": released, "replayed": False}

    def complete_once(self, reservation_id: str, count: int, operation_id: str) -> dict[str, Any]:
        with self._lock:
            signature = ("complete", str(reservation_id), int(count))
            return self._release_once(operation_id, signature, lambda: self._release_locked(reservation_id, count))

    def rollback_once(self, reservation_id: str, count: int, operation_id: str) -> dict[str, Any]:
        with self._lock:
            signature = ("rollback", str(reservation_id), int(count))
            return self._release_once(operation_id, signature, lambda: self._release_locked(reservation_id, count))

    def release_executor(self, executor_id: str) -> int:
        with self._lock:
            executor_id = str(executor_id)
            reservation_ids = [
                reservation_id
                for reservation_id, reservation in self._reservations.items()
                if reservation["executor_id"] == executor_id
            ]
            return sum(self._release_locked(reservation_id, None) for reservation_id in reservation_ids)

    def release_executor_once(self, executor_id: str, operation_id: str) -> dict[str, Any]:
        with self._lock:
            executor_id = str(executor_id)

            def release() -> int:
                reservation_ids = [
                    reservation_id
                    for reservation_id, reservation in self._reservations.items()
                    if reservation["executor_id"] == executor_id
                ]
                return sum(self._release_locked(reservation_id, None) for reservation_id in reservation_ids)

            return self._release_once(operation_id, ("release_executor", executor_id), release)


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

        self.owned = True
        self._shutdown_complete = False
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

    @staticmethod
    def _kill_handles(
        ray_module: Any,
        llm_actors: list[Any],
        router_actor: Any | None,
    ) -> tuple[list[Any], Any | None, list[str]]:
        remaining_actors: list[Any] = []
        remaining_router = router_actor
        errors: list[str] = []
        handles = [("actor", actor) for actor in llm_actors]
        if router_actor is not None:
            handles.append(("router", router_actor))
        for kind, handle in handles:
            try:
                try:
                    ray_module.kill(handle, no_restart=True)
                except TypeError:
                    ray_module.kill(handle)
            except Exception as exc:
                errors.append(f"{kind} kill failed: {type(exc).__name__}: {exc}")
                if kind == "actor":
                    remaining_actors.append(handle)
                else:
                    remaining_router = handle
            else:
                if kind == "router":
                    remaining_router = None
        return remaining_actors, remaining_router, errors

    @classmethod
    def _from_handles(cls, llm_actors: list[Any], router_actor: Any) -> LLMActors:
        instance = cls.__new__(cls)
        instance.llm_actors = llm_actors
        instance.router_actor = router_actor
        instance.owned = False
        instance._shutdown_complete = False
        return instance

    def shutdown(self) -> None:
        """Release anonymous actors owned by this executor exactly once."""
        if self._shutdown_complete:
            return
        if not self.owned:
            self._shutdown_complete = True
            return
        import ray

        remaining_actors, remaining_router, errors = self._kill_handles(ray, self.llm_actors, self.router_actor)
        self.llm_actors = remaining_actors
        self.router_actor = remaining_router
        self._shutdown_complete = not remaining_actors and remaining_router is None
        if errors:
            raise RuntimeError("vLLM pool shutdown incomplete: " + "; ".join(errors))

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


def _boolean_option(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"vllm {name} must be a boolean")
    return value


def _fractional_gpu_option(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
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
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
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
    for name in (
        "do_prefix_routing",
        "use_ray",
        "use_threading",
        "require_ray_worker",
        "ray_worker_only",
        "_force_background_thread",
    ):
        if name in merged:
            merged[name] = _boolean_option(name, merged[name])
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
    require_ray_worker = opts.get("require_ray_worker", False) or opts.get("ray_worker_only", False)
    if require_ray_worker and not _is_ray_worker():
        raise RuntimeError("vllm executor must be constructed on a Ray worker when require_ray_worker is set")

    # `use_ray` is the only routing switch. Pool/address metadata only
    # configures Ray-backed execution after it has been explicitly selected.
    if opts["use_ray"]:
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
        try:
            return RemoteVLLMExecutor(llm_actors, pool_name=pool_name)
        except Exception:
            llm_actors.shutdown()
            raise

    return LocalVLLMExecutor(
        model,
        engine_args,
        generate_args,
        on_error=on_error,
        use_threading=opts["use_threading"],
        engine_init_timeout_s=opts["engine_init_timeout_s"],
        force_background_thread=opts.get("_force_background_thread", False),
    )
