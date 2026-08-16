# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Engine-agnostic LLM executor contract and shared local-executor machinery.

``LLMExecutor`` is the base class implemented by each backend (vLLM,
SGLang, ...): the submit/result lifecycle plus the one-shot wakeup protocol
used by DuckDB's native scheduler. ``LocalEngineExecutor`` extends it with
all the engine-agnostic machinery (background event loop, task counting,
per-executor routing, abort/release) so a backend only supplies four hooks:

* ``_engine_name`` — short backend name for error messages;
* ``_materialize_sampling_params`` — build the backend's sampling params;
* ``_create_engine`` — construct the backend engine object on ``self.llm``;
* ``_run_generate`` — run one prompt and return its output text.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Mapping
from typing import Any, Callable

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

from vane._native import __standard_vector_size__ as DUCKDB_STANDARD_VECTOR_SIZE
from vane.ai.functions import _log_substituted_failure
from vane.ai.provider import _safe_provider_execution_error, _SafeProviderError


class LLMExecutor(ABC):
    """Common execution contract shared by every inference backend executor."""

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
            raise TypeError("llm wakeup callback must be callable")
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
        raise TimeoutError("query deadline expired before LLM wait")
    return remaining


def _bounded_query_timeout_s(timeout_s: float | None) -> float | None:
    deadline_remaining = _query_deadline_remaining_s()
    if timeout_s is None:
        return deadline_remaining
    timeout_s = max(0.0, float(timeout_s))
    if deadline_remaining is None:
        return timeout_s
    return min(timeout_s, deadline_remaining)


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


class LocalEngineExecutor(LLMExecutor):
    """Shared machinery for local and Ray-backed async engine executors.

    Subclasses supply the backend hooks listed in the module docstring; every
    submission, batching, backpressure, wakeup, and distributed-routing detail
    lives here.
    """

    _engine_name: str = "llm"

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
        self.model = model
        self.engine_args = dict(engine_args)
        self.llm: Any = None
        self.engine_ready = threading.Event()
        self.engine_error_message: str | None = None
        self.engine_init_timeout_s = engine_init_timeout_s

        self.sampling_params = self._materialize_sampling_params(generate_args)
        self.generate_args = generate_args

        self.counter = 0
        self.counter_lock = threading.Lock()

        self.running_task_count = 0
        self.task_count_lock = threading.Lock()

        self.completed_tasks: deque[tuple[str | None, pa.Table]] = deque()
        self.error_message: str | None = None
        self.error_lock = threading.Lock()
        self.on_error = on_error

        self._result_cv = threading.Condition(threading.RLock())
        self._ensure_wakeup_state()

        self._ray_actor_mode = self._detect_ray_actor() and not force_background_thread

        self.use_threading = use_threading
        if self._ray_actor_mode:
            self._init_engine_sync()
        elif self.use_threading:
            self.loop_ready = threading.Event()
            self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self.loop_thread.start()
            if not self.loop_ready.wait(_bounded_query_timeout_s(self.engine_init_timeout_s)):
                raise RuntimeError(
                    f"{self._engine_name} event loop did not start before {self._engine_init_deadline_description()}"
                )

        self._finished_submitting = False
        self._shutdown_called = False
        self._per_executor_deques: dict[str, deque[tuple[Any, ...]]] = {}
        self._per_executor_running_task_count: dict[str, int] = {}
        self._per_executor_finished: set[str] = set()
        self._per_executor_request_ids: dict[str, set[str]] = {}
        self._per_executor_tasks: dict[str, set[Any]] = {}
        self._per_executor_errors: dict[str, str] = {}
        self._per_executor_aborted: set[str] = set()
        self._per_executor_waiters: dict[str, int] = {}
        self._per_executor_wait_tokens_observed: dict[str, str] = {}
        self._per_executor_abort_wait_tokens: dict[str, str] = {}
        self._async_waiter_lock = threading.Lock()
        self._async_waiters: dict[str, list[tuple[Any, asyncio.Event]]] = {}

    # ---- backend hooks ----------------------------------------------------

    @abstractmethod
    def _materialize_sampling_params(self, generate_args: dict[str, Any]) -> Any:
        """Pop `sampling_params` from generate_args and build backend params."""
        pass

    @abstractmethod
    def _create_engine(self) -> None:
        """Construct the backend engine and assign it to `self.llm`."""
        pass

    @abstractmethod
    async def _run_generate(self, prompt: str, request_id: str) -> str:
        """Run one prompt and return the generated text."""
        pass

    # ---- engine lifecycle ---------------------------------------------------

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
        try:
            self._create_engine()
        except Exception as exc:
            error_message = str(_safe_provider_execution_error(self._engine_name, self.model, "engine initialization", exc))
            if self.on_error == "raise":
                with self.error_lock:
                    if self.error_message is None:
                        self.error_message = error_message
            self.engine_error_message = error_message
        finally:
            self.engine_ready.set()

    async def _init_engine(self) -> None:
        try:
            self._create_engine()
        except Exception as exc:
            error_message = str(_safe_provider_execution_error(self._engine_name, self.model, "engine initialization", exc))
            if self.on_error == "raise":
                with self.error_lock:
                    if self.error_message is None:
                        self.error_message = error_message
            self.engine_error_message = error_message
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
                raise _SafeProviderError(f"{self._engine_name} engine init failed: {self.engine_error_message}")
            if self.llm is None:
                raise _SafeProviderError(f"{self._engine_name} engine not initialized")
            with self.counter_lock:
                request_id = str(self.counter)
                self.counter += 1
            if executor_id:
                with self.task_count_lock:
                    self._per_executor_request_ids.setdefault(executor_id, set()).add(request_id)

            output_text = await self._run_generate(prompt, request_id)

            if executor_id:
                self._per_executor_deques.setdefault(executor_id, deque()).append((output_text, row, reservation_id))
            else:
                self.completed_tasks.append((output_text, row))
            self._notify_state_change()
        except Exception as exc:
            if self.on_error == "raise":
                error_message = str(_safe_provider_execution_error(self._engine_name, self.model, "generation", exc))
                if executor_id:
                    with self.task_count_lock:
                        self._per_executor_errors.setdefault(executor_id, error_message)
                else:
                    with self.error_lock:
                        if self.error_message is None:
                            self.error_message = error_message
                self._notify_state_change(force=True)
            else:
                self._log_null_substitution(exc)
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

    def _log_null_substitution(self, exc: Exception) -> None:
        _log_substituted_failure(exc, on_error="raise" if self.on_error == "raise" else "ignore")

    def _append_error_rows(
        self,
        rows: pa.Table,
        executor_id: str | None = None,
        reservation_id: str | None = None,
    ) -> None:
        rows = _ensure_table(rows)
        self._log_null_substitution(_SafeProviderError(f"{self._engine_name} engine init failed: {self.engine_error_message}"))
        for i in range(rows.num_rows):
            row = rows.slice(i, 1)
            if executor_id:
                self._per_executor_deques.setdefault(executor_id, deque()).append((None, row, reservation_id))
            else:
                self.completed_tasks.append((None, row))
        self._notify_state_change()

    # ---- native bridge lifecycle -------------------------------------------

    def submit(self, _prefix: str | None, prompts: list[str], rows: pa.Table) -> None:
        rows = _ensure_table(rows)
        if len(prompts) != rows.num_rows:
            raise ValueError("Number of prompts and rows must match")

        if not self.use_threading:
            raise ValueError("Synchronous mode not supported when use_threading is False")

        self._wait_for_engine_ready_blocking()
        if self.engine_error_message is not None:
            if self.on_error == "raise":
                raise RuntimeError(f"{self._engine_name} engine init failed: {self.engine_error_message}")
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

        if executor_id:
            with self.task_count_lock:
                if (
                    executor_id in self._per_executor_finished
                    or executor_id in self._per_executor_aborted
                    or executor_id in self._per_executor_errors
                ):
                    raise RuntimeError(f"{self._engine_name} executor {executor_id} is already finished")
                if executor_id not in self._per_executor_deques:
                    self._per_executor_deques[executor_id] = deque()
                    self._per_executor_request_ids[executor_id] = set()
                    self._per_executor_tasks[executor_id] = set()

        if self._ray_actor_mode:
            if self.engine_error_message is not None:
                if self.on_error == "raise":
                    raise RuntimeError(f"{self._engine_name} engine init failed: {self.engine_error_message}")
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
                asyncio_task = asyncio.create_task(self._generate(prompt, row, executor_id, reservation_id))
                self._track_executor_task(executor_id, asyncio_task)
        else:
            if not self.engine_ready.is_set():
                await self._wait_for_engine_ready_async()
            if executor_id:
                with self.task_count_lock:
                    if (
                        executor_id in self._per_executor_finished
                        or executor_id in self._per_executor_aborted
                        or executor_id in self._per_executor_errors
                    ):
                        raise RuntimeError(f"{self._engine_name} executor {executor_id} is already finished")
            if self.engine_error_message is not None:
                if self.on_error == "raise":
                    raise RuntimeError(f"{self._engine_name} engine init failed: {self.engine_error_message}")
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
                thread_future = asyncio.run_coroutine_threadsafe(
                    self._generate(prompt, row, executor_id, reservation_id), self.loop
                )
                self._track_executor_task(executor_id, thread_future)
        self._notify_state_change(force=True)

    def _track_executor_task(self, executor_id: str | None, task: Any) -> None:
        if not executor_id:
            return
        tasks = self._per_executor_tasks.setdefault(executor_id, set())
        tasks.add(task)

        def discard(done: Any) -> None:
            tasks.discard(done)

        task.add_done_callback(discard)

    def _raise_if_task_failed(self, executor_id: str | None = None) -> None:
        if self.on_error != "raise":
            return
        error_message = self._per_executor_errors.get(executor_id) if executor_id else self.error_message
        if error_message is not None:
            raise RuntimeError(f"{self._engine_name} task failed: {error_message}")

    def take_ready_result(self, executor_id: str | None = None) -> tuple[Any, ...] | None:
        self._raise_if_task_failed(executor_id)

        source_deque = (
            self._per_executor_deques.setdefault(executor_id, deque()) if executor_id else self.completed_tasks
        )
        if not source_deque:
            return None
        if not executor_id:
            output, row = source_deque.popleft()
            self._notify_state_change()
            return [output], row

        outputs: list[str | None] = []
        row_tables: list[pa.Table] = []
        reservation_counts: OrderedDict[str, int] = OrderedDict()
        while source_deque and len(outputs) < DUCKDB_STANDARD_VECTOR_SIZE:
            output, row, *extra = source_deque.popleft()
            if len(extra) != 1 or not isinstance(extra[0], str) or not extra[0]:
                raise RuntimeError(f"{self._engine_name} per-executor result must include a non-empty reservation_id")
            row = _ensure_table(row)
            if row.num_rows != 1:
                raise RuntimeError(f"{self._engine_name} per-executor result row must contain exactly one row")
            reservation_id = extra[0]
            outputs.append(output)
            row_tables.append(row)
            reservation_counts[reservation_id] = reservation_counts.get(reservation_id, 0) + 1

        self._notify_state_change()
        return outputs, _concat_tables(row_tables), list(reservation_counts.items())

    def finished_submitting(self) -> None:
        self._finished_submitting = True
        self._notify_state_change(force=True)

    def _engine_ready_wait_timeout_s(self) -> float | None:
        return _bounded_query_timeout_s(self.engine_init_timeout_s)

    def _engine_init_deadline_message(self) -> str:
        return f"{self._engine_name} engine init did not finish before {self._engine_init_deadline_description()}"

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
        with self.task_count_lock:
            if self._per_executor_waiters.get(executor_id, 0) > 0:
                return False
            expected_wait_token = self._per_executor_abort_wait_tokens.get(executor_id)
            if (
                expected_wait_token is not None
                and self._per_executor_wait_tokens_observed.get(executor_id) != expected_wait_token
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
            self._per_executor_wait_tokens_observed.pop(executor_id, None)
            self._per_executor_abort_wait_tokens.pop(executor_id, None)
            self._per_executor_finished.discard(executor_id)
        self._notify_state_change(force=True)
        return True

    def _control_rpc_timeout_s(self) -> float:
        default_timeout_s = 30.0
        raw_value = os.getenv(f"VANE_{self._engine_name.upper()}_CONTROL_RPC_TIMEOUT_S")
        if raw_value is None:
            return default_timeout_s
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return default_timeout_s
        if not math.isfinite(value) or value <= 0.0:
            return default_timeout_s
        return value

    async def abort_executor(self, executor_id: str, wait_token: str | None = None) -> None:
        if wait_token is not None and (not isinstance(wait_token, str) or not wait_token):
            raise ValueError(f"{self._engine_name} abort wait_token must be a non-empty string")
        with self.task_count_lock:
            self._per_executor_aborted.add(executor_id)
            self._per_executor_finished.discard(executor_id)
            if wait_token is not None:
                self._per_executor_abort_wait_tokens[executor_id] = wait_token
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
                f"{self._engine_name} executor {executor_id} abort failed: "
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
        self._notify_state_change(force=True)
        if wait_token is not None:
            deadline = time.monotonic() + self._control_rpc_timeout_s()
            while True:
                with self.task_count_lock:
                    acknowledged = (
                        self._per_executor_waiters.get(executor_id, 0) == 0
                        and self._per_executor_wait_tokens_observed.get(executor_id) == wait_token
                    )
                if acknowledged:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"{self._engine_name} executor {executor_id} abort waiter {wait_token} did not acknowledge termination"
                    )
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
