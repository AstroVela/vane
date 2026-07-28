# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Centralized, typed access to public ``VANE_*`` configuration variables.

Usage::

    from vane._env import env

    # Read (always live from os.environ)
    print(env.runner)  # "ray" or "local"
    print(env.ray_scan_task_size_grouping)  # True

    # Write (sets os.environ immediately)
    env.runner = "ray"
    env.ray_scan_task_size_grouping = False

    # Bulk snapshot
    d = env.as_dict()  # {"runner": "ray", ...}

Each variable is declared as a class-level ``_Var`` descriptor so that
attribute access on the singleton *env* object reads/writes ``os.environ``
in real time — no stale caches.
"""

from __future__ import annotations

import os
from typing import Any, Generic, TypeVar, overload

T = TypeVar("T")
_PUBLIC_RUNNER_VALUES = frozenset({"local", "ray"})
_PUBLIC_RUNNER_ERROR = "runner must be 'local' or 'ray'"

# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


class _Var(Generic[T]):
    """Descriptor that maps a Python attribute to a ``VANE_*`` env var."""

    def __init__(
        self,
        env_name: str,
        type_: type[T],
        default: T,
        doc: str = "",
    ) -> None:
        self.env_name = env_name
        self.type_ = type_
        self.default = default
        self.__doc__ = doc

    # -- read ----------------------------------------------------------------

    @overload
    def __get__(self, obj: None, _objtype: type) -> _Var[T]: ...
    @overload
    def __get__(self, obj: Any, _objtype: type) -> T: ...

    def __get__(self, obj: Any, _objtype: type = None) -> Any:
        if obj is None:
            return self  # class-level access returns the descriptor itself
        raw = os.environ.get(self.env_name)
        if raw is None or raw == "":
            return self.default
        return self._parse(raw)

    # -- write ---------------------------------------------------------------

    def __set__(self, obj: Any, value: T) -> None:
        if value is None:
            os.environ.pop(self.env_name, None)
        else:
            os.environ[self.env_name] = str(value)

    # -- delete --------------------------------------------------------------

    def __delete__(self, obj: Any) -> None:
        os.environ.pop(self.env_name, None)

    # -- parsing -------------------------------------------------------------

    def _parse(self, raw: str) -> T:
        if self.type_ is bool:
            return raw.lower() in ("1", "true", "yes", "on")  # type: ignore[return-value]
        if self.type_ is int:
            return int(raw)  # type: ignore[return-value]
        if self.type_ is float:
            return float(raw)  # type: ignore[return-value]
        return raw  # type: ignore[return-value]


class _RunnerVar(_Var[str]):
    """Runner variable with the same normalization as the C++ resolver."""

    def _parse(self, raw: str) -> str:
        return raw.strip().lower() or self.default

    def __set__(self, obj: Any, value: str) -> None:
        normalized = self._parse(str(value))
        if normalized not in _PUBLIC_RUNNER_VALUES:
            raise ValueError(_PUBLIC_RUNNER_ERROR)
        super().__set__(obj, normalized)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class EnvRegistry:
    """Typed, live-read/write access to ``VANE_*`` environment variables.

    Attributes are grouped by subsystem. This registry is the stable public
    programmatic configuration surface; internal/debug-only environment
    variables may still be read directly by their subsystem.
    """

    # -- Runner selection ---------------------------------------------------

    runner: str = _RunnerVar(
        "VANE_RUNNER",
        str,
        "ray",
        "Execution backend. Unset, empty, or 'ray' = Ray distributed; 'local' = Vane local FTE runner.",
    )

    # -- Ray runner ---------------------------------------------------------

    ray_scan_task_size_grouping: bool = _Var(
        "VANE_RAY_SCAN_TASK_SIZE_GROUPING",
        bool,
        True,
        "Enable size-based scan task grouping (merges small files into 96-384MB tasks). "
        "Set to false/0 to disable and get one task per file.",
    )
    ray_max_task_backlog: int = _Var(
        "VANE_RAY_MAX_TASK_BACKLOG",
        int,
        0,
        "Max pending tasks before back-pressure. 0 = unlimited.",
    )
    ray_scan_task_open_cost_bytes: int = _Var(
        "VANE_RAY_SCAN_TASK_OPEN_COST_BYTES",
        int,
        4 * 1024 * 1024,
        "Virtual per-file I/O cost (bytes) for Spark-style partition sizing. Default 4 MB.",
    )
    ray_scan_task_min_partition_num: int = _Var(
        "VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM",
        int,
        0,
        "Minimum number of scan partitions. 0 = use worker_slots.",
    )
    ray_init_sql: str = _Var(
        "VANE_RAY_INIT_SQL",
        str,
        "",
        "SQL to execute on each Ray worker at startup.",
    )

    # -- Fault-tolerant execution ------------------------------------------

    # -- UDF ----------------------------------------------------------

    udf_parallel: bool = _Var(
        "VANE_UDF_PARALLEL",
        bool,
        False,
        "Enable parallel UDF execution.",
    )
    udf_arrow_fastpath: bool = _Var(
        "VANE_UDF_ARROW_FASTPATH",
        bool,
        True,
        "Use Arrow zero-copy fast path for UDF I/O.",
    )

    # -- Local exchange -----------------------------------------------------

    local_exchange_buffer: str = _Var(
        "VANE_LOCAL_EXCHANGE_BUFFER",
        str,
        "32MB",
        "Buffer size for local exchange between pipeline stages.",
    )

    # -- helpers ------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """Return a snapshot of every registered variable's current value."""
        out: dict[str, Any] = {}
        for name in dir(type(self)):
            descriptor = getattr(type(self), name, None)
            if isinstance(descriptor, _Var):
                out[name] = getattr(self, name)
        return out

    def set(self, **kw: Any) -> None:
        """Bulk-set variables by attribute name.

        Example::

            env.set(runner="ray", ray_scan_task_size_grouping=False)
        """
        for key, value in kw.items():
            descriptor = getattr(type(self), key, None)
            if not isinstance(descriptor, _Var):
                raise AttributeError(
                    f"Unknown env variable attribute: {key!r}. Use env.as_dict().keys() to see available names."
                )
            setattr(self, key, value)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self.as_dict().items()))
        return f"EnvRegistry({items})"


# Module-level singleton — import as ``from vane._env import env``.
env = EnvRegistry()
