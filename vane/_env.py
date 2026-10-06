# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Centralized, typed access to public ``VANE_*`` configuration variables.

Usage::

    from vane._env import env

    env.udf_parallel = True
    settings = env.as_dict()

Execution configuration belongs to ``vane.connect(backend=..., resources=...)``
and per-query options.

Each variable is declared as a class-level ``_Var`` descriptor so that
attribute access on the singleton *env* object reads/writes ``os.environ``
in real time — no stale caches.
"""

from __future__ import annotations

import os
from typing import Any, Generic, TypeVar, overload

T = TypeVar("T")

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
    def __get__(self, obj: None, _objtype: type[Any] | None = None) -> _Var[T]: ...
    @overload
    def __get__(self, obj: object, _objtype: type[Any] | None = None) -> T: ...

    def __get__(self, obj: object | None, _objtype: type[Any] | None = None) -> _Var[T] | T:
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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class EnvRegistry:
    """Typed, live-read/write access to ``VANE_*`` environment variables.

    Attributes are grouped by subsystem. This registry is the stable public
    programmatic configuration surface; internal/debug-only environment
    variables may still be read directly by their subsystem.
    """

    ndjson_max_split_bytes = _Var(
        "VANE_NDJSON_MAX_SPLIT_BYTES",
        int,
        256 * 1024 * 1024,
        "Maximum nominal distributed NDJSON range size in bytes (minimum 2 MiB). "
        "Read during coordinator planning; line alignment can extend ranges beyond this size.",
    )
    # -- UDF ----------------------------------------------------------

    udf_parallel = _Var(
        "VANE_UDF_PARALLEL",
        bool,
        False,
        "Enable parallel UDF execution.",
    )
    udf_arrow_fastpath = _Var(
        "VANE_UDF_ARROW_FASTPATH",
        bool,
        True,
        "Use Arrow zero-copy fast path for UDF I/O.",
    )

    # -- Local exchange -----------------------------------------------------

    local_exchange_buffer = _Var(
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

            env.set(udf_parallel=True)
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
