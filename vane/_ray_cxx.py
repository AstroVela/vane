# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, cast, overload

from vane._native import ray_cxx
from vane._ray_errors import remote_ray_exception

_DEFAULT_HINT = "Ensure the C++ ray extension is built and importable."


class _CleanupFlightShuffleForQuery(Protocol):
    def __call__(
        self,
        query_id: str,
        cleanup_connection: object | None = ...,
        connection_snapshot_query_id: str = ...,
        apply_snapshot_s3_credentials: bool = ...,
        effective_session_config: object | None = ...,
    ) -> dict[str, int | str]: ...


@overload
def require_ray_cxx_attr(
    name: Literal["DistributedPhysicalPlanRunner"], *, hint: str | None = None
) -> type[ray_cxx.DistributedPhysicalPlanRunner]: ...


@overload
def require_ray_cxx_attr(name: Literal["PyLogicalPlan"], *, hint: str | None = None) -> type[ray_cxx.PyLogicalPlan]: ...


@overload
def require_ray_cxx_attr(
    name: Literal["RayResultPartitionRef"], *, hint: str | None = None
) -> type[ray_cxx.RayResultPartitionRef]: ...


@overload
def require_ray_cxx_attr(name: Literal["RayTaskResult"], *, hint: str | None = None) -> type[ray_cxx.RayTaskResult]: ...


@overload
def require_ray_cxx_attr(
    name: Literal["RayWorkerRuntime"], *, hint: str | None = None
) -> type[ray_cxx.RayWorkerRuntime]: ...


@overload
def require_ray_cxx_attr(name: Literal["RayWorkerTask"], *, hint: str | None = None) -> type[ray_cxx.RayWorkerTask]: ...


@overload
def require_ray_cxx_attr(
    name: Literal["NativeDistributedTaskResult"], *, hint: str | None = None
) -> type[ray_cxx.NativeDistributedTaskResult]: ...


@overload
def require_ray_cxx_attr(
    name: Literal["merge_scan_task_descriptors"], *, hint: str | None = None
) -> Callable[[list[bytes]], bytes]: ...


@overload
def require_ray_cxx_attr(
    name: Literal["describe_native_progress"], *, hint: str | None = None
) -> Callable[[object, ray_cxx.DistributedPhysicalPlan], dict[str, Any]]: ...


@overload
def require_ray_cxx_attr(
    name: Literal["_register_query_python_replay_state"], *, hint: str | None = None
) -> Callable[[str, ray_cxx.DistributedPhysicalPlan], bool]: ...


@overload
def require_ray_cxx_attr(
    name: Literal["_lookup_query_connection_snapshot"], *, hint: str | None = None
) -> Callable[[str], Mapping[str, Any] | None]: ...


@overload
def require_ray_cxx_attr(
    name: Literal[
        "_cleanup_query_python_replay_state",
        "begin_flight_shuffle_query_execution",
        "close_flight_shuffle_query",
        "end_flight_shuffle_query_execution",
        "retire_flight_shuffle_query",
    ],
    *,
    hint: str | None = None,
) -> Callable[[str], None]: ...


@overload
def require_ray_cxx_attr(
    name: Literal["shutdown_local_flight_service"], *, hint: str | None = None
) -> Callable[[], None]: ...


@overload
def require_ray_cxx_attr(
    name: Literal["flight_shuffle_query_status"], *, hint: str | None = None
) -> Callable[[str], dict[str, Any]]: ...


@overload
def require_ray_cxx_attr(
    name: Literal["cleanup_flight_shuffle_for_query"], *, hint: str | None = None
) -> _CleanupFlightShuffleForQuery: ...


@overload
def require_ray_cxx_attr(name: str, *, hint: str | None = None) -> object: ...


def require_ray_cxx_attr(name: str, *, hint: str | None = None) -> object:
    """Return a lazily resolved vane.ray_cxx binding or raise a clear error."""
    try:
        return cast(object, getattr(ray_cxx, name))
    except AttributeError as ex:
        raise ImportError(
            f"Required C++ binding `vane.ray_cxx.{name}` is not available. {hint or _DEFAULT_HINT}"
        ) from ex


def validate_plan_serialization_for_submission(plan: Any) -> None:
    """Validate a native physical root before Driver resource registration."""
    validator = getattr(plan, "_validate_serializable_for_submission", None)
    if not callable(validator):
        raise TypeError("distributed physical plan serialization validator must be callable")
    try:
        validator()
    except Exception as exc:
        query_id = str(plan.idx())
        message = f"distributed physical plan serialization preflight failed for query_id={query_id}"
        raise remote_ray_exception(message, exc) from exc
