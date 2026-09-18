# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit model-pool ownership, independent of the execution transport.

This registry grants lifetime borrows, not execution admission or Ray query
authorization. Adapters must continue to enforce those separate contracts.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from vane._ray_errors import RemoteRayException
from vane.execution._diagnostics import bounded_utf8_text, exception_message_from_args, safe_exception_type_name
from vane.execution.resources import ResourceVector
from vane.execution.udf_actor_pool_lifecycle import (
    OwnedActorPoolsError,
    actor_pool_cleanup_pending,
    rollback_actor_pools,
)


class ModelPool(Protocol):
    def shutdown(self, *, kill: bool = False) -> None: ...

    def cleanup_pending(self) -> bool: ...


_Pool = TypeVar("_Pool", bound=ModelPool)


class ModelPoolCapacityError(RuntimeError):
    """A resident reservation cannot fit; no model initialization was attempted."""

    def __init__(self, requested: ResourceVector, reserved: ResourceVector, limit: ResourceVector) -> None:
        self.requested = requested
        self.reserved = reserved
        self.limit = limit
        self.oversized = not requested.fits_within(limit)
        self.dimensions = (reserved + requested).exceeded_dimensions(limit)
        reason = "request exceeds resident limit" if self.oversized else "resident capacity is in use"
        super().__init__(
            f"model pool {reason} ({', '.join(self.dimensions)}): "
            f"requested={requested.to_dict()}, reserved={reserved.to_dict()}, limit={limit.to_dict()}"
        )


@dataclass(frozen=True)
class ModelPoolIdentity:
    session_id: str
    model: str
    version: str
    backend: str
    initialization: str
    configuration: str

    def __post_init__(self) -> None:
        for value in (
            self.session_id,
            self.model,
            self.version,
            self.backend,
            self.initialization,
            self.configuration,
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("model pool identity fields must be non-empty strings")


@dataclass(frozen=True)
class _InitializationFailure:
    snapshot: RemoteRayException | None
    description: str

    @classmethod
    def capture(cls, error: BaseException) -> _InitializationFailure:
        message = exception_message_from_args(error)
        detail = bounded_utf8_text(message, 2048) if message is not None else "no simple diagnostic message"
        description = f"cached model initialization failed ({safe_exception_type_name(error)}): {detail}"
        try:
            # Reuse Ray's bounded value-only snapshot without loading Ray. The
            # carrier is never raised or chained to the original exception.
            snapshot = RemoteRayException.from_exception(error)
        except BaseException:
            # Custom exception arguments/reducers need not be transportable.
            snapshot = None
        return cls(snapshot, description)

    def restore(self) -> BaseException:
        if self.snapshot is not None:
            try:
                return self.snapshot.restore()
            except BaseException:
                # Local/custom exception classes may not be reconstructable.
                pass
        return RuntimeError(self.description)


@dataclass
class _Entry(Generic[_Pool]):
    create: Callable[[], _Pool]
    resources: ResourceVector
    pool: _Pool | None = None
    initializing: bool = False
    error: _InitializationFailure | None = None
    borrowers: int = 0
    reserved: bool = False
    owners: tuple[ModelPool, ...] = ()


class ModelPoolBorrow(Generic[_Pool]):
    """A query's lifetime borrow; shutdown releases only this borrow.

    The borrower must finish/cancel its executors before releasing the borrow.
    Native query resource cleanup can use the same shutdown/cleanup_pending
    contract as an owned pool, without terminating the runtime's workers.
    """

    def __init__(self, owner: ModelPoolRegistry[_Pool], entry: _Entry[_Pool]) -> None:
        self._owner = owner
        self._entry = entry
        self._released = False

    @property
    def pool(self) -> _Pool:
        with self._owner._condition:
            if self._released:
                raise RuntimeError("model pool borrow has been released")
            assert self._entry.pool is not None
            return self._entry.pool

    def release(self) -> None:
        with self._owner._condition:
            if not self._released:
                self._released = True
                self._entry.borrowers -= 1
                self._owner._condition.notify_all()

    def shutdown(self, *, kill: bool = False) -> None:
        self.release()

    def cleanup_pending(self) -> bool:
        with self._owner._condition:
            return not self._released

    def __enter__(self) -> ModelPoolBorrow[_Pool]:
        # Detect reuse of a context manager whose borrow has already ended.
        self.pool
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class ModelPoolRegistry(Generic[_Pool]):
    """Own explicitly registered pools until a quiescent, retryable close.

    Registration is lazy and initialization is single-flight per identity.
    Initialization failure is sticky: waiting borrowers receive fresh exceptions
    from a snapshot, and partial constructor owners remain until close retries
    cleanup. No cached failure retains a request's live traceback.
    Drain fences new registrations/acquisitions but lets existing borrowers
    finish. Close never revokes active borrows, including when kill=True.
    Optional resident limits are reserved before initialization and released
    only after confirmed cleanup. Capacity refusal is immediate and retryable;
    persistent models have no eviction or admission wait queue.
    """

    def __init__(self, *, resident_limit: ResourceVector | None = None) -> None:
        self._condition = threading.Condition()
        self._entries: dict[ModelPoolIdentity, _Entry[_Pool]] = {}
        self._owned: list[ModelPool] = []
        self._resident_limit = resident_limit
        self._draining = False
        self._closing = False
        self._closed = False

    def _require_open(self) -> None:
        if self._draining:
            raise RuntimeError("model pool runtime is draining or closed")

    def register(
        self,
        identity: ModelPoolIdentity,
        create: Callable[[], _Pool],
        *,
        resources: ResourceVector = ResourceVector(),
    ) -> None:
        with self._condition:
            self._require_open()
            if identity in self._entries:
                raise ValueError("model pool identity is already registered")
            if self._resident_limit is not None and not resources.fits_within(self._resident_limit):
                raise ModelPoolCapacityError(resources, ResourceVector(), self._resident_limit)
            self._entries[identity] = _Entry(create=create, resources=resources)

    def _reserve_locked(self, entry: _Entry[_Pool]) -> None:
        reserved = self._reserved_resources_locked()
        requested_total = reserved + entry.resources
        if self._resident_limit is not None and not requested_total.fits_within(self._resident_limit):
            # Capacity refusal is not an initialization failure. A later attempt
            # may succeed after a different, failed constructor finishes cleanup.
            # Resident pools have no eviction, so waiting here could deadlock.
            raise ModelPoolCapacityError(entry.resources, reserved, self._resident_limit)
        entry.reserved = True

    def _reserved_resources_locked(self) -> ResourceVector:
        # Ownership is the ledger. Derive the total instead of repeatedly
        # subtracting fractional CPU/GPU values across constructor failures.
        resources = [entry.resources for entry in self._entries.values() if entry.reserved]
        return ResourceVector(
            cpu=math.fsum(item.cpu for item in resources),
            gpu=math.fsum(item.gpu for item in resources),
            heap_bytes=sum(item.heap_bytes for item in resources),
            object_store_bytes=sum(item.object_store_bytes for item in resources),
        )

    def resource_snapshot(self) -> dict[str, Any]:
        """Report declared resources once per pool, including unfinished owners."""
        with self._condition:
            registered = ResourceVector()
            initializing = ResourceVector()
            resident = ResourceVector()
            retained_failure = ResourceVector()
            for entry in self._entries.values():
                registered += entry.resources
                if not entry.reserved:
                    continue
                if entry.initializing:
                    initializing += entry.resources
                elif entry.pool is not None:
                    resident += entry.resources
                else:
                    retained_failure += entry.resources
            return {
                "limit": None if self._resident_limit is None else self._resident_limit.to_dict(),
                "registered_resources": registered.to_dict(),
                "reserved_resources": self._reserved_resources_locked().to_dict(),
                "initializing_resources": initializing.to_dict(),
                "resident_resources": resident.to_dict(),
                "retained_failure_resources": retained_failure.to_dict(),
                "registered_models": len(self._entries),
                "reserved_models": sum(entry.reserved for entry in self._entries.values()),
                "active_borrows": sum(entry.borrowers for entry in self._entries.values()),
                "draining": self._draining,
                "closed": self._closed,
            }

    def acquire(self, identity: ModelPoolIdentity) -> ModelPoolBorrow[_Pool]:
        cached_failure = None
        with self._condition:
            while True:
                self._require_open()
                entry = self._entries[identity]
                if entry.error is not None:
                    cached_failure = entry.error
                    break
                if entry.pool is not None:
                    entry.borrowers += 1
                    return ModelPoolBorrow(self, entry)
                if not entry.initializing:
                    self._reserve_locked(entry)
                    entry.initializing = True
                    break
                self._condition.wait()

        if cached_failure is not None:
            # Reconstruction can invoke user exception constructors; keep it
            # outside the registry lock, just like model initialization.
            raise cached_failure.restore()

        # Model constructors may block or start worker processes. They must not
        # prevent another pool's borrow from finishing or drain from fencing.
        try:
            pool = entry.create()
        except BaseException as error:
            # The runtime keeps partial constructor owners. Do not hand those
            # same owners to a query's preparation rollback through the error.
            failure = error.creation_error if isinstance(error, OwnedActorPoolsError) else error
            cached_failure = _InitializationFailure.capture(failure)
            with self._condition:
                entry.owners = tuple(getattr(error, "owned_actor_pools", ()))
                self._owned.extend(entry.owners)
                if not entry.owners:
                    entry.reserved = False
                entry.error = cached_failure
                entry.initializing = False
                self._condition.notify_all()
            if failure is not error:
                raise failure from error
            raise
        with self._condition:
            self._owned.append(pool)
            entry.owners = (pool,)
            entry.pool = pool
            entry.initializing = False
            self._condition.notify_all()
            # If drain raced initialization, keep the owner for close and do
            # not publish a new borrower after the admission fence.
            self._require_open()
            entry.borrowers += 1
            return ModelPoolBorrow(self, entry)

    def prewarm(self, identity: ModelPoolIdentity) -> None:
        with self.acquire(identity):
            pass

    def drain(self) -> None:
        with self._condition:
            self._draining = True
            self._condition.notify_all()

    def close(self, *, timeout: float = 0.0, kill: bool = False) -> None:
        """Fence borrows, wait at most timeout for quiescence, then close pools.

        The timeout bounds the wait for borrowers/initializers/another close;
        backend shutdown retains its own timeout. A timeout or cleanup failure
        leaves this registry responsible for cleanup and permits another close.
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("model pool close timeout must be finite and non-negative")
        deadline = time.monotonic() + timeout
        with self._condition:
            self._draining = True
            self._condition.notify_all()
            while self._closing or any(e.initializing or e.borrowers for e in self._entries.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("model pool runtime still has active borrows, initialization, or cleanup")
                self._condition.wait(remaining)
            if self._closed:
                return
            self._closing = True
            owned = list(self._owned)

        errors: list[BaseException] = []

        def shutdown(pool: ModelPool) -> None:
            pool.shutdown(kill=kill)
            if actor_pool_cleanup_pending(pool):
                raise RuntimeError("model pool shutdown retained unfinished owners")

        try:
            remaining_owned = rollback_actor_pools(
                owned,
                RuntimeError("model pool runtime close"),
                shutdown=shutdown,
                cleanup_pending=actor_pool_cleanup_pending,
                record_error=errors.append,
            )
            with self._condition:
                self._owned = remaining_owned
                remaining_ids = {id(pool) for pool in remaining_owned}
                for entry in self._entries.values():
                    entry.owners = tuple(pool for pool in entry.owners if id(pool) in remaining_ids)
                    if not entry.owners:
                        entry.reserved = False
                self._closed = not remaining_owned
                if self._closed:
                    self._entries.clear()
            if errors:
                raise OwnedActorPoolsError(
                    "model pool runtime cleanup failed; retry close to release retained owners",
                    owned_actor_pools=remaining_owned,
                    creation_error=errors[0],
                ) from errors[0]
        finally:
            with self._condition:
                self._closing = False
                self._condition.notify_all()

    def __enter__(self) -> ModelPoolRegistry[_Pool]:
        with self._condition:
            self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
