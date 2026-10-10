# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Transport-independent output ownership, shared by local and Ray execution."""

from __future__ import annotations

import os
import threading
from typing import Protocol

_OUTPUT_STATES = (
    "generator_pending",
    "unit_queue",
    "downstream_input",
    "external_consumer",
    "released",
)


class OutputLeaseRecord(Protocol):
    @property
    def lease_id(self) -> str: ...

    @property
    def state(self) -> str: ...


class OutputLeaseManager(Protocol):
    def transition_output_block(self, lease_id: str, state: str) -> bool: ...

    def release_output_block(self, lease_id: str) -> bool: ...


class ForkableOutputLease(Protocol):
    def fork(self) -> ForkableOutputLease: ...

    def transition_to(self, state: str) -> bool: ...

    def release(self) -> bool: ...


class OutputLeaseSet:
    """Keep independent query-policy and explicit data-limit owners together."""

    def __init__(self, leases: tuple[ForkableOutputLease, ...]) -> None:
        self.leases = leases

    def fork(self) -> OutputLeaseSet:
        copies = []
        try:
            for lease in self.leases:
                copies.append(lease.fork())
            return OutputLeaseSet(tuple(copies))
        except BaseException as error:
            cleanup_error = None
            for copy in copies:
                try:
                    copy.release()
                except BaseException as exc:
                    cleanup_error = exc
            if cleanup_error is not None:
                raise error from cleanup_error
            raise

    def transition_to(self, state: str) -> bool:
        changed = False
        for lease in self.leases:
            changed = lease.transition_to(state) or changed
        return changed

    def release(self) -> bool:
        error = None
        released = False
        for lease in self.leases:
            try:
                released = lease.release() or released
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error
        return released


class _SharedOutputLease:
    def __init__(self, owner: OutputBlockLeaseOwner) -> None:
        self.owner = owner
        self.lock = threading.Lock()
        self.refs = 1
        self.pid = os.getpid()


class RetainedOutputLease:
    """Retain one policy charge through zero-copy views without double charging."""

    def __init__(self, owner: OutputBlockLeaseOwner, *, _shared: _SharedOutputLease | None = None) -> None:
        self._shared = _shared or _SharedOutputLease(owner)
        self._released = False

    def fork(self) -> RetainedOutputLease:
        shared = self._shared
        if shared.pid != os.getpid():
            return self
        with shared.lock:
            if self._released:
                raise RuntimeError("query output lease is released")
            shared.refs += 1
            return RetainedOutputLease(shared.owner, _shared=shared)

    def transition_to(self, state: str) -> bool:
        shared = self._shared
        if shared.pid != os.getpid():
            return False
        return shared.owner.transition_to(state)

    def release(self) -> bool:
        shared = self._shared
        if shared.pid != os.getpid():
            return False
        with shared.lock:
            if self._released:
                return False
            released = shared.owner.release() if shared.refs == 1 else False
            self._released = True
            shared.refs -= 1
            return released


class OutputBlockLeaseOwner:
    """Shared lifetime owner carried with one query-produced data block."""

    def __init__(self, manager: OutputLeaseManager, lease: OutputLeaseRecord) -> None:
        self._owner_pid = os.getpid()
        self._manager = manager
        self._lease_id = str(lease.lease_id)
        self._state = str(lease.state)
        self._released = False
        self._lock = threading.Lock()

    @property
    def lease_id(self) -> str:
        return self._lease_id

    @property
    def state(self) -> str:
        if self._owner_pid != os.getpid():
            return "released"
        with self._lock:
            return "released" if self._released else self._state

    def transition_to(self, state: str) -> bool:
        if self._owner_pid != os.getpid():
            return False
        target = str(state)
        if target not in _OUTPUT_STATES or target == "released":
            raise ValueError(f"invalid output lease owner transition target: {target}")
        with self._lock:
            if self._released:
                return False
            current_index = _OUTPUT_STATES.index(self._state)
            target_index = _OUTPUT_STATES.index(target)
            if target_index < current_index:
                raise ValueError(f"output lease owner cannot move backward: {self._state} -> {target}")
            while current_index < target_index:
                next_state = _OUTPUT_STATES[current_index + 1]
                if not self._manager.transition_output_block(self._lease_id, next_state):
                    self._released = True
                    return False
                self._state = next_state
                current_index += 1
            return True

    def release(self) -> bool:
        if self._owner_pid != os.getpid():
            return False
        with self._lock:
            if self._released:
                return False
            released = self._manager.release_output_block(self._lease_id)
            self._released = True
            self._state = "released"
            return bool(released)

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            # Query teardown may already have canceled and removed the manager's
            # leases. Destructors cannot safely surface that idempotent race.
            pass
