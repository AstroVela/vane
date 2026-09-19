# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Transport-independent output ownership, shared by local and Ray execution."""

from __future__ import annotations

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


class OutputBlockLeaseOwner:
    """Shared lifetime owner carried with one query-produced data block."""

    def __init__(self, manager: OutputLeaseManager, lease: OutputLeaseRecord) -> None:
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
        with self._lock:
            return "released" if self._released else self._state

    def transition_to(self, state: str) -> bool:
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
