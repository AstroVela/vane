# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-actor prefetch slots, independent of the single executing invocation."""

from __future__ import annotations

import os

from vane.execution.udf_admission import AdmissionLease, LocalExecutionSlotPool, LocalSlotAdmissionAuthority
from vane.execution.udf_resource_policy import actor_prefetch_depth


class LocalActorAdmissionAuthority(LocalSlotAdmissionAuthority):
    def __init__(self, *, slot_pool: LocalActorExecutionSlotPool) -> None:
        self._selected_actor: int | None = None
        super().__init__(slot_pool=slot_pool)

    def select_actor(self, actor_index: int) -> None:
        pool = self._pool
        assert isinstance(pool, LocalActorExecutionSlotPool)
        with pool._lock:
            if self._state != "idle" or not 0 <= actor_index < pool.actor_count:
                raise ValueError("actor selection requires an idle authority and an existing replica")
            self._selected_actor = actor_index

    def _lease_locked(self, slot: int, request_id: str, retained: int) -> AdmissionLease:
        lease = super()._lease_locked(slot, request_id, retained)
        pool = self._pool
        assert isinstance(pool, LocalActorExecutionSlotPool)
        lease.lease["actor_index"] = slot % pool.actor_count
        # Buffered output retains its data owner, not an actor execution slot.
        lease._execution_finished_callback = lambda: pool._release(lease.lease["lease_id"])
        self._selected_actor = None
        return lease


class LocalActorExecutionSlotPool(LocalExecutionSlotPool):
    def __init__(self, actor_count: int, *, execution_slot_prefix: str) -> None:
        self.actor_count = int(actor_count)
        self.prefetch_depth = actor_prefetch_depth(os.environ)
        super().__init__(max_slots=self.actor_count * self.prefetch_depth, execution_slot_prefix=execution_slot_prefix)

    def create_authority(self) -> LocalActorAdmissionAuthority:
        return LocalActorAdmissionAuthority(slot_pool=self)

    def actor_loads(self) -> dict[int, int]:
        with self._lock:
            loads = dict.fromkeys(range(self.actor_count), self.prefetch_depth)
            for slot in self._available_slots:
                loads[slot % self.actor_count] -= 1
            return loads

    def _select_slot_locked(self, authority: LocalSlotAdmissionAuthority | None) -> int | None:
        selected = authority._selected_actor if isinstance(authority, LocalActorAdmissionAuthority) else None
        slots = [slot for slot in self._available_slots if selected is None or slot % self.actor_count == selected]
        if not slots:
            return None
        # Prefer the replica with the fewest admitted calls, then its oldest
        # free token. The query policy may instead select one exact replica.
        counts = {replica: 0 for replica in range(self.actor_count)}
        for slot in self._available_slots:
            counts[slot % self.actor_count] += 1
        return min(slots, key=lambda slot: (-counts[slot % self.actor_count], slot))
