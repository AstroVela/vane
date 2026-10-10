# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-device execution leases on the common local slot arbiter.

One replica owns one device. There is no additional semaphore or wait queue:
the existing slot is the device's execution capacity. Resident reservations
remain with ModelPoolRegistry, independently of these per-invocation records.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vane.execution.resources import ResourceVector
from vane.execution.udf_admission import AdmissionLease
from vane.execution.udf_local_actor_admission import LocalActorAdmissionAuthority, LocalActorExecutionSlotPool
from vane.execution.udf_resource_usage import UnitResourceActivity

if TYPE_CHECKING:
    from vane.execution.udf_subprocess import _SingleSubprocessExecutor


class LocalGpuExecution:
    def __init__(self, pool: LocalGpuExecutionSlotPool, lease_id: str, replica: int) -> None:
        self.pool = pool
        self.lease_id = lease_id
        self.replica = replica
        self.device = pool.devices[replica]
        self.generation: int | None = None
        self.pid: int | None = None
        self.submitted = False
        self.backend_done = False
        self.completion_requested = False
        self.cleanup_owners: tuple[_SingleSubprocessExecutor, ...] = ()
        self.activity = UnitResourceActivity({"device": self.device}).open_task()
        self.activity.transition("ready")

    def start(self, worker: _SingleSubprocessExecutor) -> None:
        assignment = worker._local_gpu_assignment
        if assignment is None or assignment[:2] != (self.device, self.replica):
            raise RuntimeError("GPU execution lease does not match its worker device")
        with self.pool._lock:
            self.generation = assignment[2]
            self.pid = getattr(worker._proc, "pid", None)
            self.activity.transition("running")

    def backend_finished(self, cleanup_owners: tuple[_SingleSubprocessExecutor, ...] = ()) -> None:
        with self.pool._lock:
            self.backend_done = True
            self.cleanup_owners = cleanup_owners
            self.activity.transition("cleanup_pending" if cleanup_owners else "completing")
            retired = self._retire_locked()
        if retired:
            self.pool._release(self.lease_id)

    def complete(self) -> None:
        with self.pool._lock:
            self.completion_requested = True
            retired = self._retire_locked()
        if retired:
            self.pool._release(self.lease_id)

    def _retire_locked(self) -> bool:
        if not self.completion_requested or (self.submitted and not self.backend_done):
            return False
        if any(owner._cleanup_finished is not True for owner in self.cleanup_owners):
            self.activity.transition("cleanup_pending")
            return False
        self.pool._executions.pop(self.lease_id, None)
        self.cleanup_owners = ()
        self.activity.finish()
        return True


class LocalGpuAdmissionAuthority(LocalActorAdmissionAuthority):
    def _lease_locked(self, slot: int, request_id: str, retained: int) -> AdmissionLease:
        lease = super()._lease_locked(slot, request_id, retained)
        pool = self._pool
        assert isinstance(pool, LocalGpuExecutionSlotPool)
        execution = LocalGpuExecution(pool, lease.lease["lease_id"], slot % pool.actor_count)
        pool._executions[execution.lease_id] = execution
        lease.lease["local_gpu_execution"] = execution
        lease._execution_finished_callback = execution.complete
        lease._release_callback = execution.complete
        return lease


class LocalGpuExecutionSlotPool(LocalActorExecutionSlotPool):
    def __init__(self, devices: tuple[str, ...], *, execution_slot_prefix: str) -> None:
        super().__init__(len(devices), execution_slot_prefix=execution_slot_prefix)
        self.devices = devices
        self._executions: dict[str, LocalGpuExecution] = {}
        self.admission_activity = UnitResourceActivity({"pool": execution_slot_prefix})

    def create_authority(self) -> LocalGpuAdmissionAuthority:
        authority = LocalGpuAdmissionAuthority(slot_pool=self)
        self.admission_activity.bind_admission(authority)
        return authority

    def claim(self, admission: AdmissionLease | None) -> LocalGpuExecution:
        execution = admission.lease.get("local_gpu_execution") if admission is not None else None
        with self._lock:
            if (
                self._closed
                or not isinstance(execution, LocalGpuExecution)
                or execution.pool is not self
                or self._executions.get(execution.lease_id) is not execution
                or execution.lease_id not in self._active_slots
                or execution.submitted
                or execution.completion_requested
            ):
                raise RuntimeError("GPU submission requires one live admission lease from its model pool")
            execution.submitted = True
            execution.activity.transition("submitted")
            return execution

    def retry_cleanup(self) -> None:
        # Physical cleanup belongs to the actor pool. Only retire records whose
        # owners have confirmed completion; snapshots never perform this work.
        with self._lock:
            retired = [e.lease_id for e in tuple(self._executions.values()) if e._retire_locked()]
        for lease_id in retired:
            self._release(lease_id)

    def cleanup_pending(self) -> bool:
        with self._lock:
            return any(execution.submitted for execution in self._executions.values())

    def close(self) -> None:
        try:
            super().close()
        finally:
            retired = []
            with self._lock:
                for execution in tuple(self._executions.values()):
                    if not execution.submitted:
                        execution.completion_requested = True
                        if execution._retire_locked():
                            retired.append(execution.lease_id)
            for lease_id in retired:
                self._release(lease_id)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            ready = [a._ready_slot % self.actor_count for a in self._authorities if a._ready_slot is not None]
            held = [slot % self.actor_count for slot, _ in self._active_slots.values()]
            records = [
                {
                    "lease_id": e.lease_id,
                    "device": e.device,
                    "replica": e.replica,
                    "generation": e.generation,
                    "pid": e.pid,
                    "state": e.activity.diagnostic_state(),
                }
                for e in self._executions.values()
            ]
            devices = []
            for replica, device in enumerate(self.devices):
                executions = [r for r in records if r["replica"] == replica]
                devices.append(
                    {
                        "device": device,
                        "replica": replica,
                        "capacity": 1,
                        "prefetch_depth": self.prefetch_depth,
                        "ready_slots": ready.count(replica),
                        "retained_slots": int(replica in held and not executions),
                        "execution_resources": ResourceVector(gpu=int(replica in ready or bool(executions))).to_dict(),
                        "executions": executions,
                    }
                )
            closed = self._closed
        return {
            "closed": closed,
            "devices": devices,
            # Pending requests are not assigned a device yet. Keep them at pool
            # scope instead of reporting each waiter once for every replica.
            "admission": self.admission_activity.snapshot(),
        }
