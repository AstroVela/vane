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
from vane.execution.udf_admission import AdmissionLease, LocalExecutionSlotPool, LocalSlotAdmissionAuthority
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
            self._retire_locked()

    def complete(self) -> None:
        with self.pool._lock:
            self.completion_requested = True
            self._retire_locked()

    def _retire_locked(self) -> None:
        if not self.completion_requested or (self.submitted and not self.backend_done):
            return
        if any(owner._cleanup_finished is not True for owner in self.cleanup_owners):
            self.activity.transition("cleanup_pending")
            return
        self.pool._executions.pop(self.lease_id, None)
        self.cleanup_owners = ()
        self.activity.finish()


class LocalGpuAdmissionAuthority(LocalSlotAdmissionAuthority):
    def _lease_locked(self, slot: int, request_id: str, retained: int) -> AdmissionLease:
        lease = super()._lease_locked(slot, request_id, retained)
        pool = self._pool
        assert isinstance(pool, LocalGpuExecutionSlotPool)
        execution = LocalGpuExecution(pool, lease.lease["lease_id"], slot)
        pool._executions[execution.lease_id] = execution
        lease.lease["local_gpu_execution"] = execution
        lease._execution_finished_callback = execution.complete
        return lease


class LocalGpuExecutionSlotPool(LocalExecutionSlotPool):
    def __init__(self, devices: tuple[str, ...], *, execution_slot_prefix: str) -> None:
        super().__init__(max_slots=len(devices), execution_slot_prefix=execution_slot_prefix)
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
            for execution in tuple(self._executions.values()):
                execution._retire_locked()

    def cleanup_pending(self) -> bool:
        with self._lock:
            return any(execution.submitted for execution in self._executions.values())

    def close(self) -> None:
        try:
            super().close()
        finally:
            with self._lock:
                for execution in tuple(self._executions.values()):
                    if not execution.submitted:
                        execution.completion_requested = True
                        execution._retire_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            ready = {a._ready_slot for a in self._authorities if a._ready_slot is not None}
            held = {slot for slot, _ in self._active_slots.values()}
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
                        "ready_slots": int(replica in ready),
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
