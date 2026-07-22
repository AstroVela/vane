# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral helpers for Vane lazy/ref-bundle payloads."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import weakref
from dataclasses import dataclass, field
from itertools import count
from multiprocessing import resource_tracker as _resource_tracker
from multiprocessing import shared_memory
from typing import Any

import pyarrow as pa

from vane.execution._common import ensure_table as _ensure_table
from vane.execution._common import estimate_table_bytes

REF_BUNDLE_RESULT_MARKER = "__vane_ref_bundle_result__"
SUBMIT_RESULT_MARKER = "__vane_submit_result__"
LOCAL_SHM_PROVIDER = "local_shm"

_IPC_HEADER_SIZE = 8
_MIB = 1 << 20
_GIB = 1 << 30
_TRUTHY_FALSE_VALUES = ("", "0", "false", "no", "off")
_RAY_LIKE_OBJECT_STORE_MEMORY_FRACTION = 0.3
_RAY_LIKE_SHM_MEMORY_FRACTION = 0.95
_RAY_LIKE_OBJECT_STORE_MAX_BYTES = 200 * _GIB
_LOCAL_SHM_REF_BUDGET_MIN_BYTES = 512 * _MIB
_LOCAL_SHM_REF_OUTPUT_PRODUCER_SOFT_LIMIT_FRACTION = 0.75
_LOCAL_SHM_REF_OUTPUT_PRODUCER_SOFT_MIN_BYTES = 16 * _MIB
_deferred_shm_close_lock = threading.Lock()
_deferred_shm_closes: list[shared_memory.SharedMemory] = []
_shm_debug_lock = threading.Lock()
_shm_debug_seq = 0
_local_shm_budget_cond = threading.Condition()
_local_shm_budget_wakeup_lock = threading.Lock()
_local_shm_budget_wakeup_callbacks: set[Any] = set()
_local_shm_budget_reserved_bytes = 0
_local_shm_budget_pending_output_bytes = 0
_local_shm_refs_created = 0
_local_shm_refs_released = 0


@dataclass
class _InputLease:
    lease_id: int
    refs: tuple[Any, ...]
    bytes: int
    name: str
    owner_operator_id: str
    consumer_operator_id: str
    submit_id: int | None
    reserve_output_credit: bool = True
    state: str = "active"


@dataclass
class _OutputGrant:
    grant_id: int
    bytes: int
    name: str
    priority: str
    state: str = "active"


@dataclass
class _InputRefHold:
    key: tuple[str, Any]
    count: int = 0
    refs: dict[int, Any] = field(default_factory=dict)
    name: str = ""
    size: int = 0


def _shm_debug_enabled() -> bool:
    raw = os.getenv("VANE_SHM_REF_DEBUG", "").strip().lower()
    return raw not in _TRUTHY_FALSE_VALUES


def _shm_debug_log(event: str, **fields: Any) -> None:
    if not _shm_debug_enabled():
        return
    global _shm_debug_seq
    with _shm_debug_lock:
        _shm_debug_seq += 1
        seq = _shm_debug_seq
    parts = [
        f"event={event}",
        f"seq={seq}",
        f"pid={os.getpid()}",
    ]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print("[vane-shm-ref] " + " ".join(parts), file=sys.stderr, flush=True)


def _parse_byte_size(value: str) -> int:
    text = str(value).strip().lower().replace("_", "")
    if not text:
        raise ValueError("byte size must be non-empty")
    suffixes = (
        ("gib", _GIB),
        ("gb", _GIB),
        ("g", _GIB),
        ("mib", _MIB),
        ("mb", _MIB),
        ("m", _MIB),
        ("kib", 1 << 10),
        ("kb", 1 << 10),
        ("k", 1 << 10),
    )
    for suffix, multiplier in suffixes:
        if text.endswith(suffix):
            parsed = int(float(text[: -len(suffix)]) * multiplier)
            if parsed <= 0:
                raise ValueError("byte size must be positive")
            return parsed
    parsed = int(float(text))
    if parsed <= 0:
        raise ValueError("byte size must be positive")
    return parsed


def _available_system_memory_bytes() -> int:
    with open("/proc/meminfo", encoding="utf-8") as meminfo:
        for line in meminfo:
            if not line.startswith("MemAvailable:"):
                continue
            parts = line.split()
            if len(parts) < 2:
                break
            available = int(parts[1]) * 1024
            if available <= 0:
                raise RuntimeError("/proc/meminfo reports no available memory")
            return available
    raise RuntimeError("/proc/meminfo is missing MemAvailable")


def _available_local_shm_bytes() -> int:
    stat = os.statvfs("/dev/shm")
    return int(stat.f_frsize) * int(stat.f_bavail)


def _auto_local_shm_ref_budget_bytes() -> int:
    available_memory = _available_system_memory_bytes()
    if available_memory <= 0:
        raise RuntimeError("system has no available memory for local shared-memory UDF output")
    available_shm = _available_local_shm_bytes()
    if available_shm <= 0:
        raise RuntimeError("/dev/shm has no available capacity for local shared-memory UDF output")
    candidates = [
        _RAY_LIKE_OBJECT_STORE_MAX_BYTES,
        int(available_memory * _RAY_LIKE_OBJECT_STORE_MEMORY_FRACTION),
        int(available_shm * _RAY_LIKE_SHM_MEMORY_FRACTION),
    ]
    capacity = min(candidates)
    return min(capacity, max(_LOCAL_SHM_REF_BUDGET_MIN_BYTES, int(capacity * 0.5)))


def _local_shm_ref_budget_limit_bytes() -> int:
    raw = os.getenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", "auto").strip().lower()
    if raw in _TRUTHY_FALSE_VALUES or raw in {"none", "unbounded", "unlimited"}:
        return 0
    if raw == "auto":
        return _auto_local_shm_ref_budget_bytes()
    try:
        return _parse_byte_size(raw)
    except Exception as exc:
        raise ValueError(f"VANE_LOCAL_SHM_REF_BUDGET_BYTES is not a valid byte size: {raw!r}") from exc


def _make_local_shm_ref_budget_limit_factory():
    auto_limit_bytes: int | None = None
    auto_limit_lock = threading.Lock()

    def resolve_auto_once() -> int:
        nonlocal auto_limit_bytes
        with auto_limit_lock:
            if auto_limit_bytes is None:
                auto_limit_bytes = _auto_local_shm_ref_budget_bytes()
            return auto_limit_bytes

    def limit_bytes() -> int:
        raw = os.getenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", "auto").strip().lower()
        if raw in _TRUTHY_FALSE_VALUES or raw in {"none", "unbounded", "unlimited"}:
            return 0
        if raw == "auto":
            return resolve_auto_once()
        try:
            return _parse_byte_size(raw)
        except Exception as exc:
            raise ValueError(f"VANE_LOCAL_SHM_REF_BUDGET_BYTES is not a valid byte size: {raw!r}") from exc

    return limit_bytes


def _local_shm_budget_usage_locked() -> int:
    return _LOCAL_SHM_BUDGET_MANAGER.snapshot()["usage_bytes"]


def _notify_local_shm_budget_wakeup_callbacks() -> None:
    with _local_shm_budget_wakeup_lock:
        callbacks = tuple(_local_shm_budget_wakeup_callbacks)
    errors: list[Exception] = []
    for callback in callbacks:
        try:
            callback()
        except Exception as exc:
            errors.append(exc)
    if not errors:
        return
    if len(errors) == 1:
        raise RuntimeError(f"local_shm budget wakeup callback failed: {errors[0]}") from errors[0]
    message = "; ".join(str(error) for error in errors)
    raise RuntimeError(f"local_shm budget wakeup callbacks failed: {message}") from errors[0]


def register_local_shm_ref_budget_wakeup(callback: Any):
    with _local_shm_budget_wakeup_lock:
        _local_shm_budget_wakeup_callbacks.add(callback)

    def unregister() -> None:
        with _local_shm_budget_wakeup_lock:
            _local_shm_budget_wakeup_callbacks.discard(callback)

    return unregister


class LocalShmBudgetManager:
    def __init__(self, *, limit_factory: Any | None = None) -> None:
        self._cond = threading.Condition()
        self._limit_factory = limit_factory or _make_local_shm_ref_budget_limit_factory()
        self._allocated_bytes = 0
        self._output_grant_bytes = 0
        self._output_credit_bytes = 0
        self._input_lease_bytes = 0
        self._input_leases: dict[int, _InputLease] = {}
        self._output_grants: dict[int, _OutputGrant] = {}
        self._output_credits: dict[int, int] = {}
        self._input_ref_holds: dict[tuple[str, Any], _InputRefHold] = {}
        self._lease_ids = count(1)
        self._grant_ids = count(1)
        self._waiting_output_grants = 0
        self._input_consumed_count = 0
        self._refs_released_by_input_ack = 0
        self._oversized_output_grants = 0

    def _limit_locked(self) -> int:
        return max(0, int(self._limit_factory()))

    def _usage_locked(self) -> int:
        return self._allocated_bytes + self._output_grant_bytes + self._output_credit_bytes

    def _output_grant_admission_usage_locked(self, input_credit: int) -> int:
        if input_credit <= 0:
            return self._usage_locked()
        return max(0, self._usage_locked() - input_credit)

    def snapshot(self) -> dict[str, int]:
        with self._cond:
            limit = self._limit_locked()
            usage = self._usage_locked()
            return {
                "limit_bytes": limit,
                "allocated_bytes": self._allocated_bytes,
                "reserved_bytes": self._allocated_bytes,
                "output_grant_bytes": self._output_grant_bytes,
                "pending_output_bytes": self._output_grant_bytes,
                "output_credit_bytes": self._output_credit_bytes,
                "input_lease_bytes": self._input_lease_bytes,
                "usage_bytes": usage,
                "available_bytes": max(0, limit - usage) if limit > 0 else 0,
                "active_input_leases": len(self._input_leases),
                "active_input_ref_holds": len(self._input_ref_holds),
                "active_input_ref_hold_count": sum(hold.count for hold in self._input_ref_holds.values()),
                "active_output_credits": len(self._output_credits),
                "waiting_output_grants": self._waiting_output_grants,
                "input_consumed_count": self._input_consumed_count,
                "refs_released_by_input_ack": self._refs_released_by_input_ack,
                "oversized_output_grants": self._oversized_output_grants,
            }

    def can_claim_output(self, size: int) -> bool:
        requested = max(0, int(size))
        if requested <= 0:
            return True
        with self._cond:
            limit = self._limit_locked()
            if limit <= 0:
                return True
            usage = self._usage_locked()
            return usage <= 0 or usage + requested <= limit

    def can_admit_output_submit(self, size: int, *, projected_output_bytes: int = 0) -> bool:
        requested = max(0, int(size))
        if requested <= 0:
            return True
        projected = max(0, int(projected_output_bytes))
        with self._cond:
            limit = self._limit_locked()
            if limit <= 0:
                return True
            if requested < _LOCAL_SHM_REF_OUTPUT_PRODUCER_SOFT_MIN_BYTES:
                return True
            usage = self._usage_locked() + projected
            if requested > limit:
                return usage <= 0
            if usage > 0 and usage + requested > limit:
                return False
            if usage <= 0:
                return True
            soft_limit = int(limit * _LOCAL_SHM_REF_OUTPUT_PRODUCER_SOFT_LIMIT_FRACTION)
            soft_limit = max(1, min(limit, soft_limit))
            soft_limit = max(soft_limit, min(limit, requested))
            return usage + requested <= soft_limit

    def acquire_allocation(self, size: int, *, name: str = "", block: bool = True) -> int:
        requested = max(0, int(size))
        if requested <= 0:
            return 0
        with self._cond:
            limit = self._limit_locked()
            if limit <= 0:
                return 0
            while block and limit > 0 and self._usage_locked() > 0 and self._usage_locked() + requested > limit:
                _shm_debug_log(
                    "budget_wait",
                    name=name or "-",
                    size=requested,
                    reserved_bytes=self._allocated_bytes,
                    pending_output_bytes=self._output_grant_bytes,
                    input_lease_bytes=self._input_lease_bytes,
                    limit_bytes=limit,
                )
                self._cond.wait()
                limit = self._limit_locked()
                if limit <= 0:
                    return 0
            if not block and self._usage_locked() > 0 and self._usage_locked() + requested > limit:
                _shm_debug_log(
                    "budget_overcommit",
                    name=name or "-",
                    size=requested,
                    reserved_bytes=self._allocated_bytes,
                    pending_output_bytes=self._output_grant_bytes,
                    input_lease_bytes=self._input_lease_bytes,
                    limit_bytes=limit,
                )
            self._allocated_bytes += requested
            _shm_debug_log(
                "budget_acquire",
                name=name or "-",
                size=requested,
                reserved_bytes=self._allocated_bytes,
                pending_output_bytes=self._output_grant_bytes,
                input_lease_bytes=self._input_lease_bytes,
                limit_bytes=limit,
            )
            return requested

    def release_allocation(self, size: int, *, name: str = "") -> None:
        released = max(0, int(size))
        if released <= 0:
            return
        with self._cond:
            self._allocated_bytes = max(0, self._allocated_bytes - released)
            _shm_debug_log(
                "budget_release",
                name=name or "-",
                size=released,
                reserved_bytes=self._allocated_bytes,
                pending_output_bytes=self._output_grant_bytes,
                input_lease_bytes=self._input_lease_bytes,
                limit_bytes=self._limit_locked(),
            )
            self._cond.notify_all()
        _notify_local_shm_budget_wakeup_callbacks()

    def create_input_lease(
        self,
        refs: tuple[Any, ...] | list[Any],
        bytes: int,
        *,
        name: str = "",
        owner_operator_id: str = "",
        consumer_operator_id: str = "",
        submit_id: int | None = None,
        reserve_output_credit: bool = True,
    ) -> int:
        lease_bytes = max(0, int(bytes))
        with self._cond:
            lease_id = next(self._lease_ids)
            lease_refs = tuple(refs)
            for ref in lease_refs:
                self._retain_input_ref_locked(ref, lease_id=lease_id)
            self._input_leases[lease_id] = _InputLease(
                lease_id=lease_id,
                refs=lease_refs,
                bytes=lease_bytes,
                name=name or f"input-lease-{lease_id}",
                owner_operator_id=owner_operator_id,
                consumer_operator_id=consumer_operator_id,
                submit_id=submit_id,
                reserve_output_credit=bool(reserve_output_credit),
            )
            self._input_lease_bytes += lease_bytes
            _shm_debug_log(
                "input_lease_create",
                name=name or "-",
                lease_id=lease_id,
                size=lease_bytes,
                input_lease_bytes=self._input_lease_bytes,
            )
            return lease_id

    def consume_input_lease(self, lease_id: int, *, name: str = "") -> int:
        return self._finish_input_lease(int(lease_id), state="consumed", name=name)

    def cancel_input_lease(self, lease_id: int, *, name: str = "") -> int:
        return self._finish_input_lease(int(lease_id), state="cancelled", name=name)

    def _finish_input_lease(self, lease_id: int, *, state: str, name: str = "") -> int:
        notify_budget_waiters = False
        refs_to_release: tuple[Any, ...] = ()
        with self._cond:
            lease = self._input_leases.pop(lease_id, None)
            if lease is None:
                released_credit = self._release_output_credit_locked(lease_id, name=name or f"input-lease-{lease_id}")
                if released_credit > 0:
                    self._cond.notify_all()
                    notify_budget_waiters = True
            else:
                lease.state = state
                self._input_lease_bytes = max(0, self._input_lease_bytes - lease.bytes)
                refs = lease.refs
                refs_to_release = self._release_input_refs_locked(refs, lease_id=lease_id, state=state)
                if state == "consumed":
                    self._input_consumed_count += 1
                    self._refs_released_by_input_ack += len(refs_to_release)
                    if lease.reserve_output_credit and lease.bytes > 0:
                        self._output_credits[lease_id] = self._output_credits.get(lease_id, 0) + lease.bytes
                        self._output_credit_bytes += lease.bytes
                        _shm_debug_log(
                            "output_credit_reserve",
                            name=name or lease.name or "-",
                            lease_id=lease_id,
                            size=lease.bytes,
                            output_credit_bytes=self._output_credit_bytes,
                            reserved_bytes=self._allocated_bytes,
                            pending_output_bytes=self._output_grant_bytes,
                            input_lease_bytes=self._input_lease_bytes,
                            limit_bytes=self._limit_locked(),
                        )
                release_bytes = lease.bytes
                _shm_debug_log(
                    "input_lease_release",
                    name=name or lease.name or "-",
                    lease_id=lease_id,
                    state=state,
                    size=release_bytes,
                    ref_count=len(refs),
                    release_ref_count=len(refs_to_release),
                    input_lease_bytes=self._input_lease_bytes,
                )
        if lease is None:
            if notify_budget_waiters:
                _notify_local_shm_budget_wakeup_callbacks()
            return 0
        for ref in refs_to_release:
            self._release_input_ack_ref(ref)
        with self._cond:
            self._cond.notify_all()
        _notify_local_shm_budget_wakeup_callbacks()
        return release_bytes

    def _release_input_ack_ref(self, ref: Any) -> None:
        release_budget = getattr(ref, "release_budget", None)
        if release_budget is not None:
            release_budget()
            return
        release = getattr(ref, "release", None)
        if release is None:
            raise TypeError("local shared-memory input ref must expose release_budget() or release()")
        release()

    def _input_ref_key_locked(self, ref: Any) -> tuple[str, Any]:
        name = getattr(ref, "name", None)
        if name:
            return ("local_shm", str(name))
        return ("object", id(ref))

    def _retain_input_ref_locked(self, ref: Any, *, lease_id: int) -> None:
        key = self._input_ref_key_locked(ref)
        hold = self._input_ref_holds.get(key)
        if hold is None:
            hold = _InputRefHold(
                key=key,
                name=str(getattr(ref, "name", "") or ""),
                size=max(0, int(getattr(ref, "size", 0) or 0)),
            )
            self._input_ref_holds[key] = hold
        hold.count += 1
        hold.refs[id(ref)] = ref
        _shm_debug_log(
            "input_ref_retain",
            name=hold.name or "-",
            lease_id=lease_id,
            ref_key=f"{key[0]}:{key[1]}",
            ref_count=hold.count,
            active_input_ref_holds=len(self._input_ref_holds),
        )

    def _release_input_refs_locked(self, refs: tuple[Any, ...], *, lease_id: int, state: str) -> tuple[Any, ...]:
        refs_to_release: list[Any] = []
        for ref in refs:
            key = self._input_ref_key_locked(ref)
            hold = self._input_ref_holds.get(key)
            if hold is None:
                refs_to_release.append(ref)
                continue
            hold.count = max(0, hold.count - 1)
            _shm_debug_log(
                "input_ref_release_decrement",
                name=hold.name or "-",
                lease_id=lease_id,
                state=state,
                ref_key=f"{key[0]}:{key[1]}",
                ref_count=hold.count,
                active_input_ref_holds=len(self._input_ref_holds),
            )
            if hold.count > 0:
                continue
            self._input_ref_holds.pop(key, None)
            refs_to_release.extend(hold.refs.values())
            _shm_debug_log(
                "input_ref_release_ready",
                name=hold.name or "-",
                lease_id=lease_id,
                state=state,
                ref_key=f"{key[0]}:{key[1]}",
                release_ref_count=len(hold.refs),
                active_input_ref_holds=len(self._input_ref_holds),
            )
        return tuple(refs_to_release)

    def _release_output_credit_locked(self, lease_id: int, *, name: str = "") -> int:
        credit = max(0, int(self._output_credits.pop(int(lease_id), 0) or 0))
        if credit <= 0:
            return 0
        self._output_credit_bytes = max(0, self._output_credit_bytes - credit)
        _shm_debug_log(
            "output_credit_release",
            name=name or f"input-lease-{lease_id}",
            lease_id=int(lease_id),
            size=credit,
            output_credit_bytes=self._output_credit_bytes,
            reserved_bytes=self._allocated_bytes,
            pending_output_bytes=self._output_grant_bytes,
            input_lease_bytes=self._input_lease_bytes,
            limit_bytes=self._limit_locked(),
        )
        return credit

    def claim_pending_output(
        self,
        size: int,
        *,
        name: str = "",
        cancel_event: threading.Event | None = None,
    ) -> int:
        requested = max(0, int(size))
        if requested <= 0:
            return 0
        with self._cond:

            def _can_claim_locked() -> bool:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError(f"local_shm output budget admission cancelled: {name or '-'}")
                limit = self._limit_locked()
                if limit <= 0:
                    return True
                usage = self._usage_locked()
                return usage <= 0 or usage + requested <= limit

            if not _can_claim_locked():
                limit = self._limit_locked()
                _shm_debug_log(
                    "budget_admission_wait",
                    name=name or "-",
                    size=requested,
                    reserved_bytes=self._allocated_bytes,
                    pending_output_bytes=self._output_grant_bytes,
                    input_lease_bytes=self._input_lease_bytes,
                    limit_bytes=limit,
                )
                self._cond.wait_for(_can_claim_locked)
            if self._limit_locked() <= 0:
                return 0
            self._output_grant_bytes += requested
            _shm_debug_log(
                "budget_admission_acquire",
                name=name or "-",
                size=requested,
                reserved_bytes=self._allocated_bytes,
                pending_output_bytes=self._output_grant_bytes,
                input_lease_bytes=self._input_lease_bytes,
                limit_bytes=self._limit_locked(),
            )
            return requested

    def release_pending_output(self, size: int, *, name: str = "") -> None:
        released = max(0, int(size))
        if released <= 0:
            return
        with self._cond:
            self._output_grant_bytes = max(0, self._output_grant_bytes - released)
            _shm_debug_log(
                "budget_admission_release",
                name=name or "-",
                size=released,
                reserved_bytes=self._allocated_bytes,
                pending_output_bytes=self._output_grant_bytes,
                input_lease_bytes=self._input_lease_bytes,
                limit_bytes=self._limit_locked(),
            )
            self._cond.notify_all()
        _notify_local_shm_budget_wakeup_callbacks()

    def request_output_grant(
        self,
        size: int,
        *,
        name: str = "",
        priority: str = "producer",
        input_lease_id: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> int:
        requested = max(0, int(size))
        if requested <= 0:
            return 0
        lease_id = int(input_lease_id) if input_lease_id is not None else None
        with self._cond:
            self._waiting_output_grants += 1
            try:

                def _grant_state_locked() -> tuple[int, int, int, int, bool]:
                    limit = self._limit_locked()
                    input_credit = self._output_credits.get(lease_id, 0) if lease_id is not None else 0
                    admission_usage = self._output_grant_admission_usage_locked(input_credit)
                    required_usage = admission_usage + requested
                    oversized_allowed = limit > 0 and requested > limit and admission_usage == 0
                    return limit, input_credit, admission_usage, required_usage, oversized_allowed

                def _can_grant_locked() -> bool:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError(f"local_shm output grant cancelled: {name or '-'}")
                    limit, _, _, required_usage, oversized_allowed = _grant_state_locked()
                    return limit <= 0 or required_usage <= limit or oversized_allowed

                if not _can_grant_locked():
                    limit, input_credit, _, _, _ = _grant_state_locked()
                    _shm_debug_log(
                        "output_grant_wait",
                        name=name or "-",
                        size=requested,
                        priority=priority,
                        input_lease_id=lease_id if lease_id is not None else "-",
                        input_credit_bytes=input_credit,
                        output_credit_bytes=self._output_credit_bytes,
                        reserved_bytes=self._allocated_bytes,
                        pending_output_bytes=self._output_grant_bytes,
                        input_lease_bytes=self._input_lease_bytes,
                        limit_bytes=limit,
                    )
                    self._cond.wait_for(_can_grant_locked)
                limit, input_credit, _, _, oversized_allowed = _grant_state_locked()
                grant_id = next(self._grant_ids)
                credit_released = 0
                credit_used = 0
                if lease_id is not None and input_credit > 0:
                    credit_released = self._release_output_credit_locked(lease_id, name=name or "-")
                    credit_used = min(requested, credit_released)
                self._output_grants[grant_id] = _OutputGrant(
                    grant_id=grant_id,
                    bytes=requested,
                    name=name or f"output-grant-{grant_id}",
                    priority=priority,
                )
                self._output_grant_bytes += requested
                if oversized_allowed:
                    self._oversized_output_grants += 1
                _shm_debug_log(
                    "output_grant_acquire",
                    name=name or "-",
                    grant_id=grant_id,
                    size=requested,
                    priority=priority,
                    input_lease_id=lease_id if lease_id is not None else "-",
                    input_credit_bytes=input_credit,
                    output_credit_used_bytes=credit_used,
                    output_credit_released_bytes=credit_released,
                    reserved_bytes=self._allocated_bytes,
                    pending_output_bytes=self._output_grant_bytes,
                    output_credit_bytes=self._output_credit_bytes,
                    input_lease_bytes=self._input_lease_bytes,
                    limit_bytes=limit,
                )
                if credit_released > 0:
                    self._cond.notify_all()
                return grant_id
            finally:
                self._waiting_output_grants = max(0, self._waiting_output_grants - 1)

    def convert_output_grant_to_allocation(self, grant_id: int, *, name: str = "") -> int:
        with self._cond:
            grant = self._output_grants.pop(int(grant_id), None)
            if grant is None:
                return 0
            self._output_grant_bytes = max(0, self._output_grant_bytes - grant.bytes)
            self._allocated_bytes += grant.bytes
            _shm_debug_log(
                "output_grant_convert",
                name=name or grant.name or "-",
                grant_id=grant.grant_id,
                size=grant.bytes,
                reserved_bytes=self._allocated_bytes,
                pending_output_bytes=self._output_grant_bytes,
                input_lease_bytes=self._input_lease_bytes,
            )
            self._cond.notify_all()
            return grant.bytes

    def release_output_grant(self, grant_id: int, *, name: str = "") -> int:
        with self._cond:
            grant = self._output_grants.pop(int(grant_id), None)
            if grant is None:
                return 0
            self._output_grant_bytes = max(0, self._output_grant_bytes - grant.bytes)
            _shm_debug_log(
                "output_grant_release",
                name=name or grant.name or "-",
                grant_id=grant.grant_id,
                size=grant.bytes,
                pending_output_bytes=self._output_grant_bytes,
                input_lease_bytes=self._input_lease_bytes,
            )
            self._cond.notify_all()
            released = grant.bytes
        _notify_local_shm_budget_wakeup_callbacks()
        return released

    def wake_waiters(self) -> None:
        with self._cond:
            self._cond.notify_all()
        _notify_local_shm_budget_wakeup_callbacks()


_LOCAL_SHM_BUDGET_MANAGER = LocalShmBudgetManager()


def local_shm_budget_manager() -> LocalShmBudgetManager:
    return _LOCAL_SHM_BUDGET_MANAGER


def create_local_shm_input_lease(
    refs: tuple[Any, ...] | list[Any],
    *,
    name: str = "",
    owner_operator_id: str = "",
    consumer_operator_id: str = "",
    submit_id: int | None = None,
    reserve_output_credit: bool = True,
) -> int:
    size = 0
    for ref in refs:
        size += max(0, int(getattr(ref, "size", 0) or 0))
    return _LOCAL_SHM_BUDGET_MANAGER.create_input_lease(
        refs,
        size,
        name=name,
        owner_operator_id=owner_operator_id,
        consumer_operator_id=consumer_operator_id,
        submit_id=submit_id,
        reserve_output_credit=reserve_output_credit,
    )


def consume_local_shm_input_lease(lease_id: int, *, name: str = "") -> int:
    return _LOCAL_SHM_BUDGET_MANAGER.consume_input_lease(lease_id, name=name)


def cancel_local_shm_input_lease(lease_id: int, *, name: str = "") -> int:
    return _LOCAL_SHM_BUDGET_MANAGER.cancel_input_lease(lease_id, name=name)


def request_local_shm_output_grant(
    size: int,
    *,
    name: str = "",
    priority: str = "producer",
    input_lease_id: int | None = None,
    cancel_event: threading.Event | None = None,
) -> int:
    return _LOCAL_SHM_BUDGET_MANAGER.request_output_grant(
        size,
        name=name,
        priority=priority,
        input_lease_id=input_lease_id,
        cancel_event=cancel_event,
    )


def convert_local_shm_output_grant_to_allocation(grant_id: int, *, name: str = "") -> int:
    return _LOCAL_SHM_BUDGET_MANAGER.convert_output_grant_to_allocation(grant_id, name=name)


def release_local_shm_output_grant(grant_id: int, *, name: str = "") -> int:
    return _LOCAL_SHM_BUDGET_MANAGER.release_output_grant(grant_id, name=name)


def local_shm_ref_lifecycle_snapshot() -> dict[str, int]:
    budget = _LOCAL_SHM_BUDGET_MANAGER.snapshot()
    with _local_shm_budget_cond:
        return {
            "created": _local_shm_refs_created,
            "released": _local_shm_refs_released,
            "live": max(0, _local_shm_refs_created - _local_shm_refs_released),
            "reserved_bytes": int(budget.get("reserved_bytes", 0)),
            "pending_output_bytes": int(budget.get("pending_output_bytes", 0)),
            "input_lease_bytes": int(budget.get("input_lease_bytes", 0)),
        }


def local_shm_ref_budget_snapshot() -> dict[str, int]:
    return _LOCAL_SHM_BUDGET_MANAGER.snapshot()


def can_claim_local_shm_ref_output_budget(size: int) -> bool:
    return _LOCAL_SHM_BUDGET_MANAGER.can_claim_output(size)


def can_admit_local_shm_ref_output_submit(size: int, *, projected_output_bytes: int = 0) -> bool:
    return _LOCAL_SHM_BUDGET_MANAGER.can_admit_output_submit(
        size,
        projected_output_bytes=projected_output_bytes,
    )


def _acquire_local_shm_ref_budget(size: int, *, name: str = "", block: bool = True) -> int:
    requested = max(0, int(size))
    if requested <= 0:
        return 0
    return _LOCAL_SHM_BUDGET_MANAGER.acquire_allocation(requested, name=name, block=block)


def wake_local_shm_ref_budget_waiters() -> None:
    _LOCAL_SHM_BUDGET_MANAGER.wake_waiters()


def claim_local_shm_ref_output_budget(
    size: int,
    *,
    name: str = "",
    cancel_event: threading.Event | None = None,
) -> int:
    requested = max(0, int(size))
    if requested <= 0:
        return 0
    return _LOCAL_SHM_BUDGET_MANAGER.claim_pending_output(requested, name=name, cancel_event=cancel_event)


def release_local_shm_ref_output_budget(size: int, *, name: str = "") -> None:
    released = max(0, int(size))
    if released <= 0:
        return
    _LOCAL_SHM_BUDGET_MANAGER.release_pending_output(released, name=name)


def _release_local_shm_ref_budget(size: int, *, name: str = "") -> None:
    released = max(0, int(size))
    if released <= 0:
        return
    _LOCAL_SHM_BUDGET_MANAGER.release_allocation(released, name=name)


def _arrow_table_to_ipc_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _arrow_table_from_ipc_bytes(data: bytes) -> pa.Table:
    reader = pa.ipc.open_stream(data)
    return reader.read_all()


def _write_ipc_to_shm(shm: shared_memory.SharedMemory, ipc_bytes: bytes) -> int:
    required = _IPC_HEADER_SIZE + len(ipc_bytes)
    if required > len(shm.buf):
        raise BufferError("shared memory segment is too small for Arrow IPC payload")
    shm.buf[:_IPC_HEADER_SIZE] = len(ipc_bytes).to_bytes(_IPC_HEADER_SIZE, "little")
    shm.buf[_IPC_HEADER_SIZE:required] = ipc_bytes
    return required


def _read_ipc_from_shm(shm: shared_memory.SharedMemory, size: int | None = None) -> bytes:
    if len(shm.buf) < _IPC_HEADER_SIZE:
        raise BufferError("shared memory segment is too small for Arrow IPC header")
    ipc_size = int.from_bytes(shm.buf[:_IPC_HEADER_SIZE], "little")
    required = _IPC_HEADER_SIZE + ipc_size
    if required > len(shm.buf):
        raise BufferError(
            f"shared memory IPC payload exceeds local mapping: required={required} capacity={len(shm.buf)}"
        )
    if size is not None and required > size:
        raise BufferError(f"shared memory IPC payload exceeds descriptor size: required={required} size={size}")
    return bytes(shm.buf[_IPC_HEADER_SIZE:required])


def _ipc_payload_bounds(shm: shared_memory.SharedMemory, size: int | None = None) -> tuple[int, int]:
    if len(shm.buf) < _IPC_HEADER_SIZE:
        raise BufferError("shared memory segment is too small for Arrow IPC header")
    ipc_size = int.from_bytes(shm.buf[:_IPC_HEADER_SIZE], "little")
    required = _IPC_HEADER_SIZE + ipc_size
    if required > len(shm.buf):
        raise BufferError(
            f"shared memory IPC payload exceeds local mapping: required={required} capacity={len(shm.buf)}"
        )
    if size is not None and required > size:
        raise BufferError(f"shared memory IPC payload exceeds descriptor size: required={required} size={size}")
    return _IPC_HEADER_SIZE, required


def _close_or_defer_shm(shm: shared_memory.SharedMemory) -> None:
    try:
        shm.close()
    except BufferError:
        _shm_debug_log("close_deferred", name=getattr(shm, "name", "-"), size=len(getattr(shm, "buf", b"")))
        with _deferred_shm_close_lock:
            _deferred_shm_closes.append(shm)


def _retry_deferred_shm_closes() -> None:
    with _deferred_shm_close_lock:
        if not _deferred_shm_closes:
            return
        handles = list(_deferred_shm_closes)
        _deferred_shm_closes.clear()

    still_open: list[shared_memory.SharedMemory] = []
    for shm in handles:
        try:
            shm.close()
        except BufferError:
            still_open.append(shm)
    if still_open:
        with _deferred_shm_close_lock:
            _deferred_shm_closes.extend(still_open)


class _LocalShmBufferOwner:
    """Own a shm mapping through the lifetime of a PyArrow foreign buffer."""

    def __init__(self, shm: shared_memory.SharedMemory) -> None:
        self._shm: shared_memory.SharedMemory | None = shm
        self._closed = False
        self._lock = threading.Lock()

    def close(self) -> None:
        shm: shared_memory.SharedMemory | None
        with self._lock:
            if self._closed or self._shm is None:
                return
            shm = self._shm
            self._shm = None
            self._closed = True
        _close_or_defer_shm(shm)

    def __del__(self) -> None:
        self.close()


def _arrow_table_from_local_shm_zero_copy(name: str, size: int) -> pa.Table:
    _retry_deferred_shm_closes()
    _shm_debug_log("materialize_open", name=name, size=size)
    shm = _open_existing_shm(name, track=False)
    owner: _LocalShmBufferOwner | None = None
    try:
        start, end = _ipc_payload_bounds(shm, size)
        address = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf, start))
        owner = _LocalShmBufferOwner(shm)
        buffer = pa.foreign_buffer(address, end - start, base=owner)
        table = pa.ipc.open_stream(pa.BufferReader(buffer)).read_all()
        _shm_debug_log("materialize_done", name=name, size=size, rows=table.num_rows)
        return table
    except Exception:
        if owner is not None:
            owner.close()
        else:
            _close_or_defer_shm(shm)
        raise


def _unlink_shared_memory_name(name: str) -> None:
    if not name:
        return
    try:
        shm = _open_existing_shm(name, track=False)
    except FileNotFoundError:
        return
    try:
        _unlink_shm(shm, track=False)
    finally:
        shm.close()


def _untrack_shm(shm: shared_memory.SharedMemory) -> shared_memory.SharedMemory:
    name = shm._name
    if not name:
        shm.close()
        raise RuntimeError("shared memory handle is missing its POSIX name")
    _resource_tracker.unregister(name, "shared_memory")
    return shm


def _create_shm(size: int, *, track: bool) -> shared_memory.SharedMemory:
    shm = shared_memory.SharedMemory(create=True, size=size)
    return shm if track else _untrack_shm(shm)


def _open_existing_shm(name: str, *, track: bool) -> shared_memory.SharedMemory:
    shm = shared_memory.SharedMemory(name=name)
    return shm if track else _untrack_shm(shm)


def _unlink_shm(shm: shared_memory.SharedMemory, *, track: bool) -> None:
    if track:
        shm.unlink()
        return
    shared_memory._posixshmem.shm_unlink(shm._name)


class LocalShmBlockRef:
    """Opaque local shared-memory ref held by C++ LazyRefDataChunk descriptors."""

    def __init__(
        self,
        name: str,
        size: int,
        *,
        owner: bool = True,
        shm: shared_memory.SharedMemory | None = None,
        budget_bytes: int | None = None,
        track: bool = False,
    ) -> None:
        self.name = str(name)
        self.size = int(size)
        self.owner = bool(owner)
        self._shm = shm
        self._track = bool(track)
        self._closed = False
        if not self.owner:
            self._budget_bytes = 0
        elif budget_bytes is None:
            self._budget_bytes = _acquire_local_shm_ref_budget(self.size, name=self.name)
        else:
            self._budget_bytes = max(0, int(budget_bytes))
        self._finalizer = weakref.finalize(
            self,
            _cleanup_local_shm_ref,
            self._shm,
            self.name,
            self.owner,
            self._budget_bytes,
            self._track,
        )
        if self.owner:
            global _local_shm_refs_created
            with _local_shm_budget_cond:
                _local_shm_refs_created += 1

    def to_table(self) -> pa.Table:
        if self._closed:
            raise RuntimeError(f"local shared-memory ref '{self.name}' is already released")
        return _arrow_table_from_local_shm_zero_copy(self.name, self.size)

    def release_budget(self) -> int:
        if self._closed:
            return 0
        budget_bytes = max(0, int(getattr(self, "_budget_bytes", 0) or 0))
        if budget_bytes <= 0:
            return 0
        self._budget_bytes = 0
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None and finalizer.alive:
            detached = finalizer.detach()
            if detached is not None:
                self._finalizer = weakref.finalize(
                    self,
                    _cleanup_local_shm_ref,
                    self._shm,
                    self.name,
                    self.owner,
                    0,
                    self._track,
                )
        _shm_debug_log("budget_detach", name=self.name, owner=self.owner, size=budget_bytes)
        _release_local_shm_ref_budget(budget_bytes, name=self.name)
        return budget_bytes

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"LocalShmBlockRef(name={self.name!r}, size={self.size}, owner={self.owner})"


def _cleanup_local_shm_ref(
    shm: shared_memory.SharedMemory | None,
    name: str,
    owner: bool,
    budget_bytes: int = 0,
    track: bool = False,
) -> None:
    cleanup_error: BaseException | None = None
    try:
        _shm_debug_log("release_start", name=name, owner=owner, has_shm=shm is not None)
        if owner:
            try:
                if shm is not None:
                    _unlink_shm(shm, track=track)
                else:
                    _unlink_shared_memory_name(name)
                _shm_debug_log("release_unlink", name=name, owner=owner)
            except FileNotFoundError:
                _shm_debug_log("release_missing", name=name, owner=owner)
            except BaseException as exc:
                cleanup_error = exc
        if shm is not None:
            try:
                shm.close()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
    finally:
        if owner:
            global _local_shm_refs_released
            with _local_shm_budget_cond:
                _local_shm_refs_released += 1
        _release_local_shm_ref_budget(budget_bytes, name=name)
    if cleanup_error is not None:
        raise cleanup_error


def make_local_shm_ref_bundle_result(table: pa.Table):
    table = _ensure_table(table)
    ipc_bytes = _arrow_table_to_ipc_bytes(table)
    required = _IPC_HEADER_SIZE + len(ipc_bytes)
    budget_bytes = _acquire_local_shm_ref_budget(required, name="local-shm-result")
    shm = None
    try:
        shm = _create_shm(required, track=False)
        _write_ipc_to_shm(shm, ipc_bytes)
        ref = LocalShmBlockRef(
            shm.name,
            required,
            owner=True,
            shm=shm,
            budget_bytes=budget_bytes,
            track=False,
        )
        budget_bytes = 0
        _shm_debug_log("create_result", name=shm.name, size=required, rows=table.num_rows, nbytes=table.nbytes)
    except Exception:
        if shm is not None:
            try:
                shm.close()
            finally:
                try:
                    _unlink_shm(shm, track=False)
                except FileNotFoundError:
                    pass
        _release_local_shm_ref_budget(budget_bytes, name="local-shm-result-create-failed")
        raise

    metadata = {
        "provider": LOCAL_SHM_PROVIDER,
        "num_rows": int(table.num_rows),
        "size_bytes": int(estimate_table_bytes(table)),
        "ipc_size_bytes": int(required),
        "shm_name": shm.name,
    }
    return (
        REF_BUNDLE_RESULT_MARKER,
        [ref],
        [metadata],
        list(table.schema.names),
    )


def make_local_shm_ref_bundle_descriptor(table: pa.Table, *, grant_id: int | None = None) -> dict[str, Any]:
    """Create a worker-safe local shm descriptor for a single Arrow table block."""
    table = _ensure_table(table)
    ipc_bytes = _arrow_table_to_ipc_bytes(table)
    required = _IPC_HEADER_SIZE + len(ipc_bytes)
    shm = _create_shm(required, track=False)
    try:
        _write_ipc_to_shm(shm, ipc_bytes)
        metadata = {
            "provider": LOCAL_SHM_PROVIDER,
            "num_rows": int(table.num_rows),
            "size_bytes": int(estimate_table_bytes(table)),
            "ipc_size_bytes": int(required),
            "shm_name": shm.name,
        }
        descriptor = {
            "block_refs": [
                {
                    "provider": LOCAL_SHM_PROVIDER,
                    "shm_name": shm.name,
                    "ipc_size_bytes": int(required),
                }
            ],
            "metadata": [metadata],
            "names": list(table.schema.names),
        }
        if grant_id is not None:
            descriptor["grant_id"] = int(grant_id)
        _shm_debug_log("create_descriptor", name=shm.name, size=required, rows=table.num_rows, nbytes=table.nbytes)
        return descriptor
    except Exception:
        try:
            _unlink_shm(shm, track=False)
        except FileNotFoundError:
            pass
        raise
    finally:
        shm.close()


def release_local_shm_ref_bundle_descriptor(descriptor: dict[str, Any]) -> None:
    """Release local shm blocks described by a worker-safe descriptor.

    The descriptor only owns shm names. Output grants are owned by the caller and
    must be released separately if descriptor creation fails before handoff.
    """
    for ref_desc in list((descriptor or {}).get("block_refs") or []):
        local_desc = _local_shm_descriptor_from_mapping(ref_desc)
        if local_desc is None:
            continue
        _unlink_shared_memory_name(str(local_desc["shm_name"]))


def estimate_local_shm_ref_bundle_ipc_size(value: Any) -> int:
    if isinstance(value, dict):
        refs = list(value.get("block_refs") or [])
        metadata = list(value.get("metadata") or [])
        total = 0
        for ref_desc in refs:
            local_desc = _local_shm_descriptor_from_mapping(ref_desc)
            if local_desc is not None:
                total += int(local_desc["ipc_size_bytes"])
        if total <= 0:
            for meta in metadata:
                if isinstance(meta, dict) and meta.get("ipc_size_bytes") is not None:
                    total += int(meta["ipc_size_bytes"])
        return total

    if not (isinstance(value, tuple) and len(value) >= 4 and value[0] == REF_BUNDLE_RESULT_MARKER):
        return 0

    refs = list(value[1] or [])
    metadata = list(value[2] or [])
    total = 0
    for ref, meta in zip(refs, metadata, strict=False):
        if isinstance(meta, dict) and meta.get("ipc_size_bytes") is not None:
            total += int(meta["ipc_size_bytes"])
        elif isinstance(ref, LocalShmBlockRef):
            total += int(ref.size)
    return total


def make_local_shm_ref_bundle_result_from_descriptor(
    descriptor: dict[str, Any],
    *,
    block_on_budget: bool = True,
):
    refs_in = list(descriptor.get("block_refs") or [])
    metadata_in = list(descriptor.get("metadata") or [{} for _ in refs_in])
    if len(refs_in) != len(metadata_in):
        raise ValueError(
            f"local_shm descriptor ref/metadata length mismatch: refs={len(refs_in)} metadata={len(metadata_in)}"
        )
    local_descs = []
    for ref_desc in refs_in:
        local_desc = _local_shm_descriptor_from_mapping(ref_desc, strict=True)
        assert local_desc is not None
        local_descs.append(local_desc)

    grant_id_raw = descriptor.get("grant_id")
    grant_id = int(grant_id_raw) if grant_id_raw is not None else None
    refs = []
    metadata = []
    grant_budget_remaining: int | None = None
    try:
        for local_desc, meta in zip(local_descs, metadata_in, strict=False):
            name = str(local_desc["shm_name"])
            size = int(local_desc["ipc_size_bytes"])
            if grant_id is not None:
                grant_budget_remaining = convert_local_shm_output_grant_to_allocation(grant_id, name=name)
                if grant_budget_remaining <= 0:
                    raise RuntimeError(f"local_shm output grant {grant_id} is not active")
                grant_id = None
            if grant_budget_remaining is not None:
                budget_bytes = min(size, grant_budget_remaining)
                grant_budget_remaining -= budget_bytes
                if budget_bytes < size:
                    try:
                        budget_bytes += _acquire_local_shm_ref_budget(
                            size - budget_bytes,
                            name=name,
                            block=block_on_budget,
                        )
                    except Exception:
                        _release_local_shm_ref_budget(budget_bytes, name=name)
                        raise
            else:
                budget_bytes = _acquire_local_shm_ref_budget(size, name=name, block=block_on_budget)
            shm = None
            try:
                shm = _open_existing_shm(name, track=False)
                refs.append(
                    LocalShmBlockRef(
                        name,
                        size,
                        owner=True,
                        shm=shm,
                        budget_bytes=budget_bytes,
                        track=False,
                    )
                )
            except Exception:
                _release_local_shm_ref_budget(budget_bytes, name=name)
                if shm is not None:
                    try:
                        shm.close()
                    except Exception:
                        pass
                raise
            merged_meta = dict(meta or {})
            _shm_debug_log(
                "wrap_descriptor",
                name=name,
                size=size,
                rows=merged_meta.get("num_rows", "-"),
            )
            merged_meta.setdefault("provider", LOCAL_SHM_PROVIDER)
            merged_meta.setdefault("shm_name", name)
            merged_meta.setdefault("ipc_size_bytes", size)
            metadata.append(merged_meta)
    except Exception:
        for ref in refs:
            ref.release()
        if grant_budget_remaining is not None and grant_budget_remaining > 0:
            _release_local_shm_ref_budget(grant_budget_remaining, name="local-shm-descriptor-grant-unused")
        raise

    if grant_budget_remaining is not None and grant_budget_remaining > 0:
        _release_local_shm_ref_budget(grant_budget_remaining, name="local-shm-descriptor-grant-unused")

    return (
        REF_BUNDLE_RESULT_MARKER,
        refs,
        metadata,
        list(descriptor.get("names") or []),
    )


def payload_requests_local_ref_bundle_output(payload: dict[str, Any]) -> bool:
    produce = bool(payload.get("produce_ref_bundle_output", False))
    mode = str(payload.get("streaming_output_mode") or "").strip().lower()
    if mode and mode != "local_shm_ref_bundle":
        raise RuntimeError(
            f"local subprocess distributed output requires streaming_output_mode='local_shm_ref_bundle', got {mode!r}"
        )
    if produce and not mode:
        raise RuntimeError("local subprocess distributed output requires streaming_output_mode='local_shm_ref_bundle'")
    if mode and not produce:
        raise RuntimeError("local subprocess distributed output requires produce_ref_bundle_output=True")
    return produce and mode == "local_shm_ref_bundle"


def _local_shm_descriptor_from_mapping(value: dict[str, Any], *, strict: bool = False) -> dict[str, Any] | None:
    if value.get("provider") != LOCAL_SHM_PROVIDER:
        return None
    name = value.get("shm_name")
    if not name:
        if strict:
            raise ValueError("local_shm ref mapping is missing shm_name")
        return None
    size = value.get("ipc_size_bytes")
    if size is None:
        if strict:
            raise ValueError("local_shm ref mapping is missing ipc_size_bytes")
        return None
    return {
        "provider": LOCAL_SHM_PROVIDER,
        "shm_name": str(name),
        "ipc_size_bytes": int(size),
    }


def _local_shm_ref_from_mapping(value: dict[str, Any]) -> LocalShmBlockRef | None:
    descriptor = _local_shm_descriptor_from_mapping(value, strict=True)
    if descriptor is None:
        return None
    return LocalShmBlockRef(str(descriptor["shm_name"]), int(descriptor["ipc_size_bytes"]), owner=False)


def _local_shm_descriptor_from_ref(ref: Any, meta: Any | None = None) -> dict[str, Any] | None:
    if isinstance(ref, LocalShmBlockRef):
        if getattr(ref, "_closed", False):
            raise RuntimeError(f"local shared-memory ref '{ref.name}' is already released")
        return {
            "provider": LOCAL_SHM_PROVIDER,
            "shm_name": ref.name,
            "ipc_size_bytes": ref.size,
        }
    if isinstance(ref, dict):
        descriptor = _local_shm_descriptor_from_mapping(ref)
        if descriptor is not None:
            return descriptor
    if isinstance(meta, dict):
        return _local_shm_descriptor_from_mapping(meta)
    return None


def _estimate_ref_bundle_num_rows(
    slices: list[Any] | tuple[Any, ...] | None,
    metadata: list[Any] | tuple[Any, ...] | None,
) -> int | None:
    if metadata is None:
        metadata = []
    if slices is None:
        total = 0
        for meta in metadata:
            if not isinstance(meta, dict) or meta.get("num_rows") is None:
                return None
            total += int(meta["num_rows"])
        return total

    total = 0
    for idx, slice_desc in enumerate(slices):
        if slice_desc is None:
            if idx >= len(metadata) or not isinstance(metadata[idx], dict) or metadata[idx].get("num_rows") is None:
                return None
            total += int(metadata[idx]["num_rows"])
            continue
        if isinstance(slice_desc, dict):
            start = int(slice_desc["start"])
            end = int(slice_desc["end"])
        else:
            start, end = slice_desc
            start = int(start)
            end = int(end)
        if start < 0 or end < start:
            return None
        total += end - start
    return total


def make_local_ref_bundle_worker_payload(
    block_refs: tuple[Any, ...] | list[Any],
    slices: list[Any] | tuple[Any, ...] | None = None,
    metadata: list[Any] | tuple[Any, ...] | None = None,
    names: list[str] | tuple[str, ...] | None = None,
    *,
    input_lease_id: int | None = None,
) -> dict[str, Any] | None:
    """Return a worker-safe descriptor payload when all blocks are local_shm."""
    refs = list(block_refs)
    if not refs:
        raise ValueError("empty ref bundle input is not supported")

    metadata_list = list(metadata or [{} for _ in refs])
    if len(metadata_list) != len(refs):
        raise ValueError(f"ref bundle ref/metadata length mismatch: refs={len(refs)} metadata={len(metadata_list)}")

    slices_list = list(slices) if slices is not None else None
    if slices_list is not None and len(slices_list) != len(refs):
        raise ValueError(f"ref bundle ref/slice length mismatch: refs={len(refs)} slices={len(slices_list)}")

    descriptors: list[dict[str, Any]] = []
    for ref, meta in zip(refs, metadata_list, strict=False):
        descriptor = _local_shm_descriptor_from_ref(ref, meta)
        if descriptor is None:
            return None
        descriptors.append(descriptor)

    payload = {
        "block_refs": descriptors,
        "slices": slices_list,
        "metadata": metadata_list,
        "names": list(names or []),
        "estimated_num_rows": _estimate_ref_bundle_num_rows(slices_list, metadata_list),
    }
    if input_lease_id is not None:
        payload["input_lease_id"] = int(input_lease_id)
    return payload


def _resolve_local_block(ref: Any, meta: Any | None = None) -> pa.Table | None:
    if isinstance(ref, pa.Table):
        return ref
    if isinstance(ref, pa.RecordBatch):
        return pa.Table.from_batches([ref])
    if isinstance(ref, LocalShmBlockRef):
        return ref.to_table()
    if isinstance(ref, dict):
        local_ref = _local_shm_ref_from_mapping(ref)
        if local_ref is not None:
            try:
                return local_ref.to_table()
            finally:
                local_ref.release()
    if isinstance(meta, dict) and meta.get("provider") == LOCAL_SHM_PROVIDER and meta.get("shm_name"):
        size = meta.get("ipc_size_bytes")
        if size is not None:
            local_ref = LocalShmBlockRef(str(meta["shm_name"]), int(size), owner=False)
            try:
                return local_ref.to_table()
            finally:
                local_ref.release()
    return None


def _is_ray_object_ref(ref: Any) -> bool:
    import ray

    return isinstance(ref, ray.ObjectRef)


def _resolve_ray_object_ref_blocks(refs: list[Any]) -> list[Any]:
    from vane.runners.ray.safe_get import resolve_object_refs_blocking

    resolved = resolve_object_refs_blocking(refs)
    if not isinstance(resolved, list | tuple):
        raise ValueError("ray ref bundle materialization returned a non-sequence result")
    if len(resolved) != len(refs):
        raise ValueError(f"ray ref bundle materialization length mismatch: refs={len(refs)} blocks={len(resolved)}")
    return list(resolved)


def _apply_ref_bundle_slices(
    blocks: tuple[Any, ...] | list[Any],
    slices: list[Any] | tuple[Any, ...] | None,
    metadata: list[Any] | tuple[Any, ...] | None = None,
    names: list[str] | tuple[str, ...] | None = None,
) -> pa.Table:
    if slices is None:
        slices = [None] * len(blocks)
    if metadata is None:
        metadata = [{} for _ in blocks]
    if len(blocks) != len(slices):
        raise ValueError(f"ref bundle block/slice length mismatch: blocks={len(blocks)} slices={len(slices)}")
    if len(blocks) != len(metadata):
        raise ValueError(f"ref bundle block/metadata length mismatch: blocks={len(blocks)} metadata={len(metadata)}")

    output_names = list(names or [])
    tables: list[pa.Table] = []
    for block_idx, (block, slice_desc, meta) in enumerate(zip(blocks, slices, metadata, strict=False)):
        table = _ensure_table(block)
        if isinstance(meta, dict) and meta.get("column_ids") is not None:
            column_ids = [int(column_id) for column_id in meta["column_ids"]]
            for column_id in column_ids:
                if column_id < 0 or column_id >= table.num_columns:
                    raise ValueError(f"invalid ref bundle column id {column_id} for table columns={table.num_columns}")
            table = table.select(column_ids)
        if output_names and len(output_names) != table.num_columns:
            raise ValueError(
                f"ref bundle names length {len(output_names)} does not match block {block_idx} columns={table.num_columns}"
            )
        if slice_desc is not None:
            if isinstance(slice_desc, dict):
                start = int(slice_desc["start"])
                end = int(slice_desc["end"])
            else:
                start, end = slice_desc
                start = int(start)
                end = int(end)
            if start < 0 or end < start or end > table.num_rows:
                raise ValueError(f"invalid ref bundle slice [{start}, {end}) for block rows={table.num_rows}")
            table = table.slice(start, end - start)
        tables.append(table)

    if not tables:
        raise ValueError("empty ref bundle input is not supported")
    result = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
    if output_names:
        if len(output_names) != result.num_columns:
            raise ValueError(
                f"ref bundle names length {len(output_names)} does not match result columns={result.num_columns}"
            )
        result = result.rename_columns(output_names)
    return result


def materialize_ref_bundle(
    block_refs: tuple[Any, ...] | list[Any],
    slices: list[Any] | tuple[Any, ...] | None = None,
    metadata: list[Any] | tuple[Any, ...] | None = None,
    names: list[str] | tuple[str, ...] | None = None,
) -> pa.Table:
    refs = list(block_refs)
    if not refs:
        raise ValueError("empty ref bundle input is not supported")

    metadata_list = list(metadata or [{} for _ in refs])
    if len(metadata_list) != len(refs):
        raise ValueError(f"ref bundle ref/metadata length mismatch: refs={len(refs)} metadata={len(metadata_list)}")

    local_blocks: list[pa.Table | None] = [None] * len(refs)
    ray_positions: list[int] = []
    ray_refs: list[Any] = []
    for idx, (ref, meta) in enumerate(zip(refs, metadata_list, strict=False)):
        block = _resolve_local_block(ref, meta)
        if block is None:
            if _is_ray_object_ref(ref):
                ray_positions.append(idx)
                ray_refs.append(ref)
                continue
            raise ValueError(
                f"unsupported ref bundle block at index {idx}: "
                "expected pyarrow table, record batch, local_shm ref, or Ray ObjectRef"
            )
        local_blocks[idx] = block

    if ray_refs:
        for idx, block in zip(ray_positions, _resolve_ray_object_ref_blocks(ray_refs), strict=True):
            local_block = _resolve_local_block(block, metadata_list[idx])
            if local_block is None:
                raise ValueError(
                    f"unsupported materialized Ray ref bundle block at index {idx}: "
                    "expected pyarrow table, record batch, or local_shm ref"
                )
            local_blocks[idx] = local_block

    resolved_blocks = [block for block in local_blocks if block is not None]
    if len(resolved_blocks) != len(refs):
        raise ValueError("ref bundle materialization did not resolve every block")

    return _apply_ref_bundle_slices(resolved_blocks, slices, metadata=metadata_list, names=names)


__all__ = [
    "LOCAL_SHM_PROVIDER",
    "REF_BUNDLE_RESULT_MARKER",
    "SUBMIT_RESULT_MARKER",
    "LocalShmBlockRef",
    "LocalShmBudgetManager",
    "can_admit_local_shm_ref_output_submit",
    "can_claim_local_shm_ref_output_budget",
    "cancel_local_shm_input_lease",
    "claim_local_shm_ref_output_budget",
    "consume_local_shm_input_lease",
    "convert_local_shm_output_grant_to_allocation",
    "create_local_shm_input_lease",
    "estimate_local_shm_ref_bundle_ipc_size",
    "local_shm_budget_manager",
    "local_shm_ref_budget_snapshot",
    "local_shm_ref_lifecycle_snapshot",
    "make_local_ref_bundle_worker_payload",
    "make_local_shm_ref_bundle_descriptor",
    "make_local_shm_ref_bundle_result",
    "make_local_shm_ref_bundle_result_from_descriptor",
    "materialize_ref_bundle",
    "payload_requests_local_ref_bundle_output",
    "register_local_shm_ref_budget_wakeup",
    "release_local_shm_output_grant",
    "release_local_shm_ref_bundle_descriptor",
    "release_local_shm_ref_output_budget",
    "request_local_shm_output_grant",
    "wake_local_shm_ref_budget_waiters",
]
