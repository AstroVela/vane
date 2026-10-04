# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared-directory objects and query-local, fenced commit decisions.

The directory must be shared by all participating processes and survive the
compute worker. This provider verifies store identity/visibility, not the
physical durability of a deployment's mount. StorePool owns automatic orphan
collection; coordinator recovery is outside the supported execution contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vane.execution.materialized_exchange import (
    AttemptManifest,
    AttemptToken,
    MaterializedTask,
    ObjectMeta,
    OutputObject,
    PartitionSpec,
    ResultManifest,
    StageManifest,
    _hex,
    _label,
    _object_key,
    canonical,
)
from vane.execution.plan import _fields, _items
from vane.execution.resource_demand import _capacity


class StorageCleanupPending(RuntimeError):
    """A native owner still holds storage; its reservation cannot be returned."""


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        handle = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    """Publish immutable metadata without replacing a competing publication."""
    payload = canonical(value)
    temporary = path.with_name(f".{uuid.uuid4().hex}.pending")
    try:
        handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(handle, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != payload:
                raise ValueError("conflicting immutable metadata publication") from None
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class StoreDescriptor:
    root: str
    store_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, str) or not Path(self.root).is_absolute():
            raise ValueError("shared store root must be absolute")
        _hex(self.store_id, 32, "store_id")

    def check(self) -> None:
        root = Path(self.root)
        identity = root / "store.json"
        if root.resolve() != root or identity.is_symlink():
            raise ValueError("shared store location changed")
        value = json.loads(identity.read_bytes())
        _fields(value, {"protocol", "store_id"}, "store identity")
        if type(value["protocol"]) is not int or value != {"protocol": 1, "store_id": self.store_id}:
            raise ValueError("shared store identity mismatch")

    def path(self, key: str) -> Path:
        _object_key(key)
        self.check()
        path = Path(self.root) / "queries" / key
        if path.resolve() != path:
            raise ValueError("materialized object path contains a symlink")
        return path

    def to_dict(self) -> dict[str, str]:
        return {"root": self.root, "store_id": self.store_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StoreDescriptor:
        _fields(value, {"root", "store_id"}, cls.__name__)
        return cls(**value)


class SharedDirectoryStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        path = Path(root)
        if not path.is_absolute():
            raise ValueError("shared store root must be absolute")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = path.resolve()
        marker = path / "store.json"
        if not marker.exists():
            candidate = {"protocol": 1, "store_id": uuid.uuid4().hex}
            try:
                _publish(marker, candidate)
            except ValueError:
                if not marker.exists():
                    raise
        value = json.loads(marker.read_bytes())
        _fields(value, {"protocol", "store_id"}, "store identity")
        if type(value["protocol"]) is not int or value["protocol"] != 1:
            raise ValueError("unsupported shared store protocol")
        self.descriptor = StoreDescriptor(str(path), value["store_id"])
        self.descriptor.check()
        queries = path / "queries"
        queries.mkdir(mode=0o700, exist_ok=True)
        if queries.is_symlink():
            raise ValueError("shared store query root contains a symlink")


@dataclass(frozen=True)
class ObjectReservation:
    output: PartitionSpec
    key: str
    max_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.output, PartitionSpec):
            raise ValueError("invalid reserved output")
        _object_key(self.key)
        _capacity(self.max_bytes, "max object bytes", minimum=40)
        if self.max_bytes > 1 << 40:
            raise ValueError("materialized objects are limited to 1 TiB")

    def to_dict(self) -> dict[str, Any]:
        return {"output": self.output.to_dict(), "key": self.key, "max_bytes": self.max_bytes}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ObjectReservation:
        _fields(value, {"output", "key", "max_bytes"}, cls.__name__)
        return cls(PartitionSpec.from_dict(value["output"]), value["key"], value["max_bytes"])


@dataclass(frozen=True)
class AttemptReservation:
    store: StoreDescriptor
    engine_identity: str
    token: AttemptToken
    objects: tuple[ObjectReservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.store, StoreDescriptor) or not isinstance(self.token, AttemptToken):
            raise ValueError("invalid attempt reservation")
        _label(self.engine_identity, "engine_identity")
        objects = _items(self.objects, "objects")
        if not 1 <= len(objects) <= 4096 or any(not isinstance(o, ObjectReservation) for o in objects):
            raise ValueError("invalid object reservations")
        if any(o.key.split("/")[1] != self.token.fence for o in objects):
            raise ValueError("object reservation has another attempt fence")
        if len({o.output.identity for o in objects}) != len(objects) or len({o.key for o in objects}) != len(objects):
            raise ValueError("duplicate output reservation")
        object.__setattr__(self, "objects", tuple(sorted(objects, key=lambda o: o.output.identity)))

    def seal(self, metadata: Mapping[str, ObjectMeta]) -> AttemptManifest:
        if set(metadata) != {o.key for o in self.objects}:
            raise ValueError("every reserved output, including empty partitions, must be sealed")
        objects = tuple(OutputObject(o.output, o.key, metadata[o.key]) for o in self.objects)
        if any(o.metadata.bytes > r.max_bytes for o, r in zip(objects, self.objects)):
            raise ValueError("sealed object exceeds its reservation")
        return AttemptManifest(self.engine_identity, self.token, objects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "store": self.store.to_dict(),
            "engine_identity": self.engine_identity,
            "token": self.token.to_dict(),
            "objects": [o.to_dict() for o in self.objects],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AttemptReservation:
        _fields(value, {"store", "engine_identity", "token", "objects"}, cls.__name__)
        return cls(
            StoreDescriptor.from_dict(value["store"]),
            value["engine_identity"],
            AttemptToken.from_dict(value["token"]),
            tuple(ObjectReservation.from_dict(o) for o in _items(value["objects"], "objects")),
        )


class ReadLease:
    def __init__(self, owner: CommitCoordinator, identity: str) -> None:
        self._owner = owner
        self._identity = identity

    def close(self) -> None:
        with self._owner._lock:
            self._owner._readers.discard(self._identity)

    def __enter__(self) -> ReadLease:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class CommitCoordinator:
    """One coordinator incarnation; commit, fencing and cancellation serialize.

    Call discard only after the attempt's native I/O has stopped, including after
    a worker exits. A read lease pins all committed query objects during delivery.
    Cleanup failures retain the directory and its full storage reservation.
    """

    def __init__(
        self,
        store: SharedDirectoryStore,
        query_id: str,
        engine_identity: str,
        tasks: tuple[MaterializedTask, ...],
        *,
        max_bytes: int,
        namespace: str | None = None,
    ) -> None:
        _label(query_id, "query_id")
        _label(engine_identity, "engine_identity")
        _capacity(max_bytes, "store capacity")
        task_list = _items(tasks, "tasks")
        if len(task_list) > 4096 or any(not isinstance(t, MaterializedTask) for t in task_list):
            raise ValueError("materialized query supports at most 4096 tasks")
        if len({t.task_id for t in task_list}) != len(task_list):
            raise ValueError("duplicate logical task")
        if sum(len(t.outputs) for t in task_list) > 4096:
            raise ValueError("materialized graph exceeds the partition metadata limit")
        self.store = store.descriptor
        self.store.check()
        self.query_id = query_id
        self.engine_identity = engine_identity
        self.max_bytes = max_bytes
        self.namespace = namespace or uuid.uuid4().hex
        _hex(self.namespace, 32, "query namespace")
        self._directory = Path(self.store.root) / "queries" / self.namespace
        self._tasks = {t.task_id: t for t in task_list}
        self._lock = threading.RLock()
        self._current: dict[str, AttemptToken] = {}
        self._counts: dict[str, int] = {}
        self._reservations: dict[str, AttemptReservation] = {}
        self._committed: dict[str, AttemptManifest] = {}
        self._stages: dict[str, StageManifest] = {}
        self._readers: set[str] = set()
        self._error = ""
        self._closing = False
        self._closed = False
        self._source_bytes = 0
        self.result: ResultManifest | None = None
        self._check_directory()
        self._directory.mkdir(mode=0o700, exist_ok=namespace is not None)

    def source_bytes(self, amount: int) -> None:
        _capacity(amount, "source bytes", minimum=0)
        with self._lock:
            self._active()
            if self._reservations or amount > self.max_bytes:
                raise ValueError("source storage must be admitted before attempts")
            self._source_bytes = amount

    def declare_stage(self, tasks: tuple[MaterializedTask, ...]) -> None:
        """Bind a stage to the now-committed upstream manifests exactly once."""
        tasks = _items(tasks, "stage tasks")
        if not tasks or any(not isinstance(t, MaterializedTask) for t in tasks):
            raise ValueError("stage requires logical tasks")
        if len({t.stage_id for t in tasks}) != 1 or len({t.task_id for t in tasks}) != len(tasks):
            raise ValueError("invalid stage task identities")
        with self._lock:
            self._active()
            if any(t.stage_id == tasks[0].stage_id for t in self._tasks.values()):
                raise ValueError("stage input identities are already fixed")
            if any(t.task_id in self._tasks for t in tasks):
                raise ValueError("logical task already declared")
            combined = tuple(self._tasks.values()) + tasks
            if len(combined) > 4096 or sum(len(t.outputs) for t in combined) > 4096:
                raise ValueError("materialized graph exceeds the partition metadata limit")
            self._tasks.update((t.task_id, t) for t in tasks)

    def _check_directory(self) -> None:
        self.store.check()
        if self._directory.resolve() != self._directory:
            raise ValueError("materialized query path contains a symlink")

    def _active(self) -> None:
        if self._error or self._closing:
            raise RuntimeError(self._error or "materialized query is closing")

    def _usage(self) -> int:
        return self._source_bytes + sum(o.max_bytes for r in self._reservations.values() for o in r.objects)

    def _publish_fence(self, task_id: str, fence: str | None) -> None:
        from vane.execution.fte_store import replace_metadata

        name = hashlib.sha256(task_id.encode()).hexdigest()
        replace_metadata(self._directory / f"task-{name}.json", {"fence": fence})

    def begin(self, task_id: str, worker_epoch: str, *, object_bytes: int) -> AttemptReservation:
        from vane._native import execution_runtime as native

        _capacity(object_bytes, "object reservation", minimum=40)
        with self._lock:
            self._active()
            task = self._tasks[task_id]
            if task_id in self._committed:
                raise ValueError("logical task already committed")
            if (
                sum(len(r.objects) for r in self._reservations.values()) + len(task.outputs) > 4096
                or self._usage() + object_bytes * len(task.outputs) > self.max_bytes
            ):
                raise RuntimeError("insufficient materialized storage capacity")
            token = AttemptToken(
                self.query_id,
                task.stage_id,
                task_id,
                self._counts.get(task_id, 0) + 1,
                uuid.uuid4().hex,
                worker_epoch,
                task.input_id,
            )
            objects = tuple(
                ObjectReservation(output, f"{self.namespace}/{token.fence}/{uuid.uuid4().hex}.mat", object_bytes)
                for output in task.outputs
            )
            reservation = AttemptReservation(self.store, self.engine_identity, token, objects)
            self._check_directory()
            (self._directory / token.fence).mkdir(mode=0o700)
            self._reservations[token.fence] = reservation
            self._counts[task_id] = token.attempt
            self._current[task_id] = token
            guard = native.StoreGuard.acquire(str(self._directory / token.fence / "io.lock"), True, True)
            assert guard is not None
            guard.close()
            self._publish_fence(task_id, token.fence)
            return reservation

    def _validate(self, manifest: AttemptManifest) -> AttemptManifest | None:
        self._active()
        if not isinstance(manifest, AttemptManifest) or manifest.engine_identity != self.engine_identity:
            raise ValueError("materialized manifest engine mismatch")
        token = manifest.token
        if self._current.get(token.task_id) != token:
            raise ValueError("stale or foreign attempt fence")
        previous = self._committed.get(token.task_id)
        if previous is not None:
            if previous != manifest:
                raise ValueError("conflicting duplicate attempt commit")
            return previous
        reserved = self._reservations[token.fence]
        if tuple(o.output for o in manifest.objects) != tuple(o.output for o in reserved.objects):
            raise ValueError("manifest does not seal the declared partitions and schemas")
        if any(o.key != r.key or o.metadata.bytes > r.max_bytes for o, r in zip(manifest.objects, reserved.objects)):
            raise ValueError("manifest object identity or reservation mismatch")
        return None

    def commit(self, manifest: AttemptManifest) -> AttemptManifest:
        from vane._native import execution_runtime as native

        with self._lock:
            previous = self._validate(manifest)
            if previous is not None:
                return previous
        # Storage work is outside the decision lock. Cancellation/new attempts
        # can fence this submission while verification is in progress.
        for obj in manifest.objects:
            native.MaterializedIO.verify(str(self.store.path(obj.key)), obj.metadata.to_dict(), obj.output.schema)
        _publish(self._directory / manifest.token.fence / "attempt.json", manifest.to_dict())
        with self._lock:
            previous = self._validate(manifest)
            if previous is None:
                self._committed[manifest.token.task_id] = manifest
            return previous or manifest

    def seal_stage(self, stage_id: str) -> StageManifest:
        with self._lock:
            self._active()
            expected = sorted(t.task_id for t in self._tasks.values() if t.stage_id == stage_id)
            if not expected or any(task not in self._committed for task in expected):
                raise RuntimeError("stage has uncommitted logical tasks")
            if stage_id in self._stages:
                return self._stages[stage_id]
            manifest = StageManifest(
                self.query_id, stage_id, self.engine_identity, tuple(self._committed[t] for t in expected)
            )
        name = hashlib.sha256(stage_id.encode()).hexdigest()
        self._check_directory()
        _publish(self._directory / f"stage-{name}.json", manifest.to_dict())
        with self._lock:
            self._active()
            return self._stages.setdefault(stage_id, manifest)

    def retain(self, stage: StageManifest) -> ReadLease:
        with self._lock:
            self._active()
            if self._stages.get(stage.stage_id) != stage:
                raise ValueError("read lease requires this query's committed stage")
            identity = uuid.uuid4().hex
            self._readers.add(identity)
            return ReadLease(self, identity)

    def publish_result(self, stage: StageManifest, names: tuple[str, ...]) -> ResultManifest:
        with self._lock:
            self._active()
            if self._stages.get(stage.stage_id) != stage:
                raise ValueError("result requires this query's sealed root stage")
            result = ResultManifest(stage, names)
            if self.result is not None:
                if self.result != result:
                    raise ValueError("conflicting root result publication")
                return self.result
        _publish(self._directory / "result.json", result.to_dict())
        with self._lock:
            self._active()
            self.result = result
            return result

    def cancel(self, reason: str = "materialized query canceled") -> None:
        with self._lock:
            self._error = self._error or reason or "materialized query canceled"

    def discard(self, token: AttemptToken) -> None:
        from vane._native import execution_runtime as native

        with self._lock:
            reserved = self._reservations.get(token.fence)
            if reserved is None:
                return
            if reserved.token != token:
                raise ValueError("foreign attempt token")
            if token.task_id in self._committed and self._committed[token.task_id].token == token:
                raise ValueError("committed output belongs to the query, not the attempt")
            if self._current.get(token.task_id) == token:
                del self._current[token.task_id]
                self._publish_fence(token.task_id, None)
            self._check_directory()
            directory = self._directory / token.fence
            if directory.is_symlink() or directory.exists():
                guard = native.StoreGuard.acquire(str(directory / "io.lock"), True, True)
                if guard is None:
                    raise StorageCleanupPending("attempt still owns native storage I/O")
                try:
                    shutil.rmtree(directory)
                finally:
                    guard.close()
            del self._reservations[token.fence]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closing = True
            if self._readers:
                raise RuntimeError("materialized read leases are still active")
            committed = {a.token.fence for a in self._committed.values()}
            if set(self._reservations) - committed:
                raise RuntimeError("stop and discard uncommitted attempts before query cleanup")
            self._check_directory()
            if self._directory.exists():
                shutil.rmtree(self._directory)
            self._reservations.clear()
            self._source_bytes = 0
            self._closed = True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "reserved_bytes": self._usage(),
                "attempts": len(self._reservations),
                "committed_tasks": len(self._committed),
                "sealed_stages": len(self._stages),
                "read_leases": len(self._readers),
                "error": self._error,
                "closing": self._closing,
                "closed": self._closed,
            }
