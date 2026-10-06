# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native bounded object I/O; batches never pass through the control plane."""

import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from threading import Event

import pytest

import vane
from vane._native import execution_runtime as native
from vane.execution.compiler import FragmentCompileOptions, compile_fragment_graph
from vane.execution.materialized_exchange import (
    AttemptManifest,
    MaterializedTask,
    ObjectMeta,
    PartitionSpec,
    StageManifest,
)
from vane.execution.materialized_store import AttemptReservation, CommitCoordinator, SharedDirectoryStore
from vane.execution.query_options import FteOptions, QueryExecutionOptions, RayExecution
from vane.execution.resource_demand import MemoryDemand, ResourceDemand
from vane.execution.submission import prepare_ray_query, prepare_worker_plan


def schema_for(sql="select 1::bigint"):
    with vane.connect(backend="local") as connection:
        graph = compile_fragment_graph(connection, sql, query_id="materialized-schema")
    return next(f.outputs[0].schema for f in graph.fragments if f.fragment_id == graph.result.fragment_id)


def channel(schema):
    value = native.DirectChannel(schema, native.DirectLimits(128, 128, 3, 1), 1, ["consumer"])
    value.add_producer("producer")
    value.seal_producers()
    return value


def wait(operation, accept=lambda value: value, timeout=5):
    deadline = time.monotonic() + timeout
    while True:
        value = operation()
        if accept(value):
            return value
        assert time.monotonic() < deadline, value
        time.sleep(0.002)


def write_object(path, schema, rows):
    rows = list(rows)
    source = channel(schema)
    writer = native.MaterializedIO.write(
        str(path), source, "consumer", 1 << 20, native.MaterializedIO.staging_bytes(128)
    )
    try:
        sequence = 0
        for offset in range(0, len(rows), 3):
            sequence += 1
            wait(
                lambda: source._write_rows("producer", sequence, rows[offset : offset + 3]),
                lambda state: state == "accepted",
            )
        source.finish("producer", sequence)
        status = wait(writer.status, lambda value: value["done"])
        assert status["error"] == ""
        assert source.snapshot()["bytes"] == 0
        return status["object"]
    finally:
        writer.close()


def read_object(path, schema, metadata):
    target = channel(schema)
    reader = native.MaterializedIO.read(
        str(path), target, "producer", 1 << 20, native.MaterializedIO.staging_bytes(128), metadata
    )
    try:
        rows = []
        while True:
            state, batch = wait(lambda: target.poll("consumer"), lambda value: value[0] != "blocked")
            if state == "end":
                break
            rows += batch.to_rows()
            batch.close()
        reader.close()
        assert reader.status()["done"]
        assert reader.status()["error"] == ""
        return rows
    finally:
        reader.close()


@pytest.mark.parametrize(
    "sql_type,value",
    [
        ("boolean", True),
        ("tinyint", -12),
        ("smallint", -32000),
        ("integer", -100000),
        ("bigint", -(2**60)),
        ("utinyint", 200),
        ("usmallint", 60000),
        ("uinteger", 2**31),
        ("ubigint", 2**63 + 1),
        ("hugeint", -(2**127)),
        ("hugeint", -(10**38)),
        ("hugeint", 10**38),
        ("hugeint", 2**127 - 1),
        ("float", 1.5),
        ("double", 2.25),
        ("varchar", "a long UTF-8 value 中文"),
        (None, None),
    ],
)
def test_materialized_native_types(tmp_path, sql_type, value):
    schema = schema_for("select null" if sql_type is None else f"select null::{sql_type}")
    path = tmp_path / "object.mat"
    expected = [(None,), (value,)]
    metadata = write_object(path, schema, expected)
    assert metadata["rows"] == 2
    assert metadata["frames"] == 1
    assert metadata["bytes"] == path.stat().st_size
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    native.MaterializedIO.verify(str(path), metadata, schema)
    assert read_object(path, schema, metadata) == expected


def test_empty_object_is_sealed_with_schema(tmp_path):
    schema = schema_for()
    path = tmp_path / "empty.mat"
    metadata = write_object(path, schema, [])
    assert metadata["rows"] == metadata["frames"] == 0
    assert read_object(path, schema, metadata) == []
    with pytest.raises(Exception, match="schema"):
        read_object(path, schema_for("select 'different'"), metadata)


@pytest.mark.parametrize("damage", ["delete", "truncate", "append", "rewrite"])
def test_changed_committed_object_fails_before_rows(tmp_path, damage):
    schema = schema_for()
    path = tmp_path / "damaged.mat"
    metadata = write_object(path, schema, [(i,) for i in range(20)])
    content = path.read_bytes()
    if damage == "delete":
        path.unlink()
    elif damage == "truncate":
        path.write_bytes(content[:-1])
    elif damage == "append":
        path.write_bytes(content + b"x")
    else:
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(Exception):
        read_object(path, schema, metadata)


@pytest.mark.parametrize("field,value", [("rows", 4), ("frames", 2), ("sha256", "0" * 64)])
def test_native_verify_rejects_false_seal(tmp_path, field, value):
    schema = schema_for()
    path = tmp_path / "seal.mat"
    metadata = write_object(path, schema, [(1,), (2,), (3,)])
    with pytest.raises(Exception, match="seal|checksum"):
        native.MaterializedIO.verify(str(path), {**metadata, field: value}, schema)


def test_cancel_empty_writer_retains_prior_input_failure(tmp_path):
    source = channel(schema_for())
    writer = native.MaterializedIO.write(
        str(tmp_path / "pending.mat"), source, "consumer", 1 << 20, native.MaterializedIO.staging_bytes(128)
    )
    source.abort("original native failure")
    writer.cancel("later cancellation")
    writer.close()
    assert "original native failure" in writer.status()["error"]


def test_cancel_reader_while_borrowed_window_is_full(tmp_path):
    schema = schema_for()
    path = tmp_path / "slow.mat"
    metadata = write_object(path, schema, [(i,) for i in range(20)])
    target = channel(schema)
    reader = native.MaterializedIO.read(
        str(path), target, "producer", 1 << 20, native.MaterializedIO.staging_bytes(128), metadata
    )
    try:
        _, batch = wait(lambda: target.poll("consumer"), lambda result: result[0] == "data")
        wait(lambda: target.snapshot()["write_blocks"])
        reader.cancel("client stopped")
        reader.close()
        assert reader.status()["done"]
        assert "client stopped" in reader.status()["error"]
        assert batch.to_rows() == [(0,), (1,), (2,)]
        assert target.snapshot()["bytes"] > 0
        batch.close()
        assert target.snapshot()["bytes"] == 0
        with pytest.raises(Exception, match="client stopped"):
            target.poll("consumer")
    finally:
        reader.close()


def test_storage_limit_and_exclusive_create(tmp_path):
    schema = schema_for()
    path = tmp_path / "quota.mat"
    source = channel(schema)
    source.finish("producer", 0)
    writer = native.MaterializedIO.write(str(path), source, "consumer", 40, native.MaterializedIO.staging_bytes(128))
    try:
        result = wait(writer.status, lambda status: status["done"])
        assert "reserved storage bytes" in result["error"]
        assert path.stat().st_size <= 40
    finally:
        writer.close()
    original = path.read_bytes()
    source = channel(schema)
    writer = native.MaterializedIO.write(
        str(path), source, "consumer", 1 << 20, native.MaterializedIO.staging_bytes(128)
    )
    try:
        assert wait(writer.status, lambda status: status["done"])["error"]
        assert path.read_bytes() == original
    finally:
        writer.close()


def test_materialized_staging_must_be_reserved(tmp_path):
    source = channel(schema_for())
    with pytest.raises(Exception, match="staging"):
        native.MaterializedIO.write(str(tmp_path / "no-reservation.mat"), source, "consumer", 1024, 1)
    assert not (tmp_path / "no-reservation.mat").exists()


def test_materialized_requires_a_native_buffer(tmp_path):
    with pytest.raises(Exception, match="limits"):
        native.MaterializedIO.write(str(tmp_path / "missing-buffer.mat"), None, "consumer", 1024, 1 << 20)


@pytest.fixture
def commits(tmp_path):
    from vane._native import execution_plan

    schema = schema_for()
    tasks = tuple(
        MaterializedTask(task, "stage", hashlib.sha256(task.encode()).hexdigest(), (PartitionSpec("edge", 0, schema),))
        for task in ("a", "b")
    )
    owner = CommitCoordinator(
        SharedDirectoryStore(tmp_path / "store"), "query", execution_plan.engine_identity(), tasks, max_bytes=4 << 20
    )
    try:
        yield owner
    finally:
        for reservation in tuple(owner._reservations.values()):
            if (
                reservation.token.task_id not in owner._committed
                or owner._committed[reservation.token.task_id].token != reservation.token
            ):
                owner.discard(reservation.token)
        owner.close()


def sealed_attempt(owner, task="a", epoch="worker-1", rows=((1,), (2,), (3,))):
    reserved = owner.begin(task, epoch, object_bytes=1 << 20)
    metadata = {
        obj.key: ObjectMeta.from_dict(write_object(reserved.store.path(obj.key), obj.output.schema, rows))
        for obj in reserved.objects
    }
    return reserved, reserved.seal(metadata)


def test_stage_barrier_idempotent_commit_and_immutable_roundtrip(commits):
    reserved, first = sealed_attempt(commits)
    assert AttemptReservation.from_dict(json.loads(json.dumps(reserved.to_dict()))) == reserved
    assert AttemptManifest.from_dict(json.loads(json.dumps(first.to_dict()))) == first
    with pytest.raises(FrozenInstanceError):
        first.token.attempt = 7
    with pytest.raises(RuntimeError, match="uncommitted"):
        commits.seal_stage("stage")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(commits.commit, first) for _ in range(2)]
        assert all(future.result(timeout=5) == first for future in futures)
    assert commits.commit(first) is first
    obj = first.objects[0]
    with pytest.raises(ValueError, match="conflicting duplicate"):
        commits.commit(replace(first, objects=(replace(obj, metadata=replace(obj.metadata, rows=4)),)))
    with pytest.raises(RuntimeError, match="uncommitted"):
        commits.seal_stage("stage")
    _, second = sealed_attempt(commits, "b", "worker-2", [])
    commits.commit(second)
    stage = commits.seal_stage("stage")
    assert commits.seal_stage("stage") is stage
    assert StageManifest.from_dict(json.loads(json.dumps(stage.to_dict()))) == stage
    assert (
        stage.identity
        == StageManifest(
            stage.query_id, stage.stage_id, stage.engine_identity, tuple(reversed(stage.attempts))
        ).identity
    )
    assert [attempt.token.task_id for attempt in stage.attempts] == ["a", "b"]
    with pytest.raises(ValueError, match="already committed"):
        commits.begin("a", "different-worker", object_bytes=1024)
    with pytest.raises(ValueError, match="belongs to the query"):
        commits.discard(first.token)


@pytest.mark.parametrize("change", ["worker_epoch", "attempt", "input_id", "engine", "schema", "key"])
def test_foreign_or_changed_manifest_cannot_commit(commits, change):
    _, manifest = sealed_attempt(commits)
    if change == "engine":
        bad = replace(manifest, engine_identity="another-engine")
    elif change in {"worker_epoch", "attempt", "input_id"}:
        value = {"worker_epoch": "another-worker", "attempt": 99, "input_id": "0" * 64}[change]
        bad = replace(manifest, token=replace(manifest.token, **{change: value}))
    elif change == "schema":
        obj = manifest.objects[0]
        bad = replace(manifest, objects=(replace(obj, output=replace(obj.output, schema=b"another-schema")),))
    else:
        obj = manifest.objects[0]
        bad = replace(manifest, objects=(replace(obj, key="0" * 32 + obj.key[32:]),))
    with pytest.raises(ValueError):
        commits.commit(bad)
    assert commits.snapshot()["committed_tasks"] == 0
    assert commits.commit(manifest) == manifest


def test_retry_fences_late_success_and_replays_fixed_input(commits):
    first, late = sealed_attempt(commits)
    second, winner = sealed_attempt(commits, epoch="replacement-worker")
    assert second.token.attempt == first.token.attempt + 1
    assert second.token.input_id == first.token.input_id
    assert second.token.fence != first.token.fence
    with pytest.raises(ValueError, match="stale"):
        commits.commit(late)
    commits.commit(winner)
    commits.discard(first.token)
    assert commits.snapshot()["committed_tasks"] == 1
    assert not first.store.path(first.objects[0].key).exists()
    assert second.store.path(second.objects[0].key).exists()


@pytest.mark.parametrize("action", ["cancel", "retry"])
def test_fencing_can_win_while_commit_verifies_storage(commits, monkeypatch, action):
    _, manifest = sealed_attempt(commits)
    entered, proceed = Event(), Event()
    verify = native.MaterializedIO.verify

    def blocked_verify(*args):
        entered.set()
        assert proceed.wait(5)
        verify(*args)

    monkeypatch.setattr(native.MaterializedIO, "verify", blocked_verify)
    with ThreadPoolExecutor(max_workers=2) as pool:
        committing = pool.submit(commits.commit, manifest)
        try:
            assert entered.wait(5)
            change = (
                pool.submit(commits.cancel, "query canceled")
                if action == "cancel"
                else pool.submit(commits.begin, "a", "new-epoch", object_bytes=1024)
            )
            change.result(timeout=2)
        finally:
            proceed.set()
        with pytest.raises((ValueError, RuntimeError), match="stale|canceled"):
            committing.result(timeout=5)
    assert commits.snapshot()["committed_tasks"] == 0


def test_read_lease_pins_committed_objects_through_cleanup(commits):
    for task in ("a", "b"):
        _, manifest = sealed_attempt(commits, task)
        commits.commit(manifest)
    stage = commits.seal_stage("stage")
    with commits.retain(stage):
        with pytest.raises(RuntimeError, match="read leases"):
            commits.close()
        for attempt in stage.attempts:
            obj = attempt.objects[0]
            assert read_object(commits.store.path(obj.key), obj.output.schema, obj.metadata.to_dict()) == [
                (1,),
                (2,),
                (3,),
            ]
        assert commits.snapshot()["reserved_bytes"] == 2 << 20
    commits.close()
    assert commits.snapshot()["reserved_bytes"] == 0


def test_cleanup_failure_keeps_storage_reservation(commits, monkeypatch):
    from vane.execution import materialized_store

    reserved, manifest = sealed_attempt(commits)
    commits.commit(manifest)
    commits.cancel("original query failure")
    remove = materialized_store.shutil.rmtree

    def fail_remove(*args):
        raise OSError("storage unavailable")

    monkeypatch.setattr(materialized_store.shutil, "rmtree", fail_remove)
    with pytest.raises(OSError, match="storage unavailable"):
        commits.close()
    assert commits.snapshot()["reserved_bytes"] == 1 << 20
    assert commits.snapshot()["error"] == "original query failure"
    assert reserved.store.path(reserved.objects[0].key).exists()
    monkeypatch.setattr(materialized_store.shutil, "rmtree", remove)
    commits.close()
    assert commits.snapshot()["reserved_bytes"] == 0


def test_attempt_quota_is_returned_only_after_discard(commits, monkeypatch):
    from vane.execution import materialized_store

    first = commits.begin("a", "epoch-1", object_bytes=2 << 20)
    commits.begin("a", "epoch-2", object_bytes=2 << 20)
    with pytest.raises(RuntimeError, match="capacity"):
        commits.begin("b", "epoch-3", object_bytes=1024)
    remove = materialized_store.shutil.rmtree
    monkeypatch.setattr(
        materialized_store.shutil, "rmtree", lambda *args: (_ for _ in ()).throw(OSError("pending cleanup"))
    )
    with pytest.raises(OSError):
        commits.discard(first.token)
    assert commits.snapshot()["reserved_bytes"] == 4 << 20
    monkeypatch.setattr(materialized_store.shutil, "rmtree", remove)
    commits.discard(first.token)
    assert commits.snapshot()["reserved_bytes"] == 2 << 20
    commits.begin("b", "epoch-3", object_bytes=1024)


@pytest.mark.parametrize("damage", ["missing", "modified", "false_count"])
def test_commit_verifies_each_object_before_acceptance(commits, damage):
    reserved, manifest = sealed_attempt(commits)
    obj = manifest.objects[0]
    path = reserved.store.path(obj.key)
    if damage == "missing":
        path.unlink()
    elif damage == "modified":
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    else:
        manifest = replace(manifest, objects=(replace(obj, metadata=replace(obj.metadata, rows=4)),))
    with pytest.raises(Exception):
        commits.commit(manifest)
    assert commits.snapshot()["committed_tasks"] == 0
    with pytest.raises(RuntimeError, match="uncommitted"):
        commits.seal_stage("stage")


def test_store_identity_and_object_paths_are_fenced(commits, tmp_path):
    reserved = commits.begin("a", "epoch", object_bytes=1024)
    assert SharedDirectoryStore(commits.store.root).descriptor == commits.store
    with pytest.raises(ValueError, match="identity"):
        replace(commits.store, store_id="0" * 32).check()
    with pytest.raises(ValueError, match="key"):
        commits.store.path("../escape")
    path = reserved.store.path(reserved.objects[0].key)
    target = tmp_path / "outside"
    target.write_text("do not touch")
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        reserved.store.path(reserved.objects[0].key)
    assert target.read_text() == "do not touch"


@pytest.mark.parametrize("change", ["identity", "query_symlink"])
def test_storage_location_changes_block_cleanup_without_returning_quota(commits, tmp_path, change):
    reserved, manifest = sealed_attempt(commits)
    commits.commit(manifest)
    if change == "identity":
        marker = tmp_path / "store" / "store.json"
        original = marker.read_bytes()
        marker.write_text(json.dumps({"protocol": True, "store_id": commits.store.store_id}))

        def restore():
            marker.write_bytes(original)
    else:
        directory = tmp_path / "store" / "queries" / commits.namespace
        outside = tmp_path / "saved-query"
        directory.rename(outside)
        directory.symlink_to(outside, target_is_directory=True)

        def restore():
            directory.unlink()
            outside.rename(directory)

    try:
        with pytest.raises(ValueError, match="identity|symlink"):
            commits.close()
        assert commits.snapshot()["reserved_bytes"] == 1 << 20
    finally:
        restore()
    assert reserved.store.path(reserved.objects[0].key).exists()
    commits.close()


def test_every_declared_partition_requires_an_explicit_object(tmp_path):
    schema = schema_for()
    task = MaterializedTask("task", "stage", "a" * 64, tuple(PartitionSpec("edge", i, schema) for i in range(2)))
    owner = CommitCoordinator(SharedDirectoryStore(tmp_path / "store"), "query", "engine", (task,), max_bytes=2 << 20)
    reservation = owner.begin("task", "epoch", object_bytes=1 << 20)
    try:
        metadata = {}
        for obj in reservation.objects:
            metadata[obj.key] = ObjectMeta.from_dict(write_object(reservation.store.path(obj.key), schema, []))
        with pytest.raises(ValueError, match="every"):
            reservation.seal({reservation.objects[0].key: metadata[reservation.objects[0].key]})
        complete = reservation.seal(metadata)
        with pytest.raises(ValueError, match="partitions"):
            owner.commit(replace(complete, objects=complete.objects[:1]))
        owner.commit(complete)
        assert len(owner.seal_stage("stage").attempts[0].objects) == 2
    finally:
        if not owner.snapshot()["committed_tasks"]:
            owner.discard(reservation.token)
        owner.close()


@pytest.mark.parametrize("mutation", ["protocol", "codec", "unknown", "bool_count", "duplicate", "empty"])
def test_manifest_wire_validation(commits, mutation):
    _, manifest = sealed_attempt(commits)
    wire = manifest.to_dict()
    if mutation == "protocol":
        wire["protocol"] = True
    elif mutation == "codec":
        wire["codec"] = "another-format"
    elif mutation == "unknown":
        wire["future"] = True
    elif mutation == "bool_count":
        wire["objects"][0]["metadata"]["rows"] = True
    elif mutation == "duplicate":
        wire["objects"].append(wire["objects"][0])
    else:
        wire["objects"] = []
    with pytest.raises(ValueError):
        AttemptManifest.from_dict(wire)


_NATIVE_PRODUCER = """
import json, sys, time
import vane
from vane._native import execution_plan, execution_runtime as native
from vane.execution.compiler import FragmentCompileOptions
from vane.execution.materialized_store import AttemptReservation
from vane.execution.query_options import QueryExecutionOptions, RayExecution, FteOptions
from vane.execution.resource_demand import MemoryDemand, ResourceDemand
from vane.execution.submission import prepare_ray_query, prepare_worker_plan

request = json.loads(sys.stdin.readline())
reservation = AttemptReservation.from_dict(request["reservation"])
assert reservation.engine_identity == execution_plan.engine_identity()
obj = reservation.objects[0]
connection = vane.connect(backend="local", config={"threads": 1})
options = QueryExecutionOptions(RayExecution("fte", FteOptions("shared", 2, 0)), 10, 30, 30)
spec = prepare_ray_query(connection, "select range from range(20)", query_id=reservation.token.query_id,
    options=options, resources=ResourceDemand(1, 8, MemoryDemand(2**26, 2**20, 2**20, 2**20), 4),
    compile_options=FragmentCompileOptions(1))
prepare_worker_plan(connection, spec)
fragment = next(f for f in spec.graph.fragments if not f.inputs)
channel = native.DirectChannel(obj.output.schema, native.DirectLimits(128, 128, 3, 1), 2, ["store"])
channel.add_producer("task")
if request["before_seal"]:
    channel.add_producer("unsealed-output")
channel.seal_producers()
writer = native.MaterializedIO.write(str(reservation.store.path(obj.key)), channel, "store",
    obj.max_bytes, native.MaterializedIO.staging_bytes(128))
tasks = native.TaskService(connection)
snapshot = next(s.payload for s in spec.source_snapshots if s.fragment_id == fragment.fragment_id)
assignments = {s.source_id: [split.split_id for split in s.splits] for s in fragment.sources}
tasks.prepare("task", fragment.native_plan, spec.connection_snapshot, snapshot, assignments, {},
    [{"channels": [channel], "producer": "task"}])
try:
    tasks.start("task", "initial")
    deadline = time.monotonic() + 20
    while not tasks.production_status()["finished"]:
        assert time.monotonic() < deadline
        tasks.pump(1)
        time.sleep(.001)
    metadata = None
    if not request["before_seal"]:
        while not writer.status()["done"]:
            assert time.monotonic() < deadline
            time.sleep(.001)
        assert not writer.status()["error"], writer.status()
        metadata = writer.status()["object"]
    print(json.dumps({"metadata": metadata}), flush=True)
    sys.stdin.readline()
finally:
    tasks.cancel("producer exiting")
    writer.close()
    tasks.release()
    connection.close()
"""


def run_producer(reservation, before_seal):
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", _NATIVE_PRODUCER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        child.stdin.write(json.dumps({"reservation": reservation.to_dict(), "before_seal": before_seal}) + "\n")
        child.stdin.flush()
        line = pool.submit(child.stdout.readline).result(timeout=30)
        assert line, child.stderr.read()
        return json.loads(line)["metadata"]
    finally:
        # Simulate worker loss, including loss after file seal but before the
        # coordinator has acknowledged the attempt's commit.
        if child.poll() is None:
            child.kill()
        pool.shutdown(wait=True)
        child.communicate(timeout=10)


@pytest.mark.parametrize("before_seal", [True, False])
def test_native_fragment_objects_survive_worker_exit(commits, before_seal):
    reservation = commits.begin("a", "producer-epoch", object_bytes=1 << 20)
    metadata = run_producer(reservation, before_seal)
    if before_seal:
        assert metadata is None
        assert commits.snapshot()["committed_tasks"] == 0
        with pytest.raises(RuntimeError, match="uncommitted"):
            commits.seal_stage("stage")
        old_input = reservation.token.input_id
        commits.discard(reservation.token)
        reservation = commits.begin("a", "replacement-epoch", object_bytes=1 << 20)
        assert reservation.token.input_id == old_input
        metadata = run_producer(reservation, False)
    manifest = reservation.seal({reservation.objects[0].key: ObjectMeta.from_dict(metadata)})
    commits.commit(manifest)
    _, empty = sealed_attempt(commits, "b", rows=[])
    commits.commit(empty)
    stage = commits.seal_stage("stage")
    with commits.retain(stage):
        obj = manifest.objects[0]
        assert read_object(reservation.store.path(obj.key), obj.output.schema, obj.metadata.to_dict()) == [
            (i,) for i in range(20)
        ]


@pytest.mark.parametrize("threads", [1, 4])
def test_materialized_input_runs_in_the_common_native_task_runtime(tmp_path, threads):
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        spec = prepare_ray_query(
            connection,
            "select range from range(20)",
            query_id="materialized-consumer",
            options=QueryExecutionOptions(RayExecution("fte", FteOptions("shared", 2, 0)), 10, 30, 30),
            resources=ResourceDemand(1, 8, MemoryDemand(2**26, 2**20, 2**20, 2**20), 4),
            compile_options=FragmentCompileOptions(2),
        )
        prepare_worker_plan(connection, spec)
        root = next(f for f in spec.graph.fragments if f.fragment_id == spec.graph.result.fragment_id)
        snapshot = next(s.payload for s in spec.source_snapshots if s.fragment_id == root.fragment_id)
        path = tmp_path / "input.mat"
        metadata = write_object(path, root.inputs[0].schema, [(i,) for i in range(20)])
        source = channel(root.inputs[0].schema)
        output = native.DirectChannel(root.outputs[0].schema, native.DirectLimits(128, 128, 3, 1), 1, ["client"])
        output.add_producer("root")
        output.seal_producers()
        service = native.TaskService(connection)
        reader = native.MaterializedIO.read(
            str(path), source, "producer", 1 << 20, native.MaterializedIO.staging_bytes(128), metadata
        )
        try:
            service.prepare(
                "root",
                root.native_plan,
                spec.connection_snapshot,
                snapshot,
                {},
                {root.inputs[0].port_id: [(source, "consumer")]},
                [{"channels": [output], "producer": "root"}],
            )
            service.start("root", "initial")

            def poll():
                service.pump(1)
                return output.poll("client")

            rows = []
            while True:
                state, batch = wait(poll, lambda result: result[0] != "blocked")
                if state == "end":
                    break
                rows += batch.to_rows()
                batch.close()
            assert rows == [(i,) for i in range(20)]
            assert service.production_status()["finished"]

            def drained():
                service.pump(1)
                return all(task["state"] == "FINISHED" for task in service.status())

            wait(drained)
            assert not reader.status()["error"]
        finally:
            service.cancel("test cleanup")
            reader.close()
            service.release()
        assert source.snapshot()["bytes"] == output.snapshot()["bytes"] == 0
