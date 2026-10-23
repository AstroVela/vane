# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real task workers: branch overlap, streaming progress and joined cleanup."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import psutil
import pyarrow as pa
import pytest

import vane
from vane.udf import zip_map_batches

pytestmark = pytest.mark.skipif(os.name != "posix", reason="generic subprocess UDF workers require POSIX")


def wait_for(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("test branch did not make progress")
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def local_only(monkeypatch, tmp_path):
    import ray

    assert not ray.is_initialized()
    monkeypatch.setenv("VANE_RUNNER", "local")

    def forbidden(*args, **kwargs):
        pytest.fail("streaming local UDFs must not start Ray")

    monkeypatch.setattr(ray, "init", forbidden)
    yield
    vane.teardown_runner()
    pids = [int(path.read_text()) for path in tmp_path.glob("*.pid")]
    deadline = time.monotonic() + 10
    while any(psutil.pid_exists(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not any(psutil.pid_exists(pid) for pid in pids), "nested workers survived runner teardown"


def combine(left, right):
    return pa.table({"key": left["key"], "value": [left["value"][0].as_py() + right["value"][0].as_py()]})


def plan(conn, left, right):
    schema = {"key": "BIGINT", "value": "VARCHAR"}
    return zip_map_batches(
        conn.from_arrow(pa.table({"video": ["one"]})),
        left,
        right,
        left_schema=schema,
        right_schema=schema,
        schema=schema,
        on="key",
        merge=combine,
        timeout=20,
    )


@pytest.mark.timeout(60)
def test_branches_overlap_and_first_pair_reaches_downstream_before_second_decode(tmp_path):
    published = tmp_path / "published"

    def branch(side):
        def process(batch):
            (tmp_path / f"{side}.pid").write_text(str(os.getpid()))
            wait_for(tmp_path / f"{1 - side}.pid")
            yield pa.table({"key": [0], "value": [str(side)]})
            wait_for(published)
            yield pa.table({"key": [1], "value": [str(side)]})

        return process

    def observe(batch):
        published.touch()
        return batch

    with vane.connect() as conn:
        conn.execute("SET threads=1")
        result = (
            plan(conn, branch(0), branch(1))
            .map_batches(
                observe,
                schema={"key": "BIGINT", "value": "VARCHAR"},
                batch_size=1,
                execution_backend="subprocess_task",
                cpus=1,
                gpus=0,
            )
            .fetchall()
        )
    assert result == [(0, "01"), (1, "01")]
    pids = [int(path.read_text()) for path in tmp_path.glob("*.pid")]
    assert len(set(pids)) == 2 and os.getpid() not in pids


@pytest.mark.timeout(60)
def test_slow_branch_retains_fast_branch_one_block_permit(tmp_path):
    first, second, release = (tmp_path / name for name in ("first", "second", "release"))

    def left(batch):
        first.touch()
        yield pa.table({"key": [0], "value": ["left"]})
        second.touch()
        yield pa.table({"key": [1], "value": ["left"]})

    def right(batch):
        wait_for(release)
        for index in range(2):
            yield pa.table({"key": [index], "value": ["right"]})

    def unblock():
        wait_for(first)
        try:
            time.sleep(0.2)
            assert not second.exists(), "fast decoder advanced before its first pair was consumed"
        finally:
            release.touch()

    with ThreadPoolExecutor(max_workers=1) as threads:
        check = threads.submit(unblock)
        with vane.connect() as conn:
            assert plan(conn, left, right).fetchall() == [(0, "leftright"), (1, "leftright")]
        check.result(timeout=10)


@pytest.mark.timeout(60)
def test_parallel_streaming_works_with_control_descriptors_above_select_limit():
    import resource

    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < 1200:
        pytest.skip("test requires at least 1200 file descriptors")

    def branch(batch):
        yield pa.table({"key": [0], "value": ["clip"]})

    with ExitStack() as handles:
        while True:
            fd = os.open(os.devnull, os.O_RDONLY)
            handles.callback(os.close, fd)
            if fd >= 1050:
                break
        with vane.connect() as conn:
            assert plan(conn, branch, branch).fetchall() == [(0, "clipclip")]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("failure", ["allocation", "key", "consumer"])
def test_branch_or_consumer_failure_reaps_both_workers(tmp_path, failure):
    def left(batch):
        (tmp_path / "left.pid").write_text(str(os.getpid()))
        wait_for(tmp_path / "right.pid")
        yield pa.table({"key": [0], "value": ["left"]})
        time.sleep(30)

    def right(batch):
        (tmp_path / "right.pid").write_text(str(os.getpid()))
        wait_for(tmp_path / "left.pid")
        if failure == "allocation":
            raise MemoryError("test decoder allocation failed")
        yield pa.table({"key": [1 if failure == "key" else 0], "value": ["right"]})
        time.sleep(30)

    def reject(batch):
        raise ValueError("test downstream refused a clip")

    with vane.connect() as conn:
        relation = plan(conn, left, right)
        if failure == "consumer":
            relation = relation.map_batches(
                reject,
                schema={"key": "BIGINT", "value": "VARCHAR"},
                batch_size=1,
                execution_backend="subprocess_task",
                cpus=1,
                gpus=0,
            )
        with pytest.raises(Exception):
            relation.fetchall()
        assert conn.execute("SELECT 1").fetchone() == (1,)
    assert len(list(tmp_path.glob("*.pid"))) == 2
