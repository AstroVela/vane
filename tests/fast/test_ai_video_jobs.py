# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Durable job recovery, fencing and data-only persistence."""

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import CancelledError, ThreadPoolExecutor

import pytest

from vane.ai import VideoJobStore, VideoTaskError
from vane.ai._video_jobs import PersistedVideoTask


def submit(store, task_id="job"):
    store.submit(task_id, "pipeline-v1", {"input": {"uri": "memory://video"}, "async": True}, {}, 1024, 10)


def expire(store):
    with sqlite3.connect(store.path) as database:
        database.execute("UPDATE video_jobs SET lease_until=0 WHERE owner IS NOT NULL")


def test_records_survive_another_process_without_pickle(tmp_path):
    path = str(tmp_path / "jobs.sqlite")
    code = """
import sys
from vane.ai import VideoJobStore
store = VideoJobStore(sys.argv[1])
store.submit('job', 'v1', {'async': True}, {}, 1024, 10)
store.claim('job', 'v1', 'owner')
store.finish('job', 'owner', result={'data': [1, 2, 3]})
"""
    subprocess.run([sys.executable, "-c", code, path], check=True)
    store = VideoJobStore(path)
    task = PersistedVideoTask("job", store, "v1")
    assert task.status == "ready" and task.result() == {"data": [1, 2, 3]}
    assert store.get_task("job", pipeline_version="v1").result() == task.result()
    assert store.list_tasks("v1")[0]["id"] == "job"
    assert "result" not in store.list_tasks("v1")[0]
    with sqlite3.connect(path) as database:
        raw = database.execute("SELECT request,result FROM video_jobs").fetchone()
    assert json.loads(raw[0]) == {"async": True}
    assert json.loads(raw[1]) == {"data": [1, 2, 3]}
    with pytest.raises(KeyError):
        store.get("job", "different-pipeline")


def test_expired_worker_cannot_publish_after_new_claim(tmp_path):
    store = VideoJobStore(tmp_path / "jobs.sqlite")
    submit(store)
    store.claim("job", "pipeline-v1", "old")
    expire(store)
    assert store.get("job", "pipeline-v1")["status"] == "interrupted"
    store.requeue("job", "pipeline-v1")
    assert store.get_task("job", pipeline_version="pipeline-v1").status == "pending"
    with pytest.raises(TimeoutError):
        store.get_task("job", pipeline_version="pipeline-v1").result(timeout=0)
    store.claim("job", "pipeline-v1", "new")
    with pytest.raises(VideoTaskError):
        store.finish("job", "old", result={"wrong": True})
    store.finish("job", "new", result={"correct": True})
    assert PersistedVideoTask("job", store, "pipeline-v1").result() == {"correct": True}


def test_claim_is_atomic_between_workers(tmp_path):
    store = VideoJobStore(tmp_path / "jobs.sqlite")
    submit(store)

    def claim(owner):
        try:
            store.claim("job", "pipeline-v1", owner)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(claim, ("a", "b"))) == 1


@pytest.mark.parametrize("running", [False, True])
def test_cancellation_is_durable_and_wins_completion(tmp_path, running):
    store = VideoJobStore(tmp_path / "jobs.sqlite")
    submit(store)
    if running:
        store.claim("job", "pipeline-v1", "owner")
    task = PersistedVideoTask("job", store, "pipeline-v1")
    task.cancel()
    if running:
        assert store.renew("job", "owner")
        store.finish("job", "owner", result={"data": [123]})
    with pytest.raises(CancelledError):
        task.result()
    assert store.get("job", "pipeline-v1")["result"] is None
    with pytest.raises(ValueError):
        store.claim("job", "pipeline-v1", "retry")


def test_failed_jobs_are_retained_and_not_resumed(tmp_path):
    store = VideoJobStore(tmp_path / "jobs.sqlite")
    submit(store)
    store.claim("job", "pipeline-v1", "owner")
    store.finish("job", "owner", error="ValueError")
    with pytest.raises(VideoTaskError, match="ValueError"):
        PersistedVideoTask("job", store, "pipeline-v1").result()
    with pytest.raises(ValueError):
        store.claim("job", "pipeline-v1", "retry")
    store.forget("job", "pipeline-v1")
    with pytest.raises(KeyError):
        store.get("job", "pipeline-v1")


def test_wait_timeout_does_not_cancel_and_running_jobs_cannot_be_forgotten(tmp_path):
    store = VideoJobStore(tmp_path / "jobs.sqlite")
    submit(store)
    store.claim("job", "pipeline-v1", "owner")
    task = PersistedVideoTask("job", store, "pipeline-v1")
    with pytest.raises(TimeoutError):
        task.result(timeout=0)
    assert task.status == "pending"
    with pytest.raises(ValueError):
        store.forget("job", "pipeline-v1")


def test_store_capacity_and_result_size_are_bounded(tmp_path):
    store = VideoJobStore(tmp_path / "jobs.sqlite", max_result_bytes=8)
    store.submit("a", "v1", {}, {}, 1, 1)
    with pytest.raises(RuntimeError, match="capacity"):
        store.submit("b", "v1", {}, {}, 1, 1)
    store.claim("a", "v1", "owner")
    with pytest.raises(ValueError, match="max_result_bytes"):
        store.finish("a", "owner", result={"too_large": True})
