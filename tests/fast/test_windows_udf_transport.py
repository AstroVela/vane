# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real Windows mapping ownership and descendant cleanup contracts."""

import gc
import subprocess
import sys
import textwrap

import psutil
import pyarrow as pa
import pytest

from vane.execution import ref_bundle
from vane.execution.udf_process_group import WorkerProcessGroup

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows kernel ownership contracts")


@pytest.mark.parametrize("track", [False, True])
def test_shared_mapping_survives_creator_close_until_last_handle(monkeypatch, track):
    def no_posix_tracker(*args):
        pytest.fail("Windows mappings must not start the POSIX resource tracker")

    monkeypatch.setattr(ref_bundle._resource_tracker, "register", no_posix_tracker)
    monkeypatch.setattr(ref_bundle._resource_tracker, "unregister", no_posix_tracker)
    owner = ref_bundle._create_shm(4096, track=track)
    reader = None
    name = owner.name
    try:
        owner.buf[:4] = b"vane"
        reader = ref_bundle._open_existing_shm(name, track=track)
        ref_bundle._unlink_shm(owner, track=track)
        owner.close()
        assert bytes(reader.buf[:4]) == b"vane"
        reopened = ref_bundle._open_existing_shm(name, track=track)
        reopened.close()
    finally:
        if reader is not None:
            reader.close()
        owner.close()
    with pytest.raises(FileNotFoundError):
        ref_bundle._open_existing_shm(name, track=track)


@pytest.mark.parametrize("cleanup", ["release", "collect", "adopt"])
def test_descriptor_retains_mapping_until_adoption_or_release(cleanup):
    descriptor = ref_bundle.make_local_shm_ref_bundle_descriptor(pa.table({"value": [1, 2, 3]}))
    name = descriptor["block_refs"][0]["shm_name"]
    result = None
    try:
        opened = ref_bundle._open_existing_shm(name, track=False)
        opened.close()
        if cleanup == "adopt":
            result = ref_bundle.make_local_shm_ref_bundle_result_from_descriptor(descriptor, block_on_budget=False)
            assert ref_bundle.materialize_ref_bundle(result[1], metadata=result[2]).to_pydict() == {"value": [1, 2, 3]}
            for ref in result[1]:
                ref.release()
        elif cleanup == "release":
            ref_bundle.release_local_shm_ref_bundle_descriptor(descriptor)
        else:
            del descriptor
            gc.collect()
        with pytest.raises(FileNotFoundError):
            ref_bundle._open_existing_shm(name, track=False)
    finally:
        if result is not None:
            for ref in result[1]:
                ref.release()
        ref_bundle._unlink_shared_memory_name(name)


def test_job_reaps_descendant_after_worker_has_exited():
    program = textwrap.dedent("""
        import subprocess
        import sys
        sys.stdin.buffer.readline()
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        print(child.pid, flush=True)
    """)
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    group = None
    try:
        group = WorkerProcessGroup(process.pid)
        stdout, stderr = process.communicate(b"start\n", timeout=10)
        assert process.returncode == 0, stderr
        descendant = psutil.Process(int(stdout.strip()))
        assert descendant.is_running()
        group.close(5)
        assert descendant.wait(timeout=5) == 1
        group.close(5)
    finally:
        if group is not None:
            group.close(5)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
