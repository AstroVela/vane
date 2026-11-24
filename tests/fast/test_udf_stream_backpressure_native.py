# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""The same slow-consumer contract for Ray and local task/actor producers."""

import os
import subprocess
import sys
import textwrap
import time

import pytest


@pytest.mark.parametrize(
    "backend",
    [
        "subprocess_task",
        "subprocess_actor",
        pytest.param("ray_task", marks=pytest.mark.real_ray),
        pytest.param("ray_actor", marks=pytest.mark.real_ray),
    ],
)
def test_fanout_producer_stops_when_downstream_cannot_accept_more(tmp_path, request, backend):
    env = {
        **os.environ,
        "VANE_RUNNER": "local-fast",
        "VANE_PROGRESS": "0",
        "VANE_LOCAL_SHM_STORE_BYTES": "64m",
        "VANE_LOCAL_SHM_REF_BUDGET_BYTES": "16m",
    }
    if backend.startswith("ray_"):
        _, address, _ = request.getfixturevalue("_ray_local_cluster")
        env.update(VANE_RUNNER="ray", RAY_ADDRESS=address)
    produced = tmp_path / "produced"
    entered = tmp_path / "entered"
    proceed = tmp_path / "proceed"
    script = textwrap.dedent(f"""
        from pathlib import Path
        import time
        import pyarrow as pa
        import vane

        produced = Path({str(produced)!r})
        entered = Path({str(entered)!r})
        proceed = Path({str(proceed)!r})

        def expand(batch):
            for value in range(128):
                with produced.open('a') as output:
                    output.write(str(value) + '\\n')
                yield pa.table({{'id': [value], 'payload': [b'x' * 32768]}})

        class Expand:
            def __call__(self, batch):
                return expand(batch)

        class Consume:
            def __call__(self, batch):
                entered.touch()
                deadline = time.monotonic() + 40
                while not proceed.exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError('test consumer was not released')
                    time.sleep(0.01)
                return batch.select(['id'])

        backend = {backend!r}
        with vane.connect(config={{'threads': 4}}) as con:
            options = {{'actor_number': 1}} if backend.endswith('_actor') else {{}}
            rel = con.sql('select 1::BIGINT as id').map_batches(
                Expand if options else expand,
                schema={{'id': 'BIGINT', 'payload': 'BLOB'}},
                execution_backend=backend, batch_size=1, output_batch_size=1,
                min_task_batch_size=1, task_input_max_bytes=1024, output_target_max_bytes=1024,
                **options)
            rel = rel.map_batches(
                Consume, schema={{'id': 'BIGINT'}}, batch_size=1, actor_number=1,
                min_task_batch_size=1, task_input_max_bytes=1024,
                execution_backend='ray_actor' if backend.startswith('ray_') else 'subprocess_actor')
            assert sorted(rel.to_arrow_table()['id'].to_pylist()) == list(range(128))
        vane.teardown_runner()
    """)
    with (tmp_path / "probe.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-c", script], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while not entered.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert entered.exists(), (tmp_path / "probe.log").read_text()
            time.sleep(0.5)
            count = len(produced.read_text().splitlines())
            # Includes native handoff/compute buffers as well as the two-block
            # generator window. It must remain independent of total fanout.
            assert 1 <= count <= 16, (backend, count)
            proceed.touch()
            assert process.wait(timeout=60) == 0, (tmp_path / "probe.log").read_text()
        finally:
            proceed.touch()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_expanding_pipeline_releases_real_storage_under_small_budget(backend):
    script = textwrap.dedent(f"""
        import gc
        import pyarrow as pa
        import vane
        from vane.execution.ref_bundle import local_shm_ref_budget_snapshot
        from vane.execution.udf_shm_store import local_shm_store_snapshot

        def expand(batch):
            for value in batch['id'].to_pylist():
                for part in range(8):
                    yield pa.table({{'id': [value * 8 + part], 'payload': [b'x' * (512 * 1024)]}})

        class Expand:
            def __call__(self, batch):
                return expand(batch)

        def consume(batch):
            return batch.select(['id'])

        class Consume:
            def __call__(self, batch):
                return consume(batch)

        def forward(batch):
            return batch

        class Forward:
            def __call__(self, batch):
                return forward(batch)

        backend = {backend!r}
        actors = {{'actor_number': 1}} if backend.endswith('_actor') else {{}}
        with vane.connect(config={{'threads': 4}}) as con:
            rel = con.sql('select range::BIGINT as id from range(8)').map_batches(
                Expand if actors else expand, schema={{'id': 'BIGINT', 'payload': 'BLOB'}},
                execution_backend=backend, batch_size=1, min_task_batch_size=1, task_input_max_bytes=1024,
                output_target_max_bytes=1024, **actors)
            # Several invocations consume slices from pooled output. No ACK
            # may release its physical charge or manufacture another credit.
            rel = rel.map_batches(
                Forward if actors else forward, schema={{'id': 'BIGINT', 'payload': 'BLOB'}},
                execution_backend=backend, batch_size=1, min_task_batch_size=1, task_input_max_bytes=1024, **actors)
            rel = rel.map_batches(
                Consume if actors else consume, schema={{'id': 'BIGINT'}},
                execution_backend=backend, batch_size=1, min_task_batch_size=1, task_input_max_bytes=1024,
                **actors)
            assert sorted(rel.to_arrow_table()['id'].to_pylist()) == list(range(64))
        vane.teardown_runner()
        gc.collect()
        assert local_shm_ref_budget_snapshot()['usage_bytes'] == 0
        assert local_shm_store_snapshot()['live_bytes'] == 0
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "VANE_RUNNER": "local-fast",
            "VANE_PROGRESS": "0",
            "VANE_LOCAL_SHM_STORE_BYTES": "16m",
            "VANE_LOCAL_SHM_REF_BUDGET_BYTES": "2m",
        },
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("release", ["fork", "notification"])
def test_output_capacity_waits_for_independent_historical_input_release(tmp_path, release):
    if release == "fork" and not hasattr(os, "fork"):
        pytest.skip("requires fork")
    gate = tmp_path / "release-input"
    script = textwrap.dedent(f"""
        import gc
        import os
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        import pyarrow as pa
        import vane
        from vane.execution import udf_shm_store as storage
        from vane.execution.ref_bundle import local_shm_ref_budget_snapshot

        gate = Path({str(gate)!r})
        release = {release!r}
        observed = threading.Event()
        waiting_store = []
        wait_for_capacity = storage.LocalShmStore.wait_for_capacity

        def observe_wait(store, *args):
            waiting_store.append(store)
            observed.set()
            return wait_for_capacity(store, *args)

        storage.LocalShmStore.wait_for_capacity = observe_wait

        def await_release():
            deadline = time.monotonic() + 20
            while not gate.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError('historical input release was not allowed')
                time.sleep(0.01)

        if release == 'notification':
            release_read = storage.ParentShmPeer.release_read
            def delayed_release(peer, token):
                lease = peer._reads.get(token)
                if lease is not None and lease.allocation.size > 65536:
                    await_release()
                release_read(peer, token)
            storage.ParentShmPeer.release_read = delayed_release

        def expand(batch):
            yield pa.table({{'payload': [b'x' * 65536]}})
            yield pa.table({{'payload': [b'second']}})

        class Consume:
            def __init__(self):
                self.calls = 0

            def __call__(self, batch):
                self.calls += 1
                if self.calls == 1:
                    if release == 'fork':
                        pid = os.fork()
                        if pid == 0:
                            try:
                                await_release()
                            finally:
                                os._exit(0)
                        threading.Thread(target=os.waitpid, args=(pid, 0), daemon=True).start()
                    return pa.table({{'payload': [b'small']}})
                return pa.table({{'payload': [b'y' * 65536]}})

        def allow_release_after_wait():
            try:
                assert observed.wait(15), 'query never reached physical capacity pressure'
                store = waiting_store[0]
                # The capacity check holds this lock until it actually waits or
                # fails. Only a real wait can expose a registered waiting writer.
                with store._capacity_condition:
                    assert store._capacity_condition.wait_for(lambda: bool(store._waiting_writes), timeout=5)
            finally:
                gate.touch()

        try:
            with ThreadPoolExecutor(max_workers=1) as threads:
                released = threads.submit(allow_release_after_wait)
                try:
                    with vane.connect(config={{'threads': 4}}) as con:
                        watchdog = threading.Timer(20, con.interrupt)
                        watchdog.daemon = True
                        watchdog.start()
                        try:
                            relation = con.sql('select 1::BIGINT as id').map_batches(
                                expand, schema={{'payload': 'BLOB'}}, execution_backend='subprocess_task',
                                batch_size=1, min_task_batch_size=1, task_input_max_bytes=8,
                                output_batch_size=1, output_target_max_bytes=1024,
                            ).map_batches(
                                Consume, schema={{'payload': 'BLOB'}}, execution_backend='subprocess_actor',
                                actor_number=1, batch_size=1, min_task_batch_size=1, task_input_max_bytes=1024,
                                output_batch_size=1, output_target_max_bytes=1024,
                            )
                            result = relation.to_arrow_table()
                            assert result['payload'].to_pylist() == [b'small', b'y' * 65536]
                            del result
                        finally:
                            watchdog.cancel()
                    released.result(timeout=10)
                finally:
                    gate.touch()
        finally:
            vane.teardown_runner()
            gc.collect()
            assert storage.local_shm_store_snapshot()['live_bytes'] == 0
            assert local_shm_ref_budget_snapshot()['usage_bytes'] == 0
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "VANE_RUNNER": "local-fast",
            "VANE_PROGRESS": "0",
            "VANE_LOCAL_SHM_STORE_BYTES": "128KiB",
            "VANE_LOCAL_SHM_REF_BUDGET_BYTES": "auto",
        },
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "source_kind,source_backend,arena,copy_input,intermediate,actors",
    [
        ("generator", "subprocess_task", "16MiB", False, False, 1),
        ("table", "subprocess_task", "16MiB", False, False, 1),
        ("table", "subprocess_actor", "16MiB", False, False, 1),
        ("generator", "subprocess_task", "96KiB", False, False, 1),
        ("table", "subprocess_actor", "96KiB", False, False, 1),
        ("generator", "subprocess_task", "96KiB", True, False, 1),
        ("table", "subprocess_actor", "96KiB", True, False, 1),
        ("table", "subprocess_actor", "16MiB", False, True, 1),
        ("table", "subprocess_actor", "16MiB", True, True, 1),
        ("generator", "subprocess_task", "16MiB", False, False, 2),
        ("generator", "subprocess_task", "16MiB", False, False, 4),
        ("table", "subprocess_actor", "16MiB", False, False, 2),
        ("table", "subprocess_actor", "16MiB", False, True, 2),
    ],
)
def test_actor_retaining_previous_input_can_receive_the_next_batch(
    source_kind, source_backend, arena, copy_input, intermediate, actors
):
    script = textwrap.dedent(f"""
        import gc
        import threading
        import time
        import pyarrow as pa
        import vane
        from vane.execution.ref_bundle import local_shm_budget_manager, local_shm_ref_budget_snapshot
        from vane.execution.udf_shm_store import local_shm_store_snapshot

        def expand(batch):
            for value in range(32):
                yield pa.table({{'id': [value], 'payload': [b'x' * 65536]}})

        def materialize(batch):
            if {intermediate!r}:
                time.sleep(0.05)
            values = batch['id'].to_pylist()
            return pa.table({{'id': values, 'payload': [b'x' * 65536 for _ in values]}})

        class Materialize:
            def __call__(self, batch):
                return materialize(batch)

        class Copy:
            def __call__(self, batch):
                return pa.table(batch.to_pydict())

        class RetainingConsumer:
            def __init__(self):
                self.last = None

            def __call__(self, batch):
                self.last = batch.to_pydict() if {copy_input!r} else batch
                time.sleep(0.05)
                return batch.select(['id'])

        try:
            with vane.connect(config={{'threads': 4}}) as con:
                watchdog = threading.Timer(15, con.interrupt)
                watchdog.daemon = True
                watchdog.start()
                try:
                    generator = {source_kind!r} == 'generator'
                    rel = con.sql('select 1::BIGINT as id' if generator else 'select range::BIGINT as id from range(32)')
                    rel = rel.map_batches(
                        expand if generator else (Materialize if {source_backend!r} == 'subprocess_actor' else materialize),
                        schema={{'id': 'BIGINT', 'payload': 'BLOB'}},
                        execution_backend={source_backend!r}, batch_size=1, min_task_batch_size=1,
                        **({{'actor_number': 1}} if {source_backend!r} == 'subprocess_actor' else {{}}),
                        task_input_max_bytes=8, output_batch_size=1, output_target_max_bytes=1024,
                    )
                    if {intermediate!r}:
                        # The last actor retains Copy's output. Source has no
                        # live output of its own when it needs its next input.
                        rel = rel.map_batches(
                            Copy, schema={{'id': 'BIGINT', 'payload': 'BLOB'}},
                            execution_backend='subprocess_actor', actor_number=1,
                            batch_size=1, min_task_batch_size=1, task_input_max_bytes=1024,
                            output_batch_size=1, output_target_max_bytes=1024,
                        )
                    # Four sleeping consumers plus the producer must fit within
                    # the 4-CPU CI runner while retaining every actor replica.
                    rel = rel.map_batches(
                        RetainingConsumer, schema={{'id': 'BIGINT'}},
                        execution_backend='subprocess_actor', actor_number={actors}, cpus=0.5,
                        batch_size=1, min_task_batch_size=1, task_input_max_bytes=1024,
                    )
                    if {arena!r} == '96KiB' and not {copy_input!r}:
                        try:
                            rel.to_arrow_table()
                        except Exception as error:
                            assert 'shared-memory output cannot make progress' in str(error), str(error)
                        else:
                            raise AssertionError('retained working set cannot fit in the arena')
                    else:
                        assert sorted(rel.to_arrow_table()['id'].to_pylist()) == list(range(32))
                finally:
                    watchdog.cancel()
        finally:
            vane.teardown_runner()
        gc.collect()
        assert local_shm_ref_budget_snapshot()['usage_bytes'] == 0
        assert local_shm_budget_manager()._owned_output_grants == {{}}
        assert local_shm_budget_manager()._owned_input_allocations == {{}}
        assert local_shm_store_snapshot()['live_bytes'] == 0
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "VANE_RUNNER": "local-fast",
            "VANE_PROGRESS": "0",
            "VANE_LOCAL_SHM_STORE_BYTES": arena,
            "VANE_LOCAL_SHM_REF_BUDGET_BYTES": "32KiB",
        },
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("arena", ["128KiB", "256KiB"])
@pytest.mark.parametrize("budget", ["32KiB", "auto"])
def test_output_capacity_tracks_queued_stream_when_current_input_has_no_views(tmp_path, arena, budget):
    queued = tmp_path / "queued-input"
    script = textwrap.dedent(f"""
        import gc
        import threading
        import time
        from pathlib import Path
        import pyarrow as pa
        import vane
        from vane.execution import udf_shm_store as storage
        from vane.execution.ref_bundle import local_shm_ref_budget_snapshot

        queued = Path({str(queued)!r})
        waits = []
        wait_for_capacity = storage.LocalShmStore.wait_for_capacity

        def observe_wait(store, size, inputs, *args, **kwargs):
            assert not inputs, 'NULL-only input must not retain a physical view'
            waits.append(size)
            return wait_for_capacity(store, size, inputs, *args, **kwargs)

        storage.LocalShmStore.wait_for_capacity = observe_wait

        def expand(batch):
            yield pa.table({{'payload': [None]}})
            yield pa.table({{'payload': [b'x' * 65536]}})
            queued.touch()

        class Consume:
            def __call__(self, batch):
                if batch['payload'][0].as_py() is None:
                    deadline = time.monotonic() + 10
                    while not queued.exists():
                        if time.monotonic() >= deadline:
                            raise TimeoutError('upstream did not publish its queued input')
                        time.sleep(0.01)
                    return pa.table({{'payload': [b'y' * 65536]}})
                return pa.table({{'payload': [b'done']}})

        try:
            with vane.connect(config={{'threads': 4}}) as con:
                watchdog = threading.Timer(15, con.interrupt)
                watchdog.daemon = True
                watchdog.start()
                try:
                    rel = con.sql('select 1::BIGINT as id').map_batches(
                        expand, schema={{'payload': 'BLOB'}}, execution_backend='subprocess_task',
                        batch_size=1, min_task_batch_size=1, task_input_max_bytes=8,
                        output_batch_size=1, output_target_max_bytes=1024,
                    ).map_batches(
                        Consume, schema={{'payload': 'BLOB'}}, execution_backend='subprocess_actor',
                        actor_number=1, batch_size=1, min_task_batch_size=1, task_input_max_bytes=1024,
                        output_batch_size=1, output_target_max_bytes=1024,
                    )
                    if {arena!r} == '128KiB':
                        try:
                            rel.to_arrow_table()
                        except Exception as error:
                            assert 'shared-memory output cannot make progress' in str(error), str(error)
                            assert 'pinned_input_bytes=0' in str(error), str(error)
                        else:
                            raise AssertionError('queued input leaves no room for the output')
                        assert waits, 'query must exercise an empty-view capacity check'
                    else:
                        result = rel.to_arrow_table()
                        assert result['payload'].to_pylist() == [b'y' * 65536, b'done']
                        del result
                finally:
                    watchdog.cancel()
        finally:
            vane.teardown_runner()
            gc.collect()
            assert storage.local_shm_store_snapshot()['live_bytes'] == 0
            assert local_shm_ref_budget_snapshot()['usage_bytes'] == 0
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "VANE_RUNNER": "local-fast",
            "VANE_PROGRESS": "0",
            "VANE_LOCAL_SHM_STORE_BYTES": arena,
            "VANE_LOCAL_SHM_REF_BUDGET_BYTES": budget,
        },
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "blocks,arena,history", [(2, "192KiB", False), (1, "192KiB", False), (2, "256KiB", False), (4, "320KiB", True)]
)
@pytest.mark.parametrize("budget", ["32KiB", "16MiB", "auto"])
def test_output_capacity_includes_inputs_queued_behind_the_current_call(blocks, arena, history, budget):
    script = textwrap.dedent(f"""
        import gc
        import threading
        import time
        import pyarrow as pa
        import vane
        from vane.execution.ref_bundle import local_shm_ref_budget_snapshot
        from vane.execution.udf_shm_store import local_shm_store_snapshot

        def expand(batch):
            for value in range({blocks}):
                yield pa.table({{'id': [value], 'payload': [b'x' * 65536]}})

        class Copy:
            def __init__(self):
                self.history = []

            def __call__(self, batch):
                if {history!r}:
                    self.history.append(batch)
                # Let the upstream publish the next input while this actor
                # holds its sole execution slot. The history case also
                # keeps previous-call views through the next allocation.
                time.sleep(0.4)
                return pa.table(batch.to_pydict())

        try:
            with vane.connect(config={{'threads': 4}}) as con:
                watchdog = threading.Timer(15, con.interrupt)
                watchdog.daemon = True
                watchdog.start()
                try:
                    rel = con.sql('select 1::BIGINT as id').map_batches(
                        expand, schema={{'id': 'BIGINT', 'payload': 'BLOB'}},
                        execution_backend='subprocess_task', batch_size=1, min_task_batch_size=1,
                        task_input_max_bytes=8, output_batch_size=1, output_target_max_bytes=1024,
                    ).map_batches(
                        Copy, schema={{'id': 'BIGINT', 'payload': 'BLOB'}},
                        execution_backend='subprocess_actor', actor_number=1,
                        batch_size=1, min_task_batch_size=1, task_input_max_bytes=1024,
                        output_batch_size=1, output_target_max_bytes=1024,
                    )
                    if {history!r} or ({blocks} == 2 and {arena!r} == '192KiB'):
                        try:
                            rel.to_arrow_table()
                        except Exception as error:
                            assert 'shared-memory output cannot make progress' in str(error), str(error)
                            if not {history!r}:
                                assert 'queued_input_bytes=69632' in str(error), str(error)
                        else:
                            raise AssertionError('input working set leaves no room for the output')
                    else:
                        result = rel.to_arrow_table()
                        assert sorted(result['id'].to_pylist()) == list(range({blocks}))
                        assert result['payload'].to_pylist() == [b'x' * 65536] * {blocks}
                        del result
                finally:
                    watchdog.cancel()
        finally:
            vane.teardown_runner()
        gc.collect()
        assert local_shm_ref_budget_snapshot()['usage_bytes'] == 0
        assert local_shm_store_snapshot()['live_bytes'] == 0
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "VANE_RUNNER": "local-fast",
            "VANE_PROGRESS": "0",
            "VANE_LOCAL_SHM_STORE_BYTES": arena,
            "VANE_LOCAL_SHM_REF_BUDGET_BYTES": budget,
        },
    )
    assert result.returncode == 0, result.stderr
