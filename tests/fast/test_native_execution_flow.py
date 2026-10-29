# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Function-independent flow, including a real native -> UDF -> native handoff."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


def _environment(**overrides):
    env = {**os.environ, "VANE_RUNNER": "local-fast", "VANE_PROGRESS": "0"}
    for name in (
        "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES",
        "VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES",
        "VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES",
    ):
        env.pop(name, None)
    env.update(VANE_NATIVE_ENABLE_EXECUTION_FLOW="1", VANE_NATIVE_MEDIA_ENABLE_SCALAR_ADMISSION="0")
    env.update(overrides)
    return env


def _run(script, *, env=None, timeout=90):
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(script)],
        env=_environment() if env is None else env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("flag", ["0", "1", "yes"])
def test_native_flow_configuration_and_host_capture(flag):
    env = _environment(VANE_NATIVE_ENABLE_EXECUTION_FLOW=flag)
    script = """
        import os
        import vane
        con = vane.connect()
        initial = os.environ['VANE_NATIVE_ENABLE_EXECUTION_FLOW'] == '1'
        os.environ['VANE_NATIVE_ENABLE_EXECUTION_FLOW'] = '0' if initial else '1'
        rel = con.sql('SELECT i + 1 AS x FROM range(4)t(i)')
        assert ('NATIVE_PROJECT' in rel.explain()) == initial, rel.explain()
        assert rel.fetchall() == [(1,), (2,), (3,), (4,)]
        con.close()
    """
    if flag == "yes":
        result = subprocess.run(
            [sys.executable, "-I", "-c", "import vane; vane.connect()"],
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "VANE_NATIVE_ENABLE_EXECUTION_FLOW must be 0 or 1" in result.stderr
    else:
        _run(script, env=env)


def test_native_flow_rejects_unimplemented_combined_progress_reservations():
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import vane; vane.connect()"],
        env=_environment(
            VANE_NATIVE_MEDIA_ENABLE_SCALAR_ADMISSION="1",
            VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES="8192",
            VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES="64",
        ),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "must be enabled separately" in result.stderr


def test_native_flow_preserves_nested_case_payloads_and_result_lifetimes():
    _run("""
        import time
        import pyarrow as pa
        import vane
        from vane.execution.request_admission import RequestAdmissionLimits
        from vane.execution.result_delivery import ResultDeliveryLimits

        with vane.connect(config={'threads': 4}) as con:
            con.execute("SET streaming_buffer_size='512B'")
            con.configure_local_runtime(
                request_limit=RequestAdmissionLimits(1, 1),
                result_limit=ResultDeliveryLimits(2, 1048576))
            def query():
                return con.sql('''
                    SELECT i, CASE WHEN i%2=0 THEN upper('row-' || i::VARCHAR) ELSE NULL END AS label,
                           [i, i+1] AS indices,
                           {'id': i, 'text': repeat('x', (i%3+1)::INTEGER*31)} AS nested
                    FROM range(24)t(i)
                ''')
            assert 'NATIVE_PROJECT' in query().explain()
            expected = [(i, None if i%2 else 'ROW-' + str(i), [i, i+1],
                         {'id': i, 'text': 'x' * ((i%3+1)*31)}) for i in range(24)]
            for _ in range(2):
                assert query().fetchall() == expected
                assert query().limit(2).fetchall() == expected[:2]
            arrow = query().to_arrow_table()
            capsule = pa.table(query())
            assert arrow.equals(capsule)
            result = query().execute_result(stream=True, rows_per_batch=1000000)
            retained = result.take()
            assert retained.num_rows == 1
            before = retained.to_pylist()
            time.sleep(0.05)
            result.close()
            assert query().fetchall() == expected
            assert con.sql('SELECT i+1 FROM range(0)t(i)').fetchall() == []
        assert retained.to_pylist() == before
        assert arrow.equals(capsule)
        assert arrow['i'].to_pylist() == list(range(24))
        vane.teardown_runner()
    """)


@pytest.mark.parametrize("failure", ["input", "output", "kernel"])
def test_native_flow_failure_is_terminal_and_connection_recovers(failure):
    query = {
        "input": "SELECT upper(s) FROM input_data",
        "output": "SELECT repeat('x', (i+1)::INTEGER*300) FROM range(1)t(i)",
        "kernel": "SELECT error('injected native execution failure ' || i::VARCHAR) FROM range(1)t(i)",
    }[failure]
    expected = {
        "input": "input row exceeds",
        "output": "output row exceeds",
        "kernel": "injected native execution failure",
    }[failure]
    _run(f"""
        import pyarrow as pa
        import vane
        with vane.connect() as con:
            con.execute("SET streaming_buffer_size='128B'")
            if {failure == "input"!r}:
                con.register('input_data', pa.table({{'s': ['x'*300]}}))
            try:
                con.sql({query!r}).fetchall()
            except Exception as exc:
                assert {expected!r} in str(exc), str(exc)
            else:
                raise AssertionError('failed query published output')
            assert con.sql('SELECT i+1 FROM range(4)t(i)').fetchall() == [(1,), (2,), (3,), (4,)]
        vane.teardown_runner()
    """)


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_native_flow_crosses_real_udf_boundary_and_reuses_after_close(backend):
    _run(
        f"""
        import time
        import pyarrow as pa
        import vane
        from vane.execution.request_admission import RequestAdmissionLimits
        from vane.execution.result_delivery import ResultDeliveryLimits
        from vane.execution.udf_runtime_admission import TaskAdmissionLimits

        def forward(batch):
            time.sleep(0.001)
            return batch

        class Forward:
            def __call__(self, batch):
                return forward(batch)

        with vane.connect(config={{'threads': 4}}) as con:
            con.execute("SET streaming_buffer_size='512B'")
            con.configure_local_runtime(
                request_limit=RequestAdmissionLimits(1, 1),
                task_limit=TaskAdmissionLimits(2, 4),
                result_limit=ResultDeliveryLimits(2, 1048576))
            def query():
                source = con.sql('''
                    SELECT i+1 AS id, repeat('x', (i%3+1)::INTEGER*31) AS payload, [i,i+1] AS indices
                    FROM range(36)t(i)
                ''')
                middle = source.map_batches(
                    Forward if {backend!r}.endswith('_actor') else forward,
                    schema={{'id': 'BIGINT', 'payload': 'VARCHAR', 'indices': 'BIGINT[]'}},
                    execution_backend={backend!r}, batch_size=2, output_batch_size=2,
                    min_task_batch_size=2, task_input_max_bytes=192,
                    output_target_max_bytes=192,
                    **({{'actor_number': 1}} if {backend!r}.endswith('_actor') else {{}}))
                return middle.project('id*7 AS value, upper(payload) AS payload, indices[2] AS next')

            expected = [(i*7, 'X' * (((i-1)%3+1)*31), i) for i in range(1,37)]
            assert query().explain().count('NATIVE_PROJECT') >= 2
            for _ in range(2):
                assert sorted(query().fetchall()) == expected
            stream = query().execute_result(stream=True, rows_per_batch=1000000)
            streamed = []
            try:
                while True:
                    try:
                        batch = stream.take()
                    except StopIteration:
                        break
                    streamed.extend(zip(batch['value'].to_pylist(), batch['payload'].to_pylist(),
                                        batch['next'].to_pylist()))
            finally:
                stream.close()
            assert sorted(streamed) == expected
            result = query().execute_result(stream=True, rows_per_batch=1000000)
            retained = result.take()
            before = retained.to_pylist()
            time.sleep(0.05)
            result.close()
            assert sorted(query().fetchall()) == expected
        assert retained.to_pylist() == before
        vane.teardown_runner()
    """,
        timeout=120,
    )


def test_native_flow_sorted_udf_handoff_requires_an_explicit_receiver():
    _run("""
        import vane

        def forward(batch):
            return batch

        with vane.connect() as con:
            con.execute("SET streaming_buffer_size='256B'")
            source = con.sql('SELECT i+1 AS x FROM range(8)t(i)').order('x')
            relation = source.map_batches(
                forward, schema={'x': 'BIGINT'}, execution_backend='subprocess_task',
                batch_size=1, min_task_batch_size=1,
                task_input_max_bytes=256, output_target_max_bytes=256)
            try:
                relation.fetchall()
            except Exception as exc:
                assert 'materialization budget handoff' in str(exc), str(exc)
            else:
                raise AssertionError('sorted source bypassed the flow handoff guard')
            assert con.sql('SELECT i+1 FROM range(2)t(i)').fetchall() == [(1,), (2,)]
        vane.teardown_runner()
    """)
