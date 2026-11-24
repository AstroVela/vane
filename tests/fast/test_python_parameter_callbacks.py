# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded cross-connection probes for implicit Python parameter callbacks."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("action", ["sql", "close", "propagate"])
@pytest.mark.parametrize(
    ("entry", "phase"),
    [
        ("execute", "length"),
        ("execute", "iterator"),
        ("execute", "nested"),
        ("execute", "key"),
        ("execute", "mapping"),
        ("sql", "length"),
        ("executemany", "outer_length"),
        ("executemany", "outer_iterator"),
        ("executemany", "outer_generator"),
        ("executemany", "length"),
        ("executemany", "prepared_length"),
        ("values", "length"),
        ("table_function", "length"),
    ],
)
def test_parameter_callbacks_reject_cross_connection_entry(monkeypatch, entry, phase, action, configured):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import faulthandler
        import sys
        import threading
        from collections.abc import Mapping
        from concurrent.futures import ThreadPoolExecutor
        import vane
        from vane.execution.request_admission import RequestAdmissionLimits

        entry, phase, action, configured = sys.argv[1:]
        faulthandler.dump_traceback_later(15, exit=True)
        connections = [vane.connect(config={"threads": 1}) for _ in range(2)]
        runtimes = []
        if configured == "True":
            runtimes = [con.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
                        for con in connections]
        barrier = threading.Barrier(2, timeout=5)
        attempted = [False, False]
        rejected = [False, False]

        def callback(index):
            if attempted[index]:
                return
            attempted[index] = True
            # Both outer operations retain their connection locks here.
            barrier.wait()
            target = connections[1 - index]
            if action == "propagate":
                target.execute("SELECT 42")
                raise AssertionError("parameter callback entered a connection")
            try:
                if action == "sql":
                    target.sql("SELECT 42")
                else:
                    target.close()
            except vane.InvalidInputException as error:
                assert "Python input callback" in str(error), str(error)
                rejected[index] = True
            else:
                raise AssertionError("parameter callback entered a connection")

        def run(index):
            class Parameters(list):
                def __len__(self):
                    if phase not in ("iterator", "outer_iterator", "outer_generator"):
                        callback(index)
                    return super().__len__()
                def __iter__(self):
                    if phase in ("iterator", "outer_iterator"):
                        callback(index)
                    return super().__iter__()

            class ParameterName:
                def __str__(self):
                    callback(index)
                    return "value"

            class ParameterSets:
                def __len__(self):
                    if phase == "outer_length":
                        callback(index)
                    return 1
                def __iter__(self):
                    if phase == "outer_iterator":
                        callback(index)
                    return iter([[index]])

            class ParametersMapping(Mapping):
                def __len__(self):
                    return 1
                def __iter__(self):
                    callback(index)
                    yield "value"
                def __getitem__(self, name):
                    return index

            query = "SELECT ?::BIGINT"
            params = Parameters([index])
            if phase == "nested":
                query, params = "SELECT ?[1]", [Parameters([index])]
            elif phase == "key":
                query, params = "SELECT $value", {ParameterName(): index}
            elif phase == "mapping":
                query, params = "SELECT $value", ParametersMapping()
            connection = connections[index]
            try:
                if entry == "sql":
                    result = connection.sql(query, params=params)
                elif entry == "executemany":
                    if phase in ("outer_length", "outer_iterator"):
                        parameter_sets = ParameterSets()
                    elif phase == "outer_generator":
                        def sets():
                            callback(index)
                            yield [index]
                        parameter_sets = sets()
                    elif phase == "prepared_length":
                        # The second set takes the reused PreparedStatement path.
                        parameter_sets = [[index], params]
                    else:
                        parameter_sets = [params]
                    result = connection.executemany(query, parameter_sets)
                elif entry == "values":
                    result = connection.values(params)
                elif entry == "table_function":
                    result = connection.table_function("range", params)
                else:
                    result = connection.execute(query, params)
            except vane.InvalidInputException as error:
                assert action == "propagate", str(error)
                assert "Python input callback" in str(error), str(error)
                rejected[index] = True
            else:
                assert action != "propagate", "callback error was swallowed"
                expected = [(i,) for i in range(index)] if entry == "table_function" else [(index,)]
                assert result.fetchall() == expected

        with ThreadPoolExecutor(2) as pool:
            list(pool.map(run, range(2)))
        assert attempted == rejected == [True, True], (attempted, rejected)
        for connection in connections:
            assert connection.execute("SELECT 7").fetchall() == [(7,)]
        for runtime in runtimes:
            state = runtime.resource_snapshot()["request_admission"]
            assert state["active_requests"] == state["cleanup_pending_requests"] == 0, state
            assert not state["draining"], state
        for connection in connections:
            connection.close()
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, entry, phase, action, str(configured)],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
