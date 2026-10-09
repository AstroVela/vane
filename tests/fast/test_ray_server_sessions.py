# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Server-owned session retirement closes actual Ray and native resources."""

import time

import pytest

from tests.fast.test_ray_recovery_runtime import assert_idle, close_result, resources
from tests.fast.test_server_sessions import identity, wait_until
from vane.execution.server_session import SessionLimits, SessionService

pytestmark = [pytest.mark.real_ray, pytest.mark.usefixtures("ray_local")]


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.parametrize("expire", [False, True])
def test_session_retirement_releases_query_but_preserves_shared_service(tmp_path, mode, expire):
    service = SessionService(resources=resources(tmp_path), limits=SessionLimits(lease_seconds=120))
    result = None
    try:
        handle = service.open_session(execution=mode)
        survivor = service.open_session()
        owner = service._sessions[handle["session_id"]].owner
        other = service._sessions[survivor["session_id"]].owner._connection
        # Query ingress is the next increment. Exercise the real native session
        # owner here so lease tests cannot pass by closing empty fake sessions.
        result = owner._connection.query("select range from range(10000)", rows_per_batch=64)
        assert service.runtime.resource_snapshot()["service"]["result_delivery"]["active_results"] == 1
        epochs = tuple(owner.pool.epochs)
        if expire:
            with service._condition:
                service._sessions[handle["session_id"]].expires_at = time.monotonic() - 1
                service._condition.notify_all()
        else:
            service.close_session(*identity(handle))
        wait_until(lambda: service.snapshot()["sessions"] == 1, timeout=20)
        assert_idle(other)
        assert other.query("select 7").collect().column(0).to_pylist() == [7]
        assert tuple(other.query_runtime.pool.epochs) == epochs
        assert_idle(other)
    finally:
        if result is not None:
            close_result(result)
        service.close(timeout=20)
