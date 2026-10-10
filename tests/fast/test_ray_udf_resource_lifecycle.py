# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real Ray scheduling with a logical GPU; no CUDA/model is needed."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.real_ray
@pytest.mark.ray_cluster_owner
@pytest.mark.parametrize("scenario", ["concurrent", "timeout", "crashed_client"])
def test_ray_udf_resource_lifecycle(scenario, tmp_path):
    pytest.importorskip("ray")
    script = textwrap.dedent(
        r"""
        import os
        import sys
        import time
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        import ray
        import vane
        from ray_test_profile import ray_test_object_store_options
        from vane.runners.ray.driver import RayQueryDriverClient

        root = Path(sys.argv[1])
        scenario = sys.argv[2]
        os.environ["VANE_RUNNER"] = "ray"
        os.environ["VANE_RAY_ACTOR_INIT_TIMEOUT_S"] = "10" if scenario == "timeout" else "45"
        os.environ["VANE_RAY_CLIENT_LEASE_TIMEOUT_S"] = "5"
        os.environ["VANE_RAY_CLIENT_HEARTBEAT_INTERVAL_S"] = "0.5"
        os.environ["VANE_QUERY_RESOURCE_REFRESH_INTERVAL_S"] = "0.1"
        info = ray.init(num_cpus=8, num_gpus=1, include_dashboard=False,
                        **ray_test_object_store_options())

        def wait_until(check, timeout=40):
            deadline = time.monotonic() + timeout
            while not check():
                if time.monotonic() >= deadline:
                    raise AssertionError("condition did not become true before deadline")
                time.sleep(0.05)

        def query(hold=False):
            class Identity:
                def __call__(self, table):
                    if hold:
                        (root / "entered").touch()
                        wait_until(lambda: (root / "release").exists())
                    return table

            with vane.connect() as connection:
                result = connection.sql("SELECT 42::BIGINT AS value").map_batches(
                    Identity, schema={"value": "BIGINT"}, execution_backend="ray_actor",
                    actor_number=1, cpus=1, gpus=1,
                )
                return result.fetchall()

        peer = RayQueryDriverClient()
        crashed_client = None
        blocker = None
        try:
            if scenario == "concurrent":
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first = executor.submit(query, True)
                    wait_until(lambda: (root / "entered").exists())
                    second = executor.submit(query)
                    wait_until(lambda: ray.get(peer.runner.runtime_lifecycle_snapshot.remote(
                        peer._owner_id))["plan_count"] >= 2)
                    assert not second.done()
                    (root / "release").touch()
                    assert first.result(timeout=40) == [(42,)]
                    assert second.result(timeout=40) == [(42,)]
            elif scenario == "timeout":
                @ray.remote(num_cpus=0, num_gpus=1)
                class OccupyGPU:
                    def ready(self):
                        return True
                blocker = OccupyGPU.remote()
                assert ray.get(blocker.ready.remote())
                started = time.monotonic()
                try:
                    query()
                except Exception as error:
                    assert "initialization timed out" in str(error), str(error)
                    assert "GPU=1" in str(error), str(error)
                else:
                    raise AssertionError("unavailable GPU must fail within the init deadline")
                assert time.monotonic() - started < 45
                ray.kill(blocker, no_restart=True)
                blocker = None
                wait_until(lambda: ray.available_resources().get("GPU", 0) == 1)
                assert query() == [(42,)]
            else:
                # Kill a client process within the same Ray job. A surviving
                # client keeps the shared query driver alive, so Ray's job
                # teardown cannot hide failures in Vane's client lease reaper.
                @ray.remote(num_cpus=0)
                class CrashClient:
                    def run(self):
                        return query(True)
                crashed_client = CrashClient.options(runtime_env={"env_vars": {
                    key: value for key, value in os.environ.items() if key.startswith("VANE_")
                }}).remote()
                result_ref = crashed_client.run.remote()
                wait_until(lambda: (root / "entered").exists())
                before = ray.get(peer.runner.runtime_lifecycle_snapshot.remote(peer._owner_id))
                assert before["client_count"] == 2, before
                assert before["plan_count"] == 1, before
                assert ray.available_resources().get("GPU", 0) == 0
                ray.kill(crashed_client, no_restart=True)
                crashed_client = None
                wait_until(lambda: ray.available_resources().get("GPU", 0) == 1, timeout=25)
                wait_until(lambda: ray.get(peer.runner.runtime_lifecycle_snapshot.remote(
                    peer._owner_id))["client_count"] == 1, timeout=25)
                snapshot = ray.get(peer.runner.runtime_lifecycle_snapshot.remote(peer._owner_id))
                assert snapshot["session_count"] == 0, snapshot
                assert snapshot["plan_count"] == 0, snapshot
                assert query() == [(42,)]
            print("lifecycle scenario passed:", scenario, flush=True)
        finally:
            (root / "release").touch()
            if crashed_client is not None:
                ray.kill(crashed_client, no_restart=True)
            if blocker is not None:
                ray.kill(blocker, no_restart=True)
            peer.close()
            ray.shutdown()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), scenario],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", ""))),
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"lifecycle scenario passed: {scenario}" in result.stdout
