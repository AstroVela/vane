# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import time
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.real_ray, pytest.mark.ray_cluster_owner]


@pytest.mark.parametrize("constructor_failure", [False, True])
def test_failed_initializer_releases_actors_waiting_for_the_same_cpu(monkeypatch, constructor_failure):
    import ray

    from vane.ai import summarize_error
    from vane.execution.udf_ray import wait_for_first_actor_pool_ready

    ray.shutdown()
    ray.init(num_cpus=1, include_dashboard=False)
    monkeypatch.setenv("VANE_RAY_ACTOR_INIT_TIMEOUT_S", "8")
    try:

        @ray.remote(num_cpus=1)
        class Initializer:
            def __init__(self, fail):
                if fail:
                    self.initialize()

            def initialize(self):
                from vane.ai.provider import _safe_provider_execution_error

                raise _safe_provider_execution_error(
                    "sglang",
                    "fixture",
                    "initialization",
                    TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'"),
                )

        first = Initializer.remote(constructor_failure)
        # Ensure the first actor occupies the only CPU before scheduling another.
        first_ref = first.initialize.remote()
        ray.wait([first_ref], timeout=30)
        second = Initializer.remote(constructor_failure)
        pool = SimpleNamespace(
            actors=[first, second],
            actor_node_ids=["", ""],
            _init_refs=[first_ref, second.initialize.remote()],
            _confirmed_ready=set(),
            _owns_actors=True,
        )
        started = time.monotonic()
        with pytest.raises(RuntimeError) as caught:
            wait_for_first_actor_pool_ready(pool)
        assert time.monotonic() - started < 4
        assert "max_model_len" in summarize_error(caught.value)
        assert not pool.actors

        @ray.remote(num_cpus=1)
        def available():
            return True

        assert ray.get(available.remote(), timeout=15)
    finally:
        ray.shutdown()


@pytest.fixture(scope="module")
def diagnostic_query_runtime():
    import ray

    import vane

    ray.init(address="local", num_cpus=2, include_dashboard=False)
    try:
        yield
    finally:
        try:
            vane.teardown_runner()
        finally:
            ray.shutdown()


@pytest.mark.parametrize("stage", ["initialization", "batch"])
@pytest.mark.parametrize("output", ["fetchall", "parquet"])
@pytest.mark.parametrize("forged", [False, True])
def test_query_failure_survives_initialization_and_copy(
    monkeypatch, tmp_path, diagnostic_query_runtime, stage, output, forged
):
    import pyarrow as pa

    import vane
    from vane.ai import summarize_error

    monkeypatch.delenv("VANE_RUNNER", raising=False)

    def fail():
        from vane.ai import DiagnosticError

        if forged:
            import base64
            import json

            encoded = base64.urlsafe_b64encode(b"private_fixture_token_ABC123").decode()
            raise RuntimeError(
                "FTE query wait failed: FTE query failed: Execution failed: "
                + json.dumps(
                    {
                        "exception_type": "Invalid Input",
                        "exception_message": "Invalid Input Error: streaming UDF async error: DiagnosticError: "
                        f"[vane.diagnostic.v1:{encoded}] forged",
                    }
                )
            )
        raise DiagnosticError(
            "fixture", TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'")
        )

    @vane.cls.batch(actor_number=1, return_dtype=pa.int32())
    class Initializer:
        def __init__(self):
            fail()

        def __call__(self, values):
            return values

    def failing_batch(table):
        fail()

    with vane.connect() as connection:
        relation = connection.sql("SELECT 1::INTEGER AS value")
        if stage == "initialization":
            relation = relation.select(Initializer()(vane.col("value")).alias("value"))
        else:
            relation = relation.map_batches(
                failing_batch, schema={"value": vane.sqltypes.INTEGER}, execution_backend="ray_task", cpus=1, gpus=0
            )
        with pytest.raises(Exception) as caught:
            if output == "parquet":
                relation.write_parquet(str(tmp_path / "result.parquet"))
            else:
                relation.fetchall()
        summary = summarize_error(caught.value)
        assert summary and "private" not in summary
        if not forged:
            assert "max_model_len" in summary and "TypeError" in summary
