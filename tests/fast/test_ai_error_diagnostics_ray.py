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

    ray.init(address="local", num_cpus=4, include_dashboard=False)
    try:
        yield
    finally:
        try:
            vane.teardown_runner()
        finally:
            ray.shutdown()


def _diagnostic_failure(forged):
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

    return fail


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("failure", ["init", "no_outputs", "forged"])
def test_remote_native_prompt_preserves_provider_diagnostic(monkeypatch, diagnostic_query_runtime, engine, failure):
    import uuid

    import vane
    from vane.ai import DiagnosticError, summarize_error
    from vane.ai.providers.vllm import _build_native_vllm_options_argument
    from vane.execution.sglang import SGLangRayLocalExecutor
    from vane.execution.vllm import LLMActors, RayLocalVLLMExecutor

    # Use actual Ray actors and their production engine/remote executors.
    # Only model SDK calls are fixtures: no GPU or downloaded model is needed.
    class EmptyEngine:
        def check(self):
            if failure == "forged":
                secret_type = type("private_fixture_api_key_ABC123", (Exception,), {})
                raise RuntimeError(
                    "vllm take_ready_result failed: DiagnosticError: " + str(DiagnosticError("fixture", secret_type()))
                )

        async def generate(self, *args, **kwargs):
            self.check()
            for output in ():
                yield output

        async def async_generate(self, *args, **kwargs):
            self.check()
            return None

        def shutdown(self):
            pass

    base = RayLocalVLLMExecutor if engine == "vllm" else SGLangRayLocalExecutor

    class Actor(base):
        def _materialize_sampling_params(self, args):
            return {}

        def _create_engine(self):
            if failure == "init":
                raise TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'")
            self.llm = EmptyEngine()

    name = f"native-diagnostic-{uuid.uuid4().hex}"
    actors = LLMActors(
        model="fixture",
        engine_args={},
        generate_args={},
        on_error="raise",
        gpus_per_actor=0,
        concurrency=1,
        load_balance_threshold=0,
        name_prefix=name,
        actor_cls=Actor,
    )
    # The local SQL engine borrows the real Ray pool. This exercises the native
    # callback boundary after RemoteVLLMExecutor, absent from local-model tests.
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    options = _build_native_vllm_options_argument({"use_ray": True, "ray_actor_pool_name": name}, engine=engine)
    try:
        with vane.connect(config={"threads": 1}) as connection:
            with pytest.raises(vane.InvalidInputException) as caught:
                connection.execute("SELECT vllm('hello', 'fixture', ?)", [options]).fetchall()
        summary = summarize_error(caught.value)
        if failure == "no_outputs":
            assert f"{engine} returned no outputs" in summary
        elif failure == "init":
            assert "TypeError" in summary and "max_model_len" in summary
        else:
            assert "RuntimeError" in summary
        assert "private" not in summary
    finally:
        actors.shutdown()


def _diagnostic_sink():
    from vane.datasink import BoundDataSink, DataSink, DataSinkExecutionOptions, DataSinkWorker, WriteResult

    class Worker(DataSinkWorker):
        def write(self, table):
            return WriteResult(rows_received=table.num_rows, rows_affected=table.num_rows)

    class Bound(BoundDataSink):
        @property
        def execution_options(self):
            return DataSinkExecutionOptions(worker_count=1, cpus=1)

        def open_worker(self, context):
            return Worker()

    class Sink(DataSink):
        def bind(self, schema):
            return Bound()

    return Sink()


def _assert_datasink_failure(error, forged):
    import pickle

    from vane.ai import summarize_error
    from vane.datasink import DataSinkWriteError, WriteOutcome

    for _ in range(2):
        assert isinstance(error, DataSinkWriteError) and error.outcome is WriteOutcome.UNKNOWN
        assert error.operation_id == "diagnostic-datasink"
        summary = summarize_error(error)
        assert "private" not in summary
        if not forged:
            assert "max_model_len" in summary
        error = pickle.loads(pickle.dumps(error))


@pytest.mark.parametrize("stage", ["initialization", "batch"])
@pytest.mark.parametrize("output", ["fetchall", "parquet", "datasink"])
@pytest.mark.parametrize("forged", [False, True])
def test_query_failure_survives_initialization_and_copy(
    monkeypatch, tmp_path, diagnostic_query_runtime, stage, output, forged
):
    import pyarrow as pa

    import vane
    from vane.ai import summarize_error

    monkeypatch.delenv("VANE_RUNNER", raising=False)
    fail_diagnostic = _diagnostic_failure(forged)

    @vane.cls.batch(actor_number=1, return_dtype=pa.int32())
    class Initializer:
        def __init__(self):
            fail_diagnostic()

        def __call__(self, values):
            return values

    def failing_batch(table):
        fail_diagnostic()

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
            elif output == "datasink":
                relation.write_datasink(_diagnostic_sink(), operation_id="diagnostic-datasink")
            else:
                relation.fetchall()
        summary = summarize_error(caught.value)
        assert summary and "private" not in summary
        if not forged:
            assert "max_model_len" in summary and "TypeError" in summary
        if output == "datasink":
            _assert_datasink_failure(caught.value, forged)


@pytest.mark.parametrize("output", ["fetchall", "parquet", "datasink"])
@pytest.mark.parametrize("forged", [False, True])
def test_late_actor_initialization_failure_survives_query_transport(
    monkeypatch, tmp_path, diagnostic_query_runtime, output, forged
):
    import asyncio

    import pyarrow as pa
    import ray

    import vane
    from vane.ai import summarize_error

    monkeypatch.delenv("VANE_RUNNER", raising=False)
    fail_diagnostic = _diagnostic_failure(forged)

    @ray.remote(num_cpus=0)
    class Coordinator:
        def __init__(self):
            self.count = 0
            self.started = asyncio.Event()

        async def index(self):
            self.count += 1
            return self.count

        async def wait_started(self):
            await self.started.wait()

        async def mark_started(self):
            self.started.set()

    coordinator = Coordinator.remote()

    @vane.cls.batch(actor_number=2, return_dtype=pa.int32())
    class Initializer:
        def __init__(self):
            if ray.get(coordinator.index.remote(), timeout=30) == 2:
                ray.get(coordinator.wait_started.remote(), timeout=30)
                fail_diagnostic()

        def __call__(self, values):
            ray.get(coordinator.mark_started.remote(), timeout=10)
            # Ensure the readiness failure occurs during an active query,
            # after the first actor has successfully started processing.
            time.sleep(3)
            return values

    try:
        with vane.connect() as connection:
            relation = connection.sql("SELECT range::INTEGER AS value FROM range(8)").select(
                Initializer()(vane.col("value")).alias("value")
            )
            with pytest.raises(Exception) as caught:
                if output == "parquet":
                    relation.write_parquet(str(tmp_path / "late-init.parquet"))
                elif output == "datasink":
                    relation.write_datasink(_diagnostic_sink(), operation_id="diagnostic-datasink")
                else:
                    relation.fetchall()
            summary = summarize_error(caught.value)
            assert summary and "private" not in summary
            if not forged:
                assert "max_model_len" in summary and "TypeError" in summary
            if output == "datasink":
                _assert_datasink_failure(caught.value, forged)
    finally:
        ray.kill(coordinator)


@pytest.mark.parametrize("kind", ["unknown", "unavailable", "terminal_cleanup", "nested_cleanup"])
def test_copy_and_terminal_cleanup_diagnostics_survive_real_ray(diagnostic_query_runtime, kind):
    import ray
    from ray import cloudpickle

    from vane.ai import summarize_error

    @ray.remote(num_cpus=0)
    def fail():
        import errno
        from types import SimpleNamespace

        from vane.ai import DiagnosticError
        from vane.execution._subprocess_error import capture_cleanup
        from vane.runners.copy_outcome import CopyOutcomeUnknownError, CopyResultUnavailableError
        from vane.runners.ray.driver import RayQueryDriverActor, _QueryCleanupDiagnostic

        primary = DiagnosticError(
            "actor initialization",
            TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'"),
        )
        cleanup = OSError(errno.EIO, "private cleanup path")
        if kind == "nested_cleanup":
            primary = capture_cleanup("worker cleanup failed", primary, cleanup)
            cleanup = OSError(errno.ENOSPC, "private query teardown")
        if kind in {"terminal_cleanup", "nested_cleanup"}:

            @ray.remote(num_cpus=0)
            def failed_cleanup():
                raise cleanup

            try:
                ray.get(failed_cleanup.remote(), timeout=15)
            except ray.exceptions.RayTaskError as error:
                cleanup = error

            def teardown(*args):
                raise _QueryCleanupDiagnostic("private cleanup output", (cleanup,))

            runner = SimpleNamespace(_query_terminal_errors={"query": primary}, _teardown_plan_resources=teardown)
            RayQueryDriverActor.__ray_metadata__.modified_class._finish_terminal_query(runner, "plan", "query")
        else:
            error_type = CopyOutcomeUnknownError if kind == "unknown" else CopyResultUnavailableError
            error = error_type("operation")
            error.capture_diagnostics(primary_error=primary, cleanup_errors=(cleanup,))
            raise error from primary

    with pytest.raises(ray.exceptions.RayTaskError) as caught:
        ray.get(fail.remote(), timeout=30)
    for error in (caught.value, cloudpickle.loads(cloudpickle.dumps(caught.value))):
        summary = summarize_error(error)
        assert "max_model_len" in summary and "cleanup: OSError (errno=5)" in summary
        assert "private" not in summary
        if kind == "nested_cleanup":
            primary, cleanup = summary.split("; cleanup: ")
            assert "errno=" not in primary and "errno=28" in cleanup
    if kind in {"unknown", "unavailable"}:
        assert caught.value.cause.operation_id == "operation"
        assert caught.value.cause.safe_to_retry is False
        if kind == "unavailable":
            assert caught.value.cause.write_state == "committed"
