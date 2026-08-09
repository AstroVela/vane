# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections import deque

import pytest

pa = pytest.importorskip("pyarrow")

_EMPTY_NATIVE_VLLM_OPTIONS_SQL = """struct_pack(
    __vane_vllm_payload_version := 1,
    __vane_vllm_public_options_json := '{}',
    __vane_vllm_secret_payload := encode('{"payload_version":1,"values":[]}')
)"""


def _packed_native_vllm_options(options):
    from vane.ai.providers.vllm import _build_native_vllm_options_argument

    return _build_native_vllm_options_argument(options)


class _RecordingExecutor:
    def __init__(self) -> None:
        self.submissions: list[tuple[str | None, tuple[str, ...]]] = []
        self.ready = deque()
        self.finished = False
        self.finished_count = 0
        self.invalid_wait = False
        self.wakeup_callbacks = []
        self.wakeup_registrations = 0

    def submit(self, prefix, prompts, rows) -> None:
        prompt_values = tuple(prompts)
        self.submissions.append((prefix, prompt_values))
        self.ready.append(([f"generated:{prompt}" for prompt in prompt_values], rows))
        self._notify_wakeups()

    def take_ready_result(self):
        try:
            return self.ready.popleft()
        except IndexError:
            return None

    def finished_submitting(self) -> None:
        self.finished_count += 1
        self.finished = True
        self._notify_wakeups()

    def all_tasks_finished(self) -> bool:
        return self.finished and not self.ready

    def wait_for_result(self) -> None:
        if not self.ready and not self.finished:
            self.invalid_wait = True
            raise AssertionError("wait_for_result called with no inflight work")

    def register_wakeup_callback(self, callback) -> bool:
        self.wakeup_registrations += 1
        if self.ready or self.all_tasks_finished():
            return False
        self.wakeup_callbacks.append(callback)
        return True

    def _notify_wakeups(self) -> None:
        if not self.ready and not self.all_tasks_finished():
            return
        callbacks, self.wakeup_callbacks = self.wakeup_callbacks, []
        for callback in callbacks:
            callback()

    def shutdown(self) -> None:
        self.finished = True


class _DeferredWakeupExecutor(_RecordingExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.pending = deque()
        self.callback_armed = threading.Event()
        self.pending_ready = threading.Event()
        self.callback_invocations = 0

    def submit(self, prefix, prompts, rows) -> None:
        prompt_values = tuple(prompts)
        self.submissions.append((prefix, prompt_values))
        self.pending.append(([f"generated:{prompt}" for prompt in prompt_values], rows))
        self.pending_ready.set()

    def all_tasks_finished(self) -> bool:
        return self.finished and not self.pending and not self.ready

    def register_wakeup_callback(self, callback) -> bool:
        self.wakeup_registrations += 1
        if self.ready or self.all_tasks_finished():
            return False
        self.wakeup_callbacks.append(callback)
        self.callback_armed.set()
        return True

    def publish_results(self) -> None:
        self.ready.extend(self.pending)
        self.pending.clear()
        callbacks, self.wakeup_callbacks = self.wakeup_callbacks, []
        for callback in callbacks:
            self.callback_invocations += 1
            callback()


class _ChunkedBatchResultExecutor(_RecordingExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.returned_arrow_batch_counts = []

    def submit(self, prefix, prompts, rows) -> None:
        prompt_values = tuple(prompts)
        self.submissions.append((prefix, prompt_values))
        outputs = []
        for prompt in prompt_values:
            row_id = int(prompt.rsplit("-", 1)[1])
            outputs.append(None if row_id % 7 == 0 else f"generated:{prompt}")
        batches = rows.to_batches(max_chunksize=511)
        self.returned_arrow_batch_counts.append(len(batches))
        self.ready.append((outputs, pa.Table.from_batches(batches)))
        self._notify_wakeups()


def _run_recording_sql(monkeypatch, prompts, options, *, executor=None, threads=1, query_suffix=""):
    import vane
    import vane.execution.vllm as vllm

    executor = executor or _RecordingExecutor()
    monkeypatch.setattr(vllm, "build_executor", lambda *_args, **_kwargs: executor)
    con = vane.connect()
    try:
        con.execute(f"PRAGMA threads={threads}")
        con.register(
            "vllm_input",
            pa.table(
                {
                    "id": pa.array(range(len(prompts)), type=pa.int64()),
                    "prompt": pa.array(list(prompts), type=pa.string()),
                }
            ),
        )
        packed = _packed_native_vllm_options(options)
        rows = con.execute(
            "SELECT id, prompt, vllm(prompt, 'recording-model', ?) AS generated FROM vllm_input" + query_suffix,
            [packed],
        ).fetchall()
        return executor, rows
    finally:
        con.close()


@pytest.mark.parametrize("do_prefix_routing", [False, True])
def test_native_vllm_propagates_null_prompts_without_submitting_them(monkeypatch, do_prefix_routing):
    executor, rows = _run_recording_sql(
        monkeypatch,
        ["alpha", None, "beta"],
        {
            "do_prefix_routing": do_prefix_routing,
            "max_buffer_size": 0,
            "min_bucket_size": 1,
            "batch_size": None,
            "inflight_limit": 0,
        },
    )

    assert {row[0]: row[2] for row in rows} == {
        0: "generated:alpha",
        1: None,
        2: "generated:beta",
    }
    assert sorted(prompt for _, prompts in executor.submissions for prompt in prompts) == ["alpha", "beta"]


def test_native_vllm_all_null_prompts_do_not_build_an_executor(monkeypatch):
    import vane
    import vane.execution.vllm as vllm

    builds = 0

    def build_executor(*_args, **_kwargs):
        nonlocal builds
        builds += 1
        return _RecordingExecutor()

    monkeypatch.setattr(vllm, "build_executor", build_executor)
    con = vane.connect()
    try:
        con.register(
            "vllm_input",
            pa.table(
                {
                    "id": pa.array([0, 1], type=pa.int64()),
                    "prompt": pa.array([None, None], type=pa.string()),
                }
            ),
        )
        rows = con.execute(f"""
            SELECT id, vllm(prompt, 'recording-model', {_EMPTY_NATIVE_VLLM_OPTIONS_SQL}) AS generated
            FROM vllm_input
            ORDER BY id
        """).fetchall()
    finally:
        con.close()

    assert rows == [(0, None), (1, None)]
    assert builds == 0


@pytest.mark.parametrize(
    ("prompts", "expected_prefix"),
    [
        (["abc1", "abc2"], "abc"),
        (["你好甲", "你好乙"], "你好"),
        (["🙂alpha", "🙂alpine"], "🙂alp"),
        (["same", "same"], "same"),
        (["alpha", "zulu"], None),
        (["", ""], None),
    ],
)
def test_native_bucket_prefix_ends_on_a_complete_utf8_character(monkeypatch, prompts, expected_prefix):
    executor, rows = _run_recording_sql(
        monkeypatch,
        prompts,
        {
            "do_prefix_routing": True,
            "max_buffer_size": 0,
            "min_bucket_size": 2,
            "prefix_match_threshold": 0.3,
            "batch_size": None,
            "inflight_limit": 0,
        },
    )

    assert [submission[0] for submission in executor.submissions] == [expected_prefix]
    assert {row[0]: row[2] for row in rows} == {index: f"generated:{prompt}" for index, prompt in enumerate(prompts)}


def test_native_bridge_rejects_zero_batch_size_even_if_python_normalization_is_bypassed(monkeypatch):
    import vane
    import vane.execution.vllm as vllm

    invalid = vllm.normalize_options(_packed_native_vllm_options({}))
    invalid["batch_size"] = 0
    monkeypatch.setattr(vllm, "normalize_options", lambda _options: invalid)
    monkeypatch.setattr(vllm, "build_executor", lambda *_args, **_kwargs: _RecordingExecutor())

    con = vane.connect()
    try:
        with pytest.raises(Exception, match="batch_size"):
            con.execute(f"SELECT vllm('hello', 'recording-model', {_EMPTY_NATIVE_VLLM_OPTIONS_SQL})").fetchall()
    finally:
        con.close()


def test_native_bridge_normalizes_json_numeric_options(monkeypatch):
    import vane
    import vane.execution.vllm as vllm

    executor = _RecordingExecutor()
    normalized = {}

    def build_executor(_model, options):
        normalized.update(options)
        return executor

    monkeypatch.setattr(vllm, "build_executor", build_executor)

    con = vane.connect()
    try:
        row = con.execute(
            """
            SELECT vllm(
                'hello',
                'recording-model',
                ?
            )
            """,
            [
                _packed_native_vllm_options(
                    {"prefix_match_threshold": 0.33, "gpus_per_actor": 0.25, "engine_init_timeout_s": 1.5}
                )
            ],
        ).fetchone()
    finally:
        con.close()

    assert row == ("generated:hello",)
    assert normalized["prefix_match_threshold"] == pytest.approx(0.33)
    assert normalized["gpus_per_actor"] == pytest.approx(0.25)
    assert normalized["engine_init_timeout_s"] == pytest.approx(1.5)
    assert type(normalized["prefix_match_threshold"]) is float
    assert type(normalized["gpus_per_actor"]) is float
    assert type(normalized["engine_init_timeout_s"]) is float


def test_native_bridge_preserves_json_boolean_options(monkeypatch):
    import vane
    import vane.execution.vllm as vllm

    executor = _RecordingExecutor()
    normalized = {}

    def build_executor(_model, options):
        normalized.update(options)
        return executor

    monkeypatch.setattr(vllm, "build_executor", build_executor)
    option_names = (
        "do_prefix_routing",
        "use_ray",
        "use_threading",
        "require_ray_worker",
        "ray_worker_only",
    )

    con = vane.connect()
    try:
        row = con.execute(
            """
            SELECT vllm(
                'hello',
                'recording-model',
                ?
            )
            """,
            [_packed_native_vllm_options(dict.fromkeys(option_names, False))],
        ).fetchone()
    finally:
        con.close()

    assert row == ("generated:hello",)
    assert {name: normalized[name] for name in option_names} == dict.fromkeys(option_names, False)
    assert all(type(normalized[name]) is bool for name in option_names)


@pytest.mark.parametrize(
    "name",
    [
        "use_ray",
        "use_threading",
        "require_ray_worker",
        "ray_worker_only",
        "_force_background_thread",
    ],
)
def test_native_bridge_rejects_non_boolean_execution_options(monkeypatch, name):
    import vane
    import vane.execution.vllm as vllm

    monkeypatch.setattr(vllm, "build_executor", lambda *_args, **_kwargs: _RecordingExecutor())

    con = vane.connect()
    try:
        with pytest.raises(Exception, match=rf"vllm {name} must be a boolean"):
            con.execute(
                "SELECT vllm('hello', 'recording-model', ?)",
                [_packed_native_vllm_options({name: "false"})],
            ).fetchall()
    finally:
        con.close()


@pytest.mark.timeout(30)
def test_native_finalizer_blocks_and_resumes_through_a_one_shot_callback(monkeypatch):
    executor = _DeferredWakeupExecutor()
    publisher_errors = []

    def publish_after_arm() -> None:
        try:
            assert executor.callback_armed.wait(timeout=20), "native finalizer did not arm a wakeup callback"
            assert executor.pending_ready.wait(timeout=20), "native producer did not submit a pending result"
            executor.publish_results()
        except BaseException as exc:
            publisher_errors.append(exc)

    publisher = threading.Thread(target=publish_after_arm, name="vllm-test-result-publisher")
    publisher.start()
    try:
        _, rows = _run_recording_sql(
            monkeypatch,
            ["prefix-alpha", "prefix-beta"],
            {
                "do_prefix_routing": True,
                "max_buffer_size": 0,
                "min_bucket_size": 2,
                "batch_size": None,
                "inflight_limit": 0,
            },
            executor=executor,
            threads=2,
        )
    finally:
        publisher.join(timeout=5)

    assert not publisher.is_alive()
    assert publisher_errors == []
    assert {row[0]: row[2] for row in rows} == {
        0: "generated:prefix-alpha",
        1: "generated:prefix-beta",
    }
    assert executor.wakeup_registrations >= 1
    assert executor.callback_invocations >= 1
    assert executor.finished_count == 1
    assert not executor.pending
    assert not executor.invalid_wait


def test_native_downstream_limit_retires_producer_when_final_execute_is_skipped(monkeypatch):
    prompts = [f"prompt-{index}" for index in range(5000)]
    executor, rows = _run_recording_sql(
        monkeypatch,
        prompts,
        {
            "do_prefix_routing": False,
            "batch_size": None,
            "inflight_limit": 0,
        },
        threads=1,
        query_suffix=" LIMIT 1",
    )

    assert rows == [(0, "prompt-0", "generated:prompt-0")]
    assert sum(len(submitted) for _, submitted in executor.submissions) < len(prompts)
    assert executor.finished_count == 1


@pytest.mark.parametrize("row_count", [2049, 6144])
def test_native_vllm_splits_oversized_ready_results_before_downstream_operators(monkeypatch, row_count):
    import vane
    import vane.execution.vllm as vllm

    executor = _ChunkedBatchResultExecutor()
    observed_type = pa.struct([pa.field("value", pa.string()), pa.field("chunk_size", pa.int64())])

    @vane.func.batch(return_dtype=observed_type)
    def observe_chunk(values):
        chunk_size = len(values)
        return pa.StructArray.from_arrays(
            [
                pa.array(values.to_pylist(), type=pa.string()),
                pa.array([chunk_size] * chunk_size, type=pa.int64()),
            ],
            fields=list(observed_type),
        )

    monkeypatch.setattr(vllm, "build_executor", lambda *_args, **_kwargs: executor)
    prompts = [f"shared-prefix-{row_id:05d}" for row_id in range(row_count)]
    con = vane.connect()
    try:
        con.execute("PRAGMA threads=1")
        con.register(
            "vllm_input",
            pa.table(
                {
                    "id": pa.array(range(row_count), type=pa.int64()),
                    "prompt": pa.array(prompts, type=pa.string()),
                }
            ),
        )
        vane.attach_function(
            observe_chunk,
            alias="observe_vllm_chunk",
            connection=con,
            parameters=["VARCHAR"],
        )
        rows = con.execute(
            """
            SELECT id, observe_vllm_chunk(generated) AS observed
            FROM (
                SELECT id, vllm(prompt, 'recording-model', ?) AS generated
                FROM vllm_input
            )
            """,
            [
                _packed_native_vllm_options(
                    {
                        "do_prefix_routing": True,
                        "max_buffer_size": 5000,
                        "min_bucket_size": 1,
                        "batch_size": None,
                        "inflight_limit": 0,
                    }
                )
            ],
        ).fetchall()
    finally:
        con.close()

    vector_size = vane.__standard_vector_size__
    assert len(rows) == row_count
    assert max(observed["chunk_size"] for _, observed in rows) <= vector_size
    assert [len(submitted) for _, submitted in executor.submissions] == [row_count]
    assert executor.returned_arrow_batch_counts[0] > 1
    assert {row_id: observed["value"] for row_id, observed in rows} == {
        row_id: None if row_id % 7 == 0 else f"generated:{prompts[row_id]}" for row_id in range(row_count)
    }


def test_distributed_collection_keeps_explicit_pool_names_query_scoped():
    import vane

    con = vane.connect()
    try:
        explicit_pool_name = "explicit-shared-vllm-pool"
        options = _packed_native_vllm_options({"use_ray": True, "ray_actor_pool_name": explicit_pool_name})

        def collect_node(query_id):
            source = con.sql("SELECT prompt FROM (VALUES ('hello')) input(prompt)")
            generated = vane.FunctionExpression(
                "vllm",
                vane.ColumnExpression("prompt"),
                vane.ConstantExpression("model"),
                vane.ConstantExpression(options),
            ).alias("generated")
            relation = source.select(generated)
            plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, query_id).to_physical_plan(con)
            nodes = plan.collect_vllm_nodes(conn=con)
            assert len(nodes) == 1
            return nodes[0]

        # These IDs have the same readable sanitized form. Their raw-ID hash
        # must still keep independently configured actor pools isolated.
        first = collect_node("query/a")
        second = collect_node("query?a")

        assert first["pool_name"] != explicit_pool_name
        assert second["pool_name"] != explicit_pool_name
        assert first["pool_name"] != second["pool_name"]
        assert first["options"]["ray_actor_pool_name"] == first["pool_name"]
        assert second["options"]["ray_actor_pool_name"] == second["pool_name"]
    finally:
        con.close()


def test_native_vllm_rejects_bare_json_options_at_bind_time():
    import vane

    con = vane.connect()
    try:
        with pytest.raises(Exception, match="versioned STRUCT envelope"):
            con.execute("SELECT vllm('hello', 'model', '{}')")
    finally:
        con.close()


def test_native_vllm_rejects_missing_or_null_options_at_bind_time():
    import vane

    con = vane.connect()
    try:
        with pytest.raises(Exception, match="vllm"):
            con.execute("SELECT vllm('hello', 'model')")
        with pytest.raises(Exception, match="options cannot be NULL"):
            con.execute("SELECT vllm('hello', 'model', NULL)")
    finally:
        con.close()


def test_distributed_collection_preserves_the_versioned_envelope():
    import vane
    from vane.execution.vllm import normalize_options

    con = vane.connect()
    try:
        relation = con.sql(
            f"SELECT vllm(prompt, 'model', {_EMPTY_NATIVE_VLLM_OPTIONS_SQL}) FROM (VALUES ('hello')) input(prompt)"
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "default-options").to_physical_plan(con)
        nodes = plan.collect_vllm_nodes(conn=con)
    finally:
        con.close()

    assert len(nodes) == 1
    options = nodes[0]["options"]
    assert options["__vane_vllm_payload_version"] == 1
    assert options["__vane_vllm_public_options_json"] == "{}"
    assert options["__vane_vllm_secret_payload"] == b'{"payload_version":1,"values":[]}'
    normalized = normalize_options(options)
    assert normalized["use_ray"] is True
    assert normalized["ray_worker_only"] is True
    assert normalized["ray_actor_pool_name"] == nodes[0]["pool_name"]


@pytest.mark.parametrize(
    "context",
    ["ctas", "insert_select", "scalar_subquery", "explain_analyze", "prepared"],
)
@pytest.mark.timeout(30)
def test_native_finalizer_has_scheduler_wakeup_in_materialized_and_nested_contexts(monkeypatch, context):
    import vane
    import vane.execution.vllm as vllm

    executor = _DeferredWakeupExecutor()
    monkeypatch.setattr(vllm, "build_executor", lambda *_args, **_kwargs: executor)
    publisher_errors = []

    def publish_after_arm() -> None:
        try:
            assert executor.pending_ready.wait(timeout=20), "native producer did not submit a pending result"
            assert executor.callback_armed.wait(timeout=20), "native finalizer did not arm a wakeup callback"
            executor.publish_results()
        except BaseException as exc:
            publisher_errors.append(exc)

    publisher = threading.Thread(target=publish_after_arm, name=f"vllm-{context}-result-publisher")
    publisher.start()
    con = vane.connect()
    try:
        con.register("vllm_input", pa.table({"prompt": ["hello"]}))
        expression = f"vllm(prompt, 'recording-model', {_EMPTY_NATIVE_VLLM_OPTIONS_SQL})"
        if context == "ctas":
            con.execute(f"CREATE TABLE vllm_output AS SELECT {expression} AS generated FROM vllm_input")
        elif context == "insert_select":
            con.execute("CREATE TABLE vllm_output(generated VARCHAR)")
            con.execute(f"INSERT INTO vllm_output SELECT {expression} FROM vllm_input")
        elif context == "scalar_subquery":
            con.execute(
                f"SELECT (SELECT vllm('hello', 'recording-model', {_EMPTY_NATIVE_VLLM_OPTIONS_SQL}))"
            ).fetchall()
        elif context == "explain_analyze":
            con.execute(f"EXPLAIN ANALYZE SELECT {expression} FROM vllm_input").fetchall()
        else:
            con.execute(
                "PREPARE vllm_statement AS SELECT "
                f"vllm(CAST($1 AS VARCHAR), 'recording-model', {_EMPTY_NATIVE_VLLM_OPTIONS_SQL})"
            )
            con.execute("EXECUTE vllm_statement('hello')").fetchall()
    finally:
        con.close()
        publisher.join(timeout=5)

    assert not publisher.is_alive()
    assert publisher_errors == []
    assert [submission[1] for submission in executor.submissions] == [("hello",)]
    assert executor.wakeup_registrations >= 1
    assert executor.callback_invocations >= 1
    assert executor.finished_count == 1
    assert not executor.pending
    assert not executor.invalid_wait


def test_native_bridge_rejects_executor_without_wakeup_callback(monkeypatch):
    import vane
    import vane.execution.vllm as vllm

    monkeypatch.setattr(vllm, "build_executor", lambda *_args, **_kwargs: object())

    con = vane.connect()
    try:
        with pytest.raises(Exception, match="vllm executor must implement register_wakeup_callback"):
            con.execute(f"SELECT vllm('hello', 'model', {_EMPTY_NATIVE_VLLM_OPTIONS_SQL})").fetchall()
    finally:
        con.close()
