# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import time
from pathlib import Path

import pyarrow as pa
from vllm import LLM

import vane

_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "_benchmarking_vllm_config",
    Path(__file__).with_name("config.py"),
)
if _CONFIG_SPEC is None or _CONFIG_SPEC.loader is None:
    raise ImportError("Failed to load benchmarking/vllm/config.py")
_CONFIG = importlib.util.module_from_spec(_CONFIG_SPEC)
_CONFIG_SPEC.loader.exec_module(_CONFIG)

MODEL_NAME = _CONFIG.MODEL_NAME
NATIVE_BATCH_SIZE = _CONFIG.NATIVE_BATCH_SIZE
SAMPLING_PARAMS = _CONFIG.SAMPLING_PARAMS
NAIVE_BATCH_SIZE = _CONFIG.NAIVE_BATCH_SIZE
INPUT_LIMIT = _CONFIG.INPUT_LIMIT
build_input_sql = _CONFIG.build_input_sql
build_vane_native_vllm_options = _CONFIG.build_vane_native_vllm_options
connect_vane = _CONFIG.connect_vane
get_vane_naive_actor_count = _CONFIG.get_vane_naive_actor_count
get_vane_naive_partition_count = _CONFIG.get_vane_naive_partition_count
get_vllm_engine_args = _CONFIG.get_vllm_engine_args
json_sql_literal = _CONFIG.json_sql_literal
print_benchmark_results = _CONFIG.print_benchmark_results
print_preview_rows = _CONFIG.print_preview_rows
sql_literal = _CONFIG.sql_literal


NAIVE_VLLM_SCHEMA = {
    "id": vane.sqltypes.UBIGINT,
    "prompt": vane.sqltypes.VARCHAR,
    "output": vane.sqltypes.VARCHAR,
    "prompt_len": vane.sqltypes.UBIGINT,
    "output_len": vane.sqltypes.UBIGINT,
}


class NaiveVLLM:
    def __init__(self):
        print("Initializing LLM...")
        start_time = time.perf_counter()
        self.llm = LLM(model=MODEL_NAME, disable_log_stats=True, **get_vllm_engine_args())
        end_time = time.perf_counter()
        print(f"LLM initialized in {end_time - start_time:.2f} seconds.")

    def __call__(self, table: pa.Table) -> pa.Table:
        prompts = table.column("prompt").to_pylist()
        ids = table.column("id").to_pylist()
        outputs = self.llm.generate(prompts, SAMPLING_PARAMS)
        texts = [output.outputs[0].text for output in outputs]
        return pa.table(
            {
                "id": ids,
                "prompt": prompts,
                "output": texts,
                "prompt_len": [len(prompt) for prompt in prompts],
                "output_len": [len(text) for text in texts],
            }
        )


def _native_vllm_query(*, do_prefix_routing: bool, sorted_by_prompt: bool) -> str:
    input_sql = build_input_sql(sorted_by_prompt=sorted_by_prompt)
    options_sql = json_sql_literal(
        build_vane_native_vllm_options(
            do_prefix_routing=do_prefix_routing,
        )
    )
    model_sql = sql_literal(MODEL_NAME)
    return f"""
WITH source AS (
    {input_sql}
),
generated AS (
    SELECT
        id,
        prompt,
        vllm(prompt, '{model_sql}', '{options_sql}') AS output
    FROM source
)
SELECT
    id,
    prompt,
    output,
    length(prompt) AS prompt_len,
    length(output) AS output_len
FROM generated
"""


def _run_relation_benchmark(script_name: str, rel, *, distributed: bool = False) -> None:
    print("Running benchmark...")
    start_time = time.perf_counter()
    if distributed:
        from vane.runners import get_or_create_runner

        runner = get_or_create_runner()
        tables = list(runner.run_iter_tables(rel))
        combined = pa.concat_tables(tables)
        rows = combined.to_pydict()
        row_count = combined.num_rows
        print(f"Distributed execution: {len(tables)} partitions, {row_count} rows")
    else:
        rows = rel.fetchall()
    end_time = time.perf_counter()
    print("Benchmark completed!")
    if not distributed:
        print_preview_rows(rows)
    print_benchmark_results(script_name, start_time, end_time)


def _prepare_gpu_input_relation(rel, *, distributed: bool = False):
    if distributed:
        # In distributed mode the RayRunner handles partitioning via scan
        # ranges.  local_exchange is a process-local operator and has no effect
        # across distributed tasks — skip it entirely.
        return rel

    # Use enough partitions to keep GPU actors fed with minimal idle gaps.
    # With only actor_count partitions (typically 2), DuckDB creates too few
    # pipeline threads and the GPUs starve between batches.
    #
    # Relation.repartition(...) currently trips an internal executor assertion in
    # this benchmark path, so only use local_exchange here.
    partition_count = get_vane_naive_partition_count()
    return rel.local_exchange(num_partitions=max(1, partition_count))


def run_vane_naive_benchmark(script_name: str, *, sorted_by_prompt: bool, distributed: bool = False) -> None:
    print("Starting benchmark...")
    if distributed and INPUT_LIMIT > 0:
        print(
            f"WARNING: INPUT_LIMIT={INPUT_LIMIT} applies per-task in distributed mode. "
            "Total rows processed will be up to num_tasks × LIMIT."
        )
    con = connect_vane()
    actor_count = get_vane_naive_actor_count()
    rel = con.sql(build_input_sql(sorted_by_prompt=sorted_by_prompt))
    rel = _prepare_gpu_input_relation(
        rel,
        distributed=distributed,
    )
    rel = rel.map_batches(
        NaiveVLLM,
        schema=NAIVE_VLLM_SCHEMA,
        execution_backend="ray_actor",
        gpus=1,
        batch_size=NAIVE_BATCH_SIZE,
        actor_number=actor_count,
    )
    _run_relation_benchmark(script_name, rel, distributed=distributed)


def run_vane_native_vllm_benchmark(
    script_name: str,
    *,
    do_prefix_routing: bool,
    sorted_by_prompt: bool,
    distributed: bool = False,
) -> None:
    print("Starting benchmark...")
    con = connect_vane()
    rel = con.sql(
        _native_vllm_query(
            do_prefix_routing=do_prefix_routing,
            sorted_by_prompt=sorted_by_prompt,
        )
    )
    _run_relation_benchmark(script_name, rel, distributed=distributed)
