# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Keep GPU compute batches independent of leased Ray transport envelopes."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.real_ray, pytest.mark.ray_cluster_owner, pytest.mark.gpu]


def test_gpu_actor_coalesces_small_compute_batches(tmp_path):
    script = textwrap.dedent(
        """
        import json
        from pathlib import Path
        import re
        import sys
        import pyarrow as pa
        import pyarrow.parquet as pq
        import ray
        import vane

        folder = Path(sys.argv[1])
        pq.write_table(pa.table({"x": list(range(2040))}), folder / "input.parquet")

        class ReportBatch:
            def __call__(self, table):
                return pa.table({"x": table["x"], "batch_rows": [len(table)] * len(table)})

        con = vane.connect()
        try:
            rows = con.read_parquet(str(folder / "input.parquet")).map_batches(
                ReportBatch,
                schema={"x": vane.sqltypes.BIGINT, "batch_rows": vane.sqltypes.BIGINT},
                batch_size=10, actor_number=1, gpus=1.0,
            ).fetchall()
            assert sorted(x for x, _ in rows) == list(range(2040))
            assert dict(rows)[0] == 10
            assert max(size for _, size in rows) > 10
            assert max(size for _, size in rows) <= 128 * 1024
            logs = Path(ray._private.worker._global_node.get_session_dir_path()) / "logs"
            submitted = []
            for path in logs.glob("worker-*.err"):
                text = path.read_text(errors="replace")
                if "udf_name=ReportBatch" in text:
                    submitted.extend(map(int, re.findall(r"\\bsubmitted_batches=(\\d+)", text)))
            assert submitted, "missing native scheduler debug counters"
            print("TRANSPORT_RESULT=" + json.dumps({
                "rows": len(rows), "submissions": max(submitted),
                "max_compute_rows": max(size for _, size in rows),
            }), flush=True)
        finally:
            con.close()
            ray.shutdown()
        """
    )
    counts = _run_gpu_script(script, tmp_path)
    assert counts["rows"] == 2040
    assert 1 <= counts["submissions"] < 10, counts


@pytest.mark.parametrize(
    "materialized,actors,empty_output,byte_limit,total_rows",
    [
        (False, 1, False, None, 2051),
        (True, 1, False, None, 2051),
        (False, 2, False, None, 2051),
        (False, 1, True, None, 2051),
        (False, 1, False, 1024, 109),
        (False, 1, False, None, 7),
    ],
)
def test_gpu_actor_coalesces_small_upstream_blocks(
    tmp_path, materialized, actors, empty_output, byte_limit, total_rows
):
    script = textwrap.dedent(
        """
        import json
        from pathlib import Path
        import sys
        import pyarrow as pa
        import ray
        import vane
        from vane.datasource import DataSource, DataSourceTask, read_datasource

        folder = Path(sys.argv[1])
        settings = json.loads((folder / "settings.json").read_text())
        total = settings["total_rows"]

        class SmallBlocksTask(DataSourceTask):
            def execute(self):
                for start in range(0, total, 9):
                    values = list(range(start, min(start + 9, total)))
                    columns = {"x": values}
                    if settings["byte_limit"] is not None:
                        columns["padding"] = [b"x" * 128] * len(values)
                    yield pa.record_batch(columns)

        class SmallBlocksSource(DataSource):
            @property
            def schema(self):
                return {"x": "BIGINT", **({"padding": "BLOB"} if settings["byte_limit"] is not None else {})}

            def get_tasks(self):
                yield SmallBlocksTask()

        class ReportBatch:
            def __call__(self, table):
                import os
                with (folder / (str(os.getpid()) + ".jsonl")).open("a") as trace:
                    trace.write(json.dumps({"rows": len(table)}) + "\\n")
                output = pa.table({"x": table["x"], "batch_rows": [len(table)] * len(table)})
                return output.slice(0, 0) if settings["empty_output"] else output

        con = vane.connect()
        try:
            source = read_datasource(SmallBlocksSource(), con=con)
            if settings["materialized"]:
                # Force a native operator between the lazy scan and the UDF.
                source = source.select("x + 1 AS x")
            options = {}
            if settings["byte_limit"] is not None:
                options["task_input_max_bytes"] = settings["byte_limit"]
            rows = source.map_batches(
                ReportBatch,
                schema={"x": vane.sqltypes.BIGINT, "batch_rows": vane.sqltypes.BIGINT},
                batch_size=32, actor_number=settings["actors"], gpus=1.0 / settings["actors"],
                **options,
            ).fetchall()
            if settings["empty_output"]:
                assert rows == []
            else:
                offset = int(settings["materialized"])
                assert sorted(x for x, _ in rows) == list(range(offset, total + offset))
            sizes = [json.loads(line)["rows"] for path in folder.glob("*.jsonl")
                     for line in path.read_text().splitlines()]
            assert sum(sizes) == total, sizes
            if total < 32 or settings["byte_limit"] is not None:
                assert max(sizes) < 32, sizes
            else:
                assert 32 in sizes, sizes
                # Merely restoring a fixed minimum of 32 would fail this:
                # subsequent envelopes must follow the Actor's growing target.
                assert max(sizes) > 32, sizes
            print("TRANSPORT_RESULT=" + json.dumps({"rows": total, "sizes": sizes}), flush=True)
        finally:
            con.close()
            ray.shutdown()
        """
    )
    (tmp_path / "settings.json").write_text(
        json.dumps(
            dict(
                materialized=materialized,
                actors=actors,
                empty_output=empty_output,
                byte_limit=byte_limit,
                total_rows=total_rows,
            )
        )
    )
    _run_gpu_script(script, tmp_path)


def _run_gpu_script(script, tmp_path):
    psutil = pytest.importorskip("psutil")
    # The short path also avoids Ray's Unix-domain socket length limit.
    with tempfile.TemporaryDirectory(prefix="vane-gpu-batch-") as ray_tmp:
        env = dict(os.environ, DUCKDB_DISTRIBUTED_DEBUG="1", VANE_RUNNER="ray", RAY_TMPDIR=ray_tmp)
        env.pop("RAY_ADDRESS", None)
        with (tmp_path / "probe.log").open("w") as log:
            proc = subprocess.Popen(
                [sys.executable, "-c", script, str(tmp_path)], env=env, stdout=log, stderr=subprocess.STDOUT
            )
            children = {}
            try:
                for _ in range(120):
                    if proc.poll() is not None:
                        break
                    try:
                        for child in psutil.Process(proc.pid).children(recursive=True):
                            children[child.pid] = child
                    except psutil.NoSuchProcess:
                        break
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                assert proc.poll() is not None, "GPU transport probe timed out"
            finally:
                if proc.poll() is None:
                    proc.terminate()
                for child in children.values():
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(list(children.values()), timeout=5)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(alive, timeout=5)
                proc.wait(timeout=10)
        output = Path(tmp_path / "probe.log").read_text()
        assert proc.returncode == 0, output[-20000:]
        result = next(
            line.removeprefix("TRANSPORT_RESULT=")
            for line in output.splitlines()
            if line.startswith("TRANSPORT_RESULT=")
        )
        counts = json.loads(result)
        return counts
