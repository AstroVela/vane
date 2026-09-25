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
    psutil = pytest.importorskip("psutil")
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
        assert counts["rows"] == 2040
        assert 1 <= counts["submissions"] < 10, counts
