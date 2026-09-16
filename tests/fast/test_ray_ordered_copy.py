# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Regression for sorted write fragments sharing one resource unit (#826)."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from tests.ray_test_profile import ray_test_object_store_options

pytestmark = [pytest.mark.real_ray, pytest.mark.ray_cluster_owner]


def test_ordered_copy_finishes_all_fragments_on_two_ray_nodes(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("VANE_RUNNER", None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            textwrap.dedent(
                """
                import json, sys
                from pathlib import Path
                import pyarrow.parquet as pq
                import ray
                from ray.cluster_utils import Cluster
                import vane

                root = Path(sys.argv[1])
                store_options = json.loads(sys.argv[2])
                cluster = Cluster(shutdown_at_exit=False)
                con = None
                try:
                    cluster.add_node(num_cpus=0, include_dashboard=False, **store_options)
                    for _ in range(2):
                        cluster.add_node(num_cpus=2, include_dashboard=False, **store_options)
                    ray.init(address=cluster.address, log_to_driver=True)
                    con = vane.connect()
                    for count, direction in [(256, 'ASC'), (8193, 'DESC'), (0, 'ASC')]:
                        path = root / f'ordered-{count}.parquet'
                        result = con.execute(
                            f"COPY (SELECT i::INTEGER AS id, (i % 17)::INTEGER AS k "
                            f"FROM range({count}) t(i) ORDER BY k {direction}, id {direction}) "
                            f"TO '{path}' (FORMAT PARQUET)"
                        ).fetchall()
                        assert result == [(count,)], result
                        if count:
                            dataset = pq.ParquetDataset(path)
                            # Each successful final write task contributes a file.
                            # Require real fan-out as well as complete row contents.
                            assert len(dataset.files) > 1, dataset.files
                            table = dataset.read()
                            assert sorted(table.column('id').to_pylist()) == list(range(count))
                            assert sorted(table.column('k').to_pylist()) == sorted(i % 17 for i in range(count))
                finally:
                    try:
                        if con is not None:
                            con.close()
                        vane.teardown_runner()
                    finally:
                        ray.shutdown()
                        cluster.shutdown()
                """
            ),
            str(tmp_path),
            json.dumps(ray_test_object_store_options()),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
