# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise replacement admission on independently configured real Ray nodes."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from tests.fast.test_native_media_extensions import _wav

pytestmark = [
    pytest.mark.real_ray,
    pytest.mark.ray_cluster_owner,
    pytest.mark.skipif(
        os.environ.get("VANE_TEST_DYNAMIC_MEDIA_RUNTIME") != "1"
        or not os.environ.get("VANE_TEST_NATIVE_RUNTIME_OVERRIDE"),
        reason="requires signed media wheels and the source-rebuilt SoXR fixture",
    ),
]


@pytest.mark.parametrize("mismatch", [False, True])
def test_ray_nodes_admit_only_the_exact_authorized_replacement(tmp_path, mismatch):
    source = tmp_path / "audio.wav"
    source.write_bytes(_wav())
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            textwrap.dedent(
                """
                import hashlib, json, os, pickle, shutil, sys
                from pathlib import Path
                from importlib.metadata import distribution
                import ray
                import pyarrow as pa
                from ray.cluster_utils import Cluster
                from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
                import vane
                from vane.extensions import _capture_dynamic_extension_snapshot_for_worker
                from vane.runners.ray.runner import RayRunner

                root, audio, mismatch = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'
                replacement = Path(os.environ['VANE_TEST_NATIVE_RUNTIME_OVERRIDE'])
                official = distribution('vane-media-runtime').locate_file('vane_media_runtime')
                node_paths = [root / 'node-a', root / 'node-b']
                for index, target in enumerate(node_paths):
                    shutil.copytree(official if mismatch and index == 1 else replacement, target)
                vane.use_native_media_runtime(replacement, allow_distributed=True)
                cluster = Cluster()
                probes = []
                try:
                    nodes = []
                    for index, path in enumerate(node_paths):
                        os.environ['VANE_NATIVE_MEDIA_RUNTIME'] = str(path)
                        nodes.append(cluster.add_node(
                            num_cpus=2, include_dashboard=False,
                            object_store_memory=80 * 1024 * 1024,
                        ))
                    ray.init(address=cluster.address, log_to_driver=False)
                    with vane.connect(config={'audio_backend': 'native'}) as con:
                        vane.load_installed_extension('native_media', connection=con)
                        sql = "SELECT native_audio_resample_profile(audio_file(?), 16000)"
                        expected = con.execute(sql, [audio]).fetchone()[0]['resampler_version_string']
                        assert expected == 'libsoxr-local-rebuild-proof'
                        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(con.sql('SELECT 1'), None)
                        serialized = pickle.dumps(plan)
                        snapshot = _capture_dynamic_extension_snapshot_for_worker(con)
                        assert pickle.loads(serialized).__getstate__()[3]['dynamic_extensions'] == snapshot
                        assert snapshot[0]['effective_runtime_sha256'] == hashlib.sha256(
                            (replacement / 'runtime-manifest.json').read_bytes()).hexdigest()

                        @ray.remote
                        class Probe:
                            def replay(self, serialized, audio):
                                import pickle
                                import vane
                                from vane import _native_runtime as runtime
                                from vane.extensions import (
                                    _capture_dynamic_extension_snapshot_for_worker,
                                    _prepare_dynamic_extension_snapshot,
                                )
                                with vane.connect() as worker:
                                    try:
                                        logical = pickle.loads(serialized)
                                        _prepare_dynamic_extension_snapshot(
                                            worker, logical.__getstate__()[3]['dynamic_extensions'])
                                        physical = logical.to_physical_plan(worker)
                                    except Exception as error:
                                        assert runtime._selected is None
                                        return {'error': str(error)}
                                    profile = worker.execute(
                                        "SELECT native_audio_resample_profile(audio_file(?), 16000)",
                                        [audio]).fetchone()[0]
                                    return {
                                        'version': profile['resampler_version_string'],
                                        'path': str(runtime._override),
                                        'snapshot': _capture_dynamic_extension_snapshot_for_worker(worker),
                                    }

                        for node in nodes:
                            probes.append(Probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                                node.node_id, soft=False)).remote())
                        results = ray.get([probe.replay.remote(serialized, audio) for probe in probes])
                        for index, observed in enumerate(results):
                            if mismatch and index == 1:
                                assert 'differs from the coordinator' in observed['error'], observed
                            else:
                                assert observed == {'version': expected, 'path': str(node_paths[index]),
                                                    'snapshot': snapshot}, observed
                        if not mismatch:
                            quoted = audio.replace("'", "''")
                            relation = con.sql(
                                "SELECT native_audio_resample_profile(audio_file('" + quoted + "'), 16000) "
                                "AS profile FROM range(8)")
                            runner = RayRunner(address=None, max_task_backlog=None)
                            try:
                                for _ in range(2):
                                    batches = list(runner.run_iter_tables(
                                        vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None)))
                                    table = pa.concat_tables([
                                        b.to_arrow() if hasattr(b, 'to_arrow') else b for b in batches])
                                    assert [v['resampler_version_string'] for v in table.column(0).to_pylist()] == [expected] * 8
                            finally:
                                runner.close()
                finally:
                    for probe in probes:
                        ray.kill(probe, no_restart=True)
                    ray.shutdown()
                    cluster.shutdown()
                """
            ),
            str(tmp_path),
            str(source),
            str(mismatch),
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, VANE_RUNNER="local-fast"),
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
