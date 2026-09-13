# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.fast.test_native_media_extensions import _wav

pytestmark = pytest.mark.skipif(
    os.environ.get("VANE_TEST_DYNAMIC_MEDIA_RUNTIME") != "1",
    reason="requires signed dynamic media provider wheels and their exact runtime",
)


def run(script, *arguments):
    environment = dict(os.environ, VANE_RUNNER="local-fast")
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(script), *map(str, arguments)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_prepared_directory_loads_in_fresh_process_without_python_runtime_hook(tmp_path):
    prepared = run(
        """
        import sys
        from importlib import import_module
        from importlib.metadata import entry_points
        import vane
        from vane.extensions import DynamicExtensionResolver
        entry = next(ep for ep in entry_points(group='vane.dynamic_extension_providers') if ep.name == 'native_media')
        module = import_module(entry.module)
        with vane.connect(config={'extension_directory': sys.argv[1]}) as connection:
            resolved = DynamicExtensionResolver(trusted_identities=[module.descriptor().trust_identity], providers=[module.provider()]).resolve(connection, module.descriptor())
            print(resolved[-1].path)
        """,
        tmp_path,
    ).strip()
    relocated = tmp_path / "relocated"
    shutil.copytree(Path(prepared).parent, relocated)
    prepared = relocated / Path(prepared).name
    run(
        """
        import sys
        import vane
        assert 'vane._native_runtime' not in sys.modules
        with vane.connect() as connection:
            connection.execute("LOAD '" + sys.argv[1].replace("'", "''") + "'")
            assert connection.execute("SELECT loaded FROM duckdb_extensions() WHERE extension_name='native_media'").fetchone()[0]
        assert 'vane._native_runtime' not in sys.modules
        """,
        prepared,
    )


def test_resolve_then_load_and_fresh_process_reuse_runtime_without_staging(tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(_wav())
    script = """
        import sys
        from importlib import import_module
        from importlib.metadata import entry_points
        from unittest.mock import patch
        import vane
        from vane.extensions import DynamicExtensionResolver
        entry = next(ep for ep in entry_points(group='vane.dynamic_extension_providers') if ep.name == 'native_media')
        module = import_module(entry.module)
        descriptor = module.descriptor()
        resolver = DynamicExtensionResolver(trusted_identities=[descriptor.trust_identity], providers=[module.provider()])
        with vane.connect(config={'extension_directory': sys.argv[1], 'audio_backend': 'native'}) as connection:
            if sys.argv[3] == 'first':
                resolver.resolve(connection, descriptor)
            with patch('vane._native_runtime.tempfile.mkdtemp', side_effect=AssertionError('cache hit staged another runtime')):
                resolved = resolver.resolve(connection, descriptor)[-1]
                loaded = resolver.load(connection, descriptor)
                assert resolved.path == loaded.path
            result = connection.execute('SELECT resample(audio_file(?), 16000)', [sys.argv[2]]).fetchone()[0]
            assert result.shape == (1600, 2), result.shape
            print(loaded.path)
        """
    first = run(script, tmp_path / "cache", source, "first").strip()
    assert run(script, tmp_path / "cache", source, "reuse").strip() == first


def test_signed_media_runtime_loads_with_restrictive_umask(tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(_wav())
    run(
        """
        import os, sys
        import vane
        with vane.connect(config={'extension_directory': sys.argv[1], 'audio_backend': 'native'}) as connection:
            previous_umask = os.umask(0o777)
            try:
                vane.load_installed_extension('native_media', connection=connection)
            finally:
                os.umask(previous_umask)
            result = connection.execute('SELECT resample(audio_file(?), 16000)', [sys.argv[2]]).fetchone()[0]
            assert result.shape == (1600, 2), result.shape
        """,
        tmp_path / "cache",
        source,
    )


@pytest.mark.parametrize("damage", ["missing", "bytes", "signature", "extra"])
def test_corrupt_installed_runtime_is_rejected_before_loading(tmp_path, damage):
    run(
        """
        import shutil, sys
        from pathlib import Path
        from importlib.metadata import distribution
        import vane
        import vane._native_runtime as runtime
        root = Path(sys.argv[1]) / 'runtime'
        shutil.copytree(distribution('vane-media-runtime').locate_file('vane_media_runtime'), root)
        library = next((root / '.libs').iterdir())
        if sys.argv[2] == 'missing':
            library.unlink()
        elif sys.argv[2] == 'bytes':
            library.write_bytes(b'not a shared library')
        elif sys.argv[2] == 'signature':
            (root / 'runtime-manifest.sig').write_bytes(bytes(256))
        else:
            (root / '.libs/extra.so').write_bytes(b'extra library')
        installed = distribution('vane-media-runtime')
        class Distribution:
            version = installed.version
            def locate_file(self, name):
                assert name == 'vane_media_runtime'
                return root
        runtime.distribution = lambda name: Distribution()
        with vane.connect(config={'extension_directory': str(Path(sys.argv[1]) / 'cache')}) as connection:
            try:
                vane.load_installed_extension('native_media', connection=connection)
            except Exception:
                pass
            else:
                raise AssertionError('corrupt runtime was accepted')
        assert runtime._selected is None, 'corrupt runtime was admitted'
        """,
        tmp_path,
        damage,
    )


@pytest.mark.parametrize("order", ["codecs-first", "vane-first"])
def test_media_runtime_coexists_with_python_codec_libraries(tmp_path, order):
    for package in ("av", "soundfile", "soxr"):
        pytest.importorskip(package)
    source = tmp_path / "audio.wav"
    source.write_bytes(_wav())
    run(
        """
        import importlib, sys
        def codecs():
            for name in ('av', 'soundfile', 'soxr'):
                importlib.import_module(name)
        if sys.argv[2] == 'codecs-first':
            codecs()
        import vane
        with vane.connect(config={'audio_backend': 'native'}) as connection:
            vane.load_installed_extension('native_media', connection=connection)
            if sys.argv[2] == 'vane-first':
                codecs()
            result = connection.execute('SELECT resample(audio_file(?), 16000)', [sys.argv[1]]).fetchone()[0]
            assert result.shape == (1600, 2), result.shape
        """,
        source,
        order,
    )


@pytest.mark.parametrize("runner_type", ["local-fast", "local"])
def test_rebuilt_soxr_changes_native_behavior_without_resigning_extension(tmp_path, runner_type):
    override = os.environ.get("VANE_TEST_NATIVE_RUNTIME_OVERRIDE")
    if not override:
        pytest.skip("set VANE_TEST_NATIVE_RUNTIME_OVERRIDE to the locally rebuilt SoXR fixture")
    source = tmp_path / "audio.wav"
    source.write_bytes(_wav())
    result = run(
        """
        import hashlib, json, pickle, sys
        from pathlib import Path
        from importlib.metadata import entry_points
        import vane
        vane.use_native_media_runtime(sys.argv[1])
        if sys.argv[3] == 'local':
            vane.set_runner_local(num_workers=1, max_running_tasks=1, execution_mode='in_process')
        from importlib import import_module
        entry = next(ep for ep in entry_points(group='vane.dynamic_extension_providers') if ep.name == 'native_media')
        module = import_module(entry.module)
        artifact = module.provider().find(module.descriptor().identity)
        before = hashlib.sha256(artifact.path.read_bytes()).hexdigest()
        with vane.connect(config={'audio_backend': 'native'}) as connection:
            vane.load_installed_extension('native_media', connection=connection)
            profile = connection.execute('SELECT native_audio_resample_profile(audio_file(?), 16000)', [sys.argv[2]]).fetchone()[0]
            assert profile['resampler_version_string'] == 'libsoxr-local-rebuild-proof', profile
            if sys.argv[3] == 'local':
                import pyarrow as pa
                import pyarrow.parquet as pq
                source = Path(sys.argv[2]).with_suffix('.parquet')
                target = source.with_name('resampled.parquet')
                pq.write_table(pa.table({'path': [sys.argv[2]]}), source)
                connection.read_parquet(str(source)).project(
                    'native_audio_resample_profile(audio_file(path), 16000).resampler_version_string AS version'
                ).write_parquet(str(target))
                assert pq.read_table(target).to_pydict() == {'version': ['libsoxr-local-rebuild-proof']}
                # Replaying a local plan into another DatabaseInstance must also
                # prepare the same selected runtime before deserialization.
                logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.read_parquet(str(source)), None)
                replay = pickle.loads(pickle.dumps(logical))
                with vane.connect() as planning_connection:
                    physical = replay.to_physical_plan(planning_connection)
                    assert physical is not None
            try:
                vane.use_native_media_runtime(sys.argv[1])
            except ValueError:
                pass
            else:
                raise AssertionError('runtime selection changed after initialization')
            from vane.extensions import _capture_dynamic_extension_snapshot_for_worker
            try:
                _capture_dynamic_extension_snapshot_for_worker(connection)
            except ValueError as error:
                assert 'Ray' in str(error)
            else:
                raise AssertionError('custom runtime silently propagated to workers')
            from vane.extensions import _capture_dynamic_extension_snapshot, _prepare_dynamic_extension_snapshot
            with vane._native._connect_with_runner('ray') as worker:
                try:
                    _prepare_dynamic_extension_snapshot(worker, _capture_dynamic_extension_snapshot(connection))
                except ValueError as error:
                    assert 'allow_distributed=True' in str(error)
                else:
                    raise AssertionError('worker preparation accepted a custom runtime')
                vane.load_installed_extension('native_media', connection=worker)
                try:
                    vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(worker.sql('SELECT 1'), None)
                except (ValueError, vane.InvalidInputException) as error:
                    assert 'allow_distributed=True' in str(error)
                else:
                    raise AssertionError('Ray plan capture accepted a custom runtime')
        after = hashlib.sha256(artifact.path.read_bytes()).hexdigest()
        assert before == after
        print(json.dumps({'extension_sha256': before, 'resampler': profile['resampler_version_string']}))
        """,
        override,
        source,
        runner_type,
    )
    assert json.loads(result)["resampler"] == "libsoxr-local-rebuild-proof"


@pytest.mark.parametrize("source_runner", ["ray", "local"])
def test_distributed_runtime_identity_survives_local_plan_replay(tmp_path, source_runner):
    override = os.environ.get("VANE_TEST_NATIVE_RUNTIME_OVERRIDE")
    if not override:
        pytest.skip("set VANE_TEST_NATIVE_RUNTIME_OVERRIDE to the locally rebuilt SoXR fixture")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(_wav())
    run(
        """
        import hashlib, pickle, sys
        from pathlib import Path
        import vane
        from vane.extensions import _capture_dynamic_extension_snapshot

        vane.use_native_media_runtime(sys.argv[1], allow_distributed=True)
        vane.set_runner_local(num_workers=1, max_running_tasks=1, execution_mode='in_process')
        digest = hashlib.sha256((Path(sys.argv[1]) / 'runtime-manifest.json').read_bytes()).hexdigest()
        with vane._native._connect_with_runner(sys.argv[3]) as source:
            vane.load_installed_extension('native_media', connection=source)
            logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(source.sql('SELECT 1'), None)
            expected = logical.__getstate__()[3]['dynamic_extensions']
            assert expected[0]['effective_runtime_sha256'] == digest
            serialized = pickle.dumps(logical.to_physical_plan(source))

            for preloaded in (False, True):
                replay = pickle.loads(serialized)
                with vane._native._connect_with_runner('local') as target:
                    if preloaded:
                        vane.load_installed_extension('native_media', connection=target)
                    # clone explicitly binds the transported physical plan to
                    # this local connection, including native replay validation.
                    assert replay.clone(target) is not None
                    if sys.argv[3] == 'local':
                        logical_replay = pickle.loads(pickle.dumps(logical))
                        assert logical_replay.to_physical_plan(target) is not None
                    assert _capture_dynamic_extension_snapshot(target) == expected
                    profile = target.execute(
                        'SELECT native_audio_resample_profile(audio_file(?), 16000)', [sys.argv[2]]
                    ).fetchone()[0]
                    assert profile['resampler_version_string'] == 'libsoxr-local-rebuild-proof'
                    assert replay.clone(target) is not None
        """,
        override,
        audio,
        source_runner,
    )
