# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
        from importlib import import_module
        from importlib.metadata import entry_points, distribution
        import vane
        import vane._native_runtime as runtime
        root = Path(sys.argv[1]) / 'runtime'
        entry = next(ep for ep in entry_points(group='vane.dynamic_extension_providers') if ep.name == 'native_media')
        module = import_module(entry.module)
        official = Path(module.__file__).parent / 'runtime'
        shutil.copytree(official, root)
        library = next((root / '.libs').iterdir())
        if sys.argv[2] == 'missing':
            library.unlink()
        elif sys.argv[2] == 'bytes':
            library.write_bytes(b'not a shared library')
        elif sys.argv[2] == 'signature':
            (root / 'runtime-manifest.sig').write_bytes(bytes(256))
        else:
            (root / '.libs/extra.so').write_bytes(b'extra library')
        installed = distribution('vane-extension-native-media')
        class Distribution:
            version = installed.version
            def locate_file(self, name):
                assert name == 'vane_extensions/' + official.parent.name + '/runtime'
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
