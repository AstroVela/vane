# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild delivered sources and prove a modified library works with the signed provider."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.prepare_local_media_runtime import prepare_local
from vane_packaging.media_release import _output_directory, verified_release
from vane_packaging.media_runtime import read_runtime_wheel
from vane_packaging.media_sources import read_source_archive, read_source_file
from vane_packaging.media_version import runtime_format

REBUILD_PROOF = "libsoxr-local-rebuild-proof"


def rebuild_soxr(source: Path, runtime: Path, directory: Path, *, jobs: int = 2) -> Path:
    """Make an observable change to SoXR's actual implementation and compile it."""
    if type(jobs) is not int or not 1 <= jobs <= 256:
        raise ValueError("jobs must be between 1 and 256")
    implementation = source / "src/soxr.c"
    contents = implementation.read_text()
    original = 'return "libsoxr-" SOXR_THIS_VERSION_STR;'
    if contents.count(original) != 1:
        raise ValueError("SoXR replacement proof no longer matches the pinned source")
    implementation.write_text(contents.replace(original, f'return "{REBUILD_PROOF}";'))
    build = directory / "local-soxr-build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DWITH_OPENMP=OFF",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(jobs)], check=True)
    output = directory / "local-runtime"
    prepare_local(runtime, {"libsoxr.so.0": build / "src/libsoxr.so"}, output)
    return output


def _verify_replacement(paths: dict[str, Path], replacement: Path, *, receipt: Path) -> None:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("PYTHON", "VANE_")) and key not in {"VIRTUAL_ENV", "__PYVENV_LAUNCHER__"}
    }
    environment.update(PIP_CONFIG_FILE=os.devnull, VANE_RUNNER="local-fast")
    with tempfile.TemporaryDirectory(prefix="vane-media-rebuilt-install-") as temporary:
        workspace = Path(temporary)
        venv = workspace / "venv"
        subprocess.run([sys.executable, "-I", "-m", "venv", "--copies", str(venv)], check=True, env=environment)
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "--isolated",
                "install",
                *(str(paths[role]) for role in ("base", "runtime", "provider")),
            ],
            check=True,
            cwd=workspace,
            env=environment,
        )
        program = """
import hashlib, json, sys, wave
from importlib.metadata import entry_points
from pathlib import Path
import vane
from vane.extensions import _capture_dynamic_extension_snapshot
vane.use_native_media_runtime(sys.argv[1])
audio = Path('proof.wav')
with wave.open(str(audio), 'wb') as stream:
    stream.setparams((1, 2, 8000, 800, 'NONE', 'not compressed'))
    stream.writeframes(bytes(1600))
provider = next(ep for ep in entry_points(group='vane.dynamic_extension_providers') if ep.name == 'native_media').load()()
with vane.connect() as connection:
    vane.load_installed_extension('native_media', connection=connection)
    descriptor = _capture_dynamic_extension_snapshot(connection)[0]
    profile = connection.execute('SELECT native_audio_resample_profile(audio_file(?), 16000)', [str(audio)]).fetchone()[0]
    assert profile['resampler_version_string'] == sys.argv[2], profile
    identity = f"{descriptor['name']}@{descriptor['extension_version']}#sha256:{descriptor['sha256']}"
    artifact = provider.find(identity)
    assert hashlib.sha256(artifact.path.read_bytes()).hexdigest() == descriptor['sha256']
    Path(sys.argv[3]).write_text(json.dumps({
        'extension_sha256': descriptor['sha256'],
        'effective_runtime_sha256': hashlib.sha256((Path(sys.argv[1]) / 'runtime-manifest.json').read_bytes()).hexdigest(),
        'resampler_version_string': profile['resampler_version_string'],
    }, sort_keys=True) + '\\n')
"""
        subprocess.run(
            [str(python), "-I", "-c", program, str(replacement), REBUILD_PROOF, str(receipt)],
            check=True,
            cwd=workspace,
            env=environment,
        )


def rebuild_release(directory: Path, *, trust_identity: str, manifest_sha256: str, output: Path, jobs: int = 2) -> dict:
    """Accept a retrieved delivery without any publisher signing key or binary cache."""
    if type(jobs) is not int or not 1 <= jobs <= 256:
        raise ValueError("jobs must be between 1 and 256")
    with verified_release(directory, trust_identity=trust_identity, manifest_sha256=manifest_sha256) as (
        delivery,
        manifest,
    ):
        paths = {role: delivery / record["filename"] for role, record in manifest["artifacts"].items()}
        with _output_directory(output) as stage:
            project = stage / "source"
            files = read_source_archive(read_source_file(paths["source"]), paths["source"].name)
            for name, contents in files.items():
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents)
                target.chmod(0o755 if name.endswith(".sh") else 0o644)
            environment = dict(os.environ, VCPKG_MAX_CONCURRENCY=str(jobs), VCPKG_BINARY_SOURCES="clear")
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
                    "import backend; backend._build_sdk(Path(sys.argv[1]))",
                    str(project),
                ],
                check=True,
                cwd=project,
                env=environment,
            )
            sources = list((project / "build/buildtrees/soxr/src").glob("*.clean"))
            if len(sources) != 1:
                raise ValueError("expected one SoXR source tree rebuilt from the delivered SDK")
            _, _, libraries, document, _ = read_runtime_wheel(paths["runtime"])
            runtime = stage / "official-runtime"
            (runtime / ".libs").mkdir(parents=True)
            (runtime / runtime_format().MANIFEST).write_bytes(document)
            for name, contents in libraries.items():
                (runtime / ".libs" / name).write_bytes(contents)
            replacement = rebuild_soxr(sources[0], runtime, stage, jobs=jobs)
            receipt_path = stage / "rebuild-verification.json"
            _verify_replacement(paths, replacement, receipt=receipt_path)
            receipt = json.loads(receipt_path.read_bytes())
            receipt.update(schema_version=1, release_manifest_sha256=manifest_sha256)
            receipt_path.write_bytes(runtime_format().canonical_json(receipt))
    return receipt
