# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest

import scripts.verify_extension_wheel as verifier
from tests.fast.test_extension_wheel import _relabel_wheel_platform, _rewrite_wheel_metadata
from tests.fast.test_media_sources import runtime_wheel as runtime_wheel
from tests.fast.test_media_sources import source_sdk as source_sdk
from vane import _native_runtime_format as runtime_format
from vane.extensions import DynamicExtensionError, _native_extension_compatibility_version, _native_platform
from vane_packaging.extension_wheel import _read_dependency_wheels, build_extension_wheel
from vane_packaging.media_runtime import read_runtime_wheel

ROOT = Path(__file__).resolve().parents[2]
TRUST_IDENTITY = "vane-tests"


@pytest.fixture
def release_runtime(runtime_wheel, source_sdk, tmp_path):
    # Exercise the release metadata path with temporary, test-key-signed fixtures.
    wheel = _rewrite_wheel_metadata(
        runtime_wheel,
        tmp_path / "release-runtime",
        lambda contents: contents.replace("Classifier: Private :: Do Not Upload\n", ""),
    )
    return wheel, source_sdk[2], read_runtime_wheel(wheel)


def _artifact(path, runtime_info=None, *, bind_runtime=True):
    source = path.with_suffix(".c")
    command = ["cc", "-shared", "-fPIC", "-o", str(path), str(source)]
    if runtime_info is None:
        source.write_text("int extension_fixture(void) { return 42; }\n")
    else:
        source.write_text("extern int soxr_fixture(void);\nint extension_fixture(void) { return soxr_fixture(); }\n")
        libraries = path.with_suffix(".libs")
        libraries.mkdir()
        for name, contents in runtime_info[2].items():
            (libraries / name).write_bytes(contents)
        soname = next(iter(runtime_info[2]))
        command.extend([f"-L{libraries}", f"-l:{soname}", "-Wl,--enable-new-dtags,-rpath,$ORIGIN/.libs"])
    subprocess.run(command, check=True)
    footer = bytearray(512)
    fields = ["", "", "", "CPP", "test-version", _native_extension_compatibility_version(), _native_platform(), "4"]
    for index, value in enumerate(fields):
        footer[index * 32 : index * 32 + len(value)] = value.encode("ascii")
    contents = path.read_bytes() + footer
    if runtime_info is not None and bind_runtime:
        contents = runtime_format.attach_trailer(contents, runtime_info[0]["manifest_sha256"])
    path.write_bytes(contents)
    return path


def _build(artifact, release_runtime, *, dependencies=(), platform_tag="manylinux_2_28_x86_64", **options):
    runtime, source, _ = release_runtime
    return build_extension_wheel(
        artifact=artifact,
        extension_name=artifact.stem,
        output_directory=artifact.parent / "wheels",
        platform_tag=platform_tag,
        trust_identity=TRUST_IDENTITY,
        license_expression=options.pop("license_expression", "Apache-2.0"),
        license_files=[ROOT / "LICENSE"],
        dependency_wheels=dependencies,
        dependency_trust_identities=[TRUST_IDENTITY] if dependencies else [],
        runtime_wheel=runtime,
        runtime_source=source,
        **options,
    )


@pytest.fixture
def media_dependency(tmp_path, release_runtime):
    artifact = _artifact(tmp_path / "native_media.duckdb_extension", release_runtime[2])
    return _build(artifact, release_runtime, license_expression="Apache-2.0 AND LGPL-2.1-or-later")


@pytest.mark.parametrize("platform_tag", ["manylinux_2_28_x86_64", "manylinux_2_39_x86_64"])
def test_ordinary_extension_graph_keeps_runtime_on_its_media_dependency(
    tmp_path, release_runtime, media_dependency, platform_tag
):
    relay = _build(
        _artifact(tmp_path / "relay.duckdb_extension"),
        release_runtime,
        dependencies=[media_dependency.path],
        platform_tag=platform_tag,
    )
    root = _build(
        _artifact(tmp_path / "root.duckdb_extension"),
        release_runtime,
        dependencies=[media_dependency.path, relay.path],
        platform_tag=platform_tag,
    )
    assert media_dependency.descriptor.format_version == 2
    assert media_dependency.descriptor.native_runtime.to_dict() == release_runtime[2][0]
    for ordinary in (relay, root):
        assert ordinary.descriptor.format_version == 1
        assert ordinary.descriptor.native_runtime is None

    # Both independent wheel readers must accept the complete dependency graph.
    _read_dependency_wheels([media_dependency.path, relay.path, root.path], runtime_info=release_runtime[2])
    layouts = [
        verifier._assert_extension_wheel_layout(wheel.path, wheel.descriptor.name, runtime_info=release_runtime[2])
        for wheel in (media_dependency, relay, root)
    ]
    by_identity = {layout.identity: layout for layout in layouts}
    for layout in layouts:
        verifier._assert_extension_requirements(layout, by_identity)
        requirements = {requirement.name for requirement in layout.requirements}
        assert ("vane-media-runtime" in requirements) == (layout.name == "native_media")


def test_dependency_runtime_does_not_exempt_an_ordinary_lgpl_root_from_release_materials(
    tmp_path, release_runtime, media_dependency
):
    with pytest.raises(ValueError, match="LGPL extension wheels require release_materials"):
        _build(
            _artifact(tmp_path / "root.duckdb_extension"),
            release_runtime,
            dependencies=[media_dependency.path],
            license_expression="Apache-2.0 AND LGPL-2.1-or-later",
        )


def test_dependency_runtime_does_not_allow_an_unbound_root_to_link_runtime_libraries(
    tmp_path, release_runtime, media_dependency
):
    with pytest.raises(ValueError, match="DT_RUNPATH|external librar"):
        _build(
            _artifact(tmp_path / "root.duckdb_extension", release_runtime[2], bind_runtime=False),
            release_runtime,
            dependencies=[media_dependency.path],
        )


def test_runtime_root_still_requires_its_exact_manifest(tmp_path, release_runtime):
    artifact = _artifact(tmp_path / "native_media.duckdb_extension", release_runtime[2])
    artifact.write_bytes(runtime_format.attach_trailer(artifact.read_bytes(), "0" * 64))
    with pytest.raises(DynamicExtensionError, match="NATIVE_RUNTIME_MISMATCH"):
        _build(artifact, release_runtime)


@pytest.mark.parametrize("reader", ["builder", "verifier"])
def test_runtime_extension_cannot_be_retagged_apart_from_its_runtime(
    tmp_path, release_runtime, media_dependency, reader
):
    relabeled = _relabel_wheel_platform(
        media_dependency.path,
        tmp_path / "relabeled",
        original="manylinux_2_28_x86_64",
        replacement="manylinux_2_39_x86_64",
    )
    with pytest.raises(
        (ValueError, RuntimeError), match="extension and media runtime must use the same platform policy"
    ):
        if reader == "builder":
            _read_dependency_wheels([relabeled], runtime_info=release_runtime[2])
        else:
            verifier._assert_extension_wheel_layout(relabeled, "native_media", runtime_info=release_runtime[2])
