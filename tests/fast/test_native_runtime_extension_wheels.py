# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import builtins
import hashlib
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

import scripts.verify_extension_wheel as verifier
from scripts.sign_media_release import sign_digest
from tests.fast.test_extension_wheel import (
    _relabel_wheel_platform,
    _rewrite_wheel,
    _rewrite_wheel_metadata,
    _write_minimal_base_wheel,
)
from tests.fast.test_media_sources import runtime_wheel as runtime_wheel
from tests.fast.test_media_sources import source_sdk as source_sdk
from vane import _native_runtime_format as runtime_format
from vane.extensions import DynamicExtensionError, _native_extension_compatibility_version, _native_platform
from vane_packaging import media_bundle
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
        runtime_wheel=runtime if runtime_format.trailer_digest(artifact.read_bytes()) is not None else None,
        runtime_source=source if runtime_format.trailer_digest(artifact.read_bytes()) is not None else None,
        **options,
    )


@pytest.fixture
def media_dependency(tmp_path, release_runtime):
    artifact = _artifact(tmp_path / "native_media.duckdb_extension", release_runtime[2])
    return _build(artifact, release_runtime, license_expression="Apache-2.0 AND LGPL-2.1-or-later")


@pytest.fixture
def alternate_runtime(tmp_path, release_runtime):
    wheel, source, info = release_runtime
    manifest = runtime_format.parse_manifest(info[3])
    # Same version, SDK and libraries, but a different signed manifest identity.
    manifest["source"]["url"] = "https://example.org/mirror/" + manifest["source"]["filename"]
    document = runtime_format.canonical_json(manifest)
    signature = sign_digest(
        ROOT / "external/duckdb/test/mbedtls/private.pem",
        hashlib.sha256(runtime_format.SIGNING_DOMAIN + document).digest(),
        tmp_path,
    )
    directory = tmp_path / "alternate-runtime"
    directory.mkdir()
    rewritten = _rewrite_wheel(
        wheel,
        directory / wheel.name,
        transforms={
            f"{runtime_format.PACKAGE}/{runtime_format.MANIFEST}": lambda _: document,
            f"{runtime_format.PACKAGE}/{runtime_format.SIGNATURE}": lambda _: signature,
        },
    )
    result = read_runtime_wheel(rewritten)
    assert result[0]["version"] == info[0]["version"]
    assert result[0]["manifest_sha256"] != info[0]["manifest_sha256"]
    return rewritten, source, result


def _replace_bundled_signature(wheel, directory, signature):
    with zipfile.ZipFile(wheel) as archive:
        member = next(name for name in archive.namelist() if name.endswith("/runtime-manifest.sig"))
    directory.mkdir()
    return _rewrite_wheel(wheel, directory / wheel.name, transforms={member: lambda _: signature})


@pytest.mark.parametrize("signature_kind", ["zeroed", "signed-other-document"])
def test_builder_rejects_same_length_invalid_bundled_signatures(
    tmp_path, release_runtime, media_dependency, signature_kind
):
    signature = bytes(256)
    if signature_kind == "signed-other-document":
        signature = sign_digest(
            ROOT / "external/duckdb/test/mbedtls/private.pem",
            hashlib.sha256(runtime_format.SIGNING_DOMAIN + b"another manifest").digest(),
            tmp_path,
        )
    damaged = _replace_bundled_signature(media_dependency.path, tmp_path / "damaged", signature)
    with pytest.raises(ValueError, match="runtime manifest signature is not trusted"):
        _build(_artifact(tmp_path / "root.duckdb_extension"), release_runtime, dependencies=[damaged])


@pytest.mark.parametrize("placement", ["siblings", "root"])
@pytest.mark.parametrize("test_only", [False, True])
def test_builder_rejects_distinct_runtime_identities_across_the_graph(
    tmp_path, release_runtime, alternate_runtime, media_dependency, placement, test_only
):
    alternate = _artifact(tmp_path / "alternate.duckdb_extension", alternate_runtime[2])
    if placement == "root":
        artifact, runtime, dependencies = alternate, alternate_runtime, [media_dependency.path]
    else:
        other = _build(alternate, alternate_runtime)
        artifact, runtime = _artifact(tmp_path / "root.duckdb_extension"), release_runtime
        dependencies = [media_dependency.path, other.path]
    with pytest.raises(ValueError, match="same exact native media runtime"):
        _build(artifact, runtime, dependencies=dependencies, test_only=test_only)


def test_multiple_media_extensions_can_share_one_exact_runtime(tmp_path, release_runtime, media_dependency):
    sibling = _build(_artifact(tmp_path / "sibling.duckdb_extension", release_runtime[2]), release_runtime)
    root = _build(
        _artifact(tmp_path / "root.duckdb_extension", release_runtime[2]),
        release_runtime,
        dependencies=[media_dependency.path, sibling.path],
    )
    dependencies = _read_dependency_wheels([media_dependency.path, sibling.path, root.path])
    assert all(dependency.descriptor.native_runtime == root.descriptor.native_runtime for dependency in dependencies)


@pytest.mark.parametrize("placement", ["siblings", "root"])
def test_clean_verifier_rejects_distinct_runtime_identities_before_environment_setup(
    tmp_path, release_runtime, alternate_runtime, media_dependency, monkeypatch, placement
):
    alternate = _artifact(tmp_path / "alternate.duckdb_extension", alternate_runtime[2])
    dependencies = [media_dependency.path]
    # Reproduce wheels emitted before the graph check: all individual bundles,
    # dependency pins and native bindings remain valid.
    with monkeypatch.context() as old_builder:
        old_builder.setattr(media_bundle, "validate_runtime_graph", lambda _: None)
        if placement == "siblings":
            dependencies.append(_build(alternate, alternate_runtime).path)
            alternate = _artifact(tmp_path / "root.duckdb_extension")
        root = _build(alternate, alternate_runtime, dependencies=dependencies)

    monkeypatch.setattr(verifier, "_run", lambda *a, **kw: pytest.fail("invalid graph reached environment setup"))
    with pytest.raises(RuntimeError, match="same exact native media runtime"):
        verifier.verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path, platform_tag="manylinux_2_28_x86_64"),
            extension_wheel=root.path,
            extension_name=root.descriptor.name,
            trust_identity=TRUST_IDENTITY,
            dependency_wheels=dependencies,
            dependency_trust_identities=[TRUST_IDENTITY],
            runtime_source=release_runtime[1],
        )


@pytest.mark.parametrize("damaged_role", [None, "root", "dependency"])
def test_clean_verifier_authenticates_every_bundle_before_provider_loading(
    tmp_path, release_runtime, media_dependency, monkeypatch, damaged_role
):
    root = _build(
        _artifact(tmp_path / "root.duckdb_extension", release_runtime[2]),
        release_runtime,
        dependencies=[media_dependency.path],
    )
    assert root.descriptor.native_runtime == media_dependency.descriptor.native_runtime
    wheels = {"root": root.path, "dependency": media_dependency.path}
    if damaged_role is not None:
        wheels[damaged_role] = _replace_bundled_signature(wheels[damaged_role], tmp_path / "damaged", bytes(256))
    base = _write_minimal_base_wheel(tmp_path, platform_tag="manylinux_2_28_x86_64")
    native_commands = []

    def run(command, *, cwd, environment=None):
        if command[1:3] != ["-I", "-c"]:
            return  # Synthetic base/extension ELFs: skip venv and pip, keep real signature verification.
        if len(command) == 5:
            native_commands.append("signatures")
            assert len(list(Path(command[-1]).glob("*.json"))) == 2
            subprocess.run(
                [sys.executable, *command[1:]], cwd=cwd, env=environment, check=True, capture_output=True, text=True
            )
        else:
            native_commands.append("providers")

    original_import = builtins.__import__

    def no_host_vane(name, *args, **kwargs):
        if name == "vane" or name.startswith("vane."):
            pytest.fail("clean verification imported the host Vane instead of using its isolated base")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(verifier, "_run", run)
    monkeypatch.setattr(builtins, "__import__", no_host_vane)
    arguments = {
        "base_wheel": base,
        "extension_wheel": wheels["root"],
        "extension_name": "root",
        "trust_identity": TRUST_IDENTITY,
        "dependency_wheels": [wheels["dependency"]],
        "dependency_trust_identities": [TRUST_IDENTITY],
        "runtime_source": release_runtime[1],
    }
    if damaged_role is None:
        verifier.verify_extension_wheel(**arguments)
        assert native_commands == ["signatures", "providers"]
    else:
        with pytest.raises(subprocess.CalledProcessError) as error:
            verifier.verify_extension_wheel(**arguments)
        assert "runtime manifest signature is not trusted by the base runtime" in error.value.stderr
        assert native_commands == ["signatures"]


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
    _read_dependency_wheels([media_dependency.path, relay.path, root.path])
    layouts = [
        verifier._assert_extension_wheel_layout(wheel.path, wheel.descriptor.name)
        for wheel in (media_dependency, relay, root)
    ]
    by_identity = {layout.identity: layout for layout in layouts}
    for layout in layouts:
        verifier._assert_extension_requirements(layout, by_identity)
        requirements = {requirement.name for requirement in layout.requirements}
        assert "vane-media-runtime" not in requirements
        assert (layout.runtime_manifest is not None) == (layout.name == "native_media")


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
            _read_dependency_wheels([relabeled])
        else:
            verifier._assert_extension_wheel_layout(relabeled, "native_media")


def test_private_roots_accept_private_dependency_graphs_but_release_readers_reject_them(tmp_path, release_runtime):
    media = _build(
        _artifact(tmp_path / "native_media.duckdb_extension", release_runtime[2]), release_runtime, test_only=True
    )
    relay = _build(
        _artifact(tmp_path / "relay.duckdb_extension"), release_runtime, dependencies=[media.path], test_only=True
    )
    artifact = _artifact(tmp_path / "root.duckdb_extension")
    root = _build(artifact, release_runtime, dependencies=[media.path, relay.path], test_only=True)
    _read_dependency_wheels([media.path, relay.path, root.path], test_only=True)
    with pytest.raises(ValueError, match="test-only extension wheels"):
        _build(artifact, release_runtime, dependencies=[media.path, relay.path])
    with pytest.raises(RuntimeError, match="test-only extension wheels"):
        verifier._assert_extension_wheel_layout(root.path, "root")


def test_private_graphs_keep_material_checks_for_public_dependencies(tmp_path, release_runtime):
    child = _build(
        _artifact(tmp_path / "child.duckdb_extension"),
        release_runtime,
        license_expression="Apache-2.0 AND LGPL-2.1-or-later",
        test_only=True,
    )
    public = _rewrite_wheel_metadata(
        child.path, tmp_path / "public", lambda contents: contents.replace("Classifier: Private :: Do Not Upload\n", "")
    )
    with pytest.raises(ValueError, match="missing its source and relinking materials"):
        _build(_artifact(tmp_path / "root.duckdb_extension"), release_runtime, dependencies=[public], test_only=True)


def test_media_wheel_contains_its_libraries_notices_and_public_source_link(media_dependency, release_runtime):
    layout = verifier._assert_extension_wheel_layout(media_dependency.path, "native_media")
    assert [requirement.name for requirement in layout.requirements] == ["vane-ai"]
    with zipfile.ZipFile(media_dependency.path) as wheel:
        metadata = BytesParser().parsebytes(
            wheel.read(next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")))
        )
        assert metadata.get_all("Project-URL") == [f"Native media sources, {release_runtime[2][1]['source']['url']}"]
        assert "LGPL-2.1-or-later" in metadata["License-Expression"]
        libraries = [name for name in wheel.namelist() if "/runtime/.libs/" in name]
        assert {name.rsplit("/", 1)[1]: wheel.read(name) for name in libraries} == release_runtime[2][2]
        assert any("/licenses/runtime/soxr.txt" in name for name in wheel.namelist())
        assert not any(name.startswith("vane_media_runtime") for name in wheel.namelist())


@pytest.mark.parametrize("reader", ["builder", "verifier"])
@pytest.mark.parametrize(
    "damage",
    [
        "library",
        "missing-library",
        "manifest",
        "signature",
        "notice",
        "undeclared-notice",
        "extra-library",
        "source-url",
        "license-expression",
    ],
)
def test_bundled_runtime_is_verified_even_with_a_recomputed_record(media_dependency, tmp_path, reader, damage):
    with zipfile.ZipFile(media_dependency.path) as wheel:
        names = wheel.namelist()
    library = next(name for name in names if "/runtime/.libs/" in name)
    notice = next(name for name in names if name.endswith("/licenses/runtime/soxr.txt"))
    metadata = next(name for name in names if name.endswith(".dist-info/METADATA"))
    transforms, removed, extra = {}, set(), {}
    if damage == "missing-library":
        removed.add(library)
    elif damage == "extra-library":
        extra[library.rsplit("/", 1)[0] + "/extra.so"] = b"unknown library"
    elif damage == "license-expression":
        transforms[metadata] = lambda value: b"\n".join(
            b"License-Expression: Apache-2.0" if line.startswith(b"License-Expression:") else line
            for line in value.split(b"\n")
        )
    elif damage == "source-url":
        transforms[metadata] = lambda value: value.replace(
            b"Project-URL: Native media sources, ", b"Project-URL: Wrong sources, "
        )
    elif damage == "undeclared-notice":
        transforms[metadata] = lambda value: b"\n".join(
            line for line in value.split(b"\n") if not line.startswith(b"License-File: runtime/soxr.txt")
        )
    else:
        target = {
            "library": library,
            "notice": notice,
            "manifest": next(name for name in names if name.endswith("/runtime-manifest.json")),
            "signature": next(name for name in names if name.endswith("/runtime-manifest.sig")),
        }[damage]
        transforms[target] = lambda value: b"changed bytes"
    directory = tmp_path / "damaged"
    directory.mkdir()
    damaged = _rewrite_wheel(
        media_dependency.path,
        directory / media_dependency.path.name,
        transforms=transforms,
        removed_members=removed,
        extra_members=extra,
    )
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        if reader == "builder":
            _read_dependency_wheels([damaged])
        else:
            verifier._assert_extension_wheel_layout(damaged, "native_media")
