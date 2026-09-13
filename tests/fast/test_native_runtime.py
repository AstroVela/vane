# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import struct
import subprocess
from dataclasses import FrozenInstanceError, replace

import pytest

from vane import _native_runtime_format as fmt
from vane.extensions import (
    DynamicExtensionDescriptor,
    DynamicExtensionError,
    NativeRuntimeReference,
)
from vane_packaging.media_runtime import stage_libraries, validate_library_graph
from vane_packaging.media_version import identity_version

GIT_COMMIT = "0123456789abcdef" * 2 + "01234567"
NAMESPACE = "vane_media_" + GIT_COMMIT
IDENTITY = {
    "git_commit": GIT_COMMIT,
    "git_dirty": False,
    "vane_version": "0.2.0.dev612",
}
RUNTIME_VERSION = identity_version(IDENTITY)
SOXR_LIBRARY = NAMESPACE + "_libsoxr.so.0"


def manifest(library=b"library bytes"):
    return {
        "schema_version": 1,
        "distribution": "vane-media-runtime",
        "version": RUNTIME_VERSION,
        **IDENTITY,
        "platform": "manylinux_2_28_x86_64",
        "namespace": NAMESPACE,
        "license_expression": "LGPL-2.1-or-later",
        "source": {
            "filename": f"vane_media_runtime-{RUNTIME_VERSION}.tar.gz",
            "sha256": "1" * 64,
            "url": f"https://pypi.org/project/vane-media-runtime/{RUNTIME_VERSION}/#files",
        },
        "components": {
            "soxr": {
                "version": "0.1.3#8",
                "license": "LGPL-2.1-or-later",
                "notice_sha256": "2" * 64,
            }
        },
        "files": {
            SOXR_LIBRARY: {
                "sha256": hashlib.sha256(library).hexdigest(),
                "size": len(library),
                "needed": [],
                "component": "soxr",
            }
        },
    }


def test_runtime_manifest_requires_canonical_bounded_identity():
    document = fmt.canonical_json(manifest())
    assert fmt.reference(document)["manifest_sha256"] == hashlib.sha256(document).hexdigest()
    for contents in (
        document.rstrip(),
        json.dumps(manifest()).encode(),
        b'{"version":1,"version":2}\n',
        b"[0]\n",
    ):
        with pytest.raises(ValueError):
            fmt.parse_manifest(contents)
    for path in ("../soxr.so", "/soxr.so", "a/b", "LPT9.dll", "COM4", "x."):
        with pytest.raises(ValueError):
            fmt.filename(path)
    for value in ("01.0", "0.01", "1.0+local", "1.0/../../", True):
        with pytest.raises(ValueError):
            fmt.version(value)


def test_runtime_manifest_rejects_unreviewed_inventory_shapes():
    for mutate in (
        lambda value: value.update(schema_version=True),
        lambda value: value.update(platform="manylinux_02_28_x86_64"),
        lambda value: value["files"][SOXR_LIBRARY].update(size=True),
        lambda value: value["files"][SOXR_LIBRARY].update(needed=["../other.so"]),
        lambda value: value.update(git_commit="invalid"),
        lambda value: value.update(git_dirty="false"),
        lambda value: value.update(namespace="vane_media_other"),
        lambda value: value["source"].update(filename="different.tar.gz"),
        lambda value: value["source"].update(url="http://example.com/source"),
        lambda value: value["source"].update(url="https://user:secret@example.com/source"),
    ):
        value = manifest()
        mutate(value)
        with pytest.raises(ValueError):
            fmt.parse_manifest(fmt.canonical_json(value))


def test_runtime_library_bytes_and_exact_directory_are_checked(tmp_path):
    value = manifest()
    name = next(iter(value["files"]))
    library = tmp_path / name
    library.write_bytes(b"library bytes")
    fmt.verify_files(tmp_path, value)
    library.write_bytes(b"tampered data")
    with pytest.raises(ValueError, match="digest mismatch"):
        fmt.verify_files(tmp_path, value)
    library.unlink()
    with pytest.raises(ValueError, match="differs"):
        fmt.verify_files(tmp_path, value)
    library.symlink_to("/dev/zero")
    with pytest.raises((OSError, ValueError)):
        fmt.verify_files(tmp_path, value)


def test_runtime_trailer_preserves_footer_and_must_precede_signing():
    footer = b"metadata".ljust(256, b"\0") + bytes(256)
    extension = b"ELF payload" + footer
    bound = fmt.attach_trailer(extension, "a" * 64)
    assert bound[-512:] == footer
    assert fmt.trailer_digest(bound) == "a" * 64
    assert fmt.attach_trailer(bound, "b" * 64) == fmt.attach_trailer(extension, "b" * 64)
    with pytest.raises(ValueError, match="before signing"):
        fmt.attach_trailer(bound[:-1] + b"s", "b" * 64)


def test_descriptor_v2_roundtrip_and_immutable_runtime_reference():
    reference = NativeRuntimeReference(RUNTIME_VERSION, "a" * 64)
    descriptor = DynamicExtensionDescriptor(
        name="native_media",
        extension_version="1",
        abi_type="CPP",
        duckdb_source_id="b" * 40,
        vane_version="0.1.0",
        platform="linux_amd64",
        sha256="c" * 64,
        trust_identity="astrovela/vane",
        format_version=2,
        native_runtime=reference,
    )
    assert DynamicExtensionDescriptor.from_json(descriptor.to_json()) == descriptor
    with pytest.raises(FrozenInstanceError):
        reference.version = "0.2.0"
    value = descriptor.to_dict()
    value["native_runtime"]["version"] = "0.2.0"
    assert descriptor.native_runtime.version == RUNTIME_VERSION
    value["format_version"] = 1
    with pytest.raises(DynamicExtensionError):
        DynamicExtensionDescriptor.from_dict(value)


@pytest.fixture
def snapshot_inputs(tmp_path, monkeypatch):
    from vane import _native
    from vane import _native_runtime as runtime

    monkeypatch.setattr(runtime, "_selected", None)
    monkeypatch.setattr(runtime, "_override", None)
    monkeypatch.setattr(runtime, "_override_manifest_sha256", None)
    monkeypatch.setattr(runtime, "_override_distributed", False)
    monkeypatch.setattr(_native, "_verify_native_runtime_signature", lambda *args: True)
    value = manifest()
    document = fmt.canonical_json(value)
    source = tmp_path / "runtime"
    (source / ".libs").mkdir(parents=True)
    (source / ".libs" / SOXR_LIBRARY).write_bytes(b"library bytes")
    (source / fmt.MANIFEST).write_bytes(document)
    monkeypatch.setattr(runtime, "_runtime_source", lambda reference: (source, document, bytes(256), document, value))
    reference = NativeRuntimeReference.from_dict(fmt.reference(document))
    artifact = tmp_path / "native_media.duckdb_extension"
    contents = fmt.attach_trailer(b"extension payload" + bytes(512), reference.manifest_sha256)
    artifact.write_bytes(contents)
    descriptor = DynamicExtensionDescriptor(
        name="native_media",
        extension_version="1",
        abi_type="CPP",
        duckdb_source_id="b" * 40,
        vane_version="0.1.0",
        platform="linux_amd64",
        sha256=hashlib.sha256(contents).hexdigest(),
        trust_identity="astrovela/vane",
        format_version=2,
        native_runtime=reference,
    )
    return runtime, artifact, descriptor, tmp_path / "cache", source


def test_native_runtime_reuses_valid_snapshot_without_staging(snapshot_inputs, monkeypatch):
    runtime, artifact, descriptor, cache, _ = snapshot_inputs
    target = runtime.prepare_snapshot(artifact, descriptor, cache)
    runtime._selected = None  # A new process must validate and reuse the same disk cache.
    monkeypatch.setattr(runtime.tempfile, "mkdtemp", lambda **kwargs: pytest.fail("cache hit recopied the runtime"))
    assert runtime.prepare_snapshot(artifact, descriptor, cache) == target
    assert runtime.prepare_snapshot(artifact, descriptor, cache) == target
    assert runtime._selected == descriptor.native_runtime.manifest_sha256


@pytest.mark.skipif(os.name == "nt", reason="Windows permissions do not use umask")
@pytest.mark.parametrize("invalid_trailer", [False, True])
def test_native_runtime_staging_handles_restrictive_umask(snapshot_inputs, invalid_trailer):
    runtime, artifact, descriptor, cache, _ = snapshot_inputs
    if invalid_trailer:
        contents = fmt.attach_trailer(b"extension payload" + bytes(512), "f" * 64)
        artifact.write_bytes(contents)
        descriptor = replace(descriptor, sha256=hashlib.sha256(contents).hexdigest())
    previous_umask = os.umask(0o777)
    try:
        if invalid_trailer:
            with pytest.raises(ValueError, match="extension trailer differs"):
                runtime.prepare_snapshot(artifact, descriptor, cache)
        else:
            target = runtime.prepare_snapshot(artifact, descriptor, cache)
            assert target.read_bytes() == artifact.read_bytes()
            assert runtime.prepare_snapshot(artifact, descriptor, cache) == target
    finally:
        os.umask(previous_umask)

    assert not list(cache.rglob(".media-*"))
    for path in (cache, *cache.rglob("*")):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o400)
    if invalid_trailer:
        assert runtime._selected is None


@pytest.mark.parametrize("in_process", [False, True])
def test_custom_runtime_preparation_checks_boundary_even_when_already_loaded(snapshot_inputs, monkeypatch, in_process):
    from vane import extensions

    runtime, _, descriptor, _, source = snapshot_inputs
    monkeypatch.setattr(runtime, "_override", source)
    snapshot = [descriptor.to_dict()]
    monkeypatch.setattr(extensions, "_capture_dynamic_extension_snapshot", lambda connection: snapshot)
    if in_process:
        extensions._prepare_dynamic_extension_snapshot(object(), snapshot, in_process=True)
    else:
        with pytest.raises(ValueError, match="allow_distributed=True"):
            extensions._prepare_dynamic_extension_snapshot(object(), snapshot)


def test_distributed_custom_runtime_requires_explicit_opt_in(snapshot_inputs, monkeypatch):
    from vane import extensions

    runtime, _, descriptor, _, source = snapshot_inputs
    monkeypatch.setattr(
        extensions, "_serialized_dynamic_extension_snapshot_entries", lambda connection: [descriptor.to_json()]
    )
    runtime.use_native_media_runtime(source)
    assert extensions._capture_dynamic_extension_snapshot(object()) == [descriptor.to_dict()]
    with pytest.raises(ValueError, match="allow_distributed=True"):
        extensions._capture_dynamic_extension_snapshot_for_worker(object())
    with pytest.raises(ValueError, match="start a new process"):
        runtime.use_native_media_runtime(source, allow_distributed=True)


def test_distributed_snapshot_preserves_effective_identity_and_cache_isolation(snapshot_inputs, monkeypatch):
    from vane import extensions

    runtime, _, descriptor, _, source = snapshot_inputs
    original = [descriptor.to_dict()]
    monkeypatch.setattr(
        extensions, "_serialized_dynamic_extension_snapshot_entries", lambda connection: [descriptor.to_json()]
    )
    runtime.use_native_media_runtime(source, allow_distributed=True)
    snapshot = extensions._capture_dynamic_extension_snapshot_for_worker(object())
    digest = hashlib.sha256((source / fmt.MANIFEST).read_bytes()).hexdigest()
    assert snapshot == [dict(descriptor.to_dict(), effective_runtime_sha256=digest)]
    assert extensions._capture_dynamic_extension_snapshot(object()) == snapshot
    assert str(source) not in json.dumps(snapshot)
    assert extensions._normalize_dynamic_extension_snapshot(snapshot) == snapshot
    assert extensions._parse_dynamic_extension_snapshot(snapshot) == (descriptor,)
    official_key = extensions._dynamic_extension_snapshot_cache_identity(original)
    custom_key = extensions._dynamic_extension_snapshot_cache_identity(snapshot)
    assert official_key == ((descriptor.name, descriptor.to_json()),)
    assert custom_key != official_key
    snapshot[0]["effective_runtime_sha256"] = "f" * 64
    assert extensions._dynamic_extension_snapshot_cache_identity(snapshot) != custom_key
    # Runtime selection is transport metadata, never a change to the signed descriptor.
    with pytest.raises(DynamicExtensionError, match="unknown"):
        DynamicExtensionDescriptor.from_dict(snapshot[0])


@pytest.mark.parametrize("digest", [None, True, "A" * 64, "a" * 63, [], "../runtime"])
def test_snapshot_rejects_malformed_runtime_selection(snapshot_inputs, digest):
    from vane import extensions

    _, _, descriptor, _, _ = snapshot_inputs
    with pytest.raises(DynamicExtensionError, match="canonical SHA-256"):
        extensions._normalize_dynamic_extension_snapshot([dict(descriptor.to_dict(), effective_runtime_sha256=digest)])


def test_snapshot_rejects_mixed_runtime_selections(snapshot_inputs):
    from vane import extensions

    _, _, descriptor, _, _ = snapshot_inputs
    custom = dict(descriptor.to_dict(), effective_runtime_sha256="a" * 64)
    other = replace(descriptor, name="other_media").to_dict()
    for entry in (other, dict(other, effective_runtime_sha256="b" * 64)):
        with pytest.raises(DynamicExtensionError, match="selections disagree"):
            extensions._normalize_dynamic_extension_snapshot([custom, entry])
    plain = replace(descriptor, native_runtime=None, format_version=1).to_dict()
    with pytest.raises(DynamicExtensionError, match="requires a native runtime"):
        extensions._normalize_dynamic_extension_snapshot([dict(plain, effective_runtime_sha256="a" * 64)])


def test_distributed_worker_requires_independent_local_authorization(snapshot_inputs, monkeypatch):
    runtime, _, descriptor, _, source = snapshot_inputs
    digest = hashlib.sha256((source / fmt.MANIFEST).read_bytes()).hexdigest()
    monkeypatch.delenv("VANE_NATIVE_MEDIA_RUNTIME", raising=False)
    with pytest.raises(ValueError, match="VANE_NATIVE_MEDIA_RUNTIME"):
        runtime.prepare_distributed_runtime([descriptor], digest)
    assert runtime._selected is None
    worker_source = source.parent / "worker-local-runtime"
    shutil.copytree(source, worker_source)
    monkeypatch.setenv("VANE_NATIVE_MEDIA_RUNTIME", str(worker_source))
    runtime.prepare_distributed_runtime([descriptor], digest)
    assert runtime._override == worker_source
    assert runtime.distributed_runtime_selection([descriptor]) == digest
    with pytest.raises(ValueError, match="differs from the coordinator"):
        runtime.prepare_distributed_runtime([descriptor], "f" * 64)
    with pytest.raises(ValueError, match="differs from the coordinator"):
        runtime.prepare_distributed_runtime([descriptor], None)


def test_distributed_worker_does_not_apply_override_to_official_queries(snapshot_inputs, monkeypatch):
    runtime, _, descriptor, _, source = snapshot_inputs
    monkeypatch.setenv("VANE_NATIVE_MEDIA_RUNTIME", str(source))
    runtime.prepare_distributed_runtime([descriptor], None)
    assert runtime._override is None
    assert runtime.distributed_runtime_selection([descriptor]) is None


@pytest.mark.parametrize("in_process", [False, True])
def test_distributed_worker_checks_library_bytes_before_already_loaded_shortcut(
    snapshot_inputs, monkeypatch, in_process
):
    from vane import extensions

    runtime, _, descriptor, _, source = snapshot_inputs
    runtime.use_native_media_runtime(source, allow_distributed=True)
    snapshot = [dict(descriptor.to_dict(), effective_runtime_sha256=runtime._override_manifest_sha256)]
    monkeypatch.setattr(extensions, "_capture_dynamic_extension_snapshot", lambda connection: [descriptor.to_dict()])
    (source / ".libs" / SOXR_LIBRARY).write_bytes(b"tampered bytes")
    with pytest.raises(ValueError, match="digest mismatch"):
        extensions._prepare_dynamic_extension_snapshot(object(), snapshot, in_process=in_process)


def test_custom_runtime_manifest_is_frozen_at_explicit_selection(tmp_path, monkeypatch):
    from vane import _native_runtime as runtime

    for name, value in (
        ("_override", None),
        ("_override_manifest_sha256", None),
        ("_selected", None),
        ("_override_distributed", False),
    ):
        monkeypatch.setattr(runtime, name, value)
    official = tmp_path / "official"
    (official / ".libs").mkdir(parents=True)
    document = fmt.canonical_json(manifest())
    (official / fmt.MANIFEST).write_bytes(document)
    (official / fmt.SIGNATURE).write_bytes(bytes(256))
    (official / ".libs" / SOXR_LIBRARY).write_bytes(b"library bytes")
    custom = tmp_path / "custom"
    shutil.copytree(official, custom)

    class Distribution:
        version = RUNTIME_VERSION

        def locate_file(self, name):
            return official

    monkeypatch.setattr(runtime, "distribution", lambda name: Distribution())
    runtime.use_native_media_runtime(custom, allow_distributed=True)
    replacement = b"a different rebuilt library"
    (custom / ".libs" / SOXR_LIBRARY).write_bytes(replacement)
    (custom / fmt.MANIFEST).write_bytes(fmt.canonical_json(manifest(replacement)))
    with pytest.raises(ValueError, match="changed after selection"):
        runtime._runtime_source(NativeRuntimeReference.from_dict(fmt.reference(document)))


@pytest.mark.parametrize(
    "damage",
    [
        "source-extension",
        "source-library",
        "source-signature",
        "manifest",
        "signature",
        "effective",
        "extension",
        "library",
        "missing-library",
        "extra-library",
        "writable-library",
        "hardlink-library",
        "symlink-library",
        "symlink-destination",
        "empty-destination",
    ],
)
def test_native_runtime_cache_hit_revalidates_sources_and_snapshot(snapshot_inputs, monkeypatch, damage):
    from vane import _native

    runtime, artifact, descriptor, cache, source = snapshot_inputs
    target = runtime.prepare_snapshot(artifact, descriptor, cache)
    runtime._selected = None
    destination = target.parent
    library = destination / ".libs" / SOXR_LIBRARY
    if damage == "source-extension":
        artifact.write_bytes(b"changed extension")
    elif damage == "source-library":
        (source / ".libs" / SOXR_LIBRARY).write_bytes(b"changed library")
    elif damage == "source-signature":
        monkeypatch.setattr(_native, "_verify_native_runtime_signature", lambda *args: False)
    elif damage == "missing-library":
        library.unlink()
    elif damage == "extra-library":
        (library.parent / "extra.so").write_bytes(b"unexpected")
    elif damage == "writable-library":
        library.chmod(0o600)
    elif damage == "hardlink-library":
        (source / "linked.so").hardlink_to(library)
    elif damage == "symlink-library":
        library.unlink()
        library.symlink_to(source / ".libs" / SOXR_LIBRARY)
    elif damage == "symlink-destination":
        moved = destination.with_name("moved")
        destination.rename(moved)
        destination.symlink_to(moved, target_is_directory=True)
    elif damage == "empty-destination":
        shutil.rmtree(destination)
        destination.mkdir(mode=0o700)
    else:
        path = {
            "manifest": destination / fmt.MANIFEST,
            "signature": destination / fmt.SIGNATURE,
            "effective": destination / "effective-runtime.json",
            "extension": target,
            "library": library,
        }[damage]
        path.chmod(0o600)
        path.write_bytes(b"tampered")
        path.chmod(0o400)
    monkeypatch.setattr(runtime.tempfile, "mkdtemp", lambda **kwargs: pytest.fail("invalid cache was restaged"))
    with pytest.raises((ValueError, OSError)):
        runtime.prepare_snapshot(artifact, descriptor, cache)
    assert runtime._selected is None


@pytest.mark.parametrize("corrupt_winner", [False, True])
def test_native_runtime_validates_a_concurrent_snapshot_publisher(snapshot_inputs, monkeypatch, corrupt_winner):
    runtime, artifact, descriptor, cache, _ = snapshot_inputs

    def publish_elsewhere(staging, destination):
        shutil.copytree(staging, destination)
        if corrupt_winner:
            library = destination / ".libs" / SOXR_LIBRARY
            library.chmod(0o600)
            library.write_bytes(b"changed library")
            library.chmod(0o400)
        raise FileExistsError("another process published first")

    monkeypatch.setattr(runtime.os, "rename", publish_elsewhere)
    if corrupt_winner:
        with pytest.raises(ValueError, match="digest mismatch"):
            runtime.prepare_snapshot(artifact, descriptor, cache)
        assert runtime._selected is None
    else:
        target = runtime.prepare_snapshot(artifact, descriptor, cache)
        assert target.read_bytes() == artifact.read_bytes()
    assert not list(cache.rglob(".media-*"))


def test_real_elf_recursive_dependency_closure_and_relocation(tmp_path):
    compiler = shutil.which("cc")
    patcher = shutil.which("patchelf")
    if compiler is None or patcher is None:
        pytest.skip("real ELF relocation needs cc and patchelf")
    sdk = tmp_path / "sdk"
    lib = sdk / "lib"
    lib.mkdir(parents=True)
    (tmp_path / "leaf.c").write_text("int media_leaf(void) { return 42; }\n")
    (tmp_path / "root.c").write_text("extern int media_leaf(void); int media_root(void) { return media_leaf(); }\n")
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            str(tmp_path / "leaf.c"),
            "-Wl,-soname,libleaf.so",
            "-o",
            str(lib / "libleaf.so"),
        ],
        check=True,
    )
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            str(tmp_path / "root.c"),
            "-Wl,-soname,libroot.so",
            f"-L{lib}",
            "-lleaf",
            "-o",
            str(lib / "libroot.so"),
        ],
        check=True,
    )
    destination = tmp_path / "relocated"
    mapping = stage_libraries(
        sdk,
        destination,
        namespace="vane_media_test",
        platform="manylinux_2_39_x86_64",
        patchelf=patcher,
    )
    libraries = {path.name: path.read_bytes() for path in destination.iterdir()}
    graph = validate_library_graph(libraries, "manylinux_2_39_x86_64")
    assert mapping["libleaf.so"] in graph[mapping["libroot.so"]]
    del libraries[mapping["libleaf.so"]]
    with pytest.raises(ValueError, match="non-policy"):
        validate_library_graph(libraries, "manylinux_2_39_x86_64")


@pytest.mark.parametrize("runpath", ["$ORIGIN", "$ORIGIN/.libs"])
@pytest.mark.parametrize("damage", ["missing", "wrong", "duplicate", "legacy-rpath"])
def test_required_media_runpath_is_present_and_exact(tmp_path, runpath, damage):
    from elftools.elf.elffile import ELFFile

    from vane_packaging.extension_wheel import _parse_elf_dynamic_linkage

    compiler, patcher = shutil.which("cc"), shutil.which("patchelf")
    if compiler is None or patcher is None:
        pytest.skip("ELF validation needs cc and patchelf")
    source = tmp_path / "library.c"
    source.write_text("int media_value(void) { return 42; }\n")
    library = tmp_path / "library.so"
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            str(source),
            "-Wl,-soname,library.so",
            "-Wl,--enable-new-dtags",
            f"-Wl,-rpath,{runpath}",
            "-o",
            str(library),
        ],
        check=True,
    )
    _parse_elf_dynamic_linkage(library.read_bytes(), description="media", allowed_runpath=runpath)
    if damage == "duplicate":
        contents = bytearray(library.read_bytes())
        dynamic = ELFFile(io.BytesIO(contents)).get_section_by_name(".dynamic")
        tags = list(dynamic.iter_tags())
        offset = next(tag.entry.d_val for tag in tags if tag.entry.d_tag == "DT_RUNPATH")
        index = next(i for i, tag in enumerate(tags) if tag.entry.d_tag == "DT_SONAME")
        struct.pack_into("<QQ", contents, dynamic.header.sh_offset + index * 16, 29, offset)
        library.write_bytes(contents)
    else:
        arguments = {
            "missing": ["--remove-rpath"],
            "wrong": ["--set-rpath", runpath + ":/tmp"],
            "legacy-rpath": ["--force-rpath", "--set-rpath", runpath],
        }[damage]
        subprocess.run([patcher, *arguments, str(library)], check=True)
    with pytest.raises(ValueError, match="RUNPATH|RPATH"):
        _parse_elf_dynamic_linkage(library.read_bytes(), description="media", allowed_runpath=runpath)
