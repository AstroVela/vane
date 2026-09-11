# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import json
import shutil
import struct
import subprocess
from dataclasses import FrozenInstanceError

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
