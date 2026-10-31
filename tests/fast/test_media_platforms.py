# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable validation of owned Mach-O and PE media dependencies."""

import ctypes
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.tags import sys_tags

import vane
from scripts import prepare_dynamic_media_extension as dynamic_media
from tests.fast.test_extension_wheel import _synthetic_macho, _synthetic_pe
from tests.fast.test_native_runtime import manifest
from vane import _native_runtime_format as fmt
from vane._native_runtime_format import filesystem_path
from vane_packaging import extension_wheel as binary
from vane_packaging import media_binary
from vane_packaging.media_binary import relocate_non_elf
from vane_packaging.media_runtime import component_notice, stage_libraries, validate_library_graph


@pytest.mark.parametrize("library", ["rstrtmgr.dll", "wsock32.dll"])
def test_windows_engine_system_imports_do_not_allow_lookalike_dlls(library):
    contents = _synthetic_pe(architecture="amd64", imports=(library,))
    assert binary._validate_windows_pe_platform(
        contents, "win_amd64", description="engine", interpreter_tag="cp312"
    ) == (library,)
    with pytest.raises(ValueError, match="non-policy Windows DLLs"):
        binary._validate_windows_pe_platform(
            _synthetic_pe(architecture="amd64", imports=("private_" + library,)),
            "win_amd64",
            description="engine",
            interpreter_tag="cp312",
        )


@pytest.mark.parametrize("persistent", [False, True])
def test_windows_binary_replacement_recovers_only_bounded_file_contention(tmp_path, monkeypatch, persistent):
    source, target = tmp_path / "new.dll", tmp_path / "old.dll"
    source.write_bytes(b"new binary")
    target.write_bytes(b"original binary")
    replace = Path.replace
    attempts = []

    def contend(path, destination):
        attempts.append(path)
        if persistent or len(attempts) <= 2:
            error = PermissionError("file is being inspected")
            error.winerror = 5
            raise error
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", contend)
    monkeypatch.setattr(media_binary, "sleep", lambda _: None)
    if persistent:
        with pytest.raises(PermissionError, match="being inspected"):
            media_binary._replace_windows_binary(source, target)
        assert 2 < len(attempts) < 10
        assert source.read_bytes() == b"new binary"
        assert target.read_bytes() == b"original binary"
    else:
        media_binary._replace_windows_binary(source, target)
        assert target.read_bytes() == b"new binary"
        assert not source.exists()


def test_component_notices_allow_only_reviewed_line_ending_variants(tmp_path):
    notice = tmp_path / "copyright"
    reviewed = b"Reviewed license\nCopyright contributors\n"
    digest = hashlib.sha256(reviewed).hexdigest()
    for contents in (reviewed, reviewed.replace(b"\n", b"\r\n")):
        notice.write_bytes(contents)
        assert component_notice(notice, digest) == reviewed
    notice.write_bytes(reviewed.replace(b"Reviewed license", b"Changed license"))
    with pytest.raises(ValueError, match="unreviewed media component notice"):
        component_notice(notice, digest)


@pytest.mark.parametrize("family", ["windows", "macos"])
def test_native_media_dependencies_require_an_explicit_bundled_library(family):
    library = "vane_media_release_codec.dll" if family == "windows" else "vane_media_release_codec.dylib"
    if family == "windows":
        contents = _synthetic_pe(architecture="amd64", imports=(library, "KERNEL32.dll"))

        def validate(names):
            return binary._validate_windows_pe_platform(
                contents, "win_amd64", description="codec", interpreter_tag="cp312", bundled_libraries=names
            )
    else:
        contents = _synthetic_macho(
            architecture="arm64", minimum_macos=(11, 0, 0), dynamic_libraries=(f"@loader_path/.libs/{library}",)
        )

        def validate(names):
            return binary._validate_macos_binary_platform(
                contents, "macosx_11_0_arm64", description="codec", bundled_libraries=names
            )

    with pytest.raises(ValueError, match="non-policy|non-system"):
        validate(())
    validate((library,))
    with pytest.raises(ValueError, match="non-policy|non-system"):
        validate(("another-codec.dll",))
    with pytest.raises(ValueError, match="plain filenames"):
        validate(("../outside.dll",))


@pytest.mark.parametrize("dependency", ["@loader_path/../codec.dylib", "/tmp/codec.dylib", "@rpath/codec.dylib"])
def test_native_media_macos_bundling_does_not_allow_foreign_dependency_paths(dependency):
    contents = _synthetic_macho(architecture="arm64", minimum_macos=(11, 0, 0), dynamic_libraries=(dependency,))
    with pytest.raises(ValueError, match="non-system"):
        binary._validate_macos_binary_platform(
            contents, "macosx_11_0_arm64", description="codec", bundled_libraries=("codec.dylib",)
        )


def test_macos_extension_relocation_preserves_the_complete_duckdb_metadata(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    (runtime / ".libs").mkdir(parents=True)
    document = manifest()
    name = document["namespace"] + "_libsoxr.dylib"
    library = _synthetic_macho(
        architecture="arm64",
        minimum_macos=(11, 0, 0),
        dynamic_libraries=(f"@loader_path/{name}",),
        dynamic_library_command=0xD,
    )
    record = next(iter(document["files"].values()))
    record.update(sha256=hashlib.sha256(library).hexdigest(), size=len(library))
    document.update(platform="macosx_11_0_arm64", files={name: record})
    contents = fmt.canonical_json(document)
    (runtime / fmt.MANIFEST).write_bytes(contents)
    (runtime / ".libs" / name).write_bytes(library)
    original = _synthetic_macho(
        architecture="arm64", minimum_macos=(11, 0, 0), dynamic_libraries=("@rpath/libsoxr.dylib",)
    )
    relocated = _synthetic_macho(
        architecture="arm64", minimum_macos=(11, 0, 0), dynamic_libraries=(f"@loader_path/.libs/{name}",)
    )
    artifact = tmp_path / "native_media.duckdb_extension"
    artifact.write_bytes(original)
    platform_file = tmp_path / "platform"
    platform_file.write_text("osx_arm64")
    # Use the production CMake appender: its WebAssembly section header is
    # outside the final 512-byte footer, and must also stay out of Mach-O tools.
    subprocess.run(
        [
            "cmake",
            f"-DEXTENSION={artifact}",
            f"-DPLATFORM_FILE={platform_file}",
            "-DABI_TYPE=CPP",
            "-DVERSION_FIELD=v1.5.5",
            "-DEXTENSION_VERSION=test",
            f"-DNULL_FILE={dynamic_media.ROOT / 'external/duckdb/scripts/null.txt'}",
            "-P",
            str(dynamic_media.ROOT / "external/duckdb/scripts/append_metadata.cmake"),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    metadata = artifact.read_bytes()[len(original) :]
    assert len(metadata) > fmt.DUCKDB_FOOTER_SIZE
    expected_inputs = iter((original, relocated))

    def relocate(target, mapping, platform):
        assert target.read_bytes() == next(expected_inputs), "Mach-O tools must receive only the native binary"
        assert mapping == {"libsoxr.dylib": name}
        assert platform == document["platform"]
        target.write_bytes(relocated)

    monkeypatch.setattr(dynamic_media, "relocate_non_elf", relocate)
    for _ in range(2):
        dynamic_media.prepare(artifact, runtime)
        assert artifact.read_bytes() == fmt.attach_trailer(relocated + metadata, hashlib.sha256(contents).hexdigest())


@pytest.mark.skipif(sys.platform not in {"win32", "darwin"}, reason="native Mach-O/PE loader regression")
@pytest.mark.parametrize("long_path", [False, True])
def test_namespaced_media_libraries_load_after_relocation(tmp_path, long_path):
    if shutil.which("cmake") is None:
        pytest.skip("requires CMake and a native C compiler")
    if sys.platform == "win32":
        pytest.importorskip("lief")
    source = tmp_path / "source"
    source.mkdir()
    (source / "codec.c").write_text(
        "#ifdef _WIN32\n__declspec(dllexport)\n#endif\nint media_answer(void) { return 21; }\n"
    )
    (source / "consumer.c").write_text(
        "#ifdef _WIN32\n__declspec(dllimport)\n#endif\nint media_answer(void);\n"
        "#ifdef _WIN32\n__declspec(dllexport)\n#endif\nint media_decode(void) { return media_answer() * 2; }\n"
    )
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.29)\nproject(media_probe C)\n"
        "if(APPLE)\nadd_link_options(-Wl,-headerpad_max_install_names)\nendif()\n"
        "add_library(codec SHARED codec.c)\nadd_library(consumer SHARED consumer.c)\n"
        "target_link_libraries(consumer PRIVATE codec)\n"
    )
    build = tmp_path / "build"
    for arguments in (
        ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"],
        ["cmake", "--build", str(build), "--config", "Release"],
    ):
        result = subprocess.run(arguments, capture_output=True, text=True, errors="replace", timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    prefix = tmp_path / "sdk"
    library_dir = prefix / ("bin" if sys.platform == "win32" else "lib")
    library_dir.mkdir(parents=True)
    suffix = ".dll" if sys.platform == "win32" else ".dylib"
    for path in build.rglob(f"*{suffix}"):
        shutil.copyfile(path, library_dir / path.name)
    platform = next(tag.platform for tag in sys_tags() if tag.platform.startswith(("win_", "macosx_")))
    staged = tmp_path / "staged"
    # ctypes keeps loaded libraries alive across parameter cases on Windows.
    # Give each probe its own names so a previous load cannot hide missing DLLs.
    namespace = "vane_media_probe_long" if long_path else "vane_media_probe"
    mapping = stage_libraries(prefix, staged, namespace=namespace, platform=platform)
    assert len(mapping) == 2
    validate_library_graph({path.name: path.read_bytes() for path in staged.iterdir()}, platform)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    staged.rename(relocated / ".libs")
    consumer_name = "consumer.dll" if sys.platform == "win32" else "libconsumer.dylib"
    extension = relocated / ("media-probe" + suffix)
    shutil.copyfile(library_dir / consumer_name, extension)
    relocate_non_elf(extension, mapping, platform)
    shutil.rmtree(prefix)
    shutil.rmtree(build)
    if long_path:
        directory = tmp_path
        while len(str(directory)) < 280:
            directory /= "long-media-path-component"
        filesystem_path(directory).mkdir(parents=True)
        relocated.rename(filesystem_path(directory / relocated.name))
        relocated = directory / relocated.name
        extension = relocated / extension.name
    if sys.platform == "win32":
        with pytest.raises(OSError):
            ctypes.CDLL(str(extension), winmode=0)
    assert vane.ray_cxx.load_media_probe_for_test(str(extension)) == 42
    consumer = mapping[consumer_name]
    library = ctypes.CDLL(str(filesystem_path(relocated / ".libs" / consumer)))
    assert library.media_decode() == 42
