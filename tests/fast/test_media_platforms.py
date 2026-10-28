# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable validation of owned Mach-O and PE media dependencies."""

import ctypes
import hashlib
import shutil
import subprocess
import sys

import pytest
from packaging.tags import sys_tags

import vane
from tests.fast.test_extension_wheel import _synthetic_macho, _synthetic_pe
from vane_packaging import extension_wheel as binary
from vane_packaging.media_binary import relocate_non_elf
from vane_packaging.media_runtime import component_notice, stage_libraries, validate_library_graph


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


@pytest.mark.skipif(sys.platform not in {"win32", "darwin"}, reason="native Mach-O/PE loader regression")
def test_namespaced_media_libraries_load_after_relocation(tmp_path):
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
    mapping = stage_libraries(prefix, staged, namespace="vane_media_probe", platform=platform)
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
    if sys.platform == "win32":
        with pytest.raises(OSError):
            ctypes.CDLL(str(extension), winmode=0)
    assert vane.ray_cxx.load_media_probe_for_test(str(extension)) == 42
    consumer = mapping[consumer_name]
    library = ctypes.CDLL(str(relocated / ".libs" / consumer))
    assert library.media_decode() == 42
