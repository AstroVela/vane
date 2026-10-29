# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual media CMake function with small stand-ins for DuckDB targets."""

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATORS = ["Visual Studio 17 2022"] if sys.platform == "win32" else ["Ninja", "Unix Makefiles"]


@pytest.fixture
def media_project_directory(tmp_path):
    if sys.platform != "win32":
        yield tmp_path
        return
    # MSBuild excludes all of AppData from dependency tracking, including
    # pytest's default temporary directory even when TEMP/TMP are overridden.
    # Keep incremental-build inputs and outputs in the checkout's build tree.
    directory = ROOT / "build" / "media-configuration-tests"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        yield Path(temporary)


@pytest.fixture
def media_project(media_project_directory, monkeypatch):
    if sys.platform != "win32" and (not shutil.which("cc") or not shutil.which("ninja")):
        pytest.skip("CMake linkage test requires a C compiler and Ninja")
    root = media_project_directory / "project"
    if sys.platform == "win32":
        # MSBuild excludes files below TEMP from dependency tracking. Keep
        # compiler scratch files separate from the fixture's source/builds.
        compiler_temp = root / "compiler-temp"
        compiler_temp.mkdir(parents=True)
        monkeypatch.setenv("TEMP", str(compiler_temp))
        monkeypatch.setenv("TMP", str(compiler_temp))
    cmake_path = root / "external/duckdb/extension/media_common/extension.cmake"
    cmake_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / cmake_path.relative_to(root), cmake_path)
    (root / "cmake").mkdir()
    (root / "cmake/FindZLIB.cmake").write_text(
        "if(NOT TARGET ZLIB::ZLIB)\nadd_library(ZLIB::ZLIB INTERFACE IMPORTED)\nendif()\nset(ZLIB_FOUND TRUE)\n"
    )
    shutil.copyfile(
        ROOT / "cmake/prepare_dynamic_media_extension.cmake", root / "cmake/prepare_dynamic_media_extension.cmake"
    )
    # Only relocation is replaced: the real CMake function must schedule it.
    for relative in (
        "scripts/prepare_dynamic_media_extension.py",
        "vane/_native_runtime_format.py",
        "vane_packaging/media_runtime.py",
        "vane_packaging/media_binary.py",
        "vane_packaging/extension_wheel.py",
        "vane_packaging/archive_safety.py",
        "vane_packaging/artifact_limits.py",
        "vane_packaging/extension_materials.py",
        "vane_packaging/manylinux_policy.py",
        "vane_packaging/_vendor/auditwheel/manylinux-policy.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    (root / "scripts/prepare_dynamic_media_extension.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "artifact = Path(sys.argv[sys.argv.index('--artifact') + 1])\n"
        "with artifact.with_suffix('.runs').open('a') as stream: stream.write('prepared\\n')\n"
    )
    sdk = root / "sdk"
    (sdk / "include/boost/multiprecision").mkdir(parents=True)
    (sdk / "include/boost/multiprecision/cpp_int.hpp").write_text("#define SDK_BOOST 42\n")
    config = sdk / "share/boost_multiprecision/boost_multiprecision-config.cmake"
    config.parent.mkdir(parents=True)
    config.write_text(
        "find_package(boost_config CONFIG REQUIRED)\n"
        "add_library(Boost::multiprecision INTERFACE IMPORTED)\n"
        'set_property(TARGET Boost::multiprecision PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../include")\n'
    )
    dependency = sdk / "share/boost_config/boost_config-config.cmake"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("add_library(Boost::config INTERFACE IMPORTED)\n")
    (root / "library.c").write_text("int fixture(void) { return 42; }\n")
    (sdk / "lib").mkdir()
    if sys.platform == "win32":
        (root / "library-probe").mkdir()
        (root / "library-probe/CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.29)\nproject(media_import_probe C)\n"
            "add_library(soxr SHARED ../library.c)\n"
            "set_target_properties(soxr PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)\n"
            "add_library(z SHARED ../library-z.c)\n"
            "set_target_properties(z PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)\n"
        )
        (root / "library-z.c").write_text("int z_fixture(void) { return 43; }\n")
        _run("cmake", "-S", str(root / "library-probe"), "-B", str(root / "library-build"), "-G", GENERATORS[0])
        _run("cmake", "--build", str(root / "library-build"), "--config", "Release")
        library = next((root / "library-build").rglob("soxr.lib"))
        library_name = "{}.lib"
    else:
        library_name = "lib{}.dylib" if sys.platform == "darwin" else "lib{}.so"
        library = sdk / "lib" / library_name.format("soxr")
        _run("cc", "-shared", "-fPIC", str(root / "library.c"), "-o", str(library))
    for name in (
        "soxr",
        "avformat",
        "avcodec",
        "avutil",
        "swscale",
        "swresample",
        "sndfile",
        "tiff",
        "jpeg",
        "z",
        "webp",
        "webpdemux",
    ):
        destination = sdk / "lib" / library_name.format(name)
        source = next((root / "library-build").rglob("z.lib")) if sys.platform == "win32" and name == "z" else library
        if destination != source:
            shutil.copyfile(source, destination)
    runtime = root / "runtime"
    runtime.mkdir()
    (runtime / ".libs").mkdir()
    (runtime / ".libs/fixture.so").write_text("shared runtime fixture\n")
    (runtime / "runtime-manifest.json").write_text("{}\n")
    (root / "extension.c").write_text(
        "#include <boost/multiprecision/cpp_int.hpp>\nint probe(void) { return SDK_BOOST; }\n"
    )
    (root / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.29)
project(media_probe LANGUAGES C CXX)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(EXTENSION_STATIC_BUILD ON)
set(VANE_MEDIA_RUNTIME_SDK "${CMAKE_CURRENT_SOURCE_DIR}/sdk")
set(VANE_MEDIA_RUNTIME_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/runtime")
add_library(file_extension INTERFACE)
function(build_static_extension name)
  add_library(${name}_extension STATIC extension.c)
endfunction()
function(build_loadable_extension name)
  add_library(${name}_loadable_extension SHARED extension.c)
  set_target_properties(${name}_loadable_extension PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endfunction()
include(external/duckdb/extension/media_common/extension.cmake)
vane_build_native_media_extension()
""")
    return root


def _run(*command):
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL export decorations")
def test_windows_static_engine_keeps_loadable_extension_entrypoints(tmp_path):
    header = ROOT / "external/duckdb/src/include/duckdb/common/winapi.hpp"
    (tmp_path / "probe.cpp").write_text(
        f'#include "{header.as_posix()}"\nextern "C" DUCKDB_EXTENSION_API int media_answer() {{ return 42; }}\n'
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.29)\nproject(extension_export_probe CXX)\n"
        "add_library(probe SHARED probe.cpp)\n"
        "target_compile_definitions(probe PRIVATE DUCKDB_STATIC_BUILD DUCKDB_BUILD_LOADABLE_EXTENSION)\n"
    )
    build = tmp_path / "build"
    _run("cmake", "-S", str(tmp_path), "-B", str(build), "-G", GENERATORS[0])
    _run("cmake", "--build", str(build), "--config", "Release")
    assert ctypes.CDLL(str(next(build.rglob("probe.dll")))).media_answer() == 42


def test_dynamic_media_sdk_and_incremental_relocation(media_project):
    root = media_project
    build = root / "build"
    # A previous static-build cache must not take precedence over the SDK.
    stale = root / "stale-boost"
    stale.mkdir()
    for name in ("boost_multiprecision", "boost_config"):
        (stale / f"{name}-config.cmake").write_text('message(FATAL_ERROR "selected cached static Boost")\n')
    _run(
        "cmake",
        "-S",
        str(root),
        "-B",
        str(build),
        "-G",
        GENERATORS[0],
        f"-Dboost_multiprecision_DIR={stale}",
        f"-Dboost_config_DIR={stale}",
    )
    command = ("cmake", "--build", str(build), "--config", "Release", "--target", "native_media_loadable_extension")
    _run(*command)
    runs = next(build.rglob("*.runs"))
    assert runs.read_text().splitlines() == ["prepared"]
    for count, dependency in enumerate(
        (
            "scripts/prepare_dynamic_media_extension.py",
            "vane/_native_runtime_format.py",
            "vane_packaging/media_runtime.py",
            "vane_packaging/_vendor/auditwheel/manylinux-policy.json",
            "runtime/runtime-manifest.json",
        ),
        start=2,
    ):
        os.utime(root / dependency, None)
        _run(*command)
        assert len(runs.read_text().splitlines()) == count
        _run(*command)
        assert len(runs.read_text().splitlines()) == count


@pytest.mark.skipif(sys.platform != "win32", reason="MSVC static-library/import-thunk collision")
@pytest.mark.parametrize("host_scope", ["before", "after", "sibling"])
def test_dynamic_media_reuses_the_hosts_windows_zlib(media_project, host_scope):
    root = media_project
    (root / "base-z.c").write_text("int z_fixture(void) { return 41; }\nint base_only(void) { return 1; }\n")
    (root / "extension.c").write_text(
        "int z_fixture(void);\nint base_only(void);\nint probe(void) { return z_fixture() + base_only(); }\n"
    )
    cmake = root / "CMakeLists.txt"
    host = (
        "add_library(base_zlib STATIC base-z.c)\n"
        "if(NOT TARGET ZLIB::ZLIB)\nadd_library(ZLIB::ZLIB ALIAS base_zlib)\nendif()\n"
        "target_link_libraries(file_extension INTERFACE base_zlib)\n"
    )
    source = cmake.read_text()
    if host_scope == "sibling":
        sibling = root / "host-zlib"
        sibling.mkdir()
        (sibling / "CMakeLists.txt").write_text(
            "add_library(base_zlib STATIC ../base-z.c)\n"
            "add_library(ZLIB::ZLIB INTERFACE IMPORTED)\n"
            "target_link_libraries(ZLIB::ZLIB INTERFACE base_zlib)\n"
            "target_link_libraries(file_extension INTERFACE ZLIB::ZLIB)\n"
        )
        host = "add_subdirectory(host-zlib)\n"
    cmake.write_text(
        source + host
        if host_scope == "after"
        else source.replace(
            "include(external/duckdb/extension/media_common/extension.cmake)",
            host + "include(external/duckdb/extension/media_common/extension.cmake)",
        )
    )
    build = root / "build"
    _run("cmake", "-S", str(root), "-B", str(build), "-G", GENERATORS[0])
    _run("cmake", "--build", str(build), "--config", "Release", "--target", "native_media_loadable_extension")
    library = next(build.rglob("native_media_loadable_extension.dll"))
    assert ctypes.CDLL(str(library)).probe() == 42


def test_dynamic_media_rejects_sdk_without_boost_even_when_host_has_it(media_project):
    root = media_project
    host = root / "host/share/boost_multiprecision"
    host.parent.mkdir(parents=True)
    shutil.move(root / "sdk/share/boost_multiprecision", host)
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(root / "build"),
            "-G",
            GENERATORS[0],
            f"-DCMAKE_PREFIX_PATH={root / 'host'}",
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    assert result.returncode != 0
    assert "Missing Boost.Multiprecision in VANE_MEDIA_RUNTIME_SDK" in result.stdout + result.stderr


@pytest.mark.parametrize("generator", GENERATORS)
def test_loadable_staging_recovers_runtime_files_without_relinking(media_project, generator):
    root = media_project
    with (root / "CMakeLists.txt").open("a") as stream:
        stream.write(
            f'\ninclude("{ROOT.as_posix()}/cmake/duckdb_loader.cmake")\n'
            "set(VANE_LOADABLE_EXTENSION_NAMES native_media)\n"
            "duckdb_stage_loadable_extensions()\n"
        )
    build = root / "build"
    _run("cmake", "-S", str(root), "-B", str(build), "-G", generator)
    command = ("cmake", "--build", str(build), "--config", "Release", "--target", "vane_loadable_extensions")
    _run(*command)
    staged = build / "vane_extensions/.libs/fixture.so"
    artifact = build / "vane_extensions/native_media.duckdb_extension"
    original_artifact = artifact.read_bytes()
    for remove in (lambda: staged.unlink(), lambda: shutil.rmtree(staged.parent)):
        remove()
        _run(*command)
        assert staged.read_text() == "shared runtime fixture\n"
        assert artifact.read_bytes() == original_artifact
    (root / "runtime/.libs/fixture.so").write_text("updated runtime fixture\n")
    _run(*command)
    assert staged.read_text() == "updated runtime fixture\n"
    _run(*command)
    assert next(build.rglob("*.runs")).read_text().splitlines() == ["prepared"]
    (root / "runtime/.libs/fixture.so").unlink()
    missing = subprocess.run(command, capture_output=True, text=True, errors="replace")
    assert missing.returncode != 0
    assert "runtime has no shared libraries" in missing.stdout + missing.stderr
