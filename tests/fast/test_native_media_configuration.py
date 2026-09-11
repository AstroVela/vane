# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual media CMake function with small stand-ins for DuckDB targets."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def media_project(tmp_path):
    if not shutil.which("cc") or not shutil.which("ninja"):
        pytest.skip("CMake linkage test requires a C compiler and Ninja")
    root = tmp_path / "project"
    cmake_path = root / "external/duckdb/extension/media_common/extension.cmake"
    cmake_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / cmake_path.relative_to(root), cmake_path)
    # Only relocation is replaced: the real CMake function must schedule it.
    for relative in (
        "scripts/prepare_dynamic_media_extension.py",
        "vane/_native_runtime_format.py",
        "vane_packaging/media_runtime.py",
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
    subprocess.run(["cc", "-shared", "-fPIC", str(root / "library.c"), "-o", str(sdk / "lib/libsoxr.so")], check=True)
    for name in (
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
        shutil.copyfile(sdk / "lib/libsoxr.so", sdk / f"lib/lib{name}.so")
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
project(media_probe LANGUAGES C)
set(EXTENSION_STATIC_BUILD ON)
set(VANE_MEDIA_RUNTIME_SDK "${CMAKE_CURRENT_SOURCE_DIR}/sdk")
set(VANE_MEDIA_RUNTIME_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/runtime")
add_library(file_extension INTERFACE)
function(build_static_extension name)
  add_library(${name}_extension STATIC extension.c)
endfunction()
function(build_loadable_extension name)
  add_library(${name}_loadable_extension SHARED extension.c)
endfunction()
include(external/duckdb/extension/media_common/extension.cmake)
vane_build_native_media_extension()
""")
    return root


def _run(*command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


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
        "Ninja",
        f"-Dboost_multiprecision_DIR={stale}",
        f"-Dboost_config_DIR={stale}",
    )
    command = ("cmake", "--build", str(build), "--target", "native_media_loadable_extension")
    _run(*command)
    runs = build / "libnative_media_loadable_extension.runs"
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


def test_dynamic_media_rejects_sdk_without_boost_even_when_host_has_it(media_project):
    root = media_project
    host = root / "host/share/boost_multiprecision"
    host.parent.mkdir(parents=True)
    shutil.move(root / "sdk/share/boost_multiprecision", host)
    result = subprocess.run(
        ["cmake", "-S", str(root), "-B", str(root / "build"), "-G", "Ninja", f"-DCMAKE_PREFIX_PATH={root / 'host'}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Missing Boost.Multiprecision in VANE_MEDIA_RUNTIME_SDK" in result.stdout + result.stderr


@pytest.mark.parametrize("generator", ["Ninja", "Unix Makefiles"])
def test_loadable_staging_recovers_runtime_files_without_relinking(media_project, generator):
    root = media_project
    with (root / "CMakeLists.txt").open("a") as stream:
        stream.write(
            f'\ninclude("{ROOT}/cmake/duckdb_loader.cmake")\n'
            "set(VANE_LOADABLE_EXTENSION_NAMES native_media)\n"
            "duckdb_stage_loadable_extensions()\n"
        )
    build = root / "build"
    _run("cmake", "-S", str(root), "-B", str(build), "-G", generator)
    command = ("cmake", "--build", str(build), "--target", "vane_loadable_extensions")
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
    assert (build / "libnative_media_loadable_extension.runs").read_text().splitlines() == ["prepared"]
    (root / "runtime/.libs/fixture.so").unlink()
    missing = subprocess.run(command, capture_output=True, text=True)
    assert missing.returncode != 0
    assert "runtime has no shared libraries" in missing.stdout + missing.stderr
