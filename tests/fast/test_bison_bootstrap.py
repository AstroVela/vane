# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Offline download, integrity and cache contracts for the manylinux build tool."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


@pytest.fixture
def bootstrap(tmp_path):
    for tool in ("bash", "tar", "make", "sha256sum"):
        if shutil.which(tool) is None:
            pytest.skip(f"Bison bootstrap requires {tool}")
    source = tmp_path / "bison-3.8.2"
    source.mkdir()
    configure = source / "configure"
    configure.write_text('#!/bin/sh\nprintf "%s\\n" "${1#--prefix=}" > prefix\n')
    configure.chmod(0o755)
    binary = source / "bison"
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "bison (GNU Bison) 3.8.2"\n')
    binary.chmod(0o755)
    (source / "Makefile").write_text(
        'all:\n\t@:\ninstall:\n\tmkdir -p "$$(cat prefix)/bin"\n\tcp bison "$$(cat prefix)/bin/bison"\n'
    )
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(source, arcname=source.name)
    # Keep the production script's fixed-digest check, using an offline SDK fixture.
    script = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_manylinux_bison.sh"
    content = script.read_text().replace(
        "06c9e13bdf7eb24d4ceb6b59205a4f67c2c7e7213119644430fe82fbd14a0abb",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    runner = tmp_path / "bootstrap.sh"
    runner.write_text(content)
    tools = tmp_path / "tools"
    tools.mkdir()
    curl = tools / "curl"
    curl.write_text(
        f"#!{sys.executable}\n"
        "import json, os, shutil, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['BISON_TEST_LOG'], 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "mode = os.environ['BISON_TEST_MODE']\n"
        "primary = sys.argv[-1].startswith('https://ftp.gnu.org/')\n"
        "target = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "if mode == 'unavailable' or (mode == 'network' and primary):\n"
        "    target.write_bytes(b'partial download')\n"
        "    sys.exit(7)\n"
        "if mode == 'corrupt' or (mode == 'checksum' and primary):\n"
        "    target.write_bytes(b'invalid archive')\n"
        "else:\n"
        "    shutil.copyfile(os.environ['BISON_TEST_ARCHIVE'], target)\n"
    )
    curl.chmod(0o755)
    log = tmp_path / "downloads.jsonl"
    cache = tmp_path / "build" / "bison-3.8.2.tar.gz"
    prefix = tmp_path / "install"

    def run(mode):
        env = dict(os.environ)
        env.update(
            PATH=str(tools) + os.pathsep + env.get("PATH", ""),
            VANE_BISON_INSTALL_PREFIX=str(prefix),
            VANE_BUILD_TOOLS_DIR=str(cache.parent),
            VANE_BUILD_JOBS="1",
            BISON_TEST_MODE=mode,
            BISON_TEST_LOG=str(log),
            BISON_TEST_ARCHIVE=str(archive),
        )
        result = subprocess.run(["bash", str(runner)], env=env, capture_output=True, text=True, timeout=30)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run, cache, archive, prefix


@pytest.mark.parametrize("mode", ["network", "checksum"])
def test_bison_bootstrap_uses_verified_mirror_after_primary_failure(bootstrap, mode):
    run, cache, archive, prefix = bootstrap
    result, calls = run(mode)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 2
    assert calls[0][-1].startswith("https://ftp.gnu.org/")
    assert calls[1][-1].startswith("https://mirrors.kernel.org/")
    assert all("--retry" in call and "--connect-timeout" in call and "--max-time" in call for call in calls)
    assert all("--retry-all-errors" not in call for call in calls)  # manylinux curl 7.61
    assert cache.read_bytes() == archive.read_bytes()
    assert (prefix / "bin" / "bison").is_file()
    assert not cache.with_suffix(".gz.part").exists()


@pytest.mark.parametrize("mode", ["unavailable", "corrupt"])
def test_bison_bootstrap_does_not_install_unverified_downloads(bootstrap, mode):
    run, cache, _, prefix = bootstrap
    result, calls = run(mode)
    assert result.returncode != 0
    assert "Failed to download verified GNU Bison" in result.stderr
    assert len(calls) == 2
    assert not cache.exists()
    assert not cache.with_suffix(".gz.part").exists()
    assert not prefix.exists()


@pytest.mark.parametrize("valid", [False, True])
def test_bison_bootstrap_validates_archive_cache_and_reuses_installed_tool(bootstrap, valid):
    run, cache, archive, _ = bootstrap
    cache.parent.mkdir()
    cache.write_bytes(archive.read_bytes() if valid else b"broken cached archive")
    result, calls = run("primary")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == (0 if valid else 1)
    assert cache.read_bytes() == archive.read_bytes()
    reused, repeated_calls = run("unavailable")
    assert reused.returncode == 0, reused.stdout + reused.stderr
    assert "Using cached GNU Bison" in reused.stdout
    assert repeated_calls == calls
