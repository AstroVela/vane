# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze Git provenance when exporting the standalone media source SDK."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

VERSION_FILE = "runtime-version.json"


def runtime_format():
    root = Path(__file__).resolve().parents[1]
    path = root / "vane/_native_runtime_format.py"
    if not path.is_file():
        path = root / "_native_runtime_format.py"
    spec = importlib.util.spec_from_file_location("_native_runtime_format", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity_version(identity: dict) -> str:
    """Reuse the version encoder used by Iceberg and all Vane provider wheels."""
    from vane_packaging.extension_wheel import (
        _extension_distribution_version_from_digest,
    )

    fmt = runtime_format()
    provenance = {key: identity[key] for key in ("git_commit", "git_dirty", "vane_version")}
    fmt.git_commit(provenance["git_commit"])
    fmt.version(provenance["vane_version"])
    if type(provenance["git_dirty"]) is not bool:
        raise ValueError("invalid media Git working-tree state")
    return _extension_distribution_version_from_digest(
        provenance["vane_version"],
        hashlib.sha256(fmt.canonical_json(provenance)).hexdigest(),
    )


def source_version(project: Path) -> dict:
    """Use the exported identity without Git, or derive it from the Vane checkout."""
    fmt = runtime_format()
    if (project / VERSION_FILE).exists():
        document = fmt.parse_json(fmt.read_file(project, VERSION_FILE, fmt.MAX_MANIFEST_BYTES))
        if set(document) != {
            "git_commit",
            "git_dirty",
            "vane_version",
            "version",
        } or document["version"] != identity_version(document):
            raise ValueError("invalid exported media Git identity")
        return document
    root = project.resolve().parents[1]
    try:
        git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root, text=True).strip()
        if Path(git_root).resolve() != root:
            raise ValueError("media version requires the Vane checkout or an exported source SDK")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        changes = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root)
    except subprocess.CalledProcessError as error:
        raise ValueError("media version requires Git metadata or an exported source SDK") from error
    # Resolve the same Vane source version as the Iceberg release preflight.
    # The exported document freezes it, so wheel rebuilds need no Git or SCM tool.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("SETUPTOOLS_SCM_PRETEND_VERSION")
    }
    vane_version = subprocess.run(
        [sys.executable, "-m", "setuptools_scm"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    identity = {
        "git_commit": commit,
        "git_dirty": bool(changes),
        "vane_version": vane_version,
    }
    return {**identity, "version": identity_version(identity)}
