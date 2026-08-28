# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
"""Installed provider for the CI-only signed DuckDB extension fixture."""

from __future__ import annotations

import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from vane.extensions import (
    LocalExtensionArtifact,
    LocalExtensionProvider,
    create_dynamic_extension_descriptor,
)

_EXTENSION_NAME = "loadable_extension_demo"
_TRUST_IDENTITY = "vane-ci-test-key"


@lru_cache(maxsize=1)
def create_provider() -> LocalExtensionProvider:
    artifact_value = os.environ.get("VANE_TEST_SIGNED_DYNAMIC_EXTENSION_PATH", "").strip()
    if not artifact_value:
        raise RuntimeError("VANE_TEST_SIGNED_DYNAMIC_EXTENSION_PATH is required")
    artifact_path = Path(artifact_value).expanduser().resolve()
    descriptor = create_dynamic_extension_descriptor(
        artifact_path,
        name=_EXTENSION_NAME,
        trust_identity=_TRUST_IDENTITY,
    )
    descriptor_sha256 = os.environ.get("VANE_TEST_DYNAMIC_EXTENSION_DESCRIPTOR_SHA256", "").strip()
    if descriptor_sha256:
        descriptor = replace(descriptor, sha256=descriptor_sha256)
    return LocalExtensionProvider(
        _TRUST_IDENTITY,
        (LocalExtensionArtifact(descriptor, artifact_path),),
    )
