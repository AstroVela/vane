# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Pinned auditwheel manylinux policy data used by release tooling."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

AUDITWHEEL_MANYLINUX_POLICY_VERSION = "6.8.1"
AUDITWHEEL_MANYLINUX_POLICY_COMMIT = "94e0693e0fcb444c7fe50f09a8a635e791be6174"
AUDITWHEEL_MANYLINUX_POLICY_SHA256 = "104863eb197685edf6407a51ccde6cbd906be736efb959a991a60d102f1ccf96"
AUDITWHEEL_MANYLINUX_POLICY_PATH = Path(__file__).resolve().parent / "_vendor" / "auditwheel" / "manylinux-policy.json"

_MAX_POLICY_BYTES = 256 * 1024
_POLICY_NAME_RE = re.compile(r"^manylinux_([0-9]+)_([0-9]+)$")
_SUPPORTED_ARCHITECTURES = frozenset({"x86_64", "aarch64"})
_EXPECTED_BASELINES = {
    "x86_64": (
        (2, 5),
        (2, 12),
        (2, 17),
        (2, 24),
        (2, 26),
        (2, 27),
        (2, 28),
        (2, 31),
        (2, 34),
        (2, 35),
        (2, 36),
        (2, 37),
        (2, 38),
        (2, 39),
        (2, 40),
        (2, 41),
    ),
    "aarch64": (
        (2, 17),
        (2, 24),
        (2, 26),
        (2, 27),
        (2, 28),
        (2, 31),
        (2, 34),
        (2, 35),
        (2, 36),
        (2, 37),
        (2, 38),
        (2, 39),
        (2, 40),
        (2, 41),
    ),
}


@dataclass(frozen=True)
class ManylinuxPolicy:
    """One exact architecture/baseline policy from the pinned snapshot."""

    name: str
    minimum_version: tuple[int, int]
    architecture: str
    external_libraries: frozenset[str]
    versioned_symbols: frozenset[str]
    undefined_symbol_blacklist: tuple[tuple[str, frozenset[str]], ...]


def manylinux_policy(minimum_version: tuple[int, int], architecture: str) -> ManylinuxPolicy:
    """Return the exact pinned policy or reject an unknown tag baseline."""
    try:
        return _manylinux_policy_index()[(architecture, minimum_version)]
    except KeyError:
        raise ValueError(
            f"manylinux_{minimum_version[0]}_{minimum_version[1]}_{architecture} is not present in "
            f"the pinned auditwheel {AUDITWHEEL_MANYLINUX_POLICY_VERSION} policy"
        ) from None


def manylinux_policy_combinations() -> tuple[tuple[str, tuple[int, int]], ...]:
    """Return every supported architecture/baseline pair in deterministic order."""
    return tuple(sorted(_manylinux_policy_index(), key=lambda item: (item[0], item[1])))


@cache
def _manylinux_policy_index() -> Mapping[tuple[str, tuple[int, int]], ManylinuxPolicy]:
    raw_policy = _read_policy_snapshot()
    try:
        payload = json.loads(raw_policy)
    except (UnicodeDecodeError, json.JSONDecodeError) as exception:
        raise RuntimeError("the bundled auditwheel manylinux policy is not valid JSON") from exception
    if not isinstance(payload, list):
        raise RuntimeError("the bundled auditwheel manylinux policy root must be a list")

    policies: dict[tuple[str, tuple[int, int]], ManylinuxPolicy] = {}
    for raw_entry in payload:
        if not isinstance(raw_entry, dict):
            raise RuntimeError("the bundled auditwheel manylinux policy contains a non-object entry")
        name = raw_entry.get("name")
        if name == "linux":
            continue
        if not isinstance(name, str) or (match := _POLICY_NAME_RE.fullmatch(name)) is None:
            raise RuntimeError("the bundled auditwheel manylinux policy contains an invalid policy name")
        minimum_version = (int(match[1]), int(match[2]))
        symbol_versions = raw_entry.get("symbol_versions")
        if not isinstance(symbol_versions, dict):
            raise RuntimeError(f"the bundled auditwheel policy {name!r} has invalid symbol versions")
        external_libraries = frozenset(
            _string_list(raw_entry.get("lib_whitelist"), description=f"{name} library allowlist")
        )
        raw_blacklist = raw_entry.get("blacklist")
        if not isinstance(raw_blacklist, dict):
            raise RuntimeError(f"the bundled auditwheel policy {name!r} has an invalid symbol blacklist")
        if any(not isinstance(library, str) or not library.isascii() or not library for library in raw_blacklist):
            raise RuntimeError(f"the bundled auditwheel policy {name!r} has an invalid blacklist library")
        blacklist = tuple(
            sorted(
                (
                    library,
                    frozenset(_string_list(symbols, description=f"{name} {library} symbol blacklist")),
                )
                for library, symbols in raw_blacklist.items()
            )
        )

        for architecture in _SUPPORTED_ARCHITECTURES.intersection(symbol_versions):
            raw_namespaces = symbol_versions[architecture]
            if not isinstance(raw_namespaces, dict):
                raise RuntimeError(f"the bundled auditwheel policy {name!r} has invalid {architecture} symbols")
            exact_symbols: set[str] = set()
            for namespace, versions in raw_namespaces.items():
                if not isinstance(namespace, str) or not namespace.isascii() or not namespace:
                    raise RuntimeError(f"the bundled auditwheel policy {name!r} has an invalid symbol namespace")
                for version in _string_list(
                    versions,
                    description=f"{name} {architecture} {namespace} versions",
                ):
                    exact_symbol = f"{namespace}_{version}"
                    if exact_symbol in exact_symbols:
                        raise RuntimeError(f"the bundled auditwheel policy {name!r} has duplicate symbols")
                    exact_symbols.add(exact_symbol)
            key = (architecture, minimum_version)
            if key in policies:
                raise RuntimeError(f"the bundled auditwheel manylinux policy repeats {name}_{architecture}")
            policies[key] = ManylinuxPolicy(
                name=name,
                minimum_version=minimum_version,
                architecture=architecture,
                external_libraries=external_libraries,
                versioned_symbols=frozenset(exact_symbols),
                undefined_symbol_blacklist=blacklist,
            )

    actual_baselines = {
        architecture: tuple(
            sorted(version for candidate_architecture, version in policies if candidate_architecture == architecture)
        )
        for architecture in _SUPPORTED_ARCHITECTURES
    }
    if actual_baselines != _EXPECTED_BASELINES:
        raise RuntimeError(
            "the bundled auditwheel manylinux policy does not contain the expected x86-64/AArch64 baselines"
        )
    return MappingProxyType(policies)


def _read_policy_snapshot() -> bytes:
    try:
        with AUDITWHEEL_MANYLINUX_POLICY_PATH.open("rb") as policy_file:
            contents = policy_file.read(_MAX_POLICY_BYTES + 1)
    except OSError as exception:
        raise RuntimeError("could not read the bundled auditwheel manylinux policy") from exception
    if len(contents) > _MAX_POLICY_BYTES:
        raise RuntimeError("the bundled auditwheel manylinux policy exceeds its 256 KiB limit")
    if hashlib.sha256(contents).hexdigest() != AUDITWHEEL_MANYLINUX_POLICY_SHA256:
        raise RuntimeError("the bundled auditwheel manylinux policy does not match its pinned SHA256")
    return contents


def _string_list(value: Any, *, description: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.isascii() or not item for item in value
    ):
        raise RuntimeError(f"the bundled auditwheel policy contains an invalid {description}")
    if len(set(value)) != len(value):
        raise RuntimeError(f"the bundled auditwheel policy contains duplicate values in {description}")
    return tuple(value)
