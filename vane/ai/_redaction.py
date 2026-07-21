# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Credential redaction shared by the AI SQL binding and provider layers.

Vane defends provider credentials in two layers: the SQL binding rejects
inline credential options outright (see ``vane.ai._sql``), while the Python
descriptor path seals credential values in :class:`Secret` so that repr, str,
logs, exception messages, and assertion diffs never show the plaintext. Both
layers share the sensitive-key table and matching logic defined here.

This module is private; :class:`Secret` is intentionally not exported from
the public ``vane.ai`` namespace.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    "REDACTED_PLACEHOLDER",
    "SENSITIVE_OPTION_KEYS",
    "Secret",
    "is_sensitive_option_key",
    "normalized_option_key",
    "unwrap_sensitive_options",
    "wrap_sensitive_options",
]

REDACTED_PLACEHOLDER = "**********"

# Normalized credential key names. A key matches when its normalized form is
# equal to, or ends with, one of these entries (see is_sensitive_option_key).
SENSITIVE_OPTION_KEYS = frozenset(
    {
        "accesskey",
        "accesskeyid",
        "accesstoken",
        "apikey",
        "apikeyid",
        "apitoken",
        "authorization",
        "authtoken",
        "bearertoken",
        "clientsecret",
        "clientsecretvalue",
        "credential",
        "credentials",
        "password",
        "passwd",
        "privatekey",
        "secret",
        "secretkey",
        "token",
    }
)


def normalized_option_key(key: Any) -> str:
    """Normalize an option key for sensitive-key matching (casefold, strip non-alphanumerics)."""
    return re.sub(r"[^a-z0-9]", "", str(key).casefold())


def is_sensitive_option_key(key: Any) -> bool:
    """Return True when ``key`` names a credential according to the shared key table."""
    normalized = normalized_option_key(key)
    return any(normalized == sensitive or normalized.endswith(sensitive) for sensitive in SENSITIVE_OPTION_KEYS)


class Secret:
    """Opaque wrapper that renders as a fixed placeholder wherever Python stringifies it.

    The wrapped value is only retrievable through an explicit :meth:`reveal`
    call. Equality compares real values against another ``Secret`` only;
    comparison with a bare ``str`` (or any non-``Secret``) is always ``False``.
    Pickling preserves the real value so workers can use it; plaintext inside
    pickle payloads is an accepted inherent property of that requirement.
    """

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        if isinstance(value, Secret):
            value = value.reveal()
        self._value = value

    def reveal(self) -> Any:
        """Return the wrapped plaintext value."""
        return self._value

    def __repr__(self) -> str:
        return REDACTED_PLACEHOLDER

    def __str__(self) -> str:
        return REDACTED_PLACEHOLDER

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Secret):
            return bool(self._value == other._value)
        return False

    def __hash__(self) -> int:
        return hash(self._value)

    def __reduce__(self) -> tuple[type[Secret], tuple[Any]]:
        return (Secret, (self._value,))


def _seal(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Secret):
        return value
    return Secret(value)


def _wrap_value(value: Any) -> Any:
    if isinstance(value, Secret):
        return value
    if isinstance(value, Mapping):
        return {key: _seal(item) if is_sensitive_option_key(key) else _wrap_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_wrap_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_wrap_value(item) for item in value)
    return value


def wrap_sensitive_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``options`` with every sensitive-keyed value sealed in a :class:`Secret`.

    Matching applies at any nesting depth, including mappings inside lists and
    tuples. A container value under a sensitive key is sealed whole. Values
    that are already ``Secret`` are kept as-is (wrapping is idempotent), and
    ``None`` values stay ``None``.
    """
    return {key: _seal(value) if is_sensitive_option_key(key) else _wrap_value(value) for key, value in options.items()}


def _unwrap_value(value: Any) -> Any:
    if isinstance(value, Secret):
        return _unwrap_value(value.reveal())
    if isinstance(value, Mapping):
        return {key: _unwrap_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unwrap_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_unwrap_value(item) for item in value)
    return value


def unwrap_sensitive_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return a plain copy of ``options`` with every :class:`Secret` replaced by its real value."""
    return {key: _unwrap_value(value) for key, value in options.items()}
