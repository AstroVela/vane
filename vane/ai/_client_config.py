# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Common validation and redaction for application-side client snapshots.

Snapshots carry actual credentials in serialized descriptors. Secret wrappers
protect display surfaces, not pickle bytes (scoped secret transport is #243).
"""

from __future__ import annotations

import copy
import os
from typing import Any

from vane.ai._redaction import wrap_sensitive_options
from vane.ai.options import _validate_base_url_option
from vane.ai.provider import _SafeProviderError


class ProviderClientConfigurationError(_SafeProviderError):
    """A client snapshot cannot authenticate without using worker state."""


def _text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"Provider client {name} must be a non-empty string without surrounding whitespace")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"Provider client {name} must not contain control characters")
    return value


def _setting(value: str | None, variable: str) -> str | None:
    return _text(value if value is not None else os.environ.get(variable), variable)


def _base_url(value: str) -> str:
    _validate_base_url_option({"base_url": value}, api="Prompt")
    return value


def copy_client_options(options: dict[str, Any]) -> dict[str, Any]:
    """Give each descriptor its own redacted snapshot, including credentials."""
    return wrap_sensitive_options(copy.deepcopy(options), extra_keys=frozenset({"organization", "project"}))


def _require_key(options: dict[str, Any], provider: str) -> None:
    if not options.get("api_key"):
        raise ProviderClientConfigurationError(
            f"{provider} API key was not configured on the application before creating the provider/descriptor"
        )
