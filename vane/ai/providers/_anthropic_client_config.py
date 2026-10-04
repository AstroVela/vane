# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Anthropic client configuration snapshots and worker-side SDK initialization."""

from __future__ import annotations

from typing import Any

from vane.ai._client_config import ProviderClientConfigurationError, _base_url, _setting, _text, copy_client_options
from vane.ai._redaction import unwrap_sensitive_options

ANTHROPIC_BASE_URL = "https://api.anthropic.com"


def capture_anthropic_client(
    *, api_key: str | None = None, auth_token: str | None = None, base_url: str | None = None
) -> dict[str, Any]:
    # An explicitly selected credential must not pick up a second auth method.
    key = _text(api_key, "api_key")
    token = _text(auth_token, "auth_token")
    if key is None and token is None:
        key = _setting(None, "ANTHROPIC_API_KEY")
        token = _setting(None, "ANTHROPIC_AUTH_TOKEN")
    if key is not None and token is not None:
        raise ValueError("Configure only one Anthropic api_key or auth_token")
    return copy_client_options(
        {
            "api_key": key,
            "auth_token": token,
            "base_url": _base_url(_setting(base_url, "ANTHROPIC_BASE_URL") or ANTHROPIC_BASE_URL),
        }
    )


def create_anthropic_client(factory: Any, snapshot: dict[str, Any], options: dict[str, Any]) -> Any:
    config = unwrap_sensitive_options(snapshot)
    if not config["api_key"] and not config["auth_token"]:
        raise ProviderClientConfigurationError(
            "Anthropic api_key or auth_token was not configured on the application before creating the provider/descriptor"
        )
    client = factory(
        api_key=config["api_key"] or "",
        auth_token=config["auth_token"] or "",
        base_url=options.get("base_url") or config["base_url"],
        max_retries=0,
        **({"timeout": options["timeout"]} if options.get("timeout") is not None else {}),
    )
    client.api_key = config["api_key"]
    client.auth_token = config["auth_token"]
    client._custom_headers = {}
    return client
