# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Google client configuration snapshots and worker-side SDK initialization."""

from __future__ import annotations

import copy
import os
import re
from typing import Any

from vane.ai._client_config import (
    ProviderClientConfigurationError,
    _base_url,
    _require_key,
    _setting,
    _text,
    copy_client_options,
)
from vane.ai._redaction import unwrap_sensitive_options


def _google_mode(vertexai: bool | None) -> bool:
    if vertexai is not None:
        if type(vertexai) is not bool:
            raise ValueError("Google client vertexai must be a boolean")
        return vertexai
    # Enterprise is the SDK's newer spelling of the same backend selection.
    for name in ("GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_USE_VERTEXAI"):
        raw = os.environ.get(name)
        if raw is not None:
            if raw.lower() not in {"true", "false", "1", "0"}:
                raise ValueError(f"{name} must be true, false, 1, or 0")
            return raw.lower() in {"true", "1"}
    return False


def capture_google_client(
    *,
    api_key: str | None = None,
    vertexai: bool | None = None,
    credentials: Any = None,
    project: str | None = None,
    location: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    mode = _google_mode(vertexai)
    if credentials is not None and api_key is not None:
        raise ValueError("Configure only one Google api_key or credentials")
    if not mode and (credentials is not None or project is not None or location is not None):
        raise ValueError("Google credentials, project, and location require vertexai=True")
    key = None
    if credentials is None:
        key = _text(
            api_key if api_key is not None else os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY")),
            "Google api_key",
        )
    resolved_project = _setting(project, "GOOGLE_CLOUD_PROJECT") if mode else None
    resolved_location = _setting(location, "GOOGLE_CLOUD_LOCATION") if mode else None
    if mode:
        resolved_location = resolved_location or "global"
        if re.fullmatch(r"[a-z0-9-]+", resolved_location) is None:
            raise ValueError("Google location must contain only lowercase letters, digits, and hyphens")
        if resolved_location == "global":
            default_url = "https://aiplatform.googleapis.com/"
        elif resolved_location in {"us", "eu"}:
            default_url = f"https://aiplatform.{resolved_location}.rep.googleapis.com/"
        else:
            default_url = f"https://{resolved_location}-aiplatform.googleapis.com/"
    else:
        default_url = "https://generativelanguage.googleapis.com/"
    return copy_client_options(
        {
            "api_key": key,
            "vertexai": mode,
            "credentials": credentials,
            "project": resolved_project,
            "location": resolved_location,
            "base_url": _base_url(
                _setting(base_url, "GOOGLE_VERTEX_BASE_URL" if mode else "GOOGLE_GEMINI_BASE_URL") or default_url
            ),
        }
    )


def create_google_client(factory: Any, types: Any, snapshot: dict[str, Any]) -> Any:
    config = copy.deepcopy(unwrap_sensitive_options(snapshot))
    if config["credentials"] is None:
        _require_key(config, "Google")
    elif not config["project"]:
        raise ProviderClientConfigurationError("Google Vertex credentials require an explicit application project")
    else:
        from google.auth.credentials import Credentials  # type: ignore[import-untyped, unused-ignore]

        if not isinstance(config["credentials"], Credentials):
            raise ProviderClientConfigurationError(
                "Google credentials must be a google.auth.credentials.Credentials object"
            )
    # In API-key mode an absent application project must not pick up a worker
    # project. The SDK's express-mode branch clears it when location is absent.
    location = config["location"] if config["project"] or config["credentials"] is not None else None
    return factory(
        api_key=config["api_key"],
        vertexai=config["vertexai"],
        credentials=config["credentials"],
        project=config["project"],
        location=location,
        http_options=types.HttpOptions(base_url=config["base_url"], retry_options=types.HttpRetryOptions(attempts=1)),
    )
