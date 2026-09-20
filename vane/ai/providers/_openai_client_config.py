# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI client configuration snapshots and worker-side SDK initialization."""

from __future__ import annotations

from typing import Any

from vane.ai._client_config import _base_url, _require_key, _setting, copy_client_options
from vane.ai._redaction import unwrap_sensitive_options

OPENAI_BASE_URL = "https://api.openai.com/v1"


def capture_openai_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    return copy_client_options(
        {
            "api_key": _setting(api_key, "OPENAI_API_KEY"),
            "base_url": _base_url(_setting(base_url, "OPENAI_BASE_URL") or OPENAI_BASE_URL),
            "organization": _setting(organization, "OPENAI_ORG_ID"),
            "project": _setting(project, "OPENAI_PROJECT_ID"),
        }
    )


def create_openai_client(factory: Any, snapshot: dict[str, Any], options: dict[str, Any]) -> Any:
    config = unwrap_sensitive_options(snapshot)
    _require_key(config, "OpenAI")
    client = factory(
        api_key=config["api_key"],
        base_url=options.get("base_url") or config["base_url"],
        organization=config["organization"] or "",
        project=config["project"] or "",
        max_retries=0,
        **({"timeout": options["timeout"]} if options.get("timeout") is not None else {}),
    )
    # SDK None defaults read ambient env. Restore explicit absence before any
    # request, and discard SDK-imported OPENAI_CUSTOM_HEADERS (including auth).
    client.organization = config["organization"]
    client.project = config["project"]
    client._custom_headers = {}
    client.admin_api_key = None
    return client
