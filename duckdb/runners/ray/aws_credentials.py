# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any


def resolve_aws_credentials_from_environment() -> dict[str, Any]:
    """Resolve one standard AWS credential chain in this isolated process."""
    from botocore.session import Session  # type: ignore[import-not-found]

    session = Session()
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("AWS credential resolver found no credentials")
    frozen = credentials.get_frozen_credentials()
    if not frozen.access_key or not frozen.secret_key:
        raise RuntimeError("AWS credential resolver returned incomplete credentials")

    resolved = {
        "AWS_ACCESS_KEY_ID": frozen.access_key,
        "AWS_SECRET_ACCESS_KEY": frozen.secret_key,
    }
    if frozen.token:
        resolved["AWS_SESSION_TOKEN"] = frozen.token
    region = os.environ.get("AWS_REGION") or session.get_config_variable("region")
    if region:
        resolved["AWS_REGION"] = str(region)

    result: dict[str, Any] = {"credentials": resolved}
    expiration = getattr(credentials, "_expiry_time", None)
    if isinstance(expiration, datetime):
        result["expiration_epoch_s"] = expiration.timestamp()
    return result


def _write_result(result: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")))


def main() -> None:
    _write_result(resolve_aws_credentials_from_environment())


if __name__ == "__main__":
    main()
