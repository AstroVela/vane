# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Pure expression construction for Vane FILE values."""

from __future__ import annotations

import vane
from vane._expressions import as_expression


def file(
    url: str | vane.Expression,
    content_type: str | vane.Expression | None = None,
    position: int | vane.Expression | None = None,
    size: int | vane.Expression | None = None,
    checksum: str | vane.Expression | None = None,
) -> vane.Expression:
    """Construct a FILE expression without accessing the referenced resource."""
    return vane.FunctionExpression(
        "file",
        as_expression(url),
        as_expression(content_type),
        as_expression(position),
        as_expression(size),
        as_expression(checksum),
    )


__all__ = ["file"]
