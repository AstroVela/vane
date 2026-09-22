# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vane.execution.resource_graph_metadata import ResourceGraphMetadataProvider


@dataclass(frozen=True)
class RayResourceGraphAdapter(ResourceGraphMetadataProvider):
    """Export the shared schema and register Ray UDF identities on the plan."""

    plan: Any

    def collect_resource_graph_metadata(self, conn: Any = None) -> dict[str, Any]:
        return self.plan.collect_resource_graph_metadata(conn=conn, annotate_udfs=True)
