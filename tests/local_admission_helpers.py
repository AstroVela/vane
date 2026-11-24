# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0


def linear_metadata(nodes):
    """A real metadata contract for fake plans used by lifecycle unit tests."""
    nodes = list(nodes)
    if not nodes:
        nodes = [{"node_id": "native", "payload": None}]
    return {
        "query_id": "test-plan",
        "nodes": [
            {
                "node_id": str(node["node_id"]),
                "node_name": "TestUDF" if node["payload"] is not None else "Scan",
                "input_node_ids": [] if index == 0 else [str(nodes[index - 1]["node_id"])],
                "is_sink": index == len(nodes) - 1,
                "is_materialization_barrier": False,
                "materialized_input_node_ids": [],
                "num_partitions": 1,
                "udf_payload": node["payload"],
            }
            for index, node in enumerate(nodes)
        ],
        "terminal_node_ids": [str(nodes[-1]["node_id"])],
        "udf_node_ids": {str(node["node_id"]): str(node["node_id"]) for node in nodes if node["payload"] is not None},
    }


class PreparedOwners:
    def __init__(self, owners):
        self.owners = owners
        self.pool = next(owner.pool for owner in owners if hasattr(owner, "pool"))

    def release(self):
        self.shutdown()

    def shutdown(self, *, kill=False):
        for owner in reversed(self.owners):
            owner.shutdown(kill=kill)
