# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Local Iceberg and MCP protocol integration; no cloud, model, or account access."""

from __future__ import annotations

import json
import sys
from contextlib import closing
from dataclasses import replace

import pytest

from vane.knowledge.config import Config, Target
from vane.knowledge.iceberg import IcebergSource, Snapshot
from vane.knowledge.service import sync_once
from vane.knowledge.state import State
from vane.knowledge.targets import deliver_openwiki


def configuration(tmp_path):
    return Config(
        name="docs",
        catalog="knowledge-test",
        table=("db", "docs"),
        branch="main",
        id_column="id",
        content_column="body",
        title_column=None,
        uri_column=None,
        state_dir=tmp_path / "state",
        targets=(Target("brain", "gbrain", ("gbrain",), str(tmp_path), "default"),),
    )


def test_real_iceberg_snapshot_to_acknowledged_documents(tmp_path, monkeypatch):
    pytest.importorskip("pyiceberg", minversion="0.12.0")
    pytest.importorskip("sqlalchemy", reason="Install the knowledge-test dependency group")
    import pyarrow as pa
    from pyiceberg.catalog import load_catalog
    from pyiceberg.utils.config import Config as IcebergConfig

    monkeypatch.setenv("PYICEBERG_CATALOG__KNOWLEDGE_TEST__TYPE", "sql")
    monkeypatch.setenv("PYICEBERG_CATALOG__KNOWLEDGE_TEST__URI", f"sqlite:///{tmp_path / 'catalog.sqlite'}")
    monkeypatch.setenv("PYICEBERG_CATALOG__KNOWLEDGE_TEST__WAREHOUSE", (tmp_path / "warehouse").as_uri())
    # PyIceberg captures environment configuration at import and normalizes
    # underscores in catalog environment keys to hyphens. Refresh its real
    # config for this fixture, including when another test imported it first.
    monkeypatch.setattr("pyiceberg.catalog._ENV_CONFIG", IcebergConfig())
    catalog = load_catalog("knowledge-test")
    catalog.create_namespace("db")
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("body", pa.string())])
    table = catalog.create_table(("db", "docs"), schema=schema)
    config = configuration(tmp_path)
    calls = []

    def send(target, arguments, _timeout, content=""):
        calls.append((arguments[0], arguments[1], content))
        return json.dumps(
            {
                "state": "committed",
                "request_id": arguments[arguments.index("--request-id") + 1],
                "outcome": {
                    "slug": arguments[1],
                    "source_id": target.destination,
                    "status": "created_or_updated" if arguments[0] == "put" else "soft_deleted",
                },
            }
        ).encode()

    monkeypatch.setattr("vane.knowledge.targets.run_command", send)
    with closing(State(config.state_dir, create=True)) as state, state.exclusive():
        assert not sync_once(config, state)
        table.append(pa.Table.from_pylist([{"id": 1, "body": "First"}, {"id": 2, "body": "Second"}], schema=schema))
        assert sync_once(config, state)
        first_slug = calls[0][1]
        assert [c[0] for c in calls] == ["put", "put"]
        assert not sync_once(config, state)
        table.overwrite(pa.Table.from_pylist([{"id": 1, "body": "Updated"}, {"id": 3, "body": "Third"}], schema=schema))
        assert sync_once(config, state)
        assert [c[0] for c in calls] == ["put", "put", "put", "put", "delete"]
        assert calls[2][1] == first_slug
        assert "Updated" in calls[2][2]
        assert state.checkpoint().snapshot_id == table.current_snapshot().snapshot_id
        table.delete("id == 1")
        assert sync_once(config, state)
        assert calls[-1][:2] == ("delete", first_slug)
        table.append(pa.Table.from_pylist([{"id": 1, "body": "Reinserted"}], schema=schema))
        assert sync_once(config, state)
        assert calls[-1][:2] == ("put", first_slug)


def test_openwiki_subprocess_reads_real_mcp_transport(tmp_path):
    pytest.importorskip("mcp.server.mcpserver", reason="Install the knowledge-test dependency group")
    config = configuration(tmp_path)
    consumer = tmp_path / "openwiki_consumer.py"
    # Exercise the installed CLI and the actual SDK's JSON-RPC framing. The
    # stand-in replaces only OpenWiki's paid LLM synthesis, not the MCP server.
    consumer.write_text("""
import json, subprocess, sys
child = subprocess.Popen(
    [sys.executable, "-m", "vane.knowledge", "serve", "--state", sys.argv[1]],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
)
counter = 0
def request(method, params):
    global counter
    counter += 1
    child.stdin.write(json.dumps({"jsonrpc":"2.0", "id":counter, "method":method, "params":params}) + "\\n")
    child.stdin.flush()
    while True:
        line = child.stdout.readline()
        if not line: raise RuntimeError("MCP server exited")
        result = json.loads(line)
        if result.get("id") == counter:
            if "error" in result: raise RuntimeError(result["error"])
            return result["result"]
try:
    request("initialize", {"protocolVersion":"2025-11-25", "capabilities":{}, "clientInfo":{"name":"contract-test", "version":"1"}})
    child.stdin.write(json.dumps({"jsonrpc":"2.0", "method":"notifications/initialized"}) + "\\n")
    child.stdin.flush()
    tools = request("tools/list", {})["tools"]
    assert {t["name"] for t in tools} == {"list_changes", "read_changes", "acknowledge_changes"}
    assert all(t["annotations"]["readOnlyHint"] == (t["name"] != "acknowledge_changes") for t in tools)
    listing = request("tools/call", {"name":"list_changes", "arguments":{}})["structuredContent"]
    batch = listing["batch"]
    page = request("tools/call", {"name":"read_changes", "arguments":{"batch_id":batch["id"]}})["structuredContent"]
    assert len(page["events"]) == listing["event_count"]
    assert "receipt" not in listing
    assert request("tools/call", {"name":"acknowledge_changes", "arguments":{"batch_id":batch["id"], "receipt":page["receipt"]}})["structuredContent"]["acknowledged"]
finally:
    child.stdin.close()
    child.wait(timeout=10)
    assert child.returncode == 0
""")
    target = replace(
        config.targets[0],
        kind="openwiki",
        destination="custom-mcp-docs",
        command=(sys.executable, str(consumer), str(config.state_dir)),
    )
    config = replace(config, targets=(target,))
    source = object.__new__(IcebergSource)
    source.config = config
    snapshot = Snapshot("test-uuid", 1)
    docs = (source.document(snapshot, {"id": n, "body": "x" * 50000}) for n in range(8))
    with closing(State(config.state_dir, create=True)) as state, state.exclusive():
        state.stage(snapshot, docs, (target.name,))
        deliver_openwiki(state, target, 60)
        assert state.delivery(target.name)["done"]
        state.finish()
        assert state.checkpoint().snapshot_id == 1
