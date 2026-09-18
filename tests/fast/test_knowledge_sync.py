# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vane.knowledge.config import Config, Target, encode
from vane.knowledge.iceberg import IcebergSource, Snapshot
from vane.knowledge.mcp import openwiki_source
from vane.knowledge.service import sync_once
from vane.knowledge.state import State
from vane.knowledge.targets import DeliveryError, deliver, deliver_openwiki, markdown, run_command


@pytest.fixture
def config(tmp_path):
    return Config(
        name="documents",
        catalog="lake",
        table=("knowledge", "documents"),
        branch="main",
        id_column="id",
        content_column="body",
        title_column="title",
        uri_column=None,
        state_dir=tmp_path / "state",
        targets=(Target("brain", "gbrain", ("gbrain",), str(tmp_path), "default"),),
    )


@pytest.fixture
def state(config):
    with closing(State(config.state_dir, create=True)) as state:
        yield state


def doc(config, key=1, body="Text", snapshot=10):
    source = object.__new__(IcebergSource)
    source.config = config
    return source.document(Snapshot("table-uuid", snapshot), {"id": key, "body": body, "title": f"Document {key}"})


def stage(state, config, *documents, snapshot=10):
    state.stage(Snapshot("table-uuid", snapshot), iter(documents), tuple(t.name for t in config.targets))


def receipt(target, arguments, _timeout, content=""):
    verb, slug = arguments[:2]
    return encode(
        {
            "state": "committed",
            "request_id": arguments[arguments.index("--request-id") + 1],
            "outcome": {
                "slug": slug,
                "source_id": target.destination,
                "status": "created_or_updated" if verb == "put" else "soft_deleted",
            },
        }
    ).encode()


def accept(state, config):
    for target in config.targets:
        state.acknowledge(target.name, 0, done=True)
    state.finish()


def test_atomic_diff_tracks_updates_deletes_and_unchanged_rows(config, state):
    stage(state, config, doc(config, 1), doc(config, 2))
    assert state.checkpoint() is None
    accept(state, config)
    stage(state, config, doc(config, 1, "Revised", 20), doc(config, 3, snapshot=20), snapshot=20)
    events = list(state.events())
    assert [(e["kind"], e["document"]["id"]) for e in events] == [("upsert", "1"), ("upsert", "3"), ("delete", "2")]
    assert events[-1]["document"]["content"] == ""
    assert state.checkpoint().snapshot_id == 10
    accept(state, config)
    stage(state, config, doc(config, 1, "Revised", 30), doc(config, 3, snapshot=30), snapshot=30)
    assert state.pending() is None
    assert state.checkpoint().snapshot_id == 30


def test_empty_snapshot_deletes_all_documents(config, state):
    stage(state, config, doc(config))
    accept(state, config)
    stage(state, config, snapshot=20)
    assert [event["kind"] for event in state.events()] == ["delete"]


@pytest.mark.parametrize("failure", ["duplicate", "read_error"])
def test_failed_scan_rolls_back_every_change(config, state, failure):
    stage(state, config, doc(config))
    accept(state, config)

    def documents():
        yield doc(config, 2)
        if failure == "duplicate":
            yield doc(config, 2)
        else:
            raise OSError("object read failed")

    with pytest.raises((ValueError, OSError)):
        state.stage(Snapshot("table-uuid", 20), documents(), ("brain",))
    assert state.pending() is None
    assert list(state.events()) == []
    assert state.checkpoint().snapshot_id == 10
    assert [r[0] for r in state.db.execute("SELECT id FROM documents")] == ["1"]


def test_pending_batch_survives_restart_and_does_not_need_the_lake(config, monkeypatch):
    with closing(State(config.state_dir, create=True)) as state:
        state.bind(config)
        stage(state, config, doc(config))
        batch_id = state.pending()["id"]
    calls = []

    def command(*args):
        calls.append(args)
        return receipt(*args)

    monkeypatch.setattr("vane.knowledge.targets.run_command", command)
    monkeypatch.setattr(
        "vane.knowledge.service.IcebergSource", lambda _: pytest.fail("must replay before reading catalog")
    )
    with closing(State(config.state_dir)) as state, state.exclusive():
        assert state.pending()["id"] == batch_id
        assert sync_once(config, state)
        assert state.pending() is None
        assert state.checkpoint().snapshot_id == 10
    assert len(calls) == 1


def test_retry_reuses_request_identity_and_skips_acknowledged_events(config, state, monkeypatch):
    stage(state, config, doc(config, 1), doc(config, 2))
    seen = []

    def command(*args):
        seen.append(args[1])
        if len(seen) == 2:
            raise DeliveryError("reply lost after commit")
        return receipt(*args)

    monkeypatch.setattr("vane.knowledge.targets.run_command", command)
    with pytest.raises(DeliveryError):
        deliver(state, config.targets, 10)
    assert state.delivery("brain")["position"] == 1
    assert state.checkpoint() is None
    deliver(state, config.targets, 10)
    assert len(seen) == 3
    assert seen[1] == seen[2]
    assert seen[0] != seen[1]


def test_each_target_has_its_own_acknowledgement(config, state, monkeypatch):
    config = replace(config, targets=(*config.targets, replace(config.targets[0], name="second")))
    stage(state, config, doc(config))
    seen = []

    def command(*args):
        seen.append(args[0].name)
        if len(seen) == 1:
            raise DeliveryError("first target unavailable")
        return receipt(*args)

    monkeypatch.setattr("vane.knowledge.targets.run_command", command)
    with pytest.raises(DeliveryError):
        deliver(state, config.targets, 10)
    assert state.delivery("second")["done"] == 1
    with pytest.raises(RuntimeError, match="every destination"):
        state.finish()
    deliver(state, config.targets, 10)
    assert seen == ["brain", "second", "brain"]


@pytest.mark.parametrize(
    "change", [{"state": "pending"}, {"request_id": "wrong"}, {"outcome": {}}, {"outcome": {"status": "error"}}]
)
def test_unconfirmed_receipt_keeps_outbox(config, state, monkeypatch, change):
    stage(state, config, doc(config))

    def command(*args):
        value = json.loads(receipt(*args))
        value.update(change)
        return encode(value).encode()

    monkeypatch.setattr("vane.knowledge.targets.run_command", command)
    with pytest.raises(DeliveryError, match="confirm"):
        deliver(state, config.targets, 10)
    assert state.delivery("brain")["position"] == 0
    assert state.checkpoint() is None


def test_openwiki_must_confirm_each_bounded_window_during_this_attempt(config, state, monkeypatch):
    target = replace(config.targets[0], kind="openwiki", destination="custom-mcp-lake")
    config = replace(config, targets=(target,))
    stage(state, config, *(doc(config, n, "x" * 50000) for n in range(8)))
    batch_id = state.pending()["id"]
    prior = state.start_attempt(target.name)
    page = state.read_changes(batch_id)
    state.acknowledge_changes(batch_id, page["receipt"])
    assert state.is_acknowledged(prior["id"])
    monkeypatch.setattr("vane.knowledge.targets.run_command", lambda *args: b"")
    with pytest.raises(DeliveryError, match="without acknowledging"):
        deliver_openwiki(state, target, 10)
    assert not state.delivery(target.name)["done"]
    observed = []

    def consume(*args):
        page = state.read_changes(batch_id)
        assert len(encode(page).encode()) < 70000
        observed.extend(e["document"]["id"] for e in page["events"])
        state.acknowledge_changes(batch_id, page["receipt"])
        return b""

    monkeypatch.setattr("vane.knowledge.targets.run_command", consume)
    deliver_openwiki(state, target, 10)
    assert state.delivery(target.name)["done"]
    assert state.get("attempt") is None
    assert observed == [str(n) for n in range(8)]


def test_stale_batch_and_inactive_reads_are_rejected(config, state):
    stage(state, config, doc(config))
    with pytest.raises(ValueError, match="active"):
        state.read_changes(state.pending()["id"])
    state.start_attempt("brain")
    batch_id = state.pending()["id"]
    previous_receipt = state.read_changes(batch_id)["receipt"]
    state.start_attempt("brain")
    with pytest.raises(ValueError, match="Receipt"):
        state.acknowledge_changes(batch_id, previous_receipt)
    with pytest.raises(ValueError, match="active"):
        state.read_changes("old-batch")


def test_openwiki_read_without_receipt_and_failed_synthesis_do_not_ack(config, state, monkeypatch):
    target = replace(config.targets[0], kind="openwiki", destination="custom-mcp-lake")
    config = replace(config, targets=(target,))
    stage(state, config, doc(config))

    def lost_response(*args):
        state.read_changes(state.pending()["id"])
        return b""

    monkeypatch.setattr("vane.knowledge.targets.run_command", lost_response)
    with pytest.raises(DeliveryError, match="without acknowledging"):
        deliver_openwiki(state, target, 10)

    def failed_synthesis(*args):
        page = state.read_changes(state.pending()["id"])
        state.acknowledge_changes(page["batch_id"], page["receipt"])
        raise DeliveryError("synthesis failed after acknowledgement")

    monkeypatch.setattr("vane.knowledge.targets.run_command", failed_synthesis)
    with pytest.raises(DeliveryError, match="synthesis failed"):
        deliver_openwiki(state, target, 10)
    assert state.delivery(target.name)["position"] == 0
    assert state.checkpoint() is None


def test_openwiki_retry_preserves_completed_windows(config, state, monkeypatch):
    target = replace(config.targets[0], kind="openwiki", destination="custom-mcp-lake")
    config = replace(config, targets=(target,))
    stage(state, config, doc(config, 1, "x" * 50000), doc(config, 2, "y" * 50000))
    seen = []

    def consume(*args):
        page = state.read_changes(state.pending()["id"])
        seen.extend(e["document"]["id"] for e in page["events"])
        if len(seen) == 2:
            raise DeliveryError("second window failed")
        state.acknowledge_changes(page["batch_id"], page["receipt"])
        return b""

    monkeypatch.setattr("vane.knowledge.targets.run_command", consume)
    with pytest.raises(DeliveryError):
        deliver_openwiki(state, target, 10)
    assert state.delivery(target.name)["position"] == 1
    deliver_openwiki(state, target, 10)
    assert seen == ["1", "2", "2"]


def test_configuration_binding_and_writer_lock(config, state):
    state.bind(config)
    with pytest.raises(ValueError, match="Configuration changed"):
        state.bind(replace(config, branch="other"))
    with state.exclusive(), closing(State(config.state_dir)) as other:
        with pytest.raises(RuntimeError, match="Another synchronizer"):
            with other.exclusive():
                pytest.fail("writer lock was not exclusive")
        with pytest.raises(RuntimeError, match="Another synchronizer"):
            State(config.state_dir, create=True)


def test_permissions_and_unknown_state_schema(config, state):
    assert config.state_dir.stat().st_mode & 0o077 == 0
    assert (config.state_dir / "state.sqlite3").stat().st_mode & 0o077 == 0
    state.db.execute("PRAGMA user_version=2")
    with pytest.raises(ValueError, match="schema"):
        State(config.state_dir)


def source(config, snapshots, head):
    result = object.__new__(IcebergSource)
    result.config = config
    result.table = SimpleNamespace(
        refresh=lambda: None,
        metadata=SimpleNamespace(table_uuid="table-uuid"),
        refs=lambda: {"main": SimpleNamespace(snapshot_ref_type="branch", snapshot_id=head)},
        snapshot_by_id=snapshots.get,
    )
    return result


def snap(parent, operation="append"):
    return SimpleNamespace(
        parent_snapshot_id=parent, summary=SimpleNamespace(operation=SimpleNamespace(value=operation))
    )


def test_snapshot_ids_are_opaque_and_only_replacements_can_skip_scan(config):
    history = {99: snap(None), 5: snap(99, "replace"), 2: snap(5, "append")}
    assert source(config, history, 5).discover(Snapshot("table-uuid", 99)).unchanged_content
    assert not source(config, history, 2).discover(Snapshot("table-uuid", 99)).unchanged_content


@pytest.mark.parametrize(
    "history,head,previous,message",
    [
        ({10: snap(None)}, 10, Snapshot("different-table", 10), "UUID"),
        ({10: snap(None)}, 10, Snapshot("table-uuid", 20), "ancestor"),
        ({20: snap(15)}, 20, Snapshot("table-uuid", 10), "expired"),
        ({20: snap(10)}, 20, Snapshot("table-uuid", 10), "expired"),
        ({10: snap(None), 20: snap(10, "unknown")}, 20, Snapshot("table-uuid", 10), "Unsupported"),
        ({20: snap(20)}, 20, Snapshot("table-uuid", 10), "ancestor"),
    ],
)
def test_history_problems_fail_without_reset(config, history, head, previous, message):
    with pytest.raises(ValueError, match=message):
        source(config, history, head).discover(previous)


def test_replacement_checkpoint_does_not_consume_documents(config, state):
    stage(state, config, doc(config))
    accept(state, config)

    def forbidden():
        pytest.fail("compaction should not read rows")
        yield

    state.stage(Snapshot("table-uuid", 20, True), forbidden(), ("brain",))
    assert state.checkpoint().snapshot_id == 20
    assert state.pending() is None


@pytest.mark.parametrize(
    "key,body", [(None, "x"), (True, "x"), (1.5, "x"), ("", "x"), (1, None), (1, " "), (1, "\x00")]
)
def test_invalid_documents_are_rejected(config, key, body):
    with pytest.raises(ValueError):
        doc(config, key, body)


def test_document_identity_hash_and_safe_frontmatter(config):
    assert doc(config, 1)["slug"] != doc(config, "1")["slug"]
    assert doc(config, 1, snapshot=20)["content_hash"] == doc(config, 1)["content_hash"]
    value = doc(config)
    value["title"] = 'A\n---\ncommand: "bad"'
    rendered = markdown(value)
    assert rendered.count("\n---\n") == 1
    assert 'title: "A\\n---\\ncommand:' in rendered
    with pytest.raises(ValueError, match="exceeds"):
        doc(replace(config, max_document_bytes=100))


def test_command_uses_stdin_and_literal_argv_and_timeout(config):
    target = replace(
        config.targets[0], command=(sys.executable, "-c", "import sys; print(sys.argv[1]); print(sys.stdin.read())")
    )
    output = run_command(target, ["$(false); `false`"], 10, "Body with secrets")
    assert output.decode() == "$(false); `false`\nBody with secrets\n"
    target = replace(target, command=(sys.executable, "-c", "import time; time.sleep(60)"))
    with pytest.raises(DeliveryError, match="timeout"):
        run_command(target, [], 0.1)


def test_openwiki_snippet_has_explicit_source_and_private_transport(config):
    target = replace(config.targets[0], kind="openwiki", destination="custom-mcp-lake")
    snippet = openwiki_source(config, target)
    assert snippet["id"] == "custom-mcp-lake"
    assert snippet["connectorId"] == "custom-mcp"
    assert snippet["connectorConfig"]["transport"]["args"][-1] == str(config.state_dir)


def test_json_config_is_strict_and_resolves_paths(tmp_path):
    path = tmp_path / "sync.json"
    value = {
        "name": "docs",
        "catalog": "lake",
        "table": ["db", "docs"],
        "state_dir": "state",
        "id_column": "id",
        "content_column": "body",
        "targets": [{"name": "brain", "kind": "gbrain", "cwd": ".", "destination": "default"}],
    }
    path.write_text(encode(value))
    loaded = Config.load(path)
    assert loaded.state_dir == tmp_path / "state"
    assert loaded.targets[0].cwd == str(tmp_path)
    for change in (
        {"poll_seconds": 0},
        {"poll_seconds": True},
        {"typo": 1},
        {"max_document_bytes": 999999},
        {"table": "db.docs"},
    ):
        path.write_text(encode({**value, **change}))
        with pytest.raises(ValueError):
            Config.load(path)
    for destination in ("all", "custom-mcp", "notion", "git-repo", "slack"):
        value["targets"][0].update(kind="openwiki", destination=destination)
        path.write_text(encode(value))
        with pytest.raises(ValueError, match="source instance"):
            Config.load(path)
