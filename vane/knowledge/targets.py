# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Acknowledged delivery through the current OpenWiki and GBrain CLIs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import uuid
from typing import Any

from vane.knowledge.config import Target, encode
from vane.knowledge.state import State


class DeliveryError(RuntimeError):
    """A durable batch remains pending and may be retried with the same identity."""


def run_command(target: Target, arguments: list[str], timeout: float, content: str = "") -> bytes:
    # Spool potentially verbose agent output to disk, never to the service log.
    # Start a process group so a timeout/interrupt also stops spawned MCP clients.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(
                [*target.command, *arguments],
                cwd=target.cwd,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        except OSError as exc:
            raise DeliveryError(f"Cannot start destination {target.name}") from exc
        try:
            try:
                process.communicate(content.encode("utf-8"), timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise DeliveryError(f"Destination {target.name} exceeded its timeout") from exc
            if process.returncode != 0:
                raise DeliveryError(f"Destination {target.name} exited with status {process.returncode}")
            stdout.seek(0)
            result = stdout.read(1048577)
            if target.kind == "gbrain" and len(result) > 1048576:
                raise DeliveryError("GBrain receipt exceeded the response limit")
            return result
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def markdown(document: dict[str, Any]) -> str:
    fields = {
        "title": document["title"],
        "source_uri": document["uri"],
        "iceberg_table_uuid": document["table_uuid"],
        "iceberg_snapshot_id": str(document["snapshot_id"]),
        "iceberg_document_id": document["id"],
    }
    # JSON string literals are valid YAML scalars and cannot inject frontmatter.
    return (
        "---\n"
        + "".join(f"{key}: {encode(value)}\n" for key, value in fields.items())
        + "---\n\n"
        + document["content"]
        + "\n"
    )


def deliver_gbrain(state: State, target: Target, timeout: float) -> None:
    batch = state.pending()
    if batch is None:
        raise ValueError("No batch to deliver")
    position = state.delivery(target.name)["position"]
    for event in state.events(position):
        doc = event["document"]
        request_id = str(uuid.uuid5(uuid.UUID(batch["id"]), encode([target.name, event["seq"]])))
        verb = "put" if event["kind"] == "upsert" else "delete"
        output = run_command(
            target,
            [verb, doc["slug"], "--source-id", target.destination, "--force", "--request-id", request_id, "--json"],
            timeout,
            markdown(doc) if verb == "put" else "",
        )
        try:
            receipt = json.loads(output)
        except ValueError as exc:
            raise DeliveryError("GBrain did not return a JSON write receipt") from exc
        if not isinstance(receipt, dict):
            raise DeliveryError("GBrain did not return a write receipt")
        outcome = receipt.get("outcome")
        expected = ("created_or_updated", "skipped") if verb == "put" else ("soft_deleted",)
        if (
            receipt.get("state") != "committed"
            or receipt.get("request_id") != request_id
            or not isinstance(outcome, dict)
            or outcome.get("slug") != doc["slug"]
            or outcome.get("source_id") != target.destination
            or outcome.get("status") not in expected
        ):
            raise DeliveryError("GBrain did not confirm the requested mutation; retaining the batch")
        position = event["seq"]
        state.acknowledge(target.name, position)
    state.acknowledge(target.name, position, done=True)


def deliver_openwiki(state: State, target: Target, timeout: float) -> None:
    last = state.db.execute("SELECT max(seq) FROM events").fetchone()[0]
    position = state.delivery(target.name)["position"]
    while position < last:
        attempt = state.start_attempt(target.name)
        try:
            run_command(target, ["ingest", target.destination, "--print"], timeout)
            if not state.is_acknowledged(attempt["id"]):
                raise DeliveryError("OpenWiki exited without acknowledging its changes; retaining the batch")
            position = attempt["end"]
            state.acknowledge(target.name, position)
        finally:
            with state.db:
                state.db.execute("DELETE FROM metadata WHERE key='attempt'")
    state.acknowledge(target.name, position, done=True)


def deliver(state: State, targets: tuple[Target, ...], timeout: float) -> None:
    errors = []
    for target in targets:
        if state.delivery(target.name)["done"]:
            continue
        try:
            if target.kind == "gbrain":
                deliver_gbrain(state, target, timeout)
            else:
                deliver_openwiki(state, target, timeout)
        except DeliveryError as exc:
            errors.append(str(exc))
    if errors:
        raise DeliveryError("; ".join(errors))
    state.finish()
