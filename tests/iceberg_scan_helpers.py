# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared native and Ray Iceberg scan qualification helpers."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

import vane


def connection(tables):
    return vane.connect(
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "allow_persistent_secrets": "false",
            "secret_directory": str(tables.directory / "secrets"),
        }
    )


def attach(con, endpoint):
    vane.load_installed_extension("iceberg", connection=con)
    con.execute(
        "ATTACH '' AS lake (TYPE iceberg, ENDPOINT $endpoint, AUTHORIZATION_TYPE 'none', READ_ONLY)",
        {"endpoint": endpoint},
    )


def freeze(con, sql):
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(con.sql(sql), "frozen-iceberg-" + uuid.uuid4().hex)
    return pickle.dumps(logical)


def execute_frozen(payload, tables, *, split_batch=None):
    """Use the existing native worker entry point with an explicit file assignments."""
    with connection(tables) as worker:
        logical = pickle.loads(payload)
        physical = logical.to_physical_plan(worker)
        batch_map = physical.scan_split_batch_map()
        assert len(batch_map) == 1
        assignment = {
            str(node): vane.ray_cxx.merge_scan_split_batches(batches) if split_batch is None else split_batch
            for node, batches in batch_map.items()
        }
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
        try:
            worker_plan = physical.clone(worker)
            result = runner.execute_native(worker, worker_plan, scan_split_batch=assignment)
            assert result.completion_status in {"ok", "empty"}
            rows = []
            for part in result.partition_payloads:
                table = part.to_arrow() if hasattr(part, "to_arrow") else part
                rows.extend(zip(*(column.to_pylist() for column in table.columns), strict=True))
            schema = result.result_schema
            result = worker_plan = physical = logical = None
            return rows, schema
        finally:
            runner.shutdown()


class RestForwarder:
    """Forward to the real service, then close the exact endpoint captured during binding."""

    def __init__(self, endpoint):
        # Binding can hold the GIL while fetching metadata. The forwarding
        # server must have its own interpreter, not a thread in this process.
        self.process = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).parent / "gravitino_iceberg_proxy.py"), endpoint],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.stopped = False
        timeout = threading.Timer(10, self.process.kill)
        timeout.start()
        try:
            line = self.process.stdout.readline()
            if not line:
                raise AssertionError("REST forwarder did not start: " + self.process.stderr.read())
            self.endpoint = json.loads(line)["endpoint"]
        except BaseException:
            self.stop()
            raise
        finally:
            timeout.cancel()
            timeout.join()

    def stop(self):
        if not self.stopped:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.communicate(timeout=5)
                raise AssertionError("REST forwarder failed to stop") from None
            self.stopped = True


@contextmanager
def unavailable_metadata(tables):
    moved = []
    try:
        for metadata in tables.directory.glob("*/metadata"):
            hidden = metadata.with_name("frozen-test-hidden-metadata")
            metadata.rename(hidden)
            moved.append((metadata, hidden))
        assert moved
        yield
    finally:
        for metadata, hidden in reversed(moved):
            hidden.rename(metadata)
