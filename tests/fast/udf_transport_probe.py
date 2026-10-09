# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise actual native transport planning in an isolated Ray client."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import ray

import vane
from vane.datasource import DataSource, DataSourceTask, read_datasource


def wait_for(path):
    deadline = time.monotonic() + 20
    while not path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"transport progress stalled waiting for {path.name}")
        time.sleep(0.01)


def payload_size(index, config):
    if config.get("oversized") and index == 35:
        return 140 * 1024
    if config.get("variable"):
        return (1, 257, 2048, 8192)[index % 4]
    return config.get("payload_bytes", 1024)


class InputTask(DataSourceTask):
    def __init__(self, config):
        self.config = config

    def execute(self):
        config = self.config
        root = Path(config["root"])
        block_rows = config["block_rows"]
        for start in range(0, config["rows"], block_rows):
            end = min(start + block_rows, config["rows"])
            ids = list(range(start, end))
            yield pa.record_batch(
                {
                    "id": pa.array(ids, type=pa.int64()),
                    "payload": [chr(65 + i % 26) * payload_size(i, config) for i in ids],
                }
            )
            if end >= 96:
                (root / "source_ready").touch()
            if config.get("gate") == "idle" and end == 32:
                wait_for(root / "seen_0")
            if config.get("gate") == "completion" and end == 128:
                # EOF is withheld until the pending short transport task runs.
                wait_for(root / "seen_96")
            if config.get("gate") == "two_actors" and end == 64:
                wait_for(root / "seen_32")


class InputSource(DataSource):
    def __init__(self, config):
        self.config = config

    @property
    def schema(self):
        return {"id": "BIGINT", "payload": "VARCHAR"}

    def get_tasks(self):
        yield InputTask(self.config)


def make_consumer(config):
    def compute(batch, task_id):
        root = Path(config["root"])
        if config["batch_format"] == "numpy":
            ids = batch["id"].tolist()
            payloads = batch["payload"].tolist()
        else:
            ids = batch["id"].to_pylist()
            payloads = batch["payload"].to_pylist()
        assert 0 < len(ids) <= 32
        sizes = []
        for index, value in zip(ids, payloads, strict=True):
            assert value == chr(65 + index % 26) * payload_size(index, config)
            sizes.append(len(value))
        event = {"task": task_id, "ids": ids, "sizes": sizes, "pid": os.getpid()}
        with (root / f"calls_{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        (root / f"seen_{ids[0]}").touch()
        if ids[0] == 0:
            if config.get("gate") == "aggregate":
                wait_for(root / "source_ready")
            elif config.get("gate") == "two_actors":
                wait_for(root / "seen_32")
        if config.get("fail") and 32 in ids:
            raise ValueError("planned second compute batch failure")
        if config.get("cancel") and 32 in ids:
            wait_for(root / "cancel_release")
        if config.get("restart") and 32 in ids:
            try:
                with (root / "actor_restarted").open("x"):
                    pass
            except FileExistsError:
                pass
            else:
                # Exit this test-owned actor after the first compute call has
                # already streamed output. Ray must replay without duplicates.
                os._exit(23)
        factor = config.get("expand", 1)
        return {"id": np.repeat(np.asarray(ids, dtype=np.int64), factor)}

    class Consumer:
        def __init__(self):
            from vane.execution.udf_actor_callable import ActorCallableRuntime

            self.task_id = None
            self.original_invoke = ActorCallableRuntime.__call__

            def observe_task(runtime, udf, batch):
                # Ray's task context belongs to the actor owner. Capture it
                # before the framework submits ordinary __call__ to its worker.
                # This instrumentation is confined to this test-owned process.
                assert udf is self
                self.task_id = ray.get_runtime_context().get_task_id()
                assert self.task_id is not None
                return self.original_invoke(runtime, udf, batch)

            ActorCallableRuntime.__call__ = observe_task

        def __call__(self, batch):
            assert self.task_id is not None
            return compute(batch, self.task_id)

        def _vane_close(self):
            from vane.execution.udf_actor_callable import ActorCallableRuntime

            ActorCallableRuntime.__call__ = self.original_invoke

    return Consumer


def main():
    config = json.loads(sys.argv[1])
    root = Path(config["root"])
    root.mkdir(exist_ok=True)

    if config.get("cancel"):
        import faulthandler

        faulthandler.dump_traceback_later(35, repeat=True)

    vane.runners.set_runner_ray(noop_if_initialized=True)
    con = vane.connect()
    try:
        con.execute("SET threads=2")
        rel = read_datasource(InputSource(config), con=con)
        if config.get("lazy"):

            def identity(table):
                return table

            rel = rel.map_batches(
                identity,
                schema={"id": "BIGINT", "payload": "VARCHAR"},
                execution_backend="ray_task",
                cpus=1,
                batch_size=config["block_rows"],
                output_batch_size=config["block_rows"],
                preserve_compute_batch_boundaries=True,
                task_input_max_bytes=256 * 1024,
                output_target_max_bytes=256 * 1024,
            )
        options = {}
        if config.get("soft_min"):
            options["min_task_batch_size"] = config["soft_min"]
        rel = rel.map_batches(
            make_consumer(config),
            schema={"id": "BIGINT"},
            execution_backend="ray_actor",
            batch_format=config["batch_format"],
            actor_number=config.get("actors", 1),
            cpus=1,
            batch_size=32,
            task_input_max_bytes=config.get("budget", 100 * 1024),
            output_target_max_bytes=config.get("output_budget", 4096),
            **options,
        )
        if config.get("slow_consumer"):

            def slow_consumer(batch):
                time.sleep(0.02)
                return batch

            rel = rel.map_batches(
                slow_consumer,
                schema={"id": "BIGINT"},
                execution_backend="ray_task",
                batch_format="numpy",
                cpus=1,
                batch_size=8,
                task_input_max_bytes=64,
                output_target_max_bytes=64,
            )
        if config.get("fail"):
            try:
                rel.to_arrow_table()
            except Exception as error:
                assert "planned second compute batch failure" in str(error), str(error)
            else:
                raise AssertionError("UDF failure was not propagated")
            result = {"failure_propagated": True}
        elif config.get("cancel"):
            cancelled = {}

            def interrupt_after_output():
                try:
                    wait_for(root / "seen_32")
                    cancelled["at"] = time.monotonic()
                    con.interrupt()
                except Exception as error:
                    cancelled["error"] = str(error)

            interrupter = threading.Thread(target=interrupt_after_output, daemon=True)
            interrupter.start()
            try:
                rel.to_arrow_table()
            except Exception as error:
                assert any(word in str(error).lower() for word in ("interrupt", "cancel")), str(error)
            else:
                raise AssertionError("query interruption was not propagated")
            finally:
                (root / "cancel_release").touch()
                interrupter.join(timeout=5)
            assert not interrupter.is_alive() and "error" not in cancelled, cancelled
            elapsed = time.monotonic() - cancelled["at"]
            assert elapsed < 15, elapsed
            result = {"cancel_propagated": True, "cancel_seconds": elapsed}
        elif config.get("limit"):
            start = time.monotonic()
            values = rel.limit(config["limit"]).to_arrow_table()["id"].to_pylist()
            assert len(values) == config["limit"]
            assert Counter(values) <= Counter({i: config.get("expand", 1) for i in range(config["rows"])})
            result = {"output_rows": len(values), "limit_seconds": time.monotonic() - start}
        else:
            values = rel.to_arrow_table()["id"].to_pylist()
            expected = Counter({i: config.get("expand", 1) for i in range(config["rows"])})
            assert Counter(values) == expected
            result = {"output_rows": len(values)}
    finally:
        con.close()
        vane.teardown_runner()
    calls = []
    for path in sorted(root.glob("calls_*.jsonl")):
        calls.extend(json.loads(line) for line in path.read_text().splitlines())
    tasks = {}
    for call in calls:
        task = tasks.setdefault(call["task"], {"rows": 0, "payload_bytes": 0, "calls": []})
        task["rows"] += len(call["ids"])
        task["payload_bytes"] += sum(call["sizes"])
        task["calls"].append(call["ids"])
    result.update(calls=calls, tasks=tasks, actor_restarted=(root / "actor_restarted").exists())
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
