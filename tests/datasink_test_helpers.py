# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Service-free SDK recording across real DataSink runner boundaries."""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from vane import DuckDBPyRelation
from vane.datasink import BoundKeyedUpsertSink, DataSink, DataSinkExecutionOptions, DataSinkWorker, WriteContext


@pytest.fixture(params=["local-fast", "local", pytest.param("ray", marks=pytest.mark.real_ray)])
def datasink_runner(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    import vane

    runner_type = request.param
    runner = None
    if runner_type == "ray":
        request.getfixturevalue("ray_local")
        from vane.runners.ray.runner import RayRunner

        runner = RayRunner(address=None, max_task_backlog=None)
    elif runner_type == "local":
        from vane.runners.local.runner import LocalRunner

        monkeypatch.setenv("VANE_LOCAL_FTE_WORKERS", "2")
        monkeypatch.setenv("VANE_LOCAL_FTE_EXECUTION_MODE", "in_process")
        runner = LocalRunner(num_workers=2)
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    if runner is not None:
        factory = "set_runner_ray" if runner_type == "ray" else "set_runner_local"
        monkeypatch.setattr(vane._native, factory, lambda *_args, **_kwargs: runner)
    try:
        yield runner_type
    finally:
        if runner_type == "ray":
            assert runner is not None
            runner.close()


def recording_sdk_sink(
    sink: DataSink,
    directory: Path,
    *,
    sdk_module: str,
    sdk_loader: str,
    sdk: tuple[type[Any], Any],
) -> DataSink:
    # Carry the fake SDK in the serialized bound sink so it reaches subprocesses
    # too. Each caller's autouse SDK fixture restores the driver module, and
    # worker actors own their SDK replacement for the duration of the operation.
    class RecordingBound(BoundKeyedUpsertSink):
        def __init__(self, bound: BoundKeyedUpsertSink) -> None:
            self.bound = bound

        @property
        def execution_options(self) -> DataSinkExecutionOptions:
            return self.bound.execution_options

        @property
        def key_columns(self) -> Sequence[str]:
            return self.bound.key_columns

        def prepare_input(self, relation: DuckDBPyRelation) -> DuckDBPyRelation:
            return self.bound.prepare_input(relation)

        def open_worker(self, context: WriteContext) -> DataSinkWorker:
            setattr(import_module(sdk_module), sdk_loader, lambda: sdk)
            worker = self.bound.open_worker(context)
            write = worker.write

            def record_schema(table: pa.Table) -> Any:
                (directory / f"{uuid.uuid4().hex}.schema").write_bytes(table.schema.serialize().to_pybytes())
                return write(table)

            worker.write = record_schema
            return worker

    class RecordingSink(DataSink):
        def bind(self, schema: pa.Schema) -> BoundKeyedUpsertSink:
            bound = sink.bind(schema)
            assert isinstance(bound, BoundKeyedUpsertSink)
            return RecordingBound(bound)

    return RecordingSink()
