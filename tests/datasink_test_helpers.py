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


@pytest.fixture
def datasink_runner() -> Iterator[str]:
    """DataSink writes use the local native Relation executor."""
    yield "local"


def recording_sdk_sink(
    sink: DataSink,
    directory: Path,
    *,
    sdk_module: str,
    sdk_loader: str,
    sdk: tuple[Any, ...],
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
