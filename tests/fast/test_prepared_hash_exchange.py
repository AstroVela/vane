# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared HASH outputs retain native routing across batches and sink tasks."""

import time

import pytest

import vane
from tests.fast.test_direct_exchange import submission
from vane._native import execution_plan
from vane._native import execution_runtime as native
from vane.execution.compiler import FragmentCompileOptions, compile_fragment_graph


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("partitions", [3, 5])
def test_multiple_hash_outputs_reuse_scratch_without_mixing_keys(threads, partitions):
    sql = (
        "select range k, case when range%11=0 then null else range%7 end n, "
        "42::integer c, 'key-'||(range%23)::varchar s, [range%19,null] a "
        "from range(12299) where range%3!=0"
    )
    keys = [(2,), (0,), (1, 0), (0, 1), (3,), (4,)]
    names = ["k", "n", "c", "s", "a"]
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        hashes = [f"hash({','.join(names[k] for k in columns)})%{partitions}" for columns in keys]
        expected = {(edge, part): [] for edge in range(len(keys)) for part in range(partitions)}
        for row in connection.execute(f"select *, {','.join(hashes)} from ({sql})").fetchall():
            for edge, part in enumerate(row[len(names) :]):
                expected[edge, part].append(row[: len(names)])
        spec = submission(connection, sql, partitions=partitions, hash_columns=keys[0])
        source = spec.graph.fragments[0]
        snapshot = next(s.payload for s in spec.source_snapshots if s.fragment_id == source.fragment_id)
        channels, outputs = {}, []
        for edge, columns in enumerate(keys):
            graph = compile_fragment_graph(
                connection, sql, query_id=f"hash-{edge}", options=FragmentCompileOptions(partitions, columns)
            )
            targets = []
            for part in range(partitions):
                channel = native.DirectChannel(
                    source.outputs[0].schema, native.DirectLimits(4096, 1024, 64, 2), 1, ["client"]
                )
                channel.add_producer("source")
                channel.seal_producers()
                channels[edge, part] = channel
                targets.append(channel)
            outputs.append({"channels": targets, "producer": "source", "partitioning": graph.exchanges[0].partitioning})
        service = native.TaskService(connection)
        actual = {key: [] for key in channels}
        sequences = {key: [] for key in channels}
        try:
            service.prepare(
                "source",
                source.native_plan,
                spec.connection_snapshot,
                snapshot,
                {s.source_id: [split.split_id for split in s.splits] for s in source.sources},
                {},
                outputs,
            )
            service.start("source", "initial")
            ended = set()
            deadline = time.monotonic() + 20
            while len(ended) != len(channels):
                service.pump(1)
                for key, channel in channels.items():
                    status, batch = channel.poll("client")
                    if status == "end":
                        ended.add(key)
                    if batch is not None:
                        try:
                            actual[key].extend(batch.to_rows())
                            sequences[key].append(batch.sequence)
                        finally:
                            batch.close()
                assert time.monotonic() < deadline, service.status()
            for key, channel in channels.items():
                assert sorted(actual[key]) == sorted(expected[key])
                assert sequences[key] == list(range(1, len(sequences[key]) + 1))
                assert channel.snapshot()["peak_bytes"] <= 4096
        finally:
            service.cancel("test cleanup")
            service.release()
            for channel in channels.values():
                channel.close_consumer("client")
        assert all(c.snapshot()["bytes"] == c.snapshot()["leased_bytes"] == 0 for c in channels.values())


@pytest.mark.parametrize("damage", ["trailing", "truncated", "foreign_engine", "schema"])
def test_hash_validation_precedes_task_start(damage):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        spec = submission(connection, "select range from range(9)", hash_columns=(0,))
        source = spec.graph.fragments[0]
        partitioning = spec.graph.exchanges[0].partitioning
        error = vane.SerializationException
        if damage == "trailing":
            partitioning += b"trailing"
        elif damage == "truncated":
            partitioning = partitioning[:-1]
        elif damage == "foreign_engine":
            identity = execution_plan.engine_identity().encode()
            assert identity in partitioning
            partitioning = partitioning.replace(identity, b"x" * len(identity))
        else:
            other = submission(connection, "select range::varchar from range(9)", hash_columns=(0,))
            partitioning = other.graph.exchanges[0].partitioning
            error = vane.InvalidInputException
        channel = native.DirectChannel(source.outputs[0].schema, native.DirectLimits(64, 64, 3, 1), 1, ["client"])
        channel.add_producer("source")
        channel.seal_producers()
        service = native.TaskService(connection)
        try:
            with pytest.raises(error):
                service.prepare(
                    "source",
                    source.native_plan,
                    spec.connection_snapshot,
                    next(s.payload for s in spec.source_snapshots if s.fragment_id == source.fragment_id),
                    {s.source_id: [split.split_id for split in s.splits] for s in source.sources},
                    {},
                    [{"channels": [channel], "producer": "source", "partitioning": partitioning}],
                )
            assert service.status() == []
            assert channel.snapshot()["accepted_rows"] == channel.snapshot()["bytes"] == 0
        finally:
            service.release()
