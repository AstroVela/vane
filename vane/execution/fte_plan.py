# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed logical inputs and output partitions for strictly separated FTE stages."""

from __future__ import annotations

from dataclasses import dataclass

from vane.execution.materialized_exchange import (
    MaterializedTask,
    OutputObject,
    PartitionSpec,
    StageManifest,
    fingerprint,
)
from vane.execution.pipelined_plan import task_id
from vane.execution.plan import FragmentSpec
from vane.execution.submission import RayQuerySpec


@dataclass(frozen=True)
class TaskBinding:
    task: MaterializedTask
    assignments: dict[str, list[str]]
    inputs: dict[str, tuple[OutputObject, ...]]


def bind_task(
    spec: RayQuerySpec,
    fragment: FragmentSpec,
    partition: int,
    upstream: dict[str, StageManifest],
) -> TaskBinding:
    if type(partition) is not int or not 0 <= partition < fragment.partition_count:
        raise ValueError("invalid materialized task partition")
    incoming = [e for e in spec.graph.exchanges if e.consumer_fragment_id == fragment.fragment_id]
    if set(upstream) != {e.producer_fragment_id for e in incoming}:
        raise ValueError("task requires exactly its committed upstream stages")
    fragments = {f.fragment_id: f for f in spec.graph.fragments}
    inputs: dict[str, tuple[OutputObject, ...]] = {}
    for edge in incoming:
        stage = upstream[edge.producer_fragment_id]
        producer = fragments[edge.producer_fragment_id]
        if (stage.query_id, stage.stage_id, stage.engine_identity) != (
            spec.query_id,
            producer.fragment_id,
            spec.graph.engine_identity,
        ) or {a.token.task_id for a in stage.attempts} != {
            task_id(producer.fragment_id, part) for part in range(producer.partition_count)
        }:
            raise ValueError("upstream stage does not match the fixed fragment graph")
        expected_schema = next(p.schema for p in fragment.inputs if p.port_id == edge.consumer_port)
        objects = []
        for attempt in stage.attempts:
            matching = [o for o in attempt.objects if o.output.identity == (edge.exchange_id, partition)]
            if len(matching) != 1 or matching[0].output.schema != expected_schema:
                raise ValueError("upstream stage is missing a partition or has a different schema")
            objects.extend(matching)
        inputs[edge.consumer_port] = inputs.get(edge.consumer_port, ()) + tuple(objects)
    assignments = {
        source.source_id: [split.split_id for split in source.splits[partition :: fragment.partition_count]]
        for source in fragment.sources
    }
    outputs: list[PartitionSpec] = []
    for edge in spec.graph.exchanges:
        if edge.producer_fragment_id == fragment.fragment_id:
            schema = next(p.schema for p in fragment.outputs if p.port_id == edge.producer_port)
            outputs.extend(
                PartitionSpec(edge.exchange_id, part, schema)
                for part in range(fragments[edge.consumer_fragment_id].partition_count)
            )
    if fragment.fragment_id == spec.graph.result.fragment_id:
        outputs.append(PartitionSpec("result", 0, spec.result_schema))
    identity = fingerprint(
        {
            "plan": spec.cache_key(),
            "fragment": fragment.fragment_id,
            "partition": partition,
            "assignments": assignments,
            "upstream": {name: manifest.identity for name, manifest in upstream.items()},
        }
    )
    task = MaterializedTask(task_id(fragment.fragment_id, partition), fragment.fragment_id, identity, tuple(outputs))
    return TaskBinding(task, assignments, inputs)
