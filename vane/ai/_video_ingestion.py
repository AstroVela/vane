# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile video ingestion using existing map_batches and aggregate operators."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any

import numpy as np
import pyarrow as pa

from vane.ai._video_content import decode_content, encode_content
from vane.ai._video_pipeline_types import TEXT_MODALITIES, ClipWindow, VideoMediaInfo, _Selection
from vane.ai.provider import _ProviderResultError, _safe_provider_execution_error
from vane.execution.udf_output_schema import _arrow_type_from_name

_CLIP_SCHEMA = {
    "clip_id": "VARCHAR",
    "asset_id": "VARCHAR",
    "source_version": "VARCHAR",
    "uri": "VARCHAR",
    "start_sec": "DOUBLE",
    "end_sec": "DOUBLE",
}
_JOB_SCHEMA = {
    **_CLIP_SCHEMA,
    "duration_sec": "DOUBLE",
    "origin": "DOUBLE",
    "available": "VARCHAR[]",
    "_group": "BIGINT",
}
_STAGE_KEYS = {"execution_backend", "actor_number", "cpus", "gpus", "batch_size", "memory_bytes"}


def _stage_options(resources: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    unknown = resources.keys() - {"probe", "preprocess", "embed", "fusion"}
    if unknown:
        raise ValueError("unknown video stages: " + ", ".join(sorted(unknown)))
    result = {}
    for name in ("probe", "preprocess", "embed", "fusion"):
        raw = resources.get(name, {})
        if not isinstance(raw, Mapping) or raw.keys() - _STAGE_KEYS:
            raise ValueError(f"invalid resource options for {name}")
        options = dict(raw)
        backend = options.setdefault(
            "execution_backend", "ray_task" if name in {"probe", "preprocess"} else "ray_actor"
        )
        if backend not in {"ray_task", "ray_actor"}:
            raise ValueError("video ingestion stages require ray_task or ray_actor")
        if backend == "ray_actor":
            options.setdefault("actor_number", 1)
        elif "actor_number" in options:
            raise ValueError("actor_number is only valid for ray_actor")
        options.setdefault("cpus", 1.0)
        options.setdefault("gpus", 0.0)
        options.setdefault("batch_size", 8 if name == "embed" else 1)
        result[name] = options
    return result


def _apply_stage(
    relation: Any,
    factory: type,
    schema: Mapping[str, str],
    options: Mapping[str, Any],
    budget: int,
    *,
    output_batch_size: int | None = None,
    min_task_batch_size: int | None = None,
) -> Any:
    if options["execution_backend"] == "ray_actor":
        function = factory
    else:

        def function(table: pa.Table) -> pa.Table:
            instance = factory()
            try:
                return instance(table)
            finally:
                close = getattr(instance, "close", None)
                if close is not None:
                    close()

    # Reuse Vane's explicit non-retryable UDF contract. Model exceptions never
    # become an empty channel or trigger execution through another backend.
    function._vane_datasink_no_task_retries = True
    return relation.map_batches(
        function,
        schema=schema,
        target_max_batch_bytes=budget,
        task_input_max_bytes=budget,
        output_target_max_bytes=budget,
        output_batch_size=output_batch_size,
        min_task_batch_size=min_task_batch_size,
        **options,
    )


def _table(rows: list[dict[str, Any]], schema: Mapping[str, str], budget: int) -> pa.Table:
    table = pa.Table.from_pylist(
        rows, schema=pa.schema([(name, _arrow_type_from_name(dtype)) for name, dtype in schema.items()])
    )
    if table.get_total_buffer_size() > budget:
        raise MemoryError("video stage output exceeds max_batch_bytes; reduce batch_size or content limits")
    return table


def validate_execution(source: Any, resources: Mapping[str, Any], max_batch_bytes: int) -> dict[str, dict[str, Any]]:
    if type(max_batch_bytes) is not int or max_batch_bytes <= 0:
        raise ValueError("max_batch_bytes must be a positive integer")
    if getattr(source, "connection", None) is not None:
        raise ValueError("distributed video sources cannot capture a live connection")
    return _stage_options(resources)


def build_video_ingestion(
    pipeline: Any, assets: Any, request: Any, resources: Mapping[str, Any], max_batch_bytes: int
) -> Any:
    from vane.ai.video import _content_in_clip, _ProfileRuntime

    source = pipeline._source
    options = validate_execution(source, resources, max_batch_bytes)
    profiles, fusion, max_clips = dict(pipeline._profiles), pipeline._fusion, pipeline._max_clips
    required = {"asset_id", "source_version", "uri"}
    input_schema = dict(zip(assets.columns, assets.types, strict=True))
    if not required <= input_schema.keys() or any(str(input_schema[c]) != "VARCHAR" for c in required):
        raise ValueError("video ingestion input requires VARCHAR asset_id, source_version and uri")
    groups = [
        group
        for selection in request.selections
        for group in (
            [_Selection("all", (c,)) for c in selection.modalities] if selection.policy == "all" else [selection]
        )
    ]

    class ProbeAndExpand:
        def __init__(self) -> None:
            self.source = copy.deepcopy(source)

        def __call__(self, table: pa.Table) -> pa.Table:
            rows = []
            for asset in table.to_pylist():
                if any(not isinstance(asset[name], str) or not asset[name].strip() for name in required):
                    raise ValueError("asset_id, source_version and uri must be nonempty strings")
                media = self.source.probe(asset["uri"])
                if not isinstance(media, VideoMediaInfo):
                    raise TypeError("video source probe must return VideoMediaInfo")
                if not request.asynchronous and media.duration > request.max_video_seconds:
                    raise ValueError("Video exceeds sync_budget.max_video_seconds; submit with async=true")
                if media.duration > request.clip_seconds * max_clips:
                    raise ValueError("video segmentation exceeds max_clips")
                for index in range(math.ceil(media.duration / request.clip_seconds)):
                    start, end = index * request.clip_seconds, min((index + 1) * request.clip_seconds, media.duration)
                    identity = [
                        "video-clips-v1",
                        asset["asset_id"],
                        asset["source_version"],
                        request.clip_seconds,
                        start,
                        end,
                    ]
                    clip_id = hashlib.sha256(
                        json.dumps(identity, allow_nan=False, separators=(",", ":")).encode()
                    ).hexdigest()
                    for group in range(len(groups)):
                        rows.append(
                            {
                                **asset,
                                "clip_id": clip_id,
                                "start_sec": start,
                                "end_sec": end,
                                "duration_sec": media.duration,
                                "origin": media.origin,
                                "available": sorted(media.available),
                                "_group": group,
                            }
                        )
            return _table(rows, _JOB_SCHEMA, max_batch_bytes)

    # Compute batch size alone does not split upstream blocks into Ray tasks.
    # Emit work-sized blocks and preserve those boundaries at preprocessing.
    jobs = _apply_stage(
        assets.project("asset_id, source_version, uri"),
        ProbeAndExpand,
        _JOB_SCHEMA,
        options["probe"],
        max_batch_bytes,
        output_batch_size=options["preprocess"]["batch_size"],
    )
    content_schema = {**_CLIP_SCHEMA, "_group": "BIGINT"}
    encoded_schema = {**_CLIP_SCHEMA, "_group": "BIGINT"}
    for channel in request.modalities:
        content_schema.update({f"{channel}_content": "BLOB", f"{channel}_metadata": "VARCHAR"})
        if request.output in {"separate", "both"}:
            encoded_schema[f"{channel}_vector"] = f"FLOAT[{profiles[channel].descriptor.get_dimensions()}]"
        encoded_schema[f"{channel}_metadata"] = "VARCHAR"
        if request.output in {"fused", "both"}:
            encoded_schema[f"{channel}_content"] = "BLOB"

    class Preprocess:
        def __init__(self) -> None:
            self.source = copy.deepcopy(source)

        def __call__(self, table: pa.Table) -> pa.Table:
            rows = []
            for row in table.to_pylist():
                clip = ClipWindow(row["start_sec"], row["end_sec"])
                media = VideoMediaInfo(row["duration_sec"], frozenset(row["available"]), row["origin"])
                result = {name: row[name] for name in (*_CLIP_SCHEMA, "_group")}
                group, winner = groups[row["_group"]], None
                for channel in group.modalities:
                    content = ()
                    if winner is not None:
                        audit = {"status": "skipped", "reason": "lower_priority", "selected_modality": winner}
                    elif channel not in media.available:
                        audit = {"status": "skipped", "reason": "no_stream"}
                    else:
                        content = _content_in_clip(self.source.extract(row["uri"], channel, clip, media), channel, clip)
                        audit = (
                            {"status": "selected", "reason": group.policy}
                            if content
                            else {"status": "skipped", "reason": "no_content"}
                        )
                        if content and group.policy == "first_available":
                            winner = channel
                    audit.update(modality=channel, start_sec=clip.start, end_sec=clip.end)
                    if request.return_timeline:
                        audit["timeline"] = [
                            {
                                "start_sec": c.start,
                                "end_sec": c.end,
                                **({"text": c.value} if channel in TEXT_MODALITIES else {}),
                            }
                            for c in content
                        ]
                    result[f"{channel}_content"] = encode_content(channel, content) if content else None
                    result[f"{channel}_metadata"] = json.dumps(audit, allow_nan=False)
                rows.append(result)
            return _table(rows, content_schema, max_batch_bytes)

    class Encode:
        def __init__(self) -> None:
            self.loop = asyncio.new_event_loop()
            self.runtimes: dict[str, Any] = {}
            self.cleanup = ExitStack()
            self.cleanup.callback(self.loop.close)

        def __call__(self, table: pa.Table) -> pa.Table:
            rows = []
            for row in table.to_pylist():
                result = {name: row[name] for name in (*_CLIP_SCHEMA, "_group")}
                for channel in groups[row["_group"]].modalities:
                    data, audit = row[f"{channel}_content"], json.loads(row[f"{channel}_metadata"])
                    if request.output in {"separate", "both"}:
                        profile = profiles[channel]
                        audit.update(
                            profile=profile.name,
                            provider=profile.descriptor.get_provider(),
                            model=profile.descriptor.get_model(),
                        )
                        vector = None
                        if data is not None:
                            if channel not in self.runtimes:
                                self.runtimes[channel] = _ProfileRuntime(profile, channel, self.loop)
                                self.cleanup.callback(self.runtimes[channel].close)
                            vector = self.runtimes[channel].embed(decode_content(channel, data))
                        result[f"{channel}_vector"] = vector
                    result[f"{channel}_metadata"] = json.dumps(audit, allow_nan=False)
                    if request.output in {"fused", "both"}:
                        result[f"{channel}_content"] = data
                rows.append(result)
            return _table(rows, encoded_schema, max_batch_bytes)

        def close(self) -> None:
            self.cleanup.close()

    contents = _apply_stage(
        jobs,
        Preprocess,
        content_schema,
        options["preprocess"],
        max_batch_bytes,
        min_task_batch_size=options["preprocess"]["batch_size"],
    )
    vectors = _apply_stage(contents, Encode, encoded_schema, options["embed"], max_batch_bytes)
    # Each group always returns a row, even when all its channels are absent.
    # Reject duplicate source identities or incomplete groups instead of picking
    # an arbitrary vector. Filter by group identity, not by vector nullness.
    count = len(groups)
    aggregates = [
        f"CASE WHEN count(*) = {count} AND count(DISTINCT _group) = {count} "
        "THEN clip_id ELSE error('duplicate asset identity or incomplete video groups') END AS clip_id"
    ]
    aggregates.extend(f"first({name}) AS {name}" for name in _CLIP_SCHEMA if name != "clip_id")
    for group_index, group in enumerate(groups):
        for channel in group.modalities:
            for suffix in ("vector", "metadata", "content"):
                name = f"{channel}_{suffix}"
                if name in encoded_schema:
                    aggregates.append(f"first({name}) FILTER (WHERE _group = {group_index}) AS {name}")
    merged = vectors.aggregate(", ".join(aggregates), "clip_id")
    if request.output == "separate":
        return merged

    assert fusion is not None
    final_schema = {
        name: str(dtype)
        for name, dtype in zip(merged.columns, merged.types, strict=True)
        if not name.endswith("_content")
    }
    final_schema.update(fused_vector=f"FLOAT[{fusion.dimensions}]", fused_metadata="VARCHAR")

    class Fuse:
        def __init__(self) -> None:
            self.loop = asyncio.new_event_loop()

        def __call__(self, table: pa.Table) -> pa.Table:
            rows = []
            for row in table.to_pylist():
                clip = ClipWindow(row["start_sec"], row["end_sec"])
                selected = {
                    channel: decode_content(channel, row[f"{channel}_content"])
                    for channel in request.modalities
                    if row[f"{channel}_content"] is not None
                }
                vector = None
                if selected:
                    if frozenset(selected) not in fusion.combinations:
                        raise ValueError("fusion model does not support this clip's selected modalities")
                    settings = copy.deepcopy(request.fusion)
                    settings.setdefault("method", fusion.method)
                    if "weights" in settings:
                        settings["weights"] = {c: w for c, w in settings["weights"].items() if c in selected}
                    try:
                        vector = fusion.embed(clip, selected, settings)
                        if inspect.isawaitable(vector):
                            vector = self.loop.run_until_complete(vector)
                    except Exception as error:
                        raise _safe_provider_execution_error(
                            fusion.provider, fusion.model, "video fusion", error
                        ) from None
                    try:
                        with np.errstate(over="ignore", invalid="ignore"):
                            vector = np.asarray(vector, dtype=np.float32)
                    except (TypeError, ValueError, OverflowError):
                        raise _ProviderResultError("Fusion model returned a non-numeric embedding") from None
                    if vector.shape != (fusion.dimensions,) or not np.isfinite(vector).all():
                        raise _ProviderResultError("Fusion model returned an invalid embedding")
                    vector = vector.tolist()
                result = {name: row[name] for name in final_schema if name in row}
                result.update(
                    fused_vector=vector,
                    fused_metadata=json.dumps(
                        {
                            "status": "selected" if selected else "skipped",
                            "reason": "joint" if selected else "no_content",
                            "modalities": list(selected),
                            "profile": fusion.name,
                            "provider": fusion.provider,
                            "model": fusion.model,
                            "start_sec": clip.start,
                            "end_sec": clip.end,
                        }
                    ),
                )
                rows.append(result)
            return _table(rows, final_schema, max_batch_bytes)

        def close(self) -> None:
            self.loop.close()

    return _apply_stage(merged, Fuse, final_schema, options["fusion"], max_batch_bytes)
