# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Video asset orchestration: clip selection, outputs and optional durable jobs.

Selection has no search AND/OR semantics. Retrieval combines the returned vectors
separately. A configured job store persists request status and results.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyarrow as pa

from vane.ai._audio_embedding import AudioClip
from vane.ai._video_embedding import VideoClip
from vane.ai._video_jobs import PersistedVideoTask, VideoJobStore
from vane.ai._video_pipeline_types import (
    MODALITIES,
    TEXT_MODALITIES,
    ClipWindow,
    TimedContent,
    VideoMediaInfo,
    VideoSource,
    _VideoRequest,
    object_fields,
)
from vane.ai._video_source import VaneVideoSource

__all__ = [
    "ClipWindow",
    "TimedContent",
    "VideoMediaInfo",
    "VideoSource",
    "VideoEmbeddingProfile",
    "VideoFusionProfile",
    "VideoEmbeddingPipeline",
    "VideoEmbeddingTask",
    "VideoEmbeddingTimeout",
    "VaneVideoSource",
    "VideoJobStore",
]


@dataclass(frozen=True)
class VideoEmbeddingProfile:
    """A named, reproducible configuration of an existing Vane descriptor.

    Providers run in distributed UDF workers for every entry point.
    The profile name should version model options
    (including revisions) so indexed vectors retain their encoding identity.
    """

    name: str
    descriptor: Any
    normalize: bool = False

    def __post_init__(self) -> None:
        from vane.ai.protocols import AudioEmbedderDescriptor, TextEmbedderDescriptor, VideoEmbedderDescriptor

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("profile name must be nonempty")
        if not isinstance(self.descriptor, (TextEmbedderDescriptor, VideoEmbedderDescriptor, AudioEmbedderDescriptor)):
            raise TypeError("profile requires a text, video or audio embedder descriptor")
        if type(self.descriptor.get_dimensions()) is not int or self.descriptor.get_dimensions() <= 0:
            raise ValueError("profile dimensions must be a positive integer")
        if type(self.normalize) is not bool:
            raise TypeError("profile normalize must be a bool")

    @classmethod
    def from_provider(
        cls,
        name: str,
        *,
        modality: str,
        provider: Any,
        model: str | None = None,
        dimensions: int | None = None,
        **options: Any,
    ) -> VideoEmbeddingProfile:
        """Resolve provider metadata without loading weights or making requests."""
        from vane.ai.functions import _prepare_embed_call

        if modality not in MODALITIES:
            raise ValueError("Unknown profile modality")
        unsupported = options.keys() & {
            "execution_backend",
            "actor_number",
            "batch_size",
            "max_chunk_chars",
            "chunk_overlap_chars",
            "max_retries",
        }
        if unsupported:
            raise ValueError(
                "Video pipeline profiles do not accept execution options: " + ", ".join(sorted(unsupported))
            )
        kind: Literal["text", "video", "audio"] = (
            "text" if modality in TEXT_MODALITIES else "video" if modality == "visual" else "audio"
        )
        descriptor, _, _, normalize, _, _, _ = _prepare_embed_call(
            provider,
            model,
            dimensions,
            "raise",
            options,
            relation=False,
            input_kind=kind,
        )
        return cls(name, descriptor, normalize)


@dataclass(frozen=True)
class VideoFusionProfile:
    """An explicitly capable joint-input model, supplied by the application.

    ``embed`` receives the clip, selected timestamped inputs, and fusion options.
    It must interpret their timestamps on the common asset timeline. Vane never
    adds independent model vectors. Only declared modality combinations may run.
    The callback and any model/client it owns remain application-owned.
    """

    name: str
    provider: str
    model: str
    dimensions: int
    combinations: tuple[frozenset[str], ...]
    embed: Callable[[ClipWindow, Mapping[str, tuple[TimedContent, ...]], Mapping[str, Any]], Any]
    method: str = "joint"
    supports_weights: bool = False

    def __post_init__(self) -> None:
        for field in (self.name, self.provider, self.model, self.method):
            if not isinstance(field, str) or not field.strip():
                raise ValueError("Fusion identity and method must be nonempty strings")
        if type(self.dimensions) is not int or self.dimensions <= 0:
            raise ValueError("Fusion dimensions must be a positive integer")
        if (
            not isinstance(self.combinations, tuple)
            or not self.combinations
            or any(
                not isinstance(combo, frozenset) or not combo or not combo <= MODALITIES for combo in self.combinations
            )
        ):
            raise ValueError("Fusion combinations must explicitly list supported nonempty modality sets")
        if not callable(self.embed) or type(self.supports_weights) is not bool:
            raise TypeError("Fusion requires an embed callback and boolean supports_weights")


def _content_in_clip(values: Any, channel: str, clip: ClipWindow) -> tuple[TimedContent, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("Video extractors must return a sequence of TimedContent; use [] for normal absence")
    if any(not isinstance(item, TimedContent) for item in values):
        raise TypeError("Video extractors must return TimedContent with absolute timestamps")
    result = []
    for item in sorted(values, key=lambda item: (item.start, item.end)):
        overlaps = (
            clip.start <= item.start < clip.end
            if item.start == item.end
            else item.start < clip.end and item.end > clip.start
        )
        if not overlaps:
            continue
        if channel in TEXT_MODALITIES:
            if not isinstance(item.value, str):
                raise TypeError("Text channels require string content")
            if not item.value.strip():
                continue
        else:
            if item.start < clip.start or item.end > clip.end:
                raise ValueError("Decoded media must be windowed to the requested clip")
            if channel == "visual":
                if not isinstance(item.value, VideoClip):
                    raise TypeError("visual content requires VideoClip")
                if not item.value.frames:
                    continue
                if len(item.value.frames) != len(item.value.frame_times) or any(
                    isinstance(t, bool)
                    or not isinstance(t, (int, float))
                    or not math.isfinite(t)
                    or not clip.start <= t < clip.end
                    for t in item.value.frame_times
                ):
                    raise ValueError("Visual frames require timestamps inside the current clip")
                indices, times = item.value.frame_indices, item.value.frame_times
                if (
                    len(indices) != len(times)
                    or any(type(i) is not int or i < 0 for i in indices)
                    or any(a >= b for a, b in zip(indices, indices[1:]))
                    or any(a > b for a, b in zip(times, times[1:]))
                ):
                    raise ValueError("Visual frame indices and timestamps must be ordered")
                if any(
                    not isinstance(frame, np.ndarray)
                    or frame.dtype != np.uint8
                    or frame.ndim != 3
                    or frame.shape[2] != 3
                    or min(frame.shape[:2]) <= 0
                    for frame in item.value.frames
                ):
                    raise ValueError("Visual content requires nonempty UInt8 RGB frames")
            else:
                if not isinstance(item.value, AudioClip):
                    raise TypeError("audio content requires AudioClip")
                if type(item.value.sample_rate) is not int or item.value.sample_rate <= 0:
                    raise ValueError("Audio sample rate must be positive")
                samples = item.value.samples
                if (
                    not isinstance(samples, np.ndarray)
                    or samples.ndim != 2
                    or samples.shape[1] < 1
                    or samples.dtype.kind != "f"
                ):
                    raise ValueError("Audio content requires frame-major floating PCM")
                if not np.isfinite(samples).all() or np.any(np.abs(samples) > 1):
                    raise ValueError("Audio content requires finite PCM in [-1, 1]")
                if not len(item.value.samples):
                    continue
                if abs((item.end - item.start) * item.value.sample_rate - len(item.value.samples)) > 1.01:
                    raise ValueError("Audio content timestamps must match its sample duration")
        result.append(TimedContent(max(item.start, clip.start), min(item.end, clip.end), item.value))
    return tuple(result)


class _ProfileRuntime:
    def __init__(self, profile: VideoEmbeddingProfile, channel: str, loop: asyncio.AbstractEventLoop) -> None:
        from vane.ai.functions import _EmbedAudioBatch, _EmbedTextBatch, _EmbedVideoBatch

        wrapper = (
            _EmbedTextBatch
            if channel in TEXT_MODALITIES
            else _EmbedVideoBatch
            if channel == "visual"
            else _EmbedAudioBatch
        )
        self.channel = channel
        self.wrapper = wrapper(
            profile.descriptor,
            "input",
            "embedding",
            profile.descriptor.get_dimensions(),
            max_retries=0,
            normalize=profile.normalize,
        )
        self.wrapper.bind_async_runtime(loop.run_until_complete)

    def close(self) -> None:
        try:
            self.wrapper.close()
        finally:
            # Task completion also releases synchronous model weights, including
            # when an exception traceback retains this runtime.
            self.wrapper._embedder = None

    def embed(self, values: tuple[TimedContent, ...]) -> list[float]:
        import vane
        from vane.execution.udf_file_contract import FileUDFContract

        if self.channel in TEXT_MODALITIES:
            array = pa.array(["\n".join(item.value for item in values)], type=pa.string())
        else:
            if len(values) != 1:
                raise ValueError("Media sources must return one windowed VideoClip or AudioClip per channel")
            value = values[0].value
            payload: list[dict[str, Any]] | dict[str, Any]
            if self.channel == "visual":
                dtype = vane.list_type(
                    vane.struct_type(
                        {
                            "frame_index": vane.sqltypes.BIGINT,
                            "frame_time": vane.sqltypes.DOUBLE,
                            "data": vane.sqltype("IMAGE('RGB')"),
                        }
                    )
                )
                if len(value.frames) != len(value.frame_indices):
                    raise ValueError("Video frame indices must cover every frame")
                payload = [
                    {"frame_index": i, "frame_time": float(t), "data": f}
                    for i, t, f in zip(value.frame_indices, value.frame_times, value.frames, strict=True)
                ]
            else:
                dtype = vane.struct_type(
                    {"sample_rate": vane.sqltypes.INTEGER, "data": vane.tensor_type(vane.sqltypes.DOUBLE, (None, None))}
                )
                payload = {"sample_rate": value.sample_rate, "data": np.asarray(value.samples, dtype=np.float64)}
            array = FileUDFContract("video_pipeline", (), (dtype,)).scalar_outputs_to_array([payload])
        return self.wrapper(pa.table({"input": array}))["embedding"][0].as_py()


class VideoEmbeddingTimeout(TimeoutError):
    """The synchronous caller's budget expired; task cleanup is cooperative."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(
            f"Video embedding exceeded the synchronous time budget; cancellation requested for task {task_id}"
        )


class VideoEmbeddingTask:
    """An in-process job. Exceptions are raised by result(), never converted to absence."""

    def __init__(self, task_id: str, future: Future[dict[str, Any]], cancelled: threading.Event) -> None:
        self.id = task_id
        self._future = future
        self._cancelled = cancelled

    @property
    def status(self) -> str:
        if self._future.cancelled():
            return "cancelled"
        if not self._future.done():
            return "cancelling" if self._cancelled.is_set() else "processing" if self._future.running() else "pending"
        error = self._future.exception()
        return (
            "cancelled"
            if isinstance(error, (CancelledError, VideoEmbeddingTimeout))
            else "failed"
            if error is not None
            else "ready"
        )

    def result(self, timeout: float | None = None) -> dict[str, Any]:
        return copy.deepcopy(self._future.result(timeout=timeout))

    def cancel(self) -> None:
        self._cancelled.set()
        self._future.cancel()


class VideoEmbeddingPipeline:
    """Video orchestration and lazy distributed ingestion over embedding profiles.

    ``create(request)`` returns a task when ``async`` is true and a response
    otherwise. The default synchronous budget is 60 seconds of wall time and
    60 seconds of video. All entry points compile the same Ray ingestion plan.
    Cancellation interrupts the query and waits for runner cleanup.
    """

    def __init__(
        self,
        *,
        profiles: Mapping[str, VideoEmbeddingProfile],
        source: VideoSource | None = None,
        fusion: VideoFusionProfile | None = None,
        max_tasks: int = 128,
        max_clips: int = 10000,
        job_store: VideoJobStore | None = None,
        pipeline_version: str | None = None,
    ) -> None:
        for name, value in (("max_tasks", max_tasks), ("max_clips", max_clips)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if any(
            channel not in MODALITIES or not isinstance(profile, VideoEmbeddingProfile)
            for channel, profile in profiles.items()
        ):
            raise TypeError("profiles must map modalities to VideoEmbeddingProfile objects")
        if fusion is not None and not isinstance(fusion, VideoFusionProfile):
            raise TypeError("fusion must be a VideoFusionProfile")
        if job_store is not None and not isinstance(job_store, VideoJobStore):
            raise TypeError("job_store must be a VideoJobStore")
        if job_store is not None and (not isinstance(pipeline_version, str) or not pipeline_version.strip()):
            raise ValueError("Durable jobs require an explicit immutable pipeline_version")
        self._job_store, self._pipeline_version = job_store, pipeline_version
        self._profiles = dict(profiles)
        self._source = source if source is not None else VaneVideoSource()
        self._fusion = fusion
        self._max_tasks = max_tasks
        self._max_clips = max_clips
        self._tasks: dict[str, VideoEmbeddingTask] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vane-video")
        self._closed = False

    def _validate(self, request: _VideoRequest) -> None:
        from vane.ai.protocols import AudioEmbedderDescriptor, TextEmbedderDescriptor, VideoEmbedderDescriptor

        missing = set(request.modalities) - self._source.supported_modalities
        if missing:
            raise ValueError("Configure extractors for requested modalities: " + ", ".join(sorted(missing)))
        if request.output in {"separate", "both"}:
            for channel in request.modalities:
                profile = self._profiles.get(channel)
                expected = (
                    TextEmbedderDescriptor
                    if channel in TEXT_MODALITIES
                    else VideoEmbedderDescriptor
                    if channel == "visual"
                    else AudioEmbedderDescriptor
                )
                if profile is None or not isinstance(profile.descriptor, expected):
                    raise ValueError(f"Configure a compatible embedding profile for {channel}")
        if request.output in {"fused", "both"}:
            fusion = self._fusion
            if fusion is None:
                raise ValueError("fused/both output requires an explicitly capable fusion model")
            if not any(combo <= set(request.modalities) for combo in fusion.combinations):
                raise ValueError("Fusion model supports no combination of the requested modalities")
            if request.fusion.get("method", fusion.method) != fusion.method:
                raise ValueError("Fusion method is not supported by the configured model")
            if "weights" in request.fusion and not fusion.supports_weights:
                raise ValueError("Fusion model does not support weights")

    def create(
        self,
        request: Mapping[str, Any],
        *,
        resources: Mapping[str, Any] | None = None,
        max_batch_bytes: int = 256 * 1024**2,
    ) -> VideoEmbeddingTask | PersistedVideoTask | dict[str, Any]:
        """Submit one asset through the ingestion plan and return its vector response."""
        request = copy.deepcopy(request)
        parsed = _VideoRequest.parse(request)
        self._validate(parsed)
        from vane.ai._video_ingestion import validate_execution

        resources = copy.deepcopy(dict(resources or {}))
        validate_execution(self._source, resources, max_batch_bytes)
        task_id, cancelled = uuid.uuid4().hex, threading.Event()
        deadline = None if parsed.asynchronous else time.monotonic() + parsed.timeout_seconds
        with self._lock:
            if self._closed:
                raise RuntimeError("Video embedding pipeline is closed")
            if len(self._tasks) >= self._max_tasks:
                raise RuntimeError("Video task capacity reached; forget completed tasks before submitting more")
            execute = self._execute
            if self._job_store is not None:
                self._job_store.submit(
                    task_id, self._pipeline_version, request, resources, max_batch_bytes, self._max_tasks
                )
                execute = self._execute_persisted
            future = self._executor.submit(execute, task_id, parsed, cancelled, deadline, resources, max_batch_bytes)
            task = VideoEmbeddingTask(task_id, future, cancelled)
            self._tasks[task_id] = task
        if parsed.asynchronous:
            return self.get_task(task_id)
        assert deadline is not None
        try:
            return task.result(timeout=max(0, deadline - time.monotonic()))
        except TimeoutError:
            # A provider's own TimeoutError is an execution error, not a wait timeout.
            if future.done():
                error = future.exception()
                if error is not None and not isinstance(error, VideoEmbeddingTimeout) and self._job_store is None:
                    self.forget_task(task_id)
                return task.result()
            self.get_task(task_id).cancel()
            task.cancel()
            raise VideoEmbeddingTimeout(task_id) from None
        except BaseException:
            # Synchronous failures do not expose a task ID. Do not consume the
            # retained-job budget with an inaccessible failed task.
            if future.done() and self._job_store is None:
                self.forget_task(task_id)
            elif not future.done():
                self.get_task(task_id).cancel()
                task.cancel()
            raise

    def get_task(self, task_id: str) -> VideoEmbeddingTask | PersistedVideoTask:
        if self._job_store is not None:
            self._job_store.get(task_id, self._pipeline_version)
            return PersistedVideoTask(task_id, self._job_store, self._pipeline_version)
        with self._lock:
            return self._tasks[task_id]

    def resume_task(self, task_id: str) -> PersistedVideoTask:
        """Explicitly rerun an interrupted asynchronous read, retaining its ID.

        Failed/cancelled/completed jobs are not resumable. Profiles and extractors
        are supplied again by the application under the same pipeline_version.
        No serialized Python code or live relation is loaded from storage.
        """
        if self._job_store is None:
            raise ValueError("resume_task requires a job_store")
        record = self._job_store.get(task_id, self._pipeline_version)
        if record["status"] not in {"pending", "interrupted"} or record["owner"] is not None:
            raise ValueError("Only unclaimed pending or interrupted jobs can be resumed")
        parsed = _VideoRequest.parse(record["request"])
        if not parsed.asynchronous:
            raise ValueError("Synchronous requests cannot be resumed; submit a new asynchronous request")
        self._validate(parsed)
        from vane.ai._video_ingestion import validate_execution

        validate_execution(self._source, record["resources"], record["max_batch_bytes"])
        with self._lock:
            if self._closed:
                raise RuntimeError("Video embedding pipeline is closed")
            previous = self._tasks.get(task_id)
            if previous is not None and not previous._future.done():
                raise ValueError("This process already has the task queued or running")
            self._job_store.requeue(task_id, self._pipeline_version)
            cancelled = threading.Event()
            future = self._executor.submit(
                self._execute_persisted,
                task_id,
                parsed,
                cancelled,
                None,
                record["resources"],
                record["max_batch_bytes"],
            )
            self._tasks[task_id] = VideoEmbeddingTask(task_id, future, cancelled)
        return PersistedVideoTask(task_id, self._job_store, self._pipeline_version)

    def ingest(self, asset: Mapping[str, Any], *, connection: Any, **options: Any) -> Any:
        """Build a single-asset ingestion relation using ``batch_ingest``.

        ``asset`` requires uri; asset_id and source_version may be omitted.
        All other options are the same as for ``batch_ingest``. Neither method
        executes the plan until a result consumer or sink is called.
        """
        return self.batch_ingest(_asset_relation(connection, asset), **options)

    def batch_ingest(
        self,
        assets: Any,
        *,
        modalities: Sequence[str],
        selection: Mapping[str, Any] | None = None,
        clip_seconds: float = 10,
        output: str = "separate",
        fusion: Mapping[str, Any] | None = None,
        return_timeline: bool = True,
        resources: Mapping[str, Any] | None = None,
        max_batch_bytes: int = 256 * 1024**2,
    ) -> Any:
        """Build one lazy Ray ingestion relation, ready for a final sink.

        ``assets`` requires VARCHAR uri; identity columns may be omitted or NULL.
        Call ``result.insert_into('namespace.main.video_clips')`` to submit one
        distributed write. Source identities must be unique within the input.
        """
        from vane.ai._video_ingestion import build_video_ingestion

        with self._lock:
            if self._closed:
                raise RuntimeError("Video embedding pipeline is closed")
        request = _VideoRequest.parse(
            {
                "input": {
                    "type": "video",
                    "uri": "relation://assets",
                    "asset_id": "relation",
                    "source_version": "relation",
                },
                "modalities": list(modalities),
                "selection": dict(selection or {}),
                "segmentation": {"clip_seconds": clip_seconds},
                "output": output,
                "fusion": dict(fusion or {}),
                "return_timeline": return_timeline,
                "async": True,
            }
        )
        self._validate(request)
        return build_video_ingestion(self, assets, request, dict(resources or {}), max_batch_bytes)

    def forget_task(self, task_id: str) -> None:
        with self._lock:
            if self._job_store is not None:
                task = self._tasks.get(task_id)
                if task is not None and not task._future.done():
                    raise ValueError("Cannot forget a task before execution and cleanup finish")
                self._job_store.forget(task_id, self._pipeline_version)
                self._tasks.pop(task_id, None)
                return
            task = self._tasks[task_id]
            if not task._future.done():
                raise ValueError("Cannot forget a task before execution and cleanup finish")
            del self._tasks[task_id]

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)

    def __enter__(self) -> VideoEmbeddingPipeline:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _execute_persisted(
        self,
        task_id: str,
        request: _VideoRequest,
        cancelled: threading.Event,
        deadline: float | None,
        resources: Mapping[str, Any],
        max_batch_bytes: int,
    ) -> dict[str, Any]:
        from vane.ai._errors import summarize_error

        store, owner = self._job_store, uuid.uuid4().hex
        assert store is not None
        store.claim(task_id, self._pipeline_version, owner)
        done = threading.Event()
        heartbeat_errors: list[Exception] = []

        def renew() -> None:
            while not done.wait(min(1, store.lease_seconds / 3)):
                try:
                    if store.renew(task_id, owner):
                        cancelled.set()
                except Exception as error:
                    heartbeat_errors.append(error)
                    cancelled.set()
                    return

        heartbeat = threading.Thread(target=renew, name="vane-video-lease", daemon=True)
        heartbeat.start()
        try:
            if store.renew(task_id, owner, processing=True):
                cancelled.set()
            result = self._execute(task_id, request, cancelled, deadline, resources, max_batch_bytes)
            if heartbeat_errors:
                raise heartbeat_errors[0]
            store.finish(task_id, owner, result=result, cancelled=cancelled.is_set())
            if store.get(task_id, self._pipeline_version)["status"] == "cancelled":
                raise CancelledError()
            return result
        except BaseException as error:
            # Store safe diagnostics, never arbitrary model responses or secrets.
            failure = heartbeat_errors[0] if heartbeat_errors and isinstance(error, CancelledError) else error
            if store.get(task_id, self._pipeline_version)["owner"] == owner:
                store.finish(
                    task_id,
                    owner,
                    error=summarize_error(failure),
                    cancelled=isinstance(failure, (CancelledError, VideoEmbeddingTimeout)),
                )
            if failure is not error:
                raise failure from None
            raise
        finally:
            done.set()
            heartbeat.join()

    def _execute(
        self,
        task_id: str,
        request: _VideoRequest,
        cancelled: threading.Event,
        deadline: float | None,
        resources: Mapping[str, Any],
        max_batch_bytes: int,
    ) -> dict[str, Any]:
        import vane
        from vane.ai._video_ingestion import build_video_ingestion

        def checkpoint() -> None:
            if cancelled.is_set():
                raise CancelledError()
            if deadline is not None and time.monotonic() >= deadline:
                cancelled.set()
                raise VideoEmbeddingTimeout(task_id)

        checkpoint()
        # Each task owns its connection. Interrupting it cannot cancel another
        # caller's query. Keep signalling until the query unwinds, including a
        # cancellation racing with plan binding or result-consumer startup.
        with vane.connect() as connection:
            finished = threading.Event()

            def interrupt() -> None:
                while not finished.wait(0.05):
                    if cancelled.is_set():
                        connection.interrupt()

            watcher = threading.Thread(target=interrupt, name="vane-video-interrupt", daemon=True)
            watcher.start()
            try:
                assets = _asset_relation(
                    connection,
                    {
                        key: value
                        for key, value in {
                            "asset_id": request.asset_id,
                            "source_version": request.source_version,
                            "uri": request.uri,
                        }.items()
                        if value is not None
                    },
                )
                relation = build_video_ingestion(self, assets, request, resources, max_batch_bytes)
                checkpoint()
                rows = relation.to_arrow_table().to_pylist()
                checkpoint()
                response = _response(task_id, rows, request)
                checkpoint()
                return response
            except VideoEmbeddingTimeout:
                raise
            except Exception:
                if cancelled.is_set():
                    raise CancelledError() from None
                raise
            finally:
                finished.set()
                watcher.join()


def _asset_relation(connection: Any, asset: Mapping[str, Any]) -> Any:
    fields = ("asset_id", "source_version", "uri")
    values = object_fields(asset, set(fields), "asset")
    if not isinstance(values.get("uri"), str) or not values["uri"].strip():
        raise ValueError("uri must be a nonempty string")
    if any(
        name in values and (not isinstance(values[name], str) or not values[name].strip())
        for name in ("asset_id", "source_version")
    ):
        raise ValueError("asset_id and source_version must be nonempty strings when supplied")
    # Bound values preserve arbitrary URI/identity text without SQL interpolation.
    return connection.sql(
        "SELECT ?::VARCHAR AS asset_id, ?::VARCHAR AS source_version, ?::VARCHAR AS uri",
        params=[values.get(name) for name in fields],
    )


def _response(task_id: str, rows: list[dict[str, Any]], request: _VideoRequest) -> dict[str, Any]:
    """Format plan output; selection, embedding and fusion live only in the plan."""
    response: dict[str, Any] = {"id": task_id, "status": "ready", "data": [], "clips": []}
    for row in sorted(rows, key=lambda row: row["start_sec"]):
        identity = {
            name: row[name] for name in ("clip_id", "asset_id", "source_version", "uri", "start_sec", "end_sec")
        }
        metadata = {channel: json.loads(row[f"{channel}_metadata"]) for channel in request.modalities}
        audit = {
            **identity,
            "channels": {
                channel: {key: value for key, value in meta.items() if key in {"status", "reason", "selected_modality"}}
                for channel, meta in metadata.items()
            },
        }
        if request.return_timeline:
            audit["timeline"] = [
                {"modality": channel, **item} for channel, meta in metadata.items() for item in meta["timeline"]
            ]
        response["clips"].append(audit)
        channels = list(request.modalities) if request.output in {"separate", "both"} else []
        if request.output in {"fused", "both"}:
            channels.append("fused")
            metadata["fused"] = json.loads(row["fused_metadata"])
        for channel in channels:
            vector, meta = row[f"{channel}_vector"], metadata[channel]
            if vector is not None:
                response["data"].append(
                    {
                        **identity,
                        "embedding": vector,
                        "modality": channel,
                        "modalities": meta["modalities"] if channel == "fused" else [channel],
                        **{name: meta[name] for name in ("profile", "provider", "model")},
                        **({"extraction": meta["extraction"]} if "extraction" in meta else {}),
                    }
                )
    return response
