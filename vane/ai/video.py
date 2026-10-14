# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Video asset orchestration: clip selection, independent outputs and local jobs.

Selection has no search AND/OR semantics. Retrieval combines the returned vectors
separately. Jobs and their IDs belong to this pipeline instance and process.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyarrow as pa

from vane.ai._audio_embedding import AudioClip
from vane.ai._video_embedding import VideoClip
from vane.ai._video_pipeline_types import (
    MODALITIES,
    TEXT_MODALITIES,
    ClipWindow,
    TimedContent,
    VideoMediaInfo,
    VideoSource,
    _VideoRequest,
)
from vane.ai._video_source import VaneVideoSource
from vane.ai.provider import _ProviderResultError, _safe_provider_execution_error

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
]


@dataclass(frozen=True)
class VideoEmbeddingProfile:
    """A named, reproducible configuration of an existing Vane descriptor.

    Providers run in the local worker for ``create()`` or distributed UDF workers
    for ``build_ingestion()``. The profile name should version model options
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
    60 seconds of video. Cancellation is checked between stages; an in-flight
    decoder/model call must return before its resources can be released.
    """

    def __init__(
        self,
        *,
        profiles: Mapping[str, VideoEmbeddingProfile],
        source: VideoSource | None = None,
        fusion: VideoFusionProfile | None = None,
        max_tasks: int = 128,
        max_clips: int = 10000,
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

    def create(self, request: Mapping[str, Any]) -> VideoEmbeddingTask | dict[str, Any]:
        parsed = _VideoRequest.parse(copy.deepcopy(request))
        self._validate(parsed)
        task_id, cancelled = uuid.uuid4().hex, threading.Event()
        deadline = None if parsed.asynchronous else time.monotonic() + parsed.timeout_seconds
        with self._lock:
            if self._closed:
                raise RuntimeError("Video embedding pipeline is closed")
            if len(self._tasks) >= self._max_tasks:
                raise RuntimeError("Video task capacity reached; forget completed tasks before submitting more")
            future = self._executor.submit(self._execute, task_id, parsed, cancelled, deadline)
            task = VideoEmbeddingTask(task_id, future, cancelled)
            self._tasks[task_id] = task
        if parsed.asynchronous:
            return task
        assert deadline is not None
        try:
            return task.result(timeout=max(0, deadline - time.monotonic()))
        except TimeoutError:
            # A provider's own TimeoutError is an execution error, not a wait timeout.
            if future.done():
                error = future.exception()
                if error is not None and not isinstance(error, VideoEmbeddingTimeout):
                    self.forget_task(task_id)
                return task.result()
            task.cancel()
            raise VideoEmbeddingTimeout(task_id) from None
        except BaseException:
            # Synchronous failures do not expose a task ID. Do not consume the
            # retained-job budget with an inaccessible failed task.
            if future.done():
                self.forget_task(task_id)
            else:
                task.cancel()
            raise

    def get_task(self, task_id: str) -> VideoEmbeddingTask:
        with self._lock:
            return self._tasks[task_id]

    def build_ingestion(
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

        ``assets`` requires VARCHAR asset_id, source_version and uri columns.
        Call ``result.insert_into('namespace.main.video_clips')`` to submit one
        distributed write. Source identities must be unique within the input.
        """
        from vane.ai._video_ingestion import build_video_ingestion

        with self._lock:
            if self._closed:
                raise RuntimeError("Video embedding pipeline is closed")
        request = _VideoRequest.parse(
            {
                "input": {"type": "video", "uri": "relation://assets"},
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

    def _execute(
        self, task_id: str, request: _VideoRequest, cancelled: threading.Event, deadline: float | None
    ) -> dict[str, Any]:
        def checkpoint() -> None:
            if cancelled.is_set():
                raise CancelledError()
            if deadline is not None and time.monotonic() >= deadline:
                cancelled.set()
                raise VideoEmbeddingTimeout(task_id)

        checkpoint()
        media = self._source.probe(request.uri)
        checkpoint()
        if not isinstance(media, VideoMediaInfo):
            raise TypeError("Video source probe must return VideoMediaInfo")
        if not request.asynchronous and media.duration > request.max_video_seconds:
            raise ValueError("Video exceeds sync_budget.max_video_seconds; submit with async=true")
        if media.duration > request.clip_seconds * self._max_clips:
            raise ValueError("Video segmentation exceeds the pipeline max_clips budget")
        count = math.ceil(media.duration / request.clip_seconds)
        response: dict[str, Any] = {"id": task_id, "status": "ready", "data": [], "clips": []}
        fusion = self._fusion
        loop = asyncio.new_event_loop()
        runtimes: dict[str, _ProfileRuntime] = {}
        try:
            with ExitStack() as cleanup:
                for index in range(count):
                    checkpoint()
                    clip = ClipWindow(
                        index * request.clip_seconds, min((index + 1) * request.clip_seconds, media.duration)
                    )
                    selected: dict[str, tuple[TimedContent, ...]] = {}
                    decisions: dict[str, dict[str, Any]] = {}
                    for selection in request.selections:
                        winner = None
                        for channel in selection.modalities:
                            checkpoint()
                            if winner is not None:
                                decisions[channel] = {
                                    "status": "skipped",
                                    "reason": "lower_priority",
                                    "selected_modality": winner,
                                }
                                continue
                            if channel not in media.available:
                                decisions[channel] = {"status": "skipped", "reason": "no_stream"}
                                continue
                            values = _content_in_clip(
                                self._source.extract(request.uri, channel, clip, media), channel, clip
                            )
                            checkpoint()
                            if not values:
                                decisions[channel] = {"status": "skipped", "reason": "no_content"}
                                continue
                            selected[channel] = values
                            decisions[channel] = {
                                "status": "selected",
                                "reason": "first_available" if selection.policy == "first_available" else "all",
                            }
                            if selection.policy == "first_available":
                                winner = channel
                    audit: dict[str, Any] = {
                        "start_sec": clip.start,
                        "end_sec": clip.end,
                        "channels": {c: decisions[c] for c in request.modalities},
                    }
                    if request.return_timeline:
                        audit["timeline"] = [
                            {
                                "modality": channel,
                                "start_sec": item.start,
                                "end_sec": item.end,
                                **({"text": item.value} if channel in TEXT_MODALITIES else {}),
                            }
                            for channel in request.modalities
                            for item in selected.get(channel, ())
                        ]
                    response["clips"].append(audit)
                    if selected and request.output in {"fused", "both"}:
                        assert fusion is not None
                        if frozenset(selected) not in fusion.combinations:
                            raise ValueError("Fusion model does not support the modalities available in this clip")
                    if request.output in {"separate", "both"}:
                        for channel in request.modalities:
                            if channel not in selected:
                                continue
                            checkpoint()
                            profile = self._profiles[channel]
                            if channel not in runtimes:
                                runtimes[channel] = _ProfileRuntime(profile, channel, loop)
                                cleanup.callback(runtimes[channel].close)
                            vector = runtimes[channel].embed(selected[channel])
                            checkpoint()
                            response["data"].append(
                                {
                                    "embedding": vector,
                                    "modality": channel,
                                    "modalities": [channel],
                                    "start_sec": clip.start,
                                    "end_sec": clip.end,
                                    "profile": profile.name,
                                    "provider": profile.descriptor.get_provider(),
                                    "model": profile.descriptor.get_model(),
                                }
                            )
                    if selected and request.output in {"fused", "both"}:
                        checkpoint()
                        assert fusion is not None
                        options = copy.deepcopy(request.fusion)
                        options.setdefault("method", fusion.method)
                        if "weights" in options:
                            options["weights"] = {c: w for c, w in options["weights"].items() if c in selected}
                        try:
                            vector = fusion.embed(clip, selected, options)
                            if inspect.isawaitable(vector):
                                vector = loop.run_until_complete(vector)
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
                        checkpoint()
                        response["data"].append(
                            {
                                "embedding": vector.tolist(),
                                "modality": "fused",
                                "modalities": [c for c in request.modalities if c in selected],
                                "start_sec": clip.start,
                                "end_sec": clip.end,
                                "profile": fusion.name,
                                "provider": fusion.provider,
                                "model": fusion.model,
                            }
                        )
        finally:
            loop.close()
        checkpoint()
        return response
