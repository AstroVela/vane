# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Clip orchestration contracts, using real embedding wrappers without weights."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import CancelledError

import numpy as np
import pytest

from vane.ai import VideoEmbeddingPipeline, VideoEmbeddingProfile, VideoFusionProfile
from vane.ai._audio_embedding import AudioClip, AudioInputSpec
from vane.ai._video_embedding import VideoClip, VideoInputSpec
from vane.ai.protocols import AudioEmbedderDescriptor, TextEmbedderDescriptor, VideoEmbedderDescriptor
from vane.ai.video import ClipWindow, TimedContent, VideoEmbeddingTask, VideoEmbeddingTimeout, VideoMediaInfo

CHANNELS = frozenset({"visual", "audio", "subtitles", "transcription", "ocr"})


class Embedder:
    def __init__(self, calls, kind):
        self.calls, self.kind = calls, kind

    def embed_text(self, values):
        self.calls.extend(values)
        return [np.array([len(value), 2, 3], dtype=np.float32) for value in values]

    def embed_video(self, values):
        self.calls.extend(values)
        return [np.array([len(value.frames), 2, 3], dtype=np.float32) for value in values]

    def embed_audio(self, values):
        self.calls.extend(values)
        return [np.array([len(value.samples), 2, 3], dtype=np.float32) for value in values]


class Descriptor:
    def __init__(self):
        self.calls = []
        self.loaded = 0

    def get_provider(self):
        return "fixture"

    def get_model(self):
        return self.kind + "-v1"

    def get_options(self):
        return {}

    def get_dimensions(self):
        return 3

    def instantiate(self):
        self.loaded += 1
        return Embedder(self.calls, self.kind)


class TextDescriptor(Descriptor, TextEmbedderDescriptor):
    kind = "text"


class VisualDescriptor(Descriptor, VideoEmbedderDescriptor):
    kind = "video"

    def get_input_spec(self):
        return VideoInputSpec(max_frames=8)


class AudioDescriptor(Descriptor, AudioEmbedderDescriptor):
    kind = "audio"

    def get_input_spec(self):
        return AudioInputSpec(sample_rate=10, max_samples=100)


class Source:
    supported_modalities = CHANNELS

    def __init__(self, contents=None, *, duration=20, available=CHANNELS):
        self.contents = contents or {}
        self.media = VideoMediaInfo(duration, available)
        self.probes = []
        self.calls = []

    def probe(self, uri):
        self.probes.append(uri)
        return self.media

    def extract(self, uri, modality, clip, media):
        self.calls.append((clip.start, modality))
        value = self.contents.get((clip.start, modality), [])
        if isinstance(value, Exception):
            raise value
        return value


def request(channels, **kwargs):
    return {"input": {"type": "video", "uri": "s3://fixture/video.mp4"}, "modalities": channels, **kwargs}


def visual(start=0):
    frame = np.full((2, 2, 3), 7, dtype=np.uint8)
    return [TimedContent(start, start + 10, VideoClip((frame,), (start + 1,), (0,)))]


@pytest.fixture
def profiles():
    return {
        channel: VideoEmbeddingProfile(
            channel + "-v1",
            VisualDescriptor()
            if channel == "visual"
            else AudioDescriptor()
            if channel == "audio"
            else TextDescriptor(),
        )
        for channel in CHANNELS
    }


def test_priority_is_per_clip_and_does_not_embed_unselected_channels(profiles):
    source = Source(
        {(0, "transcription"): [TimedContent(1, 4, "speech")], (0, "visual"): visual(), (10, "visual"): visual(10)}
    )
    with VideoEmbeddingPipeline(source=source, profiles=profiles) as pipeline:
        result = pipeline.create(
            request(
                ["transcription", "visual", "audio"],
                selection={
                    "policy": "first_available",
                    "priority": ["transcription", "visual", "audio"],
                    "unit": "clip",
                },
            )
        )
    assert source.probes == ["s3://fixture/video.mp4"]
    assert source.calls == [(0, "transcription"), (10, "transcription"), (10, "visual")]
    assert [(v["modality"], v["start_sec"], v["end_sec"]) for v in result["data"]] == [
        ("transcription", 0, 10),
        ("visual", 10, 20),
    ]
    assert result["clips"][0]["channels"]["visual"] == {
        "status": "skipped",
        "reason": "lower_priority",
        "selected_modality": "transcription",
    }
    assert result["clips"][1]["channels"]["transcription"]["reason"] == "no_content"
    assert result["data"][0]["profile"] == "transcription-v1"
    assert result["data"][0]["model"] == "text-v1"
    assert result["data"][0]["provider"] == "fixture"
    assert profiles["audio"].descriptor.loaded == 0


def test_groups_keep_visual_and_only_fallback_between_subtitles_and_asr(profiles):
    source = Source(
        {
            (0, "visual"): visual(),
            (10, "visual"): visual(10),
            (0, "subtitles"): [TimedContent(2, 5, "captions")],
            (10, "transcription"): [TimedContent(11, 12, "speech")],
        }
    )
    selection = {
        "policy": "all",
        "groups": [
            {
                "modalities": ["subtitles", "transcription"],
                "policy": "first_available",
                "priority": ["subtitles", "transcription"],
            }
        ],
    }
    with VideoEmbeddingPipeline(source=source, profiles=profiles) as pipeline:
        result = pipeline.create(request(["visual", "subtitles", "transcription", "ocr", "audio"], selection=selection))
    assert [v["modality"] for v in result["data"]] == ["visual", "subtitles", "visual", "transcription"]
    assert (0, "transcription") not in source.calls
    assert profiles["visual"].descriptor.loaded == 1
    assert result["clips"][0]["channels"]["audio"]["reason"] == "no_content"


@pytest.mark.parametrize("policy", ["all", "first_available"])
def test_track_detection_is_shared_and_absence_does_not_call_extractors(profiles, policy):
    source = Source({(0, "visual"): visual()}, duration=10, available=frozenset({"visual", "ocr"}))
    with VideoEmbeddingPipeline(source=source, profiles=profiles) as pipeline:
        result = pipeline.create(request(["transcription", "audio", "visual"], selection={"policy": policy}))
    assert source.calls == [(0, "visual")]
    assert len(source.probes) == 1
    assert result["clips"][0]["channels"]["transcription"]["reason"] == "no_stream"
    assert result["clips"][0]["channels"]["audio"]["reason"] == "no_stream"


@pytest.mark.parametrize(
    "error", [TimeoutError("ASR deadline"), RuntimeError("decoder failed"), ValueError("model failed")]
)
def test_execution_errors_propagate_and_never_trigger_fallback(profiles, error):
    source = Source({(0, "transcription"): error, (0, "visual"): visual()})
    with VideoEmbeddingPipeline(source=source, profiles=profiles) as pipeline:
        with pytest.raises(type(error), match=str(error)):
            pipeline.create(request(["transcription", "visual"], selection={"policy": "first_available"}))
    assert source.calls == [(0, "transcription")]
    assert profiles["visual"].descriptor.loaded == 0


def test_text_uses_clip_overlap_half_open_points_and_strips_empty_content(profiles):
    text = [
        TimedContent(9, 11, "crossing"),
        TimedContent(10, 10, "boundary"),
        TimedContent(1, 2, " \n"),
        TimedContent(20, 25, "outside"),
    ]
    source = Source({(0, "subtitles"): text, (10, "subtitles"): text})
    with VideoEmbeddingPipeline(source=source, profiles=profiles) as pipeline:
        result = pipeline.create(request(["subtitles"]))
    assert profiles["subtitles"].descriptor.calls == ["crossing", "crossing\nboundary"]
    assert result["clips"][0]["timeline"] == [
        {"modality": "subtitles", "start_sec": 9, "end_sec": 10, "text": "crossing"}
    ]
    assert result["clips"][1]["timeline"][0]["start_sec"] == 10


def test_audio_uses_existing_typed_embedding_contract(profiles):
    source = Source(
        {(0, "audio"): [TimedContent(0, 10, AudioClip(np.ones((100, 1), dtype=np.float32), 10))]}, duration=10
    )
    with VideoEmbeddingPipeline(source=source, profiles=profiles) as pipeline:
        result = pipeline.create(request(["audio"]))
    assert result["data"][0]["embedding"] == [100, 2, 3]


@pytest.mark.parametrize("values", [None, ["untimed"], [TimedContent(0, 1, 42)]])
def test_invalid_extractor_results_are_errors(profiles, values):
    with VideoEmbeddingPipeline(source=Source({(0, "transcription"): values}), profiles=profiles) as pipeline:
        with pytest.raises(TypeError):
            pipeline.create(request(["transcription", "visual"], selection={"policy": "first_available"}))


@pytest.mark.parametrize("start,end", [(None, 1), (0, float("nan")), (float("inf"), 1), (True, 1), (-1, 1), (2, 1)])
def test_content_requires_valid_absolute_timestamps(start, end):
    with pytest.raises(ValueError):
        TimedContent(start, end, "content")


def fusion_profile(callback, combinations=(frozenset({"visual", "transcription"}),), **options):
    return VideoFusionProfile("joint-v1", "fixture", "joint-model-v1", 3, combinations, callback, **options)


@pytest.mark.parametrize("output,expected", [("fused", ["fused"]), ("both", ["visual", "transcription", "fused"])])
def test_joint_output_is_independent_of_selection_and_receives_aligned_inputs(profiles, output, expected):
    calls = []

    async def joint(clip, content, options):
        calls.append((clip, content, options))
        return [7, 8, 9]

    source = Source({(0, "visual"): visual(), (0, "transcription"): [TimedContent(1, 11, "speech")]}, duration=10)
    with VideoEmbeddingPipeline(source=source, profiles=profiles, fusion=fusion_profile(joint)) as pipeline:
        result = pipeline.create(request(["visual", "transcription"], output=output))
    assert [v["modality"] for v in result["data"]] == expected
    assert calls[0][0] == ClipWindow(0, 10)
    assert calls[0][1]["transcription"][0].end == 10
    assert calls[0][2] == {"method": "joint"}
    assert result["data"][-1]["modalities"] == ["visual", "transcription"]
    if output == "fused":
        assert all(profile.descriptor.loaded == 0 for profile in profiles.values())


def test_first_available_and_fused_can_combine_for_explicit_single_input_model(profiles):
    source = Source({(0, "transcription"): [TimedContent(0, 1, "text")]}, duration=10)
    fusion = fusion_profile(lambda *_: [1, 2, 3], (frozenset({"transcription"}),))
    with VideoEmbeddingPipeline(source=source, profiles={}, fusion=fusion) as pipeline:
        result = pipeline.create(
            request(["transcription", "visual"], selection={"policy": "first_available"}, output="fused")
        )
    assert result["data"][0]["modalities"] == ["transcription"]
    assert source.calls == [(0, "transcription")]


def test_missing_fusion_capability_rejects_equal_dimension_models_before_io(profiles):
    source = Source()
    with VideoEmbeddingPipeline(source=source, profiles=profiles) as pipeline:
        with pytest.raises(ValueError, match="fusion model"):
            pipeline.create(request(["visual", "audio"], output="both"))
    assert not source.probes


def test_fusion_missing_input_combination_fails_without_silently_averaging(profiles):
    source = Source({(0, "visual"): visual()}, duration=10)
    with VideoEmbeddingPipeline(
        source=source, profiles=profiles, fusion=fusion_profile(lambda *_: pytest.fail("must not fuse"))
    ) as pipeline:
        with pytest.raises(ValueError, match="available in this clip"):
            pipeline.create(request(["visual", "transcription"], output="both"))
    assert profiles["visual"].descriptor.loaded == 0


@pytest.mark.parametrize("vector", [[1, 2], [1, 2, float("nan")], [[1, 2, 3]], [1, 2, float("inf")]])
def test_fusion_validates_result_vectors(vector):
    source = Source({(0, "transcription"): [TimedContent(0, 1, "text")]}, duration=10)
    fusion = fusion_profile(lambda *_: vector, (frozenset({"transcription"}),))
    with VideoEmbeddingPipeline(source=source, profiles={}, fusion=fusion) as pipeline:
        with pytest.raises(TypeError, match="invalid embedding"):
            pipeline.create(request(["transcription"], output="fused"))


@pytest.mark.parametrize(
    "update",
    [
        {"selection": {"policy": "auto"}},
        {"selection": {"policy": "fuse"}},
        {"selection": {"policy": "all", "priority": ["visual"]}},
        {"selection": {"policy": "first_available", "priority": ["audio"]}},
        {"selection": {"policy": "all", "unit": "asset"}},
        {"selection": {"groups": [{"modalities": ["visual"]}, {"modalities": ["visual"]}]}},
        {"selection": {"groups": [{"modalities": ["audio"]}]}},
        {"selection": {"policy": "first_available", "groups": [{"modalities": ["visual"]}]}},
        {"segmentation": {"clip_seconds": 0}},
        {"segmentation": {"clip_seconds": True}},
        {"segmentation": {"clip_seconds": float("nan")}},
        {"modalities": ["visual", "visual"]},
        {"output": "unknown"},
        {"output": "separate", "fusion": {"method": "mean"}},
        {"async": "false"},
        {"return_timeline": 1},
        {"search": "AND"},
        {"sync_budget": {"timeout_seconds": float("inf")}},
    ],
)
def test_request_errors_are_detected_before_io(profiles, update):
    source = Source()
    with VideoEmbeddingPipeline(source=source, profiles=profiles) as pipeline:
        with pytest.raises((ValueError, TypeError)):
            pipeline.create(
                request(["visual"], **update) if "modalities" not in update else request(update["modalities"])
            )
    assert not source.probes


def test_empty_tail_clip_is_retained_and_timeline_can_be_omitted(profiles):
    source = Source({(0, "visual"): visual()}, duration=10.5)
    with VideoEmbeddingPipeline(source=source, profiles=profiles) as pipeline:
        result = pipeline.create(request(["visual"], return_timeline=False))
    assert len(result["data"]) == 1
    assert result["clips"][1]["end_sec"] == 10.5
    assert result["clips"][1]["channels"]["visual"]["reason"] == "no_content"
    assert all("timeline" not in clip for clip in result["clips"])
    assert result["data"][0]["end_sec"] == 10


def test_async_returns_id_before_probe_finishes_and_snapshots_request(profiles):
    entered, release = threading.Event(), threading.Event()
    source = Source(duration=70)
    original = source.probe

    def probe(uri):
        entered.set()
        assert release.wait(5)
        return original(uri)

    source.probe = probe
    pipeline = VideoEmbeddingPipeline(source=source, profiles=profiles)
    try:
        payload = request(["visual"], **{"async": True})
        task = pipeline.create(payload)
        assert isinstance(task, VideoEmbeddingTask) and task.id
        assert entered.wait(5)
        assert task.status == "processing"
        assert pipeline.get_task(task.id) is task
        payload["modalities"].append("audio")
        release.set()
        result = task.result(timeout=5)
        assert len(result["clips"]) == 7
        assert task.status == "ready"
        assert all(channel == "visual" for _, channel in source.calls)
        result["clips"].clear()
        assert len(task.result()["clips"]) == 7
        pipeline.forget_task(task.id)
        with pytest.raises(KeyError):
            pipeline.get_task(task.id)
    finally:
        release.set()
        pipeline.close()


def test_sync_duration_budget_rejects_before_extraction(profiles):
    source = Source(duration=61)
    with VideoEmbeddingPipeline(source=source, profiles=profiles) as pipeline:
        with pytest.raises(ValueError, match="async=true"):
            pipeline.create(request(["visual"]))
    assert not source.calls


def test_sync_time_budget_stops_new_work_and_retains_task_for_cleanup(profiles):
    release = threading.Event()
    source = Source()
    original = source.probe
    source.probe = lambda uri: (release.wait(5), original(uri))[1]
    pipeline = VideoEmbeddingPipeline(source=source, profiles=profiles)
    try:
        with pytest.raises(VideoEmbeddingTimeout) as error:
            pipeline.create(request(["visual"], sync_budget={"timeout_seconds": 0.02}))
        task = pipeline.get_task(error.value.task_id)
        assert task.status == "cancelling"
        release.set()
        with pytest.raises(CancelledError):
            task.result(timeout=5)
        assert task.status == "cancelled"
        assert not source.calls
    finally:
        release.set()
        pipeline.close()


def test_task_capacity_and_closure(profiles):
    pipeline = VideoEmbeddingPipeline(source=Source(), profiles=profiles, max_tasks=1)
    task = pipeline.create(request(["visual"], **{"async": True}))
    task.result(5)
    with pytest.raises(RuntimeError, match="capacity"):
        pipeline.create(request(["visual"]))
    pipeline.forget_task(task.id)
    pipeline.close()
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.create(request(["visual"]))


def test_clip_budget_prevents_unbounded_allocation(profiles):
    with VideoEmbeddingPipeline(source=Source(), profiles=profiles, max_clips=2) as pipeline:
        with pytest.raises(ValueError, match="max_clips"):
            pipeline.create(request(["visual"], segmentation={"clip_seconds": 1e-300}))


def test_asynchronous_failure_is_observable(profiles):
    with VideoEmbeddingPipeline(source=Source({(0, "visual"): RuntimeError("decode")}), profiles=profiles) as pipeline:
        task = pipeline.create(request(["visual"], **{"async": True}))
        with pytest.raises(RuntimeError, match="decode"):
            task.result(5)
        assert task.status == "failed"


def test_profile_from_provider_is_lazy_and_rejects_ignored_execution_options():
    profile = VideoEmbeddingProfile.from_provider(
        "cosmos-v1",
        modality="visual",
        provider="transformers",
        model="nvidia/Cosmos-Embed1-224p",
        revision="a" * 40,
        trust_remote_code=True,
        device="cuda",
        dtype="float16",
    )
    assert profile.descriptor.get_dimensions() > 0
    with pytest.raises(ValueError, match="execution options"):
        VideoEmbeddingProfile.from_provider("x", modality="visual", provider="transformers", actor_number=2)


def test_async_model_is_initialized_reused_and_closed_on_one_loop(profiles):
    events = []

    class AsyncModel:
        async def embed_text(self, values):
            events.append(("embed", asyncio.get_running_loop()))
            return [[1, 2, 3] for _ in values]

        async def aclose(self):
            events.append(("close", asyncio.get_running_loop()))

    class AsyncDescriptor(TextDescriptor):
        def is_async(self):
            return True

        def instantiate(self):
            events.append(("load", asyncio.get_running_loop()))
            return AsyncModel()

    profiles["transcription"] = VideoEmbeddingProfile("async-v1", AsyncDescriptor())
    source = Source({(start, "transcription"): [TimedContent(start, start + 1, "speech")] for start in (0, 10)})
    with VideoEmbeddingPipeline(source=source, profiles=profiles) as pipeline:
        pipeline.create(request(["transcription"]))
    assert [event for event, _ in events] == ["load", "embed", "embed", "close"]
    assert len({id(loop) for _, loop in events}) == 1
    assert events[0][1].is_closed()


def test_bad_embedding_does_not_advance_priority(profiles):
    class InvalidModel:
        def embed_text(self, values):
            return [[float("nan"), 2, 3]]

    profiles["transcription"].descriptor.instantiate = lambda: InvalidModel()
    source = Source({(0, "transcription"): [TimedContent(0, 1, "speech")], (0, "visual"): visual()}, duration=10)
    with VideoEmbeddingPipeline(source=source, profiles=profiles) as pipeline:
        with pytest.raises(TypeError, match="non-finite"):
            pipeline.create(request(["transcription", "visual"], selection={"policy": "first_available"}))
    assert source.calls == [(0, "transcription")]


def test_fusion_weights_are_opt_in_and_only_selected_weights_are_passed():
    calls = []
    source = Source({(0, "transcription"): [TimedContent(0, 1, "speech")]}, duration=10)
    fusion = fusion_profile(
        lambda _clip, _content, options: calls.append(options) or [1, 2, 3],
        (frozenset({"transcription"}),),
        supports_weights=True,
    )
    payload = request(
        ["transcription", "visual"], output="fused", fusion={"weights": {"transcription": 2, "visual": 1}}
    )
    with VideoEmbeddingPipeline(source=source, profiles={}, fusion=fusion) as pipeline:
        pipeline.create(payload)
    assert calls == [{"method": "joint", "weights": {"transcription": 2.0}}]
    assert payload["fusion"]["weights"] == {"transcription": 2, "visual": 1}
    unsupported = fusion_profile(lambda *_: pytest.fail("must not fuse"), (frozenset({"transcription"}),))
    with VideoEmbeddingPipeline(source=source, profiles={}, fusion=unsupported) as pipeline:
        with pytest.raises(ValueError, match="weights"):
            pipeline.create(payload)


def test_cancelling_queued_task_does_not_extract_or_load_models(profiles):
    entered, release = threading.Event(), threading.Event()
    source = Source(duration=10)
    original = source.probe

    def probe(uri):
        entered.set()
        assert release.wait(5)
        return original(uri)

    source.probe = probe
    pipeline = VideoEmbeddingPipeline(source=source, profiles=profiles)
    try:
        first = pipeline.create(request(["visual"], **{"async": True}))
        assert entered.wait(5)
        second = pipeline.create(request(["audio"], **{"async": True}))
        assert second.status == "pending"
        with pytest.raises(ValueError, match="cleanup"):
            pipeline.forget_task(first.id)
        second.cancel()
        with pytest.raises(CancelledError):
            second.result()
        assert second.status == "cancelled"
        release.set()
        first.result(5)
        assert source.calls == [(0, "visual")]
    finally:
        release.set()
        pipeline.close()
