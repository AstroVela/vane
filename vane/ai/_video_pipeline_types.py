# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Clip selection contracts shared by video sources and orchestration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

MODALITIES = frozenset({"visual", "audio", "subtitles", "transcription", "ocr"})
TEXT_MODALITIES = frozenset({"subtitles", "transcription", "ocr"})


def positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def object_fields(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")
    return dict(value)


def modalities(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty list of modalities")
    if any(not isinstance(item, str) or item not in MODALITIES for item in value):
        raise ValueError(f"{name} accepts only: {', '.join(sorted(MODALITIES))}")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} cannot contain duplicate modalities")
    return tuple(value)


@dataclass(frozen=True)
class ClipWindow:
    """An absolute, half-open interval on the asset's common timeline."""

    start: float
    end: float

    def __post_init__(self) -> None:
        for value in (self.start, self.end):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Clip timestamps must be finite numbers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Clip timestamps require 0 <= start < end")


@dataclass(frozen=True)
class TimedContent:
    """Text, VideoClip or AudioClip, with absolute asset timestamps.

    Equal endpoints describe a point event. Text spanning a clip boundary is
    included in both clips; decoded media must already be windowed by its source.
    """

    start: float
    end: float
    value: Any

    def __post_init__(self) -> None:
        for value in (self.start, self.end):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Content timestamps must be finite numbers")
        if self.start < 0 or self.end < self.start:
            raise ValueError("Content timestamps require 0 <= start <= end")


@dataclass(frozen=True)
class VideoMediaInfo:
    """Container detection, performed once before either selection policy.

    ``available`` describes physical tracks, not whether speech/OCR will produce
    content. A source must include transcription when audio is present and OCR
    when video is present. ``origin`` is the container timestamp for asset zero.
    """

    duration: float
    available: frozenset[str]
    origin: float = 0.0

    def __post_init__(self) -> None:
        positive_number(self.duration, "media duration")
        if not isinstance(self.available, frozenset) or not self.available <= MODALITIES:
            raise ValueError("available must be a frozenset of known modalities")
        if isinstance(self.origin, bool) or not isinstance(self.origin, (int, float)) or not math.isfinite(self.origin):
            raise ValueError("media origin must be finite")


class VideoSource(Protocol):
    """A source owns decoding and extraction; normal absence returns an empty list.

    Source methods run serially within each worker instance. Errors must propagate.
    ``extract`` must use the probe's common timeline and bounded clip inputs.
    """

    supported_modalities: frozenset[str]

    def probe(self, uri: str) -> VideoMediaInfo: ...

    def extract(self, uri: str, modality: str, clip: ClipWindow, media: VideoMediaInfo) -> Sequence[TimedContent]: ...


@dataclass(frozen=True)
class _Selection:
    policy: str
    modalities: tuple[str, ...]


def _selection(value: Any, channels: tuple[str, ...], *, group: bool = False) -> _Selection:
    allowed = {"policy", "priority", "modalities"} if group else {"policy", "priority", "unit", "groups"}
    value = object_fields(value, allowed, "selection group" if group else "selection")
    policy = value.get("policy", "all")
    if policy not in ("first_available", "all"):
        raise ValueError("selection.policy must be first_available or all; auto has no defined policy")
    if not group and value.get("unit", "clip") != "clip":
        raise ValueError("selection.unit must be clip")
    if policy == "all":
        if "priority" in value:
            raise ValueError("priority is only valid with first_available")
        return _Selection(policy, channels)
    priority = modalities(value.get("priority", list(channels)), "priority")
    if set(priority) != set(channels):
        raise ValueError("priority must list every selected modality exactly once")
    return _Selection(policy, priority)


@dataclass(frozen=True)
class _VideoRequest:
    uri: str
    modalities: tuple[str, ...]
    selections: tuple[_Selection, ...]
    clip_seconds: float
    output: str
    fusion: dict[str, Any]
    return_timeline: bool
    asynchronous: bool
    timeout_seconds: float
    max_video_seconds: float

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> _VideoRequest:
        value = object_fields(
            value,
            {
                "input",
                "modalities",
                "selection",
                "segmentation",
                "output",
                "fusion",
                "return_timeline",
                "async",
                "sync_budget",
            },
            "video request",
        )
        source = object_fields(value.get("input"), {"type", "uri"}, "input")
        if source.get("type") != "video":
            raise ValueError("input.type must be video")
        uri = source.get("uri")
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError("input.uri must be a nonempty string")
        channels = modalities(value.get("modalities"), "modalities")
        selection = value.get("selection", {})
        outer = _selection(selection, channels)
        selections = []
        grouped: set[str] = set()
        groups = selection.get("groups", [])
        if not isinstance(groups, list):
            raise TypeError("selection.groups must be a list")
        if groups and outer.policy != "all":
            raise ValueError("selection.groups requires the outer all policy")
        for group in groups:
            if not isinstance(group, Mapping):
                raise TypeError("selection.groups must contain objects")
            members = modalities(group.get("modalities"), "group.modalities")
            if not set(members) <= set(channels) or grouped.intersection(members):
                raise ValueError("selection groups must be disjoint subsets of modalities")
            selections.append(_selection(group, members, group=True))
            grouped.update(members)
        if groups:
            selections.extend(_Selection("all", (channel,)) for channel in channels if channel not in grouped)
        else:
            selections.append(outer)
        segmentation = object_fields(value.get("segmentation", {}), {"clip_seconds"}, "segmentation")
        output = value.get("output", "separate")
        if output not in ("separate", "fused", "both"):
            raise ValueError("output must be separate, fused or both")
        fusion = object_fields(value.get("fusion", {}), {"method", "weights"}, "fusion")
        if fusion and output == "separate":
            raise ValueError("fusion requires fused or both output")
        if "method" in fusion and (not isinstance(fusion["method"], str) or not fusion["method"].strip()):
            raise ValueError("fusion.method must be a nonempty string")
        if "weights" in fusion:
            weights = object_fields(fusion["weights"], set(channels), "fusion.weights")
            if set(weights) != set(channels):
                raise ValueError("fusion.weights must cover all requested modalities")
            fusion["weights"] = {
                channel: positive_number(weight, "fusion weight") for channel, weight in weights.items()
            }
        budget = object_fields(value.get("sync_budget", {}), {"timeout_seconds", "max_video_seconds"}, "sync_budget")
        for field in ("return_timeline", "async"):
            if field in value and type(value[field]) is not bool:
                raise TypeError(f"{field} must be a bool")
        return cls(
            uri,
            channels,
            tuple(selections),
            positive_number(segmentation.get("clip_seconds", 10), "clip_seconds"),
            output,
            fusion,
            value.get("return_timeline", True),
            value.get("async", False),
            positive_number(budget.get("timeout_seconds", 60), "sync_budget.timeout_seconds"),
            positive_number(budget.get("max_video_seconds", 60), "sync_budget.max_video_seconds"),
        )
