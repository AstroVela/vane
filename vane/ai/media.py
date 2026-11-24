# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Timestamped values for model inference, independent of ingest policies/jobs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MediaWindow:
    start: float
    end: float

    def __post_init__(self) -> None:
        if any(type(t) not in (int, float) or not math.isfinite(t) for t in (self.start, self.end)):
            raise ValueError("Media timestamps must be finite numbers")
        if not 0 <= self.start < self.end:
            raise ValueError("Media windows require 0 <= start < end")


@dataclass(frozen=True)
class TimedContent:
    start: float
    end: float
    value: Any

    def __post_init__(self) -> None:
        if any(type(t) not in (int, float) or not math.isfinite(t) for t in (self.start, self.end)):
            raise ValueError("Content timestamps must be finite numbers")
        if not 0 <= self.start <= self.end:
            raise ValueError("Content timestamps require 0 <= start <= end")
