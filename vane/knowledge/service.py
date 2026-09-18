# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Polling and delivery share one exclusive writer and a durable pending batch."""

from __future__ import annotations

from vane.knowledge.config import Config
from vane.knowledge.iceberg import IcebergSource
from vane.knowledge.state import State
from vane.knowledge.targets import deliver


def sync_once(config: Config, state: State, source: IcebergSource | None = None) -> bool:
    """Run under State.exclusive(); replay durable delivery before contacting the lake."""
    state.bind(config)
    if state.pending() is None:
        source = source or IcebergSource(config)
        snapshot = source.discover(state.checkpoint())
        state.stage(snapshot, source.documents(snapshot), tuple(t.name for t in config.targets))
    if state.pending() is None:
        return False
    deliver(state, config.targets, config.timeout_seconds)
    return True
