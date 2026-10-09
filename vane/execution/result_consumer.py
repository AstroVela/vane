# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native result attachment for embedded consumers and the public gateway."""

from __future__ import annotations

import base64
import json
import secrets
from dataclasses import asdict
from typing import Any


class NativeResultConsumer:
    def __init__(self, gateway: Any = None) -> None:
        self.gateway = gateway
        self.flight: Any = None
        self.channel: Any = None
        self.ticket = ""
        self.descriptor: dict[str, Any] = {}

    def attach(self, spec: Any, resources: Any, location: str, ticket: str) -> None:
        from vane._native import execution_runtime as native
        from vane.execution.pipelined_worker import _channel

        if self.channel is not None:
            raise RuntimeError("result consumer already attached")
        self.channel = _channel(spec.result_schema, resources, "result-service", "client")
        self.flight = native.DirectFlight(
            "127.0.0.1",
            "127.0.0.1",
            1,
            native.DirectFlight.staging_per_link(resources.exchange.frame_bytes),
            resources.exchange.frame_bytes,
        )
        self.flight.subscribe(
            location,
            ticket,
            self.channel,
            "result-service",
            spec.options.admission_timeout + spec.options.execution_timeout + spec.options.delivery_timeout,
        )
        if self.gateway is not None:
            # Exact matching is the authorization check. Neither the internal
            # worker address nor its capability leaves the server.
            self.ticket = secrets.token_urlsafe(48)
            self.gateway.publish(self.ticket, self.channel, "client")
            self.descriptor = {
                "location": self.gateway.location,
                "ticket": self.ticket,
                "schema": base64.b64encode(spec.result_schema).decode("ascii"),
                "names": list(spec.result_names),
                "engine": spec.graph.engine_identity,
                "limits": asdict(resources.exchange),
                "timeout": spec.options.execution_timeout + spec.options.delivery_timeout,
            }
            if len(json.dumps(self.descriptor).encode("utf-8")) > 60000:
                raise ValueError("result schema exceeds the remote metadata limit")

    @property
    def error(self) -> str:
        return "" if self.flight is None else str(self.flight.error)

    @property
    def ready(self) -> bool:
        return self.flight is not None and bool(self.flight.ready)

    def delivered(self) -> bool:
        if self.gateway is None or not self.ticket:
            raise RuntimeError("result has no external consumer")
        return bool(self.gateway.delivered(self.ticket))

    def cancel(self, reason: str) -> None:
        if self.flight is not None:
            self.flight.cancel(reason)

    def close(self) -> None:
        if self.ticket:
            self.gateway.revoke(self.ticket)
            self.ticket = ""
        if self.flight is not None:
            self.flight.close()
