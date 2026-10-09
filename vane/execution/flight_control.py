# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Flight framing and authentication for the remote session service."""

from __future__ import annotations

import hmac
import json
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.flight as flight

from vane.execution.query_options import QueryExecutionOptions
from vane.execution.query_runtime import QueryResources
from vane.execution.server_session import SessionError, SessionService

_ACTIONS = {
    "vane.query.execute": "Submit SQL once per consecutive session sequence",
    "vane.query.status": "Read query state and native result endpoint",
    "vane.query.cancel": "Cancel execution and retain the cleanup owner",
    "vane.query.finish": "Confirm native FINISH and complete delivery",
    "vane.query.close": "Release a query handle; poll until CLOSED",
    "vane.info": "Server instance and session capacity",
    "vane.session.open": "Open a leased SQL session",
    "vane.session.renew": "Renew an unexpired session lease",
    "vane.session.close": "Close a session; poll until CLOSED",
}
_MAX_BODY_BYTES = 65536


class _Authentication(flight.ServerMiddlewareFactory):
    def __init__(self, token: str) -> None:
        self._header = ("Bearer " + token).encode("ascii")

    def start_call(self, info: Any, headers: dict[str, list[Any]]) -> None:
        supplied = headers.get("authorization", [])
        if len(supplied) == 1:
            value = supplied[0]
            if isinstance(value, str):
                value = value.encode("utf-8")
            if isinstance(value, bytes) and hmac.compare_digest(value, self._header):
                return
        raise flight.FlightUnauthenticatedError("invalid server credentials")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate request field")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("non-finite request value")


def _request(body: Any, fields: set[str]) -> dict[str, Any]:
    if len(body) > _MAX_BODY_BYTES:
        raise ValueError("control request exceeds 65536 bytes")
    request = json.loads(body.to_pybytes(), object_pairs_hook=_object, parse_constant=_constant)
    if not isinstance(request, dict) or set(request) - fields - {"protocol"}:
        raise ValueError("unknown control request fields")
    if type(request.get("protocol")) is not int or request["protocol"] != 1:
        raise ValueError("unsupported control protocol; expected 1")
    return request


def _identity(request: dict[str, Any]) -> tuple[str, str]:
    for name in ("server_id", "session_id"):
        value = request.get(name)
        if not isinstance(value, str) or not value or len(value) > 128:
            raise ValueError(f"invalid {name}")
    return request["server_id"], request["session_id"]


class FlightControlServer(flight.FlightServerBase):
    def __init__(
        self,
        service: SessionService,
        location: Any,
        token: str,
        tls_certificates: list[tuple[bytes, bytes]] | None,
    ) -> None:
        self._service = service
        super().__init__(
            location=location,
            tls_certificates=tls_certificates,
            middleware={"authentication": _Authentication(token)},
        )

    def list_actions(self, context: Any) -> list[Any]:
        return [flight.ActionType(name, description) for name, description in _ACTIONS.items()]

    def do_action(self, context: Any, action: Any) -> Iterator[Any]:
        try:
            if context.is_cancelled():
                raise flight.FlightCancelledError("control request cancelled")
            if action.type == "vane.info":
                _request(action.body, set())
                result = self._service.snapshot()
                result["capabilities"] = ["sessions"] + (
                    ["queries", "native-results"] if self._service.gateway is not None else []
                )
            elif action.type == "vane.session.open":
                request = _request(action.body, {"execution", "resources"})
                values = request.get("resources")
                if values is not None and not isinstance(values, dict):
                    raise ValueError("resources must be an object")
                resources = None if values is None else QueryResources(**values)
                result = self._service.open_session(
                    execution=request.get("execution", "pipelined"), resources=resources
                )
                if context.is_cancelled():
                    self._service.close_session(result["server_id"], result["session_id"])
                    raise flight.FlightCancelledError("session open cancelled")
            elif action.type in ("vane.session.renew", "vane.session.close"):
                request = _request(action.body, {"server_id", "session_id"})
                identity = _identity(request)
                operation = (
                    self._service.renew_session if action.type == "vane.session.renew" else self._service.close_session
                )
                result = operation(*identity)
            elif action.type == "vane.query.execute":
                request = _request(
                    action.body, {"server_id", "session_id", "sequence", "sql", "options", "rows_per_batch"}
                )
                values = request.get("options")
                options = None if values is None else QueryExecutionOptions.from_dict(values)
                result = self._service.execute(
                    *_identity(request),
                    sequence=request.get("sequence", 0),
                    sql=request.get("sql", ""),
                    options=options,
                    rows_per_batch=request.get("rows_per_batch", 1024),
                )
            elif action.type in {"vane.query.status", "vane.query.cancel", "vane.query.finish", "vane.query.close"}:
                request = _request(action.body, {"server_id", "session_id", "query_id"})
                result = self._service.query_action(
                    *_identity(request), request.get("query_id", 0), action.type.rsplit(".", 1)[1]
                )
            else:
                raise ValueError("unknown control action")
            response = {"protocol": 1, "ok": True, "result": result}
        except SessionError as error:
            response = {"protocol": 1, "ok": False, "error": {"code": error.code, "message": str(error)}}
        except (ValueError, TypeError, RecursionError) as error:
            raise pa.ArrowInvalid(str(error)) from error
        encoded = json.dumps(response, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > _MAX_BODY_BYTES:
            raise pa.ArrowInvalid("control response exceeds 65536 bytes")
        yield flight.Result(encoded)
