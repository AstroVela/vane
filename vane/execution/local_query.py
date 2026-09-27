# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Session-owned admission for the existing native SQL and Relation entry points."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from vane.execution.request_admission import RequestAdmissionLimits, _timeout
from vane.execution.resources import ResourceVector
from vane.execution.udf_data_admission import DataAdmissionLimits
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_local_request import LocalModelRequest, _NativeRequestCancellation
from vane.execution.udf_runtime_admission import TaskAdmissionLimits

if TYPE_CHECKING:
    from vane.execution.local_model import LocalQueryModel


class _NativePlan:
    """A preparation snapshot; never retain borrowed native plan pointers."""

    def __init__(self, runtime: LocalModelRuntime, nodes: list[dict[str, Any]], graph: dict[str, Any] | None) -> None:
        self._runtime = runtime
        self._nodes = nodes
        self._graph = graph
        self.handles: dict[str, Any] = {}

    def session_id(self) -> str:
        return self._runtime._session_id

    def session_config(self) -> dict[str, str]:
        return dict(self._runtime._session_config)

    def collect_udf_nodes(self, *, conn: Any = None) -> list[dict[str, Any]]:
        return self._nodes

    def collect_resource_graph_metadata(self, *, conn: Any = None, annotate_udfs: bool = False) -> dict[str, Any]:
        assert not annotate_udfs and self._graph is not None
        return self._graph

    def set_udf_actor_handles(self, handles: dict[str, Any], *, conn: Any = None) -> None:
        self.handles = handles


class _NativeQuery:
    def __init__(self, request: LocalModelRequest) -> None:
        self.request = request
        self.track_graph = request._runtime._track_graph
        self._binding = _NativeRequestCancellation(request._cancellation)
        self._prepared = False

    def prepare(self, nodes: list[dict[str, Any]], graph: dict[str, Any] | None) -> dict[str, Any]:
        if self._prepared:
            raise RuntimeError("a native runtime request can prepare only one query")
        self._prepared = True
        plan = _NativePlan(self.request._runtime, nodes, graph)
        bindings = {}
        for node in nodes:
            payload = node["payload"]
            if "local_model_token" in payload:
                if payload.get("local_model_session_id") != self.request._runtime._session_id:
                    raise ValueError("registered local model belongs to a different Vane session")
                bindings[str(node["node_id"])] = payload["local_model_name"]
        self.request._prepare_execution(plan, bindings)
        return plan.handles

    def started(self, interrupt: Callable[[], None]) -> None:
        self._binding.started_callback(interrupt)
        self.request._check_cancelled()

    def close(self) -> None:
        self._binding.close()


class LocalQueryRuntime:
    """Admission shared by a local-fast connection and all its cursors.

    Create through ``connection.configure_local_runtime``. Configuration is
    fixed for the session. Query results retain their normal Python API and
    native materialization; the byte budget covers UDF shared-memory ownership.
    """

    def __init__(
        self,
        *,
        session_id: str,
        session_config: Mapping[str, Any],
        request_limit: RequestAdmissionLimits,
        resident_limit: ResourceVector | None = None,
        task_limit: TaskAdmissionLimits | None = None,
        data_limit: DataAdmissionLimits | None = None,
        track_data: bool = False,
        track_graph: bool = False,
        execution_timeout: float | None = None,
        _connection: Any = None,
    ) -> None:
        if not isinstance(request_limit, RequestAdmissionLimits):
            raise TypeError("request_limit must be RequestAdmissionLimits")
        self._execution_timeout = (
            None if execution_timeout is None else _timeout(execution_timeout, "execution_timeout")
        )
        self._runtime = LocalModelRuntime(
            session_id=session_id,
            session_config=session_config,
            request_limit=request_limit,
            resident_limit=resident_limit,
            task_limit=task_limit,
            data_limit=data_limit,
            track_data=track_data,
            track_graph=track_graph,
        )
        self._connection = weakref.ref(_connection) if _connection is not None else lambda: None

    def register_model(
        self,
        name: str,
        model: Any,
        *,
        version: str,
        parameters: Any,
        cpus: float = 1.0,
        memory_bytes: int | None = None,
    ) -> LocalQueryModel:
        """Freeze a CPU class UDF and explicitly register it for cross-query reuse.

        The returned callable builds Relation expressions and can be passed to
        ``vane.attach_function`` without repeating its registered parameters.
        """
        from vane import _native
        from vane.execution.local_model import LocalQueryModel, collect_model_payload, prepare_model_definition

        _native._check_python_callback_entry()
        assert self._runtime._request_admission is not None
        self._runtime._request_admission.require_open()
        connection = self._connection()
        if connection is None:
            raise RuntimeError("the local runtime's owning connection is unavailable")
        definition, unnest = prepare_model_definition(
            model,
            session_id=self._runtime._session_id,
            session_config=self._runtime._session_config,
            name=name,
            version=version,
            parameters=parameters,
            cpus=cpus,
            memory_bytes=memory_bytes,
        )
        payload = collect_model_payload(connection, definition)
        registered = self._runtime.register(name, version=version, payload=payload)
        return LocalQueryModel(definition, registered, weakref.ref(connection), unnest)

    def _execute(
        self, execute: Callable[[_NativeQuery], None], publish: Callable[[LocalModelRequest | None], None]
    ) -> None:
        request = self._runtime.request()
        publish(request)
        try:

            def run() -> None:
                query = _NativeQuery(request)
                try:
                    execute(query)
                finally:
                    query.close()

            request._run_execution(run, execution_timeout=self._execution_timeout)
        finally:
            publish(None)

    def resource_snapshot(self) -> dict[str, Any]:
        return self._runtime.resource_snapshot()

    def drain(self) -> None:
        self._runtime.drain()

    def close(self, *, timeout: float = 0.0, kill: bool = False) -> None:
        self._runtime.close(timeout=timeout, kill=kill)
