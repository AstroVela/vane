# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Session-owned local models with explicit physical-plan bindings.

These are internal execution APIs. They do not install a process-global cache
or change the lifetime of unregistered class UDFs.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from vane import pickle as vane_pickle
from vane.execution.request_admission import RequestAdmissionLimits, RequestTicket, RuntimeRequestAdmission
from vane.execution.resources import ResourceVector, udf_process_resources
from vane.execution.udf_actor_pool_lifecycle import (
    OwnedActorPoolsError,
    actor_pool_cleanup_pending,
    rollback_actor_pools,
)
from vane.execution.udf_data_admission import DataAdmissionLimits
from vane.execution.udf_data_lease import QueryDataScope, RuntimeDataLedger
from vane.execution.udf_input_cleanup import QueryInputCleanup
from vane.execution.udf_model_pool import ModelPoolBorrow, ModelPoolIdentity, ModelPoolRegistry
from vane.execution.udf_runtime_admission import QueryTaskAdmission, RuntimeTaskAdmission, TaskAdmissionLimits

if TYPE_CHECKING:
    from vane.execution.udf_local_request import LocalModelRequest
    from vane.execution.udf_subprocess import LocalSubprocessActorPool


def _payload_bytes(payload: Mapping[str, Any]) -> bytes:
    # Preserve the complete payload used to initialize workers.
    return vane_pickle.dumps(dict(sorted(payload.items())))


def _model_fingerprint(payload: Mapping[str, Any]) -> str:
    # SQL binding assigns a fresh expression_id to each call. It identifies a
    # query expression, not the model. Keep exact matching for initialization,
    # schema, device and execution settings, including any unknown fields.
    compatible_payload = {key: value for key, value in payload.items() if key != "expression_id"}
    return hashlib.sha256(_payload_bytes(compatible_payload)).hexdigest()


@dataclass(frozen=True)
class RegisteredLocalModel:
    identity: ModelPoolIdentity
    pool_size: int
    resident_resources: ResourceVector
    _registry: ModelPoolRegistry[LocalSubprocessActorPool] = field(repr=False)
    _session_config: tuple[tuple[str, str], ...] = field(repr=False)
    _request_admission: RuntimeRequestAdmission | None = field(default=None, repr=False)
    _request_ticket: RequestTicket | None = field(default=None, repr=False)

    def validate(self, payload: Mapping[str, Any], pool_size: int, session_config: Mapping[str, Any] | None) -> None:
        if session_config is None or tuple(sorted(session_config.items())) != self._session_config:
            raise ValueError("registered local model belongs to a different Vane session configuration")
        if pool_size != self.pool_size or _model_fingerprint(payload) != self.identity.initialization:
            raise ValueError("registered local model payload or pool size does not match the UDF node")

    def _require_admission(self) -> None:
        if self._request_admission is not None:
            if self._request_ticket is None:
                self._request_admission.require_open()
            else:
                self._request_admission.require_claimed(self._request_ticket)

    def acquire(self) -> ModelPoolBorrow[LocalSubprocessActorPool]:
        self._require_admission()
        borrow = self._registry.acquire(self.identity)
        try:
            # Initialization can cross drain. Keep the pool owned by the
            # registry, but do not publish a new public borrow afterward.
            self._require_admission()
        except BaseException:
            borrow.release()
            raise
        return borrow

    def prewarm(self) -> None:
        with self.acquire():
            pass


class LocalModelRuntime:
    """Own CPU subprocess models for one explicitly identified Vane session.

    Register from a collected UDF payload, optionally prewarm, then prepare a
    physical plan with explicit model/node bindings. Preparation acquires borrows
    through the existing local_actor_pool path. Runtime close is explicit and
    must run after query executors have finished and released their borrows.
    """

    def __init__(
        self,
        *,
        session_id: str,
        session_config: Mapping[str, Any],
        resident_limit: ResourceVector | None = None,
        task_limit: TaskAdmissionLimits | None = None,
        track_data: bool = False,
        data_limit: DataAdmissionLimits | None = None,
        request_limit: RequestAdmissionLimits | None = None,
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("local model runtime requires a non-empty session_id")
        if resident_limit is not None and (resident_limit.gpu or resident_limit.object_store_bytes):
            raise ValueError("local resident limits support CPU and declared heap only")
        if type(track_data) is not bool:
            raise TypeError("track_data must be a bool")
        self._session_id = session_id
        self._session_config = {str(key): str(value) for key, value in session_config.items()}
        self._registry: ModelPoolRegistry[LocalSubprocessActorPool] = ModelPoolRegistry(resident_limit=resident_limit)
        self._task_admission = RuntimeTaskAdmission(task_limit) if task_limit is not None else None
        self._data_ledger = RuntimeDataLedger(data_limit) if track_data or data_limit is not None else None
        self._request_admission = RuntimeRequestAdmission(request_limit) if request_limit is not None else None
        self._request_cleanup: set[LocalModelRequest] = set()
        self._models: dict[str, RegisteredLocalModel] = {}
        self._lock = threading.Lock()

    def register(self, name: str, *, version: str, payload: Mapping[str, Any]) -> RegisteredLocalModel:
        from vane.execution.udf_subprocess import LocalSubprocessActorPool, _local_actor_pool_size_from_node

        if self._request_admission is not None:
            self._request_admission.require_open()
        if str(payload.get("execution_backend") or "").strip().lower() != "subprocess_actor":
            raise ValueError("local model registration requires a subprocess_actor UDF")
        frozen_payload = _payload_bytes(payload)
        snapshot = vane_pickle.loads(frozen_payload)
        per_actor = udf_process_resources(snapshot)
        if per_actor.gpu > 0.0:
            raise ValueError("GPU resources require a Ray UDF backend")
        pool_size = _local_actor_pool_size_from_node({}, snapshot)
        resources = ResourceVector(
            cpu=per_actor.cpu * pool_size,
            heap_bytes=per_actor.heap_bytes * pool_size,
        )
        config = dict(self._session_config)
        identity = ModelPoolIdentity(
            session_id=self._session_id,
            model=name,
            version=version,
            backend="subprocess_actor",
            initialization=_model_fingerprint(snapshot),
            configuration=hashlib.sha256(vane_pickle.dumps((pool_size, tuple(sorted(config.items()))))).hexdigest(),
        )

        def create() -> LocalSubprocessActorPool:
            return LocalSubprocessActorPool(
                vane_pickle.loads(frozen_payload), pool_size, name=f"model-{name}-{version}", session_config=config
            )

        model = RegisteredLocalModel(
            identity, pool_size, resources, self._registry, tuple(sorted(config.items())), self._request_admission
        )
        with self._lock:
            if name in self._models:
                raise ValueError(f"local model {name!r} is already registered; use a distinct name for another version")
            self._registry.register(identity, create, resources=resources)
            self._models[name] = model
        return model

    def prepare(
        self, plan: Any, bindings: Mapping[str, str], *, conn: Any = None
    ) -> list[
        LocalSubprocessActorPool
        | ModelPoolBorrow[LocalSubprocessActorPool]
        | QueryTaskAdmission
        | QueryDataScope
        | QueryInputCleanup
    ]:
        """Validate bindings, acquire query resources, and publish their handles.

        Call once per execution and retain the returned resources until all query
        executors have finished. Their shutdown releases only the query's owners.
        Different sessions are rejected even when their configurations match.
        """
        if self._request_admission is not None:
            raise RuntimeError("request-limited runtimes require request().execute()")
        return self._prepare(plan, bindings, conn=conn)

    def request(self, *, queue_timeout: float | None = None) -> LocalModelRequest:
        """Queue lightweight request metadata before borrowing execution resources."""
        from vane.execution.udf_local_request import LocalModelRequest

        if self._request_admission is None:
            raise RuntimeError("local requests require a configured request_limit")
        return LocalModelRequest(self, self._request_admission.request(queue_timeout=queue_timeout))

    def _retain_request_cleanup(self, request: LocalModelRequest, *, pending: bool) -> None:
        with self._lock:
            if pending:
                self._request_cleanup.add(request)
            else:
                self._request_cleanup.discard(request)

    def _prepare(
        self, plan: Any, bindings: Mapping[str, str], *, conn: Any = None, request_ticket: RequestTicket | None = None
    ) -> list[
        LocalSubprocessActorPool
        | ModelPoolBorrow[LocalSubprocessActorPool]
        | QueryTaskAdmission
        | QueryDataScope
        | QueryInputCleanup
    ]:
        from vane.execution.ref_bundle import payload_requests_local_ref_bundle_output
        from vane.execution.udf_subprocess import (
            _local_actor_pool_size_from_node,
            ensure_local_subprocess_actor_pools_for_nodes,
        )

        if self._request_admission is not None:
            self._request_admission.require_claimed(request_ticket)
        if (
            not bindings
            and self._task_admission is None
            and self._data_ledger is None
            and self._request_admission is None
        ):
            raise ValueError("local model preparation requires explicit model bindings")
        if plan.session_id() != self._session_id or plan.session_config() != self._session_config:
            raise ValueError("local model runtime belongs to a different Vane session")
        nodes = {str(node["node_id"]): dict(node) for node in plan.collect_udf_nodes(conn=conn)}
        unknown = set(bindings) - set(nodes)
        if unknown:
            raise ValueError(f"unknown model UDF node IDs: {sorted(unknown)}")
        # Session isolation applies to every UDF, including unregistered actors
        # and tasks. Copy options so validation cannot mutate the original plan.
        executor_options_by_node = {}
        for node_id, node in nodes.items():
            backend = str(node["payload"].get("execution_backend") or "").strip().lower()
            if (
                self._task_admission is not None or self._data_ledger is not None or self._request_admission is not None
            ) and backend not in {
                "subprocess_actor",
                "subprocess_task",
            }:
                feature = (
                    "task admission"
                    if self._task_admission is not None
                    else "data accounting"
                    if self._data_ledger is not None
                    else "request admission"
                )
                raise ValueError(f"runtime {feature} requires local subprocess UDFs")
            if (
                self._data_ledger is not None
                and self._data_ledger.limits is not None
                and not payload_requests_local_ref_bundle_output(node["payload"])
            ):
                raise ValueError("runtime byte admission requires local shared-memory ref-bundle output")
            options = dict(node.get("executor_options") or {})
            if "local_task_admission" in options:
                raise ValueError("UDF node already has a query task admission binding")
            if "local_data_scope" in options:
                raise ValueError("UDF node already has a query data binding")
            if "local_input_cleanup" in options:
                raise ValueError("UDF node already has a query input cleanup binding")
            options["session_config"] = dict(self._session_config)
            node["executor_options"] = options
            executor_options_by_node[node_id] = options
        with self._lock:
            for node_id, name in bindings.items():
                node = nodes[node_id]
                model = self._models[name]
                payload = node["payload"]
                model.validate(payload, _local_actor_pool_size_from_node(node, payload), self._session_config)
                options = node["executor_options"]
                if "local_actor_pool" in options or "local_model_pool" in options:
                    raise ValueError("UDF node already has a local actor pool binding")
                # Do not grant the externally returned handle a drain bypass.
                # The preparation copy is authorized by one live request only.
                options["local_model_pool"] = (
                    replace(model, _request_ticket=request_ticket) if request_ticket is not None else model
                )
        # Actor preparation skips task nodes. Publish their configuration too,
        # inside the helper's rollback boundary in case handle injection fails.
        query = self._task_admission.open_query() if self._task_admission is not None else None
        if query is not None:
            for options in executor_options_by_node.values():
                options["local_task_admission"] = query
        data_query = None
        # Data scopes already retain failed input cleanup. Requests without
        # accounting still need a durable owner after native executors detach.
        input_query = QueryInputCleanup() if request_ticket is not None and self._data_ledger is None else None
        try:
            if input_query is not None:
                for options in executor_options_by_node.values():
                    options["local_input_cleanup"] = input_query
            if self._data_ledger is not None:
                data_query = self._data_ledger.open_query()
                for options in executor_options_by_node.values():
                    options["local_data_scope"] = data_query
            resources, actor_options = ensure_local_subprocess_actor_pools_for_nodes(
                list(nodes.values()),
                plan_identity=id(plan),
                set_handles=lambda options: plan.set_udf_actor_handles(
                    {**executor_options_by_node, **options}, conn=conn
                ),
            )
            # The actor helper has no publication callback for a task-only plan.
            # There are no actor owners to roll back in this case.
            if not actor_options and executor_options_by_node:
                plan.set_udf_actor_handles(executor_options_by_node, conn=conn)
        except BaseException as error:
            from vane.execution.udf_local_request import _shutdown_resource

            cleanup_errors: list[BaseException] = []
            pending = rollback_actor_pools(
                [owner for owner in (query, data_query, input_query) if owner is not None],
                RuntimeError("query preparation cleanup"),
                shutdown=lambda owner: _shutdown_resource(owner, kill=True),
                cleanup_pending=actor_pool_cleanup_pending,
                record_error=cleanup_errors.append,
            )
            if cleanup_errors:
                raise OwnedActorPoolsError(
                    "query preparation cleanup failed; retain owners for retry",
                    owned_actor_pools=[*getattr(error, "owned_actor_pools", ()), *pending],
                    creation_error=error,
                ) from cleanup_errors[0]
            raise
        return [*resources, *[owner for owner in (query, data_query, input_query) if owner is not None]]

    def prewarm(self, name: str) -> None:
        if self._request_admission is not None:
            self._request_admission.require_open()
        with self._lock:
            model = self._models[name]
        model.prewarm()

    def drain(self) -> None:
        if self._request_admission is not None:
            # Requests already executing may still be preparing their model
            # borrows. Fence ingress now; drain inner gates after they finish.
            self._request_admission.drain()
        else:
            self._drain_execution()

    def _drain_execution(self) -> None:
        if self._data_ledger is not None:
            self._data_ledger.drain()
        # Every task-limited preparation passes this gate, including task-only
        # plans that never acquire a model borrow. Fence it before model drain.
        if self._task_admission is not None:
            self._task_admission.drain()
        self._registry.drain()

    def resource_snapshot(self) -> dict[str, Any]:
        snapshot = self._registry.resource_snapshot()
        if self._task_admission is not None:
            snapshot["task_admission"] = self._task_admission.snapshot()
        if self._data_ledger is not None:
            snapshot["data"] = self._data_ledger.snapshot()
        if self._request_admission is not None:
            snapshot["request_admission"] = self._request_admission.snapshot()
            snapshot["draining"] = snapshot["draining"] or snapshot["request_admission"]["draining"]
            with self._lock:
                snapshot["request_admission"]["cleanup_pending_requests"] = len(self._request_cleanup)
        return snapshot

    def close(self, *, timeout: float = 0.0, kill: bool = False) -> None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("model runtime close timeout must be finite and non-negative")
        deadline = time.monotonic() + timeout
        self.drain()
        if self._request_admission is not None:
            with self._lock:
                pending = list(self._request_cleanup)
            errors = []
            for request in pending:
                try:
                    request.shutdown(kill=kill)
                except BaseException as error:
                    errors.append(error)
            if errors:
                raise RuntimeError("request cleanup failed during runtime close; retry close") from errors[0]
            self._request_admission.close(timeout=max(0.0, deadline - time.monotonic()))
            self._drain_execution()
        if self._data_ledger is not None:
            self._data_ledger.close(timeout=max(0.0, deadline - time.monotonic()))
        if self._task_admission is not None:
            self._task_admission.close(timeout=max(0.0, deadline - time.monotonic()))
        self._registry.close(timeout=max(0.0, deadline - time.monotonic()), kill=kill)

    def __enter__(self) -> LocalModelRuntime:
        if self._request_admission is not None:
            self._request_admission.require_open()
        self._registry.__enter__()
        return self

    def __exit__(self, _type: object, error: BaseException | None, _traceback: object) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is not None:
                raise error from cleanup_error
            raise
