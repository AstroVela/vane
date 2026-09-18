# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Session-owned local models with explicit physical-plan bindings.

These are internal execution APIs. They do not install a process-global cache
or change the lifetime of unregistered class UDFs.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vane import pickle as vane_pickle
from vane.execution.udf_model_pool import ModelPoolBorrow, ModelPoolIdentity, ModelPoolRegistry

if TYPE_CHECKING:
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
    _registry: ModelPoolRegistry[LocalSubprocessActorPool] = field(repr=False)
    _session_config: tuple[tuple[str, str], ...] = field(repr=False)

    def validate(self, payload: Mapping[str, Any], pool_size: int, session_config: Mapping[str, Any] | None) -> None:
        if session_config is None or tuple(sorted(session_config.items())) != self._session_config:
            raise ValueError("registered local model belongs to a different Vane session configuration")
        if pool_size != self.pool_size or _model_fingerprint(payload) != self.identity.initialization:
            raise ValueError("registered local model payload or pool size does not match the UDF node")

    def acquire(self) -> ModelPoolBorrow[LocalSubprocessActorPool]:
        return self._registry.acquire(self.identity)

    def prewarm(self) -> None:
        self._registry.prewarm(self.identity)


class LocalModelRuntime:
    """Own CPU subprocess models for one explicitly identified Vane session.

    Register from a collected UDF payload, optionally prewarm, then prepare a
    physical plan with explicit model/node bindings. Preparation acquires borrows
    through the existing local_actor_pool path. Runtime close is explicit and
    must run after query executors have finished and released their borrows.
    """

    def __init__(self, *, session_id: str, session_config: Mapping[str, Any]) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("local model runtime requires a non-empty session_id")
        self._session_id = session_id
        self._session_config = {str(key): str(value) for key, value in session_config.items()}
        self._registry: ModelPoolRegistry[LocalSubprocessActorPool] = ModelPoolRegistry()
        self._models: dict[str, RegisteredLocalModel] = {}
        self._lock = threading.Lock()

    def register(self, name: str, *, version: str, payload: Mapping[str, Any]) -> RegisteredLocalModel:
        from vane.execution.udf_subprocess import LocalSubprocessActorPool, _local_actor_pool_size_from_node

        if str(payload.get("execution_backend") or "").strip().lower() != "subprocess_actor":
            raise ValueError("local model registration requires a subprocess_actor UDF")
        if float(payload.get("gpus") or 0.0) > 0.0:
            raise ValueError("GPU resources require a Ray UDF backend")
        frozen_payload = _payload_bytes(payload)
        snapshot = vane_pickle.loads(frozen_payload)
        pool_size = _local_actor_pool_size_from_node({}, snapshot)
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

        model = RegisteredLocalModel(identity, pool_size, self._registry, tuple(sorted(config.items())))
        with self._lock:
            if name in self._models:
                raise ValueError(f"local model {name!r} is already registered; use a distinct name for another version")
            self._registry.register(identity, create)
            self._models[name] = model
        return model

    def prepare(
        self, plan: Any, bindings: Mapping[str, str], *, conn: Any = None
    ) -> list[LocalSubprocessActorPool | ModelPoolBorrow[LocalSubprocessActorPool]]:
        """Validate bindings, acquire query resources, and publish their handles.

        Call once per execution and retain the returned resources until all query
        executors have finished. Their shutdown releases only the query's owners.
        Different sessions are rejected even when their configurations match.
        """
        from vane.execution.udf_subprocess import (
            _local_actor_pool_size_from_node,
            ensure_local_subprocess_actor_pools_for_nodes,
        )

        if not bindings:
            raise ValueError("local model preparation requires explicit model bindings")
        if plan.session_id() != self._session_id or plan.session_config() != self._session_config:
            raise ValueError("local model runtime belongs to a different Vane session")
        nodes = {str(node["node_id"]): dict(node) for node in plan.collect_udf_nodes(conn=conn)}
        unknown = set(bindings) - set(nodes)
        if unknown:
            raise ValueError(f"unknown model UDF node IDs: {sorted(unknown)}")
        with self._lock:
            for node_id, name in bindings.items():
                node = nodes[node_id]
                model = self._models[name]
                payload = node["payload"]
                model.validate(payload, _local_actor_pool_size_from_node(node, payload), self._session_config)
                options = dict(node.get("executor_options") or {})
                if "local_actor_pool" in options or "local_model_pool" in options:
                    raise ValueError("UDF node already has a local actor pool binding")
                options.update(local_model_pool=model, session_config=dict(self._session_config))
                node["executor_options"] = options
        resources, _ = ensure_local_subprocess_actor_pools_for_nodes(
            list(nodes.values()),
            plan_identity=id(plan),
            set_handles=lambda options: plan.set_udf_actor_handles(options, conn=conn),
        )
        return resources

    def prewarm(self, name: str) -> None:
        with self._lock:
            model = self._models[name]
        model.prewarm()

    def drain(self) -> None:
        self._registry.drain()

    def close(self, *, timeout: float = 0.0, kill: bool = False) -> None:
        self._registry.close(timeout=timeout, kill=kill)

    def __enter__(self) -> LocalModelRuntime:
        self._registry.__enter__()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
