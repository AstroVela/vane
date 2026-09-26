# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit resident models for the ordinary local SQL and Relation APIs."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from vane import _native
from vane import pickle as vane_pickle
from vane._expression_udf import (
    VaneClassBatchInstance,
    VaneClassInstance,
    _build_actor_map_batches_expression,
    _expand_batch_expression,
    _preflight_attach_function,
    _PreparedBatchSQLRegistration,
)
from vane._expressions import as_expression
from vane.execution.resources import udf_process_resources
from vane.execution.udf_local_model import RegisteredLocalModel
from vane.pickle import _ActorAdapterMeta


def _restore_registered_actor(adapter: bytes, binding: tuple[Any, ...], row_adapter: bool) -> type:
    class _RegisteredActor(metaclass=_ActorAdapterMeta):
        _vane_row_actor_adapter = row_adapter
        _vane_local_model_binding = binding

        def __init__(self) -> None:
            self._actor = vane_pickle.loads(adapter)()

        def __call__(self, table: Any) -> Any:
            return self._actor(table)

    _RegisteredActor._vane_actor_adapter_recipe = (_restore_registered_actor, (adapter, binding, row_adapter))
    return _RegisteredActor


@dataclass(frozen=True)
class LocalQueryModel:
    """A frozen, session-owned model used as an expression or attached SQL UDF.

    Calls build expressions using the registered positional input types. They
    never construct an eager Python model in the calling process.
    """

    _definition: _PreparedBatchSQLRegistration
    _model: RegisteredLocalModel
    _unnest: bool = False

    @property
    def name(self) -> str:
        return self._model.identity.model

    @property
    def version(self) -> str:
        return self._model.identity.version

    def prewarm(self) -> None:
        """Initialize this registration's pool without retaining a query borrow."""
        _native._check_python_callback_entry()
        self._model.prewarm()

    def __call__(self, *args: Any) -> Any:
        _native._check_python_callback_entry()
        self._model._require_admission()
        expression = _model_expression(self._definition, args)
        return _expand_batch_expression(expression, unnest=self._unnest)

    def _sql_registration(
        self, alias: str | None, *, replace_alias: bool, **options: Any
    ) -> _PreparedBatchSQLRegistration:
        _native._check_python_callback_entry()
        self._model._require_admission()
        if any(value is not None for value in options.values()):
            raise ValueError("registered model input types and execution options cannot be overridden at attachment")
        if alias is not None and (not isinstance(alias, str) or not alias):
            raise ValueError("alias must be a non-empty string")
        return replace(
            self._definition, alias=self._definition.alias if alias is None else alias, replace=replace_alias
        )


def _model_expression(definition: _PreparedBatchSQLRegistration, args: tuple[Any, ...]) -> Any:
    if len(args) != len(definition.parameters):
        raise ValueError(f"registered model requires {len(definition.parameters)} positional inputs")
    assert definition.actor_number is not None
    return _build_actor_map_batches_expression(
        definition.udf,
        name=definition.alias,
        inputs={
            name: as_expression(argument).cast(dtype)
            for name, argument, dtype in zip(definition.input_names, args, definition.parameters, strict=True)
        },
        schema=definition.schema,
        batch_size=definition.batch_size,
        row_preserving=True,
        actor_number=definition.actor_number,
        gpus=0,
    )


def prepare_model_definition(
    model: Any,
    *,
    session_id: str,
    session_config: Mapping[str, str],
    name: str,
    version: str,
    parameters: Any,
    cpus: float,
    memory_bytes: int | None,
) -> tuple[_PreparedBatchSQLRegistration, bool]:
    if type(model) not in (VaneClassInstance, VaneClassBatchInstance):
        raise TypeError("register_model requires an instantiated vane.cls or vane.cls.batch model")
    for label, value in (("name", name), ("version", version)):
        if type(value) is not str or not value.strip():
            raise ValueError(f"model {label} must be a non-empty string")
    if model.gpus:
        raise ValueError("registered local query models support CPU subprocess actors only")
    if isinstance(cpus, bool):
        raise ValueError("cpus must be a finite positive number")
    resources = udf_process_resources({"cpus": cpus, "memory_bytes": memory_bytes})
    definition = _preflight_attach_function(
        model,
        None,
        replace=False,
        parameters=parameters,
        return_dtype=None,
        input_names=None,
        schema=None,
        batch_size=None,
        gpus=None,
        actor_number=None,
    )
    assert isinstance(definition, _PreparedBatchSQLRegistration)
    # Freeze reducers, closures and constructor arguments once. Rebuilding a
    # query must not capture later mutations of the caller's model definition.
    adapter = vane_pickle.dumps(definition.udf)
    binding = (
        session_id,
        name,
        version,
        uuid.uuid4().hex,
        resources.cpu,
        memory_bytes,
        session_config.get("VANE_UDF_TARGET_MAX_BATCH_BYTES", ""),
    )
    actor = _restore_registered_actor(adapter, binding, type(model) is VaneClassInstance)
    return replace(definition, udf=actor), bool(getattr(model, "unnest", False))


def collect_model_payload(connection: Any, definition: _PreparedBatchSQLRegistration) -> dict[str, Any]:
    import vane

    # A real native plan supplies the same type and transport metadata as later
    # SQL/Relation queries. This only binds/plans; it never executes user code.
    relation = connection.sql("SELECT 1")
    if definition.parameters:
        relation = relation.project(
            *(
                vane.lit(None).cast(dtype).alias(name)
                for name, dtype in zip(definition.input_names, definition.parameters, strict=True)
            )
        )
    inputs = tuple(vane.col(name) for name in definition.input_names)
    relation = relation.project(_model_expression(definition, inputs))
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)
    nodes = plan.collect_udf_nodes(conn=connection)
    if len(nodes) != 1:
        raise RuntimeError("registered model prototype must contain exactly one UDF")
    return nodes[0]["payload"]
