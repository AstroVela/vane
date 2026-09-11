# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Interpret unresolved API requests in the authoritative driver connection."""

from __future__ import annotations

from typing import Any

from vane._unresolved import (
    CONNECTION_OPERATIONS,
    RELATION_OPERATIONS,
    SOURCES,
    TRANSFORMS,
    WRITES,
    DriverPlanReference,
    DriverStatement,
    UnresolvedInput,
    UnresolvedPlan,
    UnresolvedRequest,
    decode_argument,
    encode_argument,
)


def _arguments(value: Any, inputs: tuple[Any, ...]) -> Any:
    if isinstance(value, UnresolvedInput):
        if value.index < 0 or value.index >= len(inputs):
            raise ValueError("Unresolved operation has an invalid input index")
        return inputs[value.index]
    if isinstance(value, tuple):
        return tuple(_arguments(item, inputs) for item in value)
    if isinstance(value, list):
        return [_arguments(item, inputs) for item in value]
    if isinstance(value, dict):
        return {key: _arguments(item, inputs) for key, item in value.items()}
    return decode_argument(value)


def build_relation(connection: Any, plan: UnresolvedPlan, *, depth: int = 0) -> Any:
    from vane import _native

    if not isinstance(plan, UnresolvedPlan) or depth > 256:
        raise ValueError("Invalid or excessively nested unresolved Relation")
    inputs = tuple(build_relation(connection, child, depth=depth + 1) for child in plan.inputs)
    args = _arguments(plan.arguments, inputs)
    kwargs = _arguments(dict(plan.keywords), inputs)
    if plan.kind == "sql" and plan.operation == "query" and not inputs:
        statements = connection.extract_statements(args[0])
        if len(statements) != 1 or statements[0].type != _native.StatementType.SELECT:
            raise _native.InvalidInputException("An unresolved SQL Relation must contain one SELECT statement")
        return connection.sql(statements[0], **kwargs)
    if plan.kind == "source" and plan.operation in SOURCES and not inputs:
        return getattr(connection, plan.operation)(*args, **kwargs)
    if plan.kind == "relation" and plan.operation in TRANSFORMS and inputs:
        return getattr(inputs[0], plan.operation)(*args, **kwargs)
    if plan.kind == "write" and plan.operation in WRITES and len(inputs) == 1:
        return getattr(_native.ray_cxx, WRITES[plan.operation])(inputs[0], *args, **kwargs)
    raise _native.NotImplementedException(f"Unsupported unresolved operation: {plan.kind}.{plan.operation}")


def prepare_request(session: Any, request: UnresolvedRequest, mode: str) -> dict[str, Any]:
    """Run under the session's native operation lock on its executor thread."""
    from vane import _native

    if mode not in {"execute", "parse", "analyze"}:
        raise ValueError(f"Unknown unresolved request mode: {mode}")
    if not isinstance(request.plan, UnresolvedPlan):
        raise TypeError("Expected an unresolved operation tree")
    if mode == "parse" and (request.plan.kind != "sql" or request.plan.operation != "query"):
        raise ValueError("Only SQL requests can be parsed")
    if mode != "execute" and request.plan.kind in {"extension", "connection", "inspect", "write"}:
        raise ValueError("State-changing operations require execute mode")
    with session.operation_lock:
        if session.bootstrap != request.bootstrap:
            raise ValueError("Unresolved connection bootstrap changed after session open")
        connection = session.connection
        plan = request.plan
        with session.condition:
            if request.query in session.unresolved_plans:
                raise ValueError("Unresolved query identity was already prepared")
            if mode == "execute" and len(session.unresolved_plans) >= 128:
                raise _native.InvalidInputException("Close an outstanding Ray result before preparing another query")
        if plan.kind == "extension" and plan.operation in {"load_installed_extension", "extension_statuses"}:
            from vane import extensions

            value = getattr(extensions, plan.operation)(
                *decode_argument(plan.arguments), connection=connection, **decode_argument(dict(plan.keywords))
            )
            return {"value": value}
        if mode == "parse":
            return {
                "statements": [
                    DriverStatement(
                        statement.query,
                        statement.type,
                        frozenset(statement.named_parameters),
                        tuple(statement.expected_result_type),
                    )
                    for statement in connection.extract_statements(plan.arguments[0])
                ]
            }
        if plan.kind == "connection" and plan.operation in CONNECTION_OPERATIONS:
            args = _arguments(plan.arguments, ())
            kwargs = _arguments(dict(plan.keywords), ())
            value = getattr(connection, plan.operation)(*args, **kwargs)
            return {"value": None if value is connection else encode_argument(value)}
        if plan.kind == "inspect" and plan.operation in RELATION_OPERATIONS and len(plan.inputs) == 1:
            relation = build_relation(connection, plan.inputs[0])
            member = getattr(relation, plan.operation)
            value = (
                member
                if plan.operation == "alias"
                else member(*_arguments(plan.arguments, (relation,)), **_arguments(dict(plan.keywords), (relation,)))
            )
            return {"value": None if isinstance(value, _native.DuckDBPyRelation) else encode_argument(value)}

        source = None
        parameters = None
        if plan.kind == "sql" and mode == "execute":
            statements = connection.extract_statements(plan.arguments[0])
            if len(statements) != 1:
                raise _native.InvalidInputException("Execution requires one parsed SQL statement")
            source = statements[0]
            parameters = decode_argument(dict(plan.keywords).get("params"))
        else:
            source = build_relation(connection, plan)
        if mode == "analyze":
            return {"schema": _native.ray_cxx._unresolved_relation_schema(source)}
        prepared = _native.ray_cxx._prepare_unresolved_execution(
            connection, source, parameters, request.query, request.session, request.session_config()
        )
        if prepared["kind"] == "native":
            result = prepared.pop("result")
            try:
                prepared["table"] = result.to_arrow_table()
            finally:
                result.close()
        else:
            # The input keeps registered Python/Arrow dependencies alive through
            # fragment teardown, including cancellation and partial consumption.
            with session.condition:
                session.unresolved_plans[request.query] = (prepared.pop("plan"), source)
            prepared["reference"] = DriverPlanReference(request.session, request.config, request.query)
        return prepared


def resolve_plan_reference(session: Any, plan: Any) -> Any:
    if not isinstance(plan, DriverPlanReference):
        return plan
    with session.condition:
        prepared = session.unresolved_plans.get(plan.query)
        if prepared is None:
            raise ValueError("Driver query reference is closed or was never prepared")
        return prepared[0]
