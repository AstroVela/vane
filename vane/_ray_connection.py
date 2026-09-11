# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ray connections build unresolved requests; local-fast keeps native objects."""

from __future__ import annotations

import copy
import inspect
import os
import threading
import uuid
import weakref
from collections.abc import Iterator
from typing import Any

from vane import _native
from vane._unresolved import (
    CONNECTION_OPERATIONS,
    RELATION_OPERATIONS,
    SOURCES,
    TRANSFORMS,
    WRITES,
    DriverStatement,
    UnresolvedInput,
    UnresolvedPlan,
    UnresolvedRequest,
    decode_argument,
    encode_argument,
    register_argument_reducers,
)

_CONSUMERS = {
    "fetchone": "fetchone",
    "fetchmany": "fetchmany",
    "fetchall": "fetchall",
    "fetchnumpy": "fetchnumpy",
    "df": "df",
    "fetchdf": "df",
    "fetch_df": "df",
    "to_df": "df",
    "fetch_df_chunk": "fetch_df_chunk",
    "to_arrow_table": "to_arrow_table",
    "fetch_arrow_table": "to_arrow_table",
    "to_arrow_reader": "to_arrow_reader",
    "fetch_record_batch": "to_arrow_reader",
    "arrow": "arrow",
    "pl": "pl",
    "torch": "torch",
    "tf": "tf",
}


def _close_after_error(resource: Any, error: BaseException) -> None:
    try:
        resource.close()
    except BaseException as cleanup_error:
        from vane.runners.ray.driver import QueryExecutionCleanupError

        raise QueryExecutionCleanupError.from_errors(
            "Unresolved Ray result failed and cleanup also failed", error, [cleanup_error]
        ) from error


class _ResultStream(Iterator[Any]):
    """Own a remote query even before the native result consumes its iterator."""

    def __init__(self, client: Any, reference: Any):
        self.client = client
        self.reference = reference
        _, _, self.runner = client._ensure_session(reference)
        self.iterator = client.stream_plan(reference)
        self.first: Any = None
        self.closed = False
        try:
            partition = next(self.iterator, None)
            if partition is not None:
                self.first = partition.partition()
        except BaseException as error:
            _close_after_error(self, error)
            raise

    def __next__(self) -> Any:
        if self.closed:
            raise StopIteration
        if self.first is not None:
            first, self.first = self.first, None
            return first
        try:
            return next(self.iterator).partition()
        except StopIteration:
            self.close()
            raise
        except BaseException as error:
            _close_after_error(self, error)
            raise

    def close(self) -> None:
        if self.closed:
            return
        self.first = None
        errors = []
        try:
            self.iterator.close()
        except BaseException as error:
            errors.append(error)
        try:
            self.client._close_plan_after_stream(
                self.runner, session_id=self.reference.session, plan_id=self.reference.query
            )
        except BaseException as error:
            errors.append(error)
        if len(errors) > 1:
            from vane.runners.ray.driver import QueryExecutionCleanupError

            raise QueryExecutionCleanupError.from_errors(
                "Unresolved Ray result cleanup failed", errors[0], errors[1:]
            ) from errors[0]
        if errors:
            raise errors[0]
        self.closed = True


class RayRelation:
    """An unresolved operation tree associated with one driver connection."""

    def __init__(self, connection: RayConnection, plan: UnresolvedPlan):
        self._connection = connection
        self._plan = plan
        self._result: Any = None
        connection._relations.add(self)

    def _derive(
        self, operation: str, args: tuple[Any, ...], kwargs: dict[str, Any], *, kind: str = "relation"
    ) -> RayRelation:
        self._connection._check_open()
        inputs = [self._plan]

        def capture(value: Any) -> Any:
            if isinstance(value, RayRelation):
                if value._connection is not self._connection:
                    raise _native.InvalidInputException("Relations must belong to the same Ray connection")
                inputs.append(value._plan)
                return UnresolvedInput(len(inputs) - 1)
            if isinstance(value, tuple):
                return tuple(capture(item) for item in value)
            if isinstance(value, list):
                return [capture(item) for item in value]
            if isinstance(value, dict):
                return {key: capture(item) for key, item in value.items()}
            return encode_argument(value)

        if operation in {"map", "map_batches", "flat_map"} and kwargs.get("execution_backend") is None:
            function = args[0] if args else kwargs.get("function", kwargs.get("map_function"))
            kwargs = {**kwargs, "execution_backend": "ray_actor" if inspect.isclass(function) else "ray_task"}
        arguments = capture(args)
        keywords = tuple(capture(kwargs).items())
        return RayRelation(self._connection, UnresolvedPlan(kind, operation, arguments, keywords, tuple(inputs)))

    def _consume(self, name: str, *args: Any, **kwargs: Any) -> Any:
        with self._connection._lock:
            self._connection._check_open()
            if self._result is None:
                self._result = self._connection._execute_plan(self._plan)
            return getattr(self._result, name)(*args, **kwargs)

    def execute(self) -> RayRelation:
        with self._connection._lock:
            self.close()
            self._result = self._connection._execute_plan(self._plan)
        return self

    def close(self) -> None:
        if self._result is not None:
            self._result.close()
            self._result = None

    def _schema_result(self) -> Any:
        descriptor = self._connection._request(self._plan, "analyze")
        return self._connection._make_result(iter(()), descriptor["schema"])

    @property
    def columns(self) -> list[str]:
        return self._schema_result().columns

    @property
    def types(self) -> list[Any]:
        return self._schema_result().types

    @property
    def dtypes(self) -> list[Any]:
        return self.types

    @property
    def description(self) -> Any:
        return self._schema_result().description

    @property
    def shape(self) -> tuple[int, int]:
        return len(self), len(self.columns)

    @property
    def alias(self) -> str:
        plan = self._derive("alias", (), {}, kind="inspect")._plan
        return self._connection._request(plan)["value"]

    def __len__(self) -> int:
        return self._derive("aggregate", ("count(*)",), {}).fetchone()[0]

    def __contains__(self, name: str) -> bool:
        return name in self.columns

    def __getitem__(self, name: str) -> RayRelation:
        return self._derive("project", (_native.ColumnExpression(name),), {})

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        return self._consume("__arrow_c_stream__", requested_schema)

    def __repr__(self) -> str:
        return f"RayRelation({self._plan.kind}.{self._plan.operation}, unresolved)"

    def _get_runner_type(self) -> str:
        return "ray"

    def _run_datasink(self) -> dict[str, Any]:
        return self._connection._execute_plan(self._plan, datasink=True)

    def write_datasink(self, sink: Any, *, operation_id: str | None = None) -> Any:
        from vane.datasink import write_datasink

        return write_datasink(self, sink, operation_id=operation_id)

    def query(self, virtual_table_name: str, sql_query: str) -> RayRelation | None:
        self.create_view(virtual_table_name)
        return self._connection.sql(sql_query)

    def __getattr__(self, name: str) -> Any:
        if name in TRANSFORMS:
            return lambda *args, **kwargs: self._derive(name, args, kwargs)
        if name in _CONSUMERS:
            return lambda *args, **kwargs: self._consume(_CONSUMERS[name], *args, **kwargs)
        if name in WRITES:

            def write(*args: Any, **kwargs: Any) -> None:
                plan = self._derive(name, args, kwargs, kind="write")._plan
                result = self._connection._execute_plan(plan)
                result.close()

            return write
        if name in RELATION_OPERATIONS:

            def inspect_relation(*args: Any, **kwargs: Any) -> Any:
                plan = self._derive(name, args, kwargs, kind="inspect")._plan
                value = decode_argument(self._connection._request(plan)["value"])
                return self if name in {"create_view", "to_view"} else value

            return inspect_relation
        if name in {"embed", "prompt"}:
            from vane.ai import _relation_patch

            return lambda *args, **kwargs: getattr(_relation_patch, "_" + name)(self, *args, **kwargs)
        if name.startswith("_"):
            raise AttributeError(name)
        if hasattr(_native.DuckDBPyRelation, name):
            raise _native.NotImplementedException(f"Ray Relation operation is not implemented: {name}")
        return self[name]


class RayConnection:
    """Connection identity and unresolved requests; its catalog lives on Ray."""

    def __init__(self, database: Any = ":memory:", *, read_only: bool = False, config: dict[str, Any] | None = None):
        register_argument_reducers()
        self._bootstrap = (os.fspath(database), bool(read_only), tuple((config or {}).items()))
        self._session = uuid.uuid4().hex
        captured = {
            key: value for key, value in os.environ.items() if key.startswith(("AWS_", "DUCKDB_", "S3FS_", "VANE_"))
        }
        captured["VANE_RUNNER"] = "ray"
        self._config = tuple(sorted(captured.items()))
        self._parent_session: str | None = None
        self._runner: Any = None
        self._result_context: Any = None
        self._result: Any = None
        self._relations: weakref.WeakSet[RayRelation] = weakref.WeakSet()
        self._closed = False
        self._lock = threading.RLock()

    def _check_open(self) -> None:
        if self._closed:
            raise _native.ConnectionException("Connection already closed!")

    def _client(self) -> Any:
        self._check_open()
        if self._runner is None:
            self._runner = _native.set_runner_ray(None, True)
        return self._runner._client_for_session(self._session)

    def _envelope(self, plan: UnresolvedPlan) -> UnresolvedRequest:
        return UnresolvedRequest(
            self._session, self._config, uuid.uuid4().hex, plan, self._bootstrap, self._parent_session
        )

    def _request(self, plan: UnresolvedPlan, mode: str = "execute") -> dict[str, Any]:
        with self._lock:
            return self._client().unresolved_request(self._envelope(plan), mode)

    def _make_result(self, iterator: Iterator[Any], schema: bytes) -> Any:
        if self._result_context is None:
            # Used exclusively by native result converters, never to bind or
            # execute user SQL. In particular, no driver catalog is mirrored.
            self._result_context = _native._connect_with_runner("local-fast")
        return _native.ray_cxx._make_unresolved_result(self._result_context, iterator, schema)

    def _execute_plan(self, plan: UnresolvedPlan, *, datasink: bool = False) -> Any:
        with self._lock:
            descriptor = self._request(plan)
            kind = descriptor["kind"]
            if kind == "native":
                return self._make_result(iter((descriptor["table"],)), descriptor["schema"])
            reference = descriptor["reference"]
            client = self._client()
            if kind == "datasink":
                if not datasink:
                    _, _, runner = client._ensure_session(reference)
                    client._close_plan_after_stream(runner, session_id=reference.session, plan_id=reference.query)
                    raise _native.InvalidInputException("DataSink plans require write_datasink()")
                return client.run_datasink_plan(reference)
            if kind == "write":
                outcome = client.run_copy_plan(reference)
                try:
                    import pyarrow as pa

                    rows = outcome["rows_copied"]
                    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
                        raise ValueError("Ray write returned an invalid row count")
                    table = pa.table({"Count": pa.array([rows], type=pa.int64())})
                    return self._make_result(iter((table,)), descriptor["schema"])
                except Exception as error:
                    from vane.runners.copy_outcome import CopyResultUnavailableError

                    raise CopyResultUnavailableError(
                        outcome["copy_operation_id"],
                        f"Result conversion failed after commit: {error}",
                        tuple(outcome.get("copy_cleanup_warnings", ())),
                    ) from error
            if kind != "read":
                _, _, runner = client._ensure_session(reference)
                client._close_plan_after_stream(runner, session_id=reference.session, plan_id=reference.query)
                raise ValueError(f"Unknown driver result kind: {kind}")
            stream = _ResultStream(client, reference)
            try:
                return self._make_result(stream, descriptor["schema"])
            except BaseException as error:
                _close_after_error(stream, error)
                raise

    @staticmethod
    def _sql_plan(query: Any, parameters: Any = None, *, alias: str = "query_relation") -> UnresolvedPlan:
        if isinstance(query, (_native.Statement, DriverStatement)):
            query = query.query
        if not isinstance(query, str):
            raise _native.InvalidInputException("Please provide SQL text or a Vane Statement")
        return UnresolvedPlan(
            "sql", "query", (query,), (("params", copy.deepcopy(encode_argument(parameters))), ("alias", alias))
        )

    def execute(self, query: Any, parameters: Any = None) -> RayConnection:
        with self._lock:
            self._check_open()
            if self._result is not None:
                self._result.close()
                self._result = None
            statements = self._request(self._sql_plan(query), "parse")["statements"]
            for index, statement in enumerate(statements):
                params = parameters if index == len(statements) - 1 else None
                result = self._execute_plan(self._sql_plan(statement.query, params))
                if index == len(statements) - 1:
                    self._result = result
                else:
                    try:
                        while result.fetchmany(2048):
                            pass
                    finally:
                        result.close()
        return self

    def sql(self, query: Any, *, alias: str = "query_relation", params: Any = None) -> RayRelation | None:
        with self._lock:
            statements = self._request(self._sql_plan(query), "parse")["statements"]
            for index, statement in enumerate(statements):
                last = index == len(statements) - 1
                if last and statement.type == _native.StatementType.SELECT:
                    return RayRelation(self, self._sql_plan(statement.query, params, alias=alias))
                self.execute(statement.query, params if last else None)
                while self.fetchmany(2048):
                    pass
        return None

    query = sql
    from_query = sql

    def executemany(self, query: Any, parameters: Any = None) -> RayConnection:
        if parameters is None:
            raise _native.InvalidInputException("executemany requires parameter sets")
        executed = False
        for values in parameters:
            if executed:
                while self.fetchmany(2048):
                    pass
            self.execute(query, values)
            executed = True
        if not executed:
            raise _native.InvalidInputException("executemany requires at least one parameter set")
        return self

    @property
    def description(self) -> Any:
        return None if self._result is None else self._result.description

    @property
    def rowcount(self) -> int:
        return -1

    def _consume(self, name: str, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._check_open()
            if self._result is None:
                raise _native.InvalidInputException("No open result set")
            if "rows_per_batch" in kwargs:
                kwargs = dict(kwargs)
                kwargs["batch_size"] = kwargs.pop("rows_per_batch")
            return getattr(self._result, name)(*args, **kwargs)

    def register(self, view_name: str, python_object: Any) -> RayConnection:
        if isinstance(python_object, RayRelation):
            if python_object._connection is not self:
                raise _native.InvalidInputException("Registered Relation must belong to this Ray connection")
            python_object.create_view(view_name, replace=True)
        else:
            plan = UnresolvedPlan("connection", "register", encode_argument((view_name, python_object)))
            self._request(plan)
        return self

    def extract_statements(self, query: str) -> list[DriverStatement]:
        return self._request(self._sql_plan(query), "parse")["statements"]

    def _extension_operation(self, name: str, *args: Any, **kwargs: Any) -> Any:
        plan = UnresolvedPlan("extension", name, encode_argument(args), tuple(encode_argument(kwargs).items()))
        return self._request(plan)["value"]

    def append(self, table_name: str, df: Any, *, by_name: bool = False) -> RayConnection:
        if by_name:
            raise _native.NotImplementedException("The experimental Ray connection does not implement append by_name")
        relation = self.from_df(df)
        relation.insert_into(table_name)
        return self

    def interrupt(self) -> None:
        self._check_open()
        if self._runner is None:
            return
        from vane.runners.ray.safe_get import resolve_object_refs_blocking

        client = self._runner._client_for_session(self._session)
        resolve_object_refs_blocking(client.runner.interrupt_unresolved_session.remote(client._owner_id, self._session))

    def cursor(self) -> RayConnection:
        with self._lock:
            client = self._client()
            client._ensure_session(self._envelope(self._sql_plan("")))
            database, read_only, options = self._bootstrap
            cursor = RayConnection(database, read_only=read_only, config=dict(options))
            cursor._config = self._config
            cursor._runner = self._runner
            cursor._parent_session = self._session
            client._ensure_session(cursor._envelope(self._sql_plan("")))
            return cursor

    duplicate = cursor

    def begin(self) -> RayConnection:
        return self.execute("BEGIN TRANSACTION")

    def commit(self) -> RayConnection:
        return self.execute("COMMIT")

    def rollback(self) -> RayConnection:
        return self.execute("ROLLBACK")

    def checkpoint(self) -> RayConnection:
        return self.execute("CHECKPOINT")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._result is not None:
                self._result.close()
                self._result = None
            for relation in tuple(self._relations):
                relation.close()
            if self._runner is not None:
                self._runner.close_session(self._session)
            if self._result_context is not None:
                self._result_context.close()
                self._result_context = None
            self._closed = True

    def __enter__(self) -> RayConnection:
        self._check_open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            # Explicit close reports teardown failures; the runtime owner lease
            # is responsible for reclaiming an abandoned connection.
            pass

    def __getattr__(self, name: str) -> Any:
        if name in SOURCES:

            def source(*args: Any, **kwargs: Any) -> RayRelation:
                self._check_open()
                return RayRelation(
                    self, UnresolvedPlan("source", name, encode_argument(args), tuple(encode_argument(kwargs).items()))
                )

            return source
        if name in _CONSUMERS:
            return lambda *args, **kwargs: self._consume(_CONSUMERS[name], *args, **kwargs)
        if name in CONNECTION_OPERATIONS:

            def operation(*args: Any, **kwargs: Any) -> Any:
                plan = UnresolvedPlan("connection", name, encode_argument(args), tuple(encode_argument(kwargs).items()))
                value = decode_argument(self._request(plan)["value"])
                return self if value is None else value

            return operation
        if name.startswith("_"):
            raise AttributeError(name)
        raise _native.NotImplementedException(f"Ray connection operation is not implemented: {name}")
