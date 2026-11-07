# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributed PostgreSQL keyed upserts through ADBC.

Each worker owns one PostgreSQL connection and one temporary staging table.
Every batch is copied into the staging table with ADBC bulk ingestion and then
merged into the target table with ``INSERT ... ON CONFLICT`` inside a single
transaction, so a failed batch leaves the target unchanged. Batches from
different workers commit independently; Vane does not provide a transaction
spanning the complete distributed relation.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, quote, urlsplit

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]
import pyarrow.compute as pc  # type: ignore[import-not-found, import-untyped, unused-ignore]

from vane.datasink import (
    BoundKeyedUpsertSink,
    DataSink,
    DataSinkExecutionOptions,
    DataSinkWorker,
    EnvironmentSecret,
    WriteContext,
    WriteResult,
)
from vane.datasink._arrow_schema import same_input_type
from vane.execution._diagnostics import bounded_utf8_text, exception_message_from_args, safe_exception_type_name

if TYPE_CHECKING:
    from types import ModuleType

_MAX_INT64 = (1 << 63) - 1
_MAX_IDENTIFIER_BYTES = 63
_DEFAULT_MAX_BATCH_ROWS = 10_000
_DEFAULT_MAX_BATCH_BYTES = 64 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 300
_URI_SCHEMES = ("postgresql://", "postgres://")
_CONNINFO_SPACE = " \t\n\v\f\r"
_INVALID_DSN = "dsn is not a valid libpq connection string or URI"
_TARGET_ALIAS = '"vane_target"'
# Target column types without typmods, for the merge casts. Domains resolve to their innermost base type, since an
# explicit cast to a domain applies the domain's typmod the same truncating way.
_TARGET_TYPES_SQL = (
    "SELECT a.attname, format_type(base.oid, -1) FROM pg_attribute AS a CROSS JOIN LATERAL ("
    "WITH RECURSIVE chain(oid, typtype, typbasetype, depth) AS ("
    "SELECT t.oid, t.typtype, t.typbasetype, 0 FROM pg_type AS t WHERE t.oid = a.atttypid "
    "UNION ALL SELECT t.oid, t.typtype, t.typbasetype, chain.depth + 1 FROM chain "
    "JOIN pg_type AS t ON t.oid = chain.typbasetype WHERE chain.typtype = 'd') "
    "SELECT chain.oid FROM chain ORDER BY chain.depth DESC LIMIT 1) AS base "
    "WHERE a.attrelid = to_regclass($1) AND a.attnum > 0 AND NOT a.attisdropped"
)
_PARTIAL_VISIBILITY_WARNING = (
    "PostgreSQL commits worker batches independently; a later operation failure can leave this batch applied"
)


def _name(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")
    if value != value.strip() or "\x00" in value or "\r" in value or "\n" in value:
        raise ValueError(f"{name} must not contain surrounding whitespace, NUL, CR, or LF characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid UTF-8") from error
    return value


def _identifier(name: str, value: object) -> str:
    identifier = _name(name, value)
    # PostgreSQL silently truncates longer identifiers, which could address another object.
    if len(identifier.encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
        raise ValueError(f"{name} must be at most {_MAX_IDENTIFIER_BYTES} UTF-8 bytes")
    return identifier


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _conninfo_keywords(dsn: str) -> list[str]:
    """Return the keywords of a key/value connection string, tokenized as libpq's conninfo_parse() does.

    A quoted value needs no whitespace after it, so a regular expression cannot find every keyword.
    """

    keywords: list[str] = []
    index, length = 0, len(dsn)
    while index < length:
        if dsn[index] in _CONNINFO_SPACE:
            index += 1
            continue
        start = index
        while index < length and dsn[index] != "=" and dsn[index] not in _CONNINFO_SPACE:
            index += 1
        keyword = dsn[start:index]
        while index < length and dsn[index] in _CONNINFO_SPACE:
            index += 1
        if index >= length or dsn[index] != "=":
            raise ValueError(_INVALID_DSN)
        index += 1
        while index < length and dsn[index] in _CONNINFO_SPACE:
            index += 1
        if index < length and dsn[index] == "'":
            index += 1
            while True:
                if index >= length:
                    raise ValueError(_INVALID_DSN)
                if dsn[index] == "\\":
                    index += 2
                elif dsn[index] == "'":
                    index += 1
                    break
                else:
                    index += 1
        else:
            while index < length and dsn[index] not in _CONNINFO_SPACE:
                index += 2 if dsn[index] == "\\" else 1
        keywords.append(keyword)
    return keywords


def _dsn_has_password(dsn: str) -> bool:
    if dsn.lower().startswith(_URI_SCHEMES):
        try:
            parsed = urlsplit(dsn)
            userinfo_password = parsed.password
        except ValueError as error:
            raise ValueError(_INVALID_DSN) from error
        return userinfo_password is not None or "password" in parse_qs(parsed.query, keep_blank_values=True)
    return "password" in _conninfo_keywords(dsn)


def _dsn(value: object) -> str:
    dsn = _name("dsn", value)
    # libpq URIs have no fragment; a raw "#" would hide the rest of the URI from the password check.
    if dsn.lower().startswith(_URI_SCHEMES) and "#" in dsn:
        raise ValueError("dsn URI must not contain '#'; percent-encode it as %23")
    if _dsn_has_password(dsn):
        raise ValueError("dsn must not contain a password; use password=EnvironmentSecret(...)")
    return dsn


def _with_password(dsn: str, password: str) -> str:
    """Add a password to a DSN; the ADBC driver accepts no separate password option."""

    if not password:
        raise ValueError("password must not be empty")
    if "\x00" in password:
        raise ValueError("password must not contain NUL characters")
    if dsn.lower().startswith(_URI_SCHEMES):
        separator = "" if dsn.endswith(("?", "&")) else "&" if "?" in dsn else "?"
        return f"{dsn}{separator}password={quote(password, safe='')}"
    # Leading, so a malformed trailing value in the DSN cannot absorb the password.
    return f"password='{_escaped_password(password)}' {dsn}"


def _escaped_password(password: str) -> str:
    return password.replace("\\", "\\\\").replace("'", "\\'")


def _secret_forms(password: str | None) -> tuple[str, ...]:
    """Every spelling of the password that can appear in a connection string echoed by libpq."""

    if not password:
        return ()
    return tuple(dict.fromkeys((password, quote(password, safe=""), _escaped_password(password))))


def _redact(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        text = text.replace(secret, "***")
    return text


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0 or value > _MAX_INT64:
        raise ValueError(f"{name} must be a positive signed 64-bit integer")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0 or value > _MAX_INT64:
        raise ValueError(f"{name} must be a non-negative signed 64-bit integer")
    return value


def _field_mapping(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("field_mapping must be a mapping or None")
    return tuple(
        (_name("field_mapping source", source), _identifier("field_mapping target", target))
        for source, target in value.items()
    )


def _key_columns(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (_name("key_columns", value),)
    if not isinstance(value, Sequence):
        raise TypeError("key_columns must be a column name or a sequence of column names")
    keys = tuple(_name("key_columns", key) for key in value)
    if not keys:
        raise ValueError("key_columns must contain at least one column")
    if len(set(keys)) != len(keys):
        raise ValueError("key_columns must not contain duplicates")
    return keys


def _is_list_type(data_type: pa.DataType) -> bool:
    return pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type)


def _is_scalar_type(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_null(data_type)
        or pa.types.is_boolean(data_type)
        or pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_string_view(data_type)
        or pa.types.is_binary(data_type)
        or pa.types.is_large_binary(data_type)
        or pa.types.is_fixed_size_binary(data_type)
        or pa.types.is_date(data_type)
        or pa.types.is_timestamp(data_type)
        or pa.types.is_time(data_type)
        or pa.types.is_duration(data_type)
        or pa.types.is_decimal128(data_type)
    )


def _validate_source_type(data_type: pa.DataType, path: str) -> None:
    if pa.types.is_dictionary(data_type):
        _validate_source_type(data_type.value_type, path)
        return
    if _is_list_type(data_type):
        value_type = data_type.value_type
        if pa.types.is_dictionary(value_type):
            value_type = value_type.value_type
        if not _is_scalar_type(value_type):
            raise ValueError(
                f"PostgreSQL source field {path!r} uses unsupported Arrow type {data_type}; "
                "lists must contain scalar values"
            )
        return
    if not _is_scalar_type(data_type):
        raise ValueError(
            f"PostgreSQL source field {path!r} uses unsupported Arrow type {data_type}; supported types are "
            "booleans, integers, floating-point numbers, strings, binary values, dates, times, timestamps, "
            "durations, decimal128, dictionaries of these, and lists of these"
        )


def _wire_type(data_type: pa.DataType) -> pa.DataType:
    """Return the Arrow type sent to ADBC for an input type.

    ADBC wraps uint64 values above the int64 range into negative bigint values
    and truncates nanosecond temporal values to PostgreSQL's microseconds, so
    both are converted with a checked cast first.
    """

    if pa.types.is_dictionary(data_type):
        return _wire_type(data_type.value_type)
    if pa.types.is_uint64(data_type):
        return pa.int64()
    if pa.types.is_date64(data_type):
        return pa.date32()
    if pa.types.is_timestamp(data_type):
        return pa.timestamp("us", tz=data_type.tz)
    if pa.types.is_time(data_type):
        return pa.time64("us")
    if pa.types.is_duration(data_type):
        return pa.duration("us")
    if _is_list_type(data_type):
        value_field = data_type.value_field
        return pa.list_(pa.field(value_field.name, _wire_type(value_field.type), nullable=value_field.nullable))
    return data_type


def _load_adbc_postgresql() -> ModuleType:
    try:
        import adbc_driver_postgresql.dbapi  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ModuleNotFoundError as error:
        if error.name not in {"adbc_driver_postgresql", "adbc_driver_manager"}:
            raise
        raise ImportError("PostgresSink requires adbc-driver-postgresql; install vane-ai[postgres]") from error
    return adbc_driver_postgresql.dbapi


def _connect(dsn: str) -> Any:
    return _load_adbc_postgresql().connect(dsn)


def _adbc_postgresql_version() -> tuple[int, int] | None:
    try:
        version = metadata.version("adbc-driver-postgresql")
    except metadata.PackageNotFoundError:
        return None
    match = re.match(r"(\d+)\.(\d+)", version)
    return None if match is None else (int(match.group(1)), int(match.group(2)))


def _open_connection(dsn: str, secrets: tuple[str, ...]) -> Any:
    failure: str | None = None
    try:
        return _connect(dsn)
    except Exception as error:
        # libpq echoes some malformed connection strings, including the password added on the worker. Report a
        # redacted copy and drop the original exception, whose arguments still contain it.
        detail = _redact(exception_message_from_args(error) or "", secrets)
        failure = f"PostgreSQL connection failed: {safe_exception_type_name(error)}"
        if detail:
            failure = f"{failure}: {bounded_utf8_text(detail, 512)}"
    raise RuntimeError(failure)


@dataclass(frozen=True)
class _FieldBinding:
    source_name: str
    target_name: str
    wire_type: pa.DataType
    is_key: bool


class PostgresSink(DataSink):
    """Upsert relation batches into a PostgreSQL table by key.

    The target table must already exist and have a primary key or unique
    constraint on the mapped ``key_columns``. ``key_columns`` names input
    columns; their values must be non-null and unique across the complete
    input, which Vane validates before any worker opens. Every input column is
    written: ``field_mapping`` maps input names to target column names and
    unmapped inputs keep their names. Rows whose key already exists have every
    mapped non-key column replaced; a sink whose columns are all keys inserts
    new keys only. Identifiers are matched exactly, as quoted SQL identifiers.

    Each batch is copied into a worker-local temporary table with ADBC bulk
    ingestion and merged with ``INSERT ... ON CONFLICT`` in one transaction, so
    PostgreSQL either applies or rejects the whole batch. Values are cast to
    the target column types in SQL, which covers ``uuid``, ``jsonb``, and
    pgvector ``vector``/``halfvec`` columns from string and float list inputs.
    PostgreSQL and pgvector reject values that do not fit, as a plain
    ``INSERT`` does, such as a string longer than a ``varchar(n)`` column, an
    embedding with the wrong dimension or a NaN element.

    Input types are booleans, integers, floating-point numbers, strings, binary
    values, dates, times, timestamps, durations, decimal128, dictionaries of
    these, and lists of these. uint64 values must fit in int64, and temporal
    values are sent with microsecond precision, the precision PostgreSQL
    stores; values that would change fail the batch before it is sent. With
    adbc-driver-postgresql before 1.13, which writes NULL list elements as zero
    values, a batch whose lists contain NULL elements fails the same way.
    Nested lists, structs, maps, and intervals are rejected at bind time.

    ``dsn`` is a libpq connection string or URI without a password. It may be
    a string or an ``EnvironmentSecret`` resolved on each worker. Passwords
    must use ``password=EnvironmentSecret(...)``, which is resolved and added
    to the connection string only on the worker. Omit ``password`` when the
    server does not require one, for example with certificate authentication.
    ``timeout`` sets the ``statement_timeout`` for each worker connection.

    Framework retries are disabled unless ``max_retries`` is positive. A retry
    replays the full input with the same keys; Vane does not coordinate
    concurrent writers or provide a transaction or exactly-once delivery
    across batches. If the connection fails while a batch commits, the outcome
    of that batch is unknown.
    """

    def __init__(
        self,
        table: str,
        *,
        dsn: str | EnvironmentSecret,
        key_columns: str | Sequence[str],
        password: EnvironmentSecret | None = None,
        schema: str | None = None,
        field_mapping: Mapping[str, str] | None = None,
        worker_count: int = 1,
        max_batch_rows: int = _DEFAULT_MAX_BATCH_ROWS,
        max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
        max_retries: int = 0,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.table = _identifier("table", table)
        self.schema = None if schema is None else _identifier("schema", schema)
        self.dsn: str | EnvironmentSecret
        if isinstance(dsn, EnvironmentSecret):
            self.dsn = dsn
        else:
            self.dsn = _dsn(dsn)
        if password is not None and not isinstance(password, EnvironmentSecret):
            raise TypeError("password must be an EnvironmentSecret or None")
        self._password = password
        self._key_columns = _key_columns(key_columns)
        self._field_mapping = _field_mapping(field_mapping)
        self.worker_count = _positive_int("worker_count", worker_count)
        self.max_batch_rows = _positive_int("max_batch_rows", max_batch_rows)
        self.max_batch_bytes = _positive_int("max_batch_bytes", max_batch_bytes)
        self.max_retries = _non_negative_int("max_retries", max_retries)
        self.timeout = _positive_int("timeout", timeout)

    @property
    def key_columns(self) -> tuple[str, ...]:
        return self._key_columns

    def bind(self, schema: pa.Schema) -> BoundKeyedUpsertSink:
        if not isinstance(schema, pa.Schema):
            raise TypeError("schema must be pyarrow.Schema")
        if not schema:
            raise ValueError("PostgresSink requires at least one input column")
        if len(set(schema.names)) != len(schema.names):
            raise ValueError("PostgresSink requires unique input column names")
        for field in schema:
            _validate_source_type(field.type, field.name)

        mapping = dict(self._field_mapping)
        if len(mapping) != len(self._field_mapping):
            raise ValueError("field_mapping sources must be unique")
        unknown_mapping = set(mapping).difference(schema.names)
        if unknown_mapping:
            raise ValueError(f"field_mapping contains unknown input columns: {sorted(unknown_mapping)!r}")
        unknown_keys = set(self._key_columns).difference(schema.names)
        if unknown_keys:
            raise ValueError(f"key_columns contains unknown input columns: {sorted(unknown_keys)!r}")

        fields = tuple(
            _FieldBinding(
                source_name=field.name,
                target_name=_identifier("PostgreSQL target column", mapping.get(field.name, field.name)),
                wire_type=_wire_type(field.type),
                is_key=field.name in self._key_columns,
            )
            for field in schema
        )
        target_names = [field.target_name for field in fields]
        if len(set(target_names)) != len(target_names):
            raise ValueError("field_mapping must produce unique PostgreSQL column names")
        return _BoundPostgresSink(self, schema, fields)

    def _resolve_connection(self) -> tuple[str, tuple[str, ...]]:
        """Return the worker's connection string and the password spellings to redact from driver errors."""

        dsn = self.dsn
        if isinstance(dsn, EnvironmentSecret):
            dsn = _dsn(dsn.resolve())
        if self._password is None:
            return dsn, ()
        password = self._password.resolve()
        return _with_password(dsn, password), _secret_forms(password)


class _BoundPostgresSink(BoundKeyedUpsertSink):
    def __init__(self, sink: PostgresSink, schema: pa.Schema, fields: tuple[_FieldBinding, ...]) -> None:
        self._sink = sink
        self._schema = schema
        self._fields = fields

    @property
    def execution_options(self) -> DataSinkExecutionOptions:
        return DataSinkExecutionOptions(
            worker_count=self._sink.worker_count,
            batch_size=self._sink.max_batch_rows,
            target_max_batch_bytes=self._sink.max_batch_bytes,
            max_retries=self._sink.max_retries,
        )

    @property
    def key_columns(self) -> Sequence[str]:
        return self._sink.key_columns

    def open_worker(self, _context: WriteContext) -> DataSinkWorker:
        return _PostgresWorker(self._sink, self._schema, self._fields)


class _PostgresWorker(DataSinkWorker):
    def __init__(self, sink: PostgresSink, schema: pa.Schema, fields: tuple[_FieldBinding, ...]) -> None:
        dsn, secrets = sink._resolve_connection()
        self._sink = sink
        self._schema = schema
        self._fields = fields
        self._target = (
            _quote_identifier(sink.table)
            if sink.schema is None
            else f"{_quote_identifier(sink.schema)}.{_quote_identifier(sink.table)}"
        )
        self._stage_name = f"vane_stage_{uuid.uuid4().hex}"
        self._stage = _quote_identifier(self._stage_name)
        self._stage_created = False
        self._warning_pending = True
        self._base_metadata: dict[str, Any] = {
            "provider": "postgres",
            "table": sink.table,
            "write_mode": "upsert",
        }
        if sink.schema is not None:
            self._base_metadata["schema"] = sink.schema
        driver_version = _adbc_postgresql_version()
        # ADBC PostgreSQL before 1.13 writes NULL list elements as 0, "" or false (apache/arrow-adbc#4846).
        self._null_list_elements_unsupported = driver_version is not None and driver_version < (1, 13)
        self._connection: Any | None = _open_connection(dsn, secrets)
        del dsn, secrets
        try:
            self._merge_sql = self._prepare()
        except BaseException as error:
            try:
                self.close()
            except BaseException as close_error:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    try:
                        add_note(f"PostgreSQL connection cleanup also failed: {type(close_error).__name__}")
                    except BaseException:
                        pass
            raise

    def _connection_or_raise(self) -> Any:
        if self._connection is None:
            raise RuntimeError("PostgresSink worker is closed")
        return self._connection

    def _prepare(self) -> str:
        connection = self._connection_or_raise()
        with connection.cursor() as cursor:
            cursor.execute("SELECT set_config('statement_timeout', $1, false)", (f"{self._sink.timeout}s",))
            cursor.execute(_TARGET_TYPES_SQL, (self._target,))
            target_types = dict(cursor.fetchall())
        connection.commit()
        if not target_types:
            raise ValueError(f"PostgreSQL table {self._target} does not exist or has no columns")
        missing = [field.target_name for field in self._fields if field.target_name not in target_types]
        if missing:
            raise ValueError(f"PostgreSQL table {self._target} has no columns named {missing!r}")

        columns = ", ".join(_quote_identifier(field.target_name) for field in self._fields)
        # format_type() renders catalog type names in valid, already quoted SQL. The names carry no length,
        # precision or dimension: an explicit cast to varchar(3) truncates, and a truncated key would update
        # another row. The column's own typmod is then applied by the INSERT, which rejects such values.
        values = ", ".join(
            f"CAST({_quote_identifier(f'c{index}')} AS {target_types[field.target_name]})"
            for index, field in enumerate(self._fields)
        )
        keys = ", ".join(_quote_identifier(field.target_name) for field in self._fields if field.is_key)
        updates = ", ".join(
            f"{_quote_identifier(field.target_name)} = EXCLUDED.{_quote_identifier(field.target_name)}"
            for field in self._fields
            if not field.is_key
        )
        action = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
        # The alias keeps EXCLUDED unambiguous for a target table named "excluded".
        return (
            f"INSERT INTO {self._target} AS {_TARGET_ALIAS} ({columns}) SELECT {values} FROM {self._stage} "
            f"ON CONFLICT ({keys}) {action}"
        )

    def _validate_table(self, table: pa.Table) -> tuple[int, int]:
        if not isinstance(table, pa.Table):
            raise TypeError(f"PostgresSink expected pyarrow.Table, got {type(table).__name__}")
        if len(table.schema) != len(self._schema) or any(
            batch_field.name != bound_field.name or not same_input_type(batch_field.type, bound_field.type)
            for batch_field, bound_field in zip(table.schema, self._schema, strict=True)
        ):
            raise ValueError("PostgreSQL batch schema does not match the bound input schema")
        row_count = table.num_rows
        batch_bytes = table.nbytes
        if row_count > self._sink.max_batch_rows:
            raise ValueError("PostgreSQL batch exceeds max_batch_rows")
        if batch_bytes > self._sink.max_batch_bytes:
            raise ValueError("PostgreSQL batch exceeds max_batch_bytes")
        return row_count, batch_bytes

    def _wire_table(self, table: pa.Table) -> pa.Table:
        arrays: list[pa.ChunkedArray] = []
        for index, binding in enumerate(self._fields):
            column = table.column(index)
            if column.type != binding.wire_type:
                try:
                    column = pc.cast(column, binding.wire_type, safe=True)
                except pa.ArrowException as error:
                    raise ValueError(
                        f"PostgreSQL source field {binding.source_name!r} cannot be sent without changing its "
                        f"values as Arrow type {binding.wire_type}: {error}"
                    ) from error
            if (
                self._null_list_elements_unsupported
                and _is_list_type(column.type)
                and pc.list_flatten(column).null_count
            ):
                raise ValueError(
                    f"PostgreSQL source field {binding.source_name!r} contains NULL list elements, which "
                    "adbc-driver-postgresql before 1.13 writes as 0, an empty string or false; install "
                    "adbc-driver-postgresql 1.13 or later, which requires Python 3.11"
                )
            arrays.append(column)
        # Positional staging names keep arbitrary target identifiers out of the ingest path.
        return pa.Table.from_arrays(arrays, names=[f"c{index}" for index in range(len(arrays))])

    def _merge(self, wire_table: pa.Table) -> int:
        connection = self._connection_or_raise()
        with connection.cursor() as cursor:
            cursor.adbc_ingest(
                self._stage_name,
                wire_table,
                mode="append" if self._stage_created else "create",
                temporary=True,
            )
            cursor.execute(self._merge_sql)
            affected = cursor.rowcount
            cursor.execute(f"TRUNCATE {self._stage}")
        if isinstance(affected, bool) or not isinstance(affected, int) or affected < 0:
            raise RuntimeError("PostgreSQL upsert returned an invalid row count")
        return affected

    def _rollback(self, error: BaseException) -> None:
        connection = self._connection
        if connection is None:
            return
        try:
            connection.rollback()
        except BaseException as rollback_error:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                try:
                    add_note(f"PostgreSQL rollback also failed: {safe_exception_type_name(rollback_error)}")
                except BaseException:
                    pass

    def write(self, table: pa.Table) -> WriteResult:
        row_count, batch_bytes = self._validate_table(table)
        if row_count == 0:
            return WriteResult(
                rows_received=0,
                rows_affected=0,
                bytes_received=batch_bytes,
                metadata=self._base_metadata,
            )
        wire_table = self._wire_table(table)
        committing = False
        try:
            affected = self._merge(wire_table)
            committing = True
            self._connection_or_raise().commit()
        except BaseException as error:
            if committing:
                message = (
                    "PostgreSQL batch commit did not report its outcome; the batch may be applied, and replaying "
                    "it upserts the same keys"
                )
            else:
                self._rollback(error)
                message = "PostgreSQL batch was rolled back"
            message = f"{message}: {safe_exception_type_name(error)}"
            detail = exception_message_from_args(error)
            if detail:
                message = f"{message}: {bounded_utf8_text(detail, 512)}"
            if isinstance(error, Exception):
                raise RuntimeError(message) from error
            # Keep cancellation/interrupt identity and classification. DataSink
            # diagnostics read args, not exception notes.
            BaseException.__setattr__(error, "args", (message,))
            raise
        self._stage_created = True
        warnings = (_PARTIAL_VISIBILITY_WARNING,) if self._warning_pending else ()
        self._warning_pending = False
        return WriteResult(
            rows_received=row_count,
            rows_affected=affected,
            bytes_received=batch_bytes,
            metadata=self._base_metadata,
            warnings=warnings,
        )

    def abort(self, _error: BaseException) -> None:
        self.close()

    def close(self) -> None:
        connection = self._connection
        if connection is None:
            return
        connection.close()
        self._connection = None


__all__ = ["PostgresSink"]
