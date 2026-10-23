# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Python convenience APIs for native data catalogs.

Gravitino operations execute through the C++ extension on the supplied Vane
connection. Load the matching ``gravitino`` provider before attaching.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vane import DuckDBPyConnection, DuckDBPyRelation


def _text(value: str) -> str:
    if not isinstance(value, str) or "\0" in value:
        raise ValueError("catalog arguments must be strings without NUL")
    return value


def _literal(value: str) -> str:
    return "'" + _text(value).replace("'", "''") + "'"


def _identifier(value: str) -> str:
    if not value:
        raise ValueError("catalog identifiers cannot be empty")
    return '"' + _text(value).replace('"', '""') + '"'


class GravitinoCatalog:
    """An attached Gravitino FILESET catalog, owned by a Vane connection.

    Metadata writes require auto-commit. Dropping a MANAGED Fileset asks
    Gravitino to delete its managed files as well as its metadata.
    """

    def __init__(self, connection: DuckDBPyConnection, name: str) -> None:
        _identifier(name)
        self.connection = connection
        self.name = name

    @classmethod
    def attach(
        cls,
        connection: DuckDBPyConnection,
        name: str,
        *,
        endpoint: str,
        metalake: str,
        catalog: str,
        token: str = "",
        read_only: bool = False,
        location_name: str = "",
        timeout_ms: int = 30_000,
        max_response_bytes: int = 2 * 1024 * 1024,
    ) -> GravitinoCatalog:
        """Attach an existing remote catalog; this does not create it."""
        if type(read_only) is not bool:
            raise ValueError("read_only must be a boolean")
        for field, value, limit in (
            ("timeout_ms", timeout_ms, 300_000),
            ("max_response_bytes", max_response_bytes, 16 * 1024 * 1024),
        ):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"{field} must be an integer in 1..{limit}")
        options = [
            "TYPE gravitino",
            f"ENDPOINT {_literal(endpoint)}",
            f"METALAKE {_literal(metalake)}",
            "TOKEN $token",
            f"LOCATION_NAME {_literal(location_name)}",
            f"READ_ONLY {'true' if read_only else 'false'}",
            f"TIMEOUT_MS {timeout_ms}",
            f"MAX_RESPONSE_BYTES {max_response_bytes}",
        ]
        connection.execute(
            f"ATTACH {_literal(catalog)} AS {_identifier(name)} ({', '.join(options)})", {"token": _text(token)}
        )
        return cls(connection, name)

    def detach(self) -> None:
        self.connection.execute(f"DETACH {_identifier(self.name)}")

    def metadata(self) -> dict[str, Any]:
        return self._metadata("gravitino_catalog")

    def list_schemas(self) -> list[str]:
        return self._list("gravitino_schemas")

    def load_schema(self, name: str) -> dict[str, Any]:
        return self._metadata("gravitino_schema", name)

    def create_schema(self, name: str) -> None:
        self.connection.execute(f"CREATE SCHEMA {_identifier(self.name)}.{_identifier(name)}")

    def alter_schema(self, name: str, changes: Sequence[Mapping[str, str]]) -> None:
        self._command("gravitino_alter_schema", name, self._json({"updates": changes}))

    def drop_schema(self, name: str, *, cascade: bool = False) -> None:
        if type(cascade) is not bool:
            raise ValueError("cascade must be a boolean")
        suffix = " CASCADE" if cascade else ""
        self.connection.execute(f"DROP SCHEMA {_identifier(self.name)}.{_identifier(name)}{suffix}")

    def alter_catalog(self, changes: Sequence[Mapping[str, str]]) -> None:
        self._command("gravitino_alter_catalog", self._json({"updates": changes}))

    def list_filesets(self, schema: str) -> list[str]:
        return self._list("gravitino_filesets", schema)

    def load_fileset(self, schema: str, name: str) -> dict[str, Any]:
        return self._metadata("gravitino_fileset", schema, name)

    def create_fileset(
        self,
        schema: str,
        name: str,
        *,
        storage_location: str | None = None,
        storage_locations: Mapping[str, str] | None = None,
        kind: Literal["EXTERNAL", "MANAGED"] = "EXTERNAL",
        properties: Mapping[str, str] | None = None,
        comment: str = "",
    ) -> None:
        body: dict[str, Any] = {
            "name": name,
            "type": kind,
            "properties": {} if properties is None else properties,
            "comment": comment,
        }
        if storage_location is not None:
            body["storageLocation"] = storage_location
        if storage_locations is not None:
            body["storageLocations"] = storage_locations
        self._command("gravitino_create_fileset", schema, self._json(body))

    def alter_fileset(self, schema: str, name: str, changes: Sequence[Mapping[str, str]]) -> None:
        self._command("gravitino_alter_fileset", schema, name, self._json({"updates": changes}))

    def drop_fileset(self, schema: str, name: str) -> None:
        self._command("gravitino_drop_fileset", schema, name)

    def files(self, schema: str, name: str, path: str = "", *, recursive: bool = False) -> DuckDBPyRelation:
        """Resolve and list Fileset contents using Vane's native FILE scan.

        The resulting plan contains physical storage paths. Ray workers need
        access to those paths and storage credentials, not Gravitino access.
        """
        if type(recursive) is not bool:
            raise ValueError("recursive must be a boolean")
        args = ", ".join(_literal(arg) for arg in (self.name, schema, name, path))
        return self.connection.sql(
            f"SELECT * FROM gravitino_files({args}, recursive={'true' if recursive else 'false'})"
        )

    def _list(self, function: str, *args: str) -> list[str]:
        rows = self._query(function, *args).fetchall()
        return [row[0] for row in rows]

    def _metadata(self, function: str, *args: str) -> dict[str, Any]:
        row = self._query(function, *args).fetchone()
        if row is None:
            raise RuntimeError("Gravitino metadata query returned no resource")
        return json.loads(row[1])

    def _query(self, function: str, *args: str) -> DuckDBPyRelation:
        # SQL binding releases the GIL while the native connector performs I/O.
        values = ", ".join(_literal(arg) for arg in (self.name, *args))
        return self.connection.sql(f"SELECT * FROM {function}({values})")

    def _command(self, function: str, *args: str) -> None:
        values = ", ".join(_literal(arg) for arg in (self.name, *args))
        self.connection.execute(f"PRAGMA {function}({values})")

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class GravitinoIdentifier:
    """A resource name and its complete parent namespace, without dotted parsing."""

    namespace: tuple[str, ...]
    name: str


CapabilityState = Literal["supported", "unsupported", "unverified"]


@dataclass(frozen=True)
class GravitinoCapabilities:
    """Metadata access and Gravitino relational table execution qualification."""

    metadata_read: CapabilityState
    table_read: CapabilityState
    table_write: CapabilityState


@dataclass(frozen=True)
class GravitinoResource:
    """An observed metadata description, not an executable relation or snapshot.

    Credential-like properties are redacted by the native connector. The local
    metadata dictionary can be inspected or edited without changing Gravitino.
    """

    identifier: GravitinoIdentifier
    kind: Literal["catalog", "schema", "table"]
    catalog_type: str
    provider: str
    format: str | None
    capabilities: GravitinoCapabilities
    metadata: dict[str, Any]


class GravitinoMetalake:
    """Read-only metalake discovery backed by a private native attachment.

    Uses the supplied connection's runner. The existing Gravitino provider must
    be loaded first. Discovery does not attach a FILESET catalog or import its
    storage configuration, and descriptions do not execute table reads/writes.
    """

    def __init__(self, connection: DuckDBPyConnection, name: str, metalake: str) -> None:
        _identifier(name)
        self.connection = connection
        self.metalake = metalake
        self._name = name
        self._closed = False

    @classmethod
    def connect(
        cls,
        connection: DuckDBPyConnection,
        *,
        endpoint: str,
        metalake: str,
        token: str = "",
        timeout_ms: int = 30_000,
        max_response_bytes: int = 2 * 1024 * 1024,
    ) -> GravitinoMetalake:
        """Verify a metalake and create a connection-owned metadata handle.

        The internal READ_ONLY attachment disappears when this handle is closed
        or its underlying database is released. No remote resource is created or removed.
        Existing static Bearer tokens are optional; no credentials are vended.
        """
        for field, value, limit in (
            ("timeout_ms", timeout_ms, 300_000),
            ("max_response_bytes", max_response_bytes, 16 * 1024 * 1024),
        ):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"{field} must be an integer in 1..{limit}")
        name = "__vane_gravitino_metalake_" + uuid.uuid4().hex
        connection.execute(
            f"ATTACH {_literal(metalake)} AS {_identifier(name)} "
            "(TYPE gravitino_metalake, READ_ONLY, ENDPOINT $endpoint, TOKEN $token, "
            "TIMEOUT_MS $timeout, MAX_RESPONSE_BYTES $limit)",
            {
                "endpoint": _text(endpoint),
                "token": _text(token),
                "timeout": timeout_ms,
                "limit": max_response_bytes,
            },
        )
        return cls(connection, name, metalake)

    def close(self) -> None:
        """Detach this local metadata handle; successful closes are idempotent."""
        if not self._closed:
            self.connection.execute(f"DETACH {_identifier(self._name)}")
            self._closed = True

    def __enter__(self) -> GravitinoMetalake:
        self._check_open()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def list_catalogs(self) -> list[GravitinoIdentifier]:
        return self._list("catalogs")

    def load_catalog(self, catalog: str) -> GravitinoResource:
        return self._load("catalog", catalog)

    def list_schemas(self, catalog: str) -> list[GravitinoIdentifier]:
        return self._list("schemas", catalog)

    def load_schema(self, catalog: str, schema: str) -> GravitinoResource:
        return self._load("schema", catalog, schema)

    def list_tables(self, catalog: str, schema: str) -> list[GravitinoIdentifier]:
        return self._list("tables", catalog, schema)

    def load_table(self, catalog: str, schema: str, table: str) -> GravitinoResource:
        return self._load("table", catalog, schema, table)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Gravitino metalake discovery connection is closed")

    def _query(self, operation: str, *args: str) -> list[tuple[Any, ...]]:
        self._check_open()
        parameters = ", ".join("?" for _ in range(len(args) + 1))
        return self.connection.execute(
            f"SELECT * FROM gravitino_metalake_{operation}({parameters})",
            [self._name, *(_text(value) for value in args)],
        ).fetchall()

    def _list(self, operation: str, *args: str) -> list[GravitinoIdentifier]:
        return [GravitinoIdentifier(tuple(row[0]), row[1]) for row in self._query(operation, *args)]

    def _load(self, operation: str, *args: str) -> GravitinoResource:
        rows = self._query(operation, *args)
        if len(rows) != 1:
            raise RuntimeError("Gravitino discovery returned an unexpected detail result")
        namespace, name, kind, catalog_type, provider, encoded = rows[0]
        catalog_type = catalog_type.upper()
        metadata = json.loads(encoded)
        properties = metadata.get("properties", {})
        format = properties.get("format") if isinstance(properties, dict) else None
        if not isinstance(format, str) or not format:
            format = "iceberg" if provider == "lakehouse-iceberg" else None
        # These flags describe relational table execution through Gravitino.
        # Fileset I/O has its own attached-catalog API and support matrix.
        known_non_table = catalog_type.upper() in {"FILESET", "MODEL", "MESSAGING"}
        known_unsupported = provider in {"lakehouse-generic", "lakehouse-hive", "jdbc-postgresql", "jdbc-mysql"}
        table_state: CapabilityState = "unsupported" if known_non_table or known_unsupported else "unverified"
        return GravitinoResource(
            identifier=GravitinoIdentifier(tuple(namespace), name),
            kind=kind,
            catalog_type=catalog_type,
            provider=provider,
            format=format,
            capabilities=GravitinoCapabilities("supported", table_state, table_state),
            metadata=metadata,
        )
