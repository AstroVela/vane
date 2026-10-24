# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Python convenience APIs for native data catalogs.

Catalog operations execute through the native extension on the supplied Vane
connection. Load the matching ``gravitino`` or ``iceberg`` provider before
attaching.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vane import DuckDBPyConnection, DuckDBPyRelation

_IDENTIFIER_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


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


class IcebergCatalog:
    """A read-only Iceberg REST attachment on a Vane connection.

    Load the matching Iceberg provider before :meth:`attach`. The connection
    owns the attachment and the returned relations; this object never closes
    it. Existing attachments can be wrapped only when they are Iceberg and
    READ_ONLY. A detached or replaced attachment invalidates the wrapper.

    Anonymous Gravitino REST with local Parquet storage is qualified in
    ``GRAVITINO_ICEBERG.md``. Ray supports frozen unpartitioned format-v2
    snapshots with one flat schema and no deletes, using shared local Parquet.
    """

    def __init__(self, connection: DuckDBPyConnection, name: str) -> None:
        _identifier(name)
        self.connection = connection
        self.name = name
        row = self._attachment()
        if row is None or row[1] != "iceberg" or not row[2]:
            raise ValueError("IcebergCatalog requires an existing READ_ONLY Iceberg attachment")
        self._database_oid = row[0]

    @classmethod
    def attach(
        cls,
        connection: DuckDBPyConnection,
        name: str,
        *,
        endpoint: str,
        warehouse: str = "",
    ) -> IcebergCatalog:
        """Attach an existing anonymous REST catalog, always READ_ONLY.

        ``endpoint`` is the explicit HTTP(S) Iceberg REST service URL, including
        its service prefix. ``warehouse`` is passed to the native catalog's
        warehouse routing; an empty string selects the server default. Neither
        value is inferred from Gravitino's separate metadata API.
        """
        alias = _identifier(name)
        _text(warehouse)
        endpoint = _text(endpoint)
        if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in endpoint):
            raise ValueError("endpoint must be an HTTP(S) REST URL without credentials, query or fragment")
        try:
            parsed = urlsplit(endpoint)
            valid = (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and (parsed.port is None or parsed.port > 0)
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
                and "?" not in endpoint
                and "#" not in endpoint
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("endpoint must be an HTTP(S) REST URL without credentials, query or fragment")
        loaded = connection.execute(
            "SELECT loaded FROM duckdb_extensions() WHERE extension_name = 'iceberg'"
        ).fetchone()
        if loaded is None or not loaded[0]:
            raise RuntimeError("load the matching Iceberg provider with vane.load_installed_extension before attaching")
        connection.execute(
            f"ATTACH {_literal(warehouse)} AS {alias} "
            "(TYPE iceberg, ENDPOINT $endpoint, AUTHORIZATION_TYPE 'none', READ_ONLY)",
            {"endpoint": endpoint},
        )
        return cls(connection, name)

    def table(self, namespace: str, name: str, *, snapshot_id: int | None = None) -> DuckDBPyRelation:
        """Bind an existing table as a composable native relation.

        Names are individual native catalog/schema/table identifiers, never SQL
        fragments. ``namespace`` is the schema name exposed by the provider.
        With no snapshot ID, native transaction/snapshot selection applies.
        A positive signed 64-bit snapshot ID selects that exact version.
        """
        qualified = ".".join(_identifier(part) for part in (self.name, namespace, name))
        if snapshot_id is not None and (type(snapshot_id) is not int or not 0 < snapshot_id < 2**63):
            raise ValueError("snapshot_id must be a positive signed 64-bit integer")
        self._check_attachment()
        at = "" if snapshot_id is None else f" AT (VERSION => {snapshot_id})"
        return self.connection.sql(f"SELECT * FROM {qualified}{at}")

    def detach(self) -> None:
        """Detach this attachment, keeping the caller's connection open."""
        self._check_attachment()
        self.connection.execute(f"DETACH {_identifier(self.name)}")

    def _attachment(self) -> tuple[int, str, bool] | None:
        # Match the engine's ASCII-only identifier comparison. SQL lower() and
        # Python casefold() would conflate distinct non-ASCII catalog names.
        key = self.name.translate(_IDENTIFIER_LOWER)
        rows = self.connection.execute(
            "SELECT database_name, database_oid, type, readonly FROM duckdb_databases()"
        ).fetchall()
        for name, oid, kind, read_only in rows:
            if name.translate(_IDENTIFIER_LOWER) == key:
                return oid, kind, read_only
        return None

    def _check_attachment(self) -> None:
        row = self._attachment()
        if row is None or row[0] != self._database_oid or row[1] != "iceberg" or not row[2]:
            raise ValueError("IcebergCatalog attachment has been detached or replaced")


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
