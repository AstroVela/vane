# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real S3 SDK commits and anonymous Vane reads with no local dataset copy."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import threading
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import vane

S3_TABLES = ("items", "empty", "identity", "schema", "positions", "equality", "all_deleted")


class ObjectForwarder:
    def __init__(self, endpoint, directory):
        token = uuid.uuid4().hex
        self.audit = directory / f"s3-audit-{token}.jsonl"
        self.control = directory / f"s3-control-{token}.json"
        self.audit.touch()
        self.policy()
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                str(Path(__file__).with_name("iceberg_s3_proxy.py")),
                endpoint,
                str(self.audit),
                str(self.control),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        timeout = threading.Timer(10, self.process.kill)
        timeout.start()
        try:
            line = self.process.stdout.readline()
            if not line:
                raise AssertionError("S3 forwarder failed: " + self.process.stderr.read())
            self.endpoint = json.loads(line)["endpoint"]
        except BaseException:
            self.stop()
            raise
        finally:
            timeout.cancel()
            timeout.join()

    def policy(self, *, deny_metadata=False, denied=()):
        pending = self.control.with_suffix(".pending")
        pending.write_text(json.dumps({"deny_metadata": deny_metadata, "denied": list(denied)}))
        pending.replace(self.control)

    def events(self):
        return [json.loads(line) for line in self.audit.read_text().splitlines()]

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate(timeout=5)
            raise AssertionError("S3 forwarder failed to stop") from None


@dataclass(frozen=True)
class S3Tables:
    namespace: str
    directory: Path  # Control files only; Parquet/Avro/metadata stay in S3.
    catalog: Any
    location: str
    endpoint: str
    region: str
    objects: ObjectForwarder
    config: dict = field(repr=False)


def connection(tables, *, endpoint=None):
    return vane.connect(
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "allow_persistent_secrets": "false",
            "secret_directory": str(tables.directory / "secrets"),
            "s3_endpoint": urlsplit(endpoint or tables.objects.endpoint).netloc,
            "s3_region": tables.region,
            "s3_url_style": "path",
            "s3_use_ssl": "false",
            "s3_access_key_id": "",
            "s3_secret_access_key": "",
            "s3_session_token": "",
            "http_retries": "0",
        }
    )


def attach(con, endpoint):
    from vane.catalog import IcebergCatalog

    vane.load_installed_extension("iceberg", connection=con)
    return IcebergCatalog.attach(con, "lake", endpoint=endpoint)


def append_rows(tables, name, rows, *, omit_identity=False):
    """Use SDK partition transforms and commits, with opaque non-Hive S3 keys."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyiceberg.io.pyarrow import schema_to_pyarrow
    from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
    from pyiceberg.partitioning import partition_record_value
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.typedef import Record

    table = tables.catalog.load_table((tables.namespace, name))
    spec, schema = table.spec(), table.schema()
    grouped = defaultdict(list)
    for row in rows:
        values = tuple(
            field.transform.transform(schema.find_field(field.source_id).field_type)(
                partition_record_value(
                    partition_field=field, value=row[schema.find_field(field.source_id).name], schema=schema
                )
            )
            for field in spec.fields
        )
        grouped[values].append(row)
    omitted = {
        field.source_id for field in spec.fields if omit_identity and isinstance(field.transform, IdentityTransform)
    }
    physical = Schema(*(field for field in schema.fields if field.field_id not in omitted))
    with table.transaction() as transaction:
        with transaction.update_snapshot().fast_append() as append:
            for values, batch in grouped.items():
                path = f"{table.location()}/region=wrong.点/opaque-{uuid.uuid4().hex}.parquet"
                payload = pa.Table.from_pylist(
                    [{field.name: row[field.name] for field in physical.fields} for row in batch],
                    schema=schema_to_pyarrow(physical),
                )
                with table.io.new_output(path).create() as stream:
                    pq.write_table(payload, stream, row_group_size=2)
                data_file = DataFile.from_args(
                    content=DataFileContent.DATA,
                    file_path=path,
                    file_format=FileFormat.PARQUET,
                    partition=Record(*values),
                    record_count=len(batch),
                    file_size_in_bytes=len(table.io.new_input(path)),
                )
                data_file.spec_id = spec.spec_id
                append.append_data_file(data_file)
    table.refresh()
    return table


def populate(tables, *, names=S3_TABLES):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from iceberg_delete_helpers import (
        PATH_ID,
        POS_ID,
        DeleteFeature,
        commit_deletes,
        delete_reference,
        manifest_entries,
    )
    from pyiceberg.io.pyarrow import schema_to_pyarrow
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import IntegerType, LongType, NestedField, StringType

    schema = Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "label", StringType()),
        NestedField(3, "region", StringType()),
        NestedField(4, "rank", IntegerType()),
    )
    features = {}
    for name in names:
        spec = (
            PartitionSpec(
                PartitionField(source_id=3, field_id=1000, transform=IdentityTransform(), name="region_partition")
            )
            if name == "identity"
            else PartitionSpec()
        )
        table = tables.catalog.create_table(
            (tables.namespace, name),
            schema=schema,
            partition_spec=spec,
            location=f"{tables.location}/{name}",
            properties={"format-version": "2"},
        )
        if name == "empty":
            features[name] = DeleteFeature(name, delete_reference(tables, name), (), 0)
            continue
        rows = [
            {
                "id": key,
                "label": None if key % 7 == 0 else f"row-{key}.点",
                "region": ("east.点", "west", None)[key % 3],
                "rank": key % 4,
            }
            for key in range(27)
        ]
        history = []
        step = 9 if name == "identity" else 3
        for start in range(0, len(rows), step):
            table = append_rows(tables, name, rows[start : start + step], omit_identity=True)
            if not history:
                history.append(delete_reference(tables, name))
        if name == "schema":
            with table.update_schema() as update:
                update.rename_column("label", "renamed_label")
                update.delete_column("region")
                update.update_column("rank", field_type=LongType())
            with table.update_schema() as update:
                update.add_column("region", StringType())
            table = append_rows(
                tables, name, [{"id": 101, "renamed_label": "new", "rank": 2**33, "region": "new-field"}]
            )
        elif name == "positions":
            entries = [entry for _manifest, entry in manifest_entries(table) if entry.data_file.content == 0]
            commit_deletes(tables, name, [{PATH_ID: entry.data_file.file_path, POS_ID: 1} for entry in entries])
        elif name in {"equality", "all_deleted"}:
            if name == "equality":
                commit_deletes(tables, name, [{2: None, 4: 0}, {2: "row-9.点", 4: 1}], equality_ids=(2, 4))
            else:
                commit_deletes(tables, name, [{1: key} for key in range(27)], equality_ids=(1,))
        table = tables.catalog.load_table((tables.namespace, name))
        current = delete_reference(tables, name)
        # Independent SDK scan also checks every case it implements. Equality
        # deletes use the separate field-ID/Arrow evaluator above.
        if name not in {"equality", "all_deleted"}:
            observed = table.scan().to_arrow().sort_by("id").to_pylist()
            assert tuple(tuple(row[column] for column in current.names) for row in observed) == current.rows
        files = sum(entry.data_file.content == 0 for _manifest, entry in manifest_entries(table))
        features[name] = DeleteFeature(name, current, tuple(history), files)
        orphan = f"{table.location()}/region=wrong.点/uncommitted.parquet"
        with table.io.new_output(orphan).create() as stream:
            pq.write_table(pa.Table.from_pylist([rows[0]], schema=schema_to_pyarrow(schema)), stream)
    return features


@contextmanager
def s3_tables(directory, *, names=S3_TABLES):
    import pytest

    path = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_S3_CONFIG")
    if not path:
        pytest.skip("set VANE_TEST_GRAVITINO_ICEBERG_S3_CONFIG to an owned MinIO fixture config")
    from pyiceberg.catalog.rest import RestCatalog

    config = json.loads(Path(path).read_text())
    namespace = "vane_s3_ci_" + uuid.uuid4().hex
    properties = {
        "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
        "header.X-Iceberg-Access-Delegation": "",
        "s3.endpoint": config["s3_endpoint"],
        "s3.region": config["s3_region"],
        "s3.access-key-id": config["admin_key"],
        "s3.secret-access-key": config["admin_secret"],
    }
    catalog = RestCatalog(namespace, uri=config["endpoint"], auth={"type": "noop"}, **properties)
    objects = ObjectForwarder(config["s3_endpoint"], directory)
    tables = S3Tables(
        namespace,
        directory,
        catalog,
        f"s3://{config['bucket']}/{namespace}",
        config["endpoint"],
        config["s3_region"],
        objects,
        config,
    )
    created = False
    try:
        catalog.create_namespace(namespace)
        created = True
        yield tables, populate(tables, names=names)
    finally:
        try:
            if created:
                for identifier in catalog.list_tables(namespace):
                    catalog.drop_table(identifier)
                catalog.drop_namespace(namespace)
                # Only this unique namespace's objects; the bucket is never
                # emptied by tests, and the test configuration stays private.
                import pyarrow.fs as pafs
                from pyiceberg.io.pyarrow import PyArrowFileIO

                filesystem = PyArrowFileIO(properties).fs_by_scheme("s3", config["bucket"])
                prefix = f"{config['bucket']}/{namespace}"
                if filesystem.get_file_info(prefix).type != pafs.FileType.NotFound:
                    filesystem.delete_dir(prefix)
        finally:
            catalog.close()
            objects.stop()


def execute_frozen(payload, tables, *, split_batch=None):
    # Use the binding connection resolved from the transported bootstrap. A
    # clone explicitly bound to an unrelated cursor would override that owner.
    with vane.connect(
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "allow_persistent_secrets": "false",
        }
    ) as worker:
        logical = pickle.loads(payload)
        physical = logical.to_physical_plan(worker)
        batch_map = physical.scan_split_batch_map()
        assert len(batch_map) == 1
        assignment = {
            str(node): vane.ray_cxx.merge_scan_split_batches(batches) if split_batch is None else split_batch
            for node, batches in batch_map.items()
        }
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
        try:
            result = runner.execute_native(worker, physical, scan_split_batch=assignment)
            assert result.completion_status in {"ok", "empty"}
            rows = []
            for part in result.partition_payloads:
                table = part.to_arrow() if hasattr(part, "to_arrow") else part
                rows.extend(zip(*(column.to_pylist() for column in table.columns), strict=True))
            schema = result.result_schema
            result = physical = logical = None
            return rows, schema
        finally:
            runner.shutdown()


@contextmanager
def frozen_physical(tables, name, *, endpoint=None):
    from iceberg_scan_helpers import freeze

    with connection(tables) as con:
        attach(con, endpoint or tables.endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}"."{name}"')
    with vane.connect(
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "allow_persistent_secrets": "false",
        }
    ) as worker:
        logical = pickle.loads(payload)
        physical = logical.to_physical_plan(worker)
        node, batches = next(iter(physical.scan_split_batch_map().items()))
        try:
            yield logical, physical, str(node), [bytes(batch) for batch in batches], payload
        finally:
            physical = logical = None
