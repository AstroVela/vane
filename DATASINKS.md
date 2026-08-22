# Python DataSink contract

Vane's `DataSink` API is a provider-neutral terminal write boundary for Python
services. It can back adapters for vector databases or other remote systems
without adding their SDKs to Vane core.

The initial protocol is deliberately narrow:

- writes are immediately visible to the external service;
- every bound sink must declare `RetryMode.IDEMPOTENT`;
- a retry or speculative/losing attempt can repeat a batch, so no exactly-once
  guarantee is made;
- `DataSinkWriter.abort()` only cleans up worker-local resources and is not a
  rollback of remote effects;
- a failure after execution starts has outcome `UNKNOWN`, even when some
  selected worker results are available.

Two-phase commit and non-idempotent sinks are rejected before the execution
plan starts. There is no fallback to local execution.

## Implement a sink

Implement `DataSink.bind()`, a serializable `BoundDataSink`, and a short-lived
`DataSinkWriter`. A writer receives a stable operation identity and one Arrow
table. It must make reapplying that input converge on the same external state.

```python
import pyarrow as pa

import vane
from vane.datasink import (
    BoundDataSink,
    CommitProtocol,
    DataSink,
    DataSinkCapabilities,
    DataSinkWriter,
    RetryMode,
    WriteResult,
)


class MyWriter(DataSinkWriter):
    def __init__(self, client, operation_id: str):
        self.client = client
        self.operation_id = operation_id

    def write(self, table: pa.Table) -> WriteResult:
        # Use stable row keys and operation_id in an idempotent provider call.
        affected = self.client.upsert(table, operation_id=self.operation_id)
        return WriteResult(rows_received=table.num_rows, rows_affected=affected)


class MyBoundSink(BoundDataSink):
    @property
    def capabilities(self):
        return DataSinkCapabilities(CommitProtocol.IMMEDIATE, RetryMode.IDEMPOTENT)

    def open_writer(self, context):
        return MyWriter(make_client(), context.operation_id)


class MySink(DataSink):
    def bind(self, schema: pa.Schema):
        validate_schema(schema)
        return MyBoundSink()


summary = vane.sql("SELECT id, embedding FROM items").write_datasink(MySink())
assert summary.outcome is vane.WriteOutcome.APPLIED
```

`WriteResult` metadata, warnings, and result counts are bounded before they
cross the coordinator boundary. `rows_received` must exactly match the Arrow
batch passed to the writer. A `close()` failure after a successful write is
reported as a warning rather than changing a known applied external effect.

## Keyed upserts

Subclass `BoundKeyedUpsertSink` and declare `key_columns` for sinks such as
Turbopuffer, Milvus, or Qdrant that upsert by stable keys. Vane inserts global
window barriers that reject NULL or duplicate keys before any writer is
opened. This makes the input order-independent, but the provider adapter still
owns the idempotent upsert implementation.

## Secrets

Pass credential references, not credential values, in a bound sink.
`EnvironmentSecret("MY_SERVICE_TOKEN")` serializes only the environment
variable name and resolves its value on the worker. Custom adapters must follow
the same rule for secret-manager references. Driver and workers remain within
Vane's trusted execution boundary; do not place plaintext credentials in sink
objects, result metadata, warnings, or exceptions.

## Runners and outcomes

`local-fast`, local FTE, and Ray use the same writer result schema. Ray and
local FTE collect only the selected task attempts, while unselected attempts
may already have contacted the service. Consequently:

- `APPLIED` means all selected results were valid and execution completed;
- `ABORTED` is reserved for failures proven to occur before external execution;
- `UNKNOWN` means external visibility cannot be proven and is raised as
  `DataSinkWriteError` with the partial selected results, if any.

Because the only accepted retry mode is idempotent, retrying an `UNKNOWN`
operation with the same explicit `operation_id` is safe according to the
adapter's declared contract.
