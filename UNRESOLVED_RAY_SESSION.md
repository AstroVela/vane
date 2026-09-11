# Driver-owned Ray session experiment

This draft explores an alternative to Vane's existing client-owned connection.
Use the explicit experimental API to evaluate it:

```python
from vane.experimental import connect_driver_session

with connect_driver_session() as conn:
    conn.execute("SET VARIABLE offset = 10")
    conn.execute("CREATE VIEW source AS SELECT i FROM range(5) t(i)")
    rel = conn.table("source").filter("i < 2")
    print(rel.fetchall())
    print(conn.sql("SELECT i + getvariable('offset') FROM source").fetchall())
```

`vane.connect()`, the module-level APIs, native local-fast execution, and the
existing client-bound Ray path retain their current behavior. This experiment
is not a proposal to make the driver architecture the default at this stage.

## Request and execution flow

The experimental connection records an immutable identity, environment snapshot,
database path and bootstrap options. Creating it does not initialize Ray. A SQL
call contacts the driver to parse and classify statements under that connection's
parser settings. SQL requests contain text and parameters. Relation sources and
transforms build unresolved operation trees on the client; native expressions
are encoded as parsed expression trees. Schema access and terminals bind them
on the driver. No client catalog mirror is maintained.

The driver owns the database connection, attachments, settings, catalog changes,
registered inputs and functions. Connection operations and standalone metadata
queries execute there. Eligible query/session scalars, including current query
and connection identities, schema/database names and transaction-clock values,
are captured there for distributed plans. Unsupported context-dependent
expressions combined with distributed data still fail explicitly.

After binding, the driver admits distributed plans through the existing bound
plan checks. The bound plan stays on the driver; the client receives an owned
query reference. Reads, COPY and supported table writes reuse the existing Ray
execution, commit, cancellation and teardown protocols. Results return as Arrow
partitions with native column types and formatting properties, including timezone.
The client's native result adapter decodes rows/DataFrames/Arrow without binding
or executing the user's query locally.

## Boundaries

- SQL `execute()`, parameterized `sql()`, common Relation transformations, schema
  access, native result consumers, COPY and supported table-write terminals are
  included. Module-level APIs do not accept the experimental connection.
- Database paths, input/output paths and installed extension providers are
  resolved on the driver/workers. Python locals are not implicit remote table
  names; explicitly register or pass sources.
- Distribution retains the existing capabilities. Ordinary DuckDB in-memory
  tables and temporary tables do not gain distributed read/write providers.
  Distributed execution requires auto-commit. Metadata and connection commands
  use the driver's native transaction semantics.
- SQL `CALL`, `PREPARE`, `EXECUTE`, `EXPLAIN ANALYZE`, `COPY FROM`, unsupported
  write targets/options and unimplemented API methods fail explicitly. There is
  no automatic switch to local-fast.
- A failed result conversion after a committed COPY preserves
  `CopyResultUnavailableError`; uncertain commit outcomes retain the existing
  reconciliation behavior.
- Python DataSink orchestration continues to use the existing SDK. Its schema
  requests and SQL/Relation execution reach the driver; moving arbitrary SDK
  callbacks into a remote service is outside this experiment.

## Architectural cost

Driver-owned state removes the need to synchronize a client binder with a remote
catalog, but requires a separate connection/Relation frontend, operation protocol,
remote schema analysis, server-side object lifetime management and result metadata
transport. It also changes when binding errors occur and which filesystem and
Python process an API can access.

For the current mainline, keeping connection state and binding on the client is
the smaller change. Ray mode can classify connection operations as client work
and send supported data plans to Ray without asking users to switch runners.
This draft provides a concrete driver-owned alternative for review and comparison;
it does not replace the incremental client-state approach.
