# Local model pool lifetime

This document describes the internal execution interfaces for explicitly
registered, session-owned subprocess models. The roadmap is tracked in
[#838](https://github.com/AstroVela/vane/issues/838), with model ownership in
[#840](https://github.com/AstroVela/vane/issues/840). The connection runtime
exposes explicit CPU and fixed-device GPU model registration for ordinary SQL
and Relation queries.
Model registration and execution use the native connection APIs below. They do
not expose serialized plan execution or create a serving endpoint. Ordinary
`query()` uses a separate QueryRuntime; a session cannot install both runtimes.

The [CPU serving acceptance scenario](LOCAL_SERVING_ACCEPTANCE.md) exercises
these interfaces together with synthetic text/RGB UDFs, concurrent requests,
slow consumers, cancellation, expiry, and worker loss. It produces a JSON
report with initialization counts, latency distributions, and resource
checkpoints using an installed wheel.
Its [sustained lifecycle mode](LOCAL_SERVING_ACCEPTANCE.md#sustained-lifecycle-acceptance)
repeats load and fault recovery in one runtime under an independent process
watchdog, retaining bounded failure diagnostics and an acceptance map.
The [integrated stage acceptance record](LOCAL_SERVING_ACCEPTANCE.md#integrated-stage-acceptance-2026-10-02)
identifies the merged CPU/CUDA and streaming version, its validation evidence,
and the unresolved historical startup incident.

## Shared runtime for ordinary local queries

Enable session-wide request, task and UDF byte admission through the existing
connection. Configure it once on the owning connection, after connection setup
and before creating cursors. The configuration is immutable for that session.
For example:

```python
import vane
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.udf_runtime_admission import TaskAdmissionLimits
from vane.execution.udf_data_admission import DataAdmissionLimits, DataAdmissionWaitLimits

with vane.connect() as connection:
    runtime = connection.configure_local_runtime(
        request_limit=RequestAdmissionLimits(2, 8, queue_timeout=10.0),
        task_limit=TaskAdmissionLimits(2, 16),
        data_limit=DataAdmissionLimits(
            64 * 1024 * 1024, 8 * 1024 * 1024, 8 * 1024 * 1024,
            wait=DataAdmissionWaitLimits(16, 10.0),
        ),
        execution_timeout=30.0,
    )
    with connection.cursor() as cursor:
        rows = cursor.sql("SELECT sum(i) FROM range(100) t(i)").fetchall()
    print(runtime.resource_snapshot())
```

`execute()`, lazy SQL relations, and Relation operations share this path. A lazy
relation acquires its request when execution starts. Concurrent clients use
independent cursors belonging to the same connection. Admission precedes the
execution's native preparation; lazy relation construction and initial binding
remain outside the request budget. Preparation collects every subprocess UDF once and supplies
the session's captured configuration. A different connection has a distinct
runtime even if it opens the same database. Unconfigured connections retain
their existing behavior.

The connection integration supports auto-commit, read-only SELECT and Relation
queries with CPU UDFs and explicitly registered GPU models. Configure catalog objects and connection settings before
enabling it. Writes, explicit transactions, SQL PREPARE/EXECUTE, SQL EXPLAIN,
Relation EXPLAIN ANALYZE, PRAGMA and
reentrant execution on the same cursor are rejected. Parameterized reads and
read-only `executemany()` remain supported; each parameter set gets a fresh
request and native preparation. Unregistered local subprocess actors remain
query-owned in this entry point. Explicitly registered models use resident
pools through the same preparation and cleanup path.

An active runtime query rejects another query or relation binding on the same
cursor before taking connection locks. Concurrent clients use independent
cursors. DataSource callbacks include task deserialization, `execute()`, batch
iteration and stream teardown; pandas/NumPy callbacks include Python object
conversion. All follow the input callback contract below.

Connection-bound FILE operations include `File.open()`, `File.exists/stat/mime_type()`
and an open reader's reads, MIME detection, source identity, interrupt checks
and closure.
Query-owned DataSource readers use their execution context directly and remain
available for producing input without calling a Python connection.

Python input and registered filesystem callbacks cannot call connection query,
binding, fetch or `close()` APIs, including on idle siblings or unrelated
connections. Connection-bound FILE readers follow the same rule. Reentry raises
`InvalidInputException` before acquiring connection locks or changing connection
state, with or without a configured runtime. Move such work outside the callback.
Independent control threads can still interrupt or close running queries.

The callback scope starts before invoking Python, including filesystem `open()`,
`glob()`, metadata, read, seek, write and teardown calls. It does not depend on an
existing file handle, a held I/O lock or the target cursor's activity. Arrow,
DataSource and pandas/NumPy callbacks use the same scope, including input
metadata, copying and serialization during binding. This removes the need
to track inherited file dependencies across nested queries: those queries never
start. Per-handle I/O locks still protect seek/read/write sequences when Python
releases the GIL; unrelated handles can operate concurrently.

Input copying is part of this boundary: `read_csv()`/`read_json()` invoke file-like
`read()` and path conversion under the callback scope before native scanning.
Their Python option conversion also stays in scope, including column/type objects.
Filesystem registration also guards protocol and capability properties.
Parameter conversion follows the same rule: parameter length/iteration,
mapping copies, names and nested values run inside the callback
scope, including both outer and inner `executemany()` parameter sets. Plain
Python work remains allowed in these hooks; connection query, binding and close
operations are rejected. Filesystem provider destruction also stays in scope,
covering `__del__` and weakref callbacks during unregistration and database closure.
DataSource schema snapshots normalize structured entries, strings and tensor
dimensions inside that scope. Only built-in metadata and parsed Arrow types reach
the later connection-backed type parser; copying just the outer dictionary is
insufficient because nested methods can still execute Python.

Binding callbacks also include the first Arrow protocol probe, Relation argument
conversion (including iteration and error formatting), and UDF resource/payload
conversion. A scope covers both the call and destruction of its temporary Python
objects. DataSource task lists therefore remain in scope through normal return,
iteration errors and pickling errors. Registered input dependencies and cached
DataSource schemas guard their own final decrefs; a caller need not remember to
wrap each disposal site. Trusted SQL UDF type normalization runs before taking
the registration lock, with callback entry checked first.
The shared `try_cast` helper also guards failed type conversions: constructing
the conversion error can call a user-defined metaclass's `__str__`, including
when an invalid object is passed as a SQL statement. Shared container checks and
value conversion enforce the same boundary.
Replacement scans keep the scope through frame lookup, initial type probing,
failure diagnostics and frame-reference cleanup, including registration errors.

Exported live Arrow readers continue native execution and enforce the callback
entry rule when fetching. They also reject a busy source cursor instead of
waiting. All connection-owned results carry the source cursor lock, including
prepared `executemany()` and SQL `EXECUTE` results; their owner reference is weak
to avoid retaining the connection through its own result. Materialized readers
no longer drive a query and do not need its
connection lock. A live reader cannot be used as another query's input, even
on a sibling cursor; materialize it with `reader.read_all()` before registering
it, or execute the relation before exporting its reader. The configured runtime's
Arrow input policy still applies.

New native-to-Python input entry points must establish `PythonInputCallbackScope`
before invoking Python; connection entry must use `LockForQuery`,
`LockConnection` or the explicit callback check before any blocking operation.
`test_python_callback_entry.py` covers opening, metadata, I/O and cleanup against
connection, Relation and FILE APIs, including two concurrent imports whose input
callbacks try to enter each other's connections. Filesystem concurrency tests cover
active and idle siblings, nested scans, native workers and independent control threads.
`test_python_parameter_callbacks.py` applies the same concurrent checks to
parameter conversion, including a reused prepared statement and callback errors.
`test_python_binding_callbacks.py` extends this to Arrow protocol descriptors,
Relation expressions/options, UDF resources, task-list unwinding and registered
input disposal. Its two threads meet inside Python hooks while the outer binds
retain different cursor locks, then try to query or close each other's cursors.
Successful normal binding and subsequent cursor reuse are checked too.

Conversion scopes end before native Relation binding, including expression
projection/filter/order, joins and UDF composition. Native binding can invoke
trusted AI SQL type normalization, so it must not inherit the argument-conversion
scope. Compatibility tests bind real `AI_EMBED` specifications through these APIs
without calling a provider. Implicit type conversion keeps its callback guard:
it uses the existing default catalog when open, or a private native parser context
when that catalog is closed. It does not recreate a Python connection from inside
a conversion callback. Tests cover nested types, catalog-defined types and native
execution with a string UDF schema after default-connection closure.

When reviewing a new boundary, follow the complete ownership path:

1. Check callback entry before any connection lock or native binding.
2. Treat attribute probes, container conversion, numeric/string conversion and
   exception formatting as calls into Python, even without an explicit `()`.
3. Declare the callback scope before temporary Python owners so it outlives their
   destructors on every exit. Owners retained beyond that block must guard their
   own disposal or be released after the connection lock.
4. Leave the scope before trusted runner/connection entry. Pass normalized native
   values across that boundary, and test ordinary calls as well as rejection.

Arrow schema binding and stream callbacks use the context of the cursor executing
the query, including Arrow views created by another cursor. Each
execution retains its own callback identity while sharing the input factory's
captured format settings. Input validation also uses the executing cursor.

Configured runtimes accept materialized Arrow `Table` and `RecordBatch` inputs,
built-in `InMemoryDataset` inputs and unions composed entirely of them, and
materialized Polars `DataFrame` inputs. Opaque Arrow `RecordBatchReader` inputs,
C stream capsules/providers, prebuilt Scanners, file-backed/custom Datasets and
Polars `LazyFrame` inputs are rejected. A reader created by `from_batches()` can
still hide an asynchronous producer, and Polars collection can invoke Python
on its own worker threads. Wrapping the outer stream does not establish query
ownership on those threads.

Validation precedes schema export, stream creation and LazyFrame collection,
including initial relation binding before request admission. Views and relations
created before configuration are checked again using the executing cursor's
policy. Use native file scans such as `read_parquet(...)`, or materialize inputs
before submitting the runtime query: `reader.read_all()`, `dataset.to_table()` or
`lazy_frame.collect()`. This external materialization is outside runtime budgets
and deadlines. Unconfigured connections retain their existing input support.
Input producers must not dispatch connection operations to other threads and
wait for those operations themselves.

The `json_execute_serialized_sql()` table function is also rejected, including
inside macros and subqueries: it executes on a separate native connection that
does not inherit the request's budgets or cancellation. Execute the inner SQL
directly on the configured connection instead. JSON serialization and
deserialization remain available, and unconfigured connections retain native
JSON execution.

Results use the normal fetch/Arrow APIs and are materialized before the request
returns its execution capacity, including when the caller asks for an Arrow
reader. This is not incremental native result delivery. The UDF byte budget
covers the existing shared-memory reservations and retained allocations; it
does not bound DuckDB's materialized result, Python conversion buffers or
caller-owned copies. Native memory remains governed by DuckDB's settings.

`cursor.interrupt()` cancels that cursor's queued or running request.
Its fence covers Python cancellation through completion: overlapping queries
are rejected, and its native callback stays bound to the original request.
`cursor.close()` cancels its request before waiting for execution to retire and
leaves sibling cursors usable. Closing the owning connection drains ingress,
cancels its own and child queries, and closes the runtime after the last cursor
finishes. Failed cleanup retains the request allowance and retry owner;
`runtime.close()` or a repeated connection close retries it. `runtime.drain()`
rejects new requests while claimed requests finish. No failed execution is
automatically replayed. An optional `execution_timeout` starts when admission
is claimed; the request limit's `queue_timeout` independently bounds waiting.

The returned `LocalQueryRuntime` exposes `resource_snapshot()`, `drain()` and
`close(timeout=..., kill=...)`. It composes the same `LocalModelRuntime` request,
task, data, cancellation and cleanup policies used by the explicit plan API.
The native bridge passes preparation metadata and executor handles only;
Python never owns a borrowed native physical-plan pointer.
Preparation visits owned execution plans as well as ordinary inputs, so UDFs
inside correlated subqueries receive the same captured session environment,
task admission and byte limits as top-level UDFs. Local graph collection exposes
their individual resource units and the delim join's materialized input
dependency without changing native scan state.

## Resident models in ordinary queries

Register an instantiated `vane.cls` or `vane.cls.batch` with the runtime returned
by `configure_local_runtime()`. Registration binds a prototype using declared
input types; it does not execute rows or initialize a worker. The returned
model builds typed positional expressions and can be attached as a SQL function.
Use the default local connection for this example.

```python
import vane
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.resources import ResourceVector

@vane.cls(actor_number=1, return_dtype="BIGINT")
class Encoder:
    def __init__(self, offset):
        self.offset = offset  # Load CPU model weights here.

    def __call__(self, value):
        return value + self.offset

with vane.connect() as connection:
    runtime = connection.configure_local_runtime(
        request_limit=RequestAdmissionLimits(2, 8),
        resident_limit=ResourceVector(cpu=1, heap_bytes=64 * 1024**2),
    )
    encoder = runtime.register_model(
        "encoder", Encoder(10), version="weights-v1", parameters=["BIGINT"],
        cpus=1, memory_bytes=64 * 1024**2,
    )
    encoder.prewarm()  # Optional; the first query otherwise initializes it.
    vane.attach_function(encoder, "encode_value", connection=connection)
    with connection.cursor() as cursor:
        assert cursor.execute("SELECT encode_value(2)").fetchall() == [(12,)]
        source = cursor.sql("SELECT 3 AS x")
        assert source.project(encoder(vane.col("x"))).fetchall() == [(13,)]
```

The registration serializes the class adapter and constructor arguments once;
later expression builds reuse that snapshot. Input names,
types, return schema, actor count and batch size come from the declaration and
cannot be overridden at SQL attachment; choose another registration name and
version for a different definition. Expression calls cast inputs to the same
declared types that SQL binding uses. Batch byte settings use the captured
session configuration. SQL aliases and native passthrough columns
do not change initialization identity. Each registration is explicit: an
equivalent unregistered class keeps its existing per-query lifetime.

`cpus` and optional `memory_bytes` are declarations per actor. `resident_limit`
applies to their pool-wide sum and uses the existing registry admission policy.
An oversized registration fails before startup; prewarming another model can
refuse capacity occupied by resident pools. These declarations do not enforce
OS limits or evict models. Task and UDF byte limits retain their separate roles.

SQL and Relation queries acquire fresh borrows, with the same session snapshot,
task/byte admission, deadline, cancellation and cleanup contracts as ordinary
queries. A model cannot be used by another session, an unconfigured connection
or another runner. Finishing or cancelling one query does not retire the pool.
Worker loss fails the affected request without replay; a new request may start
a replacement worker. Initialization failure follows the existing sticky failure
contract. Drain rejects new registration, expression construction and prewarm;
already claimed requests retain their preparation capability. Connection close
drains requests and then releases resident models. An in-progress `prewarm()`
keeps its owning connection alive through initialization and borrow cleanup,
including initialization failure. An idle model handle does not retain the
connection.

Model registration, expression construction, attachment and prewarm obey the
Python input callback reentry rule. Use them outside input/filesystem callbacks.
The registration API uses positional call inputs and the SQL-compatible class
signature; eager Python evaluation and per-call option overrides are not part
of this returned model handle's contract.

## Ownership and shutdown

`ModelPoolRegistry` and `ModelPoolBorrow` contain the common ownership policy:

- Concurrent acquisitions initialize each registered identity once. Different
  models can initialize independently.
- A borrow is a lifetime reference, not an execution slot or memory reservation.
  Per-executor admission and cancellation continue to use the existing shared
  UDF contracts. Finishing or cancelling one executor does not close the model
  pool or cancel another executor's scope.
- Local worker loss uses the pool's existing replacement path. Replacement may
  initialize a new worker; it does not authorize replay of a failed request.
- `drain()` rejects new registrations and acquisitions. Existing borrowers can
  finish and release their references.
- `close(timeout=0.0, kill=False)` drains and waits for all borrowers and pending
  constructors before shutting down pools. The default fails promptly if work
  remains. The timeout bounds this quiescence wait; backend cleanup has its own
  timeout. `kill=True` selects backend forced cleanup after quiescence and does
  not revoke borrowers. Cancel and finish requests before releasing them.
- A timed-out close leaves the runtime draining. Release the outstanding
  borrowers and retry close. Close and borrow release are idempotent.
- Initialization failure is sticky for that registration. The initializing
  caller receives the original error; subsequent callers receive independent
  exceptions restored from the same bounded, value-only snapshot used by Ray.
  Exceptions that cannot be snapshotted or restored produce a fresh
  `RuntimeError` with a bounded diagnostic containing the original type and,
  when available, its string argument. The cache retains no live request
  tracebacks. Partial constructor owners stay with the runtime.
  Close retains uncertain cleanup owners for another explicit retry and
  continues attempting cleanup of the other models. It never relies on garbage
  collection to prove that cleanup completed.

Use explicit close after all query resources have been released. The runtime
context manager performs the same close on exit.


## Worker failure metrics

`runtime.resource_snapshot()["worker_failures"]` exposes cumulative counters
for subprocess worker outcomes, including prewarming and replacement. The
snapshot always includes all six fields, initialized to zero:

| Field | Observation |
| --- | --- |
| `initialization_failures` | Worker startup fails before readiness, including constructor errors and startup communication failures |
| `execution_errors` | A ready worker reports an execution error through its protocol, including UDF and worker-side conversion errors |
| `worker_losses` | Unexpected process exit, control-channel failure or invalid protocol response after readiness |
| `runtime_errors` | Parent-side execution, serialization, control handling, wakeup or resource-cleanup errors retire a ready worker, without an earlier terminal outcome |
| `cancelled_workers` | Cancellation intentionally retires a worker or interrupts its startup |
| `shutdown_workers` | Ordinary closure of a live worker without an earlier failure or cancellation |

One physical worker generation, including its startup attempt, records its
first observed terminal outcome once. Repeated exception delivery, cleanup
retries and several requests borrowing the same failed model do not add counts.
A replacement has its own lifecycle. Cancelling a queued request or a task
that leaves its worker reusable does not count a retired worker. Normal
closure and cancellation are separate from the four failure counters.
Counters describe observed boundaries, not inferred OS causes such as OOM;
an idle worker's exit is observed on a later acquisition or shutdown, without
a background monitor. A recorded outcome does not establish that cleanup has
finished: pending owners and resource charges retain their existing semantics.
Control messages and final results are classified before fallible parent cleanup:
malformed worker events, result descriptors and Arrow IPC responses count as
`worker_losses`, and a reported input-consumption failure counts as
`execution_errors`. Parent response serialization and `MemoryError` during frame
assembly, reception or result decoding count as `runtime_errors`. Subsequent
grant or lease cleanup failures cannot replace an already observed category.
Descriptor field validation runs before budgets and shared-memory ownership are
transferred to the result; failed decoding retires the affected worker.
Lazy shared-memory results retain an observer for their producing worker and
collector. Malformed Arrow contents discovered during later materialization
retire that physical worker and count as `worker_losses`; parent decoding
allocation failures count as `runtime_errors`. The result remains zero-copy
and is not decoded on arrival. A deferred observation keeps its producer's
attribution even if a cached task worker has since served another runtime;
it cannot retire a replacement worker or override an earlier terminal outcome.
For chained subprocess UDFs, the consumer reports which input block failed
Arrow decoding. The parent verifies that block belongs to the consumer's input
lease, then notifies its producer before releasing the lease or recording the
consumer's execution error. This also preserves attribution when one cached
worker produces a result and later consumes it for another runtime. A separate
consumer still records its own execution error; consumer allocation failures
and errors applying input projections do not blame the producer.

Registered and query-owned actor workers report to their owning runtime.
Cached task pools can be shared by different runtimes: an active worker
reports to its current borrower, and detaches that collector before becoming
idle. A dead idle task worker is attributed to the next borrower that discovers
it; idle pool teardown has no borrower and does not charge the previous one.
Unconfigured executions do not charge a previously configured runtime.
These counters are not per-query failure totals; `request_admission` continues
to report `failed_executions`, which also covers preparation failures before
any worker starts.

`WorkerOutcome`, `WorkerLifecycle` and `WorkerMetrics` define the common
backend-neutral accounting contract. This increment wires the subprocess
adapter; Ray integration remains separate. Collectors keep only fixed counters,
and retain no worker IDs, runtime owners, requests, exceptions or tracebacks.
The finite acceptance/soak driver still records its own injected-fault counts;
those are scenario evidence, separate from these runtime observations.

## Managed results from SQL and Relation queries

This section covers model-serving queries. The native local `query()` API is
documented in the [execution design](PIPELINED_EXECUTION_DESIGN.md#公开-api-与后续目标).

Configure `result_limit` on the owning connection to enable explicit managed
delivery for ordinary local queries. Both `connection.execute_result()`
and `relation.execute_result()` return the existing `QueryResult` handle:

```python
import vane

from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.result_delivery import ResultDeliveryLimits

with vane.connect() as connection:
    runtime = connection.configure_local_runtime(
        request_limit=RequestAdmissionLimits(2, 8),
        result_limit=ResultDeliveryLimits(max_results=4, max_bytes=64 * 1024**2),
    )
    with connection.execute_result(
        "SELECT ?::BIGINT AS value", [7], delivery_timeout=5.0,
    ) as result:
        for table in result:
            assert table.column("value").to_pylist() == [7]
    relation = connection.sql("SELECT i FROM range(3) t(i)").project("i + 1 AS value")
    with relation.execute_result(delivery_timeout=5.0) as result:
        for table in result:
            assert table.column("value").to_pylist() == [1, 2, 3]
```

These calls share the configured request, UDF task/data and registered-model
budgets with other queries in the session. Concurrent clients use independent
cursors. SQL accepts one read-only SELECT, including positional or named
parameters; multi-statement scripts are rejected before executing any statement.
SQL replacement scans keep the caller's lookup frame across admission and honor
the existing replacement-scan settings. A Relation must have no open result;
an exhausted or explicitly closed result permits a new execution.
Each explicit call executes once;
delivery never replays a query. Existing `execute()`, `fetchall()` and Arrow
fetch methods retain their ordinary result behavior and do not enter this
opt-in delivery budget.

The request waits for execution admission before reserving a result slot.
Slot refusal occurs before user execution and retires the private request
ticket, so repeated refusals do not exhaust ingress. For materialized delivery, execution and confirmed
UDF cleanup release request capacity before result encoding or consumption.
The same coupled reservation, exact Arrow IPC byte accounting, delivery
deadline and cleanup-retry policy described below governs both this entry
point and explicit physical-plan requests.

By default this adapter materializes the native result and prepares one Arrow IPC payload
for a nonempty query. An empty query preserves its schema without a payload.
`result_schema` contains native column names and type names; `completion_status`
is `ok` or `empty`. The adapter does not populate per-fragment `stats` or
`task_stats`; shared runtime diagnostics remain in `resource_snapshot()`.
Native materialization, Arrow conversion and temporary encoding overlap are
outside `max_bytes`. This is a retained delivery-buffer bound, not native
streaming or a whole-process memory limit. Opt-in streaming is described in
[managed native result streams](#managed-native-result-streams). A byte refusal can follow UDF
execution and must not be treated as permission to replay it.

`close()`, `cancel()` and delivery expiry retire unconsumed results. Exported
Arrow tables and their zero-copy views remain charged until the last reference
is released, even after connection/runtime close. Pending or failed result
cleanup remains runtime-owned for an explicit retry. Keep the connection alive
while using pending managed results; closing or collecting its final session
connection closes those results. Independently exported views remain valid.
The input/filesystem callback reentry restriction also applies to both new
entry points and their argument conversion.


## Registered local GPU models

Pass a provisioned inventory to `configure_local_runtime(gpu_devices=...)`,
declare `gpus=1` and a fixed `actor_number` on the class UDF, then assign one
inventory UUID per replica in `register_model(gpu_devices=...)`. The resulting
handle supports prewarm, SQL attachment, Relation expressions and managed
results through the existing runtime APIs. For example, on a CUDA host with
PyTorch installed:

```python
import vane
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.resources import ResourceVector
from vane.execution.udf_runtime_admission import TaskAdmissionLimits

# Replace this example UUID with a provisioned full physical GPU UUID.
devices = ["GPU-aaaaaaaa-0000-0000-0000-000000000001"]

@vane.cls(gpus=1, actor_number=1, return_dtype="DOUBLE")
class Score:
    def __init__(self):
        import torch
        self.weights = torch.arange(8, dtype=torch.float64, device="cuda")

    def __call__(self, value):
        return float((self.weights * value).sum().cpu().item())

with vane.connect() as connection:
    runtime = connection.configure_local_runtime(
        request_limit=RequestAdmissionLimits(2, 8),
        resident_limit=ResourceVector(cpu=1, gpu=1),
        gpu_devices=devices,
        task_limit=TaskAdmissionLimits(1, 16),
    )
    model = runtime.register_model(
        "score", Score(), version="v1", parameters=["DOUBLE"], gpu_devices=devices,
    )
    model.prewarm()
    vane.attach_function(model, "score", connection=connection)
    assert connection.execute("SELECT score(2)").fetchone() == (56.0,)
    print(runtime.resource_snapshot()["gpu"])
```

`vane.cls.batch` uses the same registration path. UDFs return ordinary host
values/Arrow arrays after their device work completes; moving a CUDA result
to host as above supplies that synchronization. A callable that starts device
work unrelated to its returned host result must synchronize it before returning.
The runtime does not discover arbitrary CUDA streams or synchronize background
GPU work that outlives a UDF invocation. It introduces no PyTorch dependency;
the application supplies its CUDA framework.

Unregistered local GPU UDFs, GPU task UDFs, fractional GPU declarations and
multiple GPUs per replica remain unsupported. An empty inventory is invalid;
omit `gpu_devices` for a CPU-only runtime. CPU and GPU registered models share
one registry, task budget, byte budget and request lifecycle. CPU registrations
must omit the device assignment. A resident GPU limit of zero rejects GPU
registration even if the inventory contains devices.

### Device residency

The shared `vane.execution.udf_local_gpu.LocalGpuModelAdapter` implements
the device contract tracked in [#842](https://github.com/AstroVela/vane/issues/842).

The adapter binds a provisioned inventory to one `ModelPoolRegistry` and
registers fixed replicas, each requesting exactly one GPU and assigned one
full GPU UUID. The inventory and assignments reject ordinals, abbreviated
UUIDs, duplicate identities and MIG devices. UUIDs are canonicalized before
comparison. Inventory provisioning is explicit; this increment does not
discover hardware or prove that a listed device is physically available.

Registration starts no workers. The common registry atomically reserves the
pool's CPU/GPU/declared heap and its exclusive `cuda:<UUID>` keys before calling
any model constructor. An exclusive-resource conflict raises
`ModelPoolResourceBusy` with the conflicting keys. It publishes no partial
reservation and is retryable; it is not cached as an initialization failure.
All adapters managing the same devices must share this registry. The scope is
one registry, not machine-wide arbitration across independent runtimes or
processes. Ray continues to use Ray Core placement and its own authorization.

Each subprocess receives its assigned UUID in `CUDA_VISIBLE_DEVICES` before
loading the serialized UDF, using CUDA's documented
[GPU UUID visibility selection](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).
The explicit assignment controls the child environment, including after later
parent-environment changes; the parent's environment is not modified. Session
configuration and model payloads are frozen at registration. Ordered device
assignments participate in pool identity, so changing replica placement cannot
reuse an incompatible registration.

Resident ownership follows the existing pool lifetime: query completion,
borrow release and cancellation do not free devices. Replacement closes the
old worker before starting its successor on the same device. Failed cleanup
prevents replacement. Clean initialization failure returns reservations;
partial initialization or failed shutdown retains all of that pool's device
keys and numeric resources until close retries confirm cleanup. Drain fences
new borrows and prewarms; an initializer already in progress remains owned for
close even if its caller cancels or the registry drains.

`ModelPoolRegistry.resource_snapshot()["exclusive_resources"]` maps reserved
keys to their model/version/session owner. `pool.device_snapshot()` reports
replica index, device, generation, PID and cleanup completion for current,
provisional replacement and retained cleanup workers. These are passive
component snapshots; initialization may still be in progress before a pool is
available for borrowing.

CPU-only tests use fake provisioned UUIDs and real subprocesses to verify
environment setup, reuse, concurrent prewarm, device conflicts, cancellation,
replacement and retained cleanup. They establish no CUDA-kernel or physical
VRAM guarantee. This resident GPU count is separate from per-device execution
demand and GPU-memory admission.

### GPU execution admission

GPU actor pools use the existing `LocalExecutionSlotPool` arbitration and
`AdmissionLease` lifecycle. Each slot maps to one fixed replica/device; it is
not a second semaphore or wait queue. `RuntimeTaskAdmission` can therefore
acquire a task allowance and device slot together, and the byte-wait adapter
continues to reserve a complete envelope only when that slot is available.
The existing arbitration treats limited and ordinary queries fairly within
the shared pool. Different devices can run concurrently, subject to the
runtime's task limit.

An `udf_subprocess.UDFExecutor` attached to a GPU pool passes its
admission lease when submitting work. Direct `pool.submit(...)` also requires
the `admission=` keyword for GPU pools; missing, foreign, released or already
submitted leases are rejected. The pool dispatches to the lease's replica,
independently of idle-worker ordering, and records the actual worker PID and
generation when execution begins. CPU pool submission is unchanged.

Per-device execution demand is one logical GPU slot, separate from the
registry's resident GPU reservation. Model borrows and repeated invocations
do not reserve resident GPUs again. Waiting for an initial byte envelope owns
no device or runtime task allowance. During an in-flight shared-memory wait,
the runtime task allowance can be yielded while the invocation retains its
physical device/worker. Both device and UDF-unit diagnostics observe that
wait and the subsequent wait to resume execution.

Completion returns execution demand while buffered results retain the
existing output slot until consumption. Cancellation and execution deadlines
use the ordinary executor cancellation path. Worker replacement stays on the
assigned device; unfinished replacement or failed worker cleanup retains its
execution record as well as resident ownership. The actor pool retries the
physical cleanup, and only confirmed completion retires the retained record.
A callback that has not reported execution completion also remains visible
to pool cleanup.

`pool.gpu_execution_snapshot()` is passive and reports devices, live execution
leases, states, PID/generation, logical execution resources, retained result
slots and worker generations. Queued requests appear once at pool scope
because a pending request has no assigned replica yet. These are independently
locked diagnostic samples, not an atomic reservation API. They never acquire
capacity or perform cleanup.

The public `runtime.resource_snapshot()["gpu"]` lists the frozen inventory and
each registered model's assignment plus these pool snapshots. It includes
retained owners after initialization or shutdown failure, without acquiring a
borrow or prewarming a model. During initialization the pool may not yet be
published; `initializing_resources` still records its resident reservation.
After successful close the model's configured assignment remains diagnostic
metadata, while its pool list and all reservations are empty.

CPU-only tests run real subprocess UDFs with fake provisioned UUIDs. They cover
device binding, parallel replicas, query fairness, shared task budgets,
cancellation/deadlines, byte waiting and failed-cleanup retry. The separate
CUDA acceptance module exercises actual device tensor computation through
registered row/batch models, SQL, rebuilt Relations and managed results, plus
concurrent cursors, drain, cancellation, deadlines, model failure and byte
limits. It requires PyTorch with CUDA and downloads no models:

```bash
# Optional: choose a provisioned UUID instead of the first visible GPU.
export VANE_TEST_CUDA_DEVICE=GPU-aaaaaaaa-0000-0000-0000-000000000001
scripts/run_installed_pytest.sh tests/fast/test_local_query_gpu_cuda.py -m gpu
```

These hardware tests carry the `gpu` marker and run separately from CPU CI.
For repeated load, slow consumers, cancellation, execution expiry and worker
recovery in one CUDA runtime, use the
[sustained serving runner](LOCAL_SERVING_ACCEPTANCE.md#sustained-lifecycle-acceptance).
It shares the CPU supervisor and workload, checking device/worker generations
and idle ownership each round, with bounded diagnostic artifacts on failure.
Logical resident/execution counts do not estimate or enforce physical VRAM,
reserve a device against other processes/runtimes, or provide spill or GPU
utilization scheduling. Provision exclusive devices externally when needed.

## Managed native result streams

`execute_result(..., stream=True, rows_per_batch=2048)` on a configured connection
or Relation returns a pull-driven managed result. `take()` reads the next native
Arrow batch and encodes one delivery payload; it does not collect the complete
query into an Arrow table first. Existing calls keep their materialized behavior.

```python
with connection.execute_result(
    "SELECT encode(x) FROM inputs", stream=True, rows_per_batch=2048,
    delivery_timeout=30,
) as result:
    while True:
        try:
            table = result.take()
        except StopIteration:
            break
        consume(table)
        del table
```

The request slot, UDF scopes, model borrows and native interrupt fence remain
owned until EOF or confirmed cleanup. Use an independent cursor for concurrent
queries; an open stream fences further queries on its cursor. Execution timeout
covers the entire stream, including consumer pauses. Delivery timeout starts
when the stream is ready and covers the total consumption interval.

Each IPC buffer is reserved before allocation and stays charged through exported
Arrow/NumPy views. When those views fill the delivery budget, the next `take()`
waits for capacity, cancellation or a deadline instead of reading more batches.
One decoded batch may wait for its exact IPC reservation. Release previous
views before taking another batch, or provision capacity for overlapping batches.
In particular, a Python `for table in result` loop retains its previous loop
variable while asking for the next item. Set a delivery or execution deadline
when retained views could otherwise prevent progress.

A single batch larger than the complete delivery budget fails explicitly after
execution started; it must not be replayed as a new request. Closing a stream
cancels unread work and preserves exported views. Runtime close cancels live
streams and fences streams still being prepared before waiting for request
admission. This fence remains effective if close times out. Failed native or UDF
cleanup keeps both cleanup ownership and admission until an explicit retry succeeds.

The delivery budget bounds encoded IPC buffers. Native scan/sort/aggregate/join
state, the current decoded batch and temporary encoding copies remain outside
that budget and under DuckDB's native memory policy. Blocking operators may
materialize internally before their result can stream. This interface does not
provide a whole-process memory bound. HTTP sends and disconnect handling remain
transport-adapter responsibilities.

`result_delivery` snapshots also expose `streaming_results`,
`waiting_byte_results` and `waiting_bytes` (requested capacity, not a reservation).
The [sustained streaming acceptance](LOCAL_SERVING_ACCEPTANCE.md#sustained-streaming-delivery)
reuses one CPU or CUDA runtime across mixed consumers, byte waits, deadlines,
worker replacement and shutdown.


## Request and resource ownership

Configure admission on the owning connection before creating cursors. Requests
wait under the queue deadline, start their execution deadline after admission,
and retain resource charges until native execution and cleanup finish. Use
independent cursors for concurrency. `interrupt()` cancels the active request
on its cursor; closing the session drains admission and releases model owners.
Failed cleanup remains charged and owned so a later close can retry it.

`request_limit`, `resident_limit`, `task_limit`, `data_limit` and `result_limit`
control separate lifetimes. Resident model reservations remain after a query;
per-query tasks and UDF input/output buffers retire with their owners. Result
views can outlive the cursor or connection and retain their delivery byte
charge until the final view is released. Model calls are never replayed to
recover from a result capacity refusal after execution starts.

`DataAdmissionLimits` bounds retained UDF shared memory. Optional
`DataAdmissionWaitLimits` lets producers wait for capacity, with a finite queue
and deadline. `track_data=True` observes the same ownership without imposing a
byte limit. These settings do not assert an OS process RSS limit.

## Native resource graph

`track_graph=True` records native operators and UDF units as structural
metadata. It does not schedule native work. The collector walks all owned
native plans once, assigns UDF identifiers in the same order as preparation,
and returns copied metadata. ORDER BY and TopN mark materialization barriers.
Collection changes neither scan identities nor model payloads and creates no
Ray plan, worker or runner.

Model registration binds and plans a typed prototype to collect its UDF
payload; it neither executes user rows nor initializes model workers. Prewarm
and the first model call own initialization. The connection runtime then
injects per-query resource scopes into each prepared native execution.

## Supported execution boundary

This service supports native local read-only SQL/Relation queries and the
explicit registered CPU/GPU model contracts above. Ray `query()` currently
supports the analytical fragment profile documented in the
[execution design](PIPELINED_EXECUTION_DESIGN.md), which excludes model UDFs.
No model operation falls back to another backend.

The public runtime tests are
[query admission and cancellation](tests/fast/test_local_query_runtime.py),
[model registration and reuse](tests/fast/test_local_query_models.py),
[managed results](tests/fast/test_local_query_results.py),
[native streams](tests/fast/test_local_query_streaming.py), and
[serving acceptance](LOCAL_SERVING_ACCEPTANCE.md). Internal model admission,
byte ownership and cleanup also have focused unit tests. Removed runner and
serialized-plan APIs are not part of this service's contract.
