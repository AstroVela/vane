# Local model pool lifetime

This document describes the internal execution interfaces for explicitly
registered, session-owned CPU models. The roadmap is tracked in
[#838](https://github.com/AstroVela/vane/issues/838), with model ownership in
[#840](https://github.com/AstroVela/vane/issues/840). These interfaces do not yet
add a public `vane` model-registration API or a serving endpoint.

## Registration and binding

`vane.execution.udf_local_model.LocalModelRuntime` owns local subprocess actor
pools for one Vane session. Construct it from a physical plan's `session_id()`
and `session_config()`. Register a model by an explicit name and version, using
the payload of a collected `subprocess_actor` UDF node:

```python
from vane.execution.udf_local_model import LocalModelRuntime

models = LocalModelRuntime(
    session_id=plan.session_id(),
    session_config=plan.session_config(),
)
node = plan.collect_udf_nodes(conn=connection)[0]
models.register("encoder", version="weights-v1", payload=node["payload"])
models.prewarm("encoder")
query_resources = models.prepare(
    plan, {str(node["node_id"]): "encoder"}, conn=connection
)
```

Registration is lazy; prewarm initializes the workers without keeping a query
borrow. Bindings are explicit per UDF node. A new query can bind the same model
when its session identity, captured configuration, compatible UDF payload and
actor count match. The payload fingerprint includes serialized initialization,
schema, device and execution settings, but excludes the per-query
`expression_id` assigned during SQL planning. Workers still receive the complete
registered payload. Generated `vane.cls` and `vane.cls.batch` actor adapters
serialize through explicit reconstruction recipes containing the user class,
constructor arguments, input/call layout, literal row-call keywords, and output
contract. Rebuilding equivalent projections therefore preserves the callable
payload, without depending on temporary adapter class identities or caching
adapter classes globally. Model compatibility still compares the full callable
bytes and all other payload settings. Changing a model or its version requires
a distinct registration name. Different sessions cannot share a registration,
even if their configuration dictionaries are equal.

Preparation validates all explicit bindings, then delegates to the existing
`ensure_local_subprocess_actor_pools_for_nodes` step to acquire fresh borrows and
inject pools through `local_actor_pool`. It returns the query resources to the
caller. Callers must retain these resources until executor cleanup has finished,
then call their existing `shutdown(kill=...)` cleanup path. A registered
model's returned resource releases only that query's borrow. Native query
cleanup and preparation rollback use this same resource contract.

Every collected UDF node receives the plan's captured session configuration,
including unregistered actor and function/task UDFs. Workers use this explicit
snapshot instead of inheriting another session's current process environment;
session variables absent from the snapshot remain absent in those workers.
Preparation preserves other executor options and publishes all node options
within the same rollback boundary. Unregistered actors remain query-owned,
and task pools retain their existing executor/task-runtime ownership.

Applications integrating at the internal physical-plan layer must run this
preparation/cleanup step for every execution. Preparation does not execute a
query. Do not run a second generic plan-preparation pass afterward: native node
collection does not currently expose the previously injected executor options.
Concurrent queries need independent cursors and plans constructed with those
cursors, all from the owning session. No callable or payload is automatically
registered.

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

## Resident resource admission

Pass an optional `resident_limit` to bound the sum of declared process
resources for this runtime's registered model pools:

```python
from vane.execution.resources import ResourceVector

models = LocalModelRuntime(
    session_id=plan.session_id(),
    session_config=plan.session_config(),
    resident_limit=ResourceVector(cpu=4, heap_bytes=8 * 1024**3),
)
```

Declare the per-actor heap on the relation before building its physical plan:

```python
relation = source.map_batches(
    Model,
    schema=output_schema,
    execution_backend="subprocess_actor",
    actor_number=2,
    cpus=0.5,
    memory_bytes=1024**3,
)
```

`flat_map` also accepts this declaration for `subprocess_actor`. The collected
native payload carries `memory_bytes` through registration, compatibility
validation, and execution; use that payload directly. Heap declarations remain
unsupported for `subprocess_task`, whose task-memory admission is separate work.

`ResourceVector` and UDF process-resource parsing are shared with Ray's query
resource graph. CPU may be fractional; heap uses integer bytes from the UDF's
`memory_bytes` declaration. A pool requests the per-actor resources multiplied
by its actor count. For example, two actors each declaring `cpus=0.5` and
`memory_bytes=1024**3` reserve one CPU and two GiB. Missing `memory_bytes`
reserves zero heap, matching Ray. All vector fields are finite limits, with
zero meaning zero capacity; omitting the entire limit preserves unbounded
resident admission while still reporting usage. Configured or fully reserved
zero capacity rejects every positive request, even below the shared vector's
floating-point tolerance. Fractional arithmetic retains that tolerance when
capacity remains; Ray's resource-vector comparisons are unchanged.

Registration validates each pool against the limit but starts no workers and
reserves nothing. The first acquire or prewarm atomically reserves the whole
pool before initialization, so concurrent constructors cannot each spend the
runtime's full budget. Repeated borrows, query completion, cancellation, and
worker replacement retain the same reservation. Only confirmed pool cleanup
returns capacity. A clean constructor failure returns its reservation;
partial initialization or uncertain cleanup retains the full reservation until
all associated owners have finished cleanup. Close retries return each pool's
capacity once, without releasing reservations for other unfinished pools.

An individually oversized registration raises `ModelPoolCapacityError` before
publishing the model. When other pools occupy the available capacity, acquire
or prewarm raises the same error before calling the constructor. The error
reports `requested`, `reserved`, `limit`, exceeded `dimensions`, and whether
the request is `oversized`. Capacity refusal is not cached as an initialization
failure: it can be retried if another failed constructor returns capacity.
There is no admission wait queue or automatic eviction for resident models;
waiting for a persistent pool's reservation to expire could deadlock. End the
runtime and choose a different set of models or limits to change residency.

`resource_snapshot()` reports the limit, registered resource demand, reserved
resources, active borrow count, and runtime state. Its `initializing_resources`,
`resident_resources`, and `retained_failure_resources` partition the reserved
total. These are logical declarations, not measured RSS or OS enforcement.
Models without heap declarations and memory used by decoding, DuckDB, Python
results, and mapped shared-memory blocks are outside this resident limit.
Shared mappings remain charged by the existing shared-memory budget; they are
not multiplied by the number of model borrowers. Local resident limits accept
CPU and heap only, and registrations still reject GPU resources.

The limit is shared by registered models and queries using one runtime. Separate
runtimes and unregistered query-owned UDF pools keep independent ownership.
This is the first increment of [#841](https://github.com/AstroVela/vane/issues/841).
Resident admission does not replace executor backpressure or impose a
whole-process memory cap.

## Runtime task admission

Pass a separate, optional task limit to share execution capacity across all
queries prepared by one runtime:

```python
from vane.execution.udf_runtime_admission import TaskAdmissionLimits

models = LocalModelRuntime(
    session_id=plan.session_id(),
    session_config=plan.session_config(),
    task_limit=TaskAdmissionLimits(max_running_tasks=4, max_queued_tasks=32),
)
```

Two registered models with four actors each can keep all eight processes
resident, while their queries together hold at most four execution allowances.
Ready grants count against this limit before submission, so dispatchers cannot
overbook it. Executing tasks do not charge the models' resident CPU or heap a
second time. These are logical task counts, not CPU scheduling or OS limits.

Preparation attaches one query admission owner to **every** collected local
UDF node: registered models, unregistered subprocess actors, and subprocess
tasks. Plans using other backends are rejected before actor preparation.
With a task limit, `prepare(plan, {}, conn=connection)` also supports plans
containing only unregistered UDFs. A task-only plan still receives the captured
session configuration and participates in runtime drain and close.

Pending admission requests share a bounded queue, rotating between queries.
Each eligible query gets one grant per round. A busy pool can be skipped so
another UDF in the same query can progress. A grant acquires both the runtime
allowance and a pool slot without waiting while holding just one of them.
Capacity changes wake pending dispatchers; they do not poll for capacity.

Subprocess task pools also share the global task executor's thread capacity.
Admission reserves a thread together with the pool slot, before reporting a
ready grant. This applies to tasks with and without a runtime task limit, so
another query cannot enqueue work ahead of an already reserved thread. When
all threads are occupied, a pending task receives no runtime allowance.
Suspended tasks retain their thread reservations; they can reacquire their
runtime allowance when memory becomes available. Backend completion returns
the thread reservation even while the result still holds its pool slot.

Tasks blocked on shared-memory input allocation or output grants temporarily
yield their runtime allowance. This lets a downstream consumer run and release
the bytes they need. They reacquire an allowance before resuming execution;
submitted tasks ready to resume take priority over fresh admission. The byte
budget is rechecked after reacquisition, so this does not overcommit memory.
Suspension and resumption run outside the memory-budget lock.

`max_queued_tasks` counts pending dispatcher requests, not queries or rows. It
may be zero to require immediate admission. A full queue raises
`TaskAdmissionQueueFull`; native execution surfaces this as a query error.
Clean up the failed execution before retrying it. Refusal reserves no capacity
and is not a cached model initialization failure. The queue is not a bound on
all query inputs, bytes, prepared plans, or resident processes.

The returned query resources now also contain a `QueryTaskAdmission` owner.
Retain and shut down **all** returned resources after executors finish, as in
the registration workflow. Query shutdown removes its pending requests and
unused ready grants. Running tasks retain ownership until their backend futures
finish, including cancellation/failure cleanup. A task waiting for shared-memory
capacity retains its worker slot and query owner while yielding only its
execution allowance. Cancellation wakes both memory and allowance waits without
releasing another query's capacity. Cancelling one query does not close its
shared model or release another query's allowances. A
runtime close waits for these query owners, including task-only queries;
`kill=True` does not revoke running work. Drain prevents new preparation while
allowing already prepared queries to finish.
The shared query admission gate closes before model draining starts, so
task-only preparation cannot enter between the two drain operations.

The runtime allowance is released when execution finishes, even if its result
has not yet been consumed. The existing pool slot remains attached to that
buffered result until consumption or cleanup. Shared-memory output references
keep their existing byte accounting until downstream releases them. A slow
consumer therefore still bounds buffering in its pool without monopolizing
another model's runtime execution allowance.

`resource_snapshot()["task_admission"]` reports the limits, `running_tasks`,
`ready_tasks`, `queued_tasks`, `waiting_tasks`, `resuming_tasks`, query owners,
and drain/close state. Waiting tasks include those waiting to resume; their
counts are not additive. Both remain tracked until backend completion and
are bounded by their physical pools, separately from the pending admission
queue. The running and ready counts sum to the currently reserved runtime task
capacity. Omitting `task_limit` skips the runtime-wide allowance and bounded
query queue. Pool slots and global subprocess task threads still constrain
admission.

This increment reuses the shared `AdmissionAuthority`/`AdmissionLease` wire
contract. Its fair queue consumes a backend-neutral, nonblocking
`AdmissionCapacity` adapter, initially implemented for local subprocess pools.
It does not install a runtime queue in Ray or replace Ray authorization.
Unified retained input/output budgets and output-completion reserves remain
follow-ups under [#841](https://github.com/AstroVela/vane/issues/841); this task
limit alone does not establish a whole-process memory bound. Yielding during
transport waits lets consumers with available workers use the execution
allowance. It does not pre-reserve worst-case UDF output expansion, provide
extra workers, or account worker heap buffers as shared-memory allocations.

## Retained shared-memory data

Enable an optional runtime data ledger to observe the lifetime of shared-memory
inputs and outputs across prepared queries:

```python
models = LocalModelRuntime(
    session_id=plan.session_id(),
    session_config=plan.session_config(),
    track_data=True,
)
query_resources = models.prepare(plan, bindings, conn=connection)
data = models.resource_snapshot()["data"]
```

`track_data` is independent of resident and task limits. It also permits empty
model bindings for task-only or other unregistered local subprocess plans.
Preparation attaches one `QueryDataScope` to every collected subprocess actor
and task, preserving their captured session configuration and existing pool
ownership. Other backends are rejected before preparation. Retain and shut down
all returned resources after executor cleanup, as with model and task owners.
The data-query gate closes before task and model drain, preventing new data-only
preparation while allowing prepared queries to finish.

The ledger uses transport provider and shared-memory name as an allocation
identity. It charges the descriptor's IPC size, including its header, once per
live allocation in the runtime. Repeated input slices, concurrent readers, and
overlapping input/output roles do not multiply that total. Sizes must agree
while an identity is live. The ledger holds identities, sizes, and counters;
it holds no data buffers, query objects, or request tracebacks.

- A task borrows its shared-memory inputs before dispatching them to a worker.
  The borrow lasts through backend completion, including failure or cancellation
  cleanup. A worker's input ACK can return transport-budget credit earlier; it
  does not end this data borrow.
- An output owner follows `generator_pending`, `unit_queue`,
  `downstream_input`, and `external_consumer` lifetimes. Backend completion
  returns execution allowances while output owners remain with buffered results.
  Consuming or dropping a result releases only that owner's reference.
- Materializing a tracked output forks an owner onto the Arrow foreign buffer.
  Tables, slices, arrays, buffers, and zero-copy NumPy views keep the allocation
  accounted for until their last underlying mapping is released. They remain
  valid after query cleanup and runtime close. A deferred mapping close retains
  its owner until the existing shared-memory cleanup retry succeeds.

`resource_snapshot()["data"]` reports:

| Field | Meaning |
| --- | --- |
| `retained_bytes` | Deduplicated bytes across all live input and output leases |
| `input_bytes`, `output_bytes` | Deduplicated bytes within each role |
| `output_state_bytes` | Deduplicated output bytes within each lifecycle state |
| `allocations`, `leases` | Distinct transport allocations and their live owners/borrows |
| `queries`, `tasks` | Query scopes and submitted tasks awaiting backend completion |
| `draining`, `closed` | Preparation and runtime shutdown state |

Role and state counters overlap; do not sum them to obtain total retained bytes.
For example, a 64-KiB output borrowed by two queries reports 64 KiB of input,
64 KiB of output, and 64 KiB retained. Closing the runtime waits for query/task
owners, not consumer output views. A closed runtime can therefore continue to
report retained bytes until those views are released.

This increment provides accounting without a new byte admission policy. It
observes allocations once they enter the local subprocess transport; it does
not measure Python/Arrow heap copies, worker-retained objects, model heap, or
DuckDB memory. UDF results returned directly as in-process `pa.Table` objects
instead of shared-memory ref bundles are outside output accounting. Separate
runtimes have separate ledgers. Omitting `track_data` creates no runtime data
owners and preserves the existing transport-budget behavior.

The output owner and forward-state/release policy are extracted from Ray into
`vane.execution.data_lifecycle` and used by both backends. Ray keeps its manager,
query-generation authorization, object-store reservations, and existing import
path. Common tests exercise that owner against both local and Ray managers.
Runtime retained-byte limits and output-completion reserves remain follow-ups
under [#841](https://github.com/AstroVela/vane/issues/841).

## Ray boundary and validation

The registry uses the `shutdown` and `cleanup_pending` contracts already exposed
by both local and Ray actor pools. Adapter tests exercise both implementations.
Local registration accepts only `subprocess_actor`; it cannot cache a Ray pool
or its query-generation capability. Ray's driver resource graph, actor identity,
admission and generation checks remain authoritative and query-scoped. A future
Ray registry adapter must establish valid query borrowing authorization before
enabling cross-query model reuse. This change does not resolve the separate
named vLLM ownership path in [#251](https://github.com/AstroVela/vane/issues/251).

The affected tests are `test_udf_model_pool.py`, `test_udf_local_model.py`,
`test_udf_model_resources.py`, the query resource graph/builder/manager suites,
`test_udf_runtime_admission.py`, `test_udf_task_admission.py`,
`test_udf_data_lease.py`, `test_udf_data_transport.py`,
`test_udf_actor_pool_lifecycle.py`, `test_udf_executor_lifecycle.py`,
`test_driver_udf_precreate.py`, and `test_udf_process.py` under `tests/fast/`.
Adapter serialization also has coverage in `test_pickle.py`, the expression
class test suites, and `test_ray_udf_plan_replay.py` (run its real-Ray cases in a
separate pytest process).
They cover shared contracts, real subprocess reuse, native sequential/concurrent
queries (including repeated SQL calls and rebuilt class projections),
captured session isolation in mixed actor/task plans, failed-request collection,
cancellation, worker replacement, ownership recovery, bounded fair queuing,
native concurrent mixed plans sharing task capacity across models, shared-data
deduplication, and output views that outlive query/runtime shutdown. Follow the
installed-package and release checks in [DEVELOPMENT.md](DEVELOPMENT.md).
