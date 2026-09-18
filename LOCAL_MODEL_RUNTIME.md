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
Task-slot admission, fair queuing across models, retained input/output budgets,
and output-completion capacity remain separate follow-ups. Resident admission
does not replace executor backpressure or impose a whole-process memory cap.

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
`test_udf_actor_pool_lifecycle.py`, `test_udf_executor_lifecycle.py`,
`test_driver_udf_precreate.py`, and `test_udf_process.py` under `tests/fast/`.
Adapter serialization also has coverage in `test_pickle.py`, the expression
class test suites, and `test_ray_udf_plan_replay.py` (run its real-Ray cases in a
separate pytest process).
They cover shared contracts, real subprocess reuse, native sequential/concurrent
queries (including repeated SQL calls and rebuilt class projections),
captured session isolation in mixed actor/task plans, failed-request collection,
cancellation, worker replacement, and ownership recovery. Follow the
installed-package and release checks in [DEVELOPMENT.md](DEVELOPMENT.md).
