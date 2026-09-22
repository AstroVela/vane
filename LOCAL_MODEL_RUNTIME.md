# Local model pool lifetime

This document describes the internal execution interfaces for explicitly
registered, session-owned CPU models. The roadmap is tracked in
[#838](https://github.com/AstroVela/vane/issues/838), with model ownership in
[#840](https://github.com/AstroVela/vane/issues/840). These interfaces do not yet
add a public `vane` model-registration API or a serving endpoint.

The [CPU serving acceptance scenario](LOCAL_SERVING_ACCEPTANCE.md) exercises
these interfaces together with synthetic text/RGB UDFs, concurrent requests,
slow consumers, cancellation, expiry, and worker loss. It produces a JSON
report with initialization counts, latency distributions, and resource
checkpoints using an installed wheel.

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

Adapters supplying `local_model_pool` directly must provide the owning plan's
session identity to preparation: `ensure_local_subprocess_actor_pools_for_plan`
reads `plan.session_id()`, while the node-level helper requires an explicit
`session_id`. Matching configuration alone does not authorize model reuse.
Registration publishes its registry entry and model handle together with
respect to drain/close; payload serialization runs outside that lifecycle lock.

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
query. A native plan can be prepared again after the previous execution and
its query cleanup finish; each execution acquires a fresh model borrow. Do not
run a second generic plan-preparation pass afterward: native node
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

## Bounded local requests

The first serving increment of [#843](https://github.com/AstroVela/vane/issues/843)
adds an internal CPU request entry point around native local-fast execution:

```python
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.udf_runtime_admission import TaskAdmissionLimits

models = LocalModelRuntime(
    session_id=plan.session_id(),
    session_config=plan.session_config(),
    request_limit=RequestAdmissionLimits(
        max_active_requests=2,
        max_queued_requests=8,
        queue_timeout=30.0,
    ),
    task_limit=TaskAdmissionLimits(max_running_tasks=4, max_queued_tasks=16),
    track_data=True,
)
models.register("encoder", version="weights-v1", payload=node["payload"])
request = models.request(queue_timeout=5.0)
result = request.execute(
    plan, {str(node["node_id"]): "encoder"}, conn=connection, execution_timeout=5.0
)
```

Use an independent cursor and a fresh plan bound to it for each concurrent
request. `request.execute()` is synchronous on its calling thread: it waits
for admission, prepares the plan exactly once, calls the existing native
executor, then shuts down the returned query resources. Each request executes
once. Request-limited runtimes require this entry point; calling `prepare()`
directly is rejected. Runtimes without `request_limit` retain the existing
explicit preparation/cleanup interface.

Ready and executing requests share `max_active_requests`; the FIFO queue has
at most `max_queued_requests` entries. The queue owns ticket metadata only and
acquires no model borrows, UDF slots, or task-byte reservations. Plans, request
bodies, and caller-owned inputs waiting outside execution are not covered by
this count limit or by `data_limit`. Callers manage their own ingress buffers.
`RequestQueueFull` refuses overload immediately. A finite, non-negative
`queue_timeout` starts when the ticket is created and ends when it is admitted;
expiration raises `RequestQueueTimeout`. Zero requires immediate admission.
Expired entries are removed by waiting callers or the next admission/state
operation, so no background timer thread is required. Notifications and timed
condition waits drive blocked callers, without polling.

`request.execute(..., execution_timeout=seconds)` optionally limits execution
time. The finite, non-negative duration starts when `execute()` claims its
admission ticket, before model borrowing and preparation. Time spent queued or
holding an unclaimed ready ticket is excluded. `None` disables the execution
deadline; zero expires before preparation. Queue admission continues to use
`queue_timeout` independently.

The execution deadline uses a monotonic clock and covers model preparation,
native execution, UDF execution, and task/shared-memory waits. Expiration uses
the same cancellation path as `request.cancel()` and raises
`RequestExecutionTimeout` from `vane.execution.request_admission`. Manual
cancellation raises `RequestCancelled`; queue expiration raises
`RequestQueueTimeout`. The first accepted cancellation cause wins and is exposed
as `request.cancellation_reason` (`"cancelled"` or `"execution_timeout"`).
Preparation and completion also check the deadline, with completion arbitrated
under the cancellation/finish lock, so a delayed watcher cannot start native work
after expired preparation or return an overdue result. A callback arriving after
completion cannot cancel that request or interrupt a reused cursor.

Each timed, claimed request has one interruptible deadline watcher. Request
admission bounds this population; queued requests and executions without a
deadline create no watcher. Cancellation callbacks run independently so one
request's slow cleanup cannot delay another request's expiration. Completion
stops the watcher and removes its callback. Normal post-execution cleanup and
caller-side output delivery are outside the execution deadline.

`request.cancel()` returns true for the first accepted cancellation of queued,
ready, preparing, or running work. `execute()` raises `RequestCancelled` instead
of returning a result. Running cancellation interrupts the actual native cursor
after query startup and cancels this request's UDF execution scopes, including
admission and shared-memory waits. A blocked UDF's active subprocess is terminated;
shared pools remain open and replace terminated workers through their existing
recovery path. Other requests and already returned outputs keep their owners.
Repeated cancellation, or cancellation after execution has finished, returns
false. The native interrupt binding is fenced before `execute()` returns, so a
late callback cannot interrupt the next query on that cursor.

An accepted running cancellation reports `cancelling` until execution and cleanup
release the request slot; it then reports `cancelled` for manual cancellation or
`execution_timed_out` for execution expiry. Cancellation does not
return admission capacity early. A waiter can leave shared model initialization
without cancelling its initializer. A request that started the initialization
itself waits for that constructor to finish or fail; its pool remains owned by
the runtime. A deadline triggers cancellation rather than guaranteeing a hard
return time: shared initialization and pending cleanup can still delay completion.
Queue timeouts apply only before admission. Abandoned
unstarted tickets must be cancelled or shut down; the request context manager
does this on exit. `shutdown()` remains a cleanup operation after running
`execute()` returns; use `cancel()` to interrupt execution from another thread.

Successful execution, UDF failure, and worker exit all run query cleanup.
The request slot stays charged through uncertain or concurrent cleanup; retry
`request.shutdown()` after failure. The runtime retains pending cleanup owners
and also retries them from `close()`, without retaining request exceptions or
their tracebacks. Cleanup failure does not replace a primary execution error.
Request and runtime context-manager exit preserve that error as well.
Transport cleanup ownership does not require `track_data` or `data_limit`: a
request retains running UDF tasks, failed input leases, and unconverted output
grants until cleanup succeeds, even after a failed worker leaves its pool.
Completion callbacks remain owned separately from input cleanup. If executor
shutdown times out, the request stays `cancelling` and keeps its admission slot
until callbacks release their outputs and execution slots and cleanup is retried
through `request.shutdown()` or `runtime.close()`. This also applies without
`task_limit`, `track_data`, or `data_limit`.
Retries use each lease's shared-input ownership rules and each task's grant
identities, so cleaning one request cannot release another request's data.
Shared registered models remain resident. Shared-memory UDF outputs and their
zero-copy views keep their separate byte ownership after the request slot is
returned; a slow consumer retaining those views can cause a later request to
encounter the existing explicit byte capacity refusal. Native final Arrow
results can contain copies outside this ledger; callers must bound response
buffering separately. Request management never replays a failed UDF.

For request-limited runtimes, `drain()` first fences new requests and cancels
queued/unclaimed tickets. Requests that already claimed execution may finish
preparation and execution. `close(timeout=...)` waits for those request leases
and their cleanup before draining and closing the inner task/data/model owners.
Registration, runtime prewarming, and `acquire()`/`prewarm()` on handles returned
by `register()` are refused after drain. Public acquisitions already initializing
at the fence may finish initialization, but return no new borrow; the registry
keeps their pools for cleanup. Claimed preparation uses a separate binding to
its live request ticket, leaving the returned handle fenced. That binding
expires when the request releases its slot. Existing borrows remain valid until
their owners release them. A close timeout leaves the runtime draining for an
explicit retry.

`resource_snapshot()["request_admission"]` reports ready, running, queued,
completed, cancelling, explicitly cancelled, drained, timed-out and rejected requests,
aggregate queue wait seconds, cleanup-pending requests, and drain/close state.
Running counts include cancelling requests and requests still owning cleanup;
cancelled counts increase only after their slots are returned.
`timed_out_requests` counts queue expirations; `execution_timed_out_requests`
counts execution expirations after their cleanup releases the request slot.
`executed_requests`, `failed_executions`, and `execution_seconds` record claimed
executions once at their execution boundary, before query cleanup. Preparation
and cancellation callbacks are included; accepted cancellation and expiry are
not execution failures. `cleanup_seconds` updates when the slot is returned,
including time awaiting cleanup retries. Cleanup-only errors are represented
by pending owners, not `failed_executions`. `completed_requests` continues to
count non-cancelled slots returned, including failed executions.
`request.timing_snapshot()` exposes completed queue, execution, and cleanup
intervals; unstarted or unfinished intervals are `None`. These counters retain
no per-request history or exceptions. See the acceptance guide for full timing
boundaries and the separate driver-side percentile report.
The policy and
`AdmissionLease` are common execution components; native execution is the local
adapter. Managed result delivery is described below. HTTP/RPC endpoints and
a Ray request/delivery adapter remain later increments under #843.

## Managed result delivery

Configure an independent result budget and use `execute_result()` to hand a
materialized local result to a consumer through an explicitly owned iterator:

```python
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.result_delivery import ResultDeliveryLimits
from vane.execution.udf_local_model import LocalModelRuntime

models = LocalModelRuntime(
    session_id=plan.session_id(),
    session_config=plan.session_config(),
    request_limit=RequestAdmissionLimits(4, 16),
    result_limit=ResultDeliveryLimits(max_results=8, max_bytes=64 * 1024**2),
)
try:
    with models.request() as request:
        with request.execute_result(
            plan, bindings, conn=connection,
            execution_timeout=5.0, delivery_timeout=2.0,
        ) as result:
            for table in result:
                consume(table)
finally:
    models.close()
```

`result_limit` requires `request_limit`. Both capacities are positive integers.
Each managed request first waits for request admission without reserving any
result capacity. Once ready, it reserves a result slot and claims execution
under the request admission lock. Queued requests cannot occupy result capacity
needed by earlier ready requests. Slot exhaustion raises `ResultDeliveryFull`
before consuming the ready request or starting user code; the caller can retry the
same ticket, cancel it, or leave its request context. The original `execute()`
API continues to return its native result and does not enter this delivery gate.

Execution and its cleanup release the request slot independently of result
consumption. The local delivery adapter then sizes each Arrow IPC partition,
reserves its exact bytes, and encodes it into a fixed-size buffer. Byte exhaustion
raises `ResultDeliveryFull`; partially built delivery buffers are cleaned up.
This can happen after user code has executed and never authorizes automatic
query replay. Native result collection, IPC sizing/encoding work, DuckDB memory,
and the temporary overlap with the original materialized result are outside
this retained-buffer limit. Encoding makes one IPC copy per native partition;
this increment does not stream native execution or impose a whole-process
memory bound.

Iteration, or `result.take()`, exports one partition as a zero-copy Arrow table.
The final successful transfer releases the result slot. Its buffers remain
charged until their last underlying reference is gone: tables, slices, arrays,
buffers and zero-copy NumPy views can keep bytes charged after handle or runtime
close. New results may therefore obtain a slot but fail byte admission while
older consumer views remain live. Native metadata is available as
`result.result_schema`, `result.completion_status`, `result.stats`, and
`result.task_stats`. The native column names and completion status are preserved.

`delivery_timeout` is a finite non-negative total deadline starting when the
managed result is ready, after execution cleanup and IPC preparation. `None`
disables it; zero expires results with pending partitions before consumption. Queue
time and execution use their separate deadlines. A watcher expires an abandoned
result even if no caller polls it, and synchronous checks fence expired or
already-cancelled output before transfer. Each timed result has one watcher,
bounded by `max_results`; slow cleanup in one watcher does not delay another.

The deadline ends when all partitions have been transferred to the caller.
It does not time the caller's subsequent serialization, network sends, or
retention of exported views. HTTP/RPC adapters must own those operations and
drop their own references on disconnect. Incremental native output and serving
transport integration remain subsequent work in #843.

`result.close()` discards remaining output. `result.cancel()` accepts the first
cancellation and subsequent consumption raises `ResultDeliveryCancelled`;
expiry raises `ResultDeliveryTimeout`. Explicit/runtime close raises
`ResultDeliveryClosed` on subsequent consumption. These exceptions are in
`vane.execution.result_delivery`. The first terminal cause wins. Completion,
close and expiry fence copied callbacks; they do not close shared models or
another result's output. A handle supports one active consumer at a time.

States are `preparing`, `ready`, `closing`, and terminal `delivered`, `closed`,
`cancelled`, `delivery_timed_out`, or `failed`. `closing` means an operation,
cancellation callback, or failed payload cleanup still owns the result slot.
Concurrent or failed cleanup retains the owner and reports an error; retry
`result.close()` or `models.close()`. The runtime owns abandoned handles, so
cleanup does not depend on callers retaining them or on garbage collection.
Runtime drain fences request ingress; already ready results remain consumable.
Runtime close fences all results before cleaning any of them and permits
already-exported views to outlive the runtime.

`resource_snapshot()["result_delivery"]` reports result/byte limits, active,
preparing, ready and cleanup-pending results, live buffer count, total charged
bytes, external-consumer bytes, terminal outcome counts, refusals and close
state. Delivery bytes are a separate budget from UDF shared-memory accounting.
`delivery_seconds` and `delivery_samples` count ready-to-retirement intervals
once, including cleanup retries but excluding subsequent exported-view
lifetime. Results that never become ready have no delivery sample.
`result.timing_snapshot()` reports the completed interval, or `None` until it
exists. The interval is observational and does not extend the delivery deadline.
The common layer reuses `AdmissionLease`, output lease ownership, cancellation
scopes and monotonic deadlines; Arrow IPC allocation/materialization belongs
to the local adapter. Ray authorization and transport remain with Ray's adapter.

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

Global thread grants rotate between eligible pools, with one grant per pool
per round. A continuously busy pool cannot starve another pool, including
when pools use different runtimes or omit `task_limit`. Both pending grants
and new requests respect that arbitration; returning multiple threads does
not let the first pool consume them all.

Within each shared pool, its ordinary FIFO queue and each runtime policy also
rotate, with one grant per source per round. Queries with and without
`task_limit` therefore share cached pools fairly. A runtime has one source
regardless of its query count, and preserves its own query-level ordering.
Policies at their runtime limit are skipped without reserving a pool slot;
closing one query keeps the source subscribed until its last query closes.

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
Strict input/output byte envelopes are described below. Dependency-aware byte
waiting remains a follow-up under [#841](https://github.com/AstroVela/vane/issues/841);
this task limit alone does not establish a whole-process memory bound. Yielding during
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
while an identity is live. The ledger records identities, sizes, counters, and
pending input-cleanup owners, without retaining caller or query objects or
request tracebacks.

- A task borrows its shared-memory inputs before dispatching them to a worker.
  The borrow lasts through backend completion, including failure or cancellation
  cleanup. A worker's input ACK can return transport-budget credit earlier; it
  does not end this data borrow. Failed input cleanup keeps the borrow charged
  and the query active until transport cleanup succeeds. Query `shutdown()`
  retries those input leases, including partially released inputs and failures
  before dispatch; runtime close waits for that cleanup. Pending ref releases
  remain shared with later borrowers: retrying an earlier query transfers
  cleanup to the remaining borrowers without releasing their inputs. While
  an owner's release call is running, a new borrow of that input fails before
  dispatch; retry only after cleanup returns and the input is still valid.
  This also applies to
  accounting with `track_data=True` and no byte limit.
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
| `queries`, `tasks` | Query scopes and tasks awaiting backend completion or input cleanup |
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
Strict retained-byte admission and output-completion reservations are described
below. Data accounting alone preserves the observational behavior above.

## Strict shared-memory byte admission

Pass `data_limit` to reserve a complete input/output envelope before each UDF
invocation is submitted. It automatically enables data tracking and supports
registered actors, query-owned actors, and task-only native plans:

```python
from vane.execution.udf_data_admission import DataAdmissionLimits

models = LocalModelRuntime(
    session_id=plan.session_id(),
    session_config=plan.session_config(),
    data_limit=DataAdmissionLimits(
        max_bytes=256 * 1024**2,
        max_task_input_bytes=8 * 1024**2,
        max_task_output_bytes=16 * 1024**2,
    ),
    task_limit=TaskAdmissionLimits(max_running_tasks=4, max_queued_tasks=32),
)
```

All three byte limits are positive integers, and the total must fit at least
one input/output envelope. `task_limit` remains optional. Byte-limited plans
must use local shared-memory ref-bundle output; preparation validates every
node before creating actor pools. Native local-fast plans use this transport.

Existing task/pool arbitration runs first. Before its ready state is exposed,
the byte authority reserves the full per-task input and output bounds from
both the runtime ledger and the process transport budget. A refusal returns
any unused worker/thread/task grant immediately. No worker is submitted while
waiting for bytes. Ordinary and runtime-limited queries keep their existing
pool arbitration, and ready reservations are released on executor/query
cleanup if they are never submitted.

**By default, byte admission refuses immediately when capacity is unavailable.**
`DataAdmissionCapacityError` identifies the runtime or transport owner and
reports requested, used, and limit bytes. It does not cache an initialization
failure. A refusal observed by a reentrant pool or transport wakeup is handed
back to the request/state caller using fresh exception objects, and does not
become a permanent callback failure. Native execution surfaces the refusal as
a query error; release that execution's resources and retained consumer views
before retrying. A query can
be refused partway through a pipeline when its buffered results and the next
complete envelope cannot coexist, even when each batch fits individually.
Set `DataAdmissionLimits.wait` to enable the bounded waiting mode described
below. Neither mode automatically replays a query or UDF.

The input and output bounds count exact IPC bytes, including headers. All
blocks in one task's output share its output bound; repeated input slices of
the same allocation count once. An oversized materialized input is rejected
before creating its shared-memory allocation. Existing input descriptors are
validated before scheduling. Workers request the total output bytes before
allocating output mappings; an oversized output fails there. An output refusal
can occur after user code has run, so it never authorizes replay of that UDF.
Choose smaller batches or larger explicit bounds for oversized work.

An admitted task draws input allocations and output grants from its protected
transport reservation, without a second byte wait. Legacy transport users see
that reservation in the same shared-memory budget. Input ACKs do not create
another output credit for these tasks. Cancellation, failed submission/startup,
worker exit, and unused grants return their unused envelopes exactly once.
The task reservation keeps track of consumed output grants until they become
result allocations or are released. At backend completion it also retries any
grants left by failed delivery or worker cleanup. A failed grant or unused-byte
cleanup keeps its runtime/query owner and conservative byte charge for an
explicit cleanup retry; runtime close waits for that confirmation. Cleanup
does not release result allocations or grants belonging to another task.

As descriptors enter the ledger, reserved bytes convert to actual allocation
ownership. Shared allocations remain charged once across queries and roles.
Unused input headroom remains reserved until task completion even when an
input already exists in the ledger; this is a conservative envelope, not a
second charge for that mapping. Backend completion returns unused reservations;
input borrows are released after input transport cleanup succeeds. A failed
input cleanup can return unused headroom while keeping the actual input bytes
charged. Output references and zero-copy views retain their actual bytes until
the last owner releases them, including after runtime close.

The data snapshot additionally reports `limit_bytes`, the two per-task bounds,
`input_reserved_bytes`, `output_reserved_bytes`, `reserved_bytes`,
`reservations`, and `usage_bytes`. The invariant is:

```text
usage_bytes = retained_bytes + reserved_bytes <= limit_bytes
```

This bound covers data admitted to this runtime's transport ledger. Native
inputs waiting to be submitted, Python/Arrow serialization and decoding copies,
worker/model heap, DuckDB memory, and other runtimes remain outside it. The
process transport budget continues to account its own allocations and exposes
`task_reserved_bytes`; it is a separate constraint, not another runtime limit.

The protected task/output byte arithmetic comes from Ray's resource manager
and now lives in `vane.execution.byte_budget`. Both managers use it. Ray keeps
its soft shares, liveness escapes, authorization, and object-store transport;
the local strict envelope neither changes Ray's limits nor enables its spill
or dependency-aware waiting on local execution.

The same module also exposes `allocate_resource_reservations` and
`build_byte_budget_state` for operator budgeting. Ray supplies current-phase
eligibility, resource demands, concurrency maxima, and its existing arithmetic
tolerance. The shared calculations preserve baseline reservations, equal
surplus shares, proportional allocation under pressure, integer byte rounding,
and separate protected task/output capacity. Ineligible operators' retained
bytes remain charged before the available budget is divided.

These helpers calculate policy from an accounting snapshot; they acquire no
resources and supply no spill or scheduling capability. Ray retains its graph registration,
authorization, learned output estimates, and bounded liveness decisions. Local
execution can also apply the per-UDF byte policy described below, while
retaining complete task envelopes and checking hard shared-memory capacity
independently of soft reservations.

## Shared resource graph and local execution identity

`ResourceGraphMetadataProvider.collect_resource_graph_metadata(conn=...)` is the
common internal plan-adapter interface. `LocalResourceGraphAdapter` exports
metadata without changing the plan's UDF payloads; `RayResourceGraphAdapter`
registers the existing Ray query/operator identities during collection. Both
use the same native traversal and return the same schema, including
`udf_node_ids`, a mapping from pipeline nodes to physical UDF binding IDs. The
two traversal orders can differ at joins, so consumers must use this mapping.
The old native `collect_query_resource_graph_metadata()` entry point remains
compatible with its original Ray annotation behavior and three-field result.

Both backends use `vane.execution.resource_graph.ResourceGraph` for dependency
validation, deterministic ordering, materialization barriers, and phase
eligibility. Backend-specific unit validation remains separate: local units
identify native fragments, subprocess tasks and subprocess actor pools; Ray
units retain their existing process demands, output windows and authorization.
Local graph modules do not import Ray or its cluster resource coordinator.

Enable graph and UDF diagnostics explicitly with `LocalModelRuntime(...,
track_graph=True)`. Graph tracking can be used on its own: `prepare(plan, {},
conn=conn)` accepts local subprocess plans without model bindings, data tracking,
or admission limits. Each preparation gets a fresh execution query ID, even
when the same physical plan is executed again. Its UDF resource-unit contexts
travel in executor options, independent of model initialization fingerprints.
`UDFExecutor.resource_identity()` returns this context without changing numeric
executor statistics. Models can remain resident across independently rebuilt
queries and across concurrent executions.

`runtime.resource_snapshot()["prepared_query_graphs"]` reports graphs whose
preparation diagnostic owners have not been released. Direct `prepare()` callers
must shut down every returned owner, as for other preparation resources. Requests
handle this automatically; `request.resource_graph_snapshot()` retains the
structural snapshot for that request after execution. Runtime close clears the
diagnostic registry after its existing resource cleanup gates succeed. Snapshots
contain identities and dependencies, not serialized functions or credentials.
Releasing a graph diagnostic owner does not assert that other owners' cleanup
has succeeded.

These snapshots are marked `phase_tracking="structural_only"`. They expose
`initial_eligible_unit_ids`, not a live phase or a count of running operators.
The barriers currently come from the common pipeline representation; local-fast
still executes its native plan and does not emit barrier completion events into
this graph. DuckDB manages native sort, aggregation and join-build memory
inside local execution, as it does inside each Ray worker. Per-UDF byte budgets
use UDF admission lifetimes. Bounded waiting additionally protects the UDFs of
each activated query, without inferring native phase completion.
Graph tracking does not enable
local spill or change admission
limits. Tracking requires a plan supported by the common pipeline metadata
exporter and accepts only local subprocess UDFs; unsupported UDF backends are
rejected before model acquisition.

## UDF resource attribution

With `track_graph=True`, `runtime.resource_snapshot()["udf_units"]` and each
prepared/request graph snapshot's `udf_units` report activity by the same
`query_id`, `resource_unit_id`, `physical_node_id`, and `backend` used by the
graph. Tasks are observed at submission, worker execution, and completion:

- `preparing_tasks`: preparing inputs after task admission.
- `submitted_tasks`: dispatched to the pool, before its worker callback starts.
- `running_tasks`: executing a worker callback, excluding observed memory waits.
- `completing_tasks`: worker completion and result/cleanup callbacks. A shutdown
  timeout does not erase callbacks still in progress.
- `queued_tasks` and `ready_tasks`: requests queued for task/pool capacity and
  grants ready for submission. Each executor can have one such request.
  Custom pool authorities without passive observation report these counts
  (and the `task_capacity` reason) as `None`.
- `waiting_tasks`: submitted work waiting for shared memory or reacquiring its
  execution allowance. `waiting_by_reason` separates `shared_memory_input`,
  `shared_memory_output`, `execution_capacity`, and queued `task_capacity`.
  With byte waiting enabled, it also includes queued `byte_capacity` requests
  that own no execution slot.
- `byte_refusals`: counts strict admission refusals from the runtime or process
  transport budget. With byte waiting enabled, it counts transitions into
  runtime/transport byte pressure; waiting counts are reported separately. If
  only data views survive after their diagnostic scope is collected, this
  historical counter is `None`; live byte attribution remains available.

These counters describe UDF callbacks, including transport and worker startup;
they do not measure native operators or CPU utilization. Reading them never
requests admission, reserves bytes, or invokes wakeup callbacks. Snapshots copy
independently locked owners and are diagnostic observations, not one atomic
scheduler state. They contain scalar identities/counts only and do not retain
executors, requests, futures, or input buffers.

The `data` member is `None` unless `track_data=True` or `data_limit` is also
configured. When enabled, it reports unique associated `retained_bytes`,
`input_bytes`, `output_bytes`, allocation/lease counts, unused input/output
reservation bytes, and `cleanup_pending_tasks`. Failed cleanup keeps its unit
identity until the existing cleanup owner succeeds. Consumer output views
preserve the producing UDF's identity after query or runtime shutdown.

An allocation borrowed by multiple UDF units appears in each unit's retained
bytes and `shared_retained_bytes`. These unit totals **must not be summed** to
estimate runtime usage; the runtime data ledger charges each allocation once.
Input and output roles can overlap for the same reason. Unit reservation bytes
are unused portions of existing task envelopes, not additional reservations.
Untracked/native allocations and process-transport output grants that have not
become ledger allocations are outside these retained-byte counts.

The runtime lists prepared UDF units and units with live activity or data
owners. It drops finished, empty units even if a caller keeps an old plan.
A retained request graph can still show that request's final empty units.
Graph tracking supplies attribution. Per-unit byte admission is enabled
separately through `data_limit.unit_reservation_ratio`; progress-preserving byte
waits remain subsequent work.

## Per-UDF byte budgets

Set `unit_reservation_ratio` on the existing data limit to share its capacity
among UDF units through the common byte-budget policy:

```python
models = LocalModelRuntime(
    session_id=plan.session_id(),
    session_config=plan.session_config(),
    data_limit=DataAdmissionLimits(
        max_bytes=256 * 1024**2,
        max_task_input_bytes=8 * 1024**2,
        max_task_output_bytes=24 * 1024**2,
        unit_reservation_ratio=0.5,
    ),
)
```

The default `None` preserves aggregate byte admission. A finite ratio from zero
through one enables unit budgets and automatically enables local resource-graph
collection. The plan must therefore support that metadata interface. Model
bindings, captured session configuration, local subprocess backend requirements,
and shared-memory output requirements follow the existing preparation contract.

The runtime ledger calculates shares and checks admission under the same lock
that records reservations and allocation ownership. Eligibility comes from
unfinished task/ready reservations and the UDF making the current byte request.
Prepared units alone receive no share. An open query's unit remains eligible
while it has unfinished reservations. Query shutdown fences admission and
removes its units from eligibility. Retained data and pending cleanup charges
still reduce the available budget.
This uses UDF lifetimes and does not infer live native phases from the graph.

Each eligible unit has a baseline of one complete input/output envelope and a
maximum equal to the runtime byte limit. The shared
`allocate_resource_reservations` calculation preserves fitting baselines and
divides the configured fraction of surplus equally. Under pressure it allocates
that fraction proportionally; unallocated and rounded bytes stay shared. A zero
ratio still preserves fitting baselines. Ray uses the same arithmetic with its
own demands and eligibility; its object-store baseline remains zero because it
does not require a local-style complete task envelope.

The unit's protected output portion is its share multiplied by the declared
output/envelope ratio, rounded upward. The rest protects task/input bytes. This
supports unequal input and output bounds, including a budget that fits exactly
one complete task. The common `build_byte_budget_state` accepts this explicit
split; Ray retains its existing half-output split by default. Admission applies
the common byte checks to input and then output against one candidate state.
Both portions and process transport capacity must fit before publishing a grant.
Previously admitted tasks retain their full output reservation when shares
change. Runtime and transport hard limits continue to apply to every grant.

Budget accounting assigns each allocation's entire charge and input/output
classification to its first ledger lease until the last lease is released.
The charge stays with that unit even after its query finishes or only another
unit's input borrow remains. No query or executor object is retained for this
identity. Each unused reservation belongs to its submitting unit. The sum of
these unique charges equals runtime `usage_bytes`; the overlapping per-unit
associations described above remain available for diagnostics. Borrowing an
already charged allocation keeps the borrower's unused input headroom reserved,
as in aggregate admission.

`resource_snapshot()["data"]["unit_budget"]` reports:

- `reservation_ratio`, total `usage_bytes`, and `inactive_usage_bytes` charged
  before eligible shares are calculated.
- `shared_pool_bytes`, `shared_used_bytes`, and `shared_remaining_bytes`.
- `units`: scalar UDF identities, `eligible`, unique `usage_bytes`,
  `output_usage_bytes`, `protected_task_bytes`, `protected_output_bytes`, and
  per-unit `shared_used_bytes`. Output usage includes unused output envelopes as
  well as allocations first charged as output. These counters are distinct from
  retained-output association counts.

Reading snapshots is passive. With `wait=None`, a unit-share refusal raises
`DataAdmissionCapacityError` with `resource_unit_id` and an input/output
`reason`, and returns the unused task/pool grant. It can occur while aggregate
capacity remains because another eligible unit's share is protected. Existing
task/pool arbitration is preserved. Release retained data or finish admitted
work before retrying. Bounded waiting uses the additional progress contract below.

## Bounded byte waiting

Enable automatic byte waits with a finite queue bound and timeout:

```python
from vane.execution.udf_data_admission import DataAdmissionLimits, DataAdmissionWaitLimits

runtime = LocalModelRuntime(
    session_id=plan.session_id(),
    session_config=plan.session_config(),
    data_limit=DataAdmissionLimits(
        max_bytes=512 * 1024**2,
        max_task_input_bytes=8 * 1024**2,
        max_task_output_bytes=24 * 1024**2,
        unit_reservation_ratio=0.5,
        wait=DataAdmissionWaitLimits(max_queued_tasks=32, queue_timeout=10.0),
    ),
)
```

Waiting requires bound local UDF resource units and enables graph collection
automatically. It protects at least one complete input/output envelope for
**every UDF in an activated query**, including downstream UDFs that have not
submitted a task yet. A query activates on its first successful reservation;
its protected shares last until query shutdown. Merely preparing a query does
not activate it. This is deliberately conservative: native materialization
phases do not reclaim these protected shares early.

For example, two UDFs with a 32 MiB envelope need a minimum 64 MiB runtime and
transport capacity. Preparation raises `DataAdmissionProgressError` if that
minimum cannot fit the configured hard capacities. Activation waits if other
active queries or retained results leave too little room for all of its UDFs.
Completed queries' consumer views remain charged before dividing the available
budget. Use request admission to bound the number of simultaneous queries.

The common reservation and byte-budget algorithms allocate these shares. When
waiting is enabled without `unit_reservation_ratio`, its effective ratio is
zero: fitting per-UDF baselines stay protected and surplus remains shared.
Previously admitted tasks keep their full envelopes. An upstream UDF can wait
while the downstream UDF spends its protected envelope and releases the input.
Protection concerns the runtime's own ledger; other runtimes and ungoverned
transport users still compete for the process-wide shared-memory hard capacity.
That external pressure can cause a bounded wait to time out.

Byte checks run inside the existing physical-slot arbitration, after both a
pool slot and a global task-executor thread are available. A successful check
commits the complete runtime and transport envelope together with those slots,
before the task policy publishes an allowance. A byte waiter owns none of those
resources. The native caller can still retain its pending input; this queue is
not a limit on DuckDB's native buffers. Release and cleanup notifications retry
pending work without polling or blocking a worker. Repeated rejected guards do
not keep the fair dispatcher spinning on an unchanged capacity state.

Waiting uses `RuntimeTaskAdmission`'s query ordering and the shared pools'
existing source/global round robin. With no explicit `task_limit`, it uses a
capacity-only policy (`TaskAdmissionLimits.max_running_tasks=None`): backend
slots remain authoritative. With a task limit, its running and queue limits
also apply. Both paths preserve fairness alongside ordinary cached-pool users.

`max_queued_tasks` is a non-negative count of pending admission requests across
this runtime; zero requires immediate admission. `queue_timeout` is finite,
positive, in seconds, and starts at the admission request. It includes time
waiting for task/pool capacity as well as bytes. Immediate grants create no
watcher thread. Queued requests reuse the monotonic deadline watcher; expiry
is checked again before publishing a grant. Full queues raise
`DataAdmissionQueueFull` (or the separately configured `TaskAdmissionQueueFull`).
Expired requests report `DataAdmissionTimeout`; native execution surfaces the
failure as a query error. Exact IPC input/output bounds still fail promptly
with `DataBatchTooLarge`, even in waiting mode. An output bound covers the sum
of all output blocks in a task, not each compute batch separately.

Query shutdown and request cancellation remove pending requests, stop their
watchers and release unused grants. Failed transport cleanup retains its owner
and charge for retry. Drain fences new preparation and lets already admitted
queries finish; it does not cancel their waits. Runtime close still waits for
those query owners. Consumer views remain valid and charged until released.

Snapshots expose `queued_byte_admissions`, `max_queued_byte_admissions`, and
`byte_queue_timeout` under `data`, plus the effective `unit_budget` shares.
`udf_units[*].waiting_by_reason.byte_capacity` identifies UDFs waiting for bytes.
Snapshot reads are passive. Waiting does not add spill, exceed a hard limit,
replay user code, or govern native sort/aggregation/join memory.

## Ray boundary and validation

The registry uses the `shutdown` and `cleanup_pending` contracts already exposed
by both local and Ray actor pools. Adapter tests exercise both implementations.
Local registration accepts only `subprocess_actor`; it cannot cache a Ray pool
or its query-generation capability. Ray's driver resource graph, actor identity,
admission and generation checks remain authoritative and query-scoped. A future
Ray registry adapter must establish valid query borrowing authorization before
enabling cross-query model reuse. This change does not resolve the separate
named vLLM ownership path in [#251](https://github.com/AstroVela/vane/issues/251).

The affected tests include `test_local_resource_graph.py`, `test_udf_resource_usage.py`,
`test_local_udf_resources_native.py`, `test_udf_model_pool.py`, `test_udf_local_model.py`,
`test_udf_model_resources.py`, the query resource graph/builder/manager suites,
`test_udf_runtime_admission.py`, `test_udf_task_admission.py`,
`test_udf_data_lease.py`, `test_udf_data_transport.py`,
`test_udf_data_admission.py`, `test_udf_data_admission_native.py`,
`test_udf_actor_pool_lifecycle.py`, `test_udf_executor_lifecycle.py`,
`test_driver_udf_precreate.py`, and `test_udf_process.py` under `tests/fast/`.
Request deadlines also have coverage in `test_request_deadline.py`,
`test_udf_local_request_deadline_native.py`, and the parameterized cancellation
cases in `test_udf_local_request_cancellation.py`.
Managed result delivery, retained Arrow views, deadlines, and cleanup retries
have coverage in `test_result_delivery.py` and
`test_local_result_delivery_native.py`.
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
