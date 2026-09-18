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
when its session identity, captured configuration, complete UDF payload and
actor count match. The payload fingerprint includes serialized initialization,
schema, device and execution settings. Changing a model or its version requires
a distinct registration name. Different sessions cannot share a registration,
even if their configuration dictionaries are equal.

Preparation validates all explicit bindings, then delegates to the existing
`ensure_local_subprocess_actor_pools_for_nodes` step to acquire fresh borrows and
inject pools through `local_actor_pool`. It returns the query resources to the
caller. Callers must retain these resources until executor cleanup has finished,
then call their existing `shutdown(kill=...)` cleanup path. A registered
model's returned resource releases only that query's borrow. Native query
cleanup and preparation rollback use this same resource contract.

Applications integrating at the internal physical-plan layer must run this
preparation/cleanup step for every execution. Preparation does not execute a
query. Do not run a second generic plan-preparation pass afterward: native node
collection does not currently expose the previously injected executor options.
Concurrent queries need independent cursors and plans constructed with those
cursors, all from the owning session. Unregistered UDFs keep their existing
query-owned pool lifetime. No callable or payload is automatically registered.

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
- Initialization failure is sticky for that registration. Partial constructor
  owners stay with the runtime, while query preparation receives the original
  error. Close retains uncertain cleanup owners for another explicit retry and
  continues attempting cleanup of the other models. It never relies on garbage
  collection to prove that cleanup completed.

Use explicit close after all query resources have been released. The runtime
context manager performs the same close on exit. Registration does not add
automatic eviction, model-level concurrency policy, GPU admission, or a global
memory cap; those belong to the subsequent resource-control work.

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
`test_udf_actor_pool_lifecycle.py`, `test_udf_executor_lifecycle.py`,
`test_driver_udf_precreate.py`, and `test_udf_process.py` under `tests/fast/`.
They cover shared contracts, real subprocess reuse, native sequential/concurrent
queries, cancellation, worker replacement, and ownership recovery. Follow the
installed-package and release checks in [DEVELOPMENT.md](DEVELOPMENT.md).
