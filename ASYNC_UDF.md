# Async UDFs

Vane recognizes `async def` in `vane.func`, `vane.func.batch`, `vane.cls`,
`vane.cls.batch`, Relation `map` / `map_batches`, and `vane.attach_function`.
These UDFs run on an executor-owned event loop in the existing subprocess or
Ray worker. Use asynchronous clients for I/O; blocking Python code still blocks
that loop.

```python
import asyncio
import vane

@vane.func(return_dtype="BIGINT", max_concurrency=8, timeout_s=5)
async def double(value):
    await asyncio.sleep(0.001)
    return value * 2

with vane.connect() as con:
    source = con.sql("SELECT i AS value FROM range(100) t(i)")
    result = source.select(double(vane.col("value")))
    print(result.fetchall())

    vane.attach_function(double, alias="async_double", parameters=["BIGINT"], connection=con)
    print(con.sql("SELECT async_double(42::BIGINT)").fetchall())

assert asyncio.run(double(21)) == 42
```

## Concurrency and timeouts

`max_concurrency` is a positive integer counting concurrent **user calls in one
executor**. `timeout_s` is an optional positive finite number of seconds for each
call. Boolean values are rejected for both parameters. Setting either option on
a synchronous UDF is an error.

| Entry point | Concurrent call unit | Default limit |
| --- | --- | --- |
| Async `func` / function `map` | One row | 32 |
| Async `cls` / class `map` | One row on the same instance | 1 |
| Async `func.batch` / function `map_batches` | One compute batch | 1 |
| Async `cls.batch` / class `map_batches` | One compute batch on the same instance | 1 |

`actor_number` continues to control the number of class instances. Each instance
can have up to `max_concurrency` calls; function tasks each have their own limit.
Neither parameter is a cluster-wide request rate limit. Ray actor RPCs remain
serial, and overriding `ray_options.max_concurrency` is rejected.

The timeout excludes queueing and class initialization. It uses cooperative
cancellation: code that blocks the loop or suppresses cancellation can exceed
the deadline. Concurrent calls share a class instance, so choose a limit greater
than one only when its state and clients support concurrent use.

## Class lifetime

Use optional `async aopen(self)` and `async aclose(self)` hooks to own clients.
Both must return `None`. Constructors must remain synchronous. Construction,
opening, calls and closing all happen on the same running loop and thread.
Query planning and serialization capture the definition and constructor
arguments without opening the client.

```python
@vane.cls(actor_number=2, return_dtype="BIGINT", max_concurrency=4)
class Add:
    def __init__(self, offset):
        self.offset = offset

    async def aopen(self):
        self.loop = asyncio.get_running_loop()

    async def __call__(self, value):
        assert asyncio.get_running_loop() is self.loop
        await asyncio.sleep(0)
        return value + self.offset

    async def aclose(self):
        pass  # Close an asynchronous client here.

add = Add(10)
with vane.connect() as con:
    print(con.sql("SELECT 1 AS value").select(add(vane.col("value"))).fetchall())

async def eager():
    async with Add(10) as local:
        return await local(1)

assert asyncio.run(eager()) == 11
```

Eager class calls require `async with`; eager functions return an awaitable.
Eager execution uses the caller's loop and does not create a scheduler or enforce
query concurrency. It does honor `timeout_s`. Worker instances are independent
of eager instances. `aopen` completes before an actor becomes ready. Failed
initialization attempts `aclose`; failed actor cleanup can retry, so `aclose`
should tolerate repeated calls.

## Batch calls and Relation APIs

Decorated batch UDFs take Arrow arrays and return one Arrow array of the same
length, with the existing dtype, Struct, NULL and governed-type checks.

```python
import pyarrow as pa

@vane.func.batch(return_dtype=pa.int64(), batch_size=16, max_concurrency=4)
async def batch_double(values):
    await asyncio.sleep(0)
    return pa.array([None if value is None else value * 2 for value in values.to_pylist()])
```

Raw Relation `map_batches` callables receive a `pyarrow.Table` and must return a
materialized `Table`, `RecordBatch` or dictionary of columns. They may change the
row count. Async batch iterators and async generators are not supported.

```python
async def identity(table):
    await asyncio.sleep(0)
    return table

with vane.connect() as con:
    result = con.sql("SELECT i AS value FROM range(100) t(i)").map_batches(
        identity,
        schema={"value": vane.sqltypes.BIGINT},
        execution_backend="subprocess_task",
        batch_size=16,
        max_concurrency=4,
        timeout_s=5,
    )
    print(result.fetchall())
```

The same options are accepted by Relation `map` and by raw `attach_function`.
SQL registration of decorated callables inherits their configuration; overrides
are rejected. Async `flat_map` and the legacy synchronous `create_function`
callback API are outside this protocol.

For batch concurrency `C > 1` and an explicit `batch_size=B`, payload construction
sets the default submission target to `B * C` using `min_task_batch_size`.
An explicit submission target is respected. Existing byte limits can shorten a
submission; small inputs may expose fewer than `C` batches. Concurrency does not
span worker submissions.

## Execution and failure semantics

Row execution uses at most `C` worker tasks and places results at their original
input indices. Batch execution has an ordered window of at most `C` batches;
completed batches waiting behind an earlier one continue to occupy window slots.
This bounds lookahead when a first batch is slow. Results preserve input/output
association within a submission; SQL still needs `ORDER BY` for global order.

The row-class adapter owns row scheduling. The surrounding table executor runs
one adapter call at a time, avoiding a second concurrency multiplier. Native row
calls keep existing DEFAULT NULL skipping and scalar `RETURN_NULL` behavior;
protocol and output-validation errors are not converted to NULL.

On a call failure or timeout, the scheduler cancels and drains sibling calls
before reporting failure. The executor is then retired; queued submissions or
fragment retries against that executor report a bounded summary of its first
failure without invoking user code again. Abort discards buffered input tails;
normal completion flushes them. A cleanup failure is attached as a
bounded diagnostic without replacing a primary call failure. External query
cancellation uses existing worker termination mechanisms; forced termination
cannot guarantee user cleanup hooks run.

There is no new per-row retry policy. Existing task/actor retries can replay
remote requests, so externally visible operations must tolerate replay. Async
tasks bypass the deserialized-callable cache to avoid reusing loop-bound state.
Do not keep loop-bound clients in module globals or leave detached background
tasks running across calls; the owned loop runs only while processing work and
is closed with its executor.

All new native UDF payloads use version **2**, with `execution_kind`,
`invocation_granularity`, `max_concurrency` and `timeout_s`. Version 1 plans are
rejected; there is no compatibility decoder or async fallback. Plans preserve
these fields through serialization and replay, and `EXPLAIN` shows the resolved
execution settings. Existing synchronous UDF and AI-wrapper execution paths
retain their own call protocol.

The API and bounded execution model were informed by
[Daft functions](https://docs.getdaft.io/en/stable/custom-code/func/),
[Daft classes](https://docs.getdaft.io/en/stable/custom-code/cls/), and
[Daft's async execution implementation](https://github.com/Eventual-Inc/Daft/blob/1feced9b5af78586da19dc10d32c5bc0c25855e0/src/daft-local-execution/src/streaming_sink/async_udf.rs).
Vane's scheduler and lifecycle implementation use the existing Vane executor,
admission, output and cleanup contracts.
