# Actor UDF execution and video outputs

Ordinary synchronous callable classes passed to `map_batches` use a
framework-owned serial worker in both `ray_actor` and `subprocess_actor`.
There is no additional public UDF base class or user-owned executor.

## Contract

- Construction, framework input adaptation, warmup, output encoding and
  cleanup run on the owning actor thread. `__call__` runs serially on one
  persistent worker. Input uses `map_batches(batch_format=...)`; conversion
  from Arrow happens on the owner before the compute worker runs. The
  default remains `pyarrow`.
- Existing Arrow and dict outputs retain their handling. A materialized dict
  containing multidimensional NumPy arrays can additionally use the declared
  schema for top-level fixed-shape numeric tensors with primitive/list/struct
  siblings. This path requires exactly the declared columns and list, tuple
  or NumPy column values. Empty detections and zero rows preserve their types.
- The default-format dict encoder requires tensors to match dtype and shape
  and use C-contiguous storage. This encoder performs no implicit dtype
  conversion or pixel expansion through `tolist()`.
- Returned arrays transfer ownership to the output; do not overwrite them
  while downstream consumers may still use them. Arrow retains their owners
  after the worker closes. Raw results/Futures are not cached across batches.
- A matching schema from the restricted encoder establishes canonical
  storage. Output validation remains in place; different logical storage
  contracts still use normalization. Ordinary Arrow-returning UDFs keep
  their existing output handling.
- Async-runtime adapters and generator functions retain execution on the
  actor thread. Iterators returned by a regular callable remain lazily
  consumed on the actor thread. Row-preserving/scalar/flat-map calls and task
  execution retain their existing execution path. Richer output types such
  as FILE/IMAGE and variable/nested tensors retain their Arrow path.
- `prepare_batch` is not a hook. Thread-local contexts are not inherited by
  the worker: put inference/autocast contexts needed during computation
  inside `__call__`.

```python
class AddOne:
    def __call__(self, table):
        return {"y": table.column("x").to_numpy() + 1}


result = relation.map_batches(
    AddOne,
    schema={"y": "BIGINT"},
    execution_backend="ray_actor",
    actor_number=1,
    batch_size=32,
)
```

The runtime submits one call at a time and waits for it; existing output
buffering/backpressure controls consumption. Close joins the worker even if a
UDF cleanup hook fails. No extra model-call concurrency is introduced.

## Format support and the local runner

[PR #536](https://github.com/AstroVela/vane/pull/536) adds the existing API's
`batch_format` option for Arrow, NumPy dict, pandas and cuDF. Tracking
[issue #961](https://github.com/AstroVela/vane/issues/961) tracks the roadmap
item, tensor semantics and backend parity. NumPy, pandas and cuDF outputs
must use the selected container and declared column names. The default Arrow
mode keeps its existing dict/Arrow output compatibility.

```python
class NumpyAddOne:
    def __call__(self, batch):
        return {"y": batch["x"] + 1}


result = relation.map_batches(
    NumpyAddOne,
    schema={"y": "BIGINT"},
    batch_format="numpy",
    execution_backend="subprocess_actor",  # or "ray_actor"
    actor_number=1,
    batch_size=32,
)
```

Non-null fixed-shape Tensor columns become writable NumPy arrays with shape
`(rows, *tensor_shape)`. Nullable tensors use object arrays containing one
ndarray or `None` per row; nullable ordinary columns use `numpy.ma.MaskedArray`
so SQL NULL stays distinct from floating NaN. Pandas uses per-row ndarray
tensor cells. NumPy inputs and pandas tensor cells own their mutable array
storage; user mutation must not change the source Arrow buffers. Returned
buffers must still remain unchanged while downstream consumes them. Tensor
output shape is checked and existing safe
casts to the declared element type are retained. Variable-shape tensor output
requires the Arrow format. Pandas/cuDF remain optional dependencies, and cuDF
requires a compatible CUDA environment and Arrow conversion support.

The video entrypoint still receives Arrow because `VideoFrameSource` also
supports logical IMAGE columns. A NumPy video UDF can directly consume frames
when its input column is a fixed-shape Tensor; that conversion is now performed
by the framework outside the model call.

The `feature/local-runtime` branch tracked by
[#838](https://github.com/AstroVela/vane/issues/838) shares `UDFExecutor` with
Ray. Its resident model owns the callable and compute worker across queries;
borrower completion/cancellation must not close them. This change adds no
local transport queue and leaves the runner's resource admission, shared
memory ownership and output backpressure in place. In particular,
[#941](https://github.com/AstroVela/vane/pull/941)'s streaming output must
consume runtime outputs through its existing bounded submission protocol.
The shared-memory store in [#948](https://github.com/AstroVela/vane/pull/948)
and its direct-IPC follow-up [#949](https://github.com/AstroVela/vane/pull/949)
remain transport-layer concerns; they do not need their own compute worker.
Local GPU support follows that branch's explicit model registration rules.

## Video entrypoint

Run `vane_batch_main.py` from the directory containing the benchmark scripts
and prepared YOLO weights. It passes an optional detector class to the
original `vane_main.main`, retaining the source, crop, PNG and Parquet path.
The shared numerical helpers and original default detector are unchanged.

## Recorded validation

### Format integration check

On 2026-10-04, code at `a076b94c33f4844ffefcf172e079a3fb66d336a3`
was checked with three paired Arrow/NumPy queries on one RTX 2080 Ti.
Each query used the same 64 decoded/resized RGB frames repeated four times
(256 frames), YOLO11n, batch size 32, one actor, Torch threads 1 and a 2 GiB
Ray object store. Both formats used the same inference, crop/PNG and Parquet
path. Two 32-frame warmups preceded alternating format order.

| Pair | Order | Arrow seconds | NumPy seconds | NumPy throughput change |
| --- | --- | ---: | ---: | ---: |
| 1 | Arrow, NumPy | 26.657 | 14.792 | +80.21% |
| 2 | NumPy, Arrow | 15.085 | 15.044 | +0.27% |
| 3 | Arrow, NumPy | 15.133 | 15.202 | -0.45% |

All six queries produced 1,000 rows with equal schemas and exact content
hashes. The paired median was +0.27%; the first pair's large variation and
the near-equal remaining pairs do not establish a reliable throughput gain.
Timing includes actor startup, input reading, inference, crop/PNG, Parquet
drain and connection close; it excludes Ray cluster startup, output validation
and cleanup. This finite cached-input check is not an original-video or
steady-state benchmark. Generated outputs and the input cache were removed
after validation; configurations, identities, measurements and hashes were
retained.

### Earlier experimental protocol

The frozen experiment baseline was Vane `9cccef7eefd`, Ray 2.58.0, Python 3.12,
PyArrow 25.0.1, Torch 2.7.0, YOLO11n, one RTX 2080 Ti, 640x640 RGB, model
batch32, Torch/OMP threads 1, object store 20 GiB and default allocator settings.
These measurements came from the earlier experimental base-class protocol.
They do not validate the current ordinary-class implementation or establish
a throughput gain from adding a serial worker. The format integration check
above is a separate comparison on the shared ordinary-class runtime.

- Three paired five-minute cached-pixel windows: throughput gains +6.92%,
  +7.13%, +5.90%; paired median +6.92%.
- Three paired original-video windows: +1.51%, +2.18%, +3.85%; median +2.18%.
- A finite 8,192-frame cached query, including output drain: 222.42 -> 208.90
  seconds; both produced 21,339 rows with equal schema/content hashes.
- Twelve original videos: 131.86 -> 129.44 seconds, 19,433 equal-content rows.
- Canonical output validation/conversion boundary: 0.091 -> 0.016 ms/frame.
  This local improvement did not establish a stable independent FPS gain.

Cached and original-video protocols are separate; their percentages cannot
be added. The historical default PyAV/resize source was not pixel-equivalent
to Ray's source. Later Decord/Pillow input validation was an experimental
comparison adapter, not a complete product VideoFile backend replacement.

`tests/fast/test_udf_actor_callable.py` covers thread ordering, tails, empty
output, nested NULLs, storage conversion, ownership after close, backpressure,
errors, hooks, instance isolation and the public actor APIs. Only tests
related to the review/refactor are run for this revision.
