# Experimental video BatchUDF

`vane.udf.BatchUDF` is an opt-in contract for ordinary synchronous Ray Actor
`map_batches`. The original video entrypoint keeps its existing detector.

## Contract

- Construction, `prepare_batch`, warmup, output encoding and cleanup run on
  the owning Actor thread. `__call__` runs serially on one persistent worker.
- `__call__` returns a materialized dict with exactly the declared columns.
  Supported output types are primitive values, lists/structs, and top-level
  fixed-shape numeric NumPy tensors. Empty detections and zero rows preserve
  the declared schema.
- Tensors must match dtype and shape and use C-contiguous storage. There is no
  implicit dtype conversion or pixel expansion through `tolist()`.
- Returned arrays transfer ownership to the output; do not overwrite them
  while downstream consumers may still use them. Arrow retains their owners
  after the worker closes. Raw results/Futures are not cached across batches.
- A matching schema from the restricted encoder establishes canonical
  storage. Output validation remains in place; different logical storage
  contracts still use normalization. Ordinary Arrow-returning UDFs keep
  their existing output handling.
- Async/generator callables, async-runtime hooks, FILE/IMAGE outputs,
  variable/nested tensors, row-preserving batching and other execution
  backends are outside this initial contract. An ordinary class with a
  `prepare_batch` method does not opt in. Thread-local contexts are not
  inherited by the worker.

```python
from vane.udf import BatchUDF


class AddOne(BatchUDF):
    def prepare_batch(self, table):
        return table.column("x").to_numpy()

    def __call__(self, values):
        return {"y": values + 1}


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

## Video entrypoint

Run `vane_batch_main.py` from the directory containing the benchmark scripts
and prepared YOLO weights. It passes an optional detector class to the
original `vane_main.main`, retaining the source, crop, PNG and Parquet path.
The shared numerical helpers and original default detector are unchanged.

## Recorded validation

The frozen experiment baseline was Vane `9cccef7eefd`, Ray 2.58.0, Python 3.12,
PyArrow 25.0.1, Torch 2.7.0, YOLO11n, one RTX 2080 Ti, 640x640 RGB, model
batch32, Torch/OMP threads 1, object store 20 GiB and default allocator settings.
These records predate packaging the draft on newer main; they are not tests or
performance claims for the new publication head.

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

`tests/fast/test_udf_batch_callable.py` covers thread ordering, tails, empty
output, nested NULLs, storage conversion, ownership after close, backpressure,
errors, hooks, instance isolation and the public Ray Actor API. The combined
frozen candidate passed the 1,835-test base release gate with 38 optional skips.

## Single-chunk frame input

The benchmark frame adapter reuses `chunk(0)` for a single chunk and combines
multiple chunks. This applies to the optional BatchUDF detector, the original
detector and crop adaptation. NULL, dtype, shape and contiguity validation
remain unchanged. A nonzero slice offset still identifies the correct pixels.

On 32 frames of 640x640 RGB uint8, the old combine operation copied 37.5 MiB.
A same-Actor interleaved comparison over 24,576 measured frames reduced input
adaptation from 0.3998 to 0.0082 ms/frame and detector calls from 23.3440 to
22.9432 ms/frame. Crop-only replay reduced 35.228 to 34.960 ms/frame; separate
cached-pipeline paired windows had a +0.36% median throughput gain. These are
different scopes and cannot be added into an end-to-end speedup.

The focused test verifies pixel values and storage sharing for single-chunk
tensors, slices, multi-chunk input, empty input and NULL rejection. Numerical
helpers, model configuration and the default detector selection are unchanged.
