# Variable shape Tensor contract

A Tensor column has one element type and a fixed number of dimensions. A shape
dimension declared as `NULL` in SQL or `None` in Python varies by row. Each
non-NULL value is a dense tensor in row-major order, with its own actual shape.

```python
waveform_type = vane.tensor_type(vane.sqltypes.DOUBLE, (None, None))
```

The corresponding SQL type is `TENSOR(DOUBLE, [NULL, NULL])`. The explicit shape
declaration fixes the rank without fixing the number of frames or channels.
Fixed shape Tensor declarations continue to describe one shape for the column.

Variable Tensor elements support Boolean, signed and unsigned 8/16/32/64-bit
integers, Float32, and Float64. Dimensions are nonnegative signed 32-bit integers;
rank is between 1 and 32. The product of a value's dimensions must equal its
element count and fit a signed 32-bit Arrow list offset. A non-NULL tensor has
no NULL dimensions or NULL elements. A zero dimension represents an empty
tensor and is distinct from a NULL tensor. NaN and infinity remain floating
point values.

The native storage is `STRUCT(data T[], shape INTEGER[rank])`, with the Tensor
logical identity and declared uniform dimensions retained in the type. Storage
does not authorize an implicit conversion between STRUCT and Tensor. Explicit
construction validates shape, element count, and element validity. A Tensor
value cannot silently acquire a different dtype, rank, or uniform shape.

Arrow uses the canonical
[`arrow.variable_shape_tensor`](https://arrow.apache.org/docs/format/CanonicalExtensions.html#variable-shape-tensor)
extension: a `data` list and fixed-size `shape` list of int32 values. Vane emits
row-major tensors with optional `uniform_shape` metadata. Nonidentity
permutations are rejected until an explicit layout conversion exists. The same
contract applies to Arrow import/export, IPC, persistence, Flight shuffle, and
Python row and batch UDF boundaries, including empty and all-NULL batches.
Dimension names are not accepted in this stage. Fully uniform Arrow tensors
use `arrow.fixed_shape_tensor`; a variable declaration retains at least one
unknown dimension. Extension metadata is limited to 16 KiB.

PyArrow 25 or later is required. Vane installs a Python binding for the existing
canonical variable Tensor registry entry so Arrow schemas and arrays can be
pickled by workers. The binding preserves Arrow metadata, including layouts
that Vane rejects at its own execution boundaries. Arrow scalars expose their
storage mapping; Vane row results and row UDF inputs materialize NumPy arrays.

Python row values materialize as detached NumPy arrays with their declared
dtype and actual shape. `fetchnumpy`, `fetchdf`, and `fetch_df_chunk` store those
arrays as object values in result columns, with SQL NULL represented by the
NumPy mask or pandas missing value. Batch UDFs receive and return Arrow extension arrays.
Malformed outputs fail before downstream execution.

`vane.tensor(data, shape)` / SQL `tensor(data, shape)` constructs a variable
Tensor from a flattened typed numeric list and its actual shape. A constant
shape list supplies the rank at bind time; a shape column must be a fixed-size
integer array. `tensor_data` and `tensor_shape` expose each value's elements and
actual dimensions. `vane.tensor_array(values, dtype)` constructs an Arrow Tensor
column from NumPy row values. Explicit Python values can use
`vane.Value(numpy_array, tensor_dtype)` as SQL parameters. NumPy inputs require
the declared dtype; strided inputs are copied into row-major storage.

## Audio specialization

The audio API uses Float64 tensors with shape `(frames, channels)`. Both frame
count and channel count can vary by row; mono is `(frames, 1)`. Channels must be
positive, while zero frames are allowed. The target sample rate remains the
resampling argument and can be retained as a separate column when needed.
NULL inputs propagate to a NULL waveform. Decoding, resampling, byte windows,
resource limits, and cancellation belong to the audio operators.

Tensor types and transport are base engine capabilities. The independently
loaded audio extension provides native audio computation, selected explicitly
by `audio_backend='native'`. The Python backend remains an explicit execution
choice; no compatibility layer or automatic fallback is part of this work.
