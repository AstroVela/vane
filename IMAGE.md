# Decoded Image values

Image contains decoded, interleaved HWC pixels. The supported pixel dtype is
UInt8 and the supported modes are `L`, `LA`, `RGB`, and `RGBA`. Grayscale
images retain a channel axis of length one.

## Types and storage

| Python | SQL | Storage |
| --- | --- | --- |
| `vane.image_type()` | `IMAGE` | STRUCT with variable mode and dimensions |
| `vane.image_type('RGB')` | `IMAGE('RGB')` | STRUCT with fixed mode and variable dimensions |
| `vane.image_type('RGB', H, W)` | `IMAGE('RGB', H, W)` | UInt8 ARRAY of length H × W × C |

The dynamic STRUCT fields, in order, are `data: UInt8[]`, `channel: UInt16`,
`height: UInt32`, `width: UInt32`, and `mode: UInt8`. Mode codes are L=1, LA=2,
RGB=3, and RGBA=4. A non-NULL image requires every field and pixel to be
non-NULL. Width and height must be positive. A fixed shape requires both
dimensions and a mode, and its pixel count must fit a signed 32-bit Arrow
fixed-size-list length. These representation limits do not reserve a memory
budget for an application; the number and size of materialized images still
contribute to query memory use. Fixed Image columns keep dense engine ARRAY
storage, but vector initialization does not reserve a full batch of pixels.
Pixel buffers grow to the rows actually written, including NULL padding.
Row-wise writers reuse owned capacity and grow it geometrically, so cumulative
allocation and copying remain proportional to the materialized pixel count.
Shared slices and borrowed Arrow buffers detach when they grow.
Copies and constant broadcasts operate on contiguous image rows. Materializing
many large images still consumes memory proportional to their actual pixel count.
Query descriptions display an Image's mode and dimensions instead of expanding
its pixels into strings. This also applies to Images nested inside containers.
SQL literals retain their complete pixel payload for reconstruction.

C API writers receive the raw writable capacity promised by their container:
created/reset chunks and table-function outputs reserve a standard batch;
`duckdb_create_vector` reserves its requested capacity; scalar, aggregate and
cast callbacks reserve their output span. This also applies to nested Images
and explicit list-capacity growth. Large fixed Images therefore require a
corresponding memory budget when creating a writable C API batch. Reading a
query result through the C API preserves its materialized pixel span.

`dtype.is_image()`, `dtype.is_fixed_shape_image()`, and `dtype.image_mode`
inspect the logical type. `dtype.shape` returns `(height, width)` for a fixed
Image and raises for a dynamic Image. `ImageMode`, `ImageFormat`, and
`ImageProperty` accept their string values and round-trip with `str()`.
`ImageFormat` names PNG, JPEG, TIFF, GIF, and BMP. `encode_image` currently
implements PNG; the other enum members do not imply encoder availability.

## Python values

Fetched cells are detached, C-contiguous `numpy.ndarray` values with shape
`(height, width, channels)` and dtype `numpy.uint8`. `vane.Image` is a typing
alias for that array, with no separate value wrapper or value methods.
Scalar Image parameters and plan serialization retain packed UInt8 pixels;
binding an ndarray does not allocate an engine `Value` object for each byte.
Constant constructors and casts keep one pixel payload per batch, and Image
attribute functions read metadata without expanding constant pixels.

```python
import numpy as np
import vane

pixels = np.zeros((48, 64, 3), dtype=np.uint8)
value = vane.Value(pixels, vane.image_type('RGB', 48, 64))
con = vane.connect()
image = con.execute('SELECT $1', [value]).fetchone()[0]
assert image.shape == (48, 64, 3)
```

An ndarray acquires Image semantics through a declared Image type; ordinary
ndarray inference continues to describe an ordinary array. Typed inputs may
be strided: the boundary copies their pixels into HWC order without changing
dtype or colors. Masked arrays, wrong pixel dtypes, unsupported channel
counts, and incompatible declared dimensions or mode raise an error.

A `PIL.Image.Image` in a supported mode infers dynamic `IMAGE`. Pillow is
optional for the base Image type and is required only for PIL input or the
Python codec backend. PIL inputs do not undergo implicit color conversion.
`ImageFile.decode()` and the `VideoFile` value reader methods retain their PIL
return contracts; their expression/SQL counterparts produce engine Images.

## Attributes and casts

Python functions, Expression methods, and SQL expose `image_width`,
`image_height`, `image_channel`, and `image_mode`. `image_attribute(image,
name)` accepts `height`, `width`, `channel`, or `mode`, including
`vane.ImageProperty` members in Python. Results are UINTEGER and NULL inputs
produce NULL. Attribute access requires no codec or I/O.

`expr.as_image(mode=None, height=None, width=None)` validates an existing
Image expression against the selected Image type. Ordinary casts never
convert colors or resize pixels. `TRY_CAST` produces NULL for a layout
mismatch. Raw STRUCT, ARRAY, and BLOB values cannot acquire Image semantics
through ordinary casts. The SQL `image(bytes, width, height, channels, mode)`
constructor accepts already decoded UInt8 bytes and validates their layout.

Combining equal Image types preserves that type. Different shapes with the
same known mode widen to `IMAGE(mode)`; different or unknown modes widen to
`IMAGE`. Assignment may widen constraints. Narrowing mode or dimensions
requires an explicit cast, including within nested containers.

## Crop and PNG encoding

The following functions have the same arguments in Python, Expression methods,
and SQL:

| Python function | Expression method | SQL | Result |
| --- | --- | --- | --- |
| `vane.crop(image, bbox)` | `expr.crop(bbox)` | `crop(image, bbox)` | Dynamic Image, preserving the input mode constraint |
| `vane.encode_image(image, image_format)` | `expr.encode_image(image_format)` | `encode_image(image, image_format)` | PNG bytes / BLOB |

`bbox` is `(x, y, width, height)`, following the coordinate order of
[Daft's crop API](https://docs.daft.ai/en/stable/api/functions/crop/).
Python accepts a tuple/list of integers or an Expression. SQL accepts an integer
LIST or a four-element integer ARRAY. Floating-point coordinates and booleans
are rejected instead of rounded. Origins fit signed BIGINT; width and height
are positive UINTEGER values. Pixels outside the input are filled with zero in
every channel, including alpha. Empty crops are rejected because Image requires
positive dimensions. A fixed input still produces a dynamic crop result:
`IMAGE('RGB', H, W)` becomes `IMAGE('RGB')`; generic `IMAGE` remains generic.

PNG encoding accepts `L`, `LA`, `RGB`, and `RGBA` inputs and preserves every
pixel, channel and mode. The format string is case-insensitive; Python also
accepts `ImageFormat.PNG`. Other formats raise an explicit unsupported-format
error. PNG compression and chunk layout are backend-specific; encoded bytes
are not promised to match between backends. These operators do not resize,
convert colors, read files, or write files.

NULL Images or NULL bbox/format arguments produce NULL. A non-NULL bbox must
contain exactly four non-NULL integers. Invalid arguments, resource failures,
missing dependencies, and cancellation propagate as errors.

Each operator accepts at most 100 million pixels and 256 MiB of pixel data per
input Image. Crop applies the same pixel limit to its output; both operations
limit materialized output to 256 MiB per vector batch. These checks do not
replace the application's memory budget for source data, retained results,
concurrent queries, or codec working memory. Input constants keep their single
pixel payload even when other arguments vary. Native crop copies bounded spans;
native PNG encoding streams through a bounded zlib buffer. Python crop uses
NumPy buffer views; Python PNG encoding uses Pillow and a bounded in-memory
writer. Both paths check interruption while processing data.

Backend selection uses `image_backend='python'|'native'`, with Python as the
default. The native functions are provided by the existing optional DuckDB
`image` extension. Load it explicitly before choosing native execution; an
unavailable native backend raises during binding. There is no automatic
fallback. Arrow, UDF, and Ray paths retain the declared Image result type.

```python
import vane

con = vane.connect()
vane.load_installed_extension("image", connection=con)
con.execute("SET image_backend='native'")
result = con.sql("""
    SELECT encode_image(
        crop(decode_image_file(image_file('photo.png'), 'RGBA'), [10, 20, 64, 48]),
        'PNG'
    ) AS thumbnail
""").fetchone()[0]
```

ImageFile decoding in this example performs governed FILE I/O. Crop and encoding
operate only on its decoded pixels. Byte-based `decode_image`,
additional encoders, and `image_hash` are separate API stages.

## Image to Tensor

Python `vane.image_to_tensor(image)`, Expression `expr.image_to_tensor()`, and
SQL `image_to_tensor(image)` expose the same conversion of decoded pixels:

| Input type | Result type | Shape |
| --- | --- | --- |
| `IMAGE` | `TENSOR(UTINYINT, [NULL, NULL, NULL])` | Per-row height, width, channels |
| `IMAGE(mode)` | `TENSOR(UTINYINT, [NULL, NULL, C])` | Per-row height/width, known channel count |
| `IMAGE(mode, H, W)` | `TENSOR(UTINYINT, [H, W, C])` | Fixed HWC dimensions |

Pixel values, UInt8 dtype, interleaved HWC order, and every channel are preserved.
Grayscale retains a channel dimension of one. There is no normalization,
resizing, axis permutation, or color conversion. NULL inputs produce NULL
Tensors; empty inputs retain the inferred result type. Non-Image SQL arguments
are rejected. Python accepts Image-typed values, HWC ndarrays, and supported PIL
inputs through the existing Image input boundary.

This operator runs directly in the base C++ engine, independently of
`image_backend` and without loading an optional extension. It shares the
input's dense pixel buffer and retains its owner: fixed Images also preserve
constant/dictionary representation, while dynamic Images produce LIST offsets
and three dimensions per row. It validates active Image layouts and retains
the existing per-input limit of 100 million pixels / 256 MiB. The conversion
stays in execution instead of scalar constant folding, which would expand a
Tensor into individual engine values. It does not allocate a second batch of
pixel data. Downstream materialization,
Arrow export, Python values, and consumers may copy or broadcast those pixels
and still require memory proportional to their output.

Fixed numeric Tensor vectors now allocate element storage for written rows,
using the same mechanism as fixed Images. The full writable capacity promised
by C API containers is still reserved. Plain SQL ARRAY and other Tensor element
types retain their existing allocation behavior.
Materialized relation query descriptions contain row counts and column types;
generating a description does not scan or stringify Tensor elements.

Arrow preserves `arrow.fixed_shape_tensor` or `arrow.variable_shape_tensor`,
including dtype and shape constraints, through IPC, UDFs and Ray/Flight. Python
scalar materialization follows the existing Tensor contract: variable Tensors
produce HWC ndarrays; fixed Tensors produce flat tuples with shape carried by
their declared type. Fixed Tensors are also accepted by `vane.func` and
`vane.func.batch`, including registered SQL UDFs, with logical shape validation
at the output boundary. Fixed Tensor row outputs must be flat lists or tuples
of the declared length; textual array encodings are rejected, including when
the Tensor is nested inside another output type.
Fixed Tensor Arrow batches support `to_numpy_ndarray()`
to obtain `(rows, H, W, C)` arrays. NULL rows must be handled before calling
that Arrow method.

```python
con = vane.connect()
result = con.sql("""
    SELECT image_to_tensor(convert_image(
        resize(decode_image_file(image_file('photo.png'), 'RGB')::IMAGE('RGB'), 224, 224),
        'RGB'
    )) AS pixels
""")
assert result.types == [vane.tensor_type(vane.sqltypes.UTINYINT, (224, 224, 3))]
batch = result.to_arrow_table().column('pixels').combine_chunks().to_numpy_ndarray()
```

The Image preprocessing operators in this example use the selected image
backend. `image_to_tensor` itself performs no file I/O or codec work.

## Resize and color conversion

| Python function | Expression method | SQL |
| --- | --- | --- |
| `vane.resize(image, w, h)` | `expr.resize(w, h)` | `resize(image, w, h)` |
| `vane.convert_image(image, mode)` | `expr.convert_image(mode)` | `convert_image(image, mode)` |

Both operators accept the existing UInt8 `L`, `LA`, `RGB`, and `RGBA` Images.
They operate on decoded pixels without file I/O. Python accepts Image-typed
values, HWC ndarrays and supported PIL inputs through the existing Image input
boundary. Width and height accept integers or Expressions; mode accepts a
string, `ImageMode` or Expression. Python keyword names match the table above.
SQL uses positional scalar arguments. Boolean, floating-point, decimal and
string dimensions are rejected; dimensions must be positive UINTEGER values.
Mode strings are case-insensitive, with no whitespace trimming or implicit
conversion from other SQL types. Unsupported modes raise an error.

| Operation and bind-time constraints | Result type |
| --- | --- |
| `resize`, input mode and both target dimensions known | `IMAGE(mode, h, w)` |
| `resize`, input mode known and target dimensions vary by row | `IMAGE(mode)` |
| `resize`, input mode unknown | `IMAGE` |
| `convert_image`, fixed input and target mode known | Fixed Image with the new mode and original dimensions |
| `convert_image`, dynamic input and target mode known | `IMAGE(mode)` |
| `convert_image`, target mode varies by row | `IMAGE` |

An option is known at binding when its expression is foldable and non-NULL;
supplied parameter values participate in this inference. A per-row dimension
or mode never changes the declared result type during execution. NULL Images
or NULL option arguments produce NULL. Statically invalid options raise during
binding, including for empty inputs; invalid per-row options raise when a
non-NULL row is evaluated. Empty relations retain the inferred Image type.
Ordinary casts still validate layout and never resize or convert colors.

Resize maps each output pixel center to `(index + 0.5) * source_size /
target_size - 0.5` independently on each axis, clamps it to the source edges,
and applies bilinear interpolation. It stretches to exactly the requested
width and height. There is no antialiasing prefilter for downsampling and no
gamma, transfer-function or color-profile conversion. Floating-point channel
results are clamped to `[0, 255]` and rounded to the nearest integer, with
halves rounded up.

For `LA` and `RGBA`, resize interpolates premultiplied color and alpha, then
unpremultiplies using the unrounded interpolated alpha. A zero interpolated
alpha gives zero color channels. The returned pixels use straight alpha.
Resizing to the original dimensions copies all bytes, including hidden colors
under transparent pixels. Backend floating-point arithmetic can differ at
rounding boundaries; bitwise equality across backends is not a requirement.

Color conversion preserves dimensions and uses full-range RGB luma:
`L = (299*R + 587*G + 114*B + 500) // 1000`. Gray-to-RGB copies L into each
color channel. Existing alpha is preserved when the output has alpha; adding
alpha uses 255. Dropping alpha keeps the color values without compositing
against a background. A conversion to the current mode copies all pixels.

```python
import vane

con = vane.connect()
vane.load_installed_extension("image", connection=con)
con.execute("SET image_backend='native'")
prepared = con.sql("""
    SELECT resize(convert_image(decode_image_file(image_file('photo.png')), 'RGB'),
                  224, 224) AS image
""")
assert prepared.types == [vane.image_type('RGB', 224, 224)]
```

These operators use the existing `image_backend='python'|'native'` selection.
The native implementation runs C++ pixel kernels in the explicitly loaded
DuckDB `image` extension. Python executes independent bounded NumPy helpers.
Neither pixel path requires Pillow; PIL inputs and Python codecs still do.
An unavailable native backend raises during binding, with no automatic fallback.

The existing 100-million-pixel per-image limit and 256 MiB input-image and
output-batch limits apply. Fixed output batches include NULL-row pixel padding
in their budget and are checked before pixel work. Dynamic outputs charge
each materialized row before allocation. Constant operands retain a single
source payload, and entirely constant calls produce one output payload per
batch. Kernels check cancellation in bounded pixel blocks; the Python helper
also bounds its coordinate and floating-point scratch arrays independently of
image dimensions. These payload limits do not bound process RSS: vector growth,
scratch space, input columns and downstream state consume additional memory.
Validation, allocation, resource and interruption errors propagate. Arrow,
registered row/batch UDFs, Flight and Ray retain dynamic or fixed Image types.

## Arrow, UDFs, and distributed execution

Arrow uses the `vane.image` extension type over the physical STRUCT or
FixedSizeList<UInt8>. Its metadata contains exactly `mode`, `height`, and
`width`; absent constraints are JSON null. Arrow IPC and Python pickling retain that metadata, including NULL values.
PyArrow Parquet round-trips retain it for dynamic images and non-NULL fixed
images. PyArrow 25 cannot read NULL FixedSizeList values from Parquet, including
values beneath NULL parents; use IPC or database storage for those fixed Image
columns. Import validates the
physical dtype, mode, dimensions, pixel count, and active pixel validity.
Image type and layout also survive Flight and worker plan transport.

Row UDFs receive HWC ndarray cells and accept ndarray/PIL outputs under a
declared Image return type. Batch UDFs receive Arrow columns carrying
`vane.image`. Outputs may use that exact extension type or its canonical
storage under the explicitly declared Image return type. A conflicting
extension mode or shape is rejected, even when pixel counts happen to match.
NULL parents and inactive UNION children do not expose hidden pixel payloads
for value validation. SQL-registered UDFs apply the same contracts.

Image ndarrays are unhashable. Maps whose keys contain Image use the existing
parallel `{'key': [...], 'value': [...]}` Python representation for unhashable
map keys.

The type, storage, attributes, and transport belong to the base engine. Pixel
operators and codecs belong to the existing DuckDB `image` extension.
`image_backend='python'|'native'` selects explicit pixel and codec paths;
loading the extension does not change the Image value representation.
