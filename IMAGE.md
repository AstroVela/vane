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
`ImageFormat` names PNG, JPEG, TIFF, GIF, and BMP; the enumeration does not
add byte encoding or decoding functions.

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
`image_backend='python'|'native'` continues to select explicit codec paths;
loading the extension does not change the Image value representation.
