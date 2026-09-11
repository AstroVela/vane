# Image parity audit

This audit compares installed Vane Python/native image operators with Daft's
image expressions on independently executed, checksum-verified inputs. It is
a correctness comparison, not a performance benchmark or an assertion that
the libraries promise identical contracts.

The 2026-09-11 source baselines are:

- Vane upstream main: `364212c49b8f0e48c238574be548763857cb7e2d`.
- Daft main: `b14432104c67f0ef80eab0d40b1b433ec301da56`.
- Daft release: `0.7.24`, commit `9c2b73e084356711bd74b2ca629464045737327b`.
  The image crates, relevant Python wrappers, core array/schema/file paths and
  complete Cargo dependency lock are unchanged between that release and this
  main revision. Runtime comparisons use the release wheel; they do not claim
  to exercise a separately compiled main wheel.
- Daft's Rust image dependency: `image` `0.25.10`.

See [the Chinese findings report](REPORT.zh-CN.md) for measured differences,
examples, limitations and recommendations.
The subsequent implementation and verification are recorded in
[the fixes report](FIXES.zh-CN.md). Its fresh captures and logs are under
`build/image-parity-fix/`; the original baseline captures remain unchanged.

## Run

Build and install Vane non-editably following `DEVELOPMENT.md`, with
`-Ccmake.define.VANE_LOADABLE_EXTENSIONS=image`, then build the
`vane_loadable_extensions` target. The base runtime and native_media extension must
have the same content-derived DuckDB SourceID. The audit opts into loading
its local unsigned extension fixture; it does not install or publish a
provider wheel or enable test signing keys.

Use a separate `.venv-daft` with `daft==0.7.24`. Both environments used
`numpy==2.2.6`, `pillow==12.3.0`, `tifffile==2026.9.9` and
`imagecodecs==2026.8.16`. The last two packages provide the common independent
decoder used to compare wide-pixel encoded outputs.

```bash
.venv/bin/python -I benchmarking/image_parity/audit.py generate \
  --root build/image-audit/corpus
.venv/bin/python -I benchmarking/image_parity/audit.py run \
  --root build/image-audit/corpus --engine vane-python
.venv/bin/python -I benchmarking/image_parity/audit.py run \
  --root build/image-audit/corpus --engine vane-native \
  --artifact build/python-release/vane_extensions/native_media.duckdb_extension
.venv-daft/bin/python -I benchmarking/image_parity/audit.py run \
  --root build/image-audit/corpus --engine daft
.venv/bin/python -I benchmarking/image_parity/audit.py compare \
  --root build/image-audit/corpus

.venv/bin/python -I benchmarking/image_parity/probe_contracts.py \
  --engine vane-python --output build/image-audit/vane-python-contracts.json
.venv/bin/python -I benchmarking/image_parity/probe_contracts.py \
  --engine vane-native --output build/image-audit/vane-native-contracts.json \
  --artifact build/python-release/vane_extensions/native_media.duckdb_extension
.venv-daft/bin/python -I benchmarking/image_parity/probe_contracts.py \
  --engine daft --output build/image-audit/daft-contracts.json
```

The corpus has 43 raw HWC arrays over ten modes and 33 encoded files. It
covers random, ramp, checkerboard and constant pixels, transparent boundaries,
an impulse downsampling example, primaries, 8/16-bit PNG, integer/float TIFF,
JPEG, GIF, BMP, palette/transparency, one-bit PNG, CMYK JPEG, EXIF orientation,
animation, WebP/ICO support and corrupt bytes. Hash cases include non-default
sizes and bit widths. Fourteen additional probes cover NULLs, crop boundaries,
zero-size validation, and generic versus fixed-shape tensor conversion.

Every engine verifies input checksums. File operators consume private copies.
Result records retain the input manifest, its canonical digest, library
versions and the audit script digest. Comparisons require matching manifests
and script identities and verify each compared array's checksum, shape and
dtype. Errors, unsupported input construction, layout differences, pixel
differences and encoded-byte differences are separate outcomes. An error in
both engines does not imply identical exception classes or messages.

Encoded outputs are also decoded by the same reference libraries to separate
compression/container-byte differences from decoded-pixel differences.
Metadata value methods (`ImageFile.metadata()`) are recorded separately from
SQL/expression metadata: the Daft value method uses Pillow, while its
expression implementation uses Rust `image`.

Artifacts and logs stay under ignored `build/image-audit/`. Source references
and measurements describe these pinned versions and this corpus, not every
valid file, platform, codec build, or malformed input.
