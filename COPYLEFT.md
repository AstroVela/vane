# GPL-family licenses and release obligations

Vane's own code is Apache-2.0; inherited DuckDB code remains MIT. Those
statements do not replace third-party grants. The supported native profile
uses the permissive alternatives below and LGPL media libraries. Do not
delete upstream GPL text to change a component's apparent license.

## Reviewed source and native dependencies

| Component or notice | Treatment |
| --- | --- |
| Vendored and linked Zstandard | Select BSD-3-Clause from the BSD/GPL dual grant; retain the original notice. |
| Vendored Mbed TLS | Select Apache-2.0 from `Apache-2.0 OR GPL-2.0-or-later`. |
| Generated libpg_query parser | Preserve `GPL-2.0-or-later WITH Bison-exception-2.2`; the exception permits distribution of the larger parser-using work under chosen terms. The copyright, exception, and full GPL text are also shipped in `LICENSES/Bison-parser-notice.txt`. |
| Arrow's LLVM notice | Its GPL reference describes an exception in the Apache-2.0 terms; it does not make Arrow GPL. |
| gRPC's bundled third-party notices | Preserve the MPL-2.0 terms, including their definition of GPL-family secondary licenses. A definition does not select that secondary license. |
| Retained Spark LICENSE | The path containing `spark-ganglia-lgpl` is listed under Apache-2.0 in an upstream inventory. That Java connector is not included in Vane. |
| FFmpeg | LGPL-2.1-or-later for the supported build. Default features are disabled; GPL, version3, nonfree, and unreviewed codec features are rejected. |
| libsndfile and libsoxr | LGPL-2.1-or-later; corresponding sources and relinking materials accompany native extension wheels. |
| mpg123 1.33.4 | LGPL-2.1-only, as specified by upstream COPYING and library headers. The pinned vcpkg SPDX `-or-later` conclusion is inaccurate. |
| LAME 3.100 | LGPL-2.0-or-later, as granted in libmp3lame/mpglib headers. Preserve COPYING and those headers; the pinned vcpkg summary omits the later-version grant. |

The libsndfile/SoXR audio profile is introduced by audio parity PR #787.
This license change must be integrated with that profile before publishing
its audio artifacts; it does not change audio algorithms or codec selection.

Build tools such as GCC, Bison, and `ffmpeg-bin2c` are separate from linked
runtime dependencies. GCC's runtime exception and Bison's output exception
must be evaluated when their output contains tool code. Using a GPL compiler
does not itself license the application under GPL. The TPCH, TPCDS, and TPCE
benchmark sources have separate additional terms and remain excluded from
Vane release artifacts.

## Python media dependencies

Python dependencies are installed separately; the base Vane wheel does not
redistribute their wheels. A deployment that copies those distributions must
review their exact artifacts, bundled libraries, and notices as well.

| Audited Python distribution | Wrapper and bundled-library distinction |
| --- | --- |
| SoundFile 0.14.0 | BSD-3-Clause wrapper; its binary wheel includes LGPL libsndfile and codec dependencies with their own notices. |
| python-soxr 1.1.0 | The wrapper itself is LGPL-2.1-or-later and bundles libsoxr; it is not a BSD-only wrapper. |
| PyAV 17.1.0 | BSD-3-Clause wrapper. The audited wheel reports LGPL-3.0-or-later FFmpeg and includes additional codec binaries. The maintainer describes wheel licensing as LGPL with commercial exceptions or a GPL alternative. Do not infer redistribution rights from the BSD package field or a library filename alone. |

For PyAV redistribution, retain the exact wheel and its component notices,
record the applicable codec grants/exceptions, and review those terms. If
those rights cannot be established for a target wheel, use a source build
against an independently reviewed FFmpeg configuration or the supported
native media profile. This is a distributor responsibility; merely declaring
`av` in optional Python dependencies does not copy its binaries into Vane.

Primary references: [SoundFile](https://github.com/bastibe/python-soundfile),
[python-soxr](https://github.com/dofuuz/python-soxr),
[PyAV license](https://github.com/PyAV-Org/PyAV/blob/v17.1.0/LICENSE.txt), and
[PyAV maintainer clarification](https://github.com/PyAV-Org/PyAV/issues/2270#issuecomment-4594631670).

## Source delivery and checks

Follow [the native materials workflow](NATIVE_MEDIA_EXTENSIONS.md#release-materials)
for static LGPL redistribution. Notices alone are insufficient: include exact
library sources/patches, recipes, application source or relinkable objects,
instructions, and evidence that modified libraries can be rebuilt and used.
Recipients install prebuilt wheels normally and do not need a compiler to use
Vane. LGPL and GPL do not prohibit PyPI distribution.

Complete upstream source archives can contain GPL tools or tests that are
not compiled into the LGPL library. Preserve their own grants and sources.
Record the licenses of **all delivered materials** in the inventory's
`materials_license_expression`, and include them in the wheel's overall SPDX
expression. Do not describe the complete archive as solely LGPL because its
linked library is LGPL. Keeping independent sources together is not, by
itself, a license change to Vane's own code.

`scripts/check_copyleft.py` compares GPL-family and SSPL markers in release source and
license records against `LICENSES/copyleft-review.json`. It rejects new,
changed, or missing reviewed records, changes to the pinned vcpkg baseline,
and unsupported FFmpeg defaults/features or direct GPL codecs. With
`--share-dir`, it also checks GPL-family copyright records in the installed
native dependency graph. CI runs both checks. Source-artifact validation
repeats the source/configuration checks; base-wheel validation requires the
complete, unchanged Bison notice. Optional-wheel validation verifies declared
material roles, licenses, byte sizes, and hashes at root and dependency levels.
Every file under `LICENSES/` and the source roots is considered regardless of
suffix, including Markdown notices and source templates. Test/data trees are
excluded; explicitly packaged DuckDB tool/script files remain included. The
review policy itself is the trusted input and cannot contain its own digest;
sdist validation instead compares its complete bytes against the reviewed
checkout before applying the source inventory.

Installed checks require the reviewed notices for base dependencies by default.
Pass `--feature native-audio`, `--feature native-image`, and/or
`--feature native-video` for each selected vcpkg feature. The pinned dependency
notice map includes required transitive notices; missing records and changes
that remove their GPL-family wording are rejected. Unselected optional
dependencies are not required, but any additional installed notices are still
audited. The native media CI job passes all three features explicitly.
Host build tools such as `ffmpeg-bin2c` are checked when present, but are not
required in the target triplet's share tree during a cross build.
Every checked dependency must also carry one matching `SPDXRef-port` record
in its installed `vcpkg.spdx.json`. Its package name and complete `versionInfo`,
including the port revision, must match the reviewed inventory. An unchanged
copyright file does not approve a different dependency version; missing or
ambiguous version metadata is rejected. The SBOM's license conclusion is not
used to override the reviewed upstream grant.

The dependency-review workflow separately denies the explicit GPL/AGPL
`-only` and `-or-later` variants. That action reviews dependency changes; it
does not inspect every bundled native library inside a Python wheel.

The marker inventory is a change detector, not a complete license recognition
engine. Review upstream grants, selected alternatives, compiled features,
source correspondence, and the exact wheel contents before updating its
hashes or releasing. An unchanged notice cannot prove an unchanged binary
configuration. Custom extensions outside the supported native profile need
their own compatibility review; do not treat the LGPL material validator as
approval of an arbitrary GPL-linked combined work.

Legal/source references: [FFmpeg](https://ffmpeg.org/legal.html),
[Bison conditions](https://www.gnu.org/software/bison/manual/html_node/Conditions.html),
[GNU linking FAQ](https://www.gnu.org/licenses/gpl-faq.en.html#LGPLStaticVsDynamic),
[GNU aggregation FAQ](https://www.gnu.org/licenses/gpl-faq.en.html#MereAggregation),
and [PEP 639](https://peps.python.org/pep-0639/).
