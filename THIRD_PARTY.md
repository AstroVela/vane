# Third-party software

This document describes third-party code distributed in Vane source or binary artifacts. It is an inventory, not a replacement for the referenced license text.

## Derived source

| Component | Use in Vane | License | License text |
| --- | --- | --- | --- |
| DuckDB and DuckDB Python | Core SQL engine, Python API, and the base for Vane's engine modifications | MIT | `LICENSES/DuckDB-MIT.txt` and `external/duckdb/LICENSE` |
| DuckDB HTTPFS | Statically linked HTTP/S3 filesystem extension, fetched at the pinned revision in `external/duckdb/.github/config/extensions/httpfs.cmake` | MIT | `LICENSES/DuckDB-MIT.txt` |
| ALP and ALP-RD compression algorithms | Compression implementation retained in the DuckDB source tree | MIT | `external/duckdb/src/include/duckdb/storage/compression/alp/algorithm/LICENSE` and the corresponding `alprd` path |
| Spark-compatible Python API | Compatibility layer retained from DuckDB Python | Apache-2.0 | `duckdb/experimental/spark/LICENSE` |

Vane is not affiliated with, endorsed by, or maintained by the DuckDB Foundation. DuckDB is a trademark of the DuckDB Foundation.

## Vendored native dependencies

DuckDB vendors permissively licensed native libraries below `external/duckdb/third_party/` and selected extensions below `external/duckdb/extension/`. Their license files are preserved in those directories and included in Python release metadata.

The release allowlist includes only dependencies needed by the core engine and the `core_functions`, `icu`, `jemalloc`, `json`, and `parquet` extensions. In particular, release artifacts exclude:

- `external/duckdb/extension/tpch/`
- `external/duckdb/extension/tpcds/`
- `external/duckdb/third_party/tpce-tool/`

Those upstream benchmark generators have additional terms and are not covered by Vane's Apache-2.0 license.

## Linked build dependencies

Distributed exchange and HTTP filesystem support link native libraries resolved by the pinned `vcpkg.json` manifest. The release bundle `LICENSES/vcpkg-binary-dependencies.txt` is generated from vcpkg's installed copyright records with `scripts/sync_vcpkg_licenses.py`. It must be regenerated whenever the baseline, triplet, features, or dependency graph changes.

The generated bundle excludes `vcpkg-*` helper ports because they are build
machinery and are not linked into or shipped with Vane. Zstandard is used under
the permissive BSD option in its dual-license grant; its upstream record is
reproduced without alteration.

The direct native build dependencies are Apache Arrow (including Flight), cURL, gflags, glog, gRPC, OpenSSL, and their transitive dependencies. Their individual terms and notices are reproduced in the generated bundle.

## Runtime dependencies

Python runtime dependencies such as Ray, PyArrow, NumPy, and Cloudpickle are installed separately by package managers and are not copied into the Vane source distribution. Optional provider clients are also installed separately. Their own distributions govern their licenses.

## Release rule

Do not publish an sdist or wheel unless `scripts/check_release_artifacts.py` succeeds. A release reviewer must also inspect the exact source and binary contents because an automated inventory cannot determine license compatibility by itself.
