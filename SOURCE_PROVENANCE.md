# Source provenance

Vane is an independent project. New contributions made directly to this repository are accepted under the Apache License, Version 2.0, unless a file or directory says otherwise.

The repository also contains substantial code derived from projects with compatible licenses. Those original license and attribution requirements remain in force.

| Area | Origin | License treatment |
| --- | --- | --- |
| `vane/` and Vane-specific distributed execution changes | Vane contributors | Apache-2.0 by default |
| `external/duckdb/` | `duckdb/duckdb` plus Vane-maintained engine customizations | DuckDB MIT license plus the licenses retained in its vendored directories |
| `duckdb/`, `src/duckdb_py/`, `_duckdb-stubs/`, and `adbc_driver_duckdb/` | Derived from DuckDB's Python client and subsequently modified for Vane | Original DuckDB portions remain MIT; Vane contributions are Apache-2.0 |
| Tests and benchmarks derived from DuckDB or other named suites | Their named upstream source | License noted in the source directory or `THIRD_PARTY.md` |

## File-level license markers

New Vane source files use an `Apache-2.0` SPDX identifier. Parent-repository files that combine inherited DuckDB or DuckDB Python client source with Vane modifications use `MIT AND Apache-2.0` and retain both copyright notices. New and modified source in `external/duckdb` remains under that repository's MIT license.

Existing third-party headers are preserved. Unchanged upstream source, vendored dependencies, and generated output are not mechanically relabeled. Run `python3 scripts/check_source_license_headers.py` from the repository root to validate the applicable files.

The DuckDB engine is imported under `external/duckdb` as a squashed Git subtree
from `https://github.com/duckdb/duckdb.git`. Subtree metadata records the exact
official upstream revision, while DuckDB's original history remains in its
upstream repository rather than becoming an ancestor of Vane's main branch.
The current official upstream baseline is commit
`3a3967aa8190d0a2d1931d4ca4f5d920760030b4`.

Vane's engine customizations are retained as normal commits after that subtree
snapshot. The former `AstroVela/duckdb` history maps to Vane as follows:

| Former fork commit | Parent | Corresponding Vane commit |
| --- | --- | --- |
| `398033a962719ac09868f4484ec4f97353bb0325` | `3a3967aa8190d0a2d1931d4ca4f5d920760030b4` | `57d4e3c166e307c19b28cb1bb2ea7ebd2283a030` |
| `e2d398989076fa3c6c3859e77310e5e50608b168` | `398033a962719ac09868f4484ec4f97353bb0325` | `74f8d91976c69d8861262944eb61f4b8a05abd42` |

The former fork is not used for builds or engine identity. Vane intentionally
does not retain a bundle or other archive of its Git objects. The identifiers
above are kept only as a historical mapping; maintained customization history
is the corresponding Vane commits, and the official baseline history remains
in `duckdb/duckdb`. The former fork may therefore be deleted without an
archival prerequisite.

At Vane commit `74f8d91976c69d8861262944eb61f4b8a05abd42`, the
DuckDB-rooted subtree history is described as `v1.5.0-2-g55abe0cb9e`. That
description is passed through `OVERRIDE_GIT_DESCRIBE` in `pyproject.toml` to
preserve DuckDB's human-readable version line in source archives without Git
metadata.

The exact engine identity is content-derived instead. `DUCKDB_SOURCE_ID`
records the full Git tree object for `external/duckdb`, computed with
`scripts/sync_duckdb_source_id.py`. The top-level CMake build passes that value
through its first 10 hexadecimal characters as `GIT_COMMIT_HASH` and DuckDB's
embedded `SourceID`. The tree object depends on engine paths, modes, and
contents rather than commit topology, so rebases and squash merges do not
change the SourceID.
Ordinary engine changes update only `DUCKDB_SOURCE_ID`; changes to the upstream
baseline, DuckDB version line, or historical mapping must also update this
document and `OVERRIDE_GIT_DESCRIBE`. Release reviews must record the full tree
ID and inspect subsequent Vane engine commits since the previously released
state.

The statically linked DuckDB HTTPFS extension is fetched separately during the
native build and pinned to commit
`74f954001f3a740c909181b02259de6c7b942632` by
`external/duckdb/.github/config/extensions/httpfs.cmake`. It is covered by the
DuckDB MIT license recorded in `LICENSES/DuckDB-MIT.txt`.

The imported DuckDB tree contains upstream benchmark generators with additional terms, including TPC-H, TPC-DS, and TPC-E material. They are not part of Vane release artifacts. The sdist allowlist and artifact checker enforce this boundary.

When importing code:

1. Record the upstream repository, immutable revision, and source path in the pull request.
2. Confirm that its license is compatible with distribution in this repository.
3. Preserve required copyright, license, modification, and NOTICE text.
4. Update `THIRD_PARTY.md`, the relevant file headers, and the release artifact license bundle.

Do not add or mechanically replace license headers across inherited files without first identifying their provenance.
