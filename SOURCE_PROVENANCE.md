# Source provenance

Vane is an independent project. New contributions made directly to this repository are accepted under the Apache License, Version 2.0, unless a file or directory says otherwise.

The repository also contains substantial code derived from projects with compatible licenses. Those original license and attribution requirements remain in force.

| Area | Origin | License treatment |
| --- | --- | --- |
| `vane/` and Vane-specific distributed execution changes | Vane contributors | Apache-2.0 by default |
| `external/duckdb/` | Fork of `duckdb/duckdb` | DuckDB MIT license plus the licenses retained in its vendored directories |
| `duckdb/`, `src/duckdb_py/`, `_duckdb-stubs/`, and `adbc_driver_duckdb/` | Derived from DuckDB's Python client and subsequently modified for Vane | Original DuckDB portions remain MIT; Vane contributions are Apache-2.0 |
| Tests and benchmarks derived from DuckDB or other named suites | Their named upstream source | License noted in the source directory or `THIRD_PARTY.md` |

## File-level license markers

New Vane source files use an `Apache-2.0` SPDX identifier. Parent-repository files that combine inherited DuckDB or DuckDB Python client source with Vane modifications use `MIT AND Apache-2.0` and retain both copyright notices. New and modified source in `external/duckdb` remains under that repository's MIT license.

Existing third-party headers are preserved. Unchanged upstream source, vendored dependencies, and generated output are not mechanically relabeled. Run `python3 scripts/check_source_license_headers.py` from the repository root to validate the applicable files.

The DuckDB engine fork is imported under `external/duckdb` as a non-squashed
Git subtree from `https://github.com/AstroVela/duckdb.git`. The original engine
history and commit IDs remain reachable from the subtree merge's second parent.
The current imported engine source baseline is commit
`398033a962719ac09868f4484ec4f97353bb0325`, described as
`v1.5.0-1-g398033a962`. Source archives do not contain Git metadata, so the
same description is passed through `OVERRIDE_GIT_DESCRIBE` in
`pyproject.toml`. A change to the engine source must update both records in the
same pull request. Release reviews must also record the exact imported DuckDB
commit and inspect changes since the previously released baseline.

The statically linked DuckDB HTTPFS extension is fetched separately during the
native build and pinned to commit
`74f954001f3a740c909181b02259de6c7b942632` by
`external/duckdb/.github/config/extensions/httpfs.cmake`. It is covered by the
DuckDB MIT license recorded in `LICENSES/DuckDB-MIT.txt`.

The DuckDB fork contains upstream benchmark generators with additional terms, including TPC-H, TPC-DS, and TPC-E material. They are not part of Vane release artifacts. The sdist allowlist and artifact checker enforce this boundary.

When importing code:

1. Record the upstream repository, immutable revision, and source path in the pull request.
2. Confirm that its license is compatible with distribution in this repository.
3. Preserve required copyright, license, modification, and NOTICE text.
4. Update `THIRD_PARTY.md`, the relevant file headers, and the release artifact license bundle.

Do not add or mechanically replace license headers across inherited files without first identifying their provenance.
