# Source provenance

Vane is an independent project. New contributions made directly to this repository are accepted under the Apache License, Version 2.0, unless a file or directory says otherwise.

The repository also contains substantial code derived from projects with compatible licenses. Those original license and attribution requirements remain in force.

| Area | Origin | License treatment |
| --- | --- | --- |
| Vane-specific code under `vane/` and distributed execution changes | Vane contributors | Apache-2.0 by default |
| `external/duckdb/` | `duckdb/duckdb` plus Vane-maintained engine customizations | DuckDB MIT license plus the licenses retained in its vendored directories |
| DuckDB-derived modules and stubs moved under `vane/`, plus `src/vane_py/` | Derived from DuckDB's Python client and subsequently modified for Vane | Original DuckDB portions remain MIT; Vane contributions are Apache-2.0 |
| Tests and benchmarks derived from DuckDB or other named suites | Their named upstream source | License noted in the source directory or `THIRD_PARTY.md` |
| `vane_packaging/_vendor/auditwheel/manylinux-policy.json` | `pypa/auditwheel` 6.8.1 at commit `94e0693e0fcb444c7fe50f09a8a635e791be6174` | Unmodified MIT-licensed policy data; license retained in `LICENSES/auditwheel-LICENSE.txt` |

## File-level license markers

New Vane source files use an `Apache-2.0` SPDX identifier. Parent-repository files that combine inherited DuckDB or DuckDB Python client source with Vane modifications use `MIT AND Apache-2.0` and retain both copyright notices. New and modified source in `external/duckdb` remains under that repository's MIT license.

Existing third-party headers are preserved. Unchanged upstream source, vendored dependencies, and generated output are not mechanically relabeled. Run `python3 scripts/check_source_license_headers.py` from the repository root to validate the applicable files.

The DuckDB engine is imported under `external/duckdb` as a squashed Git subtree
from `https://github.com/duckdb/duckdb.git`. Subtree metadata records the exact
official upstream revision, while DuckDB's original history remains in its
upstream repository rather than becoming an ancestor of Vane's main branch.
The squashed subtree snapshot was imported from the official v1.5.0 tag at
commit `3a3967aa8190d0a2d1931d4ca4f5d920760030b4`. The reviewed maintenance
history through the official v1.5.5 tag at commit
`d8cdaa33fda8df955cc76ef58a280f68f4cd43fa` is preserved as subsequent
commits under `external/duckdb`.

Vane's engine customizations are retained as normal commits after that subtree
snapshot. They include binding supplied SQL parameters on query relations for
distributed plan transport and taking pending-query progress snapshots under
the client context lock so query cleanup cannot invalidate progress readers.
Those engine changes retain DuckDB's MIT license.
The former `AstroVela/duckdb` history maps to Vane as follows:

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
description is retained only as historical provenance and is no longer used as
the runtime version.

Vane records the reviewed official release line in
`DUCKDB_UPSTREAM_VERSION`. The build combines that value with the first ten
characters of the last Vane commit that changed `external/duckdb`, as resolved
by `git rev-list -1 HEAD -- external/duckdb`. DuckDB therefore reports a
user-facing version such as `v1.5.5-vane.594c360bbc`, without a manually
maintained counter or `g` prefix. Uncommitted engine changes append `-dirty`;
uncommitted changes outside the engine do not affect this version. Incremental
builds refresh a generated version header, so both dirty-state transitions and
new engine commits are reflected without manually editing build metadata.

The exact engine identity remains content-derived. The build computes the full
Git tree object for `external/duckdb` with
`scripts/sync_duckdb_source_id.py`. Git builds write the short identity only to
a generated header in the CMake binary directory. The external tree is a CMake
configuration dependency, so direct Ninja and Makefile builds refresh
configure-time metadata after timestamp-visible source changes. The version
object and default in-tree static extension entry points force-include the
applicable generated headers, keeping runtime identities consistent on the
first build even for mode-only changes that leave timestamps untouched. Vane
fork versions are treated as development identities for extension lookup, so
extension compatibility continues to use the SourceID rather than the
human-readable fork commit.

When Git metadata and a source-distribution manifest are absent, the SourceID
script derives the same Git-compatible object encoding from the materialized
tree. The constant `.git_archival.txt` export template records whether that
encoding uses SHA-1 or SHA-256 without introducing a generated identity file
that changes in normal commits. A commit identity cannot be reconstructed from
files alone, so the PEP 517 backend injects both the full `DUCKDB_SOURCE_ID` and
the full `DUCKDB_FORK_REVISION` into source distributions without modifying the
checkout. Subsequent builds without Git metadata require the injected fork
revision and retain both identities. The tree object depends on engine paths,
modes, symlinks, and contents rather than commit topology, while the fork
revision intentionally records Vane history; rebases and squash merges can
therefore change the displayed revision without changing the SourceID.

Ordinary engine changes require no tracked identity update. Changes to the
upstream baseline, DuckDB version line, or historical mapping must update this
document and `DUCKDB_UPSTREAM_VERSION`. Release reviews must record the full
fork revision and tree ID and inspect subsequent Vane engine commits since the
previously released state.

The statically linked DuckDB HTTPFS extension is fetched separately during the
native build and pinned to commit
`827222fb45a043a7a852d1f7aae46901492a3cda` by
`external/duckdb/.github/config/extensions/httpfs.cmake`. It is covered by the
DuckDB MIT license recorded in `LICENSES/DuckDB-MIT.txt`.

The imported DuckDB tree contains upstream benchmark generators with additional terms, including TPC-H, TPC-DS, and TPC-E material. They are not part of Vane release artifacts. The sdist allowlist and artifact checker enforce this boundary.

The optional Iceberg qualification build fetches
[`duckdb/duckdb-iceberg@45163a28e0ed6a2071a82a1bf1dd432d0216cf9c`](https://github.com/duckdb/duckdb-iceberg/tree/45163a28e0ed6a2071a82a1bf1dd432d0216cf9c)
and its independently built dependency
[`duckdb/duckdb-avro@f9d590297485f0318f480372c70bdd852826e258`](https://github.com/duckdb/duckdb-avro/tree/f9d590297485f0318f480372c70bdd852826e258).
These are the existing engine extension configuration pins. Their MIT license
files are copied unchanged to `LICENSES/Iceberg-MIT.txt` and
`LICENSES/Avro-extension-MIT.txt`; implementation source remains externally
fetched. Vane's compatibility diff is retained in
`packages/vane-iceberg-provider/patches/iceberg-vane-local.patch` and applies to
the fixed Iceberg revision; it does not require an upstream merge.
The subsequent Vane-owned
`packages/vane-iceberg-provider/patches/iceberg-vane-frozen-scan.patch` adds
versioned frozen-snapshot transport against the same revision. Its new source
files carry Apache-2.0 SPDX headers; the inherited Iceberg source remains MIT.
The third Vane-owned patch, `iceberg-vane-file-scan.patch` in the same directory,
adds per-file assignment planning and validation to that Apache-2.0 source.
It also exposes the standard empty projection in Iceberg's inherited virtual
column catalog entry so count-only scans do not request v3 row lineage. That
file retains the pinned DuckDB copyright with MIT AND Apache-2.0 modification
attribution. The patch changes neither the upstream revision nor the two
preceding patches.
The fourth patch, `iceberg-vane-partitions.patch`, adds version-2 owned identity
partition state to the Vane adapter. Its small inherited-reader changes expose
and reuse the pinned value decoder; those source/header files retain DuckDB
attribution and identify Vane's modifications as MIT AND Apache-2.0. No extension
or SDK revision changes, copied upstream implementation, or new dependencies
are introduced.
The fifth patch, `iceberg-vane-transforms.patch`, extends known partition-spec
admission in that Vane-owned Apache-2.0 adapter and freezes source constants
only for identity transforms. Its inherited-reader adjustment also restricts
the missing-column lookup to identity fields, preventing another transform on
the same source from hiding a valid identity constant. That file retains the
DuckDB and Vane attribution and MIT AND Apache-2.0 terms introduced by the fourth
patch. All four predecessor patches and the version-2 wire structure stay
unchanged, without new dependencies or additional license grants.
The sixth patch, `iceberg-vane-schema.patch`, extends the Vane-owned adapter to
freeze selected flat schemas across evolution and validate typed NULL defaults.
Its inherited manifest decoder retains legacy partition types from schema
history after a source is dropped; inherited file/manifest filtering ignores
sources absent from the selected schema. Both files retain pinned DuckDB
attribution and identify Vane modifications as MIT AND Apache-2.0. The five
predecessor patches, SDK and extension pins, dependency roles and wire version
are unchanged.
The seventh patch, `iceberg-vane-deletes.patch`, freezes committed v2 Parquet
delete-file descriptors and applicability state, with frozen bind/assignment/file
codec version 3 and unchanged generic callback protocol 1. Its inherited reader,
optimizer, file-list and positional/equality-delete changes preserve pinned
DuckDB attribution and MIT AND Apache-2.0 terms. The delete scanner and the
frozen reader's equality-column preservation follow the pinned upstream reader
and optimizer contracts; those implementation files carry the same combined
terms and attribution. The new portable-state header remains Vane-owned
Apache-2.0. All six predecessor patches and source/SDK pins are unchanged.
The self-contained loadable provider also links the pinned in-tree Parquet
implementation, under DuckDB's existing source and dependency notices.
The separate SDK manifest pins
`duckdb/vcpkg-duckdb-ports@f13de79b66345fe31d3f9c119517613023d194a8`, whose
Avro C port builds `duckdb/duckdb-avro-c@8af400279c445a81b8552a7670d8c1ebd92ba34a`.
`LICENSES/iceberg-native-dependencies.txt` records the resulting native
dependency notices. Qualification artifacts use the existing test signing and
provider descriptor workflow; production publication remains separate.

When importing code:

1. Record the upstream repository, immutable revision, and source path in the pull request.
2. Confirm that its license is compatible with distribution in this repository.
3. Preserve required copyright, license, modification, and NOTICE text.
4. Update `THIRD_PARTY.md`, the relevant file headers, and the release artifact license bundle.

Do not add or mechanically replace license headers across inherited files without first identifying their provenance.
