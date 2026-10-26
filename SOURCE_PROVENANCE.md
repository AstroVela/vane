# Source provenance

Vane is an independent project. New contributions made directly to this repository are accepted under the Apache License, Version 2.0, unless a file or directory says otherwise.

The repository also contains substantial code derived from projects with compatible licenses. Those original license and attribution requirements remain in force.

| Area | Origin | License treatment |
| --- | --- | --- |
| Vane-specific code under `vane/` and distributed execution changes | Vane contributors | Apache-2.0 by default |
| `external/duckdb/` | Unmodified official `duckdb/duckdb` v2.0.0 integration baseline | DuckDB MIT license plus the licenses retained in its vendored directories |
| `external/duckdb-python/` | Unmodified official `duckdb/duckdb-python` v2.0.0 client baseline | DuckDB MIT license; Spark compatibility code retains its Apache-2.0 notices |
| DuckDB-derived modules and stubs moved under `vane/`, plus `src/vane_py/` | Derived from DuckDB's Python client and subsequently modified for Vane | Original DuckDB portions remain MIT; Vane contributions are Apache-2.0 |
| Tests and benchmarks derived from DuckDB or other named suites | Their named upstream source | License noted in the source directory or `THIRD_PARTY.md` |
| `vane_packaging/_vendor/auditwheel/manylinux-policy.json` | `pypa/auditwheel` 6.8.1 at commit `94e0693e0fcb444c7fe50f09a8a635e791be6174` | Unmodified MIT-licensed policy data; license retained in `LICENSES/auditwheel-LICENSE.txt` |

## File-level license markers

New Vane source files use an `Apache-2.0` SPDX identifier. Parent-repository files that combine inherited DuckDB or DuckDB Python client source with Vane modifications use `MIT AND Apache-2.0` and retain both copyright notices. New and modified source in `external/duckdb` remains under that repository's MIT license.

Existing third-party headers are preserved. Unchanged upstream source, vendored dependencies, and generated output are not mechanically relabeled. Run `python3 scripts/check_source_license_headers.py` from the repository root to validate the applicable files.

On the 2.0 integration branch, P2 must first adapt its engine baseline and
classification of the new client subtree; see
[current build status](DEVELOPMENT.md#duckdb-20-integration-baseline).

## DuckDB engine baselines

The DuckDB engine is imported under `external/duckdb` as a squashed Git subtree
from `https://github.com/duckdb/duckdb.git`. Subtree metadata records the exact
official upstream revision, while DuckDB's original history remains in its
upstream repository rather than becoming an ancestor of Vane's main branch.

The `integration/duckdb-2.0` baseline imports the official v2.0.0 tag at commit
`d54cbbe3df17e8e5ce68312f483b4155836112aa`. The complete `external/duckdb` tree
matches the official Git tree `c671df12df7e48e80306cb6a42d4761417c01695`,
including file modes and upstream license notices. No Vane engine changes are
included in this initial snapshot. The preceding Vane implementation remains
available at commit `95acdea830eb1fff407f0e09f4b97ee7b0a02e83`; its Python
bindings and other root-level integration code have not yet been migrated.
See [the migration roadmap](DUCKDB_2_MIGRATION_DESIGN.md#roadmap-and-acceptance-criteria)
and [the current build status](DEVELOPMENT.md#duckdb-20-integration-baseline).

The previous squashed subtree snapshot was imported from the official v1.5.0 tag at
commit `3a3967aa8190d0a2d1931d4ca4f5d920760030b4`. The reviewed maintenance
history through the official v1.5.5 tag at commit
`d8cdaa33fda8df955cc76ef58a280f68f4cd43fa` and Vane's customizations are
preserved in the history preceding the 2.0 import.

Those historical engine customizations include binding supplied SQL parameters
on query relations for distributed plan transport; those changes retain
DuckDB's MIT license. They are migration inputs, not part of the pure 2.0 tree.
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

## DuckDB Python client baseline

The official `duckdb/duckdb-python` repository is imported under
`external/duckdb-python` as a squashed Git subtree from
`https://github.com/duckdb/duckdb-python.git`. The v2.0.0 tag resolves to
`83c719961098611e769dc4bab0bda2bf149a2d1e`; its tree is
`b2e1819b1bd881106f5b3af31a62bc61ff8110d3`. The Vane import commit
`362db617be3ed09a9fcba9aaf23f37372a56f899` retains that exact tree, including
file modes, notices, and the engine gitlink. Squash snapshot
`5f5292f695286827847d95ac88d162402e21a76a` records the official revision in
`git-subtree-split`.

The client's `external/duckdb` entry is a mode-160000 gitlink at
`d54cbbe3df17e8e5ce68312f483b4155836112aa`, matching the engine baseline
above. Its `.gitmodules` is unchanged, including the upstream `branch = main`
setting; that setting does not change the recorded gitlink. The nested engine
is not checked out. A root `.gitmodules` entry maps its path and upstream URL
for Git metadata operations, with default updates and recursive fetches
disabled. This Vane-owned mapping does not modify either imported tree. Vane
will build from its existing `external/duckdb` tree. Client import does not
change that engine's SourceID or fork revision.

The client retains `external/duckdb-python/LICENSE` (DuckDB MIT, copyright
2018-2026) and `external/duckdb-python/duckdb/experimental/spark/LICENSE`
(Apache-2.0). The MIT license text matches `LICENSES/DuckDB-MIT.txt`.
No Vane headers or formatting changes are applied to this snapshot.

This is a source-only baseline. The old inherited bindings under `src/vane_py`
and `vane` remain as migration references; the root build does not yet use the
new client. P2 will adapt nanobind module names, public imports, engine selection,
and the upstream backend's repository/submodule version assumptions. The
[capability inventory](DUCKDB_2_MIGRATION_INVENTORY.md) records the layout and
remaining work. Source and release artifact validators must account for this
new subtree before any artifact is qualified.

## Vane engine identity contract

The following describes the existing Vane build identity contract. Its native
build hooks still need to be adapted and verified against the pure 2.0 engine;
the source import alone does not validate runtime or extension compatibility.

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

The imported DuckDB HTTPFS source configuration pins commit
`77392a72077a85897db2519fae11e2c0da972b8a` of
`https://github.com/duckdb/duckdb-httpfs` in
`external/duckdb/.github/config/extensions/httpfs.cmake`. It is covered by the
DuckDB MIT license recorded in `LICENSES/DuckDB-MIT.txt`. Building and packaging
that extension with Vane remains part of the subsequent integration stages.

The imported DuckDB tree contains upstream benchmark generators with additional terms, including TPC-H, TPC-DS, and TPC-E material. They are not part of Vane release artifacts. The sdist allowlist and artifact checker enforce this boundary.

When importing code:

1. Record the upstream repository, immutable revision, and source path in the pull request.
2. Confirm that its license is compatible with distribution in this repository.
3. Preserve required copyright, license, modification, and NOTICE text.
4. Update `THIRD_PARTY.md`, the relevant file headers, and the release artifact license bundle.

Do not add or mechanically replace license headers across inherited files without first identifying their provenance.
