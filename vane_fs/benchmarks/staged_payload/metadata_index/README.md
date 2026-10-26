# Block high-index removal

This isolated C++ candidate removes `block_versions_high` over preceding
no-hint orphan batching. **It is not adopted:** ten original non-regression
checks fail, and five reference checks fail in the independent comparison.
The production 4 KiB SQLite BLOB component and installed module are unchanged.

## Evidence and implementation

A native tmpfs probe writes 64 MiB in 8 MiB API requests at file offset 48.
It measures metadata costs, not the FUSE acceptance workload's application
buffer alignment. Six alternating pairs reduce WAL length from 5,582,632 to
4,000,552 bytes (28.3%) and VM instructions from 1,484,809 to 1,386,158 (6.6%).
Median complete write/barrier time is 111.52→98.17 ms; wall time is diagnostic.
The initial empty-history index occupies 342 pages. Tested point/range/join
plans use the covering primary key, including point queries with 10,000
historical versions before and after `ANALYZE`. Historical VM counts remain
27, 20,023 and 40,022 for oldest/middle/newest points in both configurations.
Removing the index does not fix the existing history scan cost.

Only `src/workspace.cpp` changes: skip that secondary index on fresh creation
and drop it on existing format-1004 workspaces inside the existing FULL startup
transaction. A forced later startup failure rolls the drop back. Primary keys,
foreign keys, interval checks, queries and version records are unchanged.
`staging.hpp`, `checkpoint.hpp` and `payload_file.hpp` are byte-identical to
orphan batching: publication sync, FULL commit, CLEAN fence, recovery, GC,
16/64 MiB WAL thresholds and prompt CLEAN progress retain their former behavior.
There is no payload budget, presync, hint, or new format.

## Corrected original adoption window

The original no-hint partial-edge control and all workload/gate constants
remain: 48 uninstrumented mounts, six alternating pairs per medium/state,
page-offset-48 application buffers, complete create/write/fsync/close timing,
and unchanged validation. Each variant now uses its own native GC Python module.

| Medium/state | Control aggregate MiB/s | Candidate | Change | Wins |
| --- | ---: | ---: | ---: | ---: |
| tmpfs fresh | 197.52 | 347.75 | +76.1% | 6/6 |
| tmpfs post-GC | 191.29 | 346.70 | +81.2% | 6/6 |
| ext4 fresh | 111.78 | 155.43 | +39.1% | 6/6 |
| ext4 post-GC | 84.96 | 147.27 | +73.4% | 6/6 |

Ext4 first-256 median rates improve 155.17→176.57 MiB/s fresh and
153.70→174.82 post-GC. Subsequent-64 medians improve 53.95→105.53 and
50.83→95.19, while pooled write p99 increases 63.15→149.64 ms and
66.98→151.33 ms. First-256 p99 increases 361.82→424.22 and 361.67→422.05 ms.
Used metadata pages after the complete workload decrease 8,095→6,027 fresh
and 8,561→6,322 post-GC; page count alone also includes free pages.

Ten checks fail: random writes on both tmpfs states; post-GC tmpfs small-file
fsync and rmdir; fresh ext4 subsequent-write p99, random writes, small-file
fsync and rename; post-GC ext4 subsequent-write p99 and random writes.
Fresh candidate fsync phases of 5.107 and 5.297 seconds remain, as does a
6.314-second post-GC control subsequent-write phase. That control tail raises
the post-GC aggregate improvement; it is not removed or attributed to the index.

## Independent comparison with preceding orphan batching

Another 24 uninstrumented ext4 mounts isolate the index change, using each
variant's own Python module for GC. Reference gates do not replace adoption.

| State | Combined median MiB/s, control→candidate | Aggregate MiB/s | Wins |
| --- | ---: | ---: | ---: |
| fresh | 137.05→158.47 (+15.6%) | 102.46→160.74 (+56.9%) | 6/6 |
| post-GC | 134.90→149.27 (+10.7%) | 136.02→151.09 (+11.1%) | 6/6 |

First-256 medians improve 152.68→182.82 MiB/s fresh and 155.01→173.97 post-GC.
Subsequent-64 medians are 97.32→108.07 and 91.20→96.42. The fresh control
retains a 5.459-second subsequent-write phase, inflating the aggregate gain.
Five reference checks fail: fresh read, small-file fsync and rmdir, plus post-GC
random write and small-file fsync. Fresh read median decreases 928.85→790.27
MiB/s (14.9%); post-GC random write decreases 12.30→11.43 (7.1%).

## GC module binding correction and retained first window

The initial builder aliased the controls' Python environments to the candidate.
The FUSE daemons and fresh CLI initialization used the right binaries, but the
GC helper opened each control database with the candidate module and dropped its
index. Equal post-GC metadata page counts exposed the problem. A deterministic
reproduction initializes a control with the index: shared-module GC removes it;
the corrected control module preserves it.

All initial 72 mounts, including their slow samples, remain under
`shared-gc-preflight/` and are explicitly excluded from adoption. They are not
replaced because of machine load. The correction reuses exactly the same
tested native binaries, binds three independent installed modules, freezes a
new plan, and runs another 72 mounts. Module path/hash checks run before the
window; index-state assertions run during existing inspection outside workload
timers, including immediately after GC. The measured workload bytes and all
gate constants remain unchanged. The old recipes are frozen and hashed.

## Validation, limits and retention

Related tests pass: 154 installed Python tests, 11 original-control and 11
candidate C++ tests, 11 ASAN/UBSAN C++ tests, 44 positive helper cases and one
expected old-control failure. New cases cover upgrade/retained snapshots and
GC, startup rollback and 128 snapshot layers; latest-read work is identical
(1,100 SQLite steps, 142,400 VM instructions). Five binding checks verify the
three modules and reproduce/fix the GC alias error. One existing production-v1
migration test is skipped for experimental format 1004. No broad base/Ray suite
runs. Native binaries and related test results are reused only after identity
verification; changing these Python benchmark bindings requires no rebuild.

All 144 mounts validate content, retained guards and SQLite integrity, stop,
unmount and remove their data. `audit.json` records 220 removed paths, including
test/probe/binding data. The unavailable `dbstat` probe preflight and its logs
are retained; its original source and binary are reconstructed with exact
hashes. No builds, tests or formatters overlap any measured window. There is
no global cache eviction or CPU/device setting change. JSON vectors are retained
losslessly below 500 KiB; logs, host measurements, binaries and frozen sources
remain locally with identities in `manifest.json`.

The index reduces metadata work, but these plain windows cannot identify the
syscall causing each tail. WAL policy is unchanged; a byte threshold on WAL
metadata can allow larger unpublished payload batches after metadata shrinks.
Do not infer checkpoint or payload sync latency from WAL file length alone.
Next test whether a non-blocking WAL restart after completed checkpoint work
reduces repeated checkpoints for small commits, while preserving pinned-reader
progress and every existing publication barrier. That policy is not implemented
or validated in this experiment.

## Reproduce

`prepare.py` builds a non-editable Release candidate with the original control
as `--parent` and preceding orphan batching as `--orphan-parent`; use the frozen
SQLite 3.53.2/FUSE SDKs and test-support packages. It binds each control to its
own installed module. Run `validate.py`, then `run.py` for tmpfs and ext4 and
the original `partial_sync/analyze_writeback.py`. Run `run_causal.py` and the
same analyzer on `causal/`, followed by `summarize.py`. `runner.py` preserves
the original workload and adds only module/schema assertions outside timers.

For the recorded correction, `rebind.py` verifies the first build, frozen
recipes, native binaries and related tests before making corrected aliases.
`publish.py` takes the corrected artifact, the native `--probe`, both probe
`--preflight` directories and `--shared-gc-preflight`; it reconstructs the
35 source inputs and both gate summaries and retains all original windows.
