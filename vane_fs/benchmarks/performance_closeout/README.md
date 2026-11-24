# Bounded performance closeout

The [roadmap](../../PERFORMANCE_ROADMAP.md) is complete. **One of three candidate
slots was used; the other two are closed unused.** C1 is rejected, C2 has no
independently supported follow-up mechanism, and C3 targets an optional path
absent from the shipping component. No further experiment starts under this
budget. Default VaneFS keeps its SQLite BLOB storage, existing format-1/2
compatibility, SDK and native implementation.

[inventory.json](inventory.json) records eight already adopted native changes
and 48 isolated staged-payload experiments. An audit status of PASS does not
mean a performance candidate passed or was adopted. The inventory keeps those
decisions separate. The unchanged default is qualified below; format 1005 is
only the control/candidate used in the bounded mechanism experiment.

## C1: earlier finite submission is rejected

C1 changes only `Prewriteback::START_BYTES` from 64 MiB to 16 MiB relative to
the frozen paced parent. Claims remain bounded to 4 MiB, with 16 ms after
completion; FULL pauses, publication ordering, real payload/WAL/fence barriers,
GC leases, checkpoint source and WAL admission are unchanged. The failed
BEGIN/RESTART gate is excluded from both arms. The same original fsync SDK and
22 performance checks are retained.

Six AB/BA pairs per fresh/post-GC state give 24 full mounts. Two pairs per
state with four consecutive write cycles add eight supplementary mounts.
All **32 mounts and 112 complete large-file hashes pass**, but **4/22
performance checks fail**. [C1-decision.json](C1-decision.json) retains the
complete failure details and the explicit rejection.

Complete 256+64 MiB phase rates, MiB/s:

| State | Parent median | C1 median | Parent aggregate | C1 aggregate |
| --- | ---: | ---: | ---: | ---: |
| Fresh | 254.87 | 282.59 | 164.56 | 279.56 |
| Post-GC | 255.14 | 277.09 | 255.30 | 171.01 |

The fresh aggregate includes a retained **4.147-second parent fsync**. The
post-GC candidate has a **4.694-second fsync**; its supplementary window also
has a **7.211-second fsync**. Higher common-case rates do not remove the tails.
Supplementary post-GC aggregate declines from 252.48 to 156.70 MiB/s.

| Failed primary check | Retained result |
| --- | --- |
| Fresh 4 KiB random write | Median throughput -6.02% |
| Post-GC 256 MiB write | Aggregate throughput -42.14% |
| Post-GC rename | Median throughput -11.86% |
| Post-GC combined large write | Aggregate throughput -33.02%, below the unchanged improvement target |

There is no additional parameter sweep or performance rerun. C2 remains
unused: this one threshold change does not establish a separate mechanism
that would justify another scheduling implementation. C1 is not adopted.

## C3: no shipping path to change

[checkpoint-scope.json](checkpoint-scope.json) checks the actual default source.
Its `BeforeWrite(sqlite3 *)` has no `PrepareExternal`, `BeginWrite` gate or
worker RESTART. Its single RESTART site enforces the existing 64 MiB admission
budget. The 906 ms CLEAN-prefix preparation and local gate from the preceding
diagnostic belong to the unadopted external-payload experiment.

Moving that experimental optimization or removing default backpressure would
require a different scope. C3 is not generated, built or measured. Default
checkpoint durability, bounded admission and background progress remain intact.

## Default implementation qualification

The installed default native module and existing default mount/CLI binaries
are used, independently of the format-1005 experiment. **Six C++ suites, an
explicit actual-WAL-fsync contract and 131 Python tests pass**. One optional
PyAV test skips and the existing inherited-fork warning remains.

The final matrix has **six full and four sustained mounts**, all successful,
with **44 full large-file hashes** plus read/overwrite, retained-GC-guard,
namespace, format-2 BLOB and SQLite quick-check validation. The repeated
sustained phases run in one workspace per mount. This characterizes the
shipping default; it is not a before/after performance comparison.

A first attempt completed one fresh workload and its two large-file hashes,
then its post-unmount validator incorrectly required an experimental
`.stage-fence`. Default BLOB storage does not have that file. The correction
requires its absence in both post-unmount checks; all timed function bodies,
native identities and workload settings remain unchanged. That first attempt
has no completed post-unmount database inspection. Its timings and error are
retained and **included** in the following all-observed fresh statistics.

| Workload | Fresh median | Fresh aggregate | Post-GC median | Post-GC aggregate |
| --- | ---: | ---: | ---: | ---: |
| 256 MiB write, MiB/s | 90.18 | 90.01 | 89.60 | 90.06 |
| 64 MiB write, MiB/s | 46.40 | 46.67 | 43.82 | 41.69 |
| Warm sequential read, MiB/s | 512.96 | 512.77 | 520.92 | 522.08 |
| 4 KiB random read, MiB/s | 67.65 | 67.59 | 71.75 | 71.97 |
| 4 KiB random write, MiB/s | 11.38 | 4.55 | 14.72 | 12.70 |
| Small file plus fsync, files/s | 265.49 | 87.12 | 262.50 | 261.45 |
| mkdir, ops/s | 1,906.50 | 1,889.05 | 2,008.05 | 1,943.11 |
| rename, ops/s | 2,304.28 | 2,298.68 | 2,271.62 | 2,247.13 |
| rmdir, ops/s | 2,265.29 | 2,223.55 | 2,344.12 | 1,940.74 |

Fresh uses four observed workloads: three fully qualified plus the retained
first attempt. Post-GC uses three fully qualified workloads. The first attempt
contains a random-write and small-file burst, which substantially lowers the
fresh aggregates; it is not discarded. [shipping-results.json](shipping-results.json)
also records the fixed, fully qualified 10-mount window and every sustained
cycle separately. [closeout.json](closeout.json) gives all-observed statistics,
the incomplete-validation count, decisions and cleanup.

The default remains a filesystem subset for local workspaces. It retains a
single SQLite writer, SQLite BLOB write amplification and possible slow
durable-sync/admission operations. This work does not establish a latency
bound, remote filesystem access, format-1005 migration, or physical power-cut
certification. Those are separate decisions in the existing component design.

## Related validation, provenance and cleanup

C1 passes its related Python, Release and ASan/UBSan component suites, extent
cases, scheduling/fault cases and all eight real FULL pause cases per build.
The unchanged 64 MiB parent fails the new 16 MiB wake/drain contract as
expected. The first FULL fixture failure is retained: 63 MiB now wakes both
the workspace worker and the separately attached publication worker. A probe
records their distinct TIDs/FDs. Populating one MiB below the configured trigger
isolates the attached worker and preserves every pause, error, retirement and
content assertion; all native identities remain unchanged.

Default test setup initially lacked pytest; an isolated test environment uses
the existing non-editable default package and verifies its resolved native
module hash. Default correctness tests are reused for the final format-check
correction by exact native identity and identical test-function AST. A later
post-window summary correction counts the cycle-suffixed large-file names.
The original measurement driver and manifest are retained byte-for-byte; its
frozen source reference is relocated to that snapshot. No measurement is
repeated for the summary change.

There are **42 qualified mounts / 156 large-file hashes**, plus the preserved
initial default mount / two hashes: **43 observed mounts and 158 hashes total**.
All **59 owned data directories** are removed after their processes stop,
outside application timers. Their file sizes and hashes remain in cleanup
manifests. Code, SDKs, isolated environments and incremental builds remain for
reproduction. The audit verifies 11 unchanged default identities and the
frozen experiment/measurement inputs. No owned mount or process remains.

Builds, tests, measured windows and analysis run separately. No cache eviction,
device/CPU tuning, load-based filtering or unowned process termination occurs.
Only related tests run. [Source checks](source-checks.json) cover formatting, Ruff, whitespace,
copyleft and changed-source SPDX; five unchanged historical root SPDX misses
remain separately recorded.

`audit.json` indexes deterministic gzip streams in parts below 500 KiB.
Concatenate each original entry's ordered `retained_as` files and decompress
once to recover identical bytes. Every native source snapshot, raw vector,
host sample, failed attempt, corrected fixture, command, log and identity is
retained. Local artifacts are `build/performance-closeout-C1-20261019`, both
`build/performance-closeout-shipping-20261019` and `-20261020`, and
`build/performance-closeout-final-20261020` under `vane_fs`.

The deliverable is the default implementation plus its qualified limits and
review material. The draft PR remains a draft. This closeout is finished;
unused candidate slots do not start another optimization cycle.
