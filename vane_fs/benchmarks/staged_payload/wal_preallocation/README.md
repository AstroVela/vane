# One-time cold WAL reservation

Reserving 16 MiB of WAL for a new workspace is feasible through SQLite's VFS,
but this implementation fails the cold-start performance screen. Across 48
plain native runs, the optional `fdatasync` SDK improves complete 128-small-file
median throughput 27.3%. The first durable file nevertheless slows from
21.88 to 76.37 ms. First 64 MiB write throughput falls 15.2% with the default
SDK and 13.0% with the optional SDK, with **zero paired wins out of six** in
either group. Initialization is included in every rate.

Eight of twelve frozen checks fail. The patch remains an isolated experiment;
production format 2, its installed module, both existing SDKs and default
scheduling are unchanged. No ordinary FUSE throughput acceptance is claimed
after this unsuccessful native screen. Related FUSE correctness tests still
run as part of the component suite.

## Supported allocation path and implementation

The experiment starts from the exact staged format-1004 components in the
[preceding sync comparison](../sync_primitive/README.md), including its common
durable-image test correction. Controls retain their frozen source, libraries
and non-editable wheels. Candidate components change only `src/workspace.cpp`;
an exact reverse patch verifies this. Their Release core static libraries are
identical across the two SDKs. The existing checkpoint header/worker, staging,
payload code, schemas and thresholds remain byte-identical.

The public [SQLite file-control interface](https://sqlite.org/c3ref/file_control.html)
normally targets the main database. Its
[standard opcodes](https://sqlite.org/c3ref/c_fcntl_begin_atomic_write.html)
include `JOURNAL_POINTER`, which obtains the journal/WAL file handle. The
candidate sends `CHUNK_SIZE` and `SIZE_HINT` to that handle's VFS method while
holding `BEGIN IMMEDIATE`, then restores chunk size zero before releasing the
writer lock. It does not write SQLite frames or truncate committed bytes.

The [patch](reserve.patch) reserves at most the existing 16 MiB reuse target,
once for a newly initialized `fsync` workspace. Strict mode and existing
workspaces do not reserve. The initial owner/schema registration still commits
under FULL before the hint. A failed allocation therefore preserves an unlocked
durable owner row that `RecoverOwners()` can retire, matching existing worker
startup failure handling. Unsupported controls fall back to ordinary growth;
I/O errors remain visible and partial failures restore the chunk policy.

Both SDK feasibility probes expand WAL from 12,392 to 16,777,216 bytes, preserve
all existing WAL-prefix bytes, and reopen the acknowledged `base,ack` rows.
The pinned SQLite 3.53.2 Unix VFS uses its zero-write fallback, not
`posix_fallocate`: each feasibility hint performs 4,093 one-byte writes at
filesystem-block boundaries. SQLite's source selects that fallback when
`HAVE_POSIX_FALLOCATE` is absent/false. Source inspection, undefined symbols
and actual allocation-call counts agree. The probe, SDK hashes, commands and
results are retained in [feasibility.json](validation/feasibility.json).

This means a successful size hint has real preparation work and subsequently
dirty data to synchronize. A fixed file length cannot be treated as a free
warm cache. Allocation is done inside the measured workspace initialization;
its first barrier also remains inside the complete cost.

## Frozen protocol

Both SDKs independently compare unchanged versus reserved components in six
alternating AB/BA pairs for each workload: 48 plain native processes. The same
[C++ probe](probe.cpp) is linked with each core/SDK combination. Plain binaries
have no filesystem syscall recorder. A small wrapper observes the original
WAL-hook registration for status checks, forwarding its callback/context
unchanged. All workers remain active.

- `small128`: start before workspace construction, acquire/open the root, then
  create 128 files, write 4 KiB to each, perform its FULL workspace barrier,
  flush and close one reference. Lookup references remain until validation.
- `write64`: independently initialize another workspace, then create one file,
  write 64 MiB in 1 MiB requests, perform the final FULL barrier, flush and close
  one reference.

The report separates initialization, workload phase, initialization plus the
first durable file, complete initialization plus the workload, and workspace
close. Buffer creation is common and outside timing. Whole-file content and
SQLite integrity/format validation are outside timing. All plain samples,
including slower ones, remain in every median and aggregate.

The frozen primary gate requires at least 10% gain in both median and
total-work/total-time throughput, with at least five paired wins per group.
The first durable file and close medians may not increase more than 5%.
These are native screening gates, not replacements for ordinary FUSE
acceptance. No gate or timing boundary changes after measurement.

After the plain window ends, 16 separate diagnostic processes run two balanced
orders per SDK/workload. They link the same uninstrumented core with the frozen
recorder. Only their syscall/request measurements are used for mechanism
analysis; their rates do not replace the plain measurements. Their 1,032
explicit FULL barriers each produce an actual successful foreground WAL sync
with the expected SDK primitive. The 48 plain runs execute 3,096 explicit FULL
barrier calls. Native percentiles use nearest rank.

## Complete cold-start results

Times below are medians across all six independent processes, in milliseconds.
Throughput includes initialization, root/mount setup and the entire durable
workload. Small-file rates use files/s and large-file rates use MiB/s.

| SDK / workload | Baseline initialization | Reserved initialization | Baseline total | Reserved total | Median throughput change | Aggregate throughput change | Paired wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fsync / 128 small files | 21.24 | 39.57 | 457.70 | 495.51 | −7.6% | +1.1% | 2/6 |
| fdatasync / 128 small files | 18.71 | 38.42 | 426.58 | 335.27 | +27.3% | +30.8% | 6/6 |
| fsync / first 64 MiB file | 19.29 | 41.40 | 351.87 | 414.71 | −15.2% | −15.7% | 0/6 |
| fdatasync / first 64 MiB file | 19.21 | 37.75 | 356.33 | 409.79 | −13.0% | −13.5% | 0/6 |

Small-file median initialization-to-first-durable-file time rises from
24.71 to 78.00 ms with fsync (3.16 times), and from 21.88 to 76.37 ms with
fdatasync (3.49 times). The fdatasync batch gain therefore coexists with a
substantial startup regression. The fsync aggregate's small 1.1% gain is
affected by its slow 733 ms control sample; the median and four paired losses
do not establish a general benefit.

First large-file median rates fall from 182.17 to 154.51 MiB/s with fsync and
from 179.62 to 156.23 MiB/s with fdatasync. Counting only the workload phase
would still show regressions: its medians increase 326.08 to 360.65 ms and
334.36 to 372.17 ms. Initialization is not the only changed elapsed interval.
The fsync large-file close median also increases 3.65 to 7.98 ms, failing its
separate gate. No timing is discarded to improve the result.

Both arms have identical logical WAL progress: 39 frames before each phase,
3,120 after small files and 2,496 after the large file, with zero frames
backfilled. Both remain below the 16 MiB logical wake threshold and neither
performs background checkpoint maintenance. Baseline physical WAL grows from
160,712 to 12,854,432 bytes small, or to 10,283,552 bytes large. The reserved
file stays at 16,777,216 bytes. Physical allocation does not mark the worker
as warmed or relax admission.

## Diagnostic costs and retained stalls

Initialization has 92 recorded write calls in every diagnostic control and
4,154 in every candidate: **4,062 additional writes** from the hint. Normal
candidate initialization CPU is roughly 22–26 ms, compared with 6–8 ms in
controls; a slow candidate increases to about 45 ms. These are diagnostic CPU
costs with recorder overhead, not predictions of plain CPU savings.

The two normal fsync reserved small-file first WAL syncs take 36.73 and
36.89 ms, compared with 2.26 and 2.15 ms in controls. Preparing the larger
file adds work to the first durable operation. The optional SDK still has
long sync stalls despite fixed file length:

- Reserved fdatasync small-file repetition 0 lasts 5.411 s. Initialization is
  433 ms, with 389 ms inside its sync calls. Its 128 foreground WAL syncs
  sum to 4.843 s; the first alone is 1.127 s. Initialization plus the first
  durable file is 1.562 s.
- Reserved fdatasync repetition 1 lasts 881 ms. Initialization is 486 ms,
  including 441 ms of sync time; its first foreground WAL sync is 182 ms.

All these actual sync intervals and every plain operation latency are retained.
The observations do not identify SSD-internal behavior or prove that zeroing
caused every long call. A trace from this diagnostic window cannot assign a
cause to an individual plain sample.

For the first 64 MiB file, the diagnostic final barrier has roughly
23–25 ms of WAL synchronization and 142–184 ms of payload synchronization
across both SDKs/arms. Preallocation does not remove the payload barrier. The
large-file next investigation should focus on the payload allocation/sync path,
reviewing existing writeback experiments before changing it. A separate
supported `posix_fallocate` SDK comparison could isolate the zero-write fallback
for small-file initialization, but it is not evidence of a large-write gain.

The [complete findings](findings.json) retain all plain vectors, gate results,
every measured typed diagnostic sync and request wall/CPU/write summaries.
Full raw traces retain the additional recorder fields locally with hashes.

## Correctness, cleanup and reproducibility

Each candidate SDK passes 154 related Python tests with one explicit
production-v1 migration skip, 12 native suites and 12 ASAN/UBSAN core suites.
The unchanged controls' previously successful suites are reused by exact
source/SDK/binary identity. Existing FULL/error injection, close retry,
deterministic loss models, publication, snapshots, GC and crash cases remain.
No broad base/Ray suite is run.

Thirty-six new reservation cases cover bounded allocation, restored chunk
policy, strict/reopen behavior, unsupported chunk/hint fallback, visible full
and partial allocation errors, recoverable owners/retry, SIGKILL during the
hint, and the first acknowledged file after SIGKILL. They run under both SDKs
in Release and ASAN/UBSAN. Two original-core negative checks fail as intended.
The actual workload probe also passes all 24 plain/diagnostic/sanitizer
validation cases on tmpfs before ext4 measurement. These are software fault
models, not physical power-loss certification.

A retained [preflight](preflight/identity.json) corrects the probe's cleanup
order: `ReleaseMount()` removes inode references, so the root must close first.
It fails during validation before any ext4 comparison; the original source,
log and binary identity remain. Successful exact product validations are
reused after that harness-only fix. A separate derived-report compaction
removes redundant uninstrumented-core request fields; the audit verifies that
all rates, checks, tails and typed sync records are unchanged and keeps the
original report/code locally by hash.

Both measured windows retain a 20-second baseline and one-second read-only host
counters. Builds/tests finish beforehand, windows do not overlap, and no
global cache, affinity, frequency, device or scheduler setting changes. The
[audit](audit.json) verifies sources, binaries, both SDKs, all 64 runs and
**131 unique removed owned workspaces**. Processes stop and mounts are absent
before database/WAL/payload deletion. The production source/module and shared
SDK identities remain unchanged. Frozen reports, tests, all measurements,
logs and host counters remain; raw diagnostic traces, old analysis and
preflight binaries remain local with hashes.

Use new output paths and the exact frozen parents identified in
[build.json](validation/build.json). First compile/run `feasibility.cpp` with
each SDK using `-Wl,--wrap=pwrite64`, retaining the provenance shape in
`feasibility.json`, then:

```bash
python3 vane_fs/benchmarks/staged_payload/wal_preallocation/build.py "$RUN" \
  --parent vane_fs/build/sync-primitive-20261013
python3 vane_fs/benchmarks/staged_payload/wal_preallocation/run.py "$RUN"
python3 vane_fs/benchmarks/staged_payload/wal_preallocation/run.py "$RUN" --diagnostic
python3 vane_fs/benchmarks/staged_payload/wal_preallocation/analyze.py "$RUN"
```

Do not overlap builds, tests or heavy analysis with measurement. `--resume`
only reuses an archived failed validation, never a partial comparison window.
Run publishing separately after review so reruns cannot overwrite this report.
