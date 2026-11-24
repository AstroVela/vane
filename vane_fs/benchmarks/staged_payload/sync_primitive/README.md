# SQLite sync primitive on the current staged prototype

SQLite `fdatasync` improves the common small-file synchronization cost on reused
WAL, but this comparison does not justify adopting it as the default. Ordinary
FUSE small-file median throughput improves 60.9% on fresh workspaces and 51.4%
after fragmentation and GC. The fresh 64 MiB write median falls 5.6%, warmed
read medians fall 6.2%/5.4%, and post-GC random-write total throughput falls
51.2%. Seven unchanged acceptance checks fail. All slow samples remain.

The production format-2 BLOB implementation, installed default module, shared
SQLite SDK and scheduling remain unchanged. The existing Linux
[`x64-linux-fdatasync-release` option](../../../README.md#optional-linux-sqlite-fdatasync-build)
remains opt-in. This evaluates the isolated format-1004
[pre-OPEN preparation prototype](../pre_open_backfill/README.md), not production
format adoption or a comparison with a different filesystem.

## Matched builds and durability

Both non-editable Release components use byte-identical C++ runtime sources.
Their core static libraries are identical to each other and to the frozen
prototype library: SHA-256
`110c831851b4d7c04ba18f64c4d4b684480ce9075784e9f3aaea038860f791f3`.
The original five-millisecond worker, pre-OPEN preparation, 16 MiB progress
threshold, 64 MiB admission limit, payload publication and FULL barriers are
unchanged. The worker remains active in both measured variants.

The only runtime difference is the SQLite SDK: the existing default triplet
versus the existing optional `HAVE_FDATASYNC=1` triplet. Both use SQLite 3.53.2,
the same headers, SourceID and reported compile options. The control library
matches the default SDK byte for byte. Link symbols and actual WAL calls
confirm the different primitives; the macro is absent from SQLite's reported
compile options. The SDK provenance, build commands, binaries and source hashes
are retained in [build.json](validation/build.json).

[SQLite's build documentation](https://www.sqlite.org/compile.html#have_fdatasync)
describes this flag as selecting the Unix VFS sync primitive. The application
still commits ordinary mutations under NORMAL in `fsync` durability mode and
acknowledges explicit barriers under FULL. Large external payloads already use
`fdatasync` in both components; changing SQLite's SDK does not change that path.

One test-only correction is applied identically to both copied components:
the existing [loss-fixture patch](../wal_sync/durability-fixture.patch) limits
the generic syscall image hook to payload/fence sidecars. The VFS hook captures
database/WAL images. This does not change runtime sources or installed wheels.

## Native mechanism comparison

The unchanged [C++ probe](../full_sync_contention/probe.cpp) is linked against
the same frozen instrumented core and recorder with each SQLite SDK. Each
process creates 128 files, writes 4 KiB, performs one FULL workspace barrier,
flushes and closes one reference. Lookup references are retained until content
validation. Preparation, validation and workspace close are outside the phase.
This is a native-core diagnosis without mounted FUSE.

Three states each run six alternating AB/BA pairs, for 36 independent processes:

- Cold: a new workspace below the 16 MiB progress threshold, with no background
  database sync during the phase.
- Warm: a synthetic 20 MiB committed CLEAN SQL padding table is fully
  backfilled before timing, followed by WAL reuse. This setup is not a claim
  that large format-1004 file contents are stored in SQLite.
- GC: retain distinct live/snapshot content, discard a child branch containing
  eight 1 MiB files, collect garbage, then complete the same warm setup. Every
  case reclaims 2,065 versions, 2,048 payloads and one snapshot.

Both SDK variants drain preparation and start the original worker before the
phase. They register the original commit callback/context unchanged. Every
phase stays below 64 MiB admission and contains exactly 128 successful actual
foreground WAL syncs using its expected primitive: **4,608 FULL barriers** in
total. Native latency percentiles use nearest rank.

| Native state | fsync files/s | fdatasync files/s | Median change | Total-work/total-time change | Paired wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold | 297.39 | 281.08 | −5.5% | −2.1% | 2/6 |
| Warm reused | 295.34 | 439.37 | +48.8% | +51.5% | 6/6 |
| GC reused | 285.12 | 472.72 | +65.8% | +62.8% | 6/6 |

The median sum of 128 foreground WAL sync durations changes from
293.76 to 154.60 ms warm and from 298.80 to 147.59 ms after GC. Cold changes
from 288.64 to 317.06 ms. Worker database syncs remain in both warm/GC variants.
No measured native sync exceeds 100 ms in this window. Inclusive request and
worker intervals overlap and must not be added as wall time.

There is a concrete state difference: in every cold run the 128 foreground
syncs observe 128 distinct physical WAL lengths, growing from 247,232 to
12,854,432 bytes. Every warm/GC run observes a fixed physical length of
16,777,216 bytes. These records are included in
[findings.json](native/findings.json) and [report.json](validation/report.json).

The pattern is consistent with lower metadata synchronization cost when WAL
length stays fixed; it does not isolate file length from all preparation/cache
differences. Linux documents that `fdatasync` can omit unnecessary timestamp
metadata but must synchronize a changed file size.
See [fsync(2)](https://man7.org/linux/man-pages/man2/fsync.2.html). ext4 also selects
different journal transaction identifiers for the two primitives in its
[Linux 6.11 implementation](https://github.com/torvalds/linux/blob/v6.11/fs/ext4/fsync.c).
Neither fact identifies the internal cause of each device stall.

## Ordinary FUSE comparison

After native measurement stops, 24 plain, uninstrumented mounts replay the
frozen ext4 workload: six alternating pairs each on fresh and independently
fragmented/GC workspaces. Each daemon's own isolated wheel supplies GC; binding
paths and module hashes are verified before the run. Only two stale report
labels in the frozen runner change, with exact reverse verification. Workload,
timing boundaries, preparation, validation and gate constants stay unchanged.

The first write is 256 MiB with 8 MiB application requests, then 64 MiB with
1 MiB requests; both include final file fsync and close. Application buffers
retain a 48-byte page offset. Reads use a fully validated and warmed 64 MiB
fixture. Random phases retain 4,096 seeded reads and 512 distinct 4 KiB
overwrites with final fsync. Small files each create/write/fsync/close 4 KiB;
metadata phases include a final directory fsync. Full hashes, read slices,
retained GC content and SQLite integrity/format are checked outside timers.

Rates below are medians of all six mounts. Read/write units are MiB/s;
small files use files/s and metadata uses operations/s. Baseline means the
`fsync` SQLite SDK; updated means `fdatasync`, independent of the application's
`fsync` durability mode.

| Workload | Fresh fsync | Fresh fdatasync | GC fsync | GC fdatasync |
| --- | ---: | ---: | ---: | ---: |
| First 256 MiB write + fsync | 149.45 | 155.01 | 154.04 | 153.21 |
| Following 64 MiB write + fsync | 108.79 | 102.72 | 100.39 | 102.60 |
| Warm 64 MiB read | 814.84 | 764.73 | 793.67 | 751.00 |
| 4 KiB random read | 74.48 | 72.78 | 71.97 | 70.41 |
| 4 KiB overwrite + final fsync | 12.70 | 13.04 | 10.37 | 9.25 |
| Small file + fsync | 212.13 | 341.27 | 242.22 | 366.69 |
| mkdir + final directory fsync | 2,107.09 | 2,323.82 | 1,990.30 | 2,183.04 |
| rename + final directory fsync | 1,777.09 | 2,303.22 | 1,948.99 | 2,115.53 |
| rmdir + final directory fsync | 2,223.97 | 2,600.09 | 2,302.33 | 2,525.79 |

Small-file paired wins are 6/6 fresh and 5/6 GC. Total-work/total-time rates
increase from 207.63 to 334.00 files/s fresh and from 81.18 to 126.72 after GC.
That improvement coexists with retained slow GC phases of 6.866 s in the
control and 4.377 s in the candidate. Candidate GC random-write throughput
falls from 10.55 to 5.15 MiB/s in total-time aggregation, with a 1.164-second
final fsync in repetition 3. Both improvements and regressions remain.

The fresh combined 320 MiB aggregate ratio is 1.432, driven by the control's
6.037-second individual write in a 6.562-second 64 MiB phase. Only 2/6 pairs
improve, failing the unchanged 5/6 requirement. After GC, the ratio is 1.009
with 5/6 wins, below the 1.10 aggregate target. Median/aggregate checks also
reject the fresh 64 MiB write, fresh/GC warmed read, and GC random read/write.
No gate is relaxed. These plain tails cannot be assigned causes using traces
from a different native process/window.

The complete vectors, existing percentile convention, all latency arrays,
final barriers and lifecycle times are in [results.json](plain/results.json)
and [summary.json](plain/summary.json). [report.json](validation/report.json)
enumerates every application call of at least one second and every phase of
at least four seconds. Both groups' slow phases are retained.

## Validation and retained evidence

Each variant passes 154 related Python tests with one explicit production-v1
migration skip, 12 native C++ suites and 12 ASAN/UBSAN core suites. These include
cross-connection visibility, FULL/error propagation, six WAL syscall failure
injections, close retries, deterministic durable images, fork/crash recovery,
external publication and GC. The corrected loss fixture passes in both Release
and sanitizer builds. The control's unchanged successful suites are reused
after that test-only correction; its corrected fixture is rebuilt and rerun.
Six Release and six ASAN/UBSAN probe cases separately reconcile the primitive
and 128 FULL barriers, all file contents, live/snapshot content and integrity.
Both deliberately opposite syscall expectations fail as intended. Only related
tests and source checks run. These are software fault/recovery models, not
physical power-loss certification.

Two resolved preflights precede measurement. The first harness counted worker
and WAL-reset syncs as foreground barriers; the trace showed exactly 128
explicit barriers plus those other calls. The filter is corrected without
changing storage behavior. The second reproduces the old durable-image fixture
capturing a deleted 512-byte rollback journal; the live journal is absent but
the artificial image retains it and rolls initialization back. The existing
test-only patch corrects both variants. Failure logs, configurations, code and
binary identities remain in [preflight](preflight/identity.json) and
[preflight-loss-model](preflight-loss-model/identity.json); neither is a measured
comparison. Already successful exact commands are reused rather than rerunning
unaffected suites after each harness correction.

Linux `6.11.0-24-generic`, 36 logical CPUs and ext4 on `/dev/sda2` remain in use.
Whole-host busy CPU averages 4.68% during the native window and 5.95% during
plain FUSE; I/O wait averages 0.68% and 0.93%. These host counters cannot rule
out device interference. The read-only host monitors include a 20-second
baseline and no owned builds/tests overlap either window. No cache eviction,
affinity, frequency, device or scheduler setting changes, and no sample is
filtered by load. Native and plain windows do not overlap.

The [audit](audit.json) verifies all inputs/binaries, 36 native runs, 24 plain
mounts and **86 unique cleaned owned directories**, including preflight/test
data. Every run process exits and every mount is absent before DB/WAL/payload
deletion. Approximately 87 GiB remains free. Full diagnostic traces, CPU
samples and preflight binaries remain local with hashes; phase/operation
measurements, every measured typed native sync, all plain latency arrays,
host counters, configurations, source identities and validation logs are
retained here. Default source/module identity is checked after the comparison.

The next bounded experiment should isolate cold WAL growth and evaluate
supported preallocation only if it preserves recovery and retry semantics.
Count initialization plus the first durable write, rather than hiding setup
cost. The useful warm small-file result does not by itself justify moving
checkpoint work, removing barriers or adopting the current prototype.

## Reproduce

Keep the exact frozen parent implementation, instrumented core/recorder and
SDKs identified by the build manifest. Use a new output directory. The two SDK
libraries must match the provenance hashes; source and syscall expectations
are verified before measurement. `--resume` is only for an archived failed
preflight, never a partially completed measurement window.

```bash
python3 vane_fs/benchmarks/staged_payload/sync_primitive/build.py "$RUN" \
  --parent vane_fs/build/pre-open-backfill-20261013 \
  --sdk-build vane_fs/build/fdatasync-option-20261010T030643Z/sdk-build.json
python3 vane_fs/benchmarks/staged_payload/sync_primitive/run.py "$RUN"
python3 vane_fs/benchmarks/staged_payload/sync_primitive/analyze.py "$RUN"
python3 vane_fs/benchmarks/staged_payload/sync_primitive/fuse.py "$RUN/plain" \
  --template vane_fs/build/checkpoint-progress-20261012T153141Z/fuse-screen \
  --validation "$RUN/tests.json" --baseline "$RUN/fsync" --updated "$RUN/fdatasync"
python3 vane_fs/benchmarks/staged_payload/sync_primitive/report.py "$RUN"
```

Do not run builds, tests or other heavy analysis concurrently with either
comparison window. Run publishing separately after review so repeated runs
cannot overwrite the retained report.
