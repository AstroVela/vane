# SQLite synchronization on the production BLOB format

This evaluates SQLite's `HAVE_FDATASYNC` build option against the unmodified
production VaneFS component at `f9fdb2149e240b0e95ef2253c8546a0d7da7ae2d`.
All workspaces keep format version 2 and 4 KiB SQLite BLOB payloads. No external
payload format or migration is involved.

In fsync mode, the matched candidate improves median small-file throughput from
261.42 to 361.00 files/s on fresh workspaces and from 262.76 to 381.18 files/s
after fragmentation and GC. The aged overwrite workload, including its final
fsync, falls from 11.39 to 10.73 MiB/s. Long sync stalls remain in both builds.
In strict mode, a slow candidate small-file run also makes throughput across
the total elapsed time worse despite its higher median. These results do not
pass an overall performance acceptance gate; the shared SQLite SDK, production
sources and installed module remain unchanged.

## Controls and durability

Three independent Release components use byte-identical production sources:

- `sdk` links the original SQLite SDK, retained as a calibration group.
- `fsync` links a rebuilt SQLite 3.53.2 with `HAVE_FDATASYNC=0`.
- `fdatasync` links the same amalgamation, headers and configuration with
  `HAVE_FDATASYNC=1`.

The matched builds differ only in this SQLite compile definition. The original
SDK is not an equivalent build control: differences from it are reported
separately and are not attributed to the flag. The preparer reuses the
[preceding experiment's](../staged_payload/wal_sync/README.md) SQLite build
recipe and syscall regression, without applying any experimental storage patch.

SQLite documents that this option changes the Unix VFS's selected
[synchronization primitive](https://www.sqlite.org/compile.html#have_fdatasync).
It does not change SQLite's transaction or synchronization settings. Both
components preserve VaneFS's FULL-commit `strict` default and its explicit
`fsync` mode, including FULL barriers, synchronous writes, checkpoint thresholds
and WAL retention. Defining the macro on VaneFS alone would not change an
already compiled SQLite library.

Runtime versions, SourceIDs and reported compile options match across all
three libraries. The macro is not listed by `sqlite3_compileoption_get`, so
library symbols and actual WAL syscalls are checked separately: SDK/control
use `fsync`; the candidate uses `fdatasync`.

## Measurement protocol

All runs use plain binaries, without syscall wrappers or profiling. Each group
contains six balanced variant orders, rotating all three positions and then
reversing those rotations. Every sample, including slow runs, is retained.
Throughput summaries use the median of all six independent mount rates.
Latency percentiles use nearest rank; latency medians use the conventional
median.

- `fsync`, fresh: 18 mounts with sequential 64/256 MiB writes, complete-file
  validation, warmed reads, random reads/overwrites, small files and metadata.
- `fsync`, aged: 18 independent mounts retain a 32 MiB guard with alternating
  4 KiB blocks zeroed, collect garbage, then run the read/overwrite, small-file
  and metadata workloads. The guard is checked before and after measurement.
- `strict`, fresh: 18 mounts use the same workloads with 64 MiB sequential
  writes. The 256 MiB phase is omitted from this group by design.

Sequential writes use 1 MiB application requests and include final file fsync
and close. Reads use a 64 MiB fixture, completely read and hashed before timing.
Random reads compare all 4,096 seeded 4 KiB slices. Random writes modify 512
distinct seeded blocks; request time and final fsync are retained separately,
and the complete modified file is hashed. Each of 128 small files is created,
written with 4 KiB, fsynced and closed, then read back outside timing. Metadata
phases each perform 128 operations and include a final directory fsync; they
do not issue a separate directory barrier after every operation in fsync mode.

The host monitor samples at one-second intervals, with a 20-second baseline
before each durability group. No owned builds or correctness tests overlap
measurement windows. Cleanup happens after unmount and process exit, outside
measured phases. Every workspace passes SQLite `quick_check`, retains
`format=(2,4096)` and contains only 4 KiB payloads with no external sidecar.

This host has 36 logical CPUs (Xeon E5-2686 v4), Linux `6.11.0-24-generic`,
ext4 on `/dev/sda2`, and a Fanxiang S103Pro SATA SSD with write-back caching,
`mq-deadline` and `wbt_lat_usec=2000`. Approximately 104 GB is free. Whole-host
CPU busy time averages 5.7% during the fsync window and 5.1% during strict;
average I/O wait is 1.9% and 1.5%. These host aggregates do not rule out storage
interference. No global cache, kernel, scheduler or device setting is changed.

## Results

The following are medians of all six rates. Read/write units are MiB/s; small
files use files/s and directory operations use operations/s. `fsync` and
`fdatasync` identify the SQLite builds, independently of the VaneFS durability
mode named above each table.

### Fsync mode, fresh workspaces

| Workload | Original SDK | Rebuilt fsync | Rebuilt fdatasync |
| --- | ---: | ---: | ---: |
| 64 MiB write + fsync | 52.83 | 13.85 | 12.26 |
| 256 MiB write + fsync | 94.42 | 91.81 | 94.45 |
| Warm 64 MiB read | 529.36 | 521.56 | 555.84 |
| 4 KiB random read | 67.44 | 65.94 | 69.27 |
| 4 KiB overwrites + final fsync | 12.96 | 11.20 | 13.68 |
| Small file + fsync | 258.02 | 261.42 | 361.00 |
| mkdir + final directory fsync | 1,988.41 | 2,115.73 | 1,900.95 |
| rename + final directory fsync | 2,298.23 | 2,428.21 | 2,353.26 |
| rmdir + final directory fsync | 2,363.94 | 2,415.64 | 2,399.49 |

The candidate's small-file rate exceeds its matched control in all six orders.
The 256 MiB write median improves only 2.9%. The 64 MiB results are strongly
bimodal: both matched builds range from roughly 9 to 98 MiB/s. Their medians,
and the much higher original-SDK median, cannot establish a reliable bulk-write
benefit from this flag. All six individual rates remain in the result files.
Read-rate differences likewise do not establish a synchronization-related read
optimization.

### Fsync mode, after fragmentation and GC

| Workload | Original SDK | Rebuilt fsync | Rebuilt fdatasync |
| --- | ---: | ---: | ---: |
| Warm 64 MiB read | 526.49 | 533.69 | 550.68 |
| 4 KiB random read | 69.60 | 68.85 | 70.43 |
| 4 KiB overwrites + final fsync | 14.64 | 11.39 | 10.73 |
| Small file + fsync | 251.15 | 262.76 | 381.18 |
| mkdir + final directory fsync | 1,930.98 | 1,475.10 | 1,980.46 |
| rename + final directory fsync | 2,328.03 | 1,825.13 | 2,313.97 |
| rmdir + final directory fsync | 2,380.85 | 2,099.70 | 2,337.52 |

The 45.1% small-file median gain coexists with candidate runs at 24.91 and
33.39 files/s. The control also has slow runs at 29.36 and 18.94 files/s. Pooled
per-file p95 latency is 84.39 ms for the control and 78.12 ms for the candidate,
compared with 4.31 and 4.32 ms on fresh workspaces. The longest aged final
overwrite fsync takes 1,098.03 ms in the control and 1,195.80 ms in the candidate.
The flag does not remove these tails.

### Strict mode, fresh workspaces

| Workload | Original SDK | Rebuilt fsync | Rebuilt fdatasync |
| --- | ---: | ---: | ---: |
| 64 MiB write + fsync | 30.86 | 51.06 | 59.39 |
| Warm 64 MiB read | 518.75 | 516.99 | 527.57 |
| 4 KiB random read | 70.35 | 68.98 | 70.13 |
| 4 KiB overwrites + final fsync | 1.41 | 1.44 | 2.46 |
| Small file + fsync | 173.39 | 177.59 | 290.82 |
| mkdir + final directory fsync | 342.63 | 334.01 | 527.65 |
| rename + final directory fsync | 360.05 | 345.45 | 565.15 |
| rmdir + final directory fsync | 205.29 | 196.49 | 370.09 |

Strict's ordinary mutations already include their FULL barrier. Its final
explicit fsync therefore usually has little work left; the random-write request
time contains the persistence cost. The candidate improves median rates across
these write and metadata workloads, but its first small-file run takes 6.156
seconds (20.79 files/s), including a 467.07 ms individual operation. Pooled
small-file p95 increases from 5.63 to 81.65 ms. This is retained in every summary.

To expose the effect of slow runs, this second aggregation divides total work
by the sum of the six measured phase durations. It excludes preparation and
validation, exactly as each per-run rate does:

| Total-work / total-time throughput | Rebuilt fsync | Rebuilt fdatasync |
| --- | ---: | ---: |
| Fsync fresh small files, files/s | 257.83 | 368.51 |
| Fsync aged small files, files/s | 58.86 | 75.00 |
| Strict fresh small files, files/s | 178.49 | 92.32 |
| Fsync aged overwrites + fsync, MiB/s | 5.69 | 5.33 |
| Strict fresh overwrites + fsync, MiB/s | 0.83 | 0.73 |

This window supports lower common small-operation cost, not an unconditional
speedup. It does not isolate the cause of each long call: these are plain
performance binaries, and host counters include other activity. Earlier
experimental-format traces cannot assign causes to these production samples.
Six runs and a single device are also insufficient to characterize a rare-event
distribution. The next integration should expose a Linux opt-in SQLite build
option with the syscall/error tests retained; changing the default needs an
acceptable strict/aged total-time and latency result on the intended storage.

## Correctness and compatibility

Each of the three builds passes **132 related Python tests and six C++ tests**.
The native cases cover the core, inode references, checkpointing, durability,
crash recovery and the actual SQLite WAL sync primitive. The last case injects
six WAL syscall failures per build across explicit sync, synchronous writes
and close in both modes. It checks error propagation, rollback, retries that
reach storage, and acknowledged bytes after reopening. An intentionally wrong
syscall expectation fails in every build, checking that the detector is active.

Seven compatibility cases exercise 42 subprocess operations. Each native
build creates a database in both strict and fsync modes; the existing installed
Python module creates an additional strict database. All four Python modules
then read and modify every database, run GC, check snapshots, sparse data and
merged child contents, and reopen in the original writer. Integrity checks
and all content comparisons pass.

The production Python constructor exposes strict mutations only; the C++ seed
program exercises fsync creation. The initial harness incorrectly supplied a
Python durability keyword and failed before creating a database. That failed
scaffold, its logs and cleanup are retained, alongside the corrected native
seeding matrix. It is not counted as a product failure or a passing test.

## Reproduce

Use Linux, a C++ compiler, CMake, Ninja, FUSE 3, the pinned SQLite amalgamation
and the existing SQLite SDK headers/configuration. The output must not exist:

```bash
python3 vane_fs/benchmarks/production_sync/prepare.py \
  /absolute/path/to/new-controls \
  --amalgamation /absolute/path/to/sqlite3.c \
  --sqlite-prefix /absolute/path/to/sqlite-sdk
cmake -S /absolute/path/to/new-controls/sqlite \
  -B /absolute/path/to/new-controls/sqlite/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /absolute/path/to/new-controls/sqlite/build --parallel 2
```

Build each `components/{sdk,fsync,fdatasync}` with its matching
`sqlite/{sdk,fsync,fdatasync}` prefix. Enable `VANE_FS_BUILD_PYTHON`,
`VANE_FS_BUILD_TOOLS`, `VANE_FS_BUILD_FUSE` and `VANE_FS_BUILD_TESTS`. Use a
separate Release `SKBUILD_BUILD_DIR` and a non-editable wheel installation for
each variant. The frozen build runner records the exact commands, environments,
FUSE SDK path and installed module identities used here. The preparer adds the
WAL syscall regression and `vane_fs_sync_seed` helper to the isolated component
only. Run its native tests with `ctest --test-dir <component>/build` and its
Python tests with the corresponding installed interpreter.

The compatibility runner accepts labeled interpreters and matching native
seed binaries. Python-only entries get a strict seed; native entries get both
durability modes:

```bash
python3 vane_fs/benchmarks/production_sync/round_trip.py \
  /absolute/path/to/new-compatibility-output \
  --python installed=/absolute/path/to/installed/python \
  --python fsync=/absolute/path/to/fsync/venv/bin/python \
  --python fdatasync=/absolute/path/to/fdatasync/venv/bin/python \
  --seed fsync=/absolute/path/to/fsync/build/vane_fs_sync_seed \
  --seed fdatasync=/absolute/path/to/fdatasync/build/vane_fs_sync_seed
```

Performance measurements use the frozen runner and workload identified in
[artifact-manifest.json](artifact-manifest.json). The workload derives from the preceding comparison
with only the durability flag, sequential-size schedule and console summary
parameterized. A production-format inspector replaces the external-format
inspector. Source and runner hashes preserve those exact inputs.

The final preparer is also run in a new directory after measurements. All 54
source/library/binary comparisons are identical, including all three SQLite
libraries, VaneFS static libraries and FUSE executables. Its added sync test
passes in all three builds. The CMake-built compatibility seed helpers pass
six more cases (30 subprocess operations) using the existing isolated Python
installations. The measured matrix used the same seed source compiled directly
against those libraries; this final check verifies its CMake integration.

[results.json](results.json) records build and host identities, all rate vectors,
latency summaries, total-time rates and validation. Per-mount content checks,
GC inspection, commands and cleanup are in [fsync fresh](fsync-fresh.json),
[fsync aged](fsync-aged.json) and [strict fresh](strict-fresh.json). The artifact
manifest points to complete local timing vectors, monitors, all build/test logs,
the initial failed scaffold, source snapshots and binaries. All 78 cleanup
entries are verified absent, including 54 measured workspaces. No owned mount
or run process remains, and no owned build/test window overlaps measurements.
Production source, shared SDK and installed-module hashes are unchanged. The
full base suite is not rerun; validation is limited to these affected components,
reproduction checks and repository checks.
