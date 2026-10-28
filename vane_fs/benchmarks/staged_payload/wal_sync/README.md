# SQLite WAL synchronization experiment

This experiment follows the [aged overwrite investigation](../aged_write/README.md).
It isolates the SQLite Unix VFS synchronization primitive from changes to VaneFS
SQL, checkpoint policy, payload publication and WAL retention. Both controls use
the same exact-interval UPDATE component and experimental format 1004.
**The option reduces ordinary synchronization cost, but does not resolve the
long-tail acceptance failure. The candidate remains isolated.**

## Evidence and candidate

The pinned core-only vcpkg SQLite 3.53.2 static library references `fsync` and
does not enable `HAVE_FDATASYNC`. SQLite's
[compile-time documentation](https://www.sqlite.org/compile.html#have_fdatasync)
describes this option: the Unix VFS uses `fdatasync` where appropriate when it is
enabled. This definition must be applied when compiling SQLite itself; defining
it only on the VaneFS target cannot change a prebuilt SQLite library.

SQLite's `full_fsync` implementation in the frozen amalgamation uses `fdatasync`
on Linux when available. It documents that file content and necessary size
metadata still need to reach storage, whereas unrelated inode metadata need not.
The upstream [Linux 6.11 ext4 implementation](https://github.com/torvalds/linux/blob/v6.11/fs/ext4/fsync.c)
chooses a different journal transaction identifier for data-only synchronization.
These are source-level reasons to test the option, not evidence that every
previous slow sample has the same cause. This machine runs Ubuntu kernel
`6.11.0-24-generic` with ext4 on `/dev/sda2`.

Two isolated SQLite libraries use the identical amalgamation, vcpkg feature
configuration, compiler and Release flags, differing only in
`HAVE_FDATASYNC=0` or `1`. Their SQLite source IDs and reported compile options
match the original SDK. Undefined-symbol inspection and the syscall regression
verify the actual synchronization primitive. SQLite does not report this option
in `compile_options`, so that list alone cannot verify the change.

The `fdatasync` candidate retains SQLite FULL barriers at explicit synchronization
and the existing NORMAL commits in fsync mode. The existing 16 MiB checkpoint
trigger, 64 MiB admission budget and 16 MiB retained WAL policy also apply.
SQLite's [journal-size limit](https://www.sqlite.org/pragma.html#pragma_journal_size_limit)
is enforced on WAL reset, so it is distinct from both the checkpoint trigger and
the amount of uncheckpointed data.

## Controlled file probe

Eight balanced orders compare 128 writes of 32 KiB, synchronizing each write.
Every file starts with 8 MiB of initialized, persisted data. The reuse case
overwrites that existing range; the growth case appends. Setup, full-content
verification and removal take place outside the measured operations.

| Total synchronization time per 128 writes, median | fsync | fdatasync |
| --- | ---: | ---: |
| Reuse initialized space | 272.68 ms | 118.05 ms |
| Grow the file | 285.87 ms | 277.17 ms |

Reusing initialized space reduces synchronization time by 56.7%; growth improves
by only 3.0%. These are raw-file results, not VaneFS throughput. File length and
allocated bytes are recorded after every operation, and all contents are checked.

## Same-window FUSE comparison

Six balanced orders each cover fresh and GC-aged workspaces: 36 independent
mounts in total. The original SDK provides an additional calibration group.
The matched controls use identical rebuilt SQLite inputs except for the sync
option. Workloads retain the preceding 1 MiB sequential requests, 64 MiB warm
read fixture, 512 seeded 4 KiB overwrites and 128 small-file/metadata operations.
Each aged workspace first removes alternating blocks from a 32 MiB guard and
runs GC. Builds and correctness tests finish before measurements begin.

Medians, including every slow sample:

| Workload | Original SDK | Rebuilt fsync | Rebuilt fdatasync |
| --- | ---: | ---: | ---: |
| 256 MiB write + fsync, MiB/s | 141.80 | 141.85 | 144.39 |
| Warm sequential read, MiB/s | 756.09 | 787.14 | 806.27 |
| Fresh random read, MiB/s | 67.87 | 70.26 | 71.84 |
| Fresh overwrite requests, MiB/s | 19.03 | 20.67 | 19.17 |
| Fresh overwrite + fsync, MiB/s | 14.21 | 13.96 | 14.84 |
| Aged overwrite requests, MiB/s | 16.96 | 13.35 | 15.07 |
| Aged overwrite + fsync, MiB/s | 16.43 | 13.00 | 12.93 |
| Aged random read, MiB/s | 68.82 | 69.50 | 69.10 |
| Aged small file + fsync, files/s | 256.39 | 186.69 | 392.34 |
| Aged mkdir + directory fsync, ops/s | 2,072.19 | 1,925.67 | 2,067.90 |
| Aged rename + directory fsync, ops/s | 2,453.28 | 2,478.49 | 2,552.19 |
| Aged rmdir + directory fsync, ops/s | 2,431.32 | 2,627.15 | 2,676.52 |

Against the matched fsync control, small-file throughput improves by 110.2%,
while 256 MiB writes improve by only 1.8%. Aged request throughput improves by
12.9%, but overwrite-plus-fsync is essentially unchanged and fresh request
throughput falls by 7.3%. These results do not establish an overall acceptance
pass. The original SDK and rebuilt control also differ on some workloads;
in particular, their first 64 MiB write medians are 16.98 and 154.14 MiB/s,
versus 158.77 for fdatasync. That calibration discrepancy remains unexplained;
the option's effect is assessed using the matched pair.

The fdatasync candidate's six aged final barriers take 2.23, 2.78, 2.16, 5.26,
758.90 and 1.86 ms. Its small-file rates are 418.64, 387.08, 390.76, 418.36,
29.66 and 393.92 files/s. Thus the improved medians still include a severe slow
run. The matched fsync control has two 787–837 ms aged barriers, and the original
SDK also has a 21.74 files/s small-file run. No sample is discarded or replaced.

## Remaining waits

A separate diagnostic window uses four balanced orders of the two rebuilt
variants. It records sync calls, slow reads/writes, checkpoints, every truncate,
file length/allocation at sync boundaries, and external 10 ms thread samples.
These instrumented runs are kept separate from adoption throughput.

In fast small-file runs, 127 successive WAL syncs with unchanged observed length
and allocation have medians of about 2.12 ms for fsync and 1.04 ms for fdatasync.
However, two fdatasync runs still take about four seconds for 128 small files.
Their WAL stays at 16,777,216 bytes with 16,781,312 allocated bytes, and neither
phase contains a WAL truncate. Repeated slow syncs occur after the first barrier,
so a fresh size extension or repeated truncation is not required for this stall.
Observed size/allocation does not describe every extent or device operation.

Those two slow small-file phases contain 316 and 310 foreground samples in
`submit_bio_wait` inside WAL `fdatasync`; the checkpoint worker is mostly in a
futex wait. The overwrite phases also sample `rq_qos_wait`, and one foreground
WAL sync lasts 818.66 ms. This locates remaining waiting in the kernel block-I/O
path. It does not identify the individual block request type, prove another
process caused it, or diagnose a hardware fault. Consequently, increasing WAL
retention alone is not justified as a solution to these remaining tails.

Read-only device inspection after the run records a Fanxiang S103Pro SATA SSD,
write-back caching, `mq-deadline` and a 2,000 microsecond writeback-throttling
target. No device, scheduler, cache or throttling settings are changed. During
the plain window, host CPU busy time has a 4.94% median and 14.87% maximum;
iowait peaks at 11.10%. These aggregate counters include preparation and cleanup
and do not assign waiting to a particular cause.

The next useful experiment is to correlate individual data-write and flush
completion times with the sampled block-layer waits while keeping durability
barriers intact. Production BLOB adoption of this dependency build option also
needs its own validation; these format-1004 results cannot substitute for it.

## Validation and scope

The new native regression intercepts the actual WAL syscall. It checks the
selected primitive, STRICT versus NORMAL ordinary writes, and six injected
`EIO` failures across explicit barriers, synchronous writes and workspace close.
It verifies rollback, retry and acknowledged content after reopening. Running it
with the wrong expected primitive fails, demonstrating that a missing build flag
is detectable.

Enabling SQLite `fdatasync` also exposed a defect in the previous loss-model test
fixture: its generic syscall hook captured SQLite's temporary rollback journal
and kept it after the journal was deleted. Reopening that artificial image
incorrectly replayed an obsolete journal. The fixture patch limits this hook to
payload and fence sidecars; the existing VFS hook continues to capture database
and WAL images. A captured trace records the unwanted journal and the resulting
failure. This correction changes the test model, not storage behavior.

Both variants pass 144 related Python tests with one explicit production-v1
migration skip, eleven existing native cases, and the new syscall regression.
Existing publication, checkpoint, GC and deterministic crash/loss cases remain
included. These are software fault models, not physical power-loss certification.

The preparation below creates isolated components and SDKs. Production sources,
the installed production module and the shared vcpkg SDK are not changed by
preparation. No production format migration is provided by this experiment.

## Reproduce

Supply the unmodified SQLite 3.53.2 amalgamation and the pinned core-only Linux
SDK. The preparer verifies the SHA-256 of the source, header and feature config,
then inherits the preceding experiments' VaneFS source checks. It rejects an
existing output directory.

```bash
sync_trial="$PWD/vane_fs/build/wal-sync-reproduction"
.venv/bin/python vane_fs/benchmarks/staged_payload/wal_sync/prepare.py "$sync_trial" \
  --amalgamation /absolute/path/to/sqlite3.c \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release"
cmake -S "$sync_trial/sqlite" -B "$sync_trial/sqlite/build" \
  -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$sync_trial/sqlite/build" --parallel 2
for variant in fsync fdatasync; do
  cmake -S "$sync_trial/$variant" -B "$sync_trial/$variant/build" \
    -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$sync_trial/sqlite/$variant" \
    -DVANE_FS_BUILD_PYTHON=OFF -DVANE_FS_BUILD_TESTS=ON
  cmake --build "$sync_trial/$variant/build" --parallel 2
  ctest --test-dir "$sync_trial/$variant/build" --output-on-failure
done
```

The component builds register twelve native cases each. Python measurements use
persistent Release build directories and isolated non-editable installations.
FUSE measurements additionally require `VANE_FS_BUILD_FUSE=ON`, libfuse3 and
`/dev/fuse`. Complete benchmark runners, raw timings, source/binary identities
and cleanup records are retained with the local artifacts.

## Retained evidence

[results.json](results.json) records aggregate measurements, build identities
and validation. The [fresh samples](fresh-results.json),
[aged samples](aged-results.json) and [diagnostic phases](timeline.json) retain
every scalar FUSE result and latency summaries in separate files.
The [artifact manifest](artifact-manifest.json) identifies full timing vectors,
traces, frozen sources, binaries, runners and logs retained locally. Owned
workspaces and raw-probe data are removed after their file descriptors, mounts
and processes stop, outside measurement windows. The preceding failed acceptance
window remains unchanged and is not pooled with this comparison.
