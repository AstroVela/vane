# SQLite page-layout experiment: 2026-10-08

Keep the production defaults. None of the larger page sizes improves the
256 MiB write median, and their sequential-read gains come with higher WAL
volume, slower random writes and weaker aged metadata performance. The patches
remain isolated experiments; no production source or persistent format changes.

This experiment keeps independent 4 KiB payload rows and the production version
and GC algorithms. Only copied component sources change. It compares 4, 8, 16
and 32 KiB initial SQLite pages, with a fixed 2 MiB suggested page-cache budget
per writer and checkpoint connection. Strict automatic checkpoints trigger
at the first complete WAL frame at or above 16 MiB, including WAL headers.
Fsync mode retains its existing 16 MiB wake, 64 MiB admission and 16 MiB retained
WAL budgets. All variants have the same FULL/NORMAL durability boundaries.

The 4 KiB control uses these explicit settings too. It is not a byte-identical
stock binary: stock strict mode triggers every 4,096 frames, and stock uses the
SQLite default suggested cache budget. Comparisons are within this experiment.

## Hypothesis and scope

A 4 KiB BLOB plus its record header exceeds a 4 KiB table-leaf page's local
payload threshold. The [SQLite file format](https://www.sqlite.org/fileformat2.html#b_tree_pages)
stores the remainder on overflow pages. Larger pages can keep these small
payload records entirely on the leaf page. This could reduce lookups and improve
sequential access, but also increases the bytes read or logged for each page
and changes page occupancy. Performance and space must be measured together.

The compile-time `VANE_FS_PAGE_BYTES` setting applies only when the database has
zero pages. Existing databases, including preinitialized empty databases, retain
their recorded size; there is no automatic VACUUM or migration. Application ID,
format version 2, independent payload rows and 4 KiB logical blocks are
unchanged. Original production binaries can read and write the candidate
databases. The experiment adds no dependency or payload-compaction machinery.

## Measurement design

Fresh FUSE measurements use eight rounds of four variants. The first four
Williams orders place each variant in every position and each ordered adjacent
pair once; the next four reverse those orders. Every sample uses a new database
and mount, fsync mode, live direct I/O and disabled kernel writeback. Timed
workloads are 64/256 MiB writes with final fsync and close, warm 64 MiB reads,
4,096 seeded aligned 4 KiB reads, and 512 distinct 4 KiB overwrites with final
fsync. The overwrite timer excludes close. Full hashes and all random slices
are checked outside the timer.

Four further rounds use independent mounts with an aging preparation. A 32 MiB
guard file is written in separate 4 KiB calls, alternating blocks are zeroed,
then GC runs while the surviving guard remains. The resulting free pages and
leaf slack depend on page size; their actual values are retained. A new 64 MiB
file supplies the read/overwrite workload. These mounts also measure 128 small
files with write/fsync/close per file, and 128 mkdir/rename/rmdir operations with
a final directory fsync for each group. Guard content is verified before and
after. This is one deterministic aging pattern, not a steady-state guarantee.

Four rotating native rounds separately instrument 256 MiB writes and seeded
random reads. Read diagnostics use a warm 64 MiB fresh fixture and a warm 16 MiB
fixture following the same 32 MiB guard/alternating-deletion preparation. The
smaller reused fixture stresses allocation into available holes; it differs
from the FUSE read fixture. VFS byte counts are software I/O, not physical SSD
traffic. Foreground and checkpoint-worker scopes may overlap.

Four rounds repeat the prior 8 MiB snapshot, small-overwrite and sparse-GC space
cases through the strict-mode native API. GC timing includes its FULL commit.
Freed SQLite pages may stay in the file; both live payload bytes and allocated
page usage are recorded.

No owned builds or correctness tests overlap measured comparisons. Host CPU,
memory, pressure, disk counters and free space are retained; no global cache
eviction or host configuration change is performed. Every sample and stall is
retained. Cleanup follows process/mount shutdown and validation, outside the
timed workload windows.

The host is an Intel Xeon E5-2686 v4 (36 logical CPUs), with about 62.7 GiB RAM
and an ext4 filesystem on a Fanxiang S103Pro SSD. Builds use GCC 13.3.0,
SQLite 3.53.2 and libfuse 3.14.0. The 30-second baseline records 95.0% CPU idle
and 0.18% I/O wait; the full measurement/validation interval records 90.4% idle
and 3.40% I/O wait. Minimum sampled FUSE free space is 102.8 GiB. This shared
host still exhibits stalls; the counters do not identify their individual causes.

## Native write volume and page occupancy

The instrumented native driver writes 256 MiB as 256 pairs of 1,048,528 and
48 bytes, matching the observed FUSE request split. Every variant performs
512 write commits and the same 1,280 payload-insert and 1,280 version-insert
steps. Each stores 65,792 payload rows containing 257 MiB, including one MiB
of superseded partial blocks. Logical content and row granularity are identical.

Cells below are medians from four orders; VFS bytes include setup, write, Sync,
readback and close. Payload-table allocation sums its B-tree/overflow pages.

| Metric | 4 KiB pages | 8 KiB pages | 16 KiB pages | 32 KiB pages |
| --- | ---: | ---: | ---: | ---: |
| WAL writes, MiB | 338.64 | 590.41 | 464.10 | 508.37 |
| Database writes, MiB | 307.93 | 536.73 | 367.27 | 324.69 |
| Payload-table allocation, MiB | 289.21 | 514.63 | 342.89 | 293.84 |

All payload overflow pages disappear at 8/16/32 KiB. However, an 8 KiB leaf
holds only one 4 KiB payload record because two records plus headers do not
fit. The 16/32 KiB leaves hold three/seven records in this fixture. Larger dirty
pages increase WAL bytes despite fewer VFS write calls; eliminating overflow
pages alone does not reduce the amount of data written.

## Fresh FUSE results

Cells are MiB/s, median (complete range), across eight rounds. Writes include
the final fsync; sequential writes also include close. Reads follow a full
validation/warmup pass. The same mount runs the five workloads in table order,
deleting each fixture afterward without intermediate GC.

| Workload | 4 KiB pages | 8 KiB pages | 16 KiB pages | 32 KiB pages |
| --- | ---: | ---: | ---: | ---: |
| 64 MiB sequential write | 95.97 (51.31–99.68) | 59.99 (29.39–64.18) | 99.17 (31.95–108.57) | 102.71 (20.06–107.64) |
| 256 MiB sequential write | 93.95 (44.30–96.53) | 35.68 (17.52–57.97) | 79.89 (22.47–84.51) | 88.70 (23.76–90.44) |
| Warm 64 MiB sequential read | 529.70 (498.35–658.79) | 467.34 (447.14–674.97) | 582.65 (565.70–628.36) | 644.83 (603.17–679.84) |
| Warm 4 KiB random read | 67.93 (65.54–71.88) | 70.68 (65.51–73.62) | 63.95 (59.29–66.64) | 60.50 (56.35–66.39) |
| 4 KiB random overwrite | 13.57 (8.32–16.36) | 9.93 (6.60–12.30) | 6.33 (3.80–6.78) | 3.60 (2.12–3.71) |

The 16/32 KiB variants improve median sequential read throughput by 10.0%/21.7%
but reduce random-overwrite throughput by 53.4%/73.5%. None improves the
256 MiB write median. All variants include slow sequential-write samples;
the ranges and raw samples are retained, so the medians do not imply a latency
guarantee or an explanation for individual stalls.

## FUSE after deletion and GC

Medians below use four independent aged mounts per variant. MiB/s applies to
the first three rows; the remaining rows are operations/s. Complete ranges
and per-mount samples are in [results.json](results.json).

| Workload | 4 KiB pages | 8 KiB pages | 16 KiB pages | 32 KiB pages |
| --- | ---: | ---: | ---: | ---: |
| Warm 64 MiB sequential read | 519.03 | 465.62 | 670.83 | 673.93 |
| Warm 4 KiB random read | 69.20 | 72.72 | 66.90 | 58.56 |
| 4 KiB random overwrite | 13.55 | 10.17 | 7.11 | 2.93 |
| 4 KiB file + fsync + close | 207.76 | 142.64 | 188.66 | 104.94 |
| mkdir | 1,988.22 | 1,914.77 | 1,244.46 | 646.03 |
| rename | 2,435.11 | 1,391.64 | 1,266.01 | 1,046.79 |
| rmdir | 2,382.24 | 1,719.50 | 1,177.47 | 499.17 |

The sequential-read benefit of 16/32 KiB pages survives this preparation, but
random overwrites and metadata operations still regress. Small-file results
vary substantially: the control spans 52.75–280.35 files/s, and its final
random-overwrite sample falls to 1.38 MiB/s. These stalls remain included.

Per-call latency also exposes a cost independent of the final fsync. For the
fresh random-overwrite workload, pooled p95/p99 `pwrite` latencies are
0.368/0.467, 0.385/0.459, 0.575/0.741 and 0.798/1.104 ms for 4/8/16/32 KiB
pages, respectively, with 4,096 calls per variant. The aged workload's p95/p99
values are 0.302/0.380, 0.353/0.415, 0.477/0.595 and 0.793/1.228 ms, with
2,048 calls per variant. These percentiles describe individual calls; they
exclude the final group fsync and are not percentiles of durable transactions.

## Native read amplification and space reclamation

The native diagnostic requests 4,096 seeded 4 KiB slices, totaling 16 MiB,
after a complete warm read. Medians of four rounds are below. The fresh and
aged fixtures differ in size, so compare variants within each row.

| VFS read volume for 16 MiB of requested data | 4 KiB pages | 8 KiB pages | 16 KiB pages | 32 KiB pages |
| --- | ---: | ---: | ---: | ---: |
| Fresh 64 MiB fixture, MiB | 38.01 | 51.70 | 102.16 | 204.81 |
| Aged 16 MiB fixture, MiB | 28.01 | 32.82 | 64.43 | 138.48 |

Larger pages remove overflow-page traversal but make each cache miss fetch a
larger unit, with fewer pages fitting in the fixed byte budget. The fresh
8 KiB variant makes 6,617 VFS reads versus the control's median 10,211 calls,
yet transfers more bytes. Its slightly higher FUSE random-read throughput
therefore does not imply lower read amplification. The 16/32 KiB variants
transfer substantially more data even after OS-cache warmup. VFS counters do
not imply equivalent physical-device reads.

The aged FUSE preparation retains 16 MiB of payload contents in the 32 MiB
guard after zeroing alternate blocks and running GC. The following values
are identical across all four repetitions, before creating the read fixture.
Allocation excludes free pages; leaf slack is already included in allocation.

| Guard allocation after GC | 4 KiB pages | 8 KiB pages | 16 KiB pages | 32 KiB pages |
| --- | ---: | ---: | ---: | ---: |
| Payload-table allocation, MiB | 20.01 | 32.08 | 21.36 | 36.59 |
| Unused bytes inside payload leaves, MiB | 2.04 | 15.93 | 5.29 | 20.51 |
| Database freelist pages | 4,096 | 4,096 | 1,367 | 1 |

The 32 KiB candidate leaves substantial slack inside live leaves rather than
returning whole pages to the freelist. Deleting the same logical blocks does
not create the same reusable layout across page sizes.

The separate 8 MiB space cases preserve independent-block reclamation: after
dropping the snapshot, 128 small overwrites retain exactly 8 MiB of payloads;
the sparse case retains exactly its 128 KiB of live payloads. Unlink and GC
eventually remove all payload rows. Database files keep reusable free pages.

| Strict-mode GC, median milliseconds (range) | 4 KiB pages | 8 KiB pages | 16 KiB pages | 32 KiB pages |
| --- | ---: | ---: | ---: | ---: |
| Reclaim 512 KiB of superseded blocks from mostly live data | 49.21 (49.13–49.89) | 23.14 (22.73–38.79) | 31.54 (31.12–32.19) | 43.37 (41.97–44.35) |
| Reclaim blocks after sparse replacement | 26.06 (23.91–27.75) | 43.00 (36.77–64.07) | 57.37 (56.75–63.59) | 49.38 (45.46–53.31) |

Larger pages improve the first GC case but worsen the second. Together with
the write-volume, random-I/O and allocation results, these measurements do
not support changing the default. A future external-payload prototype would
need its own durable publication and recovery protocol, orphan reclamation,
and bounded random reads before a performance comparison. It is outside this
page-layout experiment.

## Validation and retained artifacts

Each of the four variants passes 135 component Python tests, all five Release
CTest suites, and the standalone C++ configuration check. The original v1
migration test remains included. New tests cover layout, snapshot GC, reopening
existing page sizes, and reading/writing candidate databases with the unchanged
production binary. ASan/UBSan was not rerun for this isolated experiment.

The measured workloads pass 192 FUSE full-file hashes, 196,608 FUSE random-slice
comparisons, 2,048 small-file comparisons, 64 native full-content comparisons,
131,072 native random-slice comparisons, 64 space-case comparisons and 208
SQLite quick checks. The preparer reproduces all 12 checked candidate
source/test files exactly. Only related tests were run.

[results.json](results.json) retains every workload sample, latency summaries,
page inspections, identities and cleanup references. Complete per-call samples,
per-table inspections, VFS counters, frozen sources/binaries, wheel artifacts,
runner scripts, host telemetry and logs remain in
`vane_fs/build/page-layout-20261008T123136Z/`. Its `artifact-manifest.json` records
hashes and detailed cleanup. The earlier correctness-only smoke overlaps
component tests and is excluded from the performance results. All measured
comparisons start after owned builds/tests finish. All 122 owned cleanup
records verify removal, with no remaining experiment processes or mounts;
the installed production module retains its original hash.

## Reproduction

```bash
python3 vane_fs/benchmarks/page_layout/prepare.py vane_fs/build/page-layout-reproduction
```

The preparation script verifies source hashes and applies `layout.patch` to
four component copies. Build each copy as a non-editable Release wheel with
`VANE_FS_BUILD_TESTS`, `VANE_FS_BUILD_TOOLS` and `VANE_FS_BUILD_FUSE` enabled.
Set `CMAKE_CXX_FLAGS=-DVANE_FS_PAGE_BYTES=<bytes>` to the matching page size,
reuse its persistent incremental build directory, and use the same SQLite and
libfuse SDKs. Install wheels into isolated environments so `python -I` test
subprocesses use the intended variant.

Run the five CTest suites and the copied component Python suite, with
`VANE_FS_PAGE_BYTES`, `VANE_FS_MOUNT_BINARY` and `VANE_FS_PRODUCTION_PYTHON` set.
No legacy format-upgrade test is excluded. Three added Python checks verify
actual page placement and GC, preservation of existing page sizes, and content
interoperability with the unchanged production binary.

`test_configuration.cpp` links each candidate's `libvane_fs.a` and the same
SQLite static library with `-Wl,--wrap=sqlite3_exec` and
`-Wl,--wrap=sqlite3_wal_autocheckpoint`. Compile with the matching
`VANE_FS_PAGE_BYTES`, then run with a new database path. It inspects actual
writer/worker cache settings, initial FULL/NORMAL modes and strict checkpoint
frame thresholds.
