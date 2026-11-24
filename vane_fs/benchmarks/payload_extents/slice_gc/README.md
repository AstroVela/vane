# Payload slice lookup and GC experiment: 2026-10-07

This follow-up to the [extent experiment](../README.md) reclaims dead slices
and repairs random reads on fresh allocation, but **neither candidate is
adopted**. Reusing fragmented free pages restores read amplification, sequential
reads still regress, and compacting mostly live extents costs more. The changes
remain isolated C++ prototypes; the production source, 4 KiB format, durability
barriers and checkpoint policy are unchanged.

## Read-path diagnosis

SQLite 3.53.2's [`getOverflowPage()` and `accessPayload()`](https://raw.githubusercontent.com/sqlite/sqlite/version-3.53.2/src/btree.c)
explain the earlier random-read amplification. A fresh BLOB cursor locates an
interior slice by traversing overflow pages. With pointer maps enabled, SQLite
first checks whether the next physical page points back to the current page;
if so, it can skip reading the current overflow page. Nonconsecutive pages
still require the fallback traversal.

The prototype sets [`auto_vacuum=INCREMENTAL`](https://www.sqlite.org/pragma.html#pragma_auto_vacuum)
before initializing a fresh database. This persists the pointer maps without
moving or truncating pages automatically at commits. It does not issue
`incremental_vacuum`, enlarge the page cache, retain BLOB handles across
transactions, or modify SQLite. Existing format-1001 databases without pointer
maps still open with their original page-layout mode; they need a fresh
database for this performance path. Production/prototype format refusal
remains in place.

## Live-slice reclamation

After removing unreachable versions and entirely unreferenced payloads, GC
finds extents whose distinct surviving slice count is below their allocated
size. It copies those slices into a new immutable payload and remaps every
surviving version reference, including snapshots, fork bases and branches.
It then removes the old payload. All steps run within the existing GC write
transaction. A concurrent WAL reader continues to see its original payloads;
errors and process termination cannot publish only part of a remap.

The scan advances through payload IDs, bounded by the initial maximum ID;
temporary C++ data is bounded by one extent. This experimental policy compacts
every partially live extent. Total GC work and write-lock duration are not
budgeted, and reclaiming a single dead slice can rewrite almost a full extent.
This tradeoff must be measured before adopting a production policy.

## Same-window FUSE results

All cells are median MiB/s with the full six-sample range. The six order
permutations each run once, with fresh databases/mounts and unchanged fsync
mode, live direct I/O, disabled kernel writeback, metadata cache and checkpoint
settings. Fixtures, seeds, hash validation and timing boundaries match the
parent experiment. No owned tests or builds overlap measurement; OS caches
remain enabled. Every sample is retained, including long checkpoint stalls.

| Workload | 4 KiB baseline | 64 KiB + pointer maps/GC | 256 KiB + pointer maps/GC |
| --- | ---: | ---: | ---: |
| 64 MiB write + fsync + close | 93.03 (14.59–98.74) | 97.30 (15.52–105.81) | 96.56 (92.22–104.08) |
| 256 MiB write + fsync + close | 95.00 (32.99–95.62) | 106.71 (86.71–109.43) | 110.61 (106.02–112.26) |
| Warm 64 MiB sequential read | 524.03 (500.30–656.24) | 484.50 (452.23–505.18) | 468.61 (460.59–502.44) |
| Warm 4 KiB random read | 68.38 (63.34–73.41) | 70.15 (67.02–73.56) | 68.20 (65.02–71.60) |
| 4 KiB random overwrite + final fsync | 13.00 (9.69–14.86) | 13.15 (10.18–14.63) | 11.74 (9.66–14.08) |

The 256 MiB write medians improve by 12.3% and 16.4%. The larger candidate's
write range is separated from the baseline in this window; the 64 KiB ranges
overlap. Random-read medians are now within 3% of the baseline, with overlapping
ranges. Sequential reads still regress by 7.5% and 10.6%. Random-write ranges
overlap; the 256 KiB median remains 9.6% lower. These samples do not establish a
universal speedup or remove the existing possibility of long write stalls.

The shared host's 30-second baseline has 95.46% CPU idle; measurement and
validation have 92.21% idle, 1.20% I/O wait and a one-minute load median of 2.78.
Minimum pre-mount free space is 25.89 GiB. These counters do not identify the
cause of each stall.

## Fragmentation and write volume

Instrumented native probes perform 4,096 seeded 4 KiB reads, validating every
slice. Each cell below is the median VFS-submitted MiB across three rotating
rounds; the application reads 16 MiB in each case.

| Preparation | 4 KiB baseline | 64 KiB candidate | 256 KiB candidate |
| --- | ---: | ---: | ---: |
| Fresh database, warm 64 MiB file | 38.20 | 45.87 | 41.61 |
| Reused pages, warm 16 MiB file | 30.35 | 124.96 | 447.27 |

The reused-page fixture first writes 32 MiB as separate 4 KiB payloads, zeroes
alternating blocks, collects garbage and synchronizes. It retains the surviving
guard file while allocating the measured 16 MiB file. Both full files and all
random slices are verified. This fixture deliberately stresses fragmented
free-page reuse; it is not a claim about the prevalence of fragmentation in
production. Its file size differs from the fresh fixture, so compare candidates
within each row. Random-phase native time medians are 0.112/0.139/0.276 seconds
after reuse. The pointer-map optimization is therefore insufficient as the sole
random-read solution. No FUSE fragmentation-throughput comparison was run.

In the native 256 MiB write experiment, total WAL writes are
338.64/313.69/311.24 MiB, and median database writes are
307.93/278.89/277.12 MiB. Pointer maps add some writes compared with the preceding
prototype, while total writes remain below the 4 KiB baseline. There are still
512 write commits. Write + Sync medians are 2.723/2.264/2.205 seconds; full ranges
are 2.623–8.262, 2.204–7.317 and 2.201–2.233. Foreground/background scopes can
overlap. VFS bytes measure SQLite software I/O, not physical SSD traffic.

## Reclaimed space and GC cost

Three rotating runs per variant repeat the parent experiment's 8 MiB content,
snapshot and sparse-file checks. All variants retain 8.5 MiB while a snapshot
protects the original data. After dropping it, GC now returns both prototypes
to 8 MiB, matching the baseline. In the sparse case, 128 KiB of surviving
content now retains **128 KiB** in every variant; the preceding prototypes kept
2 MiB and 8 MiB. All payloads disappear after the final unlink and GC.

The timing below measures `collect_garbage()` including its strict-mode FULL
commit. Cells are median milliseconds with the full three-sample range.

| GC workload | 4 KiB baseline | 64 KiB candidate | 256 KiB candidate |
| --- | ---: | ---: | ---: |
| Reclaim 512 KiB after 128 small overwrites | 10.76 (10.74–11.13) | 153.87 (137.65–161.24) | 149.17 (148.56–155.36) |
| Reclaim all but 128 KiB of an 8 MiB file | 24.78 (23.58–27.39) | 23.64 (21.79–24.14) | 23.85 (23.05–24.30) |

The mostly live case copies 7.5 MiB to reclaim 512 KiB. GC is about 14 times
slower in that fixture, so unconditional compaction is not an appropriate
production policy. These are allocated payload contents; database files need
not shrink, because freed SQLite pages remain reusable. The next design needs
bounded read cost under fragmented allocation and a compaction budget or
density threshold before considering a persistent-format migration.

## Validation and retained artifacts

Both candidates pass 144 distinct applicable Python tests, five Release native
suites and the additional GC process-crash test. The 64 KiB first run passed
139 tests; five new checks failed because the subprocess dump helper contained
an incorrectly escaped newline. After fixing that helper, all seven new checks
passed (two overlap the first run). The 256 KiB run passed all 144 together.
Formatting-only native rebuilds produced identical installed module hashes.
ASan/UBSan was not rerun for these isolated prototypes.

The final experiments verify 72 FUSE full hashes, 9 native write readbacks,
9 fresh-read fixtures, 18 fragmented-read fixture/guard comparisons, 36 space
comparisons, 147,456 random slices and 99 SQLite quick checks. The preparation
script reproduces all 13 checked source/test files exactly. An initial
pointer-map-only diagnosis is retained separately. Another read diagnostic
completed with missing host telemetry after its monitor refused to overwrite
the previous log; that attempt is preserved separately and was rerun with a
fresh monitor log. Neither attempt is pooled into the final read results.

[results.json](results.json) contains all final samples, summaries, frozen
identities, limitations and cleanup records. Raw artifacts are under
`vane_fs/build/payload-slice-gc-20261006T155127Z/`. All owned processes and mounts
are stopped; generated databases and temporary test/preparation data are
removed after validation. Sources, binaries, wheels, logs and scripts remain.

## Reproduction

Prepare fresh copies from the same hash-guarded production baseline:

```bash
python3 vane_fs/benchmarks/payload_extents/slice_gc/prepare.py vane_fs/build/slice-gc-reproduction
```

The script first applies the original extent experiment, then the two local
follow-up patches and additional tests. Follow the parent experiment's isolated
Release wheel build/install instructions; use `VANE_FS_EXTENT_BYTES=65536` and
`262144` for the respective candidates. Run their five native suites and copied
Python tests, excluding only the incompatible production-format v1 upgrade
test. The original six extent checks remain, with the partial-GC expectation
updated to require reclamation; seven additional Python cases cover pointer-map
persistence, old experimental databases, shared branch/snapshot references,
transaction rollback, WAL readers and corruption recovery.

`test_gc_crash.cpp` also links against each candidate's `libvane_fs.a` and the
same SQLite static library, with `-Wl,--wrap=sqlite3_step`. Define the same extent
size at compilation. Run it with a new database path; its child exits after
the second successful reference update inside GC. The parent checks complete
rollback, integrity, contents, owner recovery and a successful retry.

Exact build commands, runners, binary/source identities, raw measurements,
host counters, test logs and cleanup manifests are retained in the timestamped
artifact directory identified by `results.json`.
