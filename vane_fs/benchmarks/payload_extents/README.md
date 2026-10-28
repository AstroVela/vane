# Immutable payload extent experiment: 2026-10-06

The 64 KiB and 256 KiB C++ prototypes improve 256 MiB sequential writes, but
regress reads and retain dead slices of partially live extents. Neither is
adopted into the production format. The patch and preparation script are
research artifacts; the normal component build still uses `src/workspace.cpp`
with 4 KiB payloads.

The [2026-10-07 slice lookup and GC follow-up](slice_gc/README.md) reclaims dead
slices and improves random reads on fresh allocation. It also demonstrates
that fragmented free-page reuse restores read amplification and that aggressive
compaction makes mostly live GC substantially slower. The candidates remain
experimental; the original measurements below are preserved.

## Design

The baseline is `82ce177d10`, whose production storage code is `7cccaf814f`.
The prototype keeps 4 KiB logical blocks, version intervals, per-request
transactions, immediate cross-connection visibility and existing synchronization
and checkpoint policies. Vacant writes concatenate up to 16 or 64 nonzero
blocks into one immutable BLOB. Each block version references a payload ID and
byte offset. Occupied and partial writes retain the existing 4 KiB fallback.
Small changes therefore append small payloads instead of rewriting an extent.

Reads use the read-only [incremental BLOB API](https://www.sqlite.org/c3ref/blob_read.html)
to copy the requested slice. Handles are closed within the enclosing read
transaction; no cached handle can retain an old snapshot across operations.
Merge comparisons and publication include both payload ID and slice offset.
GC removes an extent only when no version refers to any of its slices.

Experimental databases use application ID `0x56465831` and format version
1001. Production and prototype binaries reject each other's databases before
changing them. There is no migration or compaction implementation.

## Same-window FUSE results

All cells are median MiB/s followed by the full six-sample range. No samples
are filtered. Each of the six order permutations runs once; every sample uses
a fresh database and mount. Both prototypes and the baseline use fsync mode,
live direct I/O, disabled kernel writeback, a 60-second metadata cache, and
the existing 16 MiB checkpoint wake / 64 MiB admission / 16 MiB WAL retention
policy. No owned tests or builds overlap measurements.

| Workload | 4 KiB baseline | 64 KiB prototype | 256 KiB prototype |
| --- | ---: | ---: | ---: |
| 64 MiB write + fsync + close | 97.17 (94.89–101.66) | 101.24 (54.30–106.58) | 100.08 (12.46–103.60) |
| 256 MiB write + fsync + close | 94.95 (81.69–96.33) | 107.60 (104.89–110.38) | 110.92 (109.56–115.61) |
| Warm 64 MiB sequential read | 517.51 (514.91–569.97) | 470.92 (452.78–494.32) | 466.53 (453.73–474.07) |
| Warm 4 KiB random read | 68.04 (65.38–70.98) | 56.38 (52.99–58.05) | 34.78 (33.55–37.03) |
| 4 KiB random overwrite + final fsync | 14.37 (12.08–15.17) | 11.96 (9.96–15.15) | 11.76 (8.20–14.32) |

The 256 MiB write medians improve by 13.3% and 16.8%, with separated ranges
in this window. Random reads regress by 17.1% and 48.9%, also with separated
ranges. Sequential reads regress by about 9–10%. The smaller write and random
overwrite ranges overlap; their median differences are not stable guarantees.

Writes use one-MiB application calls, timing create, writes, fsync and close.
The read fixture also uses one-MiB writes so FUSE splitting does not route
nearly every preparation request through the occupied-range fallback. After
a complete hash/warm pass over a 64 MiB file, sequential reads use one-MiB
calls, followed by 4,096 seeded 4 KiB reads. Then 512 distinct seeded offsets
are overwritten and fsynced; that timer excludes close. Full hashes and all
random slices are checked outside measurement. Fixtures are deleted between
workloads, with no GC or global cache eviction.

One earlier attempt was interrupted to correct the read fixture's application
write size. Its samples, original runner, reason and cleanup manifest remain
separate in `attempt-1`; they are not pooled into these six rounds.

The host remains shared. The 30-second baseline has 95.78% CPU idle and a
one-minute load median of 2.81. Measurement and validation have 92.53% CPU idle,
1.15% I/O wait and a load median of 2.58. Minimum pre-mount free space is
26.03 GiB. These counters do not attribute individual stalls to other activity.

## Write volume and read amplification

A separate instrumented native experiment writes 256 MiB as 256 pairs of
1,048,528 and 48 bytes, reproducing the FUSE request split. Three rotating
rounds use fresh databases. The following bytes include setup, writes, Sync,
readback and close; foreground/background times overlap.

| Native metric | 4 KiB baseline | 64 KiB prototype | 256 KiB prototype |
| --- | ---: | ---: | ---: |
| Payload rows | 65,792 | 4,352 | 1,280 |
| Payload contents, MiB | 257 | 257 | 257 |
| WAL writes, MiB | 338.64 | 307.66 | 305.53 |
| Database writes, median MiB | 307.94 | 277.60 | 276.01 |
| Combined writes / 256 MiB | 2.53 | 2.29 | 2.27 |

The extra logical MiB is 256 superseded partial blocks. Reducing payload rows
does not remove the 4 KiB version records or the 512 write commits. WAL bytes
fall by 9.1% and 9.8%; the larger candidate saves little more than 64 KiB.
Native write + Sync samples are 2.593/2.593/2.622 seconds for the baseline,
2.297/2.766/7.451 for 64 KiB and 2.258/7.246/2.236 for 256 KiB. The outliers
remain; checkpoint stalls are still possible.

A second native diagnostic measures 4,096 seeded 4 KiB reads after a full
warm pass over a freshly written 64 MiB file. The application reads 16 MiB.
Three rotating rounds reproduce the same VFS call counts:

| Random-read diagnostic | 4 KiB baseline | 64 KiB prototype | 256 KiB prototype |
| --- | ---: | ---: | ---: |
| SQLite VFS read calls | 10,261 | 40,607 | 136,827 |
| VFS read bytes, MiB | 38.21 | 157.14 | 533.32 |

Copying only the requested slice does not imply reading only those bytes
through the VFS. SQLite represents large record payloads with a linked chain
of [overflow pages](https://www.sqlite.org/fileformat2.html#cell_payload_overflow_pages).
The measured amplification is consistent with extra work locating slices in
those chains with newly opened cursors. The experiment does not separately
time every internal B-tree operation. These VFS counters describe software
reads/writes, not physical SSD traffic; the underlying OS page cache is active.

## Snapshot and GC space

An 8 MiB file is snapshotted, then 128 separate 4 KiB blocks are changed. All
variants append exactly 512 KiB of payload contents. With the snapshot retained,
all retain 8.5 MiB. After dropping it and collecting garbage, the baseline
returns to 8 MiB; both prototypes retain 8.5 MiB because other slices still
refer to each old extent.

A separate sparse case keeps one original 4 KiB slice per 256 KiB and zeroes
the rest of an 8 MiB file. After dropping its snapshot and collecting garbage:

| Retained payload contents | 4 KiB baseline | 64 KiB prototype | 256 KiB prototype |
| --- | ---: | ---: | ---: |
| 128 KiB of surviving nonzero data | 128 KiB | 2 MiB | 8 MiB |

These are allocated payload contents, not database file lengths. Freed SQLite
pages can remain on the freelist. Deleting the last live references releases
all payload rows in every variant; the issue is partially live extent retention.
Snapshot and live contents are validated at each transition.

## Validation and reproduction

Both sizes pass five Release native suites and 137 distinct applicable Python
tests, including real FUSE and six new extent checks. The 64 KiB run first
passed 135 tests; after correcting the independent-payload corruption fixture
and adding the format guard, three affected checks passed, one overlapping
the initial passes. The 256 KiB final run passes all 137 together. The legacy
v1 upgrade test is excluded because this prototype intentionally rejects the
production format. The fixture patch also moves payload/version rollback
injection from the 100th row to the second, since packing reduces row counts.

Coverage includes slice boundaries, sparse and unaligned writes, small-write
copy-on-write, snapshots, branch merges across different payload layouts,
truncation, concurrent writers, allocation limits, SQL failure rollback,
process-crash recovery, checkpoint and durability behavior, corruption recovery,
GC lifetime, and bidirectional format refusal. New runs use Release builds;
ASan/UBSan was not rerun for this isolated prototype.

The completed experiments validate 72 FUSE full-file hashes, 18 native full
readbacks, 110,592 random-read slices, 12 space-test full comparisons and 54
SQLite quick checks. Owned mounts and processes are stopped and generated
databases are removed after inspection/hashing. Sources, binaries, wheels,
runners, logs, host counters, failed attempts and cleanup manifests are retained
under `vane_fs/build/payload-extents-20261006T150952Z/`.
[results.json](results.json) contains every final sample, summary, identities,
limitations and cleanup records. The raw artifact keeps individual latencies
and full instrumentation counters.

To create fresh source trees from the pinned baseline:

```bash
python3 vane_fs/benchmarks/payload_extents/prepare.py vane_fs/build/extent-reproduction
```

Build each copied component independently using its existing CMake/PEP 517
workflow, a persistent incremental build directory and a non-editable wheel.
Use `-DVANE_FS_EXTENT_BYTES=65536` or `262144` in `CMAKE_CXX_FLAGS` for the
corresponding prototype; the baseline has no extra definition. Enable
`VANE_FS_BUILD_TESTS`, `VANE_FS_BUILD_TOOLS` and `VANE_FS_BUILD_FUSE`. Use the
existing SQLite SDK and libfuse3 development package. Install each wheel into
an isolated test environment so `python -I` child processes load the same
variant. Do not replace the regular installed component with the prototype.

Run `ctest --test-dir <build> --output-on-failure`. For Python tests, set
`VANE_FS_MOUNT_BINARY` to that variant's mount executable,
`VANE_FS_EXTENT_BYTES` to its selected size and `VANE_FS_BASELINE_CLI` to the
baseline CLI, then run the copied component tests with
`-k 'not test_v1_upgrade_preserves_unverifiable_legacy_pins'`. The precise build
commands and benchmark runners used here are frozen in the raw artifact.

The next design needs cheaper random slice lookup and reclamation of partially
live extents before a persistent-format migration is justified. Increasing
extent size alone does not meet those requirements.
