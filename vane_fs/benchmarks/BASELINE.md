# Native baseline: 2026-10-05

The C++ core completed all 18 runs: three profiles, three repetitions each,
before and after adding the payload-reference index. Every run validated file
contents, parent/child isolation, zero file-row copying on fork, concurrent
writer results and retained data after GC. These are local observations, not
FUSE throughput or a production capacity guarantee.

## Configuration

- Intel Xeon E5-2686 v4 at 2.30 GHz, 36 logical CPUs, approximately 62.7 GiB RAM;
  Linux 6.11, ext4, GCC 13.3.0, CMake Release, SQLite 3.53.2.
- Warm OS cache, no cache eviction, `synchronous=FULL`, default WAL
  autocheckpointing. No other builds or test suites were started during the
  measured profile runs; the host was not otherwise isolated.
- Small-file profiles contain 1,000 or 10,000 files of 4 KiB plus one 1 MiB
  file. The large-file profile contains 100 small files plus one 64 MiB file.
- Each run creates 64 sibling branches and a chain of 16 branches, then tests
  100 repeated block updates and four independent connections writing 100
  disjoint-block updates each.

Latency percentiles below pool the three runs. Each fork/first-write column
has 192 samples per profile; lookup has 300. The percentile index is
`floor((n - 1) * p)` in sorted samples. GC has only three samples, so its
individual timings are retained and reported separately.

## Final results

All latencies are milliseconds, shown as p50 / p95.

| Profile | Sibling fork | First write after fork | Directory lookup | Four-writer operation |
| --- | ---: | ---: | ---: | ---: |
| 1,000 small files | 2.44 / 3.99 | 3.24 / 29.87 | 0.35 / 0.40 | 2.92 / 66.06 |
| 10,000 small files | 2.52 / 3.09 | 3.20 / 3.57 | 0.36 / 0.40 | 2.81 / 3.09 |
| 64 MiB file | 2.35 / 2.54 | 3.18 / 3.81 | 0.34 / 0.41 | 2.82 / 9.46 |

In every run, 64 sibling forks added **zero inode, directory-entry, block-version
or payload rows**. The first changed block in each child added one payload,
for 64 new payloads total. All parent files retained their original contents.

For the 64 MiB profile, the pooled sequential-read median was 423.81 ms
(approximately 151 MiB/s). The three complete-file writes measured
31.18–33.45 MiB/s. These operations call the native API directly.

Tail latency remains material: the largest contended write was 2.58 seconds,
and a small-file creation reached 0.96 seconds. SQLite serializes writers;
these runs do not establish a latency bound. The index adds write maintenance,
and these observations do not prove that write or read throughput improved.

## GC investigation and correction

The original collector checked every payload against an unindexed reference
column. SQLite 3.53.2 reported a correlated `SCAN block_versions` for each
payload. Adding `block_payload_references ON block_versions(payload)` changed
that step to a covering-index lookup. The outer payload/version scans remain.

| Profile | Before: median GC | After: median GC | After: min–max |
| --- | ---: | ---: | ---: |
| 1,000 small files | 198.61 ms | 16.00 ms | 15.95–16.52 ms |
| 10,000 small files | 4,537.99 ms | 72.13 ms | 69.31–82.61 ms |
| 64 MiB file | 12,404.65 ms | 60.21 ms | 57.99–61.90 ms |

Each run reclaimed 259 version rows, 560 payloads and 80 internal snapshots,
while preserving all live content. Page reuse does not imply a smaller main
database file. Physical WAL lengths and database/page counts are retained in
the measurements; checkpoint backlog was not measured.

## Reproduction and retained evidence

Build the Release tools using the [component instructions](../README.md), then
run from the repository root:

```bash
python vane_fs/benchmarks/run.py \
  --binary vane_fs/build/core/vane-fs-benchmark \
  --output vane_fs/build/benchmarks --repeat 3
```

[Machine-readable results](baseline-20261005.json) include both code identities,
all source and binary hashes, frozen configurations, pooled latency statistics,
per-run throughput, GC samples, physical row/page counts and artifact hashes.
The base commit was `3095a34737d991097ed5f9737b89761736678050`; VaneFS was an
uncommitted addition. The only native source change between these runs was the
index statement in `src/workspace.cpp`.

Full operation samples, logs and cleanup manifests remain locally under
`vane_fs/build/benchmarks-20261005/` and
`vane_fs/build/benchmarks-20261005-indexed/`. All generated databases were hashed
after their owning processes exited, then removed outside measured windows:
901,128,192 bytes across 18 runs. The cleanup manifests record the removed paths
and content hashes; approximately 109 GiB remained free after the final run.

The final indexed implementation passed both native CTest cases, the same cases
under ASan/UBSan, and all 58 installed-package tests including actual FUSE
mounts and local Vane CSV/Parquet queries. The worktree's Vane release launcher
also passed: 1,833 passed and 82 skipped across its separate shards. CI is
configured and its workflow validation passed locally; no hosted CI run is
claimed here.
