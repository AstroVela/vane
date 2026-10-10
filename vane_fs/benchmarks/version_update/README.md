# Version-row UPDATE evaluation: 2026-10-10

Neither candidate is adopted. Replacing DELETE/INSERT with UPDATE reduces
SQLite work, but the complete fresh and post-GC comparisons retain regressions.
These experiments precede the [partial-write batching evaluation](../partial_write/README.md).

The production baseline is `d31b0a151468b6d68912e2e406f2eb9892e8663d`.
The reused control was built at `c6b7d5c0df5c319dbda7872a2e7fcda7e1bd55fc`;
the component's native sources are identical. Both candidates keep format 2,
4 KiB SQLite BLOBs, the original SQLite 3.53.2 SDK and all durability settings.

`all-versions.patch` updates a row when the only overlapping version has exactly
the replacement interval. `live-blocks.patch` restricts that optimization to an
existing, non-deleted block; inode updates and tombstone transitions keep the
original path. Each isolated, non-editable Release build passes 132 related
Python tests and eight C++ suites, including fault rollback in strict and fsync
modes. The baseline fails each new UPDATE-path assertion as expected.

## What the diagnostics establish

With 512 seeded 4 KiB overwrite requests, including two unchanged blocks, the
fresh strict native probe produces the following WAL bytes. These instrumented
measurements are separate from plain FUSE timing.

| Variant | WAL MiB | Wrapped SQLite VM steps |
| --- | ---: | ---: |
| Control | 21.076 | 218,312 |
| All versions | 16.832 | 182,540 |
| Live blocks only | 18.844 | 198,932 |

The version tables have secondary indexes on their key columns and `high`.
An UPDATE can preserve those unchanged index entries. This is a code-level
explanation for reduced work, not a measured attribution of every WAL page.
Fresh 256 MiB sequential-write WAL traffic falls only about 0.9% and 0.3%,
respectively. Both controls and candidates spill 605 cache pages during the
native GC fixture. A follow-up instrumented FUSE window did not reproduce all
earlier stalls; the original slow samples remain in the results.

## Complete workload results

Each candidate has its own window: six alternating control/candidate pairs on
fresh mounts and six on independent fragmented mounts, all using `fsync`
durability. The fragmented fixture writes a 32 MiB guard, zeroes alternate
4 KiB blocks, then collects garbage through an independent strict workspace.
Each mount measures 64/256 MiB writes with fsync and close, warm reads, seeded
random reads/overwrites, per-file fsync and directory operations with a final
directory fsync. All requested bytes, guard hashes and SQLite quick checks pass.
No owned builds or tests overlap measurement. Controls from different windows
must not be pooled.

| Window / workload | Control median | Candidate median | Control combined | Candidate combined |
| --- | ---: | ---: | ---: | ---: |
| All versions / post-GC 64 MiB write, MiB/s | 96.50 | 65.68 | 96.19 | 33.00 |
| All versions / post-GC 256 MiB write, MiB/s | 94.70 | 94.22 | 71.21 | 46.07 |
| Live blocks / fresh small-file fsync, files/s | 247.20 | 142.05 | 65.29 | 44.34 |
| Live blocks / post-GC 256 MiB write, MiB/s | 94.34 | 58.05 | 94.46 | 38.84 |

Combined rate is total completed work divided by total elapsed time. Six runs
do not establish a stable tail distribution or prove the cause of every stall.
The first candidate has three post-GC collection samples near 340 ms versus
roughly 40 ms in its control; its separate diagnostic window does not reproduce
that difference. Narrowing UPDATE removes that observed GC pattern, but three
of its six post-GC 256 MiB writes are still slow. Reduced VM/WAL work alone does
not justify either change.

[Results](results.json), per-variant summaries and raw plain measurements retain
all samples, complete lifecycle/GC times, validation hashes and cleanup records.
The artifact manifest indexes frozen sources, binaries, host counters, full
SQL/VFS diagnostics and logs. Failed diagnostic attempts are retained separately:
an initial native fixture omitted its mount lease, and an initial GC JSON
encoder mishandled a multiline SQL string. Corrected reruns are not pooled with
those failures. Generated databases and test directories were removed after
their processes stopped; performance evidence remains.

Use the [shared preparer](../partial_write/prepare.py) with
`--variant all-versions` or `--variant live-blocks` to reconstruct the complete
candidate and regression-test sources. It checks the exported baseline and
patched source hashes before reporting success.
