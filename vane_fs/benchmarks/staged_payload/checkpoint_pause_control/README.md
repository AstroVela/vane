# Checkpoint pause with resumption and close included

The controlled experiment recommended by the
[range-writeback wait analysis](../range_writeback_waits/README.md) does **not
support pausing the checkpoint worker during the whole FULL-heavy phase**.
Across the extent control and retained-FD range candidate, total elapsed
medians increase **9.1–11.8%**. Pausing wins only **one of 24 paired totals**.
The production scheduler, format, SDK, FULL barriers and 64 MiB admission
budget remain unchanged. A successful evidence check is not a performance
adoption result; the frozen descriptive gain screen fails.

## Controlled workload and timing

Both format-1005 components reuse the exact instrumented core libraries and
SQLite `fsync` SDK from the preceding diagnostic. The probe captures the
original WAL-hook context with a link wrapper and forwards that same callback
and context to SQLite. It stops the original checkpoint worker after common
preparation, then either starts it at the timed phase or leaves it paused.
No checkpoint algorithm, coalescing delay or foreground operation is changed.

Each independent workspace prepares a 64 MiB file and drains its FULL barrier
and payload worker. A committed, backfilled 20 MiB SQLite padding table warms
the existing checkpoint policy. This table is a diagnostic fixture, not file
payload storage. The post-GC state first reclaims **49 versions, 32 payloads
and one snapshot**, preserving different live and snapshot contents. Both
states are prepared and warm; neither represents a fresh FUSE mount.

The timed native workload performs the same 512 seeded, distinct 4 KiB random
overwrites as the preceding FUSE diagnostic, one final FULL, then 128 small
file sequences: lookup, create, 4 KiB write, inode check, FULL, flush check and
release. Open/lookup references and the mount lease follow the FUSE lifecycle,
but no FUSE mount or kernel request/reply cost is measured here. Trace overhead
is common to both arms; these absolute times are not production throughput.

Total timing starts before the active arm's worker start and includes:

1. All random writes and 129 explicit foreground FULL barriers.
2. Worker stop/join, error reporting and WAL status inspection.
3. Worker resumption, completion of all deferred backfill without any new
   commit, and stop/join again.
4. Reference release, session close and the original workspace close barriers.

Preparation is measured separately and also included in the reported lifecycle
duration. Reopening for content/integrity validation and removing owned data
follow total timing. No deferred maintenance is hidden there. Both arms must
finish resumption with all WAL frames backfilled before close.

An identical recorder adds only a payload-worker activity counter, with an
exact reverse proof. It covers a worker that has claimed a range but has not
entered its syscall. After FULL retires pending ranges, preparation waits for
that counter to reach zero. Native source/header and library identities remain
frozen; link maps verify the replaced recorder object is not pulled from the
archive. No range syscall overlaps any measured foreground phase in either arm.

## Complete results

There are six balanced, rotated/reversed four-setting repetitions per state:
**48 serial processes**, with no overlapping owned builds, tests or analysis.
Each table entry is the median of six runs; lower elapsed time is better.
`Aggregate ratio` is total active time divided by total paused time, so a
value above one favors pausing. Every sample and long tail is retained.

| State / component | Foreground active / paused, ms | Total active / paused, ms | Paused total increase | Aggregate ratio | Paused paired wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warm / extent control | 758.5 / 837.9 | 810.6 / 906.5 | +11.8% | 0.900 | 0/6 |
| Warm / range candidate | 771.6 / 824.9 | 813.5 / 892.9 | +9.8% | 0.927 | 0/6 |
| Post-GC / extent control | 794.0 / 844.1 | 837.0 / 913.0 | +9.1% | 1.565 | 1/6 |
| Post-GC / range candidate | 764.6 / 819.4 | 808.6 / 884.7 | +9.4% | 0.909 | 0/6 |

The post-GC extent aggregate favors pausing because one active total reaches
**4.803664 s**, compared with a usual approximately 0.8 s. It is preserved,
not trimmed. Its opposite median result and one paired win do not support
a general speedup. The frozen screen required at least 1.10 aggregate gain
and five of six wins for both foreground and total in every group.

Paused resumption medians are **23.4–24.1 ms**, compared with **5.7–7.3 ms**
active; paused close medians are **44.0–44.8 ms**, compared with
**36.1–37.1 ms** active. Those costs are already inside total. The physical
WAL high-water mark reaches **39,473,752 bytes (37.65 MiB)**, below the unchanged
64 MiB limit. Paused phases leave 9,552 or 9,581 frames with zero backfilled;
all resume successfully without another commit. Active phases keep making
progress, with their actual frame counts retained.

## Final FULL and long tails

The random-write final FULL becomes consistently slower when the worker is
paused. Median root and own WAL syscall times are:

| State / component | FULL active / paused, ms | Own WAL fsync active / paused, ms |
| --- | ---: | ---: |
| Warm / extent control | 4.696 / 58.080 | 4.385 / 57.758 |
| Warm / range candidate | 5.182 / 64.981 | 4.882 / 64.687 |
| Post-GC / extent control | 5.506 / 66.729 | 5.198 / 66.452 |
| Post-GC / range candidate | 4.633 / 58.246 | 4.279 / 57.892 |

The added typical barrier cost is inside the foreground's own WAL syscall.
Pausing eliminates worker overlap but also changes backfill batching, WAL
generation reuse and how much work accumulates before FULL. This does not
isolate a kernel concurrency mechanism or justify treating every overlapping
checkpoint as harmful. Some small-file p99s improve with pausing; the full
random-plus-small workload and total still regress in every group by median.

The slow active post-GC extent run has a **1.478802 s** random final FULL,
including **1.478447 s** in its own WAL fsync. A worker WAL fsync lasts
**1.445894 s**, overlapping the FULL by **1.255797 s**. Across its foreground
phase, all foreground WAL fsyncs total **4.032058 s (84.8%)**. Intersecting
worker and foreground intervals must not be added as wall time.

The paused post-GC extent run at repetition three also has a **247.657 ms**
foreground WAL fsync with no concurrent checkpoint worker. Its total reaches
**1.157771 s**. All **15 sync intervals at least 100 ms** remain, including
background intervals. Worker overlap is neither sufficient evidence of the
cause of the active tail nor a prerequisite for a tail.

Preparation is not uniformly fast either: five preparations exceed one second,
including tails up to **7.495126 s** in the extent control and **6.873529 s**
in the range candidate. They remain in the complete per-run lifecycle results;
none is removed or attributed solely to range submission. Prepared-phase
results do not settle cold workspace or first large-write performance.

## Validation, preflight and retained evidence

Only related native checks run: **16 workload probes and four fault probes**,
split equally between Release and ASan/UBSan. They check both components,
both states and both arms; paused FULL sync errors and retry, worker write
errors, preserved progress on restart without a commit, busy close and sync
failure followed by close retry. Every workload checks the complete overwritten
64 MiB file, all 128 small files, post-GC live/snapshot contents and SQLite
integrity after reopen. Existing algorithm suites are reused by exact source
and library identity; no broad base/Ray suite runs.

All **6,192 explicit foreground FULL roots** retain successful WAL fsyncs.
There are **6,201 WAL fsync calls inside those roots** and **6,264 foreground
WAL fsync calls including other operations**. A FULL can issue multiple WAL
syncs; the counts deliberately distinguish barriers from syscalls. Every
actual sync/range call succeeds in all measured traces. **77,986 roots and
130,383 events** pass exclusive wall/CPU accounting.

The first preflight incorrectly required exactly one WAL syscall per FULL.
Its third native workload completed successfully, but one FULL issued two WAL
fsyncs, so the analyzer rejected it. The corrected check requires successful
WAL fsyncs for every FULL root and counts all syscalls independently. All
three preflight cases, original matching source bytes, traces and logs remain;
the original source hashes are verified when restoring those bytes. The
subsequent complete 20-case validation finishes before the measurement plan
is frozen. No workload, sample or native algorithm is changed by this repair.

[findings.json](findings.json) contains all per-run and group results;
[attribution.json](attribution.json) reconciles final barriers and tails;
[plan.json](plan.json) preserves order and input identities; and
[final-checks.json](final-checks.json) records source checks and the unchanged
default installation. [audit.json](audit.json) indexes every retained original
byte stream, with original sizes/hashes and ordered compressed chunks. For
each entry, concatenate its `retained_as` files, gzip-decompress once, and
verify the listed original size and SHA-256. No large database/payload output
is retained. All **71 owned data directories** are removed only after their
processes stop; no owned mount or measurement process remains. Git artifacts
are individually below 500 KiB.

CPU samples and host counters remain unfiltered. There is no cache eviction,
affinity/frequency/device tuning, load-based selection or unowned process
termination. The earlier format-1004
[pause diagnostic](../full_sync_contention/README.md) also had inconsistent
results; its timing excluded resumed maintenance and close. The present
format-1005 result includes those costs and likewise does not support adoption.
The original ordinary FUSE range-candidate result remains **seven failed gates
out of 22**; this native diagnostic does not replace or clear them.

## Next step and reproduction

Keep bounded checkpoint progress. Prioritize the separate cold OPEN-batch
path identified by the preceding trace: measure the first WAL-prefix FULL,
fence persistence and exact bytes per syscall before attempting a narrowly
scoped change. Preserve recovery ordering and all required durability barriers.
Any eventual coordination or range candidate must still pass the unchanged
ordinary FUSE gates in fresh and independently prepared post-GC states.

```bash
here=vane_fs/benchmarks/staged_payload/checkpoint_pause_control
out="$PWD/vane_fs/build/checkpoint-pause-repro"
python3 "$here/build.py" "$out" \
  --parent "$PWD/vane_fs/build/range-writeback-waits-20261018"
python3 "$here/run.py" "$out"
python3 "$here/analyze.py" "$out"
python3 "$here/attribute.py" "$out"
# Record final source/identity/cleanup checks as "$out/final-checks.json".
cp "$out/attribution.json" "$here/attribution.json"
python3 "$here/publish.py" "$out"
```

The optional build argument `--preflight` points to the retained original
preflight directory when publishing this run's complete history. New
reproductions need not manufacture a failed preflight.
