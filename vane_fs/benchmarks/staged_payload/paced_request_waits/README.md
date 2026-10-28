# Paced-candidate overwrite and small-file wait diagnosis

The observed small-file tails spend about **92% of application time in an
actual foreground WAL `fsync`**. None of the 10,256 recorded random-write,
random-final-fsync or small-file operations overlaps a range-writeback syscall.
The earlier **1.617439-second first post-GC overwrite is not reproduced**:
the four candidate first overwrites take 3.010–3.879 ms. Its original cause
remains unknown; this new window cannot attribute an earlier uninstrumented
sample retrospectively.

This is a diagnosis of the frozen [paced candidate](../paced_range_writeback/README.md),
not another scheduling change or an adoption result. All **nine failed ordinary
FUSE checks remain**. Production format 2, its installed module, the default
SQLite SDK and all durability barriers remain unchanged.

## Matched complete window

The control is the original format-1005 extent implementation without hints;
the candidate retains 64 MiB triggering, claims of at most 4 MiB, a 16 ms
completion interval and pauses during FULL. Both diagnostic builds use the
same existing 28-stage typed recorder and original fsync SQLite SDK. All six
instrumented native files per arm have exact source-reversal proofs. No
scheduling or checkpoint policy changes.

Four fixed AB/BA pairs in each fresh/post-GC state give **16 mounts**. Complete
work includes the original 256+64 MiB writes, final FULLs, reads, 512 random
4 KiB overwrites, 128 create/write/fsync/close operations, namespace work,
maintenance and close. Only the existing light owned-thread CPU/scheduler
sampler runs; the previous high-rate kernel-wait observer is not used.

The 208.443-second window yields **224,607 native roots, 209,899 events and
11,824 application intervals**. Every interval is matched by caller PID,
callback start and logical write range. A previous callback's reply tail
cannot become the next write. Exclusive stage wall/CPU totals partition
each native root; simultaneous background intervals are not added to the
foreground cost. All 32 large-file hashes, retained GC guards, reads,
overwrites, small-file checks, fences and SQLite integrity checks pass.

## First overwrite after GC

| State | Control first-write median/max, ms | Candidate first-write median/max, ms |
| --- | ---: | ---: |
| Fresh | 0.803 / 0.932 | 0.659 / 0.758 |
| Post-GC | 3.350 / 14.613 | 3.272 / 3.879 |

Each post-GC first write contains a foreground WAL `fsync`: **2.254–2.889 ms**
in the candidate, and up to **13.768 ms** in the control. There is no payload
sync, external `BeginBatch`, foreground checkpoint wait or range-call overlap
in these first-write cases. These particular 4 KiB mutations use inline
SQLite payloads; the source's complete 256 KiB external-extent path is not
entered. A WAL sync inside a small mutation must not be labeled an external
payload flush.

This identifies the measured first-write cost, not the reason for the old
1.62-second sample. It also does not establish which internal SQLite WAL
transition initiated a particular syscall; the recorder has no SQLite call
stack or WAL-header event.

## Small-file p99 and rarer tails

Each entry covers the whole create/write/fsync/close operation, pooled over
512 files per arm/state:

| State | Control p99 / maximum, ms | Candidate p99 / maximum, ms |
| --- | ---: | ---: |
| Fresh | 21.929 / 43.496 | 6.451 / 34.397 |
| Post-GC | 6.169 / 33.809 | 6.168 / 34.378 |

The candidate's earlier ordinary post-GC **21.107 ms p99 is not reproduced**.
Its new p99 interpolation brackets are 6.152 and 6.170 ms; those two cases
contain 3.437 and 3.431 ms of WAL fsync, respectively. Native handler CPU is
about 2.03–2.08 ms. The 92% finding below applies to the rarer >=20 ms cases,
not to every small file or to the p99 brackets.

The candidate has **four fresh and three post-GC cases >=20 ms**. Across
these seven cases, foreground WAL fsync consumes **219.166 of 237.798 ms**
of application time, or **92.16%**. No range syscall overlaps any of them.
The post-GC candidate maximum lasts **34.378 ms**, including **31.770 ms**
in its own WAL fsync, with **2.021 ms** native handler CPU. Local publication
wait is at most 3.356 microseconds and checkpoint-mutex wait is zero in its
three post-GC tails. The light scheduler's fully contained samples record
no runqueue wait there; sparse samples do not bound unobserved scheduling
at the interval edges.

All **2,048 small-file operations** have exactly one recorded foreground WAL
fsync apiece and no foreground payload, database or fence sync. The control
also has comparable tails, including 40.387 ms of WAL fsync in a 43.496 ms
fresh operation. Background checkpoint syncs overlap many cases, but that
overlap alone does not prove which kernel or device dependency caused a
foreground sync to wait. The previous checkpoint-pause negative control
remains relevant and is not overturned by this trace.

## Other captured costs and limits

- A fresh control random-phase final fsync lasts **1.109539 seconds**, of
  which **1.109048 seconds** is its own WAL fsync. An independent worker WAL
  fsync lasts 1.429059 seconds concurrently. Native foreground CPU is only
  3.346 ms. The recorder identifies the two calls, not a causal dependency.
- A fresh control random write lasts **175.139 ms**, with **174.734 ms**
  exclusive in `sqlite3_step` and 0.614 ms native CPU. Candidate random
  writes also reach **36.843 ms**, including **34.550 ms** exclusive in
  `sqlite3_step`. These intervals are not local publication/checkpoint-mutex
  waits or range submissions. The current recorder cannot distinguish
  SQLite busy-handler sleeps, internal locks and other time inside `step`.
- A post-GC control 64 MiB final fsync still contains **3.508081 seconds**
  of actual payload fdatasync, despite having no hint worker. Optional
  range submission is therefore not necessary for every long sync tail.
- All **174 candidate range calls** are finite, WRITE-only, at most 4 MiB
  and successful. The minimum same-worker completion-to-next-call interval
  is **16.081 ms**. Their roots hold neither publication nor checkpoint
  mutexes. This checks the observed scheduler contract; it is not a latency
  bound on the device or other coordinators.

No additional tuning follows from this two-arm trace. Next, extend only
the diagnostic recorder with slow SQL identity and SQLite's actual OS sleep
calls, preserving the existing busy handler, to distinguish busy retry from
other `sqlite3_step` waits. For WAL fsync, use targeted read-only wait
attribution rather than another continuously expensive observer. Keep this
candidate isolated and rerun the original ordinary 22-check comparison only
after a concrete mechanism is supported and changed.

## Validation, setup and retained evidence

The recorder contract passes **Release and ASan/UBSan executions**, including
nested/exclusive accounting, real FULL publication, exception unwinding,
threads, VM steps, typed ranges and errno preservation. The **four existing
attribution unit tests pass**, including previous-reply-tail rejection.
Algorithm correctness is inherited from the frozen parent's exact native
source identity; its 153 Python and component suites are not claimed as new
executions. Only related tests run.

An initial runner adapter changed the wrong level of a description-string
replacement. The original attempt and an identical-source capture both fail
before workload generation or mounting. Its original script, build manifest,
proof, stderr and correction record remain retained. Only the Python adapter
is corrected; native libraries are unchanged. The JSON publication adapter
uses compact serialization with parsed-value equality to keep the diagnostic
index below 500 KiB. Frozen measured sources and all raw gzip bytes are
unchanged by formatting the publication index.

[diagnosis.json](diagnosis.json) records every first overwrite, all 47 random
or small-file cases >=20 ms, p99 brackets, disjoint costs and overlapping
background evidence. [final-checks.json](final-checks.json) records input,
default-identity, related source and process checks. [audit.json](audit.json)
indexes all raw requests, roots/events, CPU/host samples, generated scripts,
native sources, binary identities, build/test logs, setup failures and cleanup.
For each entry, concatenate its ordered `retained_as` parts and decompress
once to recover identical bytes. Every retained file is below 500 KiB.

All **18 owned data directories** are removed after their processes stop:
two native contract fixtures and 16 complete mounts. No owned process or mount
remains. Build/tests, measurement and analysis run in separate windows;
cleanup is outside application timers. No cache eviction, CPU/device tuning,
load-based sample removal or unowned process termination occurs.

```bash
here=vane_fs/benchmarks/staged_payload/paced_request_waits
out="$PWD/vane_fs/build/paced-request-waits-repro"
python3 "$here/prepare.py" "$out" \
  --comparison "$PWD/vane_fs/build/paced-range-writeback-20261019-v2/fuse"
python3 -m unittest discover \
  -s vane_fs/benchmarks/staged_payload/fuse_sync_waits -p test_analyze.py
python3 "$here/run.py" "$out"
python3 "$here/analyze.py" "$out"
python3 "$here/summarize.py" "$out"
# Record final source/identity/process checks as "$out/final-checks.json".
python3 "$here/publish_compact.py" "$out"
```
