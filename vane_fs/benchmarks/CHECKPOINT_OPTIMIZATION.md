# Background WAL checkpoints: 2026-10-06

Fsync mode now moves routine WAL checkpoints onto a separate connection and
worker. A commit wakes the worker once committed WAL frames reach 16 MiB.
Before another mutation, a 64 MiB admission budget requires a restart checkpoint
so sustained writes cannot indefinitely outrun maintenance. Strict mode retains
its 4,096-page automatic checkpoint and FULL commits.

The budget includes page-size-dependent frame headers. It is not an absolute
file-size quota: one atomic transaction can exceed it, concurrent writers can
race admission, and strict or external connections retain their own policies.
On reuse, `journal_size_limit` reduces retained allocation to 16 MiB. A pinned
SQL reader can prevent WAL reuse even after every frame has been backfilled;
admission therefore checks total committed frames, not just outstanding copies.

Ordinary fsync-mode commits remain immediately visible under NORMAL. Explicit
FULL barriers still persist preceding commits and can run under admission
pressure. Synchronous handle writes undergo normal admission. The commit hook
always returns success because the transaction has already committed; background
errors instead reach a subsequent mutation, barrier or explicit close. Failed
close preserves a usable worker and connection for retry.

Waiting for this connection's running checkpoint I/O is outside the SQLite busy
timeout. A preliminary implementation incorrectly timed that wait: a 7.48-second
checkpoint exceeded the 5-second timeout and caused a FUSE short write. A
deterministic slow-I/O test fails on that implementation. The corrected version
waits for local storage I/O before starting the SQLite lock-contention deadline;
it retains the same 64 MiB budget. Long readers and competing checkpoint/writer
locks still produce retryable Busy errors before mutation. The failed run and
instrumented reproduction are retained in ignored build artifacts.

## Measurement method

The baseline is `f40c57742e8b755a2ab00c410ef50209a35e0c77`. Both variants use
`--durability=fsync`, live direct I/O, disabled kernel writeback and a 60-second
metadata cache. Release binaries, SQLite and libfuse are unchanged throughout
the reruns; production source and executable hashes were checked against the
tested build. Only checkpoint scheduling, admission and retained WAL allocation
differ between the VaneFS variants.

The replacement comparison consists of two independent three-round batches.
Each starts with fresh owned databases and mounts; all six order permutations
across the VaneFS variants and a separate reference system are exercised across
the batches. No samples are filtered. The VaneFS results are reported here.

Writes create a 64 or 256 MiB file through one-MiB application writes, with
open, fsync and close inside the timer. A short write fails the experiment.
Random reads issue 4,096 fixed-seed 4 KiB requests against a warmed 64 MiB file.
Preparation, full size/SHA256 validation, deletion and shutdown are outside
timers. No global caches are dropped, and no owned builds or tests overlap
measurement. One-second CPU, disk, memory and pressure counters include a
30-second idle baseline before each batch. The host remains shared.

WAL allocation is sampled after each application write. This can miss a peak
within a split FUSE request and is not a count of uncheckpointed frames.

## Six-round rerun

Each cell gives the median followed by the complete sample range. All values
are MiB/s; the write rows include fsync and close.

| Workload | Before | After |
| --- | ---: | ---: |
| 64 MiB sequential write | 59.215 (54.567–61.921) | 87.447 (85.112–93.813) |
| 256 MiB sequential write | 30.205 (15.658–61.689) | 88.061 (84.220–89.024) |
| Warm 4 KiB random read | 66.773 (62.473–69.446) | 68.426 (65.650–69.712) |

The 64 MiB write median improves by 47.7%, with separated sample ranges.
The larger write also improves, but its old-binary median includes several
stalls: the six samples are 61.69, 33.71, 19.23, 52.25, 26.70 and 15.66 MiB/s.
The median ratio is therefore not a stable 2.9-times speedup. Even the slowest
new sample, 84.22 MiB/s, exceeds the fastest old sample, 61.69 MiB/s. Random-read
ranges overlap; the 2.5% median difference does not establish a read improvement.

The largest observed individual application-write latency is 2.97 seconds
before and 0.266 seconds after in these batches. This is a finite observation,
not a latency guarantee: the earlier retained run includes a 5.47-second write
on the same corrected executable. Slow storage can still delay admission.
Peak sampled WAL allocation increases from 30.36 to 64.002 MiB. Successful
unmount times are 67.5/518.0 ms before and 66.3/268.3 ms after, across the two
batches; these two shutdown samples do not establish a performance trend.

The two 30-second baselines have about 96% CPU idle time and 1-minute load
medians of 1.47 and 2.20. During the comparisons, CPU idle averages about 87%,
I/O wait averages 4.2–4.4%, and 1-minute load ranges are 1.81–6.54 and 2.68–8.28.
Those counters include the benchmark itself and unrelated host activity. They
do not prove which activity caused any individual stall. The minimum sampled
free-space margin above the reference client's unchanged guard is 3.27 and
3.25 GiB in the respective batches.

All six samples, per-write aggregate timings, source/binary hashes, host-load
summaries and cleanup records are in
[checkpoint-rerun-20261006.json](checkpoint-rerun-20261006.json). The two complete
batches validate 36 VaneFS large-file hashes and four SQLite quick checks. All
owned mounts, processes, containers and generated data were independently
verified absent after cleanup. The reference client's unmount command reports
a warning in each batch even though its mount and process stop; the raw cleanup
records preserve those warnings.

## Correctness and retention

All 126 related component Python tests pass, including the real FUSE cases.
All five native tests pass in Release and ASan/UBSan; 13 selected fsync FUSE
tests also pass with the sanitized executable. New or extended coverage checks:

- Background progress, idle completion, sustained-write admission and complete
  readback, including concurrent fsync writers.
- Pinned SQL readers, fully backfilled but unreusable WAL, 4 KiB and 8 KiB page
  budgets, Busy before mutation, and FULL barriers under admission pressure.
- Injected background I/O and worker-startup failures, owner recovery, failed
  close retry, and maintenance resuming without another commit.
- Local checkpoint I/O longer than the busy timeout, inherited worker state
  after fork, and SIGKILL while a checkpoint is copying database pages.
- File/directory fsync, fdatasync and O_SYNC/O_DSYNC recovery after crossing the
  background threshold. These tests do not simulate a physical power cut.

The original three-round comparison is retained in
[checkpoint-optimization-20261006.json](checkpoint-optimization-20261006.json).
It includes a 5.47-second application write and large baseline write outliers.
A subsequent attempt to keep all six rounds in one set of databases completed
51 of 54 samples before the separate reference client's 1% free-space guard
rejected fixture preparation. That run remains marked incomplete; its samples
are not merged into the two replacement batches. The replacement batches keep
that guard unchanged and recreate data between batches to limit accumulation.

Full raw samples, host counters, frozen binaries, runner sources, logs, failed
attempts and cleanup manifests remain in ignored build directories. Generated
databases and service data are removed only after validation and all owned
processes stop. No source dataset or unrelated host service is removed.
