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

## Follow-up diagnosis: 2026-10-06

The production baseline is now `7cccaf814f`, including new-block batching.
Two isolated candidates tested whether WAL allocation reuse or fewer background
checkpoints could reduce the remaining write cost. Neither was adopted.
Both retained the 16 MiB wake threshold, 64 MiB admission budget, per-request
transactions, and FULL synchronization barriers. Each native diagnostic used
three alternating pairs, fresh databases and a 256 MiB file on ext4, written as
256 pairs of 1,048,528 and 48 bytes. Instrumentation and all samples are retained
in [wal-diagnosis-20261006.json](wal-diagnosis-20261006.json).

| Candidate | Baseline write + Sync, seconds | Candidate, seconds | Result |
| --- | ---: | ---: | --- |
| Retain 64 MiB WAL allocation instead of 16 MiB | 2.605 (2.588–8.107) | 2.570 (2.558–2.662) | Only 1.4% lower median, overlapping ranges |
| Coalesce checkpoint requests for 20 ms | 2.638 (2.528–2.880) | 2.805 (2.756–7.986) | 6.3% higher median despite fewer checkpoints |

The retention candidate lowers foreground truncation time, but WAL traffic
remains 338.421 MiB in the write phase. Coalescing reduces the worker's passive
checkpoints from 15 to 10 in these samples without improving throughput. The
slow samples remain in the results. These small experiments do not justify a
production tuning change.

Page inspection makes the storage cost more concrete. The split writes allocate
65,792 payloads, including 256 superseded partial blocks: 257 MiB of 4 KiB block
contents occupy about 289.19 MiB in the payload table. SQLite record headers and
[overflow pages](https://www.sqlite.org/fileformat2.html#b_tree_pages) prevent
each payload from fitting in one 4 KiB database page. Version tables and indexes
bring the database to 306.17 MiB. One baseline sample submits 338.64 MiB of WAL
writes and 307.94 MiB of database writes across setup, writes, Sync, readback and
close, about 2.53 times the application payload. These are VFS-submitted bytes,
not physical SSD writes. Background and foreground timings overlap.

The unchanged production binary was also remeasured in two independent
three-round FUSE batches, alternating with a separate reference filesystem.
Across six rounds, each system runs first three times. Each batch starts with
fresh workspaces and services; fixtures are deleted between samples, without GC or
global cache eviction. Writes time create, one-MiB application writes, fsync
and close. The read workloads retain their existing hot-cache preparation.

| Latest VaneFS workload | Median MiB/s | Complete range |
| --- | ---: | ---: |
| 64 MiB sequential write + fsync + close | 101.58 | 85.68–105.08 |
| 256 MiB sequential write + fsync + close | 92.71 | 31.74–96.92 |
| Warm 64 MiB sequential read | 754.85 | 703.25–835.54 |
| Warm 4 KiB random read | 67.85 | 67.39–69.00 |

These results describe the latest build in a new measurement window; comparing
them directly with the earlier batching A/B would not establish another speedup.
The 256 MiB writes still include an 8.07-second sample. Baseline CPU idle is
95.6–95.8%; during measurement it is 86.4–86.7%, with 2.4–3.0% I/O wait.
The host is shared, so those counters cannot identify the cause of a stall.
Minimum sampled headroom above the unchanged reference cache guard is 4.56 GiB.

A separate debug-log experiment found that the reference's two-second cache
bypass can change its read path between warmup and measurement. Its initial
DIRECT_IO read issues 129 FUSE READ requests, the subsequent KEEP_CACHE read
populates kernel pages with 512 requests, and later reads issue zero requests.
VaneFS issues 129 requests on every pass. Thus a "hot" label alone does not
establish an equivalent cache state, and the mixed-path read medians are not
stable storage-speed ratios. The debug experiment is separate from the six
uninstrumented rounds; it does not establish which path every earlier sample
took. The artifact retains both systems' samples and this diagnostic evidence.

All 12 native content comparisons, 50 FUSE file hashes, 25 additional remote
hash checks, and 15 SQLite quick checks pass. No production source changed, so
the previously validated component binaries were reused and unrelated tests
were not rerun. All owned processes, monitors, mounts, containers and generated
data were independently checked absent after cleanup. The reference unmount
command again reports a warning while its mount and process do stop; the raw
records retain it. Frozen sources, binaries, logs, host telemetry and complete
cleanup manifests remain in the artifact directories referenced by the JSON.

The subsequent [immutable payload extent experiment](payload_extents/README.md)
tests 64 KiB and 256 KiB BLOBs inside the existing SQLite transaction. Six-round
FUSE measurements improve 256 MiB writes by 13.3% and 16.8%, but random reads
regress by 17.1% and 48.9%, with additional retention of partially live extents.
Neither prototype changes the production format. The experiment retains its
source patches, measurements, read-amplification diagnosis and GC-space checks.
An external payload store would also require an explicit crash-consistency
protocol between content and metadata.

The [slice lookup and GC follow-up](payload_extents/slice_gc/README.md) enables
SQLite pointer maps and atomically repacks surviving slices in isolated
prototypes. Fresh-allocation random reads approach the baseline, and the sparse
case now retains only its 128 KiB of live content. Fragmented page reuse still
amplifies random reads, while reclaiming 512 KiB from mostly live extents makes
GC about 14 times slower in the 8 MiB fixture. Both candidates remain outside
the production format; all measurements and failure/recovery checks are retained.
