# Execution benchmarks

P5.2.2 measures the supported analytical execution paths through `query()`.
The runner is separate from [correctness acceptance](EXECUTION_ACCEPTANCE.md):
performance values are observations, never CI pass/fail thresholds. It does not
change production resource defaults or select a backend automatically.

## Reproduce

Build and install a matching, non-editable wheel using
[DEVELOPMENT.md](DEVELOPMENT.md#incremental-package-build), then run:

```bash
python -I scripts/benchmark_execution.py \
  --output "$PWD/build/execution-benchmark-100k" \
  --rows 100000 --seed 970 --repetitions 3 --warmups 1
```

The output directory must be new. Use another directory for each run. `-I`
keeps the source package from shadowing the installed wheel. The CLI owns a
local Ray cluster with `worker_count * worker_threads` CPUs and shuts down only
that cluster. Ray uses its normal object-store sizing. The lower-level
`run(Configuration(...))` function uses an already initialized cluster, which
allows the test suite to reuse its shared-cluster fixture.

The default uses two workers, one thread per worker, two source partitions and
public batches of at most 2048 rows. Local queries use the same total native
thread count and a memory limit equal to the aggregate per-query Ray operator
allowance. The driver, worker processes, result service and their memory domains
remain different; this is not a comparison with identical process overhead.
All admission, execution and delivery deadlines are finite (`--deadline`).

For a local-only run or a different data scale:

```bash
python -I scripts/benchmark_execution.py \
  --output "$PWD/build/execution-benchmark-local" \
  --rows 1000000 --modes local --scenarios cold warm slow
```

The FTE store is a dedicated directory inside the output directory. This local
mount survives an actor loss in this benchmark; the benchmark does not qualify
a production storage failure domain or a multi-node deployment.

### Shared Runtime and Flight comparison

`--interface runtime` (default) opens sessions through the application Runtime.
`--interface flight` hosts a Server and opens real Flight client sessions. SQL,
query controls and native result batches use the public network path. Server
internals are observed only to correlate worker query IDs, inject faults, and
inspect accounting outside latency samples. The Flight client and Server run in
the same process over loopback; this is not a WAN or isolated-client benchmark.
Local-mode reference queries always remain embedded.

Run the same configuration separately for each interface:

```bash
python -I scripts/benchmark_execution.py \
  --output "$PWD/build/server-benchmark-runtime" --interface runtime \
  --rows 32768 --repetitions 3 --warmups 1 --modes pipelined fte \
  --consumer-rows-per-second 8192
python -I scripts/benchmark_execution.py \
  --output "$PWD/build/server-benchmark-flight" --interface flight \
  --rows 32768 --repetitions 3 --warmups 1 --modes pipelined fte \
  --consumer-rows-per-second 8192
```

Flight cold samples include new public listeners, a client session and a worker
pool. Warm samples reuse the Server and pool; mixed samples use two client
sessions on that Server. Client sequence numbers are mapped to the server's
worker query IDs before testing reservation overlap. Submission timestamps alone
are never accepted as proof. Flight resource snapshots separate the client Arrow
view budget, server query accounting and gateway links. Tokens and tickets are
not written to benchmark artifacts.

## Data, queries and capacity profiles

The runner creates four Parquet files with deterministic integer keys, nullable
integer values and fixed-width strings. Input generation is bounded by a 65536
row batch. The seed, generator revision, row count, ordered file list, compressed
sizes and SHA-256 hashes are retained. Every workload has a standalone SQL file.

The workloads are `tiny` (`SELECT 42`), a streaming filtered scan, a grouped
COUNT/SUM and an equality join followed by TopN. They exercise startup, output
transport, shuffle/aggregation and blocking operators. The generated dataset
is intentionally small enough for repeatable development runs; vary row counts,
keys, types and deployment topology before drawing production conclusions.

Two Ray profiles use the same worker, operator, delivery and storage capacities:

| Profile | Channel window | Maximum frame | Frame rows | Frame slots |
|---|---:|---:|---:|---:|
| `default` | Current `RayResources` defaults | Current default | Current default | Current default |
| `compact` | 64 KiB | 16 KiB | 256 | 4 |

The exact defaults and every effective capacity are serialized in `report.json`.
Local execution runs once with `QueryResources` defaults; it has no distributed
frame profile. Capacity refusals and deadline failures invalidate a run and
produce a failure report. The runner never enlarges budgets or deadlines in
response to a failure.

## Measurement boundaries

- **Cold session:** a new connection and worker pool execute `tiny`. Connection
  construction is reported separately from query latency. Ray import/startup is
  measured once by the CLI and is separate from all query samples. Process,
  filesystem and page caches are not flushed, so this is not a cold disk test.
- **Warm:** full result validation precedes warmups. Every workload runs the
  configured warmup count, then repeated timed queries reuse one session per
  profile. Profiles run sequentially and release their workers before the next
  profile starts, so idle pools cannot consume the next profile's reserved CPUs.
  Use reversed `--profiles compact default` order in a repeated run to check
  profile-order effects. Pipelined and FTE share that profile's worker pool. The starting mode
  and workload rotate across repetitions; raw records preserve actual order.
- **Slow client:** the streaming scan retains each batch while sleeping for
  `batch_rows / consumer_rows_per_second`. This is a row-rate limit independent
  of frame boundaries. Requested and actual sleep are reported. The default is
  50000 rows/second; consumer pauses remain part of end-to-end latency. Pauses
  check query/delivery failures and the delivery deadline at most every 50 ms;
  canceling a mixed run wakes a sleeping consumer immediately. Resource cleanup
  is still included in failure-report latency.
- **Mixed:** a paced pipelined scan delivers its first batch, then a sibling
  cursor submits an FTE aggregate to the same pool. The runner records successful
  worker-capacity reservations and releases under the admission lock, including
  query IDs, worker IDs, demands and monotonic timestamps in `mixed_pairs`.
  `max_shared_worker_overlap_seconds` must be positive: it is the longest overlap
  of one reservation from each query on a shared worker. Submission and queue
  wait do not count. This proves concurrent charged worker capacity, including
  retained pipelined output; it does not measure simultaneous CPU execution.
  A run with too few rows to overlap fails with its reservation evidence saved.
  The hooks add local timestamp recording to mixed samples, without worker RPCs.
- **Recovery:** an FTE aggregate is paired with an otherwise identical query
  whose first dispatched downstream worker is killed before commit. Control and
  fault order alternate. The result must match, and the affected task must retry
  with the same input identity and a new fence. Fault-to-completion time includes
  failure detection, replacement, retry, remaining work and result delivery; it
  is not a standalone scheduler repair time.

The clock starts immediately before `query()`. `query_return_seconds` includes
binding, planning, source freezing, admission and preparation that occur before
that call returns; it is not a measure of admission wait alone.
`first_batch_seconds` ends when the first public batch arrives, and is `null`
for empty output. `drain_seconds` ends at EOF, including automatic cleanup done
by the result API. `close_seconds` times the subsequent explicit close.
`total_seconds` ends after that close. All values use `perf_counter`.
For cold samples, `session_first_query_seconds` also includes connection
construction. Paired recovery records retain both totals and their difference.

The timed consumer counts rows and Arrow bytes, checks schemas, and releases
each batch before requesting another. For the bounded aggregate/TopN/tiny
outputs it also copies rows to Python; exact comparison runs after the clock
stops, including for every recovery query. Scan values are fully compared in
the independent validation pass; timed scan samples check schema and row count.
Output throughput is rows or Arrow bytes divided by total seconds. It is not
Parquet input bandwidth or bytes transferred over Flight.

Resource checks and diagnostic RPCs run after timing. A separate retained-batch
pass records query/channel state and worker reservations after 100 ms. These
snapshots describe owned buffers and reserved capacity, not sampled peak RSS.
Each completed sequence must release query admissions, result bytes, worker
reservations and store leases. Warmup records are retained but excluded from
summaries. Summaries include count, min, median, nearest-rank p95 and max. With
fewer than 20 repetitions, this p95 equals the maximum; it is not a stable tail
latency estimate.

## Reports and qualification

`samples.jsonl` is appended after each successful query. `report.json` contains
the effective configuration, package/platform/build identities, raw samples,
summaries and completion status. `report.md` provides a compact table. Active
case files and replay SQL are written before timed queries. Failures preserve
partial samples and the original error with available resource state; an
incomplete report is not a successful performance result.

The benchmark tests run small inputs through the same CLI/orchestration, check
the oracle and summaries, exercise real worker loss and confirm cleanup. They
assert behavior, not timing thresholds. Full datasets and generated reports are
kept outside the repository's tracked source.

Default-capacity changes require a repeatable benefit across relevant data
scales and concurrency patterns, without new admission, memory, delivery or
recovery failures. A single machine's small-input timings are not sufficient to
change global defaults. Cross-platform, multi-node, GPU/model UDF and production
storage qualification remain separate work.

## Initial measurements (2026-10-07)

The runner at `0abbe6dfb6` completed two runs on Linux x86-64, an Intel Xeon
E5-2686 v4 host, Python 3.12.14, Ray 2.59.0 and PyArrow 25.0.1. Each cluster
exposed two CPUs to Ray; local queries used two native threads. The installed
non-editable native artifact had SHA-256
`ccde6aecab29338254421295c23b340743ddfb69936f8e14a8fbbf646ed961f4`.
Both runs used seed 970, three repetitions and one warmup per workload/mode.

- 100000 rows: all scenarios, `default` then `compact`; 134 samples including
  20 warmups, 20 full-result validations and six injected worker losses.
- 1000000 rows: `--profiles compact default --scenarios warm`; 80 samples
  including 20 warmups and 20 full-result validations.

All samples succeeded and ownership checks passed. Input generation and
correctness checks had already warmed filesystem caches. Ray's object store
used `/tmp` on this host; result data travelled through Vane's Flight/store
paths. These are single-host development measurements, with three observations
per cell, rather than hardware-normalized or production tail-latency claims.

Median end-to-end times in milliseconds for the 100000-row run:

| Mode, default capacities | Cold tiny | Warm tiny | Scan | Aggregate | Join + TopN |
|---|---:|---:|---:|---:|---:|
| Local | 2.07 | 1.30 | 31.03 | 7.72 | 13.18 |
| Ray pipelined | 1898.42 | 976.68 | 1121.57 | 1075.48 | 1119.87 |
| Ray FTE | 1960.07 | 1052.37 | 1757.57 | 1426.55 | 1582.42 |

Cold tiny excludes connection construction and the separate 4.15-second Ray
startup in this run. For warm tiny, the pipelined median `query()` return was
953.60 ms and its first batch 956.63 ms; FTE returned at 938.79 ms and delivered
its first batch at 1033.54 ms. This locates most small-query latency before
`query()` returns. These measurements do not distinguish planning, actor
creation, RPC preparation and transport setup within that interval.

Scan medians across both data scales:

| Mode / profile | 100000 rows, ms | 1000000 rows, ms | 1000000 rows, output MiB/s |
|---|---:|---:|---:|
| Local | 31.03 | 253.14 | 142.49 |
| Pipelined / default | 1121.57 | 2410.93 | 14.96 |
| Pipelined / compact | 1317.68 | 4492.06 | 8.03 |
| FTE / default | 1757.57 | 5536.41 | 6.51 |
| FTE / compact | 2254.52 | 10972.05 | 3.29 |

For 100000 rows at 50000 consumer rows/second, default pipelined/FTE scans took
2890.74/3429.23 ms. The historical mixed samples recorded 2874.75 ms for the
pipelined scan and 1409.39 ms for the FTE aggregate. Their runner checked only
submission overlap, so these timings do not establish concurrent worker use;
rerun with `mixed_pairs` evidence before treating them as mixed acceptance.
Default FTE recovery took 2519.40 ms versus a 1421.31 ms control median. The
median paired additional time was 1104.46 ms. Every injected failure retried
the affected task exactly once with a fixed input identity and a new fence.

The retained-batch scan snapshot reserved 5 MiB of worker exchange capacity and
6.25 MiB of staging with defaults, versus 320 KiB and 2.5 MiB for `compact`.
Operator reservations were 128 MiB in both cases. This is a reservation
comparison, not a process-memory saving measurement.

**Capacity decision:** retain the current defaults (1 MiB channel window,
64 KiB frame, 1024 frame rows, 16 slots). The compact profile reduced buffer
reservations but made the 1-million-row scan about 1.86 times slower in
pipelined mode and 1.98 times slower in FTE. It did not show a consistent latency
benefit on smaller queries. No global capacity or timeout values changed.
The follow-up below attributes and reduces the warm Ray submission interval.

## Result actor reuse (2026-10-07)

These are historical measurements of commit `5374cf1c4f`. The application
Runtime and multi-query result service replace that implementation. The numbers
below do not measure or validate the current service architecture.

Driver-side timing around planning, worker-pool initialization and synchronous
control waits attributed most warm submission time to waiting for a newly
created result actor. Five warm `SELECT 42::BIGINT AS answer` observations per
mode, after one initial query, gave these median times:

| Operation | Pipelined, ms | FTE, ms |
|---|---:|---:|
| Original `query()` return | 945.76 | 936.63 |
| Original new result actor readiness | 878.01 | 865.77 |
| Reused result actor checkout | 2.20 | 2.97 |

Planning took less than 1 ms in this trace. Actor readiness includes Ray
scheduling, process/import startup and the first control response; it is not a
measurement of the Python constructor alone. These diagnostic observations
were separate from the uninstrumented benchmark runs below.

Commit `5374cf1c4f` reuses an exclusive result actor within a session. Every
checkout receives a new epoch and fresh native channels, Flight endpoints and
FTE reader state. Cleanup must complete before reuse; stale control calls are
fenced and failed cleanup evicts the process. Both modes share the lazy pool,
bounded by `max_results`. Idle processes and their Ray memory reservations
remain until session close. With default frames, each cached actor reserves
4.5 MiB for a pipelined-only session, or 5.75 MiB when exchange stores are
registered. These reservations exclude interpreter/module RSS. See the
[execution design](PIPELINED_EXECUTION_DESIGN.md#p2-已实现的跨进程数据面与-ray-调度) for ownership.

The same two runs as the initial measurements completed on this commit, with
the same machine, configuration, native artifact, benchmark script and input
file hashes. The checkout was clean at the start of each run. The 100000-row
all-scenario run produced 134 samples; the 1000000-row warm run produced 80.
The 214 timed samples include 40 warmups and six injected worker losses. The
runs also completed 40 independent full-result validations. All queries, fault checks and ownership checks
succeeded. Replay using a non-editable installation and fresh output paths:

```bash
python -I scripts/benchmark_execution.py \
  --output build/execution-startup-100k --rows 100000 --repetitions 3 --warmups 1
python -I scripts/benchmark_execution.py \
  --output build/execution-startup-1m --rows 1000000 --repetitions 3 --warmups 1 \
  --profiles compact default --scenarios warm
```

Median warm end-to-end times at 100000 rows, using default capacities:

| Workload | Pipelined before, ms | Pipelined after, ms | FTE before, ms | FTE after, ms |
|---|---:|---:|---:|---:|
| Tiny | 976.68 | 92.38 | 1052.37 | 150.82 |
| Scan | 1121.57 | 270.84 | 1757.57 | 883.47 |
| Aggregate | 1075.48 | 168.74 | 1426.55 | 494.66 |
| Join + TopN | 1119.87 | 211.57 | 1582.42 | 686.44 |

The warm tiny `query()` return medians fell from 953.60 to 66.55 ms for
pipelined, and from 938.79 to 43.97 ms for FTE. Cold tiny medians remained
1944.34/2044.28 ms, compared with 1898.42/1960.07 ms previously. The first use
of each concurrent result slot still needs process startup. Ray cluster startup
was measured separately at 4.05/3.04 seconds for the two runs.

Median scan times at 1000000 rows:

| Mode / profile | Before, ms | After, ms | After, output MiB/s |
|---|---:|---:|---:|
| Pipelined / default | 2410.93 | 1542.53 | 23.38 |
| Pipelined / compact | 4492.06 | 3627.43 | 9.94 |
| FTE / default | 5536.41 | 4700.03 | 7.67 |
| FTE / compact | 10972.05 | 10187.44 | 3.54 |

At 100000 rows, default slow-client pipelined/FTE medians were
1964.21/2518.38 ms. Mixed pipelined scan/FTE aggregate medians were
1955.24/505.77 ms. These historical mixed samples also used the submission-only
check and do not establish concurrent worker use. FTE recovery took
1590.77 ms versus a 499.37 ms control
median; the median paired additional time was 1086.03 ms. Every fault still
retried only the affected attempt, with unchanged input identity and a new
fence. Worker replacement itself was not optimized.

The change passed 143 distinct related tests, including 17 new tests for
reuse, stale/in-flight controls, cancellation during checkout, cleanup failure,
capacity, mode isolation, process loss and session shutdown. Tests enforce
ownership and correctness without latency thresholds. This follow-up keeps
capacity and timeout defaults unchanged. The three-sample, single-host limits
of the initial measurements still apply; the historical intermittent Flight
timeout and release qualification remain separate work.

## Shared service and Flight measurements (2026-10-08)

This is the measurement of the current shared QueryService, replacing the actor
pool measured in the historical section above. Both runs used clean commit
`943a43b8db`, the installed non-editable `vane-ai 0.3.0.dev103` wheel, Linux x86-64,
Python 3.12.14, Ray 2.59.0, PyArrow 25.0.1 and NumPy 2.5.3. The host reports 36 CPUs;
each benchmark Ray cluster exposes two CPUs to two single-thread workers. The
native SHA-256 is
`dd3706ddb13cee62fa5fc5f26faf3e3fd9b6ebd3f1e4ed88d07e6da3530f205b`;
the benchmark script SHA-256 is
`0af39ac3303012858a056a03fbb7b7806ed477f6d058e41da957dcc4c134981e`.
Input file hashes, native identity and script hashes match between the runs.

Use the commands in [Shared Runtime and Flight comparison](#shared-runtime-and-flight-comparison):
32768 rows, seed 970, two partitions, both capacity profiles, three repetitions,
one warmup and a paced consumption rate of 8192 rows/second. Runtime ran first;
Flight ran separately after its cluster stopped. Each produced 112 samples and
16 independent full-result validations. Combined, the 224 samples include 32
warmups, 12 mixed pairs and 12 successful pre-commit worker-loss recoveries. All
correctness, ownership and overlap checks passed.

Median warm latencies with the default capacity profile, in milliseconds:

| Mode | SQL | Runtime first batch | Flight first batch | Runtime total | Flight total |
|---|---|---:|---:|---:|---:|
| pipelined | Tiny | 66.78 | 76.28 | 85.14 | 101.43 |
| pipelined | Scan | 67.30 | 83.70 | 129.80 | 158.13 |
| pipelined | Aggregate | 96.41 | 111.60 | 135.17 | 170.40 |
| pipelined | Join + TopN | 130.40 | 138.40 | 167.68 | 195.52 |
| FTE | Tiny | 122.41 | 142.45 | 142.43 | 168.87 |
| FTE | Scan | 373.65 | 384.79 | 438.05 | 453.38 |
| FTE | Aggregate | 403.60 | 418.37 | 427.03 | 445.76 |
| FTE | Join + TopN | 543.23 | 550.48 | 582.13 | 577.48 |

With compact capacities, median scan totals were 220.11/212.33 ms
(Runtime/Flight pipelined) and 618.82/612.75 ms (Runtime/Flight FTE). The differences
compare complete query lifecycles, including coordination and cleanup, rather
than isolating network overhead. These are three-sample loopback measurements;
small reversals do not establish a speed improvement. Capacity defaults remain
unchanged.

Default-profile cold session plus first-query medians were 1964.60/2011.01 ms
(Runtime/Flight pipelined) and 2105.97/2134.28 ms (Runtime/Flight FTE). Ray cluster
startup, excluded from those values, was 3.17/3.28 seconds. Flight cold includes
constructing both public listeners and its client session.

All six mixed pairs per interface held overlapping reservations on a shared
worker. The longest per-pair overlaps ranged from 102.11 to 171.41 ms for Runtime
and 89.68 to 153.58 ms for Flight. These intervals include retained pipelined output
and prove concurrent charged capacity; they are not CPU-utilization samples.
Unlike the historical submission-only checks, serial dispatch cannot pass this
criterion.

Default-profile FTE recovery medians were 1539.72/1607.71 ms (Runtime/Flight),
against control medians of 419.04/435.34 ms. Median paired additional times were
1120.68/1182.29 ms. With compact capacities, additional medians were
1107.83/1150.84 ms. Each fault retried the affected attempt with fixed inputs and
a distinct fence, and all worker/store/result ledgers retired successfully.

Raw reports, samples, resource observations and replayable inputs are in
`build/server-benchmark-runtime/` and `build/server-benchmark-flight/`; the combined
summary and review/install/test evidence are in `build/server-acceptance/`.
These generated directories are not committed. The related tests passed
**170 cases, with one optional ADBC skip**. No full release/fast suite was run.
Cross-host deployment and release qualification, plus the historical intermittent
Flight timeout investigation, remain separate work.

## Flight and versioned FTE snapshots (2026-10-14)

This comparison uses SF10 Parquet inputs: eight files totaling 3,951,712,526
bytes, with 59,986,052 lineitem rows. Local uses 36 threads and 32 GiB of operator
memory; Ray uses two 18-CPU workers, two partitions and 16 GiB of operator memory
per worker. There is one active query/result, 64 task contexts per worker,
256 I/O slots, 128 MiB of exchange buffers and 256 MiB of staging buffers per
worker. The benchmark alone pins the Ray object store to 1 GiB. Channels retain
the default 1 MiB/64 KiB/1024-row/16-slot limits. The shared host retains OS
caches; no concurrent builds or tests ran during these comparisons. These are
single-host measurements, not a TPC-H score or storage-node fault qualification.

The baseline is `0920b4472d80b1452fd12862d23fcb0e09c8ead8`, with installed
`vane-ai 0.3.0.dev89`. Flight measurements use `dca23aba162d66a7b870099118169312ec0e5dfa`
and `0.3.0.dev90`; the completed FTE qualification uses
`e152596e1c303c1f5387f40b0550b69ae9dbfcb7` and `0.3.0.dev92`.
All use Python 3.12.13, Ray 2.59.0, PyArrow 26.0.0 and NumPy 2.5.4.
Native SHA-256 identities, respectively, are:

```text
4905c50120f8bcbdc6406ddb4e6aaa3a60db45a96bdb4d15d8461aaf192ad7e1
1605c8eac585ade521462a996c2097794b5a48b01f31069de6f4bb2d0ee4dc01
1569820530f4bcc72f549dc7a931a6c8aa9f685d1b11e3dcdad86421bb0ee9b5
```

Every measured query includes complete collection and successful result cleanup.
Ray startup, the initial `select 1`, equality checks and Runtime shutdown are
outside the timing window. Results match the ordered local oracle, with float
tolerances of `rel=1e-9, abs=1e-6`; local HUGEINT `decimal128(38, 0)` results are
normalized to the native `decimal256(39, 0)` representation. Inputs, code
identities, configurations, output row counts and
hashes are retained. Each run removes its owned source copies, materialized data,
spill and Ray temporary directories after cleanup; the original inputs remain.

Flight now permits four outstanding frames and releases their leases on
cumulative acknowledgements. Final success still waits for the last ACK, and
the larger staging requirement is charged before admission. The complete
22-query pipelined pass fell from **258.07 to 117.24 seconds**; local remained
9.06/8.99 seconds. Three-sample pipelined Q17/Q21 medians changed from
36.55/38.05 seconds to **12.06/15.97 seconds**. These two completed modes contain
52 validated samples. Their enclosing run subsequently failed on uncached FTE
Q8's 120-second delivery deadline, so it is not a successful three-mode run.

The source attribution run measured the entire `stage_ray_query` call, including
copy/hash/sync, binding and compilation. Before reuse, Q6 spent **12.27 of 13.30
seconds** there (92%); Q17 spent **15.65 of 90.28 seconds** (17%). With a 4 GiB
version cache, a separate three-query Q6 run at `5796d4e9ad` measured:

| Q6 sample | Source preparation, s | Complete query, s |
|---|---:|---:|
| First copy | 18.256 | 18.754 |
| First cache hit | 0.060 | 0.536 |
| Second cache hit | 0.060 | 0.544 |

Final-code Q6 cache hits in the full passes took 0.033–0.065 seconds to prepare,
with 0.52–2.03-second total latency. First copies retain their original cost;
storage variability remains visible in complete-query times. Reuse is opt-in
through `ExchangeStore.source_cache_bytes`, defaults to zero, and applies to
unchanged input versions within one Runtime. Cached versions retain a separately
charged lease until Runtime shutdown. Full caches fall back to private copies.

Verified materialized commits now return their unused maximum allowance while
retaining the actual sealed bytes. Live, failed and stale attempts retain their
maximum allowance until safe cleanup. The qualification reduced `query_bytes`
from 160 to **32 GiB**, added a separate **4 GiB** cache reservation, and kept the
160 GiB store capacity, 4 GiB source limit and 2 GiB object limit. The largest
observed query charge was **17.52 GiB**. Q17's peak changed from 40.52 to 13.24 GiB;
Q9's fresh original control charged 89.56 GiB versus 17.52 GiB with verified-size
accounting. The cache copied eight versions once (3,951,712,526 bytes), recorded
77 hits and no capacity bypasses, and returned its entire reservation on close.

The final FTE correctness/storage qualification completed all 22 queries plus
two additional Q17/Q21 repetitions: **26 matching samples**. It explicitly uses
600-second execution and delivery limits; the original profile used a
120-second delivery limit. A final-code run at that original limit still timed
out on Q9. The original Q9 control took 143.13 seconds, including 23.51 seconds
of submission, leaving 119.61 seconds for execution, delivery and cleanup.
A separate original Q9 control using the same 600-second qualification limit
completed in 129.16 seconds; the final cached Q9 sample took 96.98 seconds.
Production timeout defaults are unchanged. The longer qualification establishes
correct results and bounded storage, not compliance with the 120-second profile.

For context, the new full FTE pass totaled 766.22 seconds, including 32.15 seconds
of submission and 30.47 seconds of source preparation. The historical successful
baseline totaled 1237.25 seconds and 460.53 seconds of submission under the
stricter delivery limit. Three-sample FTE Q17/Q21 medians were 78.88/96.93 seconds,
versus historical 115.20/113.52 seconds. Current Q17 samples ranged from 69.01 to
92.71 seconds and Q21 from 79.55 to 105.93 seconds; these figures do not establish
a universal end-to-end speedup or eliminate the remaining gap to local execution.

Two earlier cache acceptance attempts exposed metadata-lock contention. Lease
sync held the global quota lock for up to 13.04 seconds, causing heartbeats or
orphan scans to fail. Renewal now serializes with its own lease's close while
publishing outside the quota lock; opportunistic orphan scans defer if that
lock is busy. The final qualification recorded no lock/renewal errors, with a
maximum observed global hold below 0.48 seconds despite a 12.02-second renewal.
Tests cover expiry, slow renewal, independent admission and cleanup ownership.

The 64 KiB materialized I/O buffers were also measured independently on four
million BIGINT rows (33,406,689 sealed bytes). Three-sample median write/read
times changed from 0.554/0.454 to 0.497/0.440 seconds; write/read syscalls fell from
15,634/16,149 to 510/1,025. Sealed-file hashes were identical. This microbenchmark
does not establish a large-query FTE speedup by itself.

Retained evidence is under `build/pr-stack-acceptance-20261014/`:
`execution-optimization-summary.json`, `execution-optimization-timings.csv`,
the frozen drivers and per-run configuration/validation/cleanup manifests.
`sf10-source-cache-qualified-600/` is the successful final FTE qualification;
`sf10-source-cache-full/`, `sf10-source-cache-full-v2/` and
`sf10-source-cache-full-v3/` retain the failed attempts and their diagnostics.
The qualification driver SHA-256 is
`01689859fb32231ed257c673d0504550f45c7a38a8808af361bd5da328c8a6d2`.
Generated benchmark artifacts are not committed.

## Pipelined exchange planning and native parallelism (2026-10-14)

The next comparison uses the same SF10 inputs, two 18-CPU workers, two
partitions, operator budgets, one active query/result, and timing boundaries
described above. The frozen installed baseline is the `e152596e1c` runtime
(`0.3.0.dev92`); its source changes are contained in PR head `e9daa6abdc`.
No concurrent builds or tests run during measurement. The acceptance driver
does not sample live diagnostics or profile stacks. Ray startup and the initial
`select 1` are excluded, while query submission, complete collection and
successful result close are included. Every sample matches the same ordered
local oracle and returns query, result and worker reservations to zero.

Three changes reduce work before increasing native parallelism:

- Broadcast selection retains the existing tiny-build path and also accepts
  estimated builds up to 8 MiB when their extra replicas cost no more than
  shuffling the probe. It includes row widths and partition count, and avoids
  guessing nested-value widths. This preserves the probe scan and its native
  dynamic filters within the fragment.
- The default frame row limit increases from 1024 to the native vector size of
  2048. The 64 KiB frame, 1 MiB window and 16-slot limits stay the same; wide
  rows still split at the byte limit.
- Sparse exchange inputs use the existing bounded collector coalescer. It
  flushes periodically, before waiting for more input and at completion. A
  one-slot regression reduces 64 four-row inputs to fewer than 16 output frames
  and delivers a final partial frame before upstream EOF.

Native exchange sources and unordered collectors then permit the scheduler's
configured parallelism. Shared waiter groups multiplex the channel's single
read/write callback without dropping blocked tasks. Sequence allocation and
publication are serialized, while each task retains its own input and partial
output. Explicit fixed-order output keeps a serial collector. Four-thread
tests exercise empty-input wakeups, retained one-slot output credit, exact
rows/sequences, cancellation and credit return after task destruction; ordered
NULL results and LIMIT/OFFSET are compared exactly with local execution.

| Measurement, seconds | Frozen baseline | Work reduction (`935c3346b8`) | With parallelism (`b837df7137`) | Local, final run |
|---|---:|---:|---:|---:|
| All 22 queries, one complete pass | 117.172 | 94.322 | **64.832** | 9.170 |
| Q17, median of three | 11.818 | 5.352 | **5.068** | 0.384 |
| Q21, median of three | 15.397 | 11.425 | **10.370** | 1.170 |
| Q7, first pass | 5.957 | 4.193 | **1.802** | 0.409 |
| Q9, first pass | 9.801 | 6.130 | **2.607** | 0.914 |

Each full run contains 52 matching samples: both modes run all 22 queries and
two extra repetitions of Q17/Q21. Local complete-pass totals are
9.025/9.130/9.170 seconds. The pipelined total falls 44.7% from the fresh frozen
baseline; native parallelism contributes a further 31.3% reduction from the
work-reduction candidate. This remains **7.07 times local**, with substantial
shuffle and correlated-plan work still present. These single-host samples do
not establish parity with local or a speedup for every query. Q6, for example,
takes 0.390/0.397/0.416 seconds across the three first passes.

Separate three-repetition controls distinguish the changes: broadcast planning
alone gives a Q17 median of 7.991 seconds; using 2048-row frames gives 5.331
seconds. Q21 changes from 15.467 to 11.676 seconds in those controls. Adding
coalescing gives Q17/Q21 medians of 5.264/11.557 seconds; it mainly addresses
sparse output rather than those two queries. A subsequent parallel control
gives 4.945/11.057 seconds. Diagnostic and native stack-profile runs are retained
separately and are not included in these timing comparisons.

A final one-second diagnostic sampler still observes at least 57.25 million
rows on Q17's remaining large exchange, while its filtered outer input exchanges
only 61,385 rows. Q21 still sends 59.99 million and 37.93 million rows through
two separate exchanges. These counts include outgoing endpoints only; live
snapshots before completion are lower bounds, not final traffic totals. The two
workers average 13.9/14.1 CPU cores during these Q17/Q21 diagnostic samples out
of 36 configured cores. Reducing raw input before correlated/hash boundaries
and the remaining transport/wakeup costs are the next targets; parallel task
eligibility alone does not make their physical work equivalent to local.

Installed native SHA-256 identities, in baseline/work-reduction/parallel order:

```text
1569820530f4bcc72f549dc7a931a6c8aa9f685d1b11e3dcdad86421bb0ee9b5
5b225a4c34e2c1023e89aa1ceee132b3db7d051bdc764e8b3c6018b044fb15fa
11f9e4c40e39605d83848eeaa9ef9542853f3cae15548eb004fd417bb132b409
```

The candidates use installed `vane-ai 0.3.0.dev95` and `0.3.0.dev96` with
Python 3.12.13, Ray 2.59.0, PyArrow 26.0.0 and NumPy 2.5.4. Retained evidence is
under `build/pipelined-local-gap-20261014/`: `baseline-full/`, `candidate-full/`,
`parallel-full/`, `parallel-comparison.json`, `parallel-comparison.csv`, the
control and diagnostic runs, and the frozen acceptance driver. Its SHA-256 is
`7df12dcea6a469482b85f84e6bfa015d1eca299ba06d55707a1b2bad0f3b9c60`.
Configurations include installed Python-file hashes, native identity, input
hashes, SQL, resource settings, output counts/hashes and cleanup manifests.
Owned Ray processes stop before their temporary and spill directories are
removed. Original inputs are preserved; repeated full result outputs are not
retained.

## Filtering correlated inputs before exchange (2026-10-15)

Q17's partitioned delimiter still received the entire repeated lineitem scan,
although only a small set of filtered part keys could contribute. The planner
now proves a superset of the owner's correlation keys through unchanged
projections, filters and inner hash joins to a small Parquet scan. It broadcasts
that key column and applies a native null-safe SEMI join before the correlated
input's hash exchange. Probe duplicates are preserved and the original
delimiter retains all matching and aggregation semantics. The estimate must
fit 8 MiB, have fewer than one eighth of the input rows, and cost less replica
traffic than the input. Outer joins, casts, aggregates, nested scopes and
arbitrary table functions are not traversed to derive a key source.

Repeated source planning also retains the original coordinator bind after
serializing its worker copy. Otherwise, compiling the same Parquet scan twice
would enumerate an empty file list on its second use. Both copies keep explicit
source assignments and the existing pipelined/FTE source contracts.

A fresh comparison freezes the preceding `4b5d2e2d4d` checkout's installed
`0.3.0.dev96` runtime and compares it with the new `0.3.0.dev97` build. It uses
the same SF10 files, two 18-CPU workers, two partitions, resource limits, driver
and timing boundaries as the preceding experiment. No builds or tests run
concurrently with either timing pass.

| Measurement, seconds | Fresh frozen baseline | Key filter | Local, candidate pass |
|---|---:|---:|---:|
| All 22 queries, one complete pass | 63.151 | **59.351** | 9.141 |
| Q17, median of three | 4.419 | **0.611** | 0.403 |
| Q21, median of three | 9.893 | 9.445 | 1.259 |
| Q6, first pass | 0.390 | 0.404 | 0.216 |

Q17 falls **86.2%** and takes **1.52 times local**. The complete-pass total falls
6.0%, but remains **6.49 times local**. The baseline local total is 9.260
seconds. Q21's small variation is not attributed to the filter: its large
correlated inputs remain. This change does not establish general parity with
local. Q21, Q18, Q8 and Q5 remain substantial contributors to the total.

Each full comparison contains 52 matching samples, including two extra
Q17/Q21 repetitions in both modes. A separate nine-sample pipelined control
gives Q17/Q21 medians of 0.617/9.575 seconds and a Q6 median of 0.405 seconds.
Ordered results use the existing numeric tolerance; all queries return
reservations and result ownership to zero.

Separate completed native direct-exchange diagnostics confirm the work
reduction without relying on timing: Q17's repeated fact input falls from
**59,986,052 to 61,385 rows** across its two hash partitions. Total accepted
channel rows fall from 60,049,484 to 126,861, including the additional key
channel; shared broadcast channels are counted once. Both diagnostics match
local results, respect the configured byte windows and return channel credit.
These are in-process native work counts, not Ray network byte measurements or
additional timing samples.

The new regression retains the delimiter with default optimizer settings on a
selective Parquet fixture, comparing exact results and bounded exchange rows
with the former runtime. Semantic cases
cover one/three partitions, NULL-safe and ordinary equality, duplicate keys,
outer rows, join-key and payload-key lineage, and one/four native threads.
Related compiler, SQL, performance and FTE-source tests pass (1175 cases); the
145-case performance module also passes after finalizing the default-optimizer
fixture. Ten targeted public Ray cases pass across the affected run and the
corrected-fixture rerun, exercising the repeated scan in both execution modes
and existing correlated/source checks. The complete release suite is not rerun.

Native SHA-256 identities, in frozen baseline/candidate order:

```text
11f9e4c40e39605d83848eeaa9ef9542853f3cae15548eb004fd417bb132b409
671df83774f6ba8c5bbcf5b5103873d8cdbecd19715f0463abaf1fbe2c8b2422
```

Evidence is retained under `build/pipelined-key-filter-20261014/` (the run
started before midnight): `baseline-full/`, `candidate-full/`,
`candidate-focus/`, `comparison.json`, `comparison.csv`, the native work-count
reports, source patch, test logs and frozen acceptance driver. The driver's
SHA-256 remains
`7df12dcea6a469482b85f84e6bfa015d1eca299ba06d55707a1b2bad0f3b9c60`.
Configurations retain code/package/input identities and output counts/hashes.
Owned Ray processes stop before temporary/spill cleanup; source files remain
unchanged and the shared local result cache is removed after comparison.

## Reusing correlated owner keys before exchange (2026-10-15)

Q21 cannot derive a sufficiently small independent scan for either correlation:
its orders scan exceeds the existing estimate limit, and its outer owner is a
nested delimiter. The planner now falls back to the existing owner fragment's
output. A second hash edge feeds a key projection and broadcast into the native
SEMI filter; the owner's source and joins execute once. The original delimiter
still owns matching, NULL and aggregation semantics. Both delimiter variants
consume their owner before DELIM_SCAN pipelines, and the filter branch drains
that same owner into its build side before probing. This preserves progress
with shared producers and bounded output windows.

The fallback requires a nonzero owner estimate, fewer than one eighth of the
input rows, at most 8 MiB of estimated key data, and lower estimated traffic for
the additional complete owner edge plus key replicas than the input exchange.
The earlier small independent scan path remains preferred for Q17. FTE uses
the same graph and replays committed owner outputs for failed consumers.

A fresh comparison freezes `35ca1d5bf0`'s installed `0.3.0.dev97` runtime and
compares it with `0.3.0.dev98`, using the same SF10 inputs, two 18-CPU workers,
two partitions, resource limits and timing boundaries as above. This task ran
no concurrent builds or tests during timing. The host is shared, and unrelated
worktree tests were active; these runs do not claim exclusive host isolation.

| Measurement, seconds | Fresh frozen baseline | Owner reuse | Local, candidate pass |
|---|---:|---:|---:|
| All 22 queries, one complete pass | 59.127 | **52.144** | 9.149 |
| Q21, median of three | 9.632 | **4.295** | 1.263 |
| Q17, median of three | 0.602 | 0.565 | 0.395 |

Q21 falls **55.4%** and takes **3.40 times local**. The complete pass falls
11.8% but remains **5.70 times local**; the baseline local total is 9.022
seconds. The two full runs each have 52 matching samples. A separate nine-sample
pipelined control gives Q6/Q17/Q21 medians of 0.397/0.592/4.322 seconds. Q17's
small variation is not attributed to this fallback. Q18 (7.979 seconds), Q8
(6.945) and Q5 (6.599) remain the largest candidate first-pass contributors.
Ordered comparisons retain the existing numeric tolerance, and all samples
return query, result and worker reservations to zero.

Separate completed native diagnostics establish the work reduction:

| Q21 accepted rows | Frozen baseline | Owner reuse |
|---|---:|---:|
| First correlated fact input | 59,986,052 | **3,458,628** |
| Second correlated fact input | 37,929,348 | **2,401,716** |
| All channels, including added owner/key branches | 144,608,322 | **55,437,498** |

These are in-process native channel counts, with shared broadcast channels
counted once, not Ray network bytes or timing samples. Both results match local
exactly and have the same output hash; channel credit returns to zero within
the configured byte windows. The outer lineitem input to the ordinary hash join
still exchanges **37,929,348 rows**, identifying a remaining boundary for
filter pushdown. The completed diagnostics use the public 1 MiB/16-slot
exchange windows and a diagnostic-only 600-second deadline. An earlier frozen
baseline attempt with a 64 KiB window reached its 180-second execution limit;
its error and cleanup manifest are retained separately.

The 32 new default-optimizer cases compare exact local results, reduced
exchange volume and one execution of the owner source across one/four threads,
two/three partitions, ordinary/NULL-safe equality, duplicate keys, multiple
correlation fields, EXISTS, NOT EXISTS, nested and scalar queries. They fail
the work-volume assertion on the frozen former runtime. Single-slot progress
and cancellation are covered; cancellation runs in a subprocess with an
independent deadline. Related native/compiler/SQL/FTE-source/exchange/graph
modules pass **1367 distinct cases** across the main run and a corrected
cancellation-fixture rerun. The initial fixture assumed a fixed number of pump
steps before backpressure; it now waits for the observed blocked state.

**Nine targeted Ray cases pass**, including both public execution modes and
worker loss after a shared owner stage commits. The failed consumer reuses its
input identity with a new fence, and reservations return to zero. The full
release suite and large-query FTE timing qualification are not rerun.

Native SHA-256 identities, in frozen baseline/candidate order:

```text
671df83774f6ba8c5bbcf5b5103873d8cdbecd19715f0463abaf1fbe2c8b2422
9093f928fb2f3806b1b96ced415ed0320644cbd164b9d1def51b460a769b4ace
```

Evidence is retained under `build/pipelined-owner-keys-20261015/`:
`baseline-full/`, `candidate-full/`, `candidate-focus/`, `comparison.json`,
`comparison.csv`, native work reports, source patch and test logs. The frozen
driver SHA-256 remains
`7df12dcea6a469482b85f84e6bfa015d1eca299ba06d55707a1b2bad0f3b9c60`.
Configurations retain package/input identities and output counts/hashes.
Owned processes stop before temporary/spill cleanup, and the shared oracle
cache is removed after comparison. Original SF10 inputs remain unchanged.

## Selective join filters before earlier hash exchanges (2026-10-15)

Q8 and Q21 still lost small-build filtering at an earlier hash boundary. Q8
exchanged the full lineitem input before applying its selective part join;
Q21 exchanged every late lineitem before its selective supplier join. The
planner now traces an unchanged integral probe key through projections,
filters and the probe side of inner/semi joins, placing a necessary SEMI
condition before an earlier hash exchange. Computed keys, LIMIT, aggregation,
outer joins and shared scopes stop the traversal. The original join retains
all matching, duplicate and NULL semantics.

This path requires native join-filter metadata indicating a filtered build,
an existing broadcast-eligible inner/semi join, a nonzero build estimate no
larger than 8 MiB, fewer than one eighth of the target input rows, and less
additional replica traffic than that input exchange. The original join and
the early filter share the same complete build output; neither its scans nor
joins are recomputed. Both branches drain the build before probing. The
additional edge remains subject to existing credit and reservation limits.

A fresh comparison freezes `845f34bdf9`'s installed `0.3.0.dev98` runtime and
compares it with `0.3.0.dev99`. It retains the same SF10 inputs, two 18-CPU
workers, two partitions, limits and timing boundaries. This task runs no
concurrent builds or tests during measurement; the host and OS caches remain
shared.

| Measurement, seconds | Fresh frozen baseline | Early filter | Local, candidate pass |
|---|---:|---:|---:|
| All 22 queries, one complete pass | 54.299 | **44.925** | 8.921 |
| Q8, first pass | 6.739 | **0.765** | 0.507 |
| Q21, median of three | 4.226 | **2.095** | 1.178 |
| Q17, median of three | 0.571 | 0.596 | 0.385 |
| Q18, first pass | 8.312 | 8.069 | 0.686 |
| Q5, first pass | 7.240 | 6.760 | 0.408 |

The complete pass falls **17.3%**, but remains **5.04 times local**. The
baseline local total is 9.021 seconds. Each full run has 52 matching samples.
Separate three-repetition controls for Q5/Q8/Q18/Q21 each add 12 matching
samples: Q8's median falls from **7.361 to 0.824 seconds (88.8%)**, and Q21's
from **4.615 to 2.202 seconds (52.3%)**. The two sets of observations are
retained separately. Q5/Q18 control medians are 6.639/8.685 seconds before and
6.569/8.374 after; their variation is not attributed to this change. Q17 keeps
its earlier scan-key filter. Ordered comparisons use the existing numeric
tolerance and return all query, result and worker reservations to zero.

Completed native direct-exchange diagnostics confirm the work reduction:

| Accepted channel rows | Frozen baseline | Early filter |
|---|---:|---:|
| Q8 fact input before the order join | 59,986,052 | **403,487** |
| Q8 all channels, including the added build branch | 61,309,338 | **1,740,225** |
| Q21 outer fact input before the order join | 37,929,348 | **1,522,366** |
| Q21 all channels, including the added build branch | 55,437,498 | **19,034,526** |

These diagnostics use public exchange windows and count each shared broadcast
channel once; they are not network byte measurements or timing samples. All
four results match local exactly, respect byte bounds, and return all credit.
Source records identify the original tables and retain the same scan count.

Q18 remains expensive: local profiling shows 624 qualifying order keys, while
the filtered aggregate is estimated at 7,583,683 rows. The current static
broadcast bounds therefore leave large exchanges in place. Q18 and Q5 remain
the largest complete-pass contributors; this change does not solve uncertain
build cardinalities or establish parity for the full query set.

The final related native/compiler/SQL/FTE-source/exchange/graph run passes
**1385 cases**. The **20 focused filter, boundary and cancellation cases** also
pass after finalizing the fixtures. Eight new volume regressions fail on the
frozen former runtime while matching local results. They cover duplicate/NULL
keys, inner/semi joins, one/four threads, two/three partitions and single-slot
windows. Further cases check preserved outer rows, computed keys, LIMIT,
aggregates, multiple filters and cancellation under a subprocess deadline.

**Twelve targeted Ray cases pass** across the affected run and final-fixture
rerun. The new public fixtures use default optimizer settings and a wider
probe payload to preserve the intended plan. Earlier fixture attempts used a
database-only optimizer setting that worker preparation correctly rejected;
worker configuration contracts are unchanged. FTE worker-loss coverage proves
that a failed consumer reuses its committed shared build with the same input
identity and a new fence. The full release suite and large-query FTE timings
are not rerun.

Native SHA-256 identities, in frozen baseline/candidate order:

```text
9093f928fb2f3806b1b96ced415ed0320644cbd164b9d1def51b460a769b4ace
3af8f715201c469ca65296997d3fd8777ce863a99d82dd5eee5a9dde5a801c2d
```

Evidence is retained under `build/pipelined-join-filter-20261015/`: both full
and focus controls, `comparison.json`, `comparison.csv`, the four native work
reports, source patch, profiles, test logs and cleanup manifests. The frozen
driver SHA-256 remains
`7df12dcea6a469482b85f84e6bfa015d1eca299ba06d55707a1b2bad0f3b9c60`.
Configurations preserve code/package/input identities and output counts/hashes.
Owned processes stop before temporary/spill cleanup; the shared oracle is
removed after comparison, and original inputs remain unchanged.

## Projected and bounded join key domains (2026-10-15)

Q5's selective build carries wide payloads and exceeds the complete-build
broadcast cap. Q18's HAVING emits only 624 order keys but estimates 7,583,683
rows, exceeding even a projected key-domain cap. Both therefore lost local
build filtering at hash exchange. Completed baseline diagnostics count
59,986,052 fact rows crossing exchange in each query, plus 15,000,000 orders
in Q18.

The planner can now reuse the original build producer for its original hash
join and a separate key projection. Exact key domains retain the 8 MiB
estimated-key and eightfold row-selectivity guards. A filtered aggregate
subtree whose keys exceed that estimate can instead produce
`hash(key) & 65535`, deduplicated locally and globally before broadcast.
This domain never exceeds 65,536 UBIGINT entries. Collisions only retain
extra probe rows: the original join still checks exact keys, NULLs, duplicate
multiplicity and residual conditions. Dense domains may provide no pruning.

Both paths require native build-filter metadata and estimate added full-build
plus key-replica traffic below the target input exchange. Hashed domains also
require more than 524,288 estimated probe rows. Computed or repeated original
build hash keys retain their ordinary projection path. The original build is
not recomputed, the complete-build broadcast cap is unchanged, and shared
consumers drain the build before probing under existing exchange credits.

A fresh SF10 comparison freezes `58b3eacfe6`'s installed `0.3.0.dev99`
runtime and compares it with `0.3.0.dev100`. Both use two 18-CPU workers,
two partitions, 16 GiB operator memory per worker and public exchange limits.
Timing includes submission through complete collection and close; startup
and comparisons are excluded. The host and OS caches remain shared, and this
task runs no builds or tests during measurement.

| First-pass measurement, seconds | Frozen baseline | Key domains | Local, candidate pass |
|---|---:|---:|---:|
| All 22 queries | 44.641 | **30.256** | 9.003 |
| Q3 | 3.234 | **0.762** | 0.376 |
| Q5 | 7.242 | **0.899** | 0.401 |
| Q7 | 1.694 | **0.824** | 0.407 |
| Q18 | 8.254 | **2.731** | 0.697 |

The complete pass falls **32.2%**, remaining **3.36 times local**. Separate
three-repetition controls measure Q5 at **6.570 to 0.894 seconds (86.4%)**,
and Q18 at **8.316 to 2.646 seconds (68.2%)**. Q8/Q21 control medians are
0.841/2.100 before and 0.860/2.031 after; their small variation is not
attributed to this change. The two full runs each have 52 matching samples,
and the two Q5/Q18/Q8/Q21 controls each add 12.

Q11/Q12 initially rise in the single full pass (1.505/2.373 to 1.747/2.908
seconds), prompting additional three-repetition controls. Those medians are
1.756/2.796 before and 1.908/2.565 after: Q12's slowdown does not repeat,
while Q11 remains 8.7% higher with overlapping sample ranges. Their complete
serialized graphs are identical after normalizing only the engine identity.
Q3/Q7 acquire the additional key-domain stage. These controls add 12 matching
samples, for **140 matching samples total**; the timing variation is retained
rather than attributed to a changed Q11/Q12 plan.

| Accepted channel rows | Frozen baseline | Key domains |
|---|---:|---:|
| Q5 fact input | 59,986,052 | **1,825,856** |
| Q5 filtered orders | 2,275,919 | **456,771** |
| Q5 all channels, including added build/key branches | 63,119,038 | **4,653,776** |
| Q18 outer fact input | 59,986,052 | **573,177** |
| Q18 orders input | 15,000,000 | **142,848** |
| Q18 all channels, including added build/key branches | 91,488,847 | **17,222,552** |

The native diagnostics count each shared broadcast channel once, not network
bytes. All four results match local exactly and pairwise output hashes agree;
byte bounds hold and credits return to zero. Q18 still exchanges 15,000,000
partial aggregate groups and all 1,499,999 customer rows. Q12/Q18/Q10 remain
the largest full-pass contributors at 2.908/2.731/2.614 seconds.

The incremental non-editable Release build passes **1408 affected
native/compiler/SQL/performance/FTE-source/exchange/graph cases**. A focused
24-case run also passes; eight volume regressions fail on the frozen previous
runtime after matching exact results. Coverage includes one/four threads,
two/three partitions, single-slot windows, NULLs and duplicates, multiple and
repeated join keys, residual conditions, outer/anti joins and cancellation
under subprocess deadlines.

The strengthened dense-domain fixture has **262,113 distinct raw keys** and
still broadcasts at most 65,536 hash values; it and the empty-domain case
pass separately. An earlier version of that fixture selected ordinary small
build broadcast because its HAVING estimate was lower, so its assertion did
not exercise the hashed path. The fixture was corrected without changing
production code. **Eighteen targeted real-Ray cases pass**, including both
public modes and consumer loss after either projected or hashed key-domain
builds commit. Recovery preserves input identity and changes the attempt
fence. The full release suite and large-query FTE timings are not rerun.

Native SHA-256 identities, in baseline/candidate order:

```text
3af8f715201c469ca65296997d3fd8777ce863a99d82dd5eee5a9dde5a801c2d
758aaff0642304993ba57c9378a22f77214a79af5565bccbc2034465da3a2f46
```

Evidence is retained under `build/pipelined-key-domain-20261015/`: all six
timing runs, `comparison.json`, `comparison.csv`, four completed native work
reports, normalized plan comparisons, source patch, test logs and fixture
corrections. The frozen driver SHA-256 remains
`7df12dcea6a469482b85f84e6bfa015d1eca299ba06d55707a1b2bad0f3b9c60`.
Owned processes stop before temporary/spill cleanup. The shared oracle cache
is removed after recording its hashes; configurations retain input identities,
result row counts/hashes and cleanup manifests. Original SF10 inputs remain
unchanged.

## Filtering compressed join keys and batching aggregate merges (2026-10-15)

Q12 still exchanged all 15 million orders, although its filtered lineitem build
contained 310,803 rows. Native integer compression inserts computed projections
before the join. Scan filter pushdown cannot cross those projections, leaving
`build_side_has_filter` unset even though the build is filtered. The distributed
planner now also recognizes a FILTER or filtered scan beneath build-root
projections while native join-filter optimization remains enabled. It applies
the necessary SEMI filter after the computed probe projection, using the actual
join-key type, and does not move it through that projection.

Exact key-only domains now allow a 16 MiB estimate and require fewer than one
quarter of the estimated probe rows. Added full-build and key-replica traffic
must still be below the probe exchange estimate. Complete-build broadcasts keep
their previous limits. Filtered aggregates with key estimates above 8 MiB retain
the existing bounded 65,536-value hash-domain path. Estimates are planning
limits; actual execution continues to enforce operator and channel budgets.

Separately, merging exported aggregate states previously constructed temporary
vectors and called combine for each row. The merge now builds aligned private
state copies and target pointers for the current batch and calls the original
native combine once. NULL states are skipped, repeated targets retain input
order, malformed state lengths still fail, and destructive combine cannot
modify the input blobs. State serialization and finalization remain unchanged.

The fresh comparison freezes `8f2fc9c9f5`'s installed `0.3.0.dev100` runtime and
compares it with `0.3.0.dev101`. Both use `/data/tpch10`, two 18-CPU workers,
two partitions, 16 GiB operator memory per worker, a 1 MiB exchange window,
64 KiB frames, 2048 rows and 16 slots. Timing includes submission, collection
and successful close; startup and comparisons are excluded. Host load and OS
caches are shared. No task-owned builds or tests run during measured comparisons.

| Measurement, seconds | Frozen baseline | Final candidate | Local, final pass |
|---|---:|---:|---:|
| All 22 queries, one full pass | 29.887 | **27.393** | 9.085 |
| Q12, full pass | 2.512 | **0.576** | 0.284 |
| Q12, median of three focused repetitions | 2.299 | **0.538** | 0.279 |

The full pass improves **8.3%**, remaining **3.02 times local**. Q12's focused
median improves **76.6%**, reaching **1.93 times local**. All other 21 complete
serialized graphs are identical after normalizing only engine identity; Q12
adds one key-domain fragment. Q10's focused median is 2.594 to 2.571 seconds.
Q18 varies from 2.926 to 2.644 in the focused comparison but only 2.713 to 2.679
in the full pass; a separate batch-merge-only control measured 2.906. These
measurements do not establish a substantial end-to-end Q18 improvement.

Completed in-process native diagnostics, separate from Ray timing, show Q12
orders exchange falling from **15,000,000 to 291,909 rows**. Including the
extra full-build and key-domain branches, total accepted channel rows fall from
**15,310,811 to 1,224,326**. Shared broadcast channels count once; these are work
counts, not network bytes. Both results match local exactly and have the same
output hash; all tasks finish, channel bytes and leases return to zero, and
window bounds hold.

An isolated final-aggregate diagnostic merges 1,048,576 decimal-sum states into
4096 groups using one native thread. Median aggregate CPU time over three runs
falls from **213.3 ms to 32.8 ms (84.6%)**. All results match. This measurement
excludes Ray and source scans; Python row conversion is outside the reported
operator CPU time. It does not imply an equivalent query latency reduction.

The final baseline/candidate qualification contains **136 matching timing
samples**: two complete 44-sample passes and two focused 24-sample passes.
Another 48 samples from intermediate merge-only and budget-only candidates are
retained separately. The budget-only candidate did not change Q12's graph;
the additional build-filter recognition is necessary. Every run records input
identities, output counts/hashes, successful cleanup and an untriggered disk guard.

Native SHA-256 identities, in baseline/final order:

```text
758aaff0642304993ba57c9378a22f77214a79af5565bccbc2034465da3a2f46
2b7c71426f703a274d58c4a279a1781b21ff1f4ee00049924134cba23c5ccf31
```

Evidence is retained under `build/pipelined-exchange-followup-20261015/`, including
all six timing runs, `comparison.json`, normalized graph comparisons, final
aggregate profiles, completed Q12 work counts, source identities and test logs.
The frozen acceptance driver SHA-256 remains
`7df12dcea6a469482b85f84e6bfa015d1eca299ba06d55707a1b2bad0f3b9c60`.

Validation passes **716 affected native/compiler/SQL/performance cases** and
**42 targeted real-Ray cases**, including both public modes, repeated aggregate
targets, NULL and malformed states, numerical bounds, compressed-key filtering,
small windows, cancellation and shared-build consumer loss with committed input
reuse and a new fence. Two new volume regressions complete exact results on the
frozen baseline and then fail the exchange-row bound.

The Ray run first passed 40 cases; its two new aggregate assertions compared
incompatible interval result representations. The final fixture casts the
interval result to VARCHAR, as existing interval tests do, and both public
cases plus 14 focused native cases pass. No production change was needed.
The first compressed-key fixture was below the native compression threshold;
its corrected cardinality exercises the intended projection boundary. Logs
retain these fixture corrections. The full release suite and large-query FTE
timings were not rerun. Formatting, applicable pinned pre-commit hooks, copyleft
inventory and `git diff --check` pass.

Owned benchmark processes stop before temporary/spill cleanup. The shared
oracle cache is removed after recording its 22 output-file hashes. Completed
test fixtures are also removed, with paths and sizes retained in cleanup
manifests; active test files and original SF10 inputs are preserved.
