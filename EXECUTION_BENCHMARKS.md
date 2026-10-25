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

## Selecting exact key filters from join output estimates (2026-10-15)

Q10 still exchanged 14,808,183 filtered lineitem rows before its orders join.
The build estimate was 3,000,000 rows against 11,997,210 estimated probes, so
it failed the fourfold probe/build row test. However, the single-key join
estimated only 2,399,442 output rows, and its UINT32 key domain fit the existing
16 MiB estimate limit.

The planner now has an additional eligibility route for an exact key domain:
when the build/probe row test fails, a single-condition INNER/SEMI hash join
can use a nonzero join-output estimate below one quarter of its direct probe
rows. Added full-build and replicated-key traffic must fit within the probe
traffic estimated to disappear. This estimate is not used for one condition
of a multi-key/residual join or a filter moved across another join. The key
size cap, complete-build broadcast limits, hashed-domain path and original
exact join remain unchanged. Cardinalities remain estimates; actual execution
continues to enforce operator and channel budgets.

The comparison freezes `30e82b4fc4`'s installed `0.3.0.dev101` runtime and
compares it with `0.3.0.dev102`. Both use SF10, two 18-CPU workers, two
partitions, 16 GiB operator memory per worker, a 1 MiB exchange window,
64 KiB frames, 2048 rows and 16 slots. Timing includes submission, full
collection and successful close. Startup and validation are excluded; the
host and OS caches are shared. No task-owned builds or tests overlap timing.

| Measurement, seconds | Frozen baseline | Candidate | Local, candidate pass |
|---|---:|---:|---:|
| All 22 queries, one full pass | 28.091 | **25.821** | 9.126 |
| Q2, median of three focused repetitions | 1.361 | **0.380** | 0.140 |
| Q10, median of three focused repetitions | 2.597 | **1.600** | 0.465 |
| Q12 control, median of three focused repetitions | 0.542 | 0.574 | 0.285 |
| Q18 control, median of three focused repetitions | 2.728 | 2.686 | 0.707 |

The full pass improves **8.1%** and remains **2.83 times local**. Q2 and Q10
focused medians improve **72.1%** and **38.4%**, respectively. Q10 remains
**3.44 times local**. Q12/Q18 controls vary in opposite directions; no benefit
is claimed for them. Only Q2 and Q10 serialized graphs change; the other 20
are identical after normalizing engine identity. These fresh controls are a
separate measurement round from the preceding Q12/aggregate-merge comparison.

Completed native diagnostics, separate from Ray timing, show Q10 lineitem
exchange falling from **14,808,183 to 1,147,084 rows**. Including extra build
and key branches, total accepted rows fall from **18,409,613 to 5,894,828**
(**68.0%**). Q2 total accepted rows fall from **1,688,593 to 108,012**
(**93.6%**). Shared broadcast channels count once; these are channel work
counts, not network bytes. Both queries match local exactly and retain their
output hashes, all tasks finish, credits return to zero and byte bounds hold.
Q10 still exchanges 1,499,999 wide customer rows and 381,105 partial groups;
this change does not remove that remaining work.

All **136 timing samples** match local: 24 baseline focused samples split
across two runs, 24 candidate focused samples, and two complete 44-sample
passes. Input identities match across all runs; disk guards do not trigger,
queries and runtimes return to idle, and owned processes stop before cleanup.

Validation passes **165 related native/compiler tests** and **19 real-Ray
tests**, covering both public modes, duplicate/NULL keys, preserved outer
rows, narrow-probe traffic rejection, multiple-key rejection, small exchange
windows, cancellation, and FTE consumer loss after a shared build commits.
The new volume regression completes correct results on the frozen baseline
and then fails at 524,286 fact rows; the candidate satisfies the bounded
exchange-row assertion. The full release suite and large-query FTE timing
were not rerun. Formatting, applicable pinned pre-commit hooks, copyleft
inventory and `git diff --check` pass.

Native SHA-256 identities, in baseline/candidate order:

```text
2b7c71426f703a274d58c4a279a1781b21ff1f4ee00049924134cba23c5ccf31
e3d64f828f0203ca896e8113c0a9559c9e65c3d728e5065427e2edeb10c67646
```

Evidence is retained under `build/pipelined-q10-20261015/`: frozen baseline,
source/native identities, all five timing runs, normalized graph comparisons,
completed Q2/Q10 native work counts, test logs, `comparison.json/csv` and
cleanup manifests. The acceptance driver SHA-256 is
`7df12dcea6a469482b85f84e6bfa015d1eca299ba06d55707a1b2bad0f3b9c60`.
Shared oracle files are removed after retaining their hashes. Test fixtures
and owned temporary/spill data are removed outside measured windows; original
SF10 inputs are preserved.

## Broadcasting wider builds within worker memory (2026-10-15)

The fixed 8 MiB build estimate still forced Q10 to shuffle 1,499,999 wide
customer rows. The planner now accepts larger complete builds when their
estimated size is at most `min(64 MiB, worker_operator_bytes / 16)` and
`4 * (partitions - 1) * build_bytes < probe_bytes`. Unknown nested probe
widths cannot justify this path. The runtime supplies each worker's operator
memory divided by active-query capacity, in both pipelined and FTE modes.
Standalone compilation without a budget retains the previous policy, as do
builds of at most 32 rows and the original 8 MiB path. These are planning
estimates; admission, actual memory limits and spill remain enforced.

Early filters take precedence over larger broadcasts. The planner first finds
filters under the original policy. A filter moved through an earlier join,
or an ancestor's filter already reducing the current probe, retains the
original broadcast limit. A direct key filter can be replaced by a larger
broadcast that eliminates the same join's hash input. Extra early filters
keep their existing size and selectivity limits.

This guard is required by measured regressions. An initial candidate removed
Q7's early customer-key filter, increasing orders exchange from 1,205,808 to
15,000,000 rows. It also removed Q21's early supplier filter while broadcasting
7,309,184 orders. Their focused medians regressed from 0.816 to 1.651 and from
2.084 to 2.631 seconds. Four regression cases fail on that candidate and pass
with the guard, covering small/large selective builds and two/three partitions.
Final Q7/Q8/Q21 serialized graphs equal the frozen baseline; only Q2/Q3/Q5/Q10
change, with the other 18 graphs identical after normalizing engine identity.

The comparison freezes `51e38be681`'s installed `0.3.0.dev102` runtime against
the final `0.3.0.dev103` build. Both use SF10, two 18-CPU workers, two partitions,
16 GiB operator memory per worker, a 1 MiB exchange window, 64 KiB frames,
2048 rows and 16 slots. Timing includes submission, complete collection and
successful close, excluding startup and validation. Host load and OS caches
are shared; no task-owned builds or tests overlap measured comparisons.

| Measurement, seconds | Frozen baseline | Final candidate | Local, final pass |
|---|---:|---:|---:|
| All 22 queries, one full pass | 26.606 | **25.538** | 9.100 |
| Q2, median of three repetitions | 0.397 | 0.396 | 0.146 |
| Q3, median of three repetitions | 0.793 | 0.800 | 0.369 |
| Q5, median of three repetitions | 0.890 | 0.754 | 0.412 |
| Q7, median of three repetitions | 0.816 | 0.811 | 0.409 |
| Q8, median of three repetitions | 0.898 | 0.862 | 0.543 |
| Q10, median of three repetitions | 1.692 | 1.068 | 0.464 |
| Q12, median of three repetitions | 0.568 | 0.564 | 0.294 |
| Q18, median of three repetitions | 2.689 | 2.639 | 0.718 |
| Q21, median of three repetitions | 2.084 | 2.102 | 1.209 |

The full pass improves **4.0%**, remaining **2.81 times local**.
Q10's focused median improves **36.9%**, reaching **2.30 times local**;
Q5 improves **15.2%**. Q2/Q3 do not establish a material latency benefit.
Q7/Q21 return to baseline performance. The unchanged Q8/Q12/Q18 controls vary
slightly; no optimization benefit is claimed for them. One full-pass total is
not a repeated whole-suite estimate.

Completed final native diagnostics, separate from Ray timing, reduce Q10 total
accepted channel rows from **5,894,828 to 2,101,431 (64.4%)**. Wide customer rows
remain local through joining and partial aggregation; 381,105 partial groups
still cross the aggregation exchange. Both outputs match local exactly and
retain the same hash. All tasks finish, byte bounds hold, and bytes/leases
return to zero. Shared broadcasts count once; these are work counts, not
network bytes. A synthetic complete-build regression also reduces accepted
rows from 3,145,852 to 1,048,700 while preserving exact output.

Final qualification contains **196 matching timing samples** across five runs:
54 focused samples and 44 full-pass samples per runtime. An additional
98 samples from the initial candidate are retained separately, giving 294
matching samples across seven runs. Inputs and the frozen driver match in all
runs; disk guards remain untriggered, and queries/runtimes return to idle.

Validation passes **585 affected native/compiler/submission/performance tests**
and **26 targeted real-Ray cases**, including per-query memory shares, both
public modes, duplicate/NULL and outer-join semantics, small windows, early
filter retention, cancellation, and committed-input reuse after consumer loss.
The previous Python 3.11 CI failure was a test synchronization assumption:
blocking on the right input did not ensure a parallel left pipeline had
published its buffered row. The test now waits within its existing deadline
for the actual output lease before injecting the late error. Its EOF/error
assertions remain, and 100 additional repetitions pass.

Forced-optimizer guard fixtures run in process. Four attempted public versions
were rejected before execution because worker preparation deliberately rejects
mismatched database-global optimizer settings; those inapplicable fixtures were
removed. Public guard qualification uses Q7/Q21 with default optimizers. Logs
also retain the earlier reader-lifetime/storage-limit fixture corrections and
the exact-domain fanout fixture's explicit 128 MiB budget. The latter preserves
its intended shared-build recovery path; complete-broadcast recovery has a
separate passing case. The full release suite and large-query FTE timings were
not rerun. Formatting, applicable pinned pre-commit hooks, copyleft inventory
and `git diff --check` pass.

Native SHA-256 identities, in baseline/final order:

```text
e3d64f828f0203ca896e8113c0a9559c9e65c3d728e5065427e2edeb10c67646
4c3ad4751652f3f64f5318c3a5b7f3653b354df117b72e4071d8702adda0dae0
```

Evidence is retained under `build/pipelined-wide-probe-20261015/`: all seven
timing runs, source/native identities, normalized plans, native work counts,
regression controls, test and CI logs, `comparison.json/csv`, and cleanup
manifests. The frozen driver SHA-256 remains
`7df12dcea6a469482b85f84e6bfa015d1eca299ba06d55707a1b2bad0f3b9c60`.
Owned processes stop before temporary/spill cleanup. The shared oracle is
removed after preserving its 22 hashes; test fixtures are removed outside
measured windows. Original SF10 inputs are preserved.

## Independent native sink waiters (2026-10-15)

This follow-up freezes `88e29cb910a597378248e11107dfd36f95b7aefc` as
its baseline. A native collector previously used one waiter lock for all
output channels, serializing frame copies to independent targets and waking
tasks waiting for other targets. Waiter groups now belong to individual
channels. Sequence allocation, publication and blocked-task registration stay
atomic within each channel; metadata remains bounded by the fixed target set.

The frame-size experiment also exposed a lost wakeup in byte-limited windows:
a task blocked on a larger frame shared its producer with a smaller task that
could still publish. That success erased the shared callback without returning
credit. Both Q10 and Q18 stalled with 128 KiB frames; a short-deadline Q18
reproduction retained unfinished producers after all output credit had returned.
Successful publication now preserves the callback until credit, closure or
cancellation triggers it. Three deterministic tests fail on the original
runtime and pass with the fix, including a borrowed view and late release.
The default 64 KiB frame, 1 MiB window, 2048 rows and 16 slots are unchanged.

The same frozen SF10 driver and Parquet inputs are used, with two 18-CPU
workers, two partitions, 16 GiB operator memory per worker, 128 MiB exchange
and 256 MiB staging capacity. Local uses 36 threads and 32 GiB. Timings include
submission through collection and successful close, with shared-host OS caches
retained and no concurrent task-owned builds or tests. The original driver's
SHA-256 remains
`7df12dcea6a469482b85f84e6bfa015d1eca299ba06d55707a1b2bad0f3b9c60`.

| Measurement, seconds | Frozen baseline | Candidate | Candidate local |
|---|---:|---:|---:|
| All 22 queries, one complete pass | 24.935 | 24.617 | 9.118 |
| Q10, median of three | 1.090 | 1.022 | 0.473 |
| Q12, median of three | 0.550 | 0.550 | 0.284 |
| Q18, median of three | 2.693 | 2.372 | 0.708 |

Q18 improves 11.9%, remaining 3.35 times local. Fixing only the lost callback
measures 2.689 seconds for Q18, separating the throughput improvement from that
correctness fix. The single full pass improves 1.3% and remains 2.70 times local;
its small aggregate difference is not evidence of a general speedup. The
15-million-group first Q18 exchange remains; this change addresses publication
concurrency rather than reducing the query's data volume.

The full-pass Q16 sample rose from 1.729 to 2.132 seconds. A three-repetition
check of all larger full-pass outliers runs the candidate first and the frozen
baseline second, using the same oracle and default exchange limits:

| Query | Baseline median, seconds | Candidate median, seconds | Change |
|---|---:|---:|---:|
| Q2 | 0.411 | 0.410 | -0.3% |
| Q7 | 0.870 | 0.817 | -6.1% |
| Q9 | 2.423 | 2.490 | +2.8% |
| Q15 | 0.532 | 0.534 | +0.3% |
| Q16 | 1.697 | 1.762 | +3.9% |
| Q19 | 1.009 | 0.980 | -2.9% |

The large Q16 outlier does not reproduce consistently: candidate samples range
from 1.743 to 2.149 seconds and baseline samples from 1.688 to 2.044. The remaining
Q9/Q16 median costs of 2.8%/3.9% are retained rather than described as zero
regression. These short shared-host runs establish the Q18 improvement more
strongly than an all-query improvement. All nine successful acceptance/control
runs have matching input and driver identities (212 samples, including 96 on
the final native build); diagnostic/failed runs and microbenchmarks are separate.

A separate 128 KiB stress run completes eight repetitions each of Q10 and Q18
without hanging, matches local, and releases every query/worker reservation.
That setting needs 4.25 MiB of reserved Flight staging per link versus 2.25 MiB
at 64 KiB, so it is not made the default. Reusing a previously measured frame
size was also tried and discarded: a four-thread in-process aggregate-state
microbenchmark measured 228.2 versus 229.4 ms (eight samples after warmup), with
no established benefit. That single-process microbenchmark does not demonstrate
the Ray improvement from independent target locks.

The final non-editable Release build passes 262 related native, Flight,
materialized-exchange and fanout/cancellation tests plus 14 real-Ray tests.
The full release suite and large-query FTE performance were not rerun.
Native SHA-256 identities, in frozen baseline / callback-only / final order:

- `4c3ad4751652f3f64f5318c3a5b7f3653b354df117b72e4071d8702adda0dae0`
- `3fa391447df0eb2c59599b33ade3a7e50beba25a139b3093d376a790b99755f4`
- `da70e1e138e6da24939ae40070c4d4a03917d6672c5d00609b2e46e13cec4b05`

Evidence is retained under `build/pipelined-q18-frames-20261015/`: frozen
runtimes, driver/configuration/input identities, raw timings, result counts and
hashes, failed-candidate diagnostics, source patches and cleanup manifests.
Diagnostic runs are separate from acceptance; completed worker-channel counters
are unavailable after cleanup, so partial live observations are not reported
as completed exchange work. Owned processes stop before temporary/spill cleanup;
the shared oracle is removed after recording its hashes. The source dataset is
preserved and every disk guard remains untriggered.

## Broadcasting complete small MARK builds (2026-10-15)

SF10 Q16 took 1.850 seconds versus 0.146 locally in a fresh three-repetition
control at `6d187d2d86`. Its `NOT IN` became a MARK join, which unconditionally
kept a single native scope. A nested literal IN-list scan also kept the part
join inside that scope: 8,000,000 partsupp rows, 1,855,696 part rows and only
56 filtered supplier keys were gathered before the final joins and aggregation.

An uncorrelated equality MARK join now uses the existing complete-build
broadcast path when its build qualifies for the base policy: at most 32
estimated rows, or at most 8 MiB with extra replica traffic no larger than the
probe exchange saved. Each probe partition receives the entire build, so the
native MARK implementation retains NULL, empty-build and duplicate-key
semantics. Probe scans, nested literal IN filters and subsequent joins can
remain distributed. MARK does not use the larger worker-budget broadcast
allowance: retaining an early probe filter must not subsequently change it
into a hash-partitioned build. Larger builds, correlated delimiter metadata,
SINGLE and shared owner scopes retain their existing treatment.

The same frozen acceptance driver compares the old installed runtime with the
non-editable Release candidate on `/data/tpch10`, using two 18-CPU workers,
two partitions, 16 GiB operator memory per worker and unchanged default
64 KiB/2048-row frames, 1 MiB windows and 16 slots. Local uses 36 threads and
32 GiB. Timings include submission through collection and successful close;
input/oracle validation and startup are outside the measured query window.
The host is shared, OS caches remain warm, and no task-owned builds or tests
overlap the comparison runs.

| Measurement, seconds | Frozen baseline | Candidate | Candidate local |
|---|---:|---:|---:|
| All 22 queries, one complete pass | 23.591 | 23.003 | 9.046 |
| Q4, median of three | 2.122 | 2.099 | 0.304 |
| Q9, median of three | 2.368 | 2.376 | 0.860 |
| Q11, median of three | 1.699 | 1.708 | 0.093 |
| Q16, median of three | 1.850 | 0.729 | 0.143 |

Q16 improves **60.6%**, remaining **5.08 times local**. The full pass improves
2.5%, remaining 2.54 times local; that single pass is weaker evidence than the
repeated Q16 comparison. All 22 queries match the local oracle. The focused
Q9 median differs by only +0.3%, while the full-pass Q9 and Q17 samples rise
6.2% and 12.5%. A separate three-repetition check runs the candidate first and
baseline second: Q9 medians are 2.417 versus 2.427 seconds, Q17 0.576 versus
0.613, and Q16 0.650 versus 2.031. The Q9/Q17 single-pass regressions do not
reproduce in this control. Of all 22 compiled graphs, only Q16 changes its
fragment/edge topology; serialized hashes also contain runtime identity and
are not treated as a cross-build operator-equivalence proof.

The following per-query numbers use the candidate's single complete pass;
they are not mixed with the focused medians above. All results pass schema,
row-count and value comparison, with the driver's recorded floating tolerance.

| Query | Local, seconds | Pipelined, seconds | Pipelined / local |
|---|---:|---:|---:|
| Q1 | 0.484 | 0.711 | 1.47 |
| Q2 | 0.133 | 0.385 | 2.88 |
| Q3 | 0.367 | 0.762 | 2.08 |
| Q4 | 0.266 | 2.035 | 7.66 |
| Q5 | 0.407 | 0.755 | 1.85 |
| Q6 | 0.212 | 0.392 | 1.85 |
| Q7 | 0.414 | 0.827 | 2.00 |
| Q8 | 0.511 | 0.890 | 1.74 |
| Q9 | 0.836 | 2.419 | 2.90 |
| Q10 | 0.475 | 1.052 | 2.22 |
| Q11 | 0.093 | 1.787 | 19.25 |
| Q12 | 0.298 | 0.610 | 2.05 |
| Q13 | 0.550 | 1.223 | 2.22 |
| Q14 | 0.297 | 0.692 | 2.33 |
| Q15 | 0.306 | 0.501 | 1.64 |
| Q16 | 0.159 | 0.626 | 3.94 |
| Q17 | 0.403 | 0.700 | 1.74 |
| Q18 | 0.712 | 2.345 | 3.29 |
| Q19 | 0.417 | 0.891 | 2.14 |
| Q20 | 0.342 | 0.903 | 2.64 |
| Q21 | 1.225 | 2.067 | 1.69 |
| Q22 | 0.139 | 0.431 | 3.09 |
| Total | **9.046** | **23.003** | **2.54** |

Separate completed in-process diagnostics count **9,883,592 → 1,539,170**
accepted native channel rows for Q16, a reduction of **84.4%**. The 8-million-row
partsupp probe now stays in its source fragment, the filtered part build sends
296,824 rows, and 1,186,602 surviving rows still cross the DISTINCT aggregation
hash exchange. Both diagnostics return the same 27,840 result rows, release
all channel/lease bytes and stay within the original credit windows. Shared
broadcast channels count once; these work counts are not network bytes or
Ray timings. Q4 still exchanges approximately 38.5 million rows and is not
optimized by this change.

The final build passes **217 related native planner/execution tests** and
**7 real-Ray tests**, including complete MARK builds with NULLs, empty inputs,
duplicates, 1/2/5 partitions, literal IN lists, bounded backpressure, broadcast
budget fallback, scalar-subquery errors, shared/correlated scopes, early close,
interrupt and FTE consumer retry with the same input identity and a new fence.
The corrected new performance regressions fail on the frozen runtime (20
failures, with two fallback controls passing). The full release suite and
large-query FTE performance are not rerun.

Native SHA-256 identities, baseline and candidate:

- `da70e1e138e6da24939ae40070c4d4a03917d6672c5d00609b2e46e13cec4b05`
- `81dfaaf965f0984f5254bcb47cc9e5aa8598da3396897a648cb394d4073a705f`

Evidence is retained under `build/pipelined-mark-join-20261015/`, including the
frozen baseline runtime, unchanged driver, code/input identities, raw timings,
result counts/hashes, completed native work counts and per-run cleanup
manifests. Source Parquet files are preserved. Temporary/spill outputs are
removed after owned processes stop, and shared oracle hashes are retained
before deleting the generated oracle.

## Budgeted correlated scan keys (2026-10-15)

A fresh SF10 comparison at `8c7ea975f2` measures Q4 at 1.999 seconds versus
0.268 locally. Its filtered orders owner estimates 3 million rows, while the
lineitem input estimates about 12 million. The existing one-eighth owner/input
threshold rejects an early key filter, although the direct delimiter join
estimates only about 2.17 million matches. In completed native execution,
37,929,348 lineitem rows cross the hash exchange before that join.

When an independent Parquet scan is the complete delimiter owner, the planner
can now budget its projected integral keys against the declared per-worker,
per-query operator allowance. The existing larger broadcast limit must exceed
8 MiB, and the key estimate must fit `min(64 MiB, budget / 16)`. A direct
single-condition INNER join with the owner's delimiter scan must estimate
less than one-quarter of input rows, and all key replicas must cost less than
the estimated eliminated exchange. Multiple/residual conditions, unavailable
estimates, unknown nested widths and a scan that only supplies a wider owner
superset do not qualify through this cost path. The original small-scan and
shared-owner policies retain their existing thresholds.

The independent scan projects keys before broadcasting them. The existing
NULL-safe SEMI filter removes impossible fact rows before hash exchange; the
original delimiter still enforces matching, duplicate, EXISTS/NOT EXISTS and
aggregation semantics. Explicit source assignments and version checks apply
to the additional scan. It does not introduce a shared owner dependency or
change runtime memory, credit, frame, window or slot limits.

The frozen acceptance driver and SF10 input identities match the previous
round. Local uses 36 threads and 32 GiB; pipelined uses two 18-CPU workers,
two partitions, 16 GiB operator memory per worker, 64 KiB/2048-row frames,
1 MiB windows and 16 slots. Timings include submission through collection and
successful close, excluding startup and oracle comparison. The shared host
retains OS caches, and no task-owned build or test overlaps measured runs.

| Measurement, seconds | Frozen baseline | Candidate | Candidate local |
|---|---:|---:|---:|
| All 22 queries, one complete pass | 22.645 | 21.462 | 9.133 |
| Q4, median of three | 1.999 | 0.682 | 0.276 |
| Q11, median of three | 1.646 | 1.675 | 0.095 |
| Q16, median of three | 0.737 | 0.744 | 0.154 |
| Q18, median of three | 2.415 | 2.421 | 0.697 |

Q4 improves **65.9%**, remaining **2.47 times local**. The full pass improves
5.2%, remaining 2.35 times local; it is weaker evidence than Q4's repeated
comparison. The Q11/Q16/Q18 control medians increase 1.8%/0.9%/0.3%. Full-pass
Q10 and Q11 increase 6.2% and 4.6%; these single observations do not establish
their repeatability, and this round does not claim every query improved.
All 136 timing samples, including 68 on the candidate, pass schema, row-count
and value checks with relative/absolute floating tolerances of 1e-9/1e-6.

The per-query table below uses only the candidate's single complete pass,
without mixing in the focused medians.

| Query | Local, seconds | Pipelined, seconds | Pipelined / local |
|---|---:|---:|---:|
| Q1 | 0.518 | 0.704 | 1.36 |
| Q2 | 0.139 | 0.378 | 2.71 |
| Q3 | 0.370 | 0.763 | 2.06 |
| Q4 | 0.274 | 0.658 | 2.40 |
| Q5 | 0.407 | 0.705 | 1.73 |
| Q6 | 0.215 | 0.346 | 1.61 |
| Q7 | 0.411 | 0.797 | 1.94 |
| Q8 | 0.511 | 0.881 | 1.72 |
| Q9 | 0.884 | 2.415 | 2.73 |
| Q10 | 0.476 | 1.035 | 2.17 |
| Q11 | 0.093 | 1.794 | 19.35 |
| Q12 | 0.293 | 0.612 | 2.09 |
| Q13 | 0.524 | 1.235 | 2.36 |
| Q14 | 0.296 | 0.703 | 2.37 |
| Q15 | 0.317 | 0.500 | 1.58 |
| Q16 | 0.147 | 0.656 | 4.48 |
| Q17 | 0.396 | 0.533 | 1.35 |
| Q18 | 0.727 | 2.396 | 3.30 |
| Q19 | 0.423 | 0.926 | 2.19 |
| Q20 | 0.348 | 0.919 | 2.64 |
| Q21 | 1.220 | 2.087 | 1.71 |
| Q22 | 0.144 | 0.418 | 2.90 |
| Total | **9.133** | **21.462** | **2.35** |

Separate completed native diagnostics count **38,503,039 → 2,597,933**
accepted channel rows for Q4, a **93.3%** reduction. The additional scan
broadcasts 573,671 owner keys; only 1,450,571 lineitem rows then cross their
hash exchange. Both runs return the same five rows and content hash, remain
within the configured windows and release all channel/lease bytes. These
counts include each shared broadcast channel once and are not network bytes
or Ray timings. Worker counters are unavailable after Ray cleanup; completed
native work is measured separately rather than inferred from empty snapshots.

Q11 remains unchanged by this optimization. Its shared CTE keeps two
8-million-row partsupp inputs in one native scope, despite the supplier/nation
build producing only 4,049 keys. Its completed native diagnostic counts
16,004,050 channel rows. Moving filtering before those exchanges is the next
candidate, but the CTE's shared consumers must preserve build/probe ordering
and bounded-channel progress.

The final non-editable Release build (`vane-ai 0.3.0.dev106`) passes **108
related native planner/execution tests** and **9 real-Ray tests**. Coverage
includes equality and NULL-safe EXISTS/NOT EXISTS, duplicate/NULL owner keys,
single/multiple threads, tiny channel budgets, per-query budget division,
replica-traffic and residual-condition fallback, existing correlated scopes,
source-version validation, cancellation/backpressure and FTE consumer retry
with the same input identity and a new fence. Initial new regressions produce
nine failures on the frozen runtime, with three fallback controls passing.
The full release suite and large-query FTE performance are not rerun.

Native SHA-256 identities, baseline and candidate:

- `81dfaaf965f0984f5254bcb47cc9e5aa8598da3396897a648cb394d4073a705f`
- `c26c6363d27e0995b35e49a42b29a96d71647a34039064803ab1dc24e4b42b29`

Evidence is retained under `build/pipelined-q4-q11-20261015/`, including frozen
runtime/configuration/input identities, raw samples, result hashes/counts,
native work diagnostics, `comparison.json` and `local-vs-pipelined.csv`.
All query/worker/result reservations return to zero. Owned processes stop
before temporary/spill cleanup; shared oracle hashes are retained before
deletion. Source Parquet files are preserved, and all disk guards remain
untriggered.

## Filtering independent CTE inputs before gather (2026-10-15)

The fresh baseline at `700c01626e` takes 1.552 seconds for SF10 Q11 versus
0.096 locally. The optimizer shares a supplier/nation build through a native
CTE. Although that build produces only 4,049 rows, both 8-million-row partsupp
inputs are gathered into the CTE's single native scope before their joins.
Completed native diagnostics count 16,004,050 accepted channel rows.

The planner now reuses an independent CTE owner's producer to filter its
independent fact inputs before GATHER. It requires a known owner estimate of
at most 8 MiB, fewer than one-eighth of probe rows, and all additional replica
traffic below the estimated probe exchange. The original join must be
INNER/SEMI with one equality or NULL-safe equality condition over unchanged
integral references, directly against this owner's CTE_SCAN with matching
output types and column positions. Unknown widths, computed/multiple/residual
keys, preserved probe rows, nested CTE/delimiter ownership and repeated
physical nodes retain the original path.

Each added branch receives the complete owner output and uses a native
NULL-safe SEMI filter. The original CTE materialization and final joins remain
responsible for NULLs, duplicate matches, aggregation and shared state. The
owner runs once: both its native materialization and the added filter builds
drain it before probing, so bounded fanout does not depend on fact input
progress. Source assignments, operator budgets, channel credits and FTE input
identity/fence rules are unchanged. This reduces exchange rows; it does not
add Parquet row-group or coordinator split pruning.

The immutable driver uses the same SF10 input identities and resources as the
previous round: local has 36 threads/32 GiB; pipelined has two 18-CPU workers,
two partitions and 16 GiB operator memory per worker. Default 64 KiB/2048-row
frames, 1 MiB windows and 16 slots remain unchanged. The timer covers query
submission through collection and successful close, excluding startup and
oracle comparison. The host is shared, OS caches remain warm, and no task-owned
build or test overlaps the measured runs.

| Measurement, seconds | Frozen baseline | Candidate | Candidate local |
|---|---:|---:|---:|
| All 22 queries, one complete pass | 21.660 | 20.317 | 9.260 |
| Q4, median of three | 0.653 | 0.636 | 0.305 |
| Q9, median of three | 2.346 | 2.377 | 0.842 |
| Q11, median of three | 1.552 | 0.276 | 0.094 |
| Q18, median of three | 2.511 | 2.389 | 0.697 |

Q11 improves **82.2%**, remaining **2.93 times local**. The single full pass
improves **6.2%**, remaining **2.19 times local**. Other full-pass samples vary:
Q17/Q1/Q14/Q8 increase 8.9%/8.1%/7.3%/7.0%, and Q18 increases 6.5%, while its
focused median decreases 4.9%. Q9's focused median increases 1.3%. These are
shared-host observations, not evidence that every query improves or that
single-pass differences are repeatable. The repeated Q11 comparison and
completed work counts are the direct evidence for this change.

All 136 timing samples, including 68 on the candidate, pass schema, row-count
and value comparison with relative/absolute floating tolerances of 1e-9/1e-6.
The repository's fixed Q11 threshold returns zero rows on this SF10 input in
both runtimes; nonempty regression fixtures independently verify aggregates,
duplicate matches and NULL semantics. The following table uses only the
candidate's single complete pass, without mixing in focused medians.

| Query | Local, seconds | Pipelined, seconds | Pipelined / local |
|---|---:|---:|---:|
| Q1 | 0.530 | 0.769 | 1.45 |
| Q2 | 0.136 | 0.385 | 2.83 |
| Q3 | 0.373 | 0.771 | 2.07 |
| Q4 | 0.285 | 0.615 | 2.16 |
| Q5 | 0.418 | 0.723 | 1.73 |
| Q6 | 0.214 | 0.385 | 1.80 |
| Q7 | 0.409 | 0.814 | 1.99 |
| Q8 | 0.512 | 0.920 | 1.80 |
| Q9 | 0.847 | 2.466 | 2.91 |
| Q10 | 0.480 | 1.049 | 2.18 |
| Q11 | 0.098 | 0.269 | 2.74 |
| Q12 | 0.295 | 0.556 | 1.88 |
| Q13 | 0.564 | 1.210 | 2.15 |
| Q14 | 0.305 | 0.719 | 2.36 |
| Q15 | 0.299 | 0.528 | 1.76 |
| Q16 | 0.151 | 0.645 | 4.27 |
| Q17 | 0.427 | 0.592 | 1.39 |
| Q18 | 0.754 | 2.517 | 3.34 |
| Q19 | 0.430 | 0.938 | 2.18 |
| Q20 | 0.356 | 0.945 | 2.66 |
| Q21 | 1.230 | 2.090 | 1.70 |
| Q22 | 0.146 | 0.410 | 2.80 |
| Total | **9.260** | **20.317** | **2.19** |

Separate completed native diagnostics count **16,004,050 → 659,988** accepted
channel rows, a **95.9%** reduction. Each fact input now sends 323,920 rows;
the original owner GATHER and each of two added broadcasts carry 4,049 rows.
Both runs match the local oracle and content hash, remain within the original
credit windows, and release all channel/lease bytes. Shared broadcast channels
count once; these counts are not network bytes or Ray timings. They do not
infer completed work from worker counters that are unavailable after cleanup.

The final non-editable Release build (`vane-ai 0.3.0.dev107`) passes **185
related native tests** (27 focused cases plus 158 existing regressions) and
**7 real-Ray tests**. These cover INNER/SEMI filtering, equality/NULL-safe
equality, duplicates, empty/NULL builds, 1/4 threads, 2/5 partitions, tiny
windows and cancellation; fallback for cost, preserved rows, computed/residual
keys and nested ownership; existing shared/correlated/scalar semantics and
changed-source rejection; and public pipelined/FTE execution, backpressure,
interrupt and failed-consumer retry with the same input identity and a new
fence. The initial regressions fail on the frozen runtime in all 12 work-count
cases, while eight semantic/fallback cases pass. The full release suite and
large-query FTE performance are not rerun.

Native SHA-256 identities, baseline and candidate:

- `c26c6363d27e0995b35e49a42b29a96d71647a34039064803ab1dc24e4b42b29`
- `b6e88f973bca4ebc7195950168157a95e5fcc82a013a9fee1051d24fb18b3bf9`

Evidence is retained under `build/pipelined-cte-filter-20261015/`, including
frozen runtime/configuration/input identities, raw samples, result hashes and
counts, completed native work, `comparison.json` and `local-vs-pipelined.csv`.
All query/worker/result reservations return to zero. Owned processes stop
before temporary/spill cleanup, shared oracle hashes are recorded before
deletion, source Parquet files are preserved, and all disk guards remain
untriggered. Q18 and Q9 now have the largest absolute gaps in the focused set.

## Complete Parquet groups before exchange (2026-10-15)

The fresh frozen baseline at `b460ece483` spends a median 2.439 seconds on
SF10 Q18, versus 0.700 locally. Its HAVING condition retains 624 order keys,
but 15,000,000 partial groups still cross HASH before that condition can run.
The exact two Parquet splits already have nonoverlapping order-key ranges:
the first ends at 29,982,980 and the second starts at 29,982,981. The groups
are complete within their source splits; a second aggregation is unnecessary.

The planner now proves this property from the actual encoded splits. A
direct integral grouping key must trace through only unchanged projections
and filters to one Parquet source with no incoming exchanges. All selected
row groups must have known min/max, no possible NULL key, and the file column
must match the bound type. Split ranges must be strictly disjoint, including
their endpoints. The complete native aggregate then runs in the source
fragment, allowing HAVING to remove groups before exchange. This applies to
grouping generally, without matching Q18's SQL or relying on cardinality
estimates. It also preserves ordered/DISTINCT aggregates and wide accumulator
semantics.

Overlapping ranges, duplicate file ranges, missing/nullable statistics,
computed or cast keys, generated columns, schema mappings and boundaries
through other aggregates or joins retain the original distribution. Source
assignments and version checks, FTE snapshots/retries, operator memory and
exchange credits are unchanged. This optimization removes aggregate traffic;
it does not reduce input scan rows or add dynamic split pruning.

Only Q18 and the Q9 control are timed in this round. The frozen driver and
all SF10 input identities match. Local uses 36 threads/32 GiB; pipelined uses
two 18-CPU workers, two partitions and 16 GiB operator memory per worker.
Frames remain 64 KiB/2048 rows, windows 1 MiB with 16 slots. Timings cover
submission through collection and successful close, excluding startup and
oracle comparison. OS caches remain warm on a shared host, with no task-owned
build or test overlapping the measurements.

| Query, median of three | Baseline local, seconds | Baseline pipelined | Candidate local | Candidate pipelined | Pipelined change | Candidate / local |
|---|---:|---:|---:|---:|---:|---:|
| Q18 | 0.700 | 2.439 | 0.707 | 1.144 | -53.1% | 1.62 |
| Q9 | 0.853 | 2.411 | 0.853 | 2.416 | +0.2% | 2.83 |

The completed native Q18 diagnostic reduces accepted channel rows from
**17,222,552 to 2,222,552 (87.1%)**. Exactly 15,000,000 partial-group rows
disappear; the remaining hash inputs include 1,499,999 customers, 142,848
orders and 573,177 outer lineitems. The output remains 100 rows with the same
content hash. Shared broadcast channels count once, and channel row counts
are separate from network-byte or disk-I/O measurements.

All **24 timing samples** pass schema, row-count and value comparison with
relative/absolute floating tolerances of 1e-9/1e-6, including 12 on the
candidate. Q9 is effectively unchanged in this sample and remains the larger
absolute local/pipelined gap. No new all-22-query total is inferred from these
focused runs. Validation stays within the requested related-test scope;
the all-22-query pass, full release suite and large-query FTE timings were
not rerun.

The incremental non-editable Release build (`vane-ai 0.3.0.dev108`) passes
**93 related native tests** (26 new cases and 67 existing regressions) and
**7 real-Ray tests**. Coverage includes 1/4 threads, 2/3/5 partitions,
bounded channels, unsigned/negative keys, multiple/empty/repeated files,
overlap at a boundary, NULLs, missing statistics, cast/computed/virtual keys,
file schema mappings, ordered/DISTINCT/wide aggregates, changed-source
rejection, cancellation/backpressure and FTE worker retry with identical
input identity and a new fence. The initial performance regression fails on
the frozen runtime: 8 result groups require 4,112 accepted rows rather than
the candidate's 16 rows (8 qualifying groups plus 8 final results).

Native SHA-256 identities, baseline and candidate:

- `b6e88f973bca4ebc7195950168157a95e5fcc82a013a9fee1051d24fb18b3bf9`
- `35e4e517f4f2e769143c013f0723f06d406a0fdd5c8f1550502029c30c5ec8c3`

Evidence is retained under `build/pipelined-disjoint-aggregate-20261015/`,
including the frozen runtime, code/configuration/input identities, raw
samples, output hashes/counts, completed native work, `comparison.json`,
`local-vs-pipelined.csv` and test logs. Query/worker/result reservations return
to zero. Owned processes stop before temporary/spill cleanup; oracle hashes
are retained before deletion, source datasets are preserved, and disk guards
remain untriggered.

## Share selective join builds across hash inputs (2026-10-15)

The fresh frozen baseline at `054070296b` still sends all 8,000,000 SF10
partsupp rows across Q9's final HASH join, although its other input has
already been restricted to 108,782 part keys by the green-name predicate.
The planner now reuses that complete small build to filter the opposite
input before HASH. It traces unchanged integral keys through projections,
filters and INNER joins, including an equality-equivalent probe key when
the optimizer omits the build's copy. An output build payload can supply its
own complete domain as well. The original multi-key/residual join still
decides matches; the added NULL-safe SEMI join preserves duplicates.

The shared build must have an explicit filter, a known estimate at most
8 MiB, no shared scope, and qualify for the existing small-build broadcast.
It must estimate fewer than one-eighth of target rows and less replica
traffic than the target exchange. Computed keys, LIMIT, aggregates, outer
joins, unknown nested widths and uneconomic replication retain the original
path. Existing probe-side filters take priority. A parent that can fuse its
build by broadcast, even after its probe gathers to one partition, cannot
use this extra opposite-input filter, keeping both
consumers in independent HASH inputs. The seed is planned once, with source
assignments, channel limits and FTE committed-input ownership preserved.

Only Q9 and the Q18 control are timed, three times per mode and runtime.
The frozen driver, installed Python code, resource settings and all SF10
input hashes match. Local uses 36 threads/32 GiB; pipelined uses two 18-CPU
workers, two partitions and 16 GiB operator memory per worker. Frames remain
64 KiB/2048 rows, windows 1 MiB with 16 slots. Timing includes submission,
collection and successful close, excluding startup and result comparison.
The shared host retains warm OS caches; no task-owned build or test overlaps
measurement.

| Query, median of three | Baseline local, seconds | Baseline pipelined | Candidate local | Candidate pipelined | Pipelined change | Candidate / local |
|---|---:|---:|---:|---:|---:|---:|
| Q9 | 0.875 | 2.394 | 0.856 | 2.095 | -12.5% | 2.45 |
| Q18 | 0.705 | 1.119 | 0.697 | 1.102 | -1.5% | 1.58 |

Q9 improves **12.5%**, while the Q18 control remains effectively unchanged.
The candidate Q9 samples are 2.195, 2.095 and 2.044 seconds. All **24 timing
samples** match local schema, row count and values, with floating tolerances
of 1e-9 relative and 1e-6 absolute. These focused results do not establish a
new all-22-query total or local/pipelined parity.

Completed native Q9 diagnostics reduce accepted channel rows from
**29,732,733 to 22,276,643 (25.1%)**. The partsupp input falls from 8,000,000
to **435,128 rows (94.6%)**, at the cost of one additional 108,782-row shared
build channel. Output remains 175 rows with the same content hash. Shared
broadcast channels count once; these are separate from network bytes, scan
rows and disk I/O. The 15,000,000-row orders input still crosses HASH and is
the largest remaining exchange in this diagnostic.

The incremental non-editable Release build (`vane-ai 0.3.0.dev109`) passes
**82 related native tests** and **7 real-Ray tests**. Coverage includes
1/4 threads, 2/3 partitions, tiny windows, cancellation, direct/equivalent
keys and build payloads, multi-key/residual conditions, duplicates, NULL-safe
equality, empty/all-NULL domains, preserved rows, unsupported lineage and
cost fallbacks. Two further regressions check that sorting a descendant
probe into one partition cannot fuse the shared consumers; the earlier
candidate fails both plan checks. Existing probe/key-domain/large-broadcast
behavior and source contracts remain covered. Public pipelined/FTE execution and a failed shared
consumer pass with the same input identity and a new retry fence. The Ray
fixture uses default optimizers and the original 4 MiB materialized-object
bound. Two new work-count checks fail as intended on the frozen baseline,
which sends all 262,144 target rows.

Native SHA-256 identities, baseline and candidate:

- `35e4e517f4f2e769143c013f0723f06d406a0fdd5c8f1550502029c30c5ec8c3`
- `8166bf69b4160c17ac6118ce9233c24e3a34b73a42cafd2a6d4a02158e4f22b0`

Evidence is retained under `build/pipelined-q9-20261015/`: frozen runtime,
code/configuration/input identities, raw timing samples, completed native
work, result counts/hashes, `comparison.json`, `local-vs-pipelined.csv`,
`test-validation.json` and test logs. All query/worker/result reservations
return to zero. Owned processes stop before temporary/spill cleanup; shared
oracle hashes are retained before deletion and source datasets are preserved.
Disk guards remain untriggered. The full release suite, all-22-query pass and
large-query FTE timings are not rerun, following the related-test scope.

## Reuse existing hash-key payload columns (2026-10-15)

The frozen baseline at `329e47c0f7` projects a second copy of every join or
complete-aggregate hash key into its exchange payload, including direct
references to columns already present. Q9's 15,000,000-row orders input
therefore sends `(orderkey, orderdate, orderkey)`. `ProjectKeys` now reuses
existing columns and omits the added projection when every key can be
reused. Computed keys and repeated occurrences of a key retain separate
projected positions, preserving multi-key hash order and multiplicity even
when one join side repeats a column and the other uses distinct columns.
Original column positions, join conditions and aggregate bindings stay valid.

Only related Q9/Q18 queries are timed, three times per mode and runtime.
The driver, installed Python code, SF10 inputs and resource settings match
between the frozen and candidate runtimes. Local uses 36 threads/32 GiB;
pipelined uses two 18-CPU workers, two partitions and 16 GiB operator memory
per worker. Frames remain 64 KiB/2048 rows and windows 1 MiB/16 slots.
Timing covers submission through collection and successful close, excluding
startup and comparison. The shared host retains warm OS caches; no
task-owned build, test or diagnostic overlaps timing.

| Query, median of three | Baseline local, seconds | Baseline pipelined | Candidate local | Candidate pipelined | Pipelined change | Candidate / local |
|---|---:|---:|---:|---:|---:|---:|
| Q9 | 0.857 | 2.105 | 0.856 | 1.944 | -7.6% | 2.27 |
| Q18 | 0.700 | 1.114 | 0.699 | 1.094 | -1.8% | 1.57 |

Q9 improves **7.6%**; Q18's small change does not establish a meaningful
speedup. Q9 candidate samples are 2.065, 1.904 and 1.944 seconds. All **24
timing samples** match local schema, row counts and values, with floating
tolerances of 1e-9 relative and 1e-6 absolute. There is no fresh all-query
total, and Q9 remains 2.27 times local latency.

Completed native Q9 diagnostics retain exactly **22,276,643 accepted channel
rows**, including 15,000,000 orders rows, and the same 175-row output hash.
The four affected HASH payloads shrink as follows:

| Input | Accepted rows | Columns before → after | Removed key types | Removed value-buffer bytes |
|---|---:|---:|---|---:|
| Filtered lineitem | 3,261,613 | 7 → 6 | uint32 | 13,046,452 |
| Orders | 15,000,000 | 3 → 2 | uint32 | 60,000,000 |
| Joined fact rows | 3,261,613 | 8 → 6 | two int64 | 52,185,808 |
| Filtered partsupp | 435,128 | 6 → 4 | two int64 | 6,962,048 |

The sum is **132,194,308 bytes (126.1 MiB)**, calculated from actual accepted
rows and removed fixed-width Arrow types. It excludes validity buffers and
IPC framing and is not a measurement of total network bytes. Row counts,
partitioning and scan work stay the same; the large orders HASH exchange
still remains.

The incremental non-editable Release build (`vane-ai 0.3.0.dev110`) passes
**199 related native tests** and **7 real-Ray tests**. The 32 new native
cases cover typed and nullable keys, NULL-safe/full joins, repeated and
computed keys, reordered multi-key conditions, complete aggregates and tiny
bounded channels at 1/4 threads and 2/3 partitions. On the frozen runtime,
30 cases fail the payload-width checks; two ordinary partial-aggregate
controls pass. Existing joins, aggregate/ordered/collated expressions,
key-domain filters and shared-build cancellation add 167 passing cases.
Public pipelined/FTE execution and a failed consumer preserve results and
retry input identity with a new fence, under the existing 4 MiB materialized
object limit. Source-contract, backpressure, early-close and interruption
checks also pass.

Native SHA-256 identities, baseline and candidate:

- `8166bf69b4160c17ac6118ce9233c24e3a34b73a42cafd2a6d4a02158e4f22b0`
- `18a1cb5ce5859c1eff53101177e088614bbc9ceac7c741956f8fcbbfb91d213b`

Evidence is retained under `build/pipelined-order-filter-20261015/`, including
the frozen runtime, code/configuration/input identities, raw samples,
completed native work and schemas, output counts/hashes, `comparison.json`,
`local-vs-pipelined.csv`, `test-validation.json` and logs. Query/worker/result
reservations return to zero. Owned processes stop before temporary/spill
cleanup; oracle hashes are retained before deletion, source inputs are
preserved and disk guards remain untriggered. Only related tests and Q9/Q18
timings are run; full release/all-query and large-query FTE timings are not
repeated.

## Prepared hash routing, bulk frame copies and Flight window (2026-10-15)

The frozen baseline is `ca93b80d5c`. Task preparation now validates each HASH
output once and retains its ordered column indices. Batches use native
Hash/CombineHash with scratch owned by each sink task, without re-decoding
expressions or retaining input leases. Fixed-width, flat, fully valid columns
with a validated contiguous row selection copy into the owned frame in one
span. Nullable, selected, constant and nested vectors retain the existing
value-wise path; frame layout, byte limits and ownership stay unchanged.

Separate completed native Q9 diagnostics measured cumulative thread CPU
inside identical temporary probes, one run per build. Hash routing fell from **1.742 to 0.679
CPU seconds** and frame allocation/copy from **1.544 to 0.706 CPU seconds**.
These are scoped in-process CPU measurements, not Ray query latency or total
worker CPU. Probes were removed before acceptance. CPU-only Q9 timings
changed from 1.980 to 1.953 seconds across six samples per runtime, with
opposite signs in forward/reverse rounds: they do not establish a latency
speedup. The hash-only exploration is retained separately.

A separate Ray Q9 timeline found orders finished at 1.203 seconds while
filtered-fact and joined-fact remote channels still held 16 outstanding
frames, including 12 queued frames. Their four unacknowledged Flight frames
limited progress. Raising that bound to **eight frames** lets the reader
continue further while cumulative ACKs travel on the control connection.
The ninth frame still waits, and native byte/slot credit continues to bound
all outstanding data. Sequence checks, cancellation and FINISH are unchanged.

This increases each link's staging reservation from
`32 * frame_bytes + 256 KiB` to `48 * frame_bytes + 256 KiB`: **2.25 to
3.25 MiB** at 64 KiB frames. Configured worker capacity is unchanged and
admission charges the larger requirement before execution. A graph that
previously fit a smaller staging capacity can now be rejected. Q9 has 20/19
remote links per worker; reservations increase from 45/42.75 to 65/61.75 MiB,
within the same declared 256 MiB per-worker capacity. Local channels have no
Flight staging. This is an explicit throughput/memory tradeoff, not a
reduction in total reserved memory.

Only Q9/Q18 are timed. The final comparison uses a baseline/candidate round
followed by candidate/baseline, three samples per query/mode/runtime in each
round. Driver, installed Python, SF10 input hashes and declared resources
match: local 36 threads/32 GiB; two pipelined 18-CPU workers, two partitions
and 16 GiB operator memory per worker; 64 KiB/2048-row frames, 1 MiB windows
and 16 slots. Submission through collection and successful close is timed;
startup and comparison are excluded. OS caches remain warm on the shared
host, and no task-owned build/test/diagnostic overlaps the timing windows.

| Query, median of six | Baseline local, seconds | Baseline pipelined | Candidate local | Candidate pipelined | Pipelined change | Candidate / local |
|---|---:|---:|---:|---:|---:|---:|
| Q9 | 0.857 | 1.930 | 0.861 | 1.601 | -17.1% | 1.86 |
| Q18 | 0.693 | 1.092 | 0.716 | 1.062 | -2.8% | 1.48 |

Q9 improves in both rounds: 1.928 to 1.620 seconds and 1.932 to 1.588
seconds. Q18's smaller change remains modest relative to shared-host
variation. All **48 final comparison samples**, including **24 on the final
build**, match local schema, row counts and values (relative/absolute float
tolerances 1e-9/1e-6). Earlier CPU/hash explorations bring retained timing
samples to 96; they are not pooled into the final candidate median.
Completed native Q9 work retains **22,276,643 accepted channel rows**,
identical per-channel counts/schemas and the same 175-row output hash.
The 15,000,000 orders rows still cross HASH. No new all-query total or
local parity is inferred.

The incremental non-editable Release build (`vane-ai 0.3.0.dev111`) passes
**270 related native tests**. New cases cover different simultaneous HASH
outputs and private scratch at 1/4 threads and 3/5 partitions, malformed
partitioning rejected before task start, fixed-width/nullable/constant/filtered
values at 63/64/65-row frame boundaries, and Flight's eight-frame progress,
ninth-frame blocking, duplicate ACKs, FINISH, close and staging admission.
The frozen baseline times out in the new window test before reaching eight
frames. Existing tiny-window, wakeup/cancellation, nullable Parquet, type,
materialized-input and Flight failure regressions pass. **11 real-Ray tests**
also pass: pipelined/FTE reused-key execution and failed-consumer retry,
analytical types, source contracts, slow-client backpressure/early close,
interruption/delivery timeout, and admission rollback for insufficient I/O
or staging capacity. Retry keeps input identity and changes the fence.
Formatting, applicable pinned pre-commit hooks, copyleft inventory and
`git diff --check` pass.

Native SHA-256 identities, baseline and final candidate:

- `18a1cb5ce5859c1eff53101177e088614bbc9ceac7c741956f8fcbbfb91d213b`
- `9c2c8cf4b228d8dec52f11e04d100698027e019e372f7f67e862725d98b8542d`

Evidence is retained under `build/pipelined-exchange-cpu-20261015/`: frozen
runtime, source/build/configuration/input identities, samples, CPU diagnostic
patches and logs, native work/schema reports, `comparison.json`,
`local-vs-pipelined.csv`, test logs and cleanup manifests. Temporary probes
are absent from production sources. All measured query/worker/result
reservations return to zero; owned processes stop before temporary/spill
cleanup. Shared oracle hashes are retained before deletion, source inputs
are preserved and disk guards remain untriggered. Only related tests and
Q9/Q18 timings are run; the full release/all-query suites and large-query
FTE timings are not repeated.


## Compact decimal exchange and one full decode validation (2026-10-15)

The frozen baseline is `5791895980`. Internal precision 1–9 DECIMAL columns
now use Arrow decimal32 and precision 10–18 use decimal64, preserving scale
and nulls recursively through containers. Precision 19–38 keeps decimal256
with precision 39 for full signed-128-bit aggregate intermediates. Public
DECIMAL results remain decimal128 at their declared SQL precision, including
empty and nested results. The native v1.5 exporter provides the compact
coefficients; its binary views are normalized to the existing binary layout
for BLOB and aggregate states. The common decoder performs full Arrow
validation once; Flight and materialized reads omit their duplicate pass.
Sequence, byte/row limits, checksums and materialized seals remain checked.

This changes the exchange type profile from `vane.analytical-types:4` to `:5`.
Workers with the previous profile are rejected before execution, and the
codec participates in fragment build identity. Frame, window, slot and
staging limits are unchanged in this round.

A separate completed Q9 diagnostic on the baseline measured **0.706 CPU
seconds** in encoding and **0.565 CPU seconds** in common decoding across
worker processes. The redundant outer Flight validation used **0.117 CPU
seconds**, in addition to **0.102 CPU seconds** for full validation inside
decode. These scoped thread-CPU measurements include a small warmup; nested
probes overlap and cannot be summed into whole-query CPU or latency. Probes
were removed before acceptance. There is no measured candidate CPU delta.

The final comparison uses baseline/candidate followed by candidate/baseline,
three samples per query/mode/runtime in each round. Only Q9/Q18 are timed.
The frozen driver, SF10 input hashes and declared resources match: local
36 threads/32 GiB; two pipelined 18-CPU workers, two partitions and 16 GiB
operator memory per worker; 64 KiB/2048-row frames, 1 MiB/16-slot windows,
eight Flight frames, 128 MiB exchange and 256 MiB staging per worker.
Installed Python differs only in the type-profile version. Timing includes
submission, collection and successful close, excluding startup/comparison.
OS caches remain warm on a shared host; no task-owned build, test or
diagnostic overlaps a timing window.

| Query, median of six | Baseline local, seconds | Baseline pipelined | Candidate local | Candidate pipelined | Pipelined change | Candidate / local |
|---|---:|---:|---:|---:|---:|---:|
| Q9 | 0.871 | 1.620 | 0.855 | 1.589 | -1.9% | 1.86 |
| Q18 | 0.699 | 1.082 | 0.691 | 1.069 | -1.2% | 1.55 |

Q9 improves 2.5% and 2.3% in the two rounds, but its local control also
improves about 1.8% across all six samples. These small changes do **not**
establish a meaningful narrowing of the pipelined/local gap. Q18's result
is similarly modest. All **48 comparison samples**, including **24 on the
candidate**, match local schema, rows and values (relative/absolute float
tolerances 1e-9/1e-6). There is no new all-query total or FTE latency claim.

Completed native Q9 diagnostics retain **22,276,643 accepted channel rows**,
identical per-channel counts and public schemas, and the same 175-row output
hash. Observed Flight wire types and actual completed channel counts show:

| Input | Accepted rows | Decimal columns narrowed | Removed value-buffer bytes |
|---|---:|---:|---:|
| Filtered lineitem | 3,261,613 | Three decimal128 → decimal64 | 78,278,712 |
| Joined fact rows | 3,261,613 | Three decimal128 → decimal64 | 78,278,712 |
| Filtered partsupp | 435,128 | One decimal128 → decimal64 | 3,481,024 |

The corresponding exchange decimal value buffers shrink by **160,038,448
bytes (152.6 MiB)**. This calculation includes logical channel rows, including
local channels that do not encode Arrow, and excludes validity buffers and
IPC framing. It is not measured network traffic. The 15,000,000-row orders
HASH exchange and overall row work remain.

The incremental non-editable Release build (`vane-ai 0.3.0.dev112`) passes
**359 related native tests** and **13 real-Ray tests**. New cases check compact
wire widths, boundary coefficients, NULLs, empty/nested/selected values,
embedded BLOBs, unchanged public schemas and rejection of corrupt decimal
coefficients/UTF-8 before publishing rows. Existing full-width intermediates,
temporal/analytical types, bounded exchanges, Flight failures and materialized
integrity checks pass. Ray covers analytical/partial aggregate states,
pipelined/FTE execution and failed-consumer retry, remote public schemas,
slow-client backpressure/early close and interruption/delivery timeout.
Two new wire-width cases fail on the frozen baseline as expected.

Native SHA-256 identities, baseline and accepted candidate:

- `9c2c8cf4b228d8dec52f11e04d100698027e019e372f7f67e862725d98b8542d`
- `c37910d5a5c53e28f04c49dec7f6fee72c91c3c90a6b6257e537ce38d80c3662`

Evidence is retained under `build/pipelined-arrow-codec-20261015/`: frozen
runtime, source/build/configuration/input identities, raw samples, CPU
diagnostic patch/logs, completed native work and actual wire schemas,
`comparison.json`, `local-vs-pipelined.csv`, test logs and cleanup manifests.
All measured query/worker/result reservations return to zero. Owned
processes stop before temporary/spill cleanup; oracle hashes are retained
before deletion, source inputs are preserved and disk guards remain
untriggered. Only related tests and Q9/Q18 timings are run; the full release,
all-query suites and large-query FTE timings are not repeated.

A separate ACK-notification experiment consumes a pending notification even
when receipt advances during a control RPC. Three controlled traces remove
an immediate redundant status request, and 52 Flight tests pass. Its one
three-sample Q9 comparison is 1.597 seconds versus 1.582 without the change,
with local controls 0.847/0.840 seconds. This does not establish a latency
benefit, so the ACK change is not retained. Its 12 matching samples, patch,
identities and cleanup are retained separately and excluded from the
accepted candidate medians.


## Sixteen-frame Flight window (2026-10-15)

The frozen baseline is `f907c138f4`. A separate Q9 timeline shows the orders
exchange completing around 0.8 seconds, followed by roughly 0.6 seconds of
filtered/joined fact transfer. Remote fact outputs repeatedly hold eight
unacknowledged frames and another eight queued frames while their receivers
usually have little queued data. Temporary per-stream probes measure
0.272/0.280 seconds waiting for the Flight window on the two joined-fact
senders. With sixteen frames, those waits fall to 0.031/0.036 seconds in a
separate completed diagnostic run. Corresponding stream spans change from
0.629/0.629 to 0.589/0.596 seconds. These per-stream wall times overlap
upstream computation and control traffic; they are not additive query
latency or an isolated estimate of ACK cost. Row counts and wire schemas
match. Diagnostic probes are removed before the final build is restored.

Flight now permits **sixteen unacknowledged frames**; the seventeenth waits.
Existing native byte/slot credit, cumulative ACKs, FINISH and cancellation
still govern ownership and completion. Staging is charged as
`80 * frame_bytes + 256 KiB` per remote link: **5.25 MiB** at 64 KiB frames,
up from 3.25 MiB. Q9's 20/19 remote links require **105/99.75 MiB** per
worker instead of 65/61.75 MiB, within the same configured 256 MiB capacity.
Previously sufficient smaller capacities can now be rejected before
execution. This is a throughput/memory tradeoff; the configured worker
capacity and native channel limits are unchanged.

Only Q9/Q18 are timed, in baseline/candidate then candidate/baseline order,
with three samples per query/mode/runtime in each round. Driver, installed
Python, SF10 input hashes and declared resources match: local 36 threads/
32 GiB; two pipelined 18-CPU workers, two partitions and 16 GiB operator
memory per worker; 64 KiB/2048-row frames, 1 MiB/16-slot channels, 128 MiB
exchange and 256 MiB staging per worker, and a 1 GiB Ray object store.
Ray's automatically reported available host memory varies by about
0.3 GiB; operator limits remain fixed. Timings include submission,
collection and successful close, excluding startup/comparison. The shared
host has warm OS caches; no task-owned build/test/diagnostic overlaps a
measured comparison window.

| Query, median of six | Baseline local, seconds | Baseline pipelined | Candidate local | Candidate pipelined | Pipelined change | Candidate / local |
|---|---:|---:|---:|---:|---:|---:|
| Q9 | 0.857 | 1.609 | 0.859 | 1.495 | -7.1% | 1.74 |
| Q18 | 0.696 | 1.076 | 0.715 | 1.069 | -0.7% | 1.49 |

Q9 improves in both rounds, from 1.583 to 1.495 seconds and from 1.627 to
1.496 seconds; its combined local control is essentially unchanged. Q18's
small difference and opposite signs across rounds do not establish a
latency benefit. All **48 comparison samples**, including **24 on the
accepted candidate**, match local schema, row counts and values with
relative/absolute floating tolerances 1e-9/1e-6. No new all-query total,
large-query FTE result or local parity is inferred.

The incremental non-editable Release build (`vane-ai 0.3.0.dev113`) passes
**53 related native Flight tests** and **13 real-Ray tests**. They cover the
sixteen-frame window, seventeenth-frame blocking, duplicate cumulative ACKs,
FINISH/close, insufficient staging, malformed data/control replies, process
loss and lease retirement. Ray checks public/empty results, tiny-window
backpressure/early close, worker/result-process loss, admission rollback,
interrupt/delivery timeout, HASH/Parquet routing, pipelined/FTE shared-input
retry and retained remote client views. Two tests fail on the frozen
baseline as expected: receiving sixteen frames without ACK times out, and
the old 6.5 MiB two-link staging budget is still admitted.

Native SHA-256 identities, baseline and final candidate:

- `c37910d5a5c53e28f04c49dec7f6fee72c91c3c90a6b6257e537ce38d80c3662`
- `0972960bda0376fc02b514419cfa834e79120260cbbeb84b6236617adb3ef365`

Evidence is retained under `build/pipelined-flight-tail-20261015/`: frozen
runtime, source/configuration/input identities, raw samples, timeline and
per-stream diagnostic patches/metrics, `comparison.json`,
`local-vs-pipelined.csv`, test logs and cleanup manifests. All measured
query/worker/result reservations return to zero. Owned processes stop
before temporary/spill cleanup; oracle hashes are retained before deletion,
source inputs are preserved and disk guards remain untriggered. Formatting,
applicable pre-commit hooks, copyleft and diff checks pass. The full release,
fast and all-query suites are not rerun, following the related-test scope.


## Bounded source wakeups (2026-10-16)

The frozen baseline is `75c3954c5e`. A separate instrumented Q9 run shows
67,108 empty source polls returning BLOCKED for 17,138 consumed frames:
each publication wakes the entire source task group, which competes for
the source lock before most tasks block again. The accepted implementation
wakes at most two readers per publication. A reader that obtains a frame
hands off to one more waiter before running downstream operators, so
queued frames can use more readers even without another publication.
Completion and errors wake the whole group. The source's shared waiter
state also participates in early pipeline termination. If a forced
coalescer flush blocks on sink credit after registering an input wait,
the executor drains the source waiters before yielding, preventing an
obsolete input registration from consuming a later publication callback.

With the accepted implementation, a separate Q9 diagnostic records
24,313 source BLOCKED returns for 16,988 frames, **63.8% fewer blocked
polls**. Coalescing changes frame counts; result schemas, row counts and
values still match. Aggregate source-lock acquisition wall time falls
from 3.738 to 1.385 seconds across all source tasks. These instrumented
times overlap across threads and cannot be added to query latency. The
source callback's task-wakeup counter excludes the executor's new
coalescer/termination wakeups, so it is not used to claim a reduction in
total scheduler wakeups. Probes are removed before restoring the accepted
production build; diagnostic runs are excluded from latency medians.

Only Q9/Q18 are timed, in baseline/candidate then candidate/baseline order,
with three samples per query/mode/runtime in each round. Driver, installed
Python, SF10 input hashes and declared resources match: local 36 threads/
32 GiB; two pipelined 18-CPU workers, two partitions and 16 GiB operator
memory per worker; 64 KiB/2048-row frames, 1 MiB/16-slot channels, 128 MiB
exchange and 256 MiB staging per worker, and a 1 GiB Ray object store.
Both builds use the sixteen-frame Flight window and 5.25 MiB staging per
remote link. Ray's automatically reported available host memory varies
by about 0.14 GiB; declared operator limits remain fixed. Timings include
submission, collection and successful close, excluding startup/comparison.
The shared host has warm OS caches; no task-owned build/test/diagnostic
overlaps a measured comparison window.

| Query, median of six | Baseline local, seconds | Baseline pipelined | Candidate local | Candidate pipelined | Pipelined change | Candidate / local |
|---|---:|---:|---:|---:|---:|---:|
| Q9 | 0.866 | 1.501 | 0.861 | 1.430 | -4.7% | 1.66 |
| Q18 | 0.691 | 1.064 | 0.712 | 1.070 | +0.6% | 1.50 |

Q9 improves in both rounds, from 1.497 to 1.392 seconds and from 1.506 to
1.460 seconds; its combined local control changes by -0.6%. Q18 has no
established latency improvement. All **48 comparison samples**, including
**24 on the accepted candidate**, match local schema, row counts and
values with relative/absolute floating tolerances 1e-9/1e-6. These results
do not establish local parity, an all-query total or large-query FTE
performance. A one-reader publication variant had inconsistent Q9 gains
and a 2.6% Q18 slowdown; it is rejected and its measurements are retained
separately, as are preliminary and screening runs.

The incremental non-editable Release build (`vane-ai 0.3.0.dev114`) passes
**215 related native exchange/Flight tests** and **18 real-Ray tests**.
New regressions exercise sparse-frame wakeup bounds, queued-frame handoff,
coalescer input-to-output blocking, normal completion, failure and early
sink termination at 4/16 threads. The sparse-frame wakeup check fails on
the frozen baseline as expected: one frame produces 16 new blocked polls
instead of at most three. Ray covers public/empty results, backpressure/
early close, worker/result-process loss, staging/I/O admission rollback,
interrupt/delivery timeout, HASH/Parquet routing, pipelined/FTE shared-input
retry, budgeted broadcast and retained client views. The broadcast budget
test explicitly reserves 64 MiB staging: its eight remote endpoints need
42 MiB with the previously introduced sixteen-frame window, exceeding
the fixture's 32 MiB default. Production capacities and rejection of
insufficient budgets are unchanged.

Native SHA-256 identities, baseline and accepted candidate:

- `0972960bda0376fc02b514419cfa834e79120260cbbeb84b6236617adb3ef365`
- `9fce9b0c9f764f9cc5a0ae9b4eb4aa1bd9031ba128a1da633de8ece6dbc8a981`

The accepted DuckDB SourceID is
`ff00fa7d41e2817abdc7e24a48081472f2a351f8`. Evidence is retained under
`build/pipelined-wakeup-20261015/`: frozen runtime/source/configuration/input
identities, raw samples, separate diagnostic patches and metrics,
`pair-comparison.json`, `pair-local-vs-pipelined.csv`, test logs and cleanup
manifests. The `pair-*` artifacts identify the accepted two-reader variant;
the earlier `candidate-*` and `comparison.json` describe the rejected
one-reader variant. All measured query/worker/result reservations return
to zero. Owned processes stop before temporary/spill cleanup; oracle
hashes are retained before deletion, source inputs are preserved and disk
guards remain untriggered. Formatting, applicable pre-commit hooks,
copyleft and diff checks pass. Only related tests and queries are run;
the full release, fast and all-query suites are not repeated.

## Batched pipelined worker starts (2026-10-16)

The frozen baseline is `7f2f9f59aa`. After every worker and result endpoint
is connected and ready, the coordinator previously waited for start RPCs
at each fragment boundary. Q9/Q18 required 19/23 RPCs for their 10/13
fragments on two workers. The scheduler now sends one batch per worker,
preserving reverse topological order within each batch. Native start does
not wait for exchange input; bounded channels can hold data until the peer
starts. Preparation, source validation, admission, fixed attempts/splits,
readiness checks, cancellation and resource ownership retain their contracts.
A failed or ambiguous start still releases every prepared worker, including
tasks whose batch has only partially started.

Separate coordinator diagnostics, three samples per query/runtime, record
start RPC counts of **19 to 2 for Q9** and **23 to 2 for Q18**. Median
cumulative waits for their replies fall from **39.2 to 2.8 ms** and
**19.2 to 3.2 ms**, respectively. These diagnostics are excluded from the
latency comparison. In the uninstrumented comparison, median submission
time falls from **180.4 to 153.0 ms** for Q9 and **228.9 to 201.2 ms** for
Q18, with lower submission medians in both rounds.

The comparison uses baseline/candidate then candidate/baseline order,
with three samples per query/mode/runtime in each round. SF10 input hashes,
driver and declared resources match: local 36 threads/32 GiB; two pipelined
18-CPU workers, two partitions and 16 GiB operator memory per worker;
64 KiB/2048-row frames, 1 MiB/16-slot channels, 128 MiB exchange and 256 MiB
staging per worker, and a 1 GiB Ray object store. The Flight window remains
sixteen frames and staging remains 5.25 MiB per remote link. Ray's reported
available host memory varies by about 0.22 GiB; declared operator limits
are fixed. Timings include submission, collection and successful close,
excluding startup and validation, with warm OS caches on the shared host.
No task-owned build, test or diagnostic overlaps a measured comparison.

| Query, median of six | Baseline local, seconds | Baseline pipelined | Candidate local | Candidate pipelined | Pipelined change | Candidate / local |
|---|---:|---:|---:|---:|---:|---:|
| Q9 | 0.851 | 1.447 | 0.862 | 1.414 | -2.3% | 1.64 |
| Q18 | 0.697 | 1.094 | 0.701 | 1.067 | -2.5% | 1.52 |

Q18 improves in both rounds, 1.092 to 1.066 seconds and 1.096 to 1.068
seconds. Q9 changes from 1.452 to 1.403 seconds in the first round but
1.442 to 1.450 seconds in the reverse round: its combined median is lower,
but a consistent end-to-end improvement is not established. Combined
local controls change by +1.2%/+0.5% for Q9/Q18. All **48 timing samples**,
including **24 on the candidate**, match local schema, row counts and
values with relative/absolute floating tolerances 1e-9/1e-6. This is a
startup improvement; local parity, an all-query total and large-query FTE
performance are not established.

The non-editable incremental Release install (`vane-ai 0.3.0.dev115`) and
frozen baseline have the same native SHA-256:
`c876bd8ff35b0d994182586544b98725323693f525b7c3b31b90795a82ec5bf2`.
Only the installed `execution/pipelined_runtime.py` differs. The baseline
was rebuilt from clean `7f2f9f59aa` sources before freezing; its build
identity differs from the preceding round's binary. DuckDB SourceID is
still `ff00fa7d41e2817abdc7e24a48081472f2a351f8`.

**21 related real-Ray tests pass**, including three new cases for a
delayed peer under tiny channel limits, a lost start reply after native
startup and a failure partway through a batch before its pump thread
exists. Each verifies result correctness or failure propagation, complete
release and pool reuse. The batching regression fails on the baseline's
single-fragment start as expected. Existing coverage includes backpressure,
early close, process loss, admission rollback, cancellation/deadlines,
HASH/Parquet, shared build/CTE inputs and budgeted broadcast. Formatting,
applicable pre-commit hooks, copyleft and diff checks pass. Only related
tests and Q9/Q18 are run; the full release, fast and all-query suites are
not repeated.

Two preceding receiver experiments are rejected and fully reverted:
borrowing fixed-width Arrow values did not improve Q9 latency despite
lower decode CPU, and reusing expected schemas/removing duplicate frame
fitting left latency effectively unchanged. A 128 KiB frame screen showed
no established gain sufficient to justify doubling per-link staging from
5.25 to 10.25 MiB. Their patches, identities, tests, diagnostics and
measurements remain under `build/pipelined-receive-20261016/`; none of those
native changes or frame settings is included in this round.

Accepted evidence is under `build/pipelined-start-20261016/`: baseline and
candidate identities, frozen configurations, raw samples, `comparison.json`,
`local-vs-pipelined.csv`, separate `control-comparison.json`, regression and
check logs, and cleanup manifests. Query/worker/result reservations return
to zero. Owned processes stop before temporary/spill cleanup; oracle row
counts, schemas and hashes are retained before output deletion. Rejected
binary copies are removed, source inputs are preserved, and disk guards
remain untriggered, with about 74 GiB free after cleanup.

## Concurrent pipelined worker connections (2026-10-16)

The frozen baseline is `f9ec2c4daa`. After all workers finish preparation
and publish their endpoints, the scheduler now dispatches every worker's
input connection RPC before awaiting the replies. Previously it waited
for each worker before dispatching the next. Result connections, readiness
checks and batched starts still follow successful worker connections.
A failed or ambiguous connection reply releases the whole prepared group,
including workers whose connection calls remain in flight.

Separate coordinator diagnostics, three samples per query/runtime, show
median cumulative connection reply waits falling from **13.3 to 8.8 ms for
Q9** and **14.5 to 10.3 ms for Q18**. There are still two RPCs per query.
These diagnostics establish less serial connection waiting; the
uninstrumented comparison does **not establish a consistent end-to-end
or overall submission improvement**.

The comparison runs baseline/candidate and then candidate/baseline, with
three samples per query/mode/runtime in each round. SF10 input hashes,
driver and declared resources match: local 36 threads/32 GiB; two pipelined
18-CPU workers, two partitions and 16 GiB operator memory per worker;
64 KiB/2048-row frames, 1 MiB/16-slot channels, 128 MiB exchange and 256 MiB
staging per worker, and a 1 GiB Ray object store. The Flight window remains
sixteen frames and staging remains 5.25 MiB per remote link. Ray's reported
available host memory varies by about 0.36 GiB. Timings include submission,
collection and successful close, excluding startup and validation, with
warm OS caches on the shared host. No task-owned build, test or diagnostic
overlaps the measured comparison.

| Query, median of six | Baseline local, seconds | Baseline pipelined | Candidate local | Candidate pipelined | Pipelined change | Candidate / local |
|---|---:|---:|---:|---:|---:|---:|
| Q9 | 0.855 | 1.419 | 0.860 | 1.410 | -0.7% | 1.64 |
| Q18 | 0.696 | 1.044 | 0.693 | 1.045 | +0.2% | 1.51 |

Q9 changes from 1.424 to 1.389 seconds in the first round and 1.414 to
1.418 seconds in the reverse round. Q18 changes from 1.034 to 1.047
seconds and 1.054 to 1.038 seconds. Both queries' rounds disagree.
Combined local controls change by +0.6%/-0.4% for Q9/Q18. Uninstrumented
submission medians increase from **165.1 to 167.2 ms** and **188.8 to
196.3 ms**, respectively. The isolated connection saving is small relative
to preparation and run-to-run variation; it does not demonstrate a narrower
gap to local. All **48 timing samples**, including **24 on the candidate**,
match local schemas, row counts and values with relative/absolute floating
tolerances 1e-9/1e-6. All-query and large-query FTE timings are not repeated.

A separate baseline worker probe measures three repetitions per query on
two workers, six worker observations per query. Median preparation wall
times are:

| Per-worker stage, milliseconds | Q9 | Q18 |
|---|---:|---:|
| Fresh database connection | 38.5 | 35.5 |
| Whole-plan validation | 8.6 | 9.2 |
| Flight service creation | 1.7 | 1.6 |
| Channel setup | 0.7 | 0.7 |
| Task preparation loop | 6.5 | 6.1 |
| Total query preparation | 56.4 | 54.1 |

Thread CPU time for database creation is close to its wall time, pointing
to initialization work rather than waiting for thread startup. Native
fragment inspection accounts for about 6.3/6.4 ms within whole-plan
validation, and native task preparation for 6.1/5.7 ms within the task loop;
these nested intervals must not be added to their parents. Worker intervals
overlap across workers and are not additive query latency. The probes use
temporary coordinator wrappers and a worker setup hook, without changing
the installed source; all instrumented samples are excluded from the
timing table. Database creation remains the main worker preparation cost.
This change does not reuse databases or alter query isolation or memory
ownership.

The non-editable incremental Release install (`vane-ai 0.3.0.dev116`) and
frozen baseline have identical native SHA-256
`c876bd8ff35b0d994182586544b98725323693f525b7c3b31b90795a82ec5bf2`.
Only installed `execution/pipelined_runtime.py` differs. DuckDB SourceID
remains `ff00fa7d41e2817abdc7e24a48081472f2a351f8`.
**23 related real-Ray tests pass**, including two new connection cases
that check dispatch before waiting, a lost reply with other connections
in flight, no premature starts, complete release and pool reuse. The
dispatch regression fails against the baseline as expected. Existing
coverage includes delayed/partial starts, backpressure, early close,
process loss, admission rollback, cancellation/deadlines, HASH/Parquet,
shared build/CTE inputs and budgeted broadcast. Only affected tests run;
the full release and fast suites are not repeated. Formatting, applicable
pre-commit hooks, copyleft and diff checks pass.

Evidence is retained under `build/pipelined-prepare-20261016/`: frozen
identities/configurations, raw samples, `comparison.json`,
`local-vs-pipelined.csv`, separate `connect-comparison.json` and
`prepare-breakdown.json`, probe sources, test logs and cleanup manifests.
The accepted worker breakdown uses `baseline-diagnostic-complete`; two
earlier harness setup/oracle errors are recorded and excluded. All measured
query/worker/result reservations return to zero. Owned processes stop
before temporary/spill cleanup. Oracle schemas, row counts and hashes
are retained before output deletion, source inputs are preserved and
disk guards remain untriggered, with about 74 GiB free after cleanup.


## Complete Parquet groups through broadcast joins (2026-10-21)

A broadcast join preserves the probe scan's exact Parquet split assignments.
The planner now retains that scan lineage and proves complete groups through
probe output references, filters and checked integral compression/decompression.
When one grouping key has non-null, disjoint ranges across the assigned splits,
all rows of each group already belong to one partition. The native aggregate,
HAVING and partial Top-N can finish there before the final global merge.
Previously, any incoming exchange or internal compression projection discarded
this proof and introduced a partial-state hash exchange and merge aggregate.

Every exchange clears scan lineage. The proof follows only probe columns of
INNER, LEFT, SEMI, ANTI and MARK joins; build output columns, FULL/RIGHT joins,
scoped state, arbitrary casts and computed keys do not establish ownership.
Internal integral transformations require the builtin function signature and a
constant offset. Checked endpoint arithmetic against the actual file bounds
proves every transformation is injective without narrowing overflow. Overlapping
ranges, NULLs, repeated inputs, missing statistics and schema mappings retain
the ordinary distributed aggregation. There is no SQL, table or query-number
special case, cross-query plan cache, or new execution mode.

The initial full comparison identified Q9, Q21 and Q10 as the largest absolute
pipelined/local gaps. Separate diagnostics show compilation plus scheduler
preparation at about 103/123/85 ms respectively. Worker preparation is nested
inside those scheduler waits, with concurrent workers; the durations are not
summed again. Task timelines and producer-side terminal channel counters show:

- Q9 still hashes 15,000,000 order rows into its first large join, then hashes
  3,261,613 joined rows toward its next join. Its fragment graph is unchanged.
- Q21 still has successive correlated build barriers and hash exchanges. Its
  fragment graph and per-edge row counts are unchanged.
- Q10 removes a hash exchange carrying **381,105 wide aggregate-state rows**
  in **2,098 frames** in the diagnostic sample. That edge recorded 4,220
  blocked publications. The graph shrinks from six fragments to five; only
  **40 local Top-N rows** need the final gather, followed by the global Top-20.
  The existing build broadcasts retain the same row counts.

These are channel publication rows and frames, counting each outgoing route
once, including local routes and broadcast replicas. They are not encoded wire
bytes; channel current/peak bytes do not measure cumulative traffic. Diagnostic
runs are excluded from all timing claims.

The frozen baseline is the merged integration tree at `73e90088f0` (identical
to the previously built `348fb5d789` tree). The non-editable Release binaries are
identified by SHA-256: baseline
`b6955e3a3b325d44752e0d1cba79a43d100c2c053e5de1859e7d055eae3f5fd0`,
candidate
`f265a1d4d496f02320ef2b4ac138045571c78b9384cc474fdffd551dbe437382`.
Their DuckDB SourceID prefixes are `ff00fa7d41` and `54dcd9e0af`.
Installed Python sources are identical between the two environments.

Both use the unchanged SF10 SQL with AS NOT MATERIALIZED Parquet input CTEs:
local 36 threads/32 GiB; two pipelined workers with 18 CPUs and 16 GiB operator
memory each; two partitions; 64 KiB/2048-row frames; 1 MiB/16-slot channels;
128 MiB exchange and 256 MiB staging per worker; a 1 GiB Ray object store.
Input hashes, driver and declared resources match. Each runtime receives a
SELECT 1 warmup. The first execution of each SQL is recorded separately, and
steady medians exclude it. Timings include submission, complete collection and
successful close, excluding result comparison and diagnostics. OS caches stay
warm on the shared host; no task-owned build, test or diagnostic overlaps a
measured run. Cold service startup and SF10 FTE performance are not measured.

The focused acceptance uses baseline/candidate/candidate/baseline order, with
local/pipelined then pipelined/local mode orders. Each query has eight steady
samples per runtime/mode, plus its separately recorded first executions:

| Query | Baseline local, s | Baseline pipelined, s | Candidate local, s | Candidate pipelined, s | Pipelined change |
|---|---:|---:|---:|---:|---:|
| Q9 | 0.841 | 1.273 | 0.844 | 1.283 | +0.7% |
| Q21 | 1.163 | 1.518 | 1.174 | 1.503 | -1.0% |
| Q10 | 0.472 | 0.780 | 0.476 | 0.631 | -19.1% |

Q10 improves in both orders: 0.760 to 0.635 seconds (-16.4%) and 0.787 to
0.627 seconds (-20.3%). Its combined steady median improves **19.1%** and is
**1.32 times** its matching local control. Q9/Q21 do not establish a material
performance improvement from this change.

The complete 22-query comparison uses six steady samples per mode/version,
plus first executions, with forward and reverse mode order. Those full-matrix
baseline rounds precede the candidate rounds; the focused interleaved runs
above support the optimization claim. Full-matrix steady medians are:

| Query | Candidate local, s | Baseline pipelined, s | Candidate pipelined, s | Candidate / local |
|---|---:|---:|---:|---:|
| Q1 | 0.457 | 0.565 | 0.563 | 1.23 |
| Q2 | 0.132 | 0.250 | 0.259 | 1.96 |
| Q3 | 0.364 | 0.542 | 0.553 | 1.52 |
| Q4 | 0.276 | 0.508 | 0.505 | 1.83 |
| Q5 | 0.409 | 0.574 | 0.578 | 1.41 |
| Q6 | 0.211 | 0.284 | 0.288 | 1.36 |
| Q7 | 0.404 | 0.621 | 0.632 | 1.57 |
| Q8 | 0.499 | 0.701 | 0.699 | 1.40 |
| Q9 | 0.852 | 1.269 | 1.270 | 1.49 |
| Q10 | 0.470 | 0.773 | 0.624 | 1.33 |
| Q11 | 0.096 | 0.166 | 0.171 | 1.78 |
| Q12 | 0.288 | 0.434 | 0.425 | 1.48 |
| Q13 | 0.533 | 0.759 | 0.753 | 1.41 |
| Q14 | 0.300 | 0.501 | 0.495 | 1.65 |
| Q15 | 0.301 | 0.395 | 0.407 | 1.35 |
| Q16 | 0.157 | 0.329 | 0.318 | 2.03 |
| Q17 | 0.382 | 0.475 | 0.470 | 1.23 |
| Q18 | 0.688 | 0.921 | 0.908 | 1.32 |
| Q19 | 0.418 | 0.654 | 0.663 | 1.59 |
| Q20 | 0.341 | 0.567 | 0.570 | 1.67 |
| Q21 | 1.181 | 1.550 | 1.533 | 1.30 |
| Q22 | 0.143 | 0.246 | 0.244 | 1.71 |

The sum of per-query steady medians is **13.09 to
12.93 seconds** for pipelined, with candidate local
**8.90 seconds**: **1.45 times** local.
This is a sum of medians, not a throughput measurement. All **824 accepted timing
executions** (704 full-matrix and 120 focused) match the local oracle's schema,
row counts and values, with floating relative/absolute tolerances 1e-9/1e-6.
Q11 returns zero rows with the unchanged SF10 threshold. Q9 and Q21 remain the
largest absolute gaps; improving them requires additional execution work.

Related validation covers complete-group lineage, compression with signed and
unsigned offsets, duplicate build matches, LEFT/SEMI/ANTI/MARK, empty/NULL builds,
FULL joins, build-side grouping, lossy/computed keys, overlapping/nullable/repeated
splits, missing statistics, ordered/DISTINCT/wide aggregates and tiny bounded
channels. A total of **631 distinct related tests pass**, including ten real-Ray tests
covering pipelined execution, FTE, worker loss/retry,
source assignments and complete cleanup. The new plan-shape regression fails on
the frozen baseline. Only related tests run; no full suite or GPU tests run.
The sdist includes the three helper modules imported by the curated analytical
release tests, including the group-proof fixtures.

Evidence is retained under `build/pipelined-gap-20261021/`: frozen identities and
configurations, raw samples, `comparison.json`, `local-vs-pipelined.csv`, focused
rounds, separate preparation/timeline/terminal-channel diagnostics, regression
logs and cleanup manifests. Owned processes stop before temporary/spill removal;
query, result and worker reservations return to zero. Oracle row counts, schemas
and hashes are retained before deleting comparison outputs. Source inputs are
preserved, disk guards stay untriggered, and cleanup happens outside timing.

## Reusing proven hash partitions (2026-10-21)

The compiler retains ordered routing columns through unchanged projections,
row filters, INNER/SEMI probe outputs and proven correlated scopes. A subsequent
HASH request becomes PASSTHROUGH only when the output schema, native key
representation, key order and partition count match. Injective compression can
preserve complete groups while changing HASH; these are separate proofs.

PASSTHROUGH routes producer partition i to consumer partition i. Pipelined
execution reserves only those routes, and FTE writes and reads only the
corresponding partition objects. Native channel credit, frame sequences,
cancellation and stage commit rules are unchanged. Q21 reuses four exchange
boundaries; Q20 reuses one. Q9 retains its original exchanges, including the
15-million-row orders input.

The following controls isolate partition reuse on `73e90088f0`; the complete
Parquet-group optimization above is evaluated separately. Both widths use 36
native threads and 32 GiB total operator memory. Two workers use 18 threads and
16 GiB each, with 256 MiB staging capacity per worker. Four workers use nine
threads and 8 GiB each, with 512 MiB staging capacity per worker so both versions
can be admitted. Frames remain 64 KiB/2048 rows, channel windows 1 MiB/16 slots,
and the Ray object store 1 GiB. SQL, inputs, comparison tolerance and measurement
boundaries match the preceding SF10 controls.

| Query | Two-worker baseline | Two-worker reuse | Four-worker baseline | Four-worker reuse |
| --- | ---: | ---: | ---: | ---: |
| Q9 | 1.263 s | 1.276 s | 1.493 s | 1.514 s |
| Q20 | 0.559 s | 0.556 s | 0.634 s | 0.622 s |
| Q21 | 1.516 s | 1.504 s | 1.730 s | 1.676 s |

Baseline/candidate/candidate/baseline controls contain 192 two-worker timing
executions and 48 four-worker executions. First SQL executions are excluded
from medians: ten steady samples per query/mode/version at two workers, six at
four workers. All 240 executions match full oracle values, schemas and row
counts. These are shared-host observations without CPU isolation. Two-worker
latency changes are below 1% on the targets; four-worker Q21 changes by about
3.1%. Q9 has no measured improvement.

The deterministic benefit is fewer reserved routes and transport buffers:

| Q21 configuration | Routes, before/after | Maximum worker staging reservation, before/after |
| --- | ---: | ---: |
| Two workers/partitions | 59 / 51 | 157.5 / 115.5 MiB |
| Four workers/partitions | 229 / 181 | 462 / 336 MiB |
| Eight workers/partitions | 905 / 681 | 1071 / 777 MiB |

These are declared worst-case transport reservations, rather than observed RSS.
The eight-worker row is a compiled routing/capacity probe, without a latency
run. With four workers and a fixed 384 MiB staging limit, the baseline rejects
Q21 before execution; reuse completes three oracle-checked executions. The
final combined implementation also completes Q21 under that limit. Resource
admission continues to enforce the configured capacity.

The final combined build preserves the preceding Q10 improvement: a separate
compatibility screen has pipelined medians 0.652 s for the complete-group parent
and 0.634 s for the combined build. This 64-execution screen is not another
interleaved speedup claim. All results match the same local oracle. Combined
resource demands match the isolated reuse component for Q9/Q20/Q21 at two,
four and eight partitions.

The integration baseline native SHA-256 is `b6955e3a3b32...`, the isolated reuse
component is `1729d1fca05a...`, the complete-group parent is `f265a1d4d496...`,
and the combined build is `0db69f419324...`. Complete identities, configurations,
drivers, samples and admission probes are retained in
`build/pipelined-shuffle-20261021/`. Final combined validation passes 1257
native/compiler/contract cases and 15 real-Ray cases, including FTE retry,
three-partition routing, early close and batched-start failures. Only related
tests run; no full suite or GPU tests run. Owned processes stop before spill,
temporary data and comparison outputs are removed; row counts, schemas, hashes
and cleanup manifests are retained.

## Q9 key-domain feasibility and pipelined setup overlap (2026-10-21)

Q9's filtered lineitem/part join produces 3,261,613 rows with 2,930,317
distinct order keys. An exact necessary-condition filter could retain that
many of the 15,000,000 orders, removing about 80.5% before their exchange.
The existing 65,536-bucket hash domain is completely occupied by these keys;
using it for this input would retain every order.

Native diagnostic probes also evaluate bounded single-hash bitmaps:

| Bitmap payload | Candidate orders retained | Missing true keys |
| --- | ---: | ---: |
| 0.5 MiB | 8,997,774 | 0 |
| 2 MiB | 4,862,551 | 0 |
| 8 MiB | 3,445,603 | 0 |

These probes build the bitmap from a materialized exact domain, so their
individual timings exclude upstream query work and transport. They establish
filter selectivity and correctness, not an end-to-end optimization. Generating
the exact domain from Parquet takes about 253 ms in the separate local probe.

An isolated native prototype regenerates the filtered probe subtree and
broadcasts its keys before the other join input's exchange. This avoids a
cyclic dependency on a probe consumer waiting for the filtered build. However,
the 64-execution screening control moves Q9's steady pipelined median from
1.296 s to 1.557 s, about 20% slower. Every result matches the oracle. These
are sequential screening rounds with three steady samples per mode/version,
not a paired acceptance comparison. The prototype is removed. Its patch and
binary identity remain in the investigation artifacts; it is not in the PR.
A future large-domain filter needs to reuse existing join work and account for
distribution, filter construction and dependency costs together.

The accepted change instead overlaps independent pipelined setup operations:
result-service preparation with worker preparation; worker, relay and client
connections after every endpoint is published; and worker/relay readiness
probes. Every preparation and connection receipt is still checked. Tasks start
only after all native endpoints and the client are ready. Existing lifecycle
locks, late-call fences, reservations, cancellation and channel limits apply.
The installed native binary is unchanged from the combined partitioning build,
SHA-256 `0db69f419324ad1c25cba93a1bb3cdcc5d723cf44f0c68822c6ae2310cb9a2f7`.

The acceptance comparison uses baseline/candidate/candidate/baseline order,
forward then reverse mode order, with eight steady samples per
query/mode/version. It contains 240 timing executions. A separate 96-execution
screen is excluded from these medians. Resources and measurement boundaries
match the preceding two-worker controls: 36 local threads/32 GiB, two workers
with 18 threads/16 GiB each, 256 MiB staging per worker, unchanged frame/window
limits and SQL, the same frozen SF10 inputs, and full result collection and
successful close. First SQL executions are recorded separately and excluded.

| Query | Baseline Local | Candidate Local | Baseline pipelined | Candidate pipelined | Pipelined change | Submission time saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2 | 0.136 s | 0.135 s | 0.249 s | 0.239 s | -3.9% | 10.1 ms |
| Q6 | 0.214 s | 0.211 s | 0.289 s | 0.281 s | -2.6% | 7.0 ms |
| Q9 | 0.854 s | 0.872 s | 1.273 s | 1.276 s | +0.2% | 3.7 ms |
| Q10 | 0.474 s | 0.482 s | 0.642 s | 0.644 s | +0.2% | 10.0 ms |
| Q20 | 0.344 s | 0.346 s | 0.574 s | 0.569 s | -0.8% | 8.3 ms |
| Q21 | 1.193 s | 1.207 s | 1.522 s | 1.522 s | 0.0% | 5.0 ms |

Q2 and Q6 improve in both orders. Q9/Q10/Q20/Q21 do not establish a material
end-to-end improvement; Q9's large join/exchange gap remains. These are
shared-host observations without CPU isolation. Local controls vary by up to
2.1%, and no task-owned build, test or diagnostic runs during measurements.

All 400 timing executions across the rejected native prototype, setup screen
and setup acceptance controls match the full Local oracle's schemas, row
counts and values. All 80 related tests pass: 49 real-Ray lifecycle, fault and
source/retry cases, and 31 result-service unit cases. No full suite or GPU tests
run. The setup regression also reproduces the serialized preparation
order on the frozen baseline. Frozen configurations, SQL, Python/native
identities, patches, drivers, samples and test logs are retained under
`build/pipelined-q9-filter-20261021/`. Cleanup records the output schemas,
counts and hashes before removing comparison outputs, after each run's owned
processes have stopped; source inputs are preserved.

## SF100 Parquet cardinality and source balance (2026-10-21)

Distributed Parquet scan estimates now use a bound scan's cardinality callback,
which counts the remaining files' selected row groups when the optimizer requests
statistics. File pruning runs first; pruned files remain snapshot dependencies.
Explicit cardinality overrides and repeated file references retain their meaning.
The Local planner and source SQL are unchanged.

Early key-only broadcasts and independent correlated key scans can use the existing
per-query worker replication allowance, bounded by worker operator memory and the
64 MiB ceiling. Existing key-lineage, join-kind, selectivity and traffic checks
apply. Split assignment gives larger row-group chunks to less-loaded strided
partition slots, preserving each chunk's contents and deterministic assignments.
The lineitem partition-count probe changes 252,370,944 / 347,666,958 rows into
300,408,432 / 299,629,470 rows (50.06% / 49.94%).

The full SF100 comparison below uses original repository SQL over the same 307
Parquet files (37.5033 GiB). Local has 36 native threads and 32 GiB operator memory;
pipelined has two workers with 18 threads and 16 GiB each. Exchange/staging budgets
remain 128/256 MiB per worker, frames 64 KiB/2048 rows, windows 1 MiB/16 slots,
and the specialized Ray object store 1 GiB. Each mode keeps a persistent connection
within its phase. Production worker cleanup still clears query/file caches.
OS caches are retained; this is a shared host without CPU isolation.

Two fresh phase processes use Local→pipelined and pipelined→Local order, with two
SQL executions per query/mode/phase. Each query's first execution in each phase is
excluded, leaving two steady samples per cell. Timing covers submission, complete
collection and successful close. No task-owned build, test, native diagnostic or
Doris/HDFS service runs during measured windows. All 176 executions match full
oracle schemas, row counts and values under the existing float tolerance
(relative 1e-9, absolute 1e-6).

| Query | Local | Pipeline before* | Pipeline now | Now / Local | Change* |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 4.204 s | 4.515 s | 4.417 s | 1.05× | -2.2% |
| Q2 | 0.927 s | 3.305 s | 2.938 s | 3.17× | -11.1% |
| Q3 | 3.759 s | 14.422 s | 5.543 s | 1.47× | -61.6% |
| Q4 | 2.834 s | 9.963 s | 8.824 s | 3.11× | -11.4% |
| Q5 | 3.919 s | 26.850 s | 5.915 s | 1.51× | -78.0% |
| Q6 | 2.025 s | 2.303 s | 2.239 s | 1.11× | -2.8% |
| Q7 | 4.339 s | 9.067 s | 5.683 s | 1.31× | -37.3% |
| Q8 | 4.755 s | 34.117 s | 6.513 s | 1.37× | -80.9% |
| Q9 | 10.392 s | 45.642 s | 16.743 s | 1.61× | -63.3% |
| Q10 | 4.368 s | 13.271 s | 12.959 s | 2.97× | -2.4% |
| Q11 | 0.942 s | 1.130 s | 1.141 s | 1.21× | +1.0% |
| Q12 | 3.039 s | 10.079 s | 9.817 s | 3.23× | -2.6% |
| Q13 | 6.549 s | 8.849 s | 8.865 s | 1.35× | +0.2% |
| Q14 | 3.177 s | 4.482 s | 4.420 s | 1.39× | -1.4% |
| Q15 | 3.221 s | 3.749 s | 3.767 s | 1.17× | +0.5% |
| Q16 | 1.183 s | 7.889 s | 2.435 s | 2.06× | -69.1% |
| Q17 | 3.633 s | 38.387 s | 4.054 s | 1.12× | -89.4% |
| Q18 | 7.070 s | 33.999 s | 18.237 s | 2.58× | -46.4% |
| Q19 | 3.976 s | 6.388 s | 5.864 s | 1.47× | -8.2% |
| Q20 | 3.570 s | 6.774 s | 5.222 s | 1.46× | -22.9% |
| Q21 | 11.844 s | 32.365 s | 28.780 s | 2.43× | -11.1% |
| Q22 | 1.170 s | 4.299 s | 4.057 s | 3.47× | -5.6% |

*Before values are the earlier same-host SF100 control on `4222d95b`, rather than
an interleaved A/B experiment. Most baseline cells contain four steady samples;
Q9 has two and one separately recorded control-RPC timeout. Large changes are
consistent with the separate native channel diagnostics: Q17's two fact exchanges
fall from 600,037,902 rows each to 600,982 rows each. These diagnostics are outside
the acceptance timings. Small latency changes are not established by this sample
size. Q11 retains the repository's fixed threshold and returns zero rows; this is
not an official TPC-H score or scale-adjusted parameter qualification.

The sum of query medians is 168.433 s for pipelined and
90.895 s for Local, a 1.85× ratio. The historical
pipelined sum is 321.844 s (47.7% lower now).
No query's current median is faster than Local. Q1/Q6/Q17 are close; Q21, Q18,
Q10 and Q12 retain large absolute gaps. This round does not establish that
pipelined beats Local under the same host resource budget.

Code is `e6533bdefae02d9d1d1f44e3707a220a030ee84f`, native SHA-256
`015287289eae8bfe195ebde518e9644b4864f67c2a8c0964da8dea375ca03bce`. All 1161 distinct related cases pass, including
17 real-Ray cases for retries, source assignments and cleanup. Tests cover
NULLs, duplicates, empty files, explicit cardinality, bounded channels and a
pruned unreadable footer. Only related tests run; no full suite or GPU tests run.
The affected test modules already belong to the source-package whitelist.

Evidence is retained in `build/pipeline-sf100-20261021/`: frozen identities,
configurations, raw timing samples, `comparison.csv`, separate channel/CPU probes,
JUnit reports, validation results and cleanup manifests. Owned processes stop
before spill, temporary data and oracle outputs are removed; oracle schemas,
row counts and hashes are retained. Source datasets are preserved.

## SF100 bounded key filters (2026-10-21)

Correlated owner-key filters now use the existing per-query worker replication
allowance rather than a separate 8 MiB cap. Owners remain shared and are neither
rescanned nor recomputed. Existing lineage, selectivity and full-traffic checks
remain in force. HAVING key filters choose a power-of-two hash domain separately
for each probe, bounded by declared worker memory, probe cardinality and extra
exchange traffic. The conservative default stays at 65,536 buckets; a 64 MiB
replication allowance permits up to 524,288 buckets, reserving 128 working bytes
per bucket. Domain and probe expressions serialize the same chosen mask.

This comparison covers eight affected/control queries with original repository
SQL over the same 307 SF100 Parquet files (37.5033 GiB). Baseline pipelined phases
bracket candidate Local→pipelined and pipelined→Local phases. Each fresh phase
runs two executions per query/mode and excludes the first query-specific one,
leaving two steady samples per cell. All 96 comparison executions, eight oracle
executions and two final diagnostic executions pass full schema/row/value checks
under the existing float tolerance (relative 1e-9, absolute 1e-6).

| Query | Local | Pipeline before | Pipeline now | Now / Local | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 4.218 s | 4.343 s | 4.405 s | 1.04× | +1.4% |
| Q6 | 2.052 s | 2.237 s | 2.263 s | 1.10× | +1.2% |
| Q9 | 9.872 s | 16.733 s | 16.781 s | 1.70× | +0.3% |
| Q17 | 3.665 s | 4.364 s | 4.260 s | 1.16× | -2.4% |
| Q18 | 7.349 s | 18.470 s | 17.369 s | 2.36× | -6.0% |
| Q20 | 3.608 s | 5.585 s | 5.237 s | 1.45× | -6.2% |
| Q21 | 11.949 s | 28.945 s | 15.328 s | 1.28× | -47.0% |
| Q22 | 1.227 s | 3.950 s | 2.713 s | 2.21× | -31.3% |

Separate native diagnostics show Q21's two correlated fact exchanges fall from
979,394,376 to 58,393,804 rows combined (94.0% fewer). Q18's order/fact exchanges
fall from 13,996,510 / 55,999,985
to 1,824,198 / 7,316,103 rows,
but its initial aggregate still exchanges 150,001,994 opaque states.
This remains Q18's main measured bottleneck. Diagnostic timings are excluded
from the comparison table.

The selected-query median sums are 84.627 s before,
68.356 s now and 43.939 s Local. These are eight-query
sums, not a new 22-query TPC-H score. The full 22-query reference remains in the
preceding section. Two steady observations do not establish small latency
changes. This is a shared host with OS caches retained and no CPU isolation.

Local uses 36 native threads and 32 GiB operator memory. Pipelined uses two
workers × 18 threads / 16 GiB each, 128/256 MiB exchange/staging per worker,
64 KiB/2048-row frames, 1 MiB/16-slot windows and a specialized 1 GiB Ray object
store. Production query/file cache cleanup is unchanged. No owned builds, tests,
native diagnostics or Doris/HDFS run during comparison windows. Final native
diagnostics run separately afterward. No SF100 FTE timing claim is made.

Frozen code: `d80fba27ad60a131f71b3ae7e9e58e296a67d9dc`; native SHA-256
`275b1c71a2eeb33f18fb0f6338af2f103537fc2684545404ff917c7626a8e3b8`. Baseline is the retained native binary for the
previous PR head, SHA-256
`015287289eae8bfe195ebde518e9644b4864f67c2a8c0964da8dea375ca03bce`,
using the same Python execution code. Both driver and worker native plan
identities are enforced by the existing preparation protocol.

All 1196 distinct related tests pass, including 23 real-Ray cases.
They cover NULLs/duplicates, scalar/semi/anti/nested correlations, low/high memory
controls, medium-probe filter retention, dense/empty domains, bounded channels,
early close and shared-producer FTE retries. No full suite or GPU tests run.
The existing test modules are already included in the source-package allowlist.

Evidence is retained under `build/pipeline-sf100-exchange-20261021/`: frozen
configurations and identities, raw samples, `comparison.csv`, validation results,
native channel probes, JUnit reports and cleanup manifests. Source files remain
unchanged. Owned spill/temp data and oracle outputs are removed only after their
processes stop; schemas, row counts and hashes are retained.

## SF100 direct binary exchange export (2026-10-21)

Internal exchanges need Arrow v1.5 for compact decimal32/decimal64 values, but
previously inherited binary views for BLOBs and aggregate states, then rebuilt
those columns as ordinary binary arrays. A separate native binary-view preference
now lets the exchange exporter produce its target binary layout directly,
including nested columns. Public Arrow defaults and wire types are unchanged.
The receiver's owned frame and the merge aggregate's private aligned copies
remain in place. No query-name rules, SQL rewrites, Local execution fallback,
new dependency or production resource-default changes are introduced.

The following control covers original Q1/Q9/Q18/Q21 SQL over the same 307 SF100
Parquet inputs. Baseline pipeline phases bracket candidate Local/pipeline and
pipeline/Local phases. Three executions per query/mode in each fresh phase,
excluding the first query-specific execution, leave four steady samples per cell.
All 72 comparison executions and four Local oracle executions pass schema,
row-count and value checks at the existing float tolerance (relative 1e-9,
absolute 1e-6).

| Query | Local | Pipeline before | Pipeline now | Change | Now / Local |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 4.148 s | 4.471 s | 4.482 s | +0.3% | 1.08× |
| Q9 | 10.018 s | 16.757 s | 16.755 s | -0.0% | 1.67× |
| Q18 | 7.145 s | 17.346 s | 17.170 s | -1.0% | 2.40× |
| Q21 | 12.227 s | 14.854 s | 14.902 s | +0.3% | 1.22× |

The Q18 median reduction is only 1.0%; this shared-host, four-sample control does
not establish a material improvement. Q1/Q9/Q21 are effectively unchanged.
Q18 still takes 2.40 times Local. This is a selected four-query control, not a
new full 22-query score or an update to the preceding eight-query totals.

Local uses 36 native threads and 32 GiB operator memory. Pipeline uses two
workers with 18 threads / 16 GiB each, 128/256 MiB exchange/staging per worker,
64 KiB/2048-row frames, 1 MiB/16-slot windows, and the specialized 1 GiB Ray
object store. Query/file-cache cleanup is unchanged; OS caches are retained.
No owned builds, tests, native sampling or Doris/HDFS run concurrently with
comparison queries. Timing includes submission, full collection and successful
close. CPU diagnostics run separately afterward.

All 816 distinct related tests pass on the clean production binary: 796 across
analytical/compiler, compact-decimal, Flight and materialized-exchange modules;
eight bounded-channel aggregate-merge cases; twelve real-Ray cases covering
aggregate states, wide values, Parquet, early close and FTE retry. New tests
preserve exact COUNT/SUM/AVG state bytes, including NULLs and repeated values,
through both transports. Existing tests cover nested BLOB/decimal layouts and
invalid Arrow data. No full suite, GPU tests or SF100 FTE performance run.

Separate all-native clock sampling with gprofng, using an unstripped binary with
identical code-section hash, confirms substantial extra aggregate work. For two
Q18 executions, mean OS CPU use is 230.29 CPU seconds per Local query versus
476.025 CPU seconds for both pipeline workers (2.067 times). Group-table
find/create has 161.30 Local versus 270.92 worker inclusive sampled CPU seconds
across the two executions. Frame measurement has 55.52 inclusive CPU seconds,
5.94% of sampled worker CPU. Inclusive costs overlap and are not additive or
query latency. The next priority is native aggregate-state import and merging,
reducing repeated group-table construction, state initialization and conversion.

Footer ranges cannot safely remove this merge: the 144 lineitem files / 628
row groups overlap over 99.8543% of their bounded integer key domain. This is
not a fraction of rows or actual duplicate groups. The conservative complete-group
proof remains necessary.

Frozen code: `f8bbee0a1caa7d77756de1a8e4e8311605a9a356`; native SHA-256
`e19748e29f50414a083c0974ec449f093a13f283117b17975aaac2f0ee8059c9`.
Baseline native SHA-256:
`275b1c71a2eeb33f18fb0f6338af2f103537fc2684545404ff917c7626a8e3b8`.
The Python execution implementation is unchanged. Evidence is retained under
`build/pipeline-sf100-states-clean-20261021/`: frozen identities/configurations,
raw samples, output counts/hashes, JUnit reports, native CPU reports and cleanup
manifests. Earlier exploratory data is excluded from this control. Owned spill,
temp data and oracle outputs are removed after processes stop; source datasets
and the active Doris/HDFS deployment are preserved.

## SF100 native aggregate-state import (2026-10-21)

Final grouped aggregates previously processed exported partial states through
ordinary sink aggregation: group lookup, empty-state initialization, private
scratch copying and combine. The radix source then combined the resulting rows
again. An explicit native state-import callback now lets eligible grouped
aggregates append complete states directly to owned radix tuples. It hashes the
group keys for partitioning, validates state sizes and copies into private,
aligned storage. NULL inputs initialize an empty child state. All duplicate
keys are combined in the source phase, including with one execution thread.

This path requires every aggregate to support import, consume input columns and
have no destructor, DISTINCT modifier or aggregate filter. Distributed merge
wrappers restore the callback through their existing serialization constructor.
Existing temporary-memory accounting, repartitioning, spill and exchange
backpressure remain in force. General merge updates retain private scratch
copies. Original SQL, public Python execution APIs, wire layouts and production
budgets are unchanged.

The control below compares this import path with the preceding direct-binary
export revision. All 72 timing executions and four Local oracle executions
match schema, row count and values (float tolerance relative 1e-9, absolute
1e-6). Three executions per query/mode in each fresh phase, discarding the first
query-specific execution, give four steady observations per cell. Baseline
pipeline phases bracket candidate Local/pipeline and pipeline/Local phases.

| Query | Local | Pipeline before | Pipeline now | Change | Now / Local |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 4.219 s | 4.490 s | 4.581 s | +2.0% | 1.09× |
| Q9 | 9.896 s | 16.840 s | 17.107 s | +1.6% | 1.73× |
| Q18 | 7.257 s | 17.189 s | 16.966 s | -1.3% | 2.34× |
| Q21 | 12.207 s | 14.877 s | 14.861 s | -0.1% | 1.22× |

The Q18 median falls only 1.3%, while Q1/Q9 medians rise 2.0%/1.6%. Four steady
observations on a shared host do not establish a material latency improvement
or resolve these small control differences. Q9 also has a 19.156-second steady
candidate sample; all scheduled samples are retained. Q18 remains 2.34 times
Local. These selected-query results do not replace the earlier full 22-query
score or eight-query totals.

Six separate result-checked CPU diagnostic executions run after timing acceptance:
two baseline pipeline queries and two candidate queries in each mode. Mean OS
user-plus-system CPU time for the two pipeline workers falls from 471.655 to
449.665 CPU seconds per Q18 query (-4.66%). Candidate Local averages 227.465 CPU
seconds, leaving worker CPU at 1.977 times Local. These totals exclude the
pipeline coordinator/result service and are not wall time. Diagnostic timings
are excluded from the table. The evidence supports a modest CPU reduction;
substantial work remains in the final aggregate merge.

All 594 distinct related tests pass on the frozen installed binary: 489
analytical/native compiler and Parquet aggregate cases, 93 merge/interval cases,
and twelve real-Ray cases covering pipeline/FTE states, wide values, early close
and shared-producer retry. Coverage includes 1/2/4 native threads, 2/3 partitions,
NULL keys/states, repeated targets and malformed states. Two regressions configure
32 MB and merge 524,299 repeated input states with one/four threads, assert actual
spill, validate results, and require empty spill directories after connection
close. No full suite, GPU tests or SF100 FTE performance run.

Resource settings match the preceding control: Local 36 threads / 32 GiB operator
memory; two pipeline workers with 18 threads / 16 GiB each, 128/256 MiB
exchange/staging per worker, 64 KiB/2048-row frames, 1 MiB/16-slot windows and a
specialized 1 GiB Ray object store. Inputs remain the same 307 SF100 Parquet files.
OS caches are retained; no owned builds/tests, CPU diagnostics or Doris/HDFS
services overlap timing windows. Timing includes submission, full collection
and successful close.

Frozen implementation: `670a59db29c27321555f304ceaa16af14060415e`; native SHA-256
`3fa6c796cf18ebe4f5e26e39c58804add5efdd1bb6408e6a6172911364af86c4`.
Baseline is `81624b6f6b` (documentation after `f8bbee0a1c`), native SHA-256
`e19748e29f50414a083c0974ec449f093a13f283117b17975aaac2f0ee8059c9`.
The Python execution implementation is unchanged. Evidence is retained under
`build/pipeline-sf100-state-import-20261021/`: frozen code/native identities,
configurations, raw samples, output counts/hashes, CPU records, JUnit reports
and cleanup manifests. Prototype validation is kept separately and is not
added to the distinct test count. Owned temporary data is cleaned after process
exit; source datasets and the restored Doris/HDFS deployment are preserved.

## Rejected SF100 final-state copy experiment (2026-10-21)

A final-merge prototype copied the complete native aggregate state into each
new group, avoiding empty-state initialization and its first combine. An explicit
copy capability restricted this to self-contained destructor-free merge states;
repeated groups still combined normally. The prototype was tested and then
removed because neither query latency nor total worker CPU improved. The PR
retains the preceding native state-import implementation.

The following is a rejected trial, not a new retained implementation score.
It compares the prototype with the preceding state-import revision. Original
Q1/Q9/Q18/Q21 SQL, the same 307 SF100 Parquet inputs and the previous resource
settings are retained. Baseline pipeline phases bracket candidate Local/pipeline
and pipeline/Local phases; excluding the first query-specific execution gives
four steady observations per cell. All 72 timing executions and four Local
oracle executions match schema, row count and values.

| Query | Local | Pipeline before | Rejected trial | Change | Trial / Local |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 4.227 s | 4.496 s | 4.549 s | +1.2% | 1.08× |
| Q9 | 10.002 s | 16.758 s | 16.862 s | +0.6% | 1.69× |
| Q18 | 7.168 s | 16.717 s | 16.774 s | +0.3% | 2.34× |
| Q21 | 12.159 s | 14.731 s | 14.808 s | +0.5% | 1.22× |

Q18 remains 2.34 times Local; its median rises 0.3%. Control medians rise
0.5–1.2%. These small shared-host differences do not establish a gain.
Six separate Q18 CPU diagnostic executions also match. Mean OS user-plus-system
CPU for the two pipeline workers rises from 446.160 to 450.805 CPU seconds
(+1.04%); candidate Local averages 225.575 CPU seconds. Worker CPU excludes the
coordinator/result service. Diagnostic timings are excluded from the table.

Matched-text gprofng 2.42 native sampling (20 ms, all descendants) adds four
baseline and two prototype Q18 executions, all result-checked. Across two
pipeline queries per binary, attributed InitializeStates CPU falls from
30.14 to 19.80 seconds and CombineStates from 12.18 to 4.56 seconds. New copying
costs 5.80 sampled CPU seconds under final combine, and exclusive group-lookup
CPU rises from 143.78 to 150.46 seconds. These are diagnostic samples with
inclusive overlap, not additive query latency or proof of one compiler cause.
The intended work was removed, but the overall measurements do not justify
retaining the additional native path.

The retained regression coverage deliberately constructs colliding long-string
keys and exercises NULL keys/values, repeated producer states across vectors,
one/four native threads, ordinary/forced-external execution and spill cleanup.
The final retained binary is validated separately after removing the prototype;
prototype test counts are not added to final distinct test totals.

Trial code: `a576611f30f142c76e30554037a6e466e0db92a0`; native SHA-256
`7ea2f095fd438e62dd55d83ef538fa6d20b1e0b638f17218a95e19e45dacd444`.
Baseline is `93fa268299` (documentation after `670a59db29`), native SHA-256
`3fa6c796cf18ebe4f5e26e39c58804add5efdd1bb6408e6a6172911364af86c4`.
Python execution files, dependency versions, input identities, driver and resource
capacities match. Only validated per-phase temporary storage paths differ.
Timing includes submission, complete collection and successful close. OS caches
are retained; no owned build/test, diagnostic or Doris/HDFS services overlap
measured windows. The complete-group proof remains conservative; observed
near-uniqueness does not permit skipping final duplicate-group merging.

Evidence is retained under `build/pipeline-sf100-state-copy-20261021/`: the rejected
patch and code identity, both measured binaries, raw samples, CPU reports,
configuration comparison, output hashes/counts, test logs and cleanup manifests.
Source datasets are preserved and Doris/HDFS is restored with the requested caches
disabled. The next investigation should reduce repeated final-merge group
construction and tuple movement; further initialization-only tuning has limited
observed benefit.

## SF100 final aggregation in owned tuples: rejected follow-up (2026-10-22)

A second experiment removed fresh tuple allocation and group-key copying during
final merging. It indexed canonical rows directly in owned, imported-state radix
partitions, combined duplicate states into those rows, and reused the hidden hash
column as an output marker after matching. Only fully imported, destructor-free
layouts were eligible. Whole-partition pinning accounted for allocated row/heap
blocks, metadata, the fixed hash index and workspace; oversized/skewed partitions
kept streaming compaction. Query memory capped soft reservations, and pin-handle
slots were reconstructed when rescanning previously unpinned blocks.

**This native implementation is rejected and absent from the final PR diff.**
The four selected latency medians total 54.746 → 54.711 seconds (-0.064%).
Q18 improves slightly, but the overall result does not justify adding the extra
pinning and output path for the current SF100 objective. The preceding accepted
state-import implementation remains the production engine.

| Query | Local | Pipeline before | Pipeline trial | Change | Trial / Local |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 4.298 s | 4.646 s | 4.649 s | +0.1% | 1.08× |
| Q9 | 10.281 s | 17.432 s | 17.269 s | -0.9% | 1.68× |
| Q18 | 7.483 s | 17.164 s | 16.817 s | -2.0% | 2.25× |
| Q21 | 12.506 s | 15.504 s | 15.976 s | +3.0% | 1.28× |

Original Q1/Q9/Q18/Q21 SQL reads the same 307 Parquet files / 37.5033 GiB.
Local uses 36 native threads / 32 GiB; pipeline uses two workers with 18 threads
and 16 GiB each. Exchange/staging budgets are 128/256 MiB per worker, frames are
64 KiB / 2048 rows, windows are 1 MiB / 16 slots, and the specialized Ray object
store is 1 GiB. Baseline pipeline phases bracket candidate Local/pipeline and
pipeline/Local phases. Each timing phase executes each query/mode three times;
the first query-specific execution is excluded, leaving four steady samples per
cell. All scheduled samples remain in the results.

The shared host varied visibly: candidate reverse-order Q9/Q18 samples include
19.234/18.160 seconds, while the later baseline and Local phases also slow down.
Other host workloads continued. The +3.0% Q21 latency result is therefore not
established as a computation regression. Separate, result-checked Q21 controls
measure 425.360 → 426.140 worker CPU seconds (+0.18%). Q18 controls measure
449.990 → 441.560 CPU seconds (-1.87%); Local averages 226.545 CPU seconds,
leaving candidate pipeline CPU at 1.949× Local. These are OS user-plus-system
deltas summed over the two workers, excluding the coordinator/result service.
Diagnostic timings are excluded from the table.

A preliminary Q18-only screen ran ten result-checked queries and observed
16.637 → 16.403 seconds (-1.41%). It is recorded separately from the four-query
matrix. Two additional Q18 native-profile executions used gprofng at 20 ms with
all descendants and a symbol library whose machine-code section matches the
candidate. The new path was active: CombineInPlace accounts for 106.22 inclusive
and 42.70 exclusive sampled CPU seconds, including 46.40 attributed to hash
probing. Ordinary lookup attributed to AddChunk is 61.66 seconds, compared with
64.36 in the preceding baseline profile; the data does not establish a large
ordinary-path compiler regression. Inclusive samples overlap and are not query
latency. Hash lookup and memory traversal remain substantial.

All **98 SF100 executions** match schema, row count and values: 72 timing queries,
four Local oracles, ten preliminary screen queries, six Q18 CPU controls, four
Q21 CPU controls and two native-profile queries. No SF100 FTE speed claim or
full 22-query score update is made. Frozen Python files, package versions, SQL,
input identities and resource capacities match; only validated temporary paths
differ. Timing includes submission, full collection and successful close. OS
caches remain enabled. Owned builds/tests, diagnostics and Doris/HDFS do not
overlap timing windows.

The trial passes **604 distinct related tests**: 103 aggregate-state cases,
489 compiler/Parquet cases, and twelve real-Ray cases. New regressions cover
sparse output vectors, long/NULL keys, duplicate-only tails, repeated fragment
execution and one/four native threads at 32/256 MB. Actual-spill tests now cover
both one and eight keys with 524,299 states at a configured 32 MB, verify results,
and require empty spill directories after close. The configured budget is not a
claim about peak process RSS. These regressions also pass on the retained engine.
Trial and retained runs are not added together as distinct tests. No full/GPU
suite was run; the modified test module already belongs to the sdist allowlist.

Trial code: `8e866d890e2754891d20d5a39db2557d3fc287bf`; native SHA-256
`9b2f154ff15e35151f7c3aa79dde3020acc74dd8feb3f30fff6f109a0a231295`.
Baseline: `d69804e514`, engine code `670a59db29`, native SHA-256
`3fa6c796cf18ebe4f5e26e39c58804add5efdd1bb6408e6a6172911364af86c4`.
Evidence under `build/pipeline-sf100-inplace-finalize-20261022/` retains the
rejected patch, frozen identities/binaries, configuration audit, raw samples,
CPU/native profiles, output hashes/counts, JUnit reports and cleanup manifests.
Only tests and this negative result are retained in the PR. The next performance
investigation should attribute partial aggregation and exchange work by stage;
these two final-state experiments have not established a material end-to-end gain.

## SF100 costed partial-state output (2026-10-22)

Grouped partial aggregates previously performed a complete worker-local radix
merge before exporting states to a distributed consumer that merges those keys
again. The compiler now explicitly marks eligible grouped exported-state
operators as permitting repeated groups. Their source can scan existing owned
partial tuples directly; the consumer still merges every duplicate before SQL
results and HAVING filters are produced. Ordinary SQL EXPORT_STATE retains one
row per group. The marker is serialized and validated for a single complete,
nonempty grouping set and unordered, non-distinct, destructor-free states.

The choice to skip local consolidation uses the existing bounded HLL samples.
Their sketches are merged across native threads, including overlapping group
sets, and compared with the state counts from the matching sampling windows.
Local consolidation remains when the estimate predicts more than a twofold
reduction in exchange rows. One/two-thread grow strategies without these samples
retain their existing consolidation. This adds no input sampling pass. It is a
cost heuristic, not a hard bound on expansion: changing distributions after the
sampling window can affect its accuracy. Correctness always depends on the final
consumer merging states, never on an estimated group-completeness assumption.

The final candidate is retained. The selected four-query median sum changes
**56.921 → 54.990 seconds (-3.4%)**. Q18 improves **7.9%**, with an independent
worker CPU reduction of **9.1%**. Q1/Q9 changes are small relative to shared-host
variation. The Q21 latency reduction is observed in this control but has not been
independently attributed to the new implementation.

| Query | Local | Pipeline before | Pipeline now | Change | Now / Local |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 4.443 s | 4.758 s | 4.784 s | +0.5% | 1.08× |
| Q9 | 10.680 s | 18.149 s | 18.102 s | -0.3% | 1.69× |
| Q18 | 7.672 s | 18.041 s | 16.610 s | -7.9% | 2.17× |
| Q21 | 13.225 s | 15.973 s | 15.494 s | -3.0% | 1.17× |

The table uses original Q1/Q9/Q18/Q21 SQL and the same 307 Parquet inputs /
37.5033 GiB. Local uses 36 native threads / 32 GiB; pipeline uses two workers
with 18 native threads and 16 GiB each. Per-worker exchange/staging budgets are
128/256 MiB, frames are 64 KiB / 2048 rows, windows are 1 MiB / 16 slots, and the
specialized Ray object store is 1 GiB. Baseline phases bracket Local/pipeline and
pipeline/Local candidate phases. Three executions per query/mode in each phase,
excluding each first execution, give four steady observations per table cell.
All scheduled samples are retained. Timing covers submission, full collection
and successful close; diagnostic timings are excluded.

The shared host varied: candidate Local Q18 steady samples range from 7.416 to
8.314 seconds. Unrelated jobs continued. OS caches were retained and owned
builds/tests, diagnostics and Doris/HDFS did not overlap timing windows. Frozen
Python files, package versions, SQL, input identities and resource capacities
match across phases; only validated temporary store paths differ. These selected
controls do not update the full 22-query score or establish SF100 FTE performance.

Separate Q18 controls average **454.225 → 413.070 worker CPU seconds (-9.06%)**.
Local averages 231.160 CPU seconds, leaving pipeline at 1.787× Local CPU and
2.17× Local latency in the table. CPU is OS user-plus-system deltas summed over
the two pipeline workers, excluding the coordinator/result service. Exchange0
rows change from 150,001,994 to 150,009,034 / 150,008,973: less than 0.005% growth.

An unconditional prototype exposed why costing is necessary: an interleaved
32,771-group control emitted 524,299 states. The final policy preserves 32,771
output rows for that control. Actual-spill validation at a configured 32 MB
covers both strategies: the interleaved case spills 5,111,808 bytes and consolidates
locally; the clustered case spills 4,718,592 bytes and exports 524,307 states for
524,274 final groups. Both match the ordinary SQL oracle and leave empty spill
directories after close. A configured memory budget is not a peak-RSS guarantee.

**614 distinct related tests pass**: 111 aggregate-state cases, 489 compiler and
Parquet cases, twelve real-Ray cases, and two additional producer spill cases.
Coverage includes NULL/filter states, long keys, reordered/repeated inputs,
ordinary SQL EXPORT_STATE, clustered/interleaved groups, external execution,
bounded channels, early close and shared-producer retry. The installed-package
launchers keep Ray in its own process. No full/GPU suite was run. Test binaries
use the final implementation before its commit-only native version refresh;
performance controls use the frozen committed binary below.

Final acceptance has **82 result-checked SF100 executions**: 72 timing queries,
four Local oracles and six CPU controls. Four initial operator-profile queries
and six superseded unconditional-prototype CPU controls also match, for 92 total.
The operator probe is absent from the final binary. Its timers are cumulative
elapsed operator time, including possible lock waits, rather than OS CPU; primary
fragment0/1 profiles are complete for both workers and repetitions, while one
noncritical fragment2 profile is absent. These diagnostics identified repeated
aggregation and exchange work without treating pooled function samples as one
logical stage.

Candidate code: `a28a36057d48bcb045355adcbfd133c2251ddc48`; native SHA-256
`4864af76ebd15d71cb5a80033ba32f8a374edd0fe5303df5c7e04d518b959a20`.
Baseline: `8ed2e3e96a`, engine `670a59db29`, native SHA-256
`3fa6c796cf18ebe4f5e26e39c58804add5efdd1bb6408e6a6172911364af86c4`.
Evidence in `build/pipeline-sf100-fragment-cost-20261022/` retains frozen identities
and binaries, raw samples, operator/CPU diagnostics, row counts, configuration
and output validation, test reports and cleanup manifests. Owned temporary data,
oracle files and the duplicate baseline package were removed after the runs
stopped. Original inputs are preserved; Doris/HDFS is restored with the requested
file and Parquet page caches disabled. Remaining Q18 exchange traffic and final
consumer work still leave a substantial Local gap.

## SF100 compact join-key domains (2026-10-25)

Conservative filtered-build estimates previously prevented early exact-key
broadcasts, leaving large probe inputs to cross hash exchanges. The compiler
now builds bounded word/mask domains for oversized selective integral-key
builds. A native hash selects a word and one of its 64 bits from disjoint hash
bits. Partial and global BIT_OR aggregates produce exactly one mask per word.
The probe matches populated words before extracting its bit, then the original
join enforces exact keys, NULL behavior and residual predicates. This preserves
probe duplicates and retains only false positives from hash collisions. Small
exact-key domains retain their existing planning. Domain rows are two UBIGINT
columns and use the existing native serialization and bounded channels.

The cost rule accounts for existing early filters: an ancestor's reduced input
cannot fund another blocking bitmap with its original cardinality. New streaming
build domains require a fourfold traffic margin, including the extra build and
broadcast edges, to cover hashing and the complete-build barrier. Existing
filtered-aggregate domain costing is retained. The word count uses a 128-byte
per-word cost estimate for grouping and lookup overhead; this bounds domain
rows, not total allocation or process RSS. Native operator memory/spill limits
still apply, including thread-local aggregate states.

The selected four-query median sum changes **47.299 → 38.912 seconds (-17.7%)**.
Q10 improves **27.0%**, Q12 **49.7%**; Q4 and Q18 are effectively unchanged.
The remaining Local gaps are explicit below. These four controls do not update
the full 22-query score or establish SF100 FTE performance.

| Query | Local | Pipeline before | Pipeline now | Change | Now / Local |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4 | 2.926 s | 8.872 s | 8.880 s | +0.1% | 3.03× |
| Q10 | 4.351 s | 12.800 s | 9.342 s | -27.0% | 2.15× |
| Q12 | 2.922 s | 9.598 s | 4.829 s | -49.7% | 1.65× |
| Q18 | 7.059 s | 16.029 s | 15.861 s | -1.0% | 2.25× |

The original SQL reads the same 307 Parquet inputs / 37.5033 GiB. Local uses
36 native threads / 32 GiB; pipeline uses two workers with 18 threads / 16 GiB
each. Per-worker exchange/staging budgets are 128/256 MiB, frames 64 KiB /
2048 rows, windows 1 MiB / 16 slots, and the specialized Ray object store is
1 GiB. Baseline phases bracket candidate Local/pipeline and pipeline/Local
phases. Three executions per query/mode/phase, excluding each first execution,
give four steady observations per table cell. All scheduled samples are retained.
Timing includes submission, complete collection and successful close.

Shared-host variation remains visible: Q12's baseline phase medians drift by
-5.7%, with individual steady samples of 9.312–10.092 seconds; candidate samples
are 4.793–5.120 seconds. Q4's 9.486-second candidate outlier is retained. OS
caches remain enabled and unrelated jobs continue. Owned builds/tests,
diagnostics and Doris/HDFS do not overlap primary timing windows. Driver,
installed Python files, packages, SQL, input identities and resource capacities
match across controls, apart from the intended native binary and validated
per-run temporary paths.

Separate diagnostics measure OS user-plus-system deltas summed over the two
pipeline workers, excluding the coordinator/result service. Mean CPU seconds
from two executions per variant, both retained, are:

| Query | Baseline worker CPU | Candidate worker CPU | Change |
| --- | ---: | ---: | ---: |
| Q10 | 302.920 | 253.715 | -16.24% |
| Q12 | 181.120 | 140.495 | -22.43% |
| Q18 | 411.385 | 410.000 | -0.34% |

Diagnostic wall times are excluded from latency. One late baseline status RPC
reached an already cleaned query and logged KeyError. All baseline queries,
result comparisons and resource checks succeeded, and every CPU observation
has complete before/after counters for both workers. No production diagnostic
handling was changed for this experiment.

A superseded trial demonstrates why the cost checks are necessary: Q7/Q19/Q21
regressed about 10–20% when additional bitmap barriers were costed against
inputs already reduced by earlier filters, or had insufficient traffic margin.
Its 65 returned results all matched; a final Q21 repetition was intentionally
cancelled when rejecting the trial. Six new regressions reproduce four failures
and two passing controls on that trial, then all pass with the cost correction.

After correcting costs, all 22 serialized graphs were compared offline,
normalizing only their native build identity strings and same-length embedded
occurrences. Nineteen graphs, including Q7/Q19/Q21, match the baseline; only
Q10/Q12/Q18 change. Raw hashes and engine identities are retained. Execution
identity guards remain enforced; normalized payloads are never executed.
Short regression screens confirm the three previously affected queries below.
Each screen has one steady sample per variant, so small changes are not claimed
as computation improvements.

| Query | Pipeline before | Pipeline now |
| --- | ---: | ---: |
| Q7 | 5.670 s | 5.609 s |
| Q19 | 6.000 s | 5.877 s |
| Q21 | 14.827 s | 14.389 s |

**650 distinct related tests pass** on the frozen final binary: 74 key-domain
cases, 25 lifecycle/boundary cases, 539 compiler/Parquet/join cases, and twelve
real-Ray cases. Coverage includes signed/wide keys, NULL-safe equality,
duplicate rows, cross-partition mask merging, existing early filters, cost
margins, residual/outer semantics, bounded channels, cancellation and shared
producer/FTE retry. The wide-key fixture preserves HUGEINT with DECIMAL storage
before restoring the integral type. The new fixtures do not claim forced-spill
or a hard peak-memory guarantee. No full or GPU suite was run; both changed test
functions live in a module already included in the sdist allowlist.

All **96 final SF100 executions** match the independent Local references:
72 primary timing queries, twelve regression screens and twelve separate CPU
controls. Schemas are checked with the existing wide-integer Arrow normalization,
row counts match, and values pass the frozen comparison policy. Superseded
trials and earlier profiles are retained separately and are not added to this
final-revision count.

Candidate code: `0f4b8e4dc688d2ce3fb8eb9e787ce50b1e7164b6`; native SHA-256
`fb8806ff2635bcea258b2c77117d3c2f6741427800c15c1797e3f5c815f64d95`.
Baseline: `7723e7e97c99cec9e8a9ab87ed2bc2d113db0738`; native SHA-256
`0eeda197744f3188dd92c8acc54fbec1272ac193a1848bb673e0f949111275e1`.
Evidence in `build/pipeline-sf100-join-exchange-20261025/` retains raw samples,
frozen identities/configurations, normalized/raw graph hashes, SQL/output hashes,
row counts, CPU/operator diagnostics, rejected-trial evidence, JUnit reports and
cleanup manifests. Owned temporary data, oracle outputs and duplicate baseline
files are removed after all run-owned processes stop. Original inputs are
preserved; Doris/HDFS is restored with file and Parquet page caches disabled.

Q4's scoped correlated path is unchanged: earlier native profiles show about
379 million fact rows before its delimiter join, versus about 24 million after
Local's dynamic scan filters. Q18 still exchanges about 150 million partial
states. These are separate remaining targets; this change introduces no
query-specific SQL rules or Local execution fallback.
