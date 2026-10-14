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
