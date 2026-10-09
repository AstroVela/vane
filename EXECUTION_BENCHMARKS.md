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
