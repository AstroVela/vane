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
  50000 rows/second; consumer pauses remain part of end-to-end latency.
- **Mixed:** a paced pipelined scan delivers its first batch, then a sibling
  cursor submits an FTE aggregate to the same pool. The runner checks that the
  executions overlap and reports each query independently. A run with too few
  rows to overlap fails rather than reporting sequential execution as mixed.
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
