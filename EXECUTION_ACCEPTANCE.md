# Execution acceptance

P5.2 compares the declared analytical SQL subset with native local execution,
then checks repeated query lifecycles on the shared Ray worker pool. The
[roadmap](PIPELINED_EXECUTION_ROADMAP.md#p52-差分与性能验收) tracks performance
measurements separately. This acceptance does not expand the supported SQL or
type surface.

Artifact-level local, Runtime and TLS Flight smoke tests and platform boundaries
are described in the [execution release matrix](EXECUTION_RELEASE.md). They
supplement this deeper acceptance suite for each installed candidate wheel.

## Reproduce

Use an installed, non-editable wheel matching the checkout, following
[DEVELOPMENT.md](DEVELOPMENT.md#incremental-package-build). Keep non-Ray and
shared-cluster Ray runs in separate processes:

```bash
export VANE_TEST_DIAGNOSTICS_DIR="$PWD/build/execution-acceptance"
scripts/run_installed_pytest.sh tests/fast/test_execution_acceptance.py tests/fast/test_direct_flight.py
scripts/run_installed_pytest.sh tests/fast/test_ray_execution_acceptance.py
```

The Ray tests use the repository's shared cluster fixture and its normal object
store sizing. Each query has bounded admission, execution and delivery deadlines;
each matrix or lifecycle case also has a pytest watchdog. The new modules are
included in the release launcher and source distribution. Full release/fast
execution remains a separate qualification step.

To repeat one recorded matrix entry, use its pytest id, for example:

```bash
scripts/run_installed_pytest.sh 'tests/fast/test_ray_execution_acceptance.py::test_seeded_sql_and_type_differential[3-1-970]'
```

## Correctness matrix

The checked-in seeds are `0` and `970`. Each produces 59 rows distributed among
four Parquet files, including one empty file. Row order, nullable/skewed keys,
decimal coefficients, floats and strings are deterministic. The scan permutes
the file references and repeats one reference deliberately. FTE must preserve
this scan multiplicity.

Each seed runs with `(partitions, worker threads)` equal to `(1,1)`, `(2,2)` and
`(3,1)`. Two workers execute both pipelined and FTE queries in the same session.
Frames and public batches contain at most three rows, exercising repeated
encoding, backpressure and nested-vector resizing.

Fourteen query shapes cover scan/filter/project, unordered duplicates,
FILTER/DISTINCT aggregates, ordered and ordinary floating aggregates, full hash
join, TopN/OFFSET, LIST/ARRAY/STRUCT/MAP, typed empty output, empty and all-NULL
aggregates, nested DECIMAL SUM with 39-digit intermediates, the complete HUGEINT
domain, TIME/INTERVAL boundaries and nonfinite floats. The matrix performs 168
distributed comparisons against native local references.

Comparison rules are explicit:

- Column names and Arrow types must match. The oracle widens native HUGEINT's
  `decimal128(38,0)` declaration to the documented distributed
  `decimal256(39,0)` representation, recursively. It never casts actual results
  to hide a schema mismatch.
- Unordered rows are compared as multisets, retaining duplicate counts and NULL
  distinctions. Queries with deterministic ORDER BY are compared in sequence.
- Only the two floating aggregate cases permit relative and absolute error of
  `1e-12`. NaN, infinity and NULL are checked explicitly. Ordered aggregates use
  deterministic input keys.
- Full-domain TIME and INTERVAL are cast to VARCHAR after aggregation. Their
  native values cross the internal exchange first; the oracle avoids the native
  Arrow exporter's narrower temporal representation.

A separate native test starts source partitions in all six permutations. Each
producer is fully drained before the next starts. Both HUGEINT and DECIMAL
SUM/AVG must retain the correct result even when positive prefixes exceed the
128-bit accumulator range. This controls actual arrival order instead of relying
on a timing delay.

## TPCH SF10 coverage acceptance

On 2026-10-12, all 22 Parquet TPC-H queries completed through local execution,
Ray pipelined and Ray FTE. Both Ray modes use the fragment compiler and native
TaskRuntime. Shared CTE, delimiter and MARK/SINGLE state stays in one native
subtree; independent inputs are still partitioned. This extends SQL coverage
without a local or legacy-FTE execution fallback. It does not establish support
for windows, ASOF joins, recursive CTEs or every native function.

Local used its default 36 threads and 50.1 GiB memory limit. Ray used two workers
with 18 CPU slots each and two partitions; operator memory used the node's
advertised memory after exchange, staging and result-service reservations. The
per-worker operator budgets were 28,414,316,544 bytes for pipelined and
27,768,844,288 bytes for the initial FTE sweep. Available memory varies at Ray
startup; each run records its actual budget. The object store was 1 GiB, with
128 MiB exchange and 256 MiB staging per worker. The worker pool was reused
within each mode.

These are single-machine, single-sample measurements with OS caches left intact,
not an official TPC-H score. Elapsed time covers `query()`, complete collection
and successful close; result comparison and Ray startup are excluded. Local
`query()` does most work synchronously, so its submission time is not comparable
to asynchronous Ray submission. All rows, duplicate counts, schemas and returned
row sequences matched local. Floating values use relative tolerance `1e-9` and
absolute tolerance `1e-6`. Q12's SQL HUGEINT reference column is widened from
local `decimal128(38,0)` to the documented Ray `decimal256(39,0)` profile; actual
results are not cast for comparison.

| Query | Local (s) | Pipelined (s) | FTE (s) |
| --- | ---: | ---: | ---: |
| Q01 | 0.523 | 1.611 | 21.567 |
| Q02 | 0.140 | 9.668 | 17.756 |
| Q03 | 0.371 | 12.141 | 54.268 |
| Q04 | 0.279 | 18.086 | 34.689 |
| Q05 | 0.406 | 19.920 | 81.892 |
| Q06 | 0.211 | 0.429 | 20.901 |
| Q07 | 0.415 | 9.244 | 53.785 |
| Q08 | 0.506 | 16.479 | 95.864 |
| Q09 | 0.836 | 21.732 | 125.515 |
| Q10 | 0.468 | 7.672 | 47.930 |
| Q11 | 0.063 | 2.606 | 10.799 |
| Q12 | 0.282 | 6.696 | 32.038 |
| Q13 | 0.544 | 5.514 | 13.647 |
| Q14 | 0.298 | 2.824 | 23.682 |
| Q15 | 0.295 | 0.641 | 20.528 |
| Q16 | 0.172 | 6.330 | 10.545 |
| Q17 | 0.376 | 49.236 | 97.866 |
| Q18 | 0.649 | 25.203 | 69.960 |
| Q19 | 0.407 | 4.622 | 26.439 |
| Q20 | 0.332 | 7.283 | 36.420 |
| Q21 | 1.163 | 58.828 | 89.903 |
| Q22 | 0.146 | 7.413 | 11.261 |
| **Total** | **8.882** | **294.178** | **997.256** |

The initial FTE store/query accounting ceiling was 128 GiB, with 4 GiB for
sources and 2 GiB per output object. Q8 has 63 outputs including broadcast
copies, requiring up to 130 GiB of reservations. It first failed at 95.452 s
with insufficient materialized storage capacity. Its successful retry used a
fresh store with a 160 GiB store/query ceiling and the same CPU allocation,
memory sizing policy, source allowance and per-object ceiling. The FTE total
sums the successful samples; it excludes that failed attempt and is not one
uninterrupted sweep. These limits are accounting ceilings, not preallocated
disk space. No storage accounting or cross-query snapshot cache was added.

Every query checked that query/result ownership and worker reservations had
retired. Closing each Runtime left no registered sessions, result contexts,
worker reservations, store allocations or source snapshots. The original
Parquet sizes and nanosecond modification times remained unchanged.

Artifacts are retained in this checkout at
`build/tpch-sf10-coverage-20261012/`: `samples.jsonl`, `timings.csv`,
`final-validation.json`, SQL and full Arrow results, per-query diagnostics,
run metadata and closed-runtime snapshots. Initial benchmark setup failures
(worker import path and an omitted result-reader memory reservation) are kept
separately; neither executed SQL. Production source is `4549ccaf9e`; the installed
native SHA256 is
`848049001800c07779e0c9e541ecdac681848da5e1ed26bf9135148f29f19703`.

Related regressions passed 1,094 non-Ray and 94 real-Ray cases; 23 optional
`fsspec` cases were skipped. No complete fast/release suite was run. The new
subquery and CTE regression modules are included in the release launcher and
source-package manifest.

## TPCH SF10 broadcast and checksum throughput

On 2026-10-13, a fresh baseline at `5dbc8f68b9` and the throughput increment
both ran Q6/Q8/Q17/Q21 in that order. Both environments used PyArrow 26.0.0,
NumPy 2.5.3 and Ray 2.59.0. The increment preserves broadcast probe fragments,
avoids value traversal when measuring fixed-width frames, and uses OpenSSL EVP
SHA-256 for frozen files and materialized objects.

CPU and memory sizing follow the preceding acceptance: 36 local threads,
two Ray workers with 18 CPU slots each, and all Ray-advertised memory after
the required reservations. Both runs use the same 160 GiB store/query ceiling,
4 GiB source allowance and 2 GiB object ceiling. Per-worker operator memory was
27,507,070,976 / 28,095,027,200 bytes for baseline pipelined / FTE, and
25,845,207,040 / 26,659,778,560 bytes for the increment. Available memory varies
at startup. This is a single sample on a shared host with OS caches retained;
elapsed time includes submission, complete collection and successful close.
Q6 also includes the first query's actor startup within each Ray mode.

| Query | Local after (s) | Pipelined before → after (s) | FTE before → after (s) |
| --- | ---: | ---: | ---: |
| Q06 | 0.252 | 1.432 → 1.351 | 22.415 → 13.112 |
| Q08 | 0.602 | 15.561 → 13.853 | 93.481 → 56.513 |
| Q17 | 0.416 | 44.307 → 40.764 | 100.796 → 59.926 |
| Q21 | 1.322 | 58.582 → 47.770 | 93.822 → 58.702 |

The increment continued through all 22 queries in each mode: local elapsed
9.814 seconds, pipelined 238.258 seconds and FTE 616.156 seconds. All 44 Ray
results matched local values, schemas and returned row sequences under the
previously documented floating tolerance and HUGEINT Arrow profile. Per-query
ownership and worker/store reservations retired; both closed Runtime snapshots
had no remaining sessions or results. Source sizes and nanosecond modification
times were unchanged. Only the four queries above have a same-day before/after
baseline; the complete sweep supplies additional correctness coverage.

The four-query elapsed sum changed from 119.882 to 103.739 seconds for pipelined,
and from 310.514 to 188.253 seconds for FTE. FTE Q6 submission changed from
21.732 to 12.406 seconds. Snapshot ownership, SHA-256 identities and full
materialized-object verification remain in place.

Nine TPC-H plans have fewer fragments. Q8 changes from 17 fragments / 16
exchanges to 14 / 13; Q21 changes from 10 / 9 to 9 / 8. Q17 retains its
five-fragment plan. A separate bounded-channel regression joins a 100,000-row
probe with two small builds and aggregates before exchange: its channels record
seven accepted rows in total. These counters are from that synthetic query,
not from the SF10 timings. Correlated shared-state scopes still run in a single
partition; their distribution and distributed runtime filters remain future work.

Related regressions passed 1,295 non-Ray and 94 real-Ray cases, with 23 optional
`fsspec` skips. Two static review rounds preceded one non-editable native build.
Test SQL aliases and two fixtures depending on the old broadcast graph were
corrected and rerun; production code did not change after that build. The
complete fast/release suites were not run.

Baseline artifacts are in `build/tpch-throughput-baseline-20261013/`; increment
artifacts are in `build/tpch-throughput-after-20261013/`. They retain SQL, complete
Arrow results, dependency/resource metadata, timings and ownership snapshots.
The increment's installed native SHA-256 is
`037b0a352a805ac5a84f21e7ced41b5d40ee51e80b4a945490b9210595421e47`.

## TPCH SF10 correlated partitions

On 2026-10-13, Q17 and Q21 were measured before and after partitioning native
delimiter scopes by a proven integer correlation key. The baseline installed
native came from `1f5e7731a2`; the increment was built on `0c5b165b08` plus the
correlated-partition change. The intervening review fixes are included, so this
is a stack comparison rather than an isolated single-patch experiment.

Local used 36 threads. Ray used two workers with 18 CPU slots each, two
partitions and all advertised memory after required reservations. Baseline
pipelined / FTE operator memory per worker was 27,323,357,184 / 26,905,368,576
bytes; the increment used 25,289,068,544 / 25,097,426,944 bytes. The object store
was 1 GiB, with 128 MiB exchange and 256 MiB staging budgets per worker.
FTE kept the 160 GiB store/query accounting ceiling, 4 GiB source allowance
and 2 GiB object ceiling. Each mode reused its runtime for alternating Q17/Q21
runs. Timings include submission, full collection and successful close;
comparison is excluded. The host was shared and OS caches were retained.

| Query | Local after (s) | Pipelined median before → after (s) | FTE median before → after (s) |
| --- | ---: | ---: | ---: |
| Q17 | 0.421 | 39.577 → 31.076 | 60.807 → 51.079 |
| Q21 | 1.203 | 50.277 → 31.789 | 59.348 → 47.785 |

Each Ray median has three successful samples except baseline pipelined Q17,
which has two; an earlier run missing the comparison schema was excluded.
Local has one sample. All twelve increment Ray runs matched the full local
values and schemas, with the documented HUGEINT profile and floating tolerance.
Worker reservations, query/store ownership and result contexts retired after
every query, and both Runtime snapshots were clean after close. Source sizes
and nanosecond modification times remained unchanged. FTE submission still
spends roughly 10–13 seconds creating query-owned snapshots; snapshot reuse is
not part of this increment.

Q17 now has six fragments, with its delimiter scope running in two partitions
and a final gather aggregate. Q21 has twelve fragments, with both delimiter
scopes partitioned by order key. Its 59,986,052-row inner lineitem exchange
changed from one target to 29,992,594 and 29,993,458 rows across two targets.
One-second diagnostic samples observed both Q17 delimiter tasks running.
Other sampled exchange counters are lower bounds when cleanup preceded the
last sample; they are not exact total traffic measurements.

Two clean manual static reviews preceded one non-editable incremental Release
build. Related tests passed 933 non-Ray and 60 real-Ray cases, with 18 optional
`fsspec` skips. Test-only corrections addressed reserved aliases, unsupported
macros and connection/optimizer setup; production code stayed unchanged after
the build. The complete fast/release suites were not run.

Artifacts are in `build/tpch-correlated-baseline-20261013/`,
`build/tpch-correlated-after-20261013/` and
`build/correlated-partition-20261013/`. They retain SQL, full Arrow results,
graphs, timing samples, resource settings and lifecycle evidence. The increment
native SHA-256 is
`b34a0f6831a52cbcd1c6133d6223aa6a2690e7c771ea03b1ce47a10594c4d179`.

After adding the separately reviewed IE_JOIN serialization and parallel outer
payload fixes from #999, the combined stack passed 965 related non-Ray and
64 real-Ray tests, with 18 optional `fsspec` skips. The Ray rerun explicitly
disabled GPU discovery because host NVML failed before fixture startup; these
SQL tests require only CPU resources. Its native SHA-256 is
`12e32106ef84ecee5736f62369e404f815b1a9b6fac61209bef720958a7a5304`.
This later regression build does not replace the benchmark identity above.

## Repeated lifecycle checks

The same connection and worker pool are reused across each sequence:

- Three rounds of backpressured pipelined output while a short FTE aggregate
  completes, followed by early close or interrupt. Exported Arrow views remain
  valid and charged until their final release.
- Three rounds of cancellation during queued session admission, followed by a
  successful query in the same mode.
- Two FTE worker losses before a downstream attempt commits. Retries must use
  the same input identity and a new fence, deliver exact results without
  duplicates, and leave the pool usable for pipelined queries.
- Unsupported window, median, ASOF-join and UUID queries are rejected
  without falling back to local execution; all admission/storage ownership is
  released and a supported query still succeeds.
- A real long-running filter alternates a 60-second execution deadline and a
  two-second deadline, twice per mode. Native execution, status monitoring and
  Flight remain enabled. A short query follows every success or cancellation.

At each idle boundary the checks require no active queries, queued or active
admissions, retained result bytes, cleanup-pending results, worker reservations,
worker waiters, store leases or store reservations. Completion counters are
allowed to increase.

## Failure evidence and Flight investigation

Each run prints its evidence directory before executing. When
`VANE_TEST_DIAGNOSTICS_DIR` is set, evidence survives the launcher's disposable
working directory and is included in CI's existing diagnostic artifact. The
directory contains:

- Configuration, Python/platform/package versions and native engine identity.
- Seeded Parquet input files, the ordered file references and SHA-256 hashes.
- The active case and standalone `replay.sql`, written before execution.
- Completed comparisons and a success report, or the original failure, Python
  thread dump and available query/session diagnostics captured before explicit
  caller cleanup. An exception in a diagnostic probe preserves the original
  failure.

The historical P5.1 FTE slow-filter failure reported a generic Flight timeout.
Flight now names `data open`, `data schema`, `data next` or the control operation
(`status`, `ack`, `close`) in errors. Controlled stalled-server tests verify
status, ACK and data-read timeout attribution. They retain the existing timeout
and failure semantics; attribution alone does not establish or fix the cause of
the historical timeout.

### Flight proxy isolation

Vane's native exchange/result connections and public Python `Client` connect
directly to their explicit Flight endpoints. Each channel sets
`grpc.enable_http_proxy=0`; Vane does not change process-wide proxy variables.
This avoids routing cluster traffic or a public session through an unrelated
HTTP proxy. Deployments must make both advertised server ports and the internal
worker endpoints directly reachable. TLS certificate/hostname verification and
the existing RPC/data deadlines still apply.

The [gRPC proxy mapper](https://grpc.github.io/grpc/core/md_doc_core_default_http_proxy_mapper.html)
otherwise consults `grpc_proxy`, `https_proxy` and `http_proxy`, in that order,
unless the target is excluded. The test machine had a loopback proxy configured
and excluded localhost, but not its Ray node address. An earlier native Flight
error recorded that proxy as the peer.

`test_flight_proxy_isolation.py` launches each probe in a fresh interpreter and
uses a loopback CONNECT proxy restricted to the test's registered listeners.
It verifies a healthy initial handshake, pauses forwarding, and checks all
three environment variables independently. Before the fix, the native stream
aborted with `direct Flight control status` after about 2.03 seconds despite a
20-second data deadline; the public Client also timed out. The baseline produced
six expected failures and three passing raw-Arrow negative controls. The
regression requires Vane to make zero proxy connections and complete native
data, ACK and FINISH. Raw Arrow must still time out through the same proxy,
proving the fault injection remains active. The child environment must stay
unchanged after Vane creates and closes its connections.

After two clean review rounds and one incremental Release build, the nine proxy
cases passed. The focused validation on 2026-10-09 completed with 119 non-Ray
and 25 shared-Ray tests passing; one optional ADBC test was skipped. This covers
native operation-attributed timeouts, public session controls, dual-port TLS,
and the original/repeated long-filter success and execution-timeout cases in
both modes. No complete release/fast suite was run.

Run the focused regression and native transport tests with the installed wheel:

```bash
scripts/run_installed_pytest.sh tests/fast/test_flight_proxy_isolation.py tests/fast/test_direct_flight.py
```

This establishes a reproducible proxy-induced control timeout. The historical
P5.1 failure did not record its operation or a proxy trace, so it is not possible
to assert that the same cause explains that particular event. Keep its diagnosis
open and retain operation-labelled evidence from repeated long-filter tests;
passing reruns alone do not resolve that uncertainty.

The subsequent [execution benchmark](EXECUTION_BENCHMARKS.md) measures startup,
warm execution, slow clients, mixed modes and recovery with explicit boundaries.
CUDA/model UDF acceptance and multi-node deployment qualification remain separate.

## Remote server acceptance

P5.2.4c exercises the public Flight client against the same shared Runtime. Run
these related modules with the installed wheel; the CLI tests own their clusters
and must run in a separate process:

```bash
scripts/run_installed_pytest.sh tests/fast/test_flight_server.py tests/fast/test_server_sessions.py tests/fast/test_server_queries.py tests/fast/test_execution_benchmark.py
scripts/run_installed_pytest.sh tests/fast/test_ray_server_acceptance.py tests/fast/test_ray_server_queries.py tests/fast/test_ray_server_sessions.py tests/fast/test_ray_execution_benchmark.py
scripts/run_installed_pytest.sh tests/fast/test_server_cli.py tests/fast/test_execution_benchmark_cli.py
```

The remote failure matrix checks these ownership boundaries:

| Event | Evidence required |
|---|---|
| Kill a separate client process during admission or streaming, in both modes | Actual session lease expires; query admission, worker reservations, gateway capabilities and FTE store reservations disappear; another session and the shared workers remain usable |
| Lose Execute acknowledgement and stop heartbeat | The accepted sequence remains owned until expiry; no new submission or caller close is needed to reclaim it |
| Cancel an opening session RPC | The late native connection is closed, even though no handle reached the caller |
| Worker release submission or ObjectRef temporarily fails | Closing session and charged resources survive multiple failures; a fresh release acknowledgement allows cleanup after recovery |
| Result release ObjectRef temporarily fails | Query/result admission is retained until a fresh RPC succeeds; the persistent result actor remains usable |
| Kill the result actor | Both resident pipelined and FTE results fail; the service does not silently replace it or report successful delivery |
| Kill a pipelined worker after delivery starts | The public reader fails; remaining results are not presented as a complete answer |
| Kill an FTE worker before downstream commit | Benchmark validates exact values, unchanged input identity and a fresh retry fence |
| Slow or stalled client | Native receive/gateway window peaks stay within configured bounds; retained Arrow views remain charged; another session can finish |
| Restart on the same control address | A new server ID rejects all old session/query controls and Execute requests without reserving new owners |

Failure injection uses real Ray actors and real ObjectRefs. Transient RPC tests
control the acknowledgement path; they do not simulate a network partition or
prove availability during one. Failure cases have watchdogs, and idle assertions
inspect the coordinator, actual worker actors, result service and store ledgers.

Deployment checks include a separate client without `ray.init()`, the standalone
CLI's Ray ownership and SIGTERM cleanup, database reopening from another process,
and wildcard binding with a reachable advertised hostname and verified TLS on
both public endpoints. This is single-host acceptance. Cross-host networking,
external certificate lifecycle and abrupt server-process recovery are separate
deployment qualification; sessions are not recoverable across a server restart.

The [benchmark](EXECUTION_BENCHMARKS.md) has explicit `runtime` and `flight`
interfaces. Both use the same workload, correctness oracle, worker budgets and
mixed-mode reservation evidence. Flight timing includes actual control RPCs and
native result transfer, with the client colocated in the server's process; the
separate-process correctness tests must not be interpreted as latency samples.
