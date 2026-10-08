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
- Unsupported window, median, inequality-join and UUID queries are rejected
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
