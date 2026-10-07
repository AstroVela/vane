# Execution acceptance

P5.2 compares the declared analytical SQL subset with native local execution,
then checks repeated query lifecycles on the shared Ray worker pool. The
[roadmap](PIPELINED_EXECUTION_ROADMAP.md#p52-差分与性能验收) tracks performance
measurements separately. This acceptance does not expand the supported SQL or
type surface.

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

Performance reports, capacity-default changes, CUDA/model UDF acceptance and
multi-node deployment qualification are outside this first P5.2 PR.
