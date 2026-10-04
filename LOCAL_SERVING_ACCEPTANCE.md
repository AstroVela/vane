# Local serving acceptance

This scenario validates the public local-fast SQL and Relation lifecycle in
[LOCAL_MODEL_RUNTIME.md](LOCAL_MODEL_RUNTIME.md), continuing
[#843](https://github.com/AstroVela/vane/issues/843) under
[#838](https://github.com/AstroVela/vane/issues/838). It combines model reuse,
request/task/data admission, cancellation, and managed result delivery in one
session. Separate sessions validate short queue and zero execution deadlines,
because public runtime configuration is fixed for the session. The scenario
continues both issues without introducing an HTTP/RPC endpoint.
The default workload uses CPU. The sustained runner also supports an explicitly
assigned CUDA device through the same public APIs and lifecycle checks.

The [runnable example](scripts/validate_local_serving.py) uses
`connection.configure_local_runtime()`, `runtime.register_model()`,
`model.prewarm()`, and `vane.attach_function()` for setup. Clients execute
parameterized SQL through `cursor.execute_result()` or build projections with
the registered model and call `relation.execute_result()`. They consume and
close `QueryResult` handles. Independent control threads cancel queries with
`cursor.interrupt()`. The driver never constructs physical plans, model payloads,
node bindings, or internal request tickets. See the
[public API example](LOCAL_MODEL_RUNTIME.md#managed-results-from-sql-and-relation-queries)
for the result-delivery configuration and ownership contract.

## Run the installed candidate

Install a non-editable wheel following [DEVELOPMENT.md](DEVELOPMENT.md). From
the checkout, using the Python interpreter containing that wheel:

```bash
python -I scripts/validate_local_serving.py \
  --requests 20 --concurrency 4 --report build/local-serving-report.json
```

The CLI sets `VANE_RUNNER=local-fast` in its own process and starts workers from
a temporary directory so that the source package cannot shadow the installed
native extension. It removes its generated fixtures after cleanup. The report
path is relative to the original working directory. No downloaded weights,
external services, credentials, image decoder, GPU, or additional Python
dependency is required. Use the installed Python-only candidate when changing
the runtime; changing the checkout alone does not update the tested package.

The command fails on an incorrect result, unexpected acceptance/rejection,
timeout, or retained owner at a quiescence checkpoint. It writes a success
report only after every scenario and final runtime cleanup succeeds. Exit
status is the acceptance signal; an old report from an earlier invocation is
not evidence that a failed invocation passed.

## Workload and checks

The fixture model consumes the text `red green blue` and an 8 by 8 RGB image
represented by raw interleaved pixel bytes. It returns word count and normalized
mean color as four numeric features. A fixed hash loop provides repeatable CPU
work. These are synthetic features, with no learned embedding or retrieval
quality claim. Both inputs and expected features are generated locally.

One explicitly registered subprocess actor is shared by the main session's
SQL and Relation queries. Each client uses an independent cursor and rebuilds
its query through the public API. The runtime has two active request slots,
two queued request slots, one running task slot, eight queued task slots, and a declared resident budget
of one CPU and 16 MiB heap. The shared-memory data budget is 2 MiB, with 64 KiB
input and 128 KiB output envelopes per task. Managed delivery has two result
slots and 64 KiB of retained IPC buffers. Queue and execution timeouts are
30 seconds. Heap declarations are logical reservations, not operating-system
limits. The two deadline sessions use the same resource limits, with a
two-second queue timeout or a zero execution timeout, respectively.

The phases run in this order:

1. Registration binds the model without initializing a worker. A cold SQL
   request initializes one worker. Explicit prewarm and subsequent sequential
   SQL and Relation requests add no initializations and use the same worker PID.
2. Concurrent clients mix one-row serving queries with 32-row analysis queries.
   All complete with correct features and the same resident model. Request
   admission is FIFO and task arbitration uses the existing shared policy;
   there is no preemption or short-query priority. One long running UDF can
   delay short requests. This scenario reports that cost without claiming a
   latency bound or exercising every possible scheduling interleaving.
   The driver retries a result-slot refusal for at most 30 seconds, only after
   checking `error.reason == "slots"` and `error.execution_started is False`.
   Fixture call markers independently assert that the refused attempt ran no
   UDF; they do not select the retry policy. Each public retry creates a new
   admission ticket. Byte refusal, unknown execution state and execution errors
   are not retried. Argument/binding callbacks before admission are outside this
   query-execution guarantee; the driver does not provide general exactly-once
   semantics for arbitrary caller-side effects.
3. Gated native queries fill both active and both queued request slots. Excess
   work is rejected before running user code; queued work occupies no result
   slots. Interrupting a queued cursor releases its ingress slot without running
   its UDF. Releasing the gates lets the remaining accepted clients finish
   exactly once.
   Request capacity can return before the earlier result is consumed, so this
   phase also permits the verified slot retries above. Completion order across
   these new tickets is not an assertion about FIFO admission of the old ones.
4. Two unconsumed results occupy both delivery slots. Three public calls refuse
   without running user code or retaining private request tickets, then a call
   executes exactly once after a result is consumed. A retained Arrow table,
   followed by its zero-copy NumPy view after the table is dropped, keeps its
   bytes charged after final handoff. A second large result fails byte admission
   after its UDF runs once. Releasing the view restores capacity for a new call.
5. Delivery cancellation and abandoned-result expiry release pending results.
   Expiry can also be reported while publishing the result, before the caller
   receives its handle; both paths must count one timeout and finish cleanup.
   `cursor.interrupt()` cancels a gated, running subprocess UDF. Subsequent
   requests succeed using the still-registered model.
6. An ordinary UDF exception is counted and leaves the registration and pool
   reusable. The local adapter retires that worker gracefully; a new request
   initializes one replacement. An intentional worker exit also fails its
   request without replay; another new request initializes one replacement
   and succeeds. Recovery initializations are reported separately from
   healthy model reuse.
7. Every recovered phase has zero query borrows, request/task owners,
   shared-memory reservations and leases, and managed result bytes. Resident
   resources stay charged until runtime close. Drain rejects new requests;
   final close returns the resident reservation as well.
8. In a separate session, two gated queries hold admission while another cursor
   queues and expires without running its UDF or acquiring result ownership.
   A final session sets execution timeout to zero: both SQL and Relation calls
   expire without initializing a worker.
   Each session returns all resources on close and has its own report counters.

The script deliberately kills only its own fixture worker via `os._exit(23)`
inside that worker's designated failure request. Constructor/call markers live
in the temporary fixture directory and are not copied into the report.

## Report and measurement boundaries

The version-2 JSON report includes configuration, environment versions,
initialization counts, one observed injected worker-exit failure, per-API load
counts, latency/delivery distributions, mixed throughput, and resource snapshots
at pressure and recovery checkpoints. `phase_request_metrics` contains queue,
execution, and cleanup totals from public runtime snapshots for cold, warm,
and mixed load phases. Public calls do not expose their request tickets, so
these totals are not reported as per-request distributions or separate
mixed-short/mixed-analysis timings. Slot retries can increase admitted counts
without increasing executed counts. `deadline_sessions` contains separate
configurations and counters; those sessions do not contribute to the main
model reuse counts or load latency samples.
Worker-exit observations in the driver are scenario evidence. The runtime's
`worker_failures` snapshot separately reports initialization failures,
worker-reported execution errors, worker losses, adapter errors, cancellation
and ordinary closure. These are once-per-worker-generation observations;
`request_admission.failed_executions` still includes preparation failures that
start no worker. See the [worker metric boundaries](LOCAL_MODEL_RUNTIME.md#worker-failure-metrics)
for shared-pool attribution and observation limits.
No per-request exception, plan, Arrow view, or unbounded sample history is
stored in runtime metrics. The finite benchmark collects scalar samples in the
driver to compute its report.

For each cold, warm, mixed-short, and mixed-analysis group, latency and delivery
distributions give count, mean, maximum, and nearest-rank P95/P99 in seconds.
An empty group has count zero and null statistics. Small samples are descriptive; use more
requests and repeat the command for performance work. Correctness checks do
not assert throughput or latency thresholds.

| Measurement | Interval |
| --- | --- |
| Request latency | Cursor creation through SQL/Relation binding, admission, result consumption and cursor close; includes verified slot retries |
| Queue wait | Ticket creation through promotion to ready; excludes time held ready before execution claim |
| Execution | Successful claim through preparation, native execution, and cancellation callback completion; excludes subsequent query cleanup |
| Cleanup | End of execution through confirmed return of the request slot; includes time awaiting explicit cleanup retries |
| Delivery | Result ready through confirmed result-slot retirement; includes pending-result cleanup, excludes subsequent external view lifetime |
| Mixed throughput | Completed mixed requests divided by phase wall time, including cursor creation, binding, consumption and close |

The execution counters update once when execution ends, even if cleanup still
owns the request slot. `failed_executions` counts preparation/native errors
after a claim, excluding accepted cancellation, execution expiry, cleanup-only
errors, result-slot refusal, and post-execution result encoding/byte refusal.
`completed_requests` keeps its existing meaning: non-cancelled request slots
returned after cleanup, including failed executions. Delivery totals include
only results that became ready and subsequently retired; failed preparations
have no delivery sample. Unstarted or unfinished per-handle intervals are null.

Managed results are materialized, with one IPC payload per nonempty native query.
Native collection, DuckDB memory, temporary encoding overlap, external
serialization, and network sends are outside the retained-result budget.
Exported views stay byte-charged after slot retirement, but cannot be forcibly
freed while a caller retains them. Delivery expiry ends at handoff to the
caller. Slow consumers here mean delayed iterator consumption and retained
Arrow/NumPy views; a real transport must own sends, disconnect cancellation, and
its own references. Managed native streaming is available through
[`execute_result(stream=True)`](LOCAL_MODEL_RUNTIME.md#managed-native-result-streams);
the sustained serving fixture's `--streaming` mode validates it alongside
materialized delivery, as described below.
Transport adapters remain subsequent work. Fixed-device GPU models are supported as described in
[the runtime guide](LOCAL_MODEL_RUNTIME.md#registered-local-gpu-models).

## Regression gate

### Sustained lifecycle acceptance

Run repeated healthy load, pressure, cancellation and worker recovery in **one**
runtime, supervised by a separate process:

```bash
python -I scripts/validate_local_serving_soak.py \
  --output /tmp/vane-serving-soak-new \
  --rounds 20 --requests 40 --concurrency 4 --timeout 600
```

The output directory must be new. The watchdog requires POSIX process groups
and `SIGUSR1`; the existing short acceptance scenario remains available on
other platforms. `--timeout` bounds the whole child run, including startup and
cleanup, rather than resetting whenever a diagnostic thread is alive. Increase
it explicitly for longer runs. A timeout or unsuccessful child returns a
nonzero exit code, even if cleanup hangs or the child leaves no final report.

Each round runs warm SQL/Relation queries and concurrent mixed-size queries,
checks healthy worker identity, and repeats ingress pressure, retained Arrow
and NumPy views, result deadlines, cancellation, UDF errors and worker exit.
After releasing results, request/task/data/result ownership must return to the
same idle baseline. The registered model's CPU/heap reservation remains resident
until final drain and close. Passive physical transport and pool snapshots must
also show no shared-memory usage, input holds, pending grants or occupied worker
slots at those idle checkpoints. Fault recovery is counted separately from healthy
reuse; an injected failed UDF must run once, without automatic replay.
The short CPU scenario's separate queue/execution-deadline sessions remain
separate coverage, since session configuration cannot change during the soak.

### Sustained streaming delivery

Add `--streaming` to the same supervisor and workload:

```bash
python -I scripts/validate_local_serving_soak.py \
  --streaming --output /tmp/vane-streaming-soak-new \
  --rounds 20 --requests 40 --concurrency 4 --timeout 600
```

The default mode keeps materialized delivery. Streaming mode uses the same
registered model, request/task/data budgets, independent client cursors, fault
probes and watchdog. Warm load alternates SQL and Relation streams. Concurrent
load mixes these streams with short materialized queries through one runtime.
Each stream returns 48 text/RGB feature rows with 2 KiB of padding per row,
delivered in batches of at most 16 rows. The logical result exceeds the 64 KiB
delivery budget, while one UDF output fits its separate 128 KiB envelope.
Every complete query checks all row identities, feature values and worker PIDs.
Consumers release each Arrow table before pulling the next batch.

Each round also retains a batch until the public snapshot reports a byte
wait, transfers that ownership to a zero-copy NumPy view, and verifies that
the same bytes remain charged. Releasing the view permits the next batch;
closing the partially read stream then discards its unread output. Separate
byte waits exercise cursor cancellation, a two-second delivery deadline and
a five-second execution deadline. Retained consumer views stay valid and
charged through each outcome. Recovery queries must succeed before proceeding.
The model is prewarmed before the first query so initialization is outside the
short execution deadline; the report's first-query timing is not cold startup.

UDF errors and worker exits are exercised through streaming execution and
consumption, with the existing exactly-one-call marker and worker-outcome
checks. Each injected failure has one replacement and no automatic replay.
Healthy load and byte-pressure release add no initialization. Cancellation
and deadline replacement counts are reported separately. Every round returns
request, task, data, result and physical transport ownership to idle. Final
drain/close runs with an outstanding byte-waiting stream, verifies that the
waiter exits and admission returns, and releases its surviving consumer view
before checking the closed baseline.

The report's `configuration.streaming` selects the mode. Each bounded
`recent_rounds[].streaming` entry records query/row/batch/logical-byte counts,
first-batch and consumption distributions, pressure recovery timings and
worker replacements. First-batch latency starts before cursor creation and
includes binding, admission and native preparation. Consumption starts when
the managed result is returned and ends after EOF/cleanup and cursor close.
The control timings start after observing byte pressure: cancellation timing
includes the interrupt and cleanup, while expiry timing also includes the
remaining deadline wait. They are observations, not latency guarantees.
`sampled_peak_delivery_bytes` is the running maximum of sampled IPC reservation
bytes; it excludes DuckDB operator memory, decoded batches, encoding overlap
and network buffering. `stream_shutdown` records the final close probe. Only
scalar summaries and the last eight round records are retained.

The CPU release gate runs both modes. The real-CUDA test below also runs both,
and `--streaming` can be combined with `--gpu-device` for a longer device run.
The public stream ownership and memory boundaries are documented in
[managed native result streams](LOCAL_MODEL_RUNTIME.md#managed-native-result-streams).

### Sustained CUDA delivery

For real CUDA acceptance, provision one otherwise available physical GPU and
pass its full UUID. PyTorch with compatible CUDA support must be installed in
the same environment as the wheel; no models or weights are downloaded:

```bash
python -I scripts/validate_local_serving_soak.py \
  --gpu-device GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  --output /tmp/vane-cuda-serving-soak-new \
  --rounds 20 --requests 40 --concurrency 4 --timeout 900

scripts/run_installed_pytest.sh tests/fast/test_local_serving_soak_cuda.py -m gpu
```

Replace the placeholder with a full provisioned UUID, for example from
`nvidia-smi --query-gpu=uuid --format=csv,noheader`. Ordinals, UUID prefixes and
MIG devices are rejected. The CUDA path uses the same synthetic text/RGB
features as CPU; RGB means and scaling run on device tensors, with a host copy
waiting for device completion before the UDF returns Arrow data.

One runtime owns one fixed GPU replica throughout all rounds. CUDA initialization
is explicitly prewarmed and timed before the first query and after expected
worker replacement. A fixed five-second execution deadline then applies to warm
queries. Each CUDA round adds a gated execution-timeout/recovery probe to the
existing cancellation, UDF-error and worker-exit probes. Healthy requests must
keep their PID/generation; each observed replacement must advance the generation
once and preserve the device assignment. Failed UDFs are never automatically
replayed. The GPU execution records, ready slots and retained slots must be
empty at idle checkpoints; the resident GPU reservation stays charged until
final close. Final close must remove both the pool and its worker process.

`worker-report.json` includes the CUDA/PyTorch/device identity, prewarm time,
per-round worker generations and sampled allocator peaks. The latest worker
sample is overwritten in `work/gpu-worker.json`. CUDA allocated/reserved bytes
describe PyTorch's allocator, including its cache; they are observations, not a
VRAM limit, a whole-device measurement, or a requirement to return to zero while
the model remains resident. The CPU/heap declarations also remain logical
admission budgets. The two-round CUDA pytest is marked `gpu` and runs separately
from CPU CI; the default CPU soak and watchdog regression tests remain in the
base release gate.

A local run on 2026-09-29 used Python 3.12.14, PyTorch 2.7.0+cu126 and one
RTX 2080 Ti, with the command above (20 rounds, 40 requests per load phase,
four clients):

- One runtime completed 1,601 load requests plus pressure/fault probes in
  435.28 seconds of supervised wall time. Each round returned to the same idle
  ownership baseline, with no additional initialization during healthy work.
- The run observed 81 initializations: one cold worker and 20 replacements each
  for cancellation, warm execution expiry, reported UDF error and worker exit.
  Runtime counters reported 20 execution errors, 20 worker losses, 40 cancelled
  workers and zero initialization/adapter errors. No failed UDF was replayed.
- Final close left zero resident resources, data/result/transport bytes and GPU
  workers. The retained round history stayed at eight entries.
- Sampled PyTorch allocator peaks were 2,560 allocated bytes and 2,097,152
  reserved bytes for this small synthetic tensor fixture. These exclude CUDA
  context/driver memory and do not estimate a production model's VRAM needs.

The 971 result-slot refusals in the load phases were verified pre-execution
refusals and retried as new requests. They are included in request latency and
admission counters, not successful UDF execution counts. This is one machine's
lifecycle evidence, not a throughput target or evidence that #841 is resolved.

Reports and diagnostics are written incrementally:

| File | Meaning |
| --- | --- |
| `report.json` | Supervisor outcome, child exit status, wall time and successful child report |
| `worker-report.json` | Final round counts, initialization counts and closed-runtime ownership |
| `progress.json` | Last entered phase and round, including startup and shutdown |
| `resources.json` | Latest completed passive runtime, transport and pool snapshot, with sampling times |
| `idle-owners.json` | Last synchronous transport/pool ownership check, including evidence if the check fails |
| `rounds.json` | Most recent eight completed rounds and their idle snapshots/latency summaries |
| `threads.log` | Python thread stacks on failure or watchdog expiry |
| `failure.json` / `worker.log` | Primary failure before outer teardown, and child diagnostics |

The pytest soak stores evidence in a unique `serving-soak-*` (CPU) or
`cuda-soak-*` (CUDA) subdirectory of `VANE_TEST_DIAGNOSTICS_DIR` when configured,
so CI uploads the files even after a watchdog timeout. Without that setting it uses pytest's temporary directory.
The installed, release and fast-test launchers resolve relative diagnostic roots
against the caller's working directory before entering their temporary test
directories. The test prints the evidence path before launching the supervisor.

Snapshots do not create pools, obtain task grants or call active admission
callbacks. They are observations of separate components, not an atomic global
state. A snapshot can be stale or unavailable if its locks are blocked; inspect
the sampling timestamps and thread stacks. The sampler is a daemon with bounded
shutdown waiting, and it cannot extend the supervisor's deadline. On expiry the
supervisor requests stacks, then terminates the isolated child process group,
including its inherited actor processes. This forced teardown is failure
containment, not evidence that runtime cleanup succeeded.
The worker keeps the signal handler and stack-log descriptor alive through
interpreter shutdown, including blocked thread joins and `atexit` hooks; a
completed worker report alone does not establish a successful process exit.

Per-request samples and fixture markers are discarded after each round. Only
eight round summaries, scalar totals and the most recent resource snapshots
remain; no unbounded sample or exception history is kept. Quantiles describe
individual retained rounds, not the whole run. Passing a soak demonstrates the
tested schedule and budget configuration; it does not establish a latency SLO
or prove the root cause of a historical timeout.

### Integrated stage acceptance, 2026-10-02

The original runtime contracts in [#839](https://github.com/AstroVela/vane/issues/839)
through [#843](https://github.com/AstroVela/vane/issues/843) have merged into
`feature/local-runtime`. Stage acceptance includes the explicit unresolved
historical timeout described below and tracked in
[#929](https://github.com/AstroVela/vane/issues/929).
The parent [#838](https://github.com/AstroVela/vane/issues/838) retains final
integration into `main` and review of that disposition.

| Version identity | Verified value |
| --- | --- |
| Integration commit after #928 | `6a4b1bbe7d9dc9c4d5ebb2d19bede531093279b3` |
| Final reviewed/tested #928 head | `f8408b86ec517479ad44e239086fe229b083f504` |
| Identical Git source tree for both commits | `fd4a0d93d11c40fad2dab1a036ad006f1949bed6` |
| Local installed environment | Linux x86-64, Python 3.12.14, non-editable `vane` 0.3.0.dev72 |
| Native engine | `v1.5.5-vane.fbdcbd56fa`, source ID `a1b4927e0ad741903521aacc7fcf82a74620a269` |
| Installed Python/type source comparison | All 261 tracked `.py`/`.pyi` files match the integration commit |

The [final-head CI run](https://github.com/AstroVela/vane/actions/runs/36876707323)
and [Required CI](https://github.com/AstroVela/vane/actions/runs/36876707323/job/110502643747)
passed. Evidence covers Python 3.10–3.14 native builds/tests and source packages,
Linux fast-test shards, macOS arm64 and Windows x64 native tests, separate shared
and isolated-owner Ray processes, AI tests, and Doris/Qdrant/Milvus integration.
Changed-file lint/format/type checks, workflow checks and dependency review also
passed. These checks ran on `f8408b86ec`; their source tree equals the merge
commit's tree. They are not separate CI executions on the merge commit.
Real CUDA acceptance runs separately from the CPU CI shards.

| Latest delivery | Merged into `feature/local-runtime` |
| --- | --- |
| [#915](https://github.com/AstroVela/vane/pull/915): sustained CUDA serving acceptance | 2026-09-30, `db3166a3ade4e6f8787c20282012d9ac490fd57c` |
| [#918](https://github.com/AstroVela/vane/pull/918): managed native result streams | 2026-10-01, `28ca906916d8b5534d203c51624cb696f68335c9` |
| [#928](https://github.com/AstroVela/vane/pull/928): sustained CPU/CUDA streaming acceptance | 2026-10-02, `6a4b1bbe7d9dc9c4d5ebb2d19bede531093279b3` |

| Original step | Acceptance evidence |
| --- | --- |
| #839: preparation rollback | #844's common helper; local/Ray adapter tests cover reverse order, identity deduplication, borrowed pools, primary errors and retained cleanup ownership |
| #840: resident model ownership | #847 registry and #901 public registration; native SQL/rebuilt Relation reuse, concurrent borrows, captured sessions, cancellation isolation, worker replacement and prewarm/shutdown ownership |
| #841: resources and backpressure | Common leases/byte arithmetic, fair task and worker admission, retained input/output ownership, per-UDF attribution, strict envelopes and bounded byte waiting; #892 native progress and Local/Ray contract acceptance |
| #842: fixed GPU admission | #910–#912 resident/execution/device ownership; CPU contract tests and real CUDA model reuse, failure recovery, cancellation, deadlines and final release; #915 sustained device scenario |
| #843: serving lifecycle | Bounded request/queue/result admission, cancellation/deadlines and cleanup retries, worker metrics, public SQL/Relation results; #918 streams and #928 slow-consumer/fault/close soaks |

At the final #928 head, the installed-package release gate passed **3,471 tests**:
3,395 non-Ray, 74 shared-Ray and two isolated cluster-owner tests. Eight optional
dependency checks skipped (seven Qdrant and one ADBC). Its 115 related tests
included native streaming cleanup regressions and CPU/CUDA materialized and
streaming soaks. The source archive matched all eight changed files and its
extracted streaming CLI scenario passed. This evidence applies to the identical
integration source tree above.

The 2026-10-02 closeout audit additionally passed **384 related tests**, with
no failures or skips, in 736.37 seconds. Fifteen were real CUDA checks across
`test_local_query_gpu_cuda.py` and `test_local_serving_soak_cuda.py`, including
both materialized and streaming serving. The other modules cover preparation
rollback, model registry/resources, native model reuse, request cancellation
and CPU soak/watchdog behavior. The original startup replay is recorded below.
The closeout repeated the complete installed release gate: **3,471 passed**
(3,395 non-Ray, 74 shared-Ray and two isolated cluster-owner tests), with the
same eight optional dependency skips. All three pytest processes exited
successfully; this is additional validation of the integration runtime sources.

The longer #928 streaming measurements were taken at its earlier
`de0851b073` candidate: one runtime per device mode, 20 rounds, 1,601 load requests
plus pressure/fault probes, four clients. CPU worker time was 268.83 seconds and
CUDA worker time was 593.24 seconds; both supervised processes exited
successfully. Healthy work preserved model identity. The 61 CPU and 81 CUDA
initializations match one cold worker and the workload's expected replacements.
Final resident, request/task, data/result and physical transport ownership
returned to zero. CUDA used RTX 2080 Ti, PyTorch 2.7.0+cu126 and CUDA 12.6.
The final-head two-round regressions above validate the later cleanup fix;
the 20-round timings are not measurements of the final head.

Supported scope is configured, auto-commit, read-only local-fast SQL/Relation
execution with explicitly registered CPU or fixed-device GPU models, native
materialized results and opt-in managed streams. Stateful reuse remains
explicit and session-owned. Input callbacks reject connection reentry;
unsupported asynchronous input paths are rejected as documented in the
[runtime guide](LOCAL_MODEL_RUNTIME.md#shared-runtime-for-ordinary-local-fast-queries).
Ray keeps its query/generation authorization and object-store/liveness policy;
shared contract tests do not enable a persistent Ray model registry.

Resident CPU/GPU/declared heap limits are logical admission estimates. UDF
shared-memory admission accounts exact IPC allocations and task envelopes;
result delivery has a separate IPC-buffer ledger, including caller-held
zero-copy views. DuckDB operator memory, worker/model copies, decoded batches,
serialization overlap, sockets and network buffers remain outside those byte
ledgers. These controls do not enforce RSS or physical VRAM. Local UDF shared
memory has no spill mechanism; native sort/aggregation/join memory and spill
remain DuckDB-owned. The tested schedules establish no latency SLO.
HTTP/RPC endpoints, retrieval-index implementations, automatic model eviction,
replica scaling and cross-request dynamic batching require separate scope.

### Historical model-entry timeout: investigation status

The 2026-09-28 investigation started from integration commit `a63231b3a1`
after #908. The preserved #887 affected-test log shows one failure with
`unit_reservation_ratio=0.5`: the first query did not create its model-entry
marker within 15 seconds, before the test requested cancellation. That run
reported 345 passed, one failed and 13 deselected. It did not capture stacks or
resource state for that event, so it cannot distinguish worker initialization,
upstream execution, output admission, or native scheduling.

The pending-query retirement defect and notification-loss window fixed in
#892 have their own controlled regressions. They remain separate findings;
neither those fixes nor later passing runs establish the original event's
cause. The preserved later GDB replay passed and is not a trace of that
first-query failure.

During the 2026-09-28 investigation, on the installed Python 3.12.14 package
matching that integration tree, both original parameterizations passed,
followed by 40 repetitions alternating
`None` and `0.5` in one process. The repeated workload completed in 145.98
seconds under the existing process-group watchdog's 240-second limit, with
the original per-query deadlines unchanged. This is a bounded negative
reproduction result, not a timeout fix or a latency guarantee.

The same native test now retains per-query progress milestones with its
pre-cleanup diagnostics, as described in the
[runtime guide](LOCAL_MODEL_RUNTIME.md#backpressure-acceptance-gate).
Controlled stops before preparation returns and before an upstream output
grant confirm that different progress states and still-held resources survive
in the artifacts. Those controls validate evidence capture; they do not
reproduce the historical scheduling failure.

The 2026-10-02 closeout repeated the unchanged test at integration commit
`6a4b1bbe7d` with `delay_native_start=False`: 40 cases alternating `None` and
`0.5` passed in one process, in 154.83 seconds of worker time, with successful
process exit under the 240-second watchdog. The original 15-second model-entry
and reuse deadlines remain unchanged. The original failure log has SHA-256
`d7d52f407b63cdf04dd07a4f38f58d18ea5f444d232f8bd32af3db1725c8a6c3`.

The explicit disposition is **stage acceptance with an unresolved historical
incident**, retained in [#929](https://github.com/AstroVela/vane/issues/929).
That issue records the original failure, negative reproductions and the evidence
required on recurrence. Review it at the final integration into `main` or on
the next matching timeout, whichever comes first. #841's implementation
completion does not confirm a root cause or claim this incident is fixed.

### Commands

```bash
scripts/run_installed_pytest.sh \
  tests/fast/test_local_serving_acceptance.py \
  tests/fast/test_local_query_models.py \
  tests/fast/test_local_query_results.py \
  tests/fast/test_result_delivery.py \
  tests/fast/test_local_serving_soak.py
scripts/run_installed_pytest.sh tests/fast/test_udf_worker_metrics.py
scripts/run_release_tests.sh
```

The release gate includes the ordinary-publication acceptance case, with four
requests per load phase and the CLI's fault scenarios. The affected/fast suite
also forces delivery expiry before publication to avoid assuming that the
publisher always wins that race. Driver tests reject unsafe retries; the
model/result suites cover shared ownership and cleanup-failure contracts.
Both capacity diagnostics are replaced with opaque text during the native
acceptance tests to verify that retry decisions use structured fields.
The release gate adds a two-round native soak and its watchdog fault tests;
longer runs use the standalone command above.
Keep release Ray shards separate as required by the
[development workflow](DEVELOPMENT.md#python-tests).
