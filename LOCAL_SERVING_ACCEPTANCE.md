# CPU serving acceptance

This scenario validates the public local-fast SQL and Relation lifecycle in
[LOCAL_MODEL_RUNTIME.md](LOCAL_MODEL_RUNTIME.md), continuing
[#843](https://github.com/AstroVela/vane/issues/843) under
[#838](https://github.com/AstroVela/vane/issues/838). It combines model reuse,
request/task/data admission, cancellation, and managed result delivery in one
session. Separate sessions validate short queue and zero execution deadlines,
because public runtime configuration is fixed for the session. The scenario
continues both issues without introducing an HTTP/RPC endpoint.

The [runnable example](scripts/validate_local_serving.py) uses
`connection.configure_local_runtime()`, `runtime.register_model()`,
`model.prewarm()`, and `vane.attach_function()` for setup. Clients execute
parameterized SQL through `cursor.execute_result()` or build projections with
the registered model and call `relation.execute_result()`. They consume and
close `ManagedResult` handles. Independent control threads cancel queries with
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
Worker-exit observations are scenario evidence; generic runtime execution
failures do not distinguish worker loss from user-code or preparation errors.
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
its own references. Native streaming, transport adapters, and local GPU support
remain subsequent work.

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
The short scenario's separate queue/execution-deadline sessions remain separate
coverage, since session configuration cannot change during the soak.

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

Snapshots do not create pools, obtain task grants or call active admission
callbacks. They are observations of separate components, not an atomic global
state. A snapshot can be stale or unavailable if its locks are blocked; inspect
the sampling timestamps and thread stacks. The sampler is a daemon with bounded
shutdown waiting, and it cannot extend the supervisor's deadline. On expiry the
supervisor requests stacks, then terminates the isolated child process group,
including its inherited actor processes. This forced teardown is failure
containment, not evidence that runtime cleanup succeeded.

Per-request samples and fixture markers are discarded after each round. Only
eight round summaries, scalar totals and the most recent resource snapshots
remain; no unbounded sample or exception history is kept. Quantiles describe
individual retained rounds, not the whole run. Passing a soak demonstrates the
tested schedule and budget configuration; it does not establish a latency SLO
or prove the root cause of a historical timeout.

### CPU stage acceptance status

The following describes the development branch after #905. It does not mark
the parent tracker complete or imply these changes are available on `main`.

| #843 criterion | Evidence and remaining work |
| --- | --- |
| Active/queued request bounds and queue expiry | Public ingress and separate queue-deadline scenarios |
| Deadline/cancellation propagation | Native request regressions, cancellation, execution expiry and result-delivery expiry |
| Bounded slow consumers | Result-slot pressure and retained Arrow/NumPy views remain byte-charged; caller-held views survive shutdown |
| Mixed analysis/serving resource policy | Shared limits and fair admission; mixed-load measurements document FIFO head-of-line delay, without a latency bound |
| Request cleanup preserves shared models | Healthy worker identity and resident reservations survive sequential/concurrent calls; fault recovery is explicit |
| Public configuration and supported capabilities | SQL/Relation runtime, registered CPU models and managed results; see `LOCAL_MODEL_RUNTIME.md` |
| Runtime metrics | Queue/execution/cleanup/delivery totals, active owners, bytes and cancellation exist. **Dedicated runtime worker-failure classification/counting remains open**; injected-fault counts in the driver do not satisfy it |
| Reproducible multimodal-UDF scenario | Deterministic CPU text/RGB fixture reports cold/warm counts and latency; sustained runs add repeated recovery and bounded diagnostics |

For [#841](https://github.com/AstroVela/vane/issues/841), acceptance PR #892
merged and its [CI run](https://github.com/AstroVela/vane/actions/runs/35818116064)
passed native build/tests on Python 3.10–3.14, the shared/isolated Ray shards,
and Required CI. The previously observed mixed-pipeline model-start timeout
still has no confirmed root cause. Keep that item open with its original
evidence; do not attribute it to the notification-loss fix from repeated passing
runs alone. This public serving soak complements the existing small-budget
mixed-pipeline regression, rather than reproducing that historical workload.

#841 remains open for that timeout disposition; #843 remains open for runtime
worker-failure metrics and final integration acceptance. Local GPU admission
continues separately under #842.

### Commands

```bash
scripts/run_installed_pytest.sh \
  tests/fast/test_local_serving_acceptance.py \
  tests/fast/test_local_query_models.py \
  tests/fast/test_local_query_results.py \
  tests/fast/test_result_delivery.py \
  tests/fast/test_local_serving_soak.py
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
