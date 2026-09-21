# CPU serving acceptance

This scenario validates the internal local-fast lifecycle described in
[LOCAL_MODEL_RUNTIME.md](LOCAL_MODEL_RUNTIME.md), continuing
[#843](https://github.com/AstroVela/vane/issues/843) under
[#838](https://github.com/AstroVela/vane/issues/838). It combines model reuse,
request/task/data admission, cancellation, execution deadlines, and managed
result delivery in one session. It does not close either issue or introduce a
public model-registration API or HTTP/RPC endpoint.

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

One explicitly registered subprocess actor is resident throughout the run.
Every query rebuilds its physical plan on an independent cursor from the same
session. The runtime has two active request slots, two queued request slots,
one running task slot, eight queued task slots, and a declared resident budget
of one CPU and 16 MiB heap. The shared-memory data budget is 2 MiB, with 64 KiB
input and 128 KiB output envelopes per task. Managed delivery has two result
slots and 64 KiB of retained IPC buffers. Heap declarations are logical
reservations, not operating-system limits.

The phases run in this order:

1. A cold request initializes one worker. Explicit prewarm and subsequent
   sequential requests add no initializations and use the same worker PID.
2. Concurrent clients mix one-row serving queries with 32-row analysis queries.
   All complete with correct features and the same resident model. Request
   admission is FIFO and task arbitration uses the existing shared policy;
   there is no preemption or short-query priority. One long running UDF can
   delay short requests. This scenario reports that cost without claiming a
   latency bound or exercising every possible scheduling interleaving.
   The driver retries a refused result slot only while the original ticket is
   still ready, for at most 30 seconds, and reports these refusals. It never
   retries an execution that has already been claimed. This consumer policy
   uses the existing explicit-refusal API, without adding a runtime scheduler.
3. Full ingress rejects excess work. FIFO promotion, queued cancellation and
   queue expiry leave no model, task, or result owners for unexecuted requests.
4. Two unconsumed results occupy both delivery slots. A ready request refuses
   three times without running user code, then executes exactly once when a
   result is consumed. A retained Arrow view keeps its bytes charged after
   final handoff, so a second large result fails byte admission. That UDF has
   already run and its ticket cannot replay. Releasing the view restores
   capacity for a new request.
5. Delivery cancellation and abandoned-result expiry release pending results.
   Expiry can also be reported while publishing the result, before the caller
   receives its handle; both paths must count one timeout and finish cleanup.
   Cancellation interrupts a gated, running subprocess UDF. A zero execution
   deadline prevents another UDF from starting. Subsequent requests succeed.
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

The script deliberately kills only its own fixture worker via `os._exit(23)`
inside that worker's designated failure request. Constructor/call markers live
in the temporary fixture directory and are not copied into the report.

## Report and measurement boundaries

The JSON report includes configuration, environment versions, initialization
counts, one observed injected worker-exit failure, phase distributions, mixed
throughput, and resource snapshots at pressure and recovery checkpoints.
Worker-exit observations are scenario evidence; generic runtime execution
failures do not distinguish worker loss from user-code or preparation errors.
No per-request exception, plan, Arrow view, or unbounded sample history is
stored in runtime metrics. The finite benchmark collects scalar samples in the
driver to compute its report.

For each cold, warm, mixed-short, and mixed-analysis group, the report gives
count, mean, maximum, and nearest-rank P95/P99 in seconds. An empty group has
count zero and null statistics. Small samples are descriptive; use more
requests and repeat the command for performance work. Correctness checks do
not assert throughput or latency thresholds.

| Measurement | Interval |
| --- | --- |
| Request latency | Ticket creation through result consumption and request context cleanup; excludes plan construction |
| Queue wait | Ticket creation through promotion to ready; excludes time held ready before execution claim |
| Execution | Successful claim through preparation, native execution, and cancellation callback completion; excludes subsequent query cleanup |
| Cleanup | End of execution through confirmed return of the request slot; includes time awaiting explicit cleanup retries |
| Delivery | Result ready through confirmed result-slot retirement; includes pending-result cleanup, excludes subsequent external view lifetime |
| Mixed throughput | Completed mixed requests divided by phase wall time, including cursor/plan construction |

The execution counters update once when execution ends, even if cleanup still
owns the request slot. `failed_executions` counts preparation/native errors
after a claim, excluding accepted cancellation, execution expiry, cleanup-only
errors, result-slot refusal, and post-execution result encoding/byte refusal.
`completed_requests` keeps its existing meaning: non-cancelled request slots
returned after cleanup, including failed executions. Delivery totals include
only results that became ready and subsequently retired; failed preparations
have no delivery sample. Unstarted or unfinished per-handle intervals are null.

Managed results are materialized, with one IPC copy per native partition.
Native collection, DuckDB memory, temporary encoding overlap, external
serialization, and network sends are outside the retained-result budget.
Exported views stay byte-charged after slot retirement, but cannot be forcibly
freed while a caller retains them. Delivery expiry ends at handoff to the
caller. Slow consumers here mean delayed iterator consumption and retained
Arrow views; a real transport must own sends, disconnect cancellation, and
its own references. Native streaming, transport adapters, local GPU support,
and public API consolidation remain subsequent work.

## Regression gate

```bash
scripts/run_installed_pytest.sh \
  tests/fast/test_local_serving_acceptance.py \
  tests/fast/test_request_admission.py \
  tests/fast/test_udf_local_request.py \
  tests/fast/test_result_delivery.py
scripts/run_release_tests.sh
```

The native acceptance test uses four requests per load phase and the same
fault scenarios as the CLI. Deterministic clock and cleanup-fault tests verify
the metric boundaries, unclaimed requests, and exactly-once accounting across
cleanup retries. The native scenario also forces delivery expiry before result
publication to avoid assuming that the publisher always wins that race.
Keep release Ray shards separate as required by the
[development workflow](DEVELOPMENT.md#python-tests).
