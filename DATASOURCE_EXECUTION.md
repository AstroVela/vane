# Python source execution

Python DataSource tasks run on native pipeline threads. A source must return
control to the scheduler when it cannot produce data because another task owns
a resource. Blocking that thread on a semaphore can prevent the resource owner
from resuming after downstream backpressure.

## Batch and readiness contract

`datasource_scan` uses the `TableFunction::poll_function` callback. A poll returns
one of three results:

- `HAVE_MORE_OUTPUT`: the output chunk contains rows.
- `BLOCKED`: the output is empty and the source has registered a readiness callback
  using the supplied `InterruptState`.
- `FINISHED`: the source has exhausted its tasks and the output is empty.

The polling callback is mutually exclusive with the ordinary table-function and
in/out callbacks. The scan retains its local stream and Arrow conversion state
while blocked. A `DataSourceStream::Poll` result is null for pending readiness;
an Arrow array without a release callback denotes EOF.

The Python bridge stages one batch through `_DataSourceIterator` before Arrow
pulls it. An internal context-aware source may yield `_DataSourceWait` to suspend
instead of producing a batch. Its `subscribe` method atomically reports readiness
or registers the current callback; it must not wait for that readiness itself.
Wait tokens never enter Arrow. Ordinary `DataSourceTask.execute` generators keep
their RecordBatch contract.

Readiness may arrive before the native task has returned `BLOCKED`. Native
interrupt epochs preserve that wakeup and reject callbacks from older executions.
Callbacks hold weak task/signal references and release the GIL before entering
the scheduler. A cancelled task can therefore be destroyed while a callback is
in flight without retaining its query or resuming a later execution.

The table scan also registers blocked polls under their local-state identity.
Pipeline completion (for example, `LIMIT`) wakes those tasks even if external
admission never becomes ready. A registration is removed on the next poll, so
this bookkeeping is bounded by waiting tasks rather than total batches/files.
The executor checks query cancellation even when every task is blocked, so
teardown does not depend on memory recovery or another decoder releasing a slot.

## Decoder admission

Each process owns a FIFO decoder admission queue. A coordinator checks host
memory outside native pipeline threads and admits requests up to
`VANE_MAX_CONCURRENT_DECODES`. It retains the existing available-memory override
and high/low memory watermarks. The coordinator only grants permits: it neither
opens decoders nor buffers frames, and it exits when the pending queue is empty.
Forked children initialize their own queue and resource count.

A video task yields its permit request, resumes when admitted, checks query
cancellation again, and opens its decoder. The permit remains owned until that
decoder closes, including while downstream backpressure suspends the generator.
Increasing scheduler threads or releasing permits between batches is unnecessary.

Closing a source cancels its pending wait and closes its generator. Closing a
permit is idempotent: a pending request leaves the queue, while an admitted
request returns exactly one slot. The native stream invalidates its execution
context before teardown. Decoder and memory-probe failures return through the
source error path; governed video exceptions retain their public categories.

## Regression checks

Use an installed build, following [Development](DEVELOPMENT.md):

```bash
scripts/run_installed_pytest.sh tests/fast/test_datasource_readiness.py \
  tests/fast/test_video_decoder_admission.py tests/fast/test_video_reader.py \
  tests/fast/test_video_optional_deps.py -m 'not real_ray'
```

The isolated CPU regression runs four queries sharing two native execution
threads and one decoder permit. A subprocess UDF applies bounded streaming
backpressure. Every query must return all 1,536 frame indices with their expected
multiplicities, without changing the resource limits. It has a deadline and
cancels its own queries on failure. Separate tests cover memory recovery, FIFO
admission, cancellation, and callbacks arriving before the scan returns `BLOCKED`.
