# Advisory prewriteback: final fsync and sequential-read attribution

The remaining multi-second final fsyncs occur in **the foreground payload
`fdatasync`**, with a background payload sync concurrently active on the same
file. They are not publication/checkpoint mutex waits or large WAL syncs in
this diagnostic window. The control also has a **7.212-second payload sync**
without a concurrent prewriteback worker, so these traces do not establish
concurrent syncs as the sole cause or identify a kernel/device root cause.

The original [plain prewriteback trial](../payload_prewriteback/README.md)
still fails **4 of 22 acceptance checks**. All results here are instrumented
attribution, not adoption throughput. Production format 2, its native module
and default SQLite SDK remain unchanged. Both diagnostic components use the
same recorder and format 1005, with exact reversals to the frozen extent control
and advisory-prewriteback candidate.

## Final FULL barriers

These are the three candidate final barriers exceeding one second, all during
the first-256 phase. Every actual sync succeeds. Times on different threads
overlap and must not be added as independent costs.

| State / repetition | Application fsync, s | Own payload fdatasync, s | Own WAL fsync, ms | Background payload fdatasync, s | Caller CPU, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh / 1 | 5.694264 | 5.690852 | 2.082 | 6.041450 | 69.390 |
| Fresh / 3 | 3.539912 | 3.536269 | 2.216 | 3.806868 | 54.775 |
| Post-GC / 3 | 4.727887 | 4.722933 | 3.318 | 5.316563 | 66.968 |

Payload sync occupies **99.90–99.94%** of each application interval. Their own
CLEAN-fence syncs take 0.834–0.862 ms. Database, checkpoint, publication and
payload-guard acquisition spans together remain below 0.020 ms per barrier;
application time outside the handler stays below 0.102 ms. There is no measured
multi-second queue gap or application mutex wait in these examples.

The full 5.691-second foreground payload interval overlaps the worker's
6.041-second sync. The other two examples similarly overlap a long worker
sync. Worker roots have **zero checkpoint/publication-held spans**: the data
worker is operating outside those mutexes as intended. The barrier nevertheless
executes its required payload sync; advisory progress never replaces it.
The control's longest final barrier takes 7.217 seconds, including 7.212 seconds
payload sync, 3.384 ms WAL sync and 1.564 ms fence sync, without another thread's
concurrent sync. Data synchronization is the demonstrated bottleneck; concurrent
whole-file sync activity is an observed property, not proof of its kernel cause.

All eight first-256 control phases have **one payload fdatasync, two WAL fsyncs
and two fence fdatasyncs**. The candidate retains those foreground counts and
adds **two or three background payload fdatasyncs** per phase. It adds no worker
WAL/fence synchronization in these phases. This differs from the rejected
[full-publication trigger](../payload_waits/README.md), which also performed
extra WAL/fence work while holding publication/checkpoint mutexes.

## Timed sequential reads

Each 64 MiB warm read has **129 native requests**, including one zero-byte EOF
read, **22,081 SQLite VM steps**, and **1,286 sqlite3_step calls**. These counts
are identical in all 16 measurements. The pread count is 64 in fresh state and
192 after GC for both variants. No timed read overlaps a prewriteback/checkpoint
worker root or any fsync/fdatasync. Read checkpoint/publication wait counters
are zero. An active background sync or extra SQL work does not explain these
particular read intervals.

| Four-pair read medians | Fresh control | Fresh candidate | Post-GC control | Post-GC candidate |
| --- | ---: | ---: | ---: | ---: |
| Application interval, ms | 73.246 | 74.291 | 77.289 | 74.689 |
| Native handler wall, ms | 62.389 | 63.952 | 66.145 | 63.821 |
| Native handler CPU, ms | 62.133 | 63.670 | 65.895 | 63.575 |
| Outside handler, ms | 10.567 | 10.560 | 11.144 | 10.868 |
| pread inclusive wall, ms | 19.358 | 18.961 | 22.455 | 20.740 |

Medians of components are separate statistics and do not sum to a median total.
The fresh median interval grows **1.4%**, while the post-GC median falls **3.4%**.
Fresh aggregate read throughput still falls **951.99 to 855.49 MiB/s (-10.1%)**;
post-GC rises 829.96 to 853.67 MiB/s (+2.9%). The fresh control includes a
48.108 ms read (41.898 ms handler CPU); its other three reads take 73.116–74.311
ms. The candidate's four fresh intervals span 69.343–81.317 ms. All samples
remain included. Native CPU varies with these durations; outside-handler time
includes application work, transport, copies and scheduling together.

Thus the original fresh median and post-GC aggregate read gate failures do not
repeat uniformly, while fresh aggregate loss remains. The read cause is
**unresolved**, not fixed or dismissed as noise. A narrower repeated native/FUSE
read comparison with frozen payload layout can separate CPU/copy-path variation
from transport effects before changing read logic.

## Next bounded optimization

Evaluate **range-based asynchronous payload writeback**, beginning only already
appended data and retaining the original final payload fdatasync, WAL and
OPEN/CLEAN sequence. It would test whether starting data I/O without a background
whole-file durability sync changes the remaining tail. This is a hypothesis,
not an implementation or an established improvement. Baseline long syncs show
why simply removing duplicate foreground synchronization is not justified.
Keep the original fresh/post-GC, complete-write, p99, read and lifecycle gates;
do not retune the 64 MiB threshold using these samples.

## Evidence and reproduction

Sixteen ext4 mounts run four AB/BA pairs per state, with the same complete
first-256/subsequent-64, warm/random read, overwrite, small-file and namespace
workloads. The application buffer offset, request sizes, per-variant GC
modules, SQLite fsync SDK and durability stay fixed. No cache eviction, CPU/device
tuning or load-filtered samples are used. Owned builds/tests finish before
measurement; all analysis begins after the mounts and monitors stop.

The recorder adds a prewriteback-worker root, payload-lease acquisition span
and ReadBytes span. Sequential reads gain exact monotonic boundaries; their
operations and validation remain unchanged. Release and ASan/UBSan integration
tests pass, including a real read with VM/pread counters. All **16 mounts and
32 large-file content checks pass**. **221,904 native roots and 201,787 events**
pass exclusive wall/CPU accounting. Algorithm correctness suites are reused by
frozen source/binary identity and exact instrumentation reversal. No broad
base/Ray suite is rerun.

An analysis preflight incorrectly equated requested FUSE bytes with returned
bytes. The final EOF request asks for more data but returns zero. The frozen
analyzer is retained unchanged; [analyze_reads.py](analyze_reads.py) applies an
exactly reversible post-measurement correction, clips logical coverage to the
validated 64 MiB size, and preserves EOF timing, CPU and VM work. The
[preflight explanation](validation/analysis-preflight.json) and
[adapter proof](validation/analysis-adapter.json) retain the correction.
No workload or measurement is rerun or modified.

```bash
here=vane_fs/benchmarks/staged_payload/prewriteback_waits
out="$PWD/vane_fs/build/prewriteback-waits-repro"
python3 "$here/prepare.py" "$out" \
  --comparison "$PWD/vane_fs/build/payload-prewriteback-20261018/fuse"
python3 "$here/run.py" "$out"
python3 "$here/analyze_reads.py" "$out"
# After recording final source checks:
python3 "$here/publish.py" "$out"
```

The preceding frozen components and SDK must be present. The
[findings](findings.json) retain every final fsync and timed read, including
handler IDs and concurrent sync intervals. Indexed traces retain every root
and event. [The audit](audit.json) verifies source/runner reversals, frozen
inputs, stopped processes and **18 cleaned data directories** (16 mounts and
two recorder tests). Test logs, configurations, host observations, hashes and
cleanup manifests remain in files smaller than 500 KiB.
