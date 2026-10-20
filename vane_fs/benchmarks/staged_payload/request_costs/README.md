# Short-request attribution after budgeted writeback

The nine failed checks in the [budgeted-writeback comparison](../writeback_budget/README.md)
remain failures. This investigation adds C++ request counters and separate CPU
diagnostics; it makes no production algorithm or writeback-parameter change.
The 4 MiB submission limit, 16 ms interval, publication pause, SQLite SDK and
FULL durability barriers retain their frozen identities. Diagnostic rates do
not replace the earlier 48-mount comparison or change its adoption gates.

## What the measurements establish

Short-request regressions cannot be attributed to the optional hint call or a
large new mutex wait. In 32 complete instrumented mounts, the random-write,
rename and rmdir phases have no overlapping hints. Fresh 512-request random
writes execute exactly 224,448 SQLite VM steps and 10,230 statement steps in
both variants; the corresponding foreground pwrite count is 10,715. GC changes
the file layout, so its counts are reported separately. Random-write statement
and syscall counts are stable between variants within each state.

The ext4 fresh random-write median handler CPU time is 120.532 ms in the control
and 137.086 ms in the budget candidate. Median cumulative database-mutex waits
are only 51.453 and 57.934 microseconds per 512 writes. Checking the candidate's
background writeback error costs 45.108 microseconds in total. Those observed
costs cannot account for the millisecond-scale regression. SQLite step and
commit wall times include nested I/O; adding all stage counters would count the
same interval multiple times.

The host uses `intel_cpufreq` with `schedutil`, with a permitted frequency range
of 1.2–3.0 GHz. A separate 16-mount window using the unchanged native binaries
samples the owned daemon's CPU residency, scheduler runtime and frequency.
CPU frequency differs even when the machine has no other application workload.
Slow random-write samples run mostly near 1.3–1.4 GHz; faster samples can run
near 2.0 GHz. Runqueue wait is small. This is evidence of the benchmark's own
CPU-state variation, not evidence of unrelated workload or a defective SSD.

The after-window observer check records `sched_schedstats=0`, while this kernel
is compiled with `CONFIG_SCHED_INFO=y`. These are separate controls: the
[Linux 6.11 per-task interface](https://github.com/torvalds/linux/blob/v6.11/fs/proc/base.c#L450)
reports execution runtime and `sched_info.run_delay`, and
[`sched_info_on()`](https://github.com/torvalds/linux/blob/v6.11/include/linux/sched/stat.h#L22)
depends on the compiled `CONFIG_SCHED_INFO`. Its
[per-task updates](https://github.com/torvalds/linux/blob/v6.11/kernel/sched/stats.h#L187)
continue independently of the separate runtime `sched_schedstats` toggle.
The small observed counter deltas therefore have a valid interpretation; they
are not disabled scheduler-statistics zeros. CPU runtime and sampled frequency
remain estimates over complete sampling intervals.

To test that explanation, another 16-mount window places the caller, FUSE main
thread, checkpoint worker and hint worker on four distinct physical cores.
Only owned processes/threads get an affinity setting. Before each short phase,
both conditions wait two seconds: the idle condition sleeps; the warm condition
runs a separate CPU-only SHA-256 process on the FUSE core. The warm process
exits before the request timer starts. Metadata, transaction logic and native
binaries are unchanged. Frequency policy, kernel/device settings and global
caches are unchanged.

Results below pair idle and warm preparation **within the same binary, state
and repetition**. They are diagnostic CPU effects, not optimization results:

| Phase | Warm preparation wins | Median paired request-stage speed ratio |
|---|---:|---:|
| 64 MiB warm read | 8/8 | 1.833 |
| 4 KiB random read | 5/8 | 1.064 |
| 512 × 4 KiB random write | 8/8 | 1.461 |
| 128 mkdir operations | 7/8 | 1.400 |
| 128 rename operations | 8/8 | 1.335 |
| 128 rmdir operations | 8/8 | 1.602 |

The pooled random-write request-stage medians are 238.720 ms after idle
preparation and 131.668 ms after warm preparation. The corresponding median
runtime-weighted sampled frequencies are 1.219 and 2.099 GHz. Pooled medians and
the median of paired ratios are different statistics and are not interchangeable.
Sysfs frequency and scheduler accounting are sampled estimates, not hardware
cycle counters. Warm preparation does not hold frequency constant throughout
a phase. Cross-variant residual differences remain; these results do not prove
that CPU frequency explains all nine failures.

There is also a concrete maintenance cost to investigate. The WAL hook wakes
once its **total** committed frame count exceeds 16 MiB, even if most frames
have already been backfilled. In the request diagnosis, 128 ext4 rename
operations overlap 15–32 checkpoint-worker passes per mount. Tmpfs 512-write
phases overlap 5–81 passes. The CLEAN case preserves the earlier immediate
checkpoint policy; it can create short checkpoint work repeatedly. These counts
identify an optimization candidate, but do not establish that checkpoint
activity caused the candidate's entire CPU regression.

Long tails remain visible. In the controlled window, fresh `updated-warm`
repetition 1 takes 3.531 s for the complete 64 MiB write, including a 1.397 s
application write request and a 1.141 s final fsync. This separate affinity/sampling
configuration is diagnostic and cannot be compared as plain throughput. The
sample is retained with all other measurements. Neither the budget nor this
investigation establishes a worst-case latency bound.

The next C++ experiment should reduce redundant CLEAN checkpoint wakeups based
on uncopied WAL progress. It must preserve FULL barriers, failed-close/error
retry, pinned-reader backpressure and 64 MiB admission, and pass the existing
frozen workload gates. CPU state needs to be recorded alongside that experiment;
frequency-adjusted numbers must not replace measured results.

## Artifacts and validation

- [Request costs on tmpfs](request-summary-tmpfs.json) and
  [request costs on ext4](request-summary-ext4.json) retain per-mount stage
  counts, inclusive times, VM counts, worker overlaps and slowest handlers.
- [Owned-CPU sampling](cpu-summary.json) records the original uninstrumented
  binaries' sampled runtime, frequency weights, migrations and runqueue waits.
- [Controlled idle/warm diagnostics](controlled-summary.json) retain every
  paired result and preparation record, including warmup process identities.
- The `requests/`, `cpu/` and `controlled/` directories retain frozen plans,
  source/binary identities, complete workload arrays and window records.
  Arrays split by medium/state/variant reproduce the original parsed records
  exactly, without rounding or dropping slow samples.
- [The audit](audit.json) verifies all 64 mounts' content checks, format 1004,
  stopped processes, absent mounts and 64 removed owned data directories.
  Production source, installed module and SDK hashes match the prior report.
  Instrumented native builds finish before every measurement window. No owned
  build or correctness test runs during a measurement window.
- [The manifest](manifest.json) verifies every published artifact and indexes
  local full request vectors, CPU samples, host logs, generated native sources,
  libraries and binaries. Large database/payload fixtures are deleted after
  validation. Each published JSON is below 500 KiB.

The prior component/Python/sanitizer results are inherited for the unchanged
native implementations; this turn does not claim another run of those suites.
Related validation here executes complete workloads, verifies content and
source identities, checks counter/request counts and compares split records
losslessly. Formatting, Ruff and the repository copyleft check also pass.

## Reproduce

Use the same CPython 3.12 environment and frozen control/candidate from the
budgeted-writeback report. All programs run outside the production component;
diagnostic libraries are built in their own persistent CMake directories.

```bash
REPO=/home/kaka/astrovela_vane-fs
BENCH="$REPO/vane_fs/benchmarks/staged_payload/request_costs"
PARENT="$REPO/vane_fs/build/writeback-budget-20261011T165013Z"
REQUEST_OUT="$REPO/vane_fs/build/request-costs-new"
CPU_OUT="$REPO/vane_fs/build/request-cpu-new"
CONTROLLED_OUT="$REPO/vane_fs/build/request-controlled-new"

"$REPO/.venv/bin/python" "$BENCH/prepare.py" "$REQUEST_OUT" \
  --parent "$PARENT" \
  --sqlite-prefix "$REPO/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$REPO/.cache/fuse-sdk/root/usr"
"$REPO/.venv/bin/python" "$BENCH/run.py" "$REQUEST_OUT" --medium tmpfs
"$REPO/.venv/bin/python" "$BENCH/run.py" "$REQUEST_OUT" --medium ext4
"$REPO/.venv/bin/python" "$BENCH/analyze.py" "$REQUEST_OUT"
"$REPO/.venv/bin/python" "$BENCH/run_cpu.py" "$CPU_OUT" --parent "$PARENT"
"$REPO/.venv/bin/python" "$BENCH/analyze_cpu.py" "$CPU_OUT"
"$REPO/.venv/bin/python" "$BENCH/run_controlled.py" "$CONTROLLED_OUT" \
  --parent "$CPU_OUT"
"$REPO/.venv/bin/python" "$BENCH/analyze_cpu.py" "$CONTROLLED_OUT"
"$REPO/.venv/bin/python" "$BENCH/publish.py" \
  --requests "$REQUEST_OUT" --cpu "$CPU_OUT" --controlled "$CONTROLLED_OUT"
```

Each measurement command owns and cleans its mount/data paths and preserves
its complete logs. `publish.py` copies generated evidence into this report
directory and refuses a missing validation, changed frozen identity, uncleared
owned path or oversized published JSON.
