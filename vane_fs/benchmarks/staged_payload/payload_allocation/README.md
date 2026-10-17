# Exact-range payload allocation

**Keep ordinary append allocation.** Explicit `fallocate(KEEP_SIZE)` before
256 KiB or 1 MiB ranges does not pass the complete ordinary-file screen.
One of eight candidate/workload groups passes; 20 of 32 frozen checks fail.
The 1 MiB candidate improves a fresh 64 MiB file, but neither allocation
policy consistently improves 256 MiB files or the simulated post-GC state.
No storage-component patch is adopted from these results.

This follows the [cold WAL reservation experiment](../wal_preallocation/README.md),
whose separate native diagnostics locate 142–184 ms in the first large file's
payload barrier versus 23–25 ms in WAL synchronization. The question here is
whether allocating payload blocks earlier reduces that complete cost. These
ordinary-file rates are **not VaneFS or FUSE throughput measurements** and
must not be substituted into the existing filesystem comparison.

## Protocol

The measured window is October 14, 2026, 06:22:04–06:26:13 UTC. Seventy-two
plain processes compare all six permutations of three policies for 64 MiB
and 256 MiB files, fresh and after simulated reclamation. Every process has
an independently created owned directory on `/dev/sda2`, ext4, Linux
`6.11.0-24-generic`. The frozen plan retains available space, mount options,
compiler and exact source/binary/default-module identities.

- `delayed`: append normally and leave allocation to the filesystem.
- `reserve256k`: reserve exactly the next 256 KiB before each append.
- `reserve1m`: reserve exactly the next 1 MiB before its four appends.

All policies issue the same 256 KiB buffered `pwrite` appends, with a file-size
lookup before each append, then the same final whole-file `fdatasync`. The
buffer starts 48 bytes into a page. Reservations never extend beyond the
known workload's final EOF. The diagnostic independently verifies that
`KEEP_SIZE` leaves the logical EOF unchanged before writing. Its
[documented semantics](https://man7.org/linux/man-pages/man2/fallocate.2.html)
permit reserving blocks without changing file size; this is essential when
physical offsets are allocated from the current logical EOF.

Fresh-file timing includes open, a 4 KiB header write, its data barrier,
directory fsync, all allocation and writes, the final data barrier and close.
The post-GC fixture first persists 64 MiB, punches a middle 16 MiB hole,
truncates the last 16 MiB, and synchronizes. It retains 32 MiB of live guard
data; fixture preparation is separately reported and outside the append
timer. This fixture exercises ordinary-file allocation after hole punching
and truncation; it does not exercise VaneFS snapshots or its GC protocol.

Complete timing is primary, so moving work from the barrier to allocation
does not qualify as an improvement. Before measurement, the screen requires
at least 10% gains in both median and aggregate throughput, at least five of
six paired wins, and total-operation p99 no greater than 1.2 times control.
Nearest-rank p99 for six complete operations is their maximum. These are
screening gates, not significance estimates or filesystem acceptance gates.
Every sample, including slow controls and candidates, remains included.

Twenty-four later diagnostic processes run two balanced orders per group
and retain every allocation/append/sync interval. Their rates do not replace
plain results. Build, probe validation, plain measurement and diagnostic
measurement do not overlap. No cache eviction, device/CPU tuning, or
load-based sample exclusion occurs; read-only host counters remain in each
window. This is a local filesystem observation, not an isolated-device or
general filesystem guarantee.

## Complete plain results

Each cell reports median / total-work-over-total-time throughput in MiB/s.

| State / bytes | Ordinary append | Reserve 256 KiB | Reserve 1 MiB |
| --- | ---: | ---: | ---: |
| Fresh / 64 MiB | 297.63 / 212.26 | 320.41 / 315.35 | 339.45 / 327.18 |
| Fresh / 256 MiB | 344.29 / 343.40 | 337.57 / 334.94 | 339.02 / 136.45 |
| After reclamation / 64 MiB | 302.23 / 292.67 | 312.30 / 305.17 | 307.43 / 301.33 |
| After reclamation / 256 MiB | 337.28 / 254.77 | 330.89 / 292.65 | 323.58 / 189.42 |

Fresh 64 MiB with 1 MiB reservation passes its four checks: median gains
14.0%, aggregate gains 54.1%, and five pairs win. The control's retained
779 ms sample materially affects its aggregate. The 256 KiB reservation's
7.6% median gain and four paired wins fail its target. Neither result
establishes a general large-file improvement.

For fresh 256 MiB, the 1 MiB candidate loses 1.5% median throughput and 60.3%
aggregate throughput, winning only two pairs. After reclamation it loses
4.1% median and 25.7% aggregate throughput, again winning only two pairs.
The 256 KiB candidate also loses about 2% median throughput in both 256 MiB
states. Its post-GC aggregate improves 14.9% because of a slow control, but
only two pairs win. All checks and full timing vectors are in
[summary.json](summary.json).

## Allocation and barrier evidence

After each timed close, FIEMAP records the physical layout without requesting
another sync. To avoid counting a split mapping as fragmentation, the
[analysis](physical-runs.json) merges mappings that are adjacent in both
logical and physical address. All measured extent flags are zero or LAST;
none are unknown, delayed or unwritten. The original full maps remain in
the plain and diagnostic results.

| Median contiguous physical runs | Ordinary append | Reserve 256 KiB | Reserve 1 MiB |
| --- | ---: | ---: | ---: |
| Fresh / 256 MiB | 25.5 | 871.0 | 179.0 |
| After reclamation / 256 MiB | 27.5 | 839.5 | 225.5 |

Thus these small explicit reservations increase fragmentation on the tested
filesystem. This is consistent with the ext4
[allocation-policy documentation](https://www.kernel.org/doc/html/v6.11/filesystems/ext4/overview.html#block-and-inode-allocation-policy),
which describes delaying placement until writeback to make broader allocation
decisions. Inferring that explicit small allocations interfere with that
coalescing is plausible; the measurements do not isolate the allocator's
internal decisions or prove that fragmentation causes a particular slow sync.

Plain fresh-256-MiB median write times are 154.55 / 157.81 / 152.63 ms;
final sync medians are 580.10 / 593.04 / 595.13 ms in the same policy order.
After reclamation, final sync medians are 574.97 / 592.50 / 615.10 ms.
Preallocation does not remove the common final-barrier cost.

The slow fresh 1 MiB-reservation plain operation takes **7.492 seconds**,
including **7.064 seconds in final sync**. After reclamation, another takes
3.314 seconds, including 2.839 seconds in sync. A retained ordinary-append
post-GC operation also takes 2.191 seconds, including 1.991 seconds in sync.
These are actual plain phase timers, not causes assigned from another run.

In the separate diagnostic, a fresh 256 KiB-reservation operation takes
5.550 seconds. Its 1,024 allocation calls total **12.79 ms** and its appends
total **140.99 ms**, whereas final `fdatasync` takes **5.308 seconds**.
All diagnostic allocation calls succeed. Across the 256 MiB diagnostic
groups their totals range from 5.27 to 16.70 ms; the final barriers still
dominate. No SQLite, WAL, checkpoint worker or publication mutex exists in
this ordinary-file probe. The observed wait can therefore occur without
those components, although the probe does not locate every kernel/device
cause or quantify their contribution to a particular VaneFS sample.

## Decision and validation

Do not add per-256-KiB or per-1-MiB fallocate to VaneFS. The complete screen
fails, physical allocation becomes more fragmented, and the durable-barrier
tail remains. These results also do not rule out all larger reservations,
different devices, or other filesystems. They give no reason to continue
tuning these two allocation granularities for the current append path.

The next useful software investigation is to measure CPU/SQL and copy costs
in matched 64/256 MiB native writes, including a tmpfs control, before changing
batching. Keep payload allocation, metadata batching and physical sync waits
separate in that attribution, and retain the complete FULL durability boundary.

All **18 related probe validations** pass on tmpfs: three policies, two file
states, plain/diagnostic/ASan+UBSan builds. Every measured file also passes a
complete byte check of its header, retained guard, hole and new data; hashes
agree across policies for each workload. This validates the ordinary-file
harness, not a new VaneFS recovery protocol. No production code, format,
SDK, installed module or durability policy changes; the published
[audit](audit.json) verifies their identities and **114 unique cleaned owned
directories**, with processes stopped before deletion. Only relevant probe
validation runs; no broad suite or component rebuild is needed.

## Reproduction

Run on a Linux ext4 output path with the C++ compiler available. The output
directory must not exist before preparation:

```bash
RUN="$PWD/vane_fs/build/payload-allocation-reproduction"
python3 vane_fs/benchmarks/staged_payload/payload_allocation/run.py "$RUN" prepare
python3 vane_fs/benchmarks/staged_payload/payload_allocation/run.py "$RUN" plain
python3 vane_fs/benchmarks/staged_payload/payload_allocation/run.py "$RUN" diagnostic
python3 vane_fs/benchmarks/staged_payload/payload_allocation/run.py "$RUN" analyze
python3 vane_fs/benchmarks/staged_payload/payload_allocation/publish.py "$RUN"
```

The last command refreshes this report directory's evidence files. Published
results retain every measurement, allocation/sync interval, physical map,
host counter, command, code identity, validation and cleanup hash. JSON
compaction preserves values exactly and records the original file hashes.
Each window's `index.json` retains the original process order and references
scenario/policy groups, keeping each artifact below the repository's large-file
limit without dropping a row or physical mapping. Reconstructing those records
must equal the original complete report before publication succeeds.
Large generated payloads are removed outside measured phases after their
owning processes exit. Compiler binaries and original artifacts remain local
with identities; no complete repeated payload outputs are retained.
