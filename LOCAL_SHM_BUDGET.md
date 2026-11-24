# Local shared-memory capacity

The local runner resolves one arena capacity and uses that same decision for
transport and query object-store budgets. Its first query maps the arena during
preparation, before creating actors or starting task workers. Mapping failure
releases the query owner and propagates a resource error; no query is retried.
Context can already have entered its durable `committing` state before invoking
Vane preparation. This change does not redefine uncertain publication receipts.

`VANE_LOCAL_SHM_STORE_BYTES=auto` takes 30% of the tightest available memory
headroom, capped at 200 GiB and, on Linux, 95% of available `/dev/shm` space.
The remaining fraction leaves room for Python, decoding and intermediate copies.
Headroom is a snapshot, not a reservation against other processes.

- All platforms: available physical memory, minus an optional physical reserve.
- Windows: `GlobalMemoryStatusEx` available commit for the current process and
  available virtual address space. Available commit includes system pressure;
  physical RAM alone is insufficient for a page-file-backed mapping.
- Linux: visible cgroup v1/v2 memory limits and their visible ancestors, plus
  finite `RLIMIT_AS` headroom. Limits outside a container's visible namespace
  cannot be discovered here. If membership cannot be resolved against visible
  mounts, report that limitation instead of guessing a namespace or bind-mount
  root; expose a namespace-relative cgroup mount for automatic sizing.
- macOS: available physical memory and finite `RLIMIT_AS` headroom.

Applications can set `VANE_LOCAL_MEMORY_LIMIT_BYTES` to an integer process
memory ceiling and `VANE_LOCAL_MEMORY_RESERVE_BYTES` to an integer physical
reserve. Remaining application budget subtracts Windows private bytes or POSIX
resident bytes, matching `scripts/run_with_memory_guard.py`. The guard passes
these policies to its child and preserves stricter inherited settings. They do
not change worker concurrency or relax OS limits; they are sizing inputs, not a
replacement for supervision. Windows private bytes do not include all shared
mapping commitments, hence the separate OS commit bound.

An explicit `VANE_LOCAL_SHM_STORE_BYTES` value remains exact: it is checked
against observable headroom and either accepted or rejected, never silently
reduced. It continues to support byte suffixes such as `256MiB`. Transport's
automatic reference budget uses half of the pinned arena; query admission uses
the existing `VANE_QUERY_OBJECT_STORE_FRACTION`. Capacity stays fixed while
clients or retained views own the arena and is recomputed for the next arena.

Allocation diagnostics include requested bytes, the initial headroom snapshot,
current headroom on failure, errno and Windows error number. Payloads, mapping
names and credentials are excluded. An OS allocation race can still fail; the
error retains its original cause and does not trigger ingestion replay.

After two clean reviews, install an ordinary wheel with the unchanged qualified
native binary. Run `test_local_shm_capacity.py`, the affected store/capacity,
query admission, memory guard and local transport tests through the installed
launcher. Keep real Ray and GPU qualifications separate. Native Windows and
Linux results qualify those systems only; simulated Darwin branches are not a
native macOS qualification. No additional macOS CI is required for this change.
