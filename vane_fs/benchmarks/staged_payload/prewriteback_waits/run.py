# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse the matched diagnostic window and add exact monotonic read boundaries."""

from pathlib import Path

from prepare import TRACE

if __name__ == "__main__":
    parent = Path(__file__).resolve().parent.parent / "payload_waits/run.py"
    code = parent.read_text()
    old = "Independent 64 MiB payload progress: full ordinary FUSE window; both formats 1005"
    new = "Advisory 64 MiB payload prewriteback: full ordinary FUSE window; both formats 1005"
    assert code.count(old) == 1
    code = code.replace(old, new)
    code = code.replace(
        "Payload progress sync/hold attribution; all original 11 ordinary FUSE failures remain",
        "Advisory prewriteback FULL/read attribution; all original 4 ordinary FUSE failures remain",
    )
    # Transform the generated runner before it freezes the workload and hashes.
    before = "    reverse = source\n"
    inserted = r'''
    replace(
        '(out / "workload.py").write_bytes((template / "workload.py").read_bytes())',
        """workload_source = (template / "workload.py").read_text()
    read_changes = [
        ('        count = 0\\n        while b := os.read(fd, MIB):', '        read_begin_ns = time.monotonic_ns()\\n        count = 0\\n        while b := os.read(fd, MIB):'),
        ('            count += len(b)\\n        seconds = time.perf_counter() - t', '            count += len(b)\\n        read_end_ns = time.monotonic_ns()\\n        seconds = time.perf_counter() - t'),
        ('                "workload": "read_64",', '                "workload": "read_64",\\n                "application_begin_ns": read_begin_ns,\\n                "application_end_ns": read_end_ns,\\n                "application_pid": os.getpid(),'),
    ]
    read_reverse = workload_source
    for before, after in read_changes:
        assert workload_source.count(before) == 1, before
        workload_source = workload_source.replace(before, after)
    read_reverse = workload_source
    for before, after in reversed(read_changes):
        assert read_reverse.count(after) == 1
        read_reverse = read_reverse.replace(after, before)
    assert read_reverse == (template / "workload.py").read_text()
    (out / "workload.py").write_text(workload_source)
    save(out / "read-workload-proof.json", dict(status="PASS", exact_reverse=True, changes=read_changes, original_sha256=digest(template / "workload.py")))""",
    )
'''
    assert code.count(before) == 1
    code = code.replace(before, inserted + before)
    exec(compile(code, str(parent), "exec"), dict(__name__="__main__", __file__=__file__, TRACE=TRACE))
