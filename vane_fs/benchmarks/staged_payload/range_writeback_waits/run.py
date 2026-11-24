# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run matched full diagnostics with exact random-write and small-file intervals."""

import sys
from pathlib import Path

from prepare import TRACE, digest, save


def workload(template, out):
    original = (template / "workload.py").read_text()
    source, pairs = original, []

    def replace(before, after):
        nonlocal source
        assert source.count(before) == 1, before
        source = source.replace(before, after)
        pairs.append([before, after])

    replace("        latencies = []", "        latencies, diagnostic_intervals = [], []")
    replace(
        """            t2 = time.perf_counter()
            assert os.pwrite(fd, replacement, off) == 4096
            latencies.append(time.perf_counter() - t2)""",
        """            begin_ns = time.monotonic_ns()
            t2 = time.perf_counter()
            assert os.pwrite(fd, replacement, off) == 4096
            latencies.append(time.perf_counter() - t2)
            diagnostic_intervals.append(dict(kind="write", begin_ns=begin_ns, end_ns=time.monotonic_ns(), offset=off, bytes=4096))""",
    )
    replace(
        """        t2 = time.perf_counter()
        os.fsync(fd)
        sync = time.perf_counter() - t2""",
        """        begin_ns = time.monotonic_ns()
        t2 = time.perf_counter()
        os.fsync(fd)
        sync = time.perf_counter() - t2
        diagnostic_intervals.append(dict(kind="fsync", begin_ns=begin_ns, end_ns=time.monotonic_ns()))""",
    )
    replace(
        '            "workload": "random_write_4k",',
        '            "workload": "random_write_4k",\n            "diagnostic_intervals": diagnostic_intervals,\n            "application_pid": os.getpid(),',
    )
    replace(
        """        for i in range(128):
            t = time.perf_counter()
            fd = os.open(point / f"small-{i}",""",
        """        diagnostic_intervals = []
        for i in range(128):
            begin_ns = time.monotonic_ns()
            t = time.perf_counter()
            fd = os.open(point / f"small-{i}",""",
    )
    replace(
        """            finally:
                os.close(fd)
            samples.append(time.perf_counter() - t)""",
        """            finally:
                os.close(fd)
            samples.append(time.perf_counter() - t)
            diagnostic_intervals.append(dict(kind="small_file", index=i, begin_ns=begin_ns, end_ns=time.monotonic_ns()))""",
    )
    replace(
        '                "workload": "small_file_fsync",',
        '                "workload": "small_file_fsync",\n                "diagnostic_intervals": diagnostic_intervals,\n                "application_pid": os.getpid(),',
    )
    reverse = source
    for before, after in reversed(pairs):
        assert reverse.count(after) == 1
        reverse = reverse.replace(after, before)
    assert reverse == original
    compile(source, str(out / "workload.py"), "exec")
    (out / "workload.py").write_text(source)
    save(
        out / "workload-proof.json",
        dict(status="PASS", exact_reverse=True, original_sha256=digest(template / "workload.py"), changes=pairs),
    )


def main():
    out = Path(sys.argv[1]).resolve()
    original = (TRACE / "run.py").read_text()
    source, changes = original, []

    def replace(before, after):
        nonlocal source
        assert source.count(before) == 1, before
        source = source.replace(before, after)
        changes.append([before, after])

    replace('template = Path(build["parent"]) / "fuse-screen"', 'template = Path(build["template"])')
    replace(
        'replace("(1 if args.diagnostic else 3)", "(2 if args.diagnostic else 3)")',
        'replace(\'m.ORDERS = m.FIRST = [["baseline", "updated"], ["updated", "baseline"]] * 3\', \'m.ORDERS = m.FIRST = [["baseline", "updated"], ["updated", "baseline"]] * 2\')',
    )
    before = '    replace("import types", "import types\\nimport threading")'
    replace(
        before,
        before
        + "\n    replace('assert not args.diagnostic, \"Ordinary uninstrumented components only\"', 'assert args.diagnostic, \"Diagnostic components only\"')",
    )
    replace(
        'interpretation="Borrowed input buffers in matched format-1004 FUSE components on ext4; native CPU screen is separate",',
        'interpretation="Finite-range 64 MiB payload writeback: full ordinary FUSE window; both formats 1005",',
    )
    replace(
        'interpretation="Wait attribution for both frozen buffer implementations; original FUSE gate failures remain",',
        'interpretation="Range writeback phase/wait attribution; all original 7 ordinary FUSE failures remain",',
    )
    replace(
        '    (out / "workload.py").write_bytes((template / "workload.py").read_bytes())',
        "    workload(template, out)",
    )
    reverse = source
    for before, after in reversed(changes):
        assert reverse.count(after) == 1
        reverse = reverse.replace(after, before)
    assert reverse == original
    save(
        out / "wrapper-proof.json",
        dict(status="PASS", original_sha256=digest(TRACE / "run.py"), exact_reverse=True, changes=changes),
    )
    exec(
        compile(source, str(TRACE / "run.py"), "exec"), dict(__name__="__main__", __file__=__file__, workload=workload)
    )


if __name__ == "__main__":
    main()
