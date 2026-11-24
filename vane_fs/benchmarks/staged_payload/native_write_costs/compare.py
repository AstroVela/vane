# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare borrowed buffers with the original core using the frozen native runner."""

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

from prepare import digest, save


def analyze(out):
    result = json.loads((out / "compare/results.json").read_text())
    assert result["status"] == "PASS" and len(result["runs"]) == 48
    groups = []
    for medium in ("ext4", "tmpfs"):
        for size in (64, 256):
            rows = [r for r in result["runs"] if r["medium"] == medium and r["mib"] == size]
            base = [r for r in rows if r["variant"] == "baseline"]
            candidate = [r for r in rows if r["variant"] == "borrowed"]
            assert len(base) == len(candidate) == 6

            def cpu(row):
                return next(p for p in row["phases"] if p["name"] == "writes")["thread_cpu_ns"]

            def close(row):
                return next(p for p in row["phases"] if p["name"] == "close")["wall_ns"]

            def p99(rows):
                values = sorted(q[1] - q[0] for r in rows for q in r["requests"])
                return values[min(len(values) - 1, int(len(values) * 0.99))]

            paired = [cpu(next(b for b in base if b["rep"] == r["rep"])) / cpu(r) for r in candidate]
            cpu_median = statistics.median(size * 1e9 / cpu(r) for r in candidate) / statistics.median(
                size * 1e9 / cpu(r) for r in base
            )
            cpu_aggregate = sum(cpu(r) for r in base) / sum(cpu(r) for r in candidate)
            total_median = statistics.median(size * 1e9 / r["total_ns"] for r in candidate) / statistics.median(
                size * 1e9 / r["total_ns"] for r in base
            )
            total_aggregate = sum(r["total_ns"] for r in base) / sum(r["total_ns"] for r in candidate)
            checks = dict(
                cpu_median=cpu_median >= 1.10,
                cpu_aggregate=cpu_aggregate >= 1.10,
                cpu_wins=sum(v > 1 for v in paired) >= 5,
                total_median=total_median >= 0.95,
                total_aggregate=total_aggregate >= 0.95,
                request_p99=p99(candidate) <= max(p99(base) * 1.2, p99(base) + 1000000),
                close_median=statistics.median(close(r) for r in candidate)
                <= statistics.median(close(r) for r in base) * 1.05,
            )
            groups.append(
                dict(
                    medium=medium,
                    mib=size,
                    cpu_median_ratio=cpu_median,
                    cpu_aggregate_ratio=cpu_aggregate,
                    cpu_paired_ratios=paired,
                    cpu_wins=sum(v > 1 for v in paired),
                    total_median_ratio=total_median,
                    total_aggregate_ratio=total_aggregate,
                    checks=checks,
                    screen_pass=all(checks.values()),
                    arms={
                        v: dict(
                            write_cpu_ns=[cpu(r) for r in data],
                            total_ns=[r["total_ns"] for r in data],
                            close_ns=[close(r) for r in data],
                            request_p99_ns=p99(data),
                            median_mib_s=statistics.median(size * 1e9 / r["total_ns"] for r in data),
                            aggregate_mib_s=size * 1e9 * len(data) / sum(r["total_ns"] for r in data),
                        )
                        for v, data in [("baseline", base), ("borrowed", candidate)]
                    },
                )
            )
    summary = dict(
        status="PASS",
        groups=groups,
        screen_pass=all(g["screen_pass"] for g in groups),
        failed_checks=sum(not v for g in groups for v in g["checks"].values()),
    )
    save(out / "compare-summary.json", summary)
    for g in groups:
        print(
            g["medium"],
            g["mib"],
            "CPU median/aggregate",
            round(g["cpu_median_ratio"], 4),
            round(g["cpu_aggregate_ratio"], 4),
            "wins",
            g["cpu_wins"],
            "checks",
            g["checks"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    if args.analyze:
        analyze(out)
        return
    build = json.loads((out / "borrow-build.json").read_text())
    assert build["status"] == "PASS"
    original = (here / "run.py").read_text()
    source = original
    edits = []

    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)
        edits.append((old, new))

    replace(
        "from prepare import clean, digest, save, validate",
        f"sys.path.insert(0, {str(here)!r})\nfrom prepare import clean, digest, save, validate",
    )
    replace(
        "out, here = args.artifacts.resolve(), Path(__file__).resolve().parent",
        f"out, here = args.artifacts.resolve(), Path({str(here)!r})",
    )
    replace('mode = "profile" if args.profile else "plain"', 'assert not args.profile\n    mode = "compare"')
    replace('dest = out / mode.replace("profile", "diagnostic")', 'dest = out / "compare"')
    replace(
        "    assert all(digest(Path(p)) == sha for p, sha in inputs.items())\n    mounts = {",
        '    candidate = json.loads((out / "borrow-build.json").read_text())\n    assert candidate["status"] == "PASS"\n    inputs.update(candidate["sources"])\n    inputs.update(candidate["binaries"])\n    inputs.update(candidate["inputs"])\n    inputs[str(out / "borrow-build.json")] = digest(out / "borrow-build.json")\n    assert all(digest(Path(p)) == sha for p, sha in inputs.items())\n    mounts = {',
    )
    replace(
        'scope="unchanged format-1004 native core attribution; CPU-clock measurement, not FUSE adoption",',
        'scope="borrowed-input native CPU screen; unchanged format-1004 and FULL barriers; FUSE acceptance separate",\n        variants=["baseline", "borrowed"],\n        cpu_gates=dict(median_and_aggregate_rate_minimum=1.10, paired_wins_minimum=5),\n        regression_gates=dict(total_median_and_aggregate_rate_minimum=0.95, request_p99_maximum_ratio=1.2, request_p99_absolute_allowance_ns=1000000, close_median_maximum_ratio=1.05),',
    )
    start = source.index("                for medium in order:\n")
    end = source.index("        assert all(digest(Path(p))", start)
    old = source[start:end]
    body = old.split("\n", 1)[1]
    body = body.replace('label = f"{size}-{rep}-{medium}"', 'label = f"{size}-{rep}-{medium}-{variant}"')
    body = body.replace(
        'str(out / ("probe-" + mode)),', 'str(out / ("probe-plain" if variant == "baseline" else "probe-borrowed")),'
    )
    body, count = re.subn(
        r"(^[ \t]*)medium=medium,\n", r"\1medium=medium,\n\1variant=variant,\n", body, flags=re.MULTILINE
    )
    assert count == 1
    new = (
        '                for medium in order:\n                    for variant in (["baseline", "borrowed"] if rep % 2 == 0 else ["borrowed", "baseline"]):\n'
        + "".join("    " + line if line.strip() else line for line in body.splitlines(True))
    )
    replace(old, new)
    restored = source
    for old, new in reversed(edits):
        assert restored.count(new) == 1
        restored = restored.replace(new, old)
    assert restored == original
    runner = out / "compare-runner.py"
    runner.write_text(source)
    save(
        out / "compare-runner-proof.json",
        dict(
            status="PASS",
            reverse_exact=True,
            source_sha256=digest(here / "run.py"),
            runner_sha256=digest(runner),
            edits=edits,
        ),
    )
    subprocess.run([sys.executable, str(runner), str(out)], check=True)


if __name__ == "__main__":
    main()
