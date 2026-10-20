# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Measure misses, retained pins and first pins without concurrent owned builds/tests."""

import argparse
import hashlib
import json
import shutil
import sqlite3
import statistics
import subprocess
import tempfile
import time
from contextlib import closing
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--updated", type=Path, required=True)
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    args = parser.parse_args()
    out, sdk = args.output.resolve(), args.sqlite_prefix.resolve()
    out.mkdir()
    here = Path(__file__).resolve().parent
    sources = {"baseline": args.baseline.resolve(), "updated": args.updated.resolve()}
    identities = {str(p): digest(p) for p in (Path(__file__).resolve(), here / "probe.cpp", sdk / "lib/libsqlite3.a")}
    for name, source in sources.items():
        for path in (
            source / "src/workspace.cpp",
            source / "src/checkpoint.hpp",
            source / "build/python-release/libvane_fs.a",
        ):
            identities[str(path)] = digest(path)
        argv = [
            "c++",
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-I" + str(source / "include"),
            "-I" + str(sdk / "include"),
            str(here / "probe.cpp"),
            str(source / "build/python-release/libvane_fs.a"),
            str(sdk / "lib/libsqlite3.a"),
            "-ldl",
            "-pthread",
            "-o",
            str(out / name),
        ]
        subprocess.run(argv, check=True)
        identities[str(out / name)] = digest(out / name)
    plan = dict(
        identities=identities,
        pairs=6,
        orders=[["baseline", "updated"], ["updated", "baseline"]] * 3,
        media=["tmpfs", "ext4"],
        durability=["fsync", "strict"],
        frozen_epoch=time.time(),
        owned_build_test_overlap=False,
        cache_eviction=False,
        frequency_affinity_changes=False,
        scope="Native lookup latency; first-pin release and validation are outside request timers",
    )
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    result = dict(status="RUNNING", started_epoch=time.time(), samples=[], cleanup=[], summary={})

    def save():
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        for medium in plan["media"]:
            root = Path("/dev/shm") if medium == "tmpfs" else out
            assert (
                subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(root)], text=True).strip() == medium
            )
            for durability in plan["durability"]:
                for rep, order in enumerate(plan["orders"]):
                    for variant in order:
                        work = Path(tempfile.mkdtemp(prefix="owned-lookup-", dir=root))
                        label = f"{medium}-{durability}-{rep}-{variant}"
                        try:
                            start = time.time()
                            done = subprocess.run(
                                [str(out / variant), str(work / "db.sqlite"), durability],
                                capture_output=True,
                                text=True,
                                timeout=120,
                            )
                            (out / (label + ".log")).write_text(done.stderr)
                            done.check_returncode()
                            samples = [json.loads(line) for line in done.stdout.splitlines()]
                            raw = out / (label + ".json")
                            raw.write_text(json.dumps(samples) + "\n")
                            with closing(sqlite3.connect(f"file:{work / 'db.sqlite'}?mode=ro", uri=True)) as db:
                                assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
                                assert db.execute("SELECT count(*) FROM open_inodes").fetchone() == (0,)
                            row = dict(
                                medium=medium,
                                durability=durability,
                                rep=rep,
                                variant=variant,
                                start=start,
                                end=time.time(),
                                raw=raw.name,
                                sha256=digest(raw),
                                means_ns={s["case"]: statistics.mean(s["nanoseconds"]) for s in samples},
                            )
                            result["samples"].append(row)
                            print(label, "PASS", flush=True)
                        finally:
                            shutil.rmtree(work)
                            result["cleanup"].append(dict(path=str(work), removed=not work.exists()))
                            save()
                values = {}
                rows = [r for r in result["samples"] if r["medium"] == medium and r["durability"] == durability]
                for case in ("missing", "retained", "first"):
                    timings = {v: [r["means_ns"][case] / 1000 for r in rows if r["variant"] == v] for v in sources}
                    medians = {v: statistics.median(a) for v, a in timings.items()}
                    values[case] = dict(
                        round_means_us=timings,
                        median_us=medians,
                        updated_over_baseline=medians["updated"] / medians["baseline"],
                    )
                result["summary"][medium + "/" + durability] = values
        assert all(digest(Path(p)) == h for p, h in identities.items())
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        result["ended_epoch"] = time.time()
        save()


if __name__ == "__main__":
    main()
