# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the preceding builder and add post-reuse commit progress in isolation."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--restart-parent", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    previous = here.parent / "checkpoint_restart"
    parent = args.restart_parent.resolve()
    old_build = json.loads((parent / "build.json").read_text())
    assert old_build["status"] == "PASS"
    out = args.output.resolve()
    generator = out.parent / (out.name + "-generator")
    generator.mkdir()
    source = (previous / "prepare.py").read_text()
    old = "    here = Path(__file__).resolve().parent"
    assert source.count(old) == 1
    source = source.replace(old, "    here = Path(" + repr(str(previous)) + ")")
    marker = '        report["source_inputs"]["updated"] = {'
    assert source.count(marker) == 1
    extra = """        run(
            "progress-patch",
            ["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-i", PROGRESS_PATCH],
            cwd=candidate,
        )
""".replace("PROGRESS_PATCH", repr(str(here / "progress.patch")))
    source = source.replace(marker, extra + marker)
    prepared = generator / "prepare.py"
    prepared.write_text(source)
    identity = dict(
        status="PASS",
        adaptation="Byte-preserved preceding builder, original controls and eight restart cases, plus one isolated progress patch",
        inputs={str(p): digest(p) for p in (here / "prepare.py", here / "progress.patch", previous / "prepare.py")},
        generated_sha256=digest(prepared),
    )
    (generator / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    support = here.parents[4] / "astrovela_vane-batch-udf-20260930/.venv-gpu/lib/python3.12/site-packages"
    subprocess.run(
        [
            sys.executable,
            str(prepared),
            str(out),
            "--parent",
            old_build["parent"],
            "--orphan-parent",
            old_build["orphan_parent"],
            "--sqlite-prefix",
            old_build["sdk"]["sqlite"],
            "--fuse-prefix",
            old_build["sdk"]["fuse"],
            "--test-support",
            str(support),
        ],
        check=True,
    )
    build = json.loads((out / "build.json").read_text())
    build["restart_parent"] = str(parent)
    build["generator"] = str(generator)
    build["inputs"].update(identity["inputs"])
    build["inputs"][str(prepared)] = digest(prepared)
    build["inputs"][str(here / "fixture.hpp")] = digest(here / "fixture.hpp")
    sdk = Path(build["sdk"]["sqlite"])
    variants = {
        "orphan-control": Path(build["orphan_parent"]) / "updated",
        "restart-control": parent / "updated",
        "updated": out / "updated",
    }
    for name, component in variants.items():
        (out / name).mkdir(exist_ok=True)
        for helper in ("test_progress", "probe"):
            source = here / (helper + ".cpp")
            build["inputs"][str(source)] = digest(source)
            binary = out / name / helper
            argv = [
                "c++",
                "-std=c++17",
                "-O3",
                "-DNDEBUG",
                "-I" + str(component / "include"),
                "-I" + str(component / "src"),
                "-I" + str(sdk / "include"),
                str(source),
                str(component / "build/python-release/libvane_fs.a"),
                str(sdk / "lib/libsqlite3.a"),
                "-ldl",
                "-pthread",
                "-Wl,--wrap=sqlite3_wal_checkpoint_v2",
                "-Wl,--wrap=fsync",
                "-Wl,--wrap=fdatasync",
                "-o",
                str(binary),
            ]
            if helper == "probe":
                trace = here.parent / "sync_costs"
                for filename in ("trace.hpp", "trace.cpp"):
                    build["inputs"][str(trace / filename)] = digest(trace / filename)
                argv[1:1] = [
                    "-I" + str(trace),
                    str(trace / "trace.cpp"),
                    "-Wl,--wrap=pread",
                    "-Wl,--wrap=pread64",
                    "-Wl,--wrap=pwrite",
                    "-Wl,--wrap=pwrite64",
                    "-Wl,--wrap=sync_file_range",
                ]
            started = time.time()
            with (out / (name + "-" + helper + "-build.log")).open("w") as log:
                done = subprocess.run(argv, stdout=log, stderr=log, timeout=120)
            build["commands"].append(
                dict(
                    label=name + "-" + helper,
                    argv=argv,
                    returncode=done.returncode,
                    started_epoch=started,
                    ended_epoch=time.time(),
                )
            )
            (out / "build.json").write_text(json.dumps(build, indent=2) + "\n")
            if done.returncode:
                build.update(status="FAIL", error="Helper link failed: " + name + "-" + helper)
                (out / "build.json").write_text(json.dumps(build, indent=2) + "\n")
                done.check_returncode()
            build["binaries"][str(binary.relative_to(out))] = digest(binary)
    (out / "build.json").write_text(json.dumps(build, indent=2) + "\n")


if __name__ == "__main__":
    main()
