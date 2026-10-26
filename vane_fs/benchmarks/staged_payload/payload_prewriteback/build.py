# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse the frozen component validation and add payload scheduling fault tests."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from generate import digest, save


def main():
    here = Path(__file__).resolve().parent
    out = Path(sys.argv[1]).resolve()
    parent = here.parent / "versioned_extents/build.py"
    code = parent.read_text()
    changes = [
        (
            'Path(source["original"]) / "venv/lib/python3.12/site-packages"',
            'Path(source["dependency_support"]) / "venv/lib/python3.12/site-packages"',
        ),
        ('helper = here / "test_extents.cpp"', f"helper = Path({str(parent.parent / 'test_extents.cpp')!r})"),
    ]
    for before, after in changes:
        assert code.count(before) == 1
        code = code.replace(before, after)
    (out / "component-build.py").write_text(code)
    save(
        out / "build-adapter.json",
        dict(
            parent=str(parent),
            parent_sha256=digest(parent),
            changes=changes,
            prepared_sha256=digest(out / "component-build.py"),
        ),
    )
    # Keep the new adapter's directory for its package/build/test paths.
    exec(compile(code, str(parent), "exec"), {"__name__": "__main__", "__file__": str(Path(__file__).resolve())})
    stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "all"
    if stage not in ("custom", "all"):
        return
    report = json.loads((out / "build.json").read_text())
    source = json.loads((out / "source.json").read_text())
    assert report["status"] == "PASS"
    report["status"] = "RUNNING"
    save(out / "build.json", report)
    target, sdk = Path(source["candidate"]), Path(source["sdk"])
    helper = here / "test_prewriteback.cpp"

    def run(label, argv, env=None):
        log = out / f"{len(report['commands']):03}-{label}.log"
        begin = time.time()
        with log.open("w") as stream:
            process = subprocess.run(argv, env=env, stdout=stream, stderr=stream, timeout=180)
        report["commands"].append(
            dict(
                label=label,
                argv=argv,
                source_fingerprint=report["source_fingerprint"],
                started_epoch=begin,
                ended_epoch=time.time(),
                returncode=process.returncode,
                log=str(log),
                log_sha256=digest(log),
            )
        )
        save(out / "build.json", report)
        assert process.returncode == 0, log
        print(label, "PASS", flush=True)

    try:
        for mode, directory in [("release", target / "build/python-release"), ("asan", target / "sanitize")]:
            binary = out / ("test-prewriteback-" + mode)
            flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if mode == "asan" else []
            run(
                mode + "-prewriteback-compile",
                [
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-I" + str(target / "include"),
                    "-I" + str(target / "src"),
                    "-I" + str(sdk / "include"),
                    str(helper),
                    str(directory / "libvane_fs.a"),
                    str(sdk / "lib/libsqlite3.a"),
                    "-ldl",
                    "-pthread",
                    "-Wl,--wrap=fdatasync",
                    "-Wl,--wrap=flock",
                    *flags,
                    "-o",
                    str(binary),
                ],
            )
            owned = Path(tempfile.mkdtemp(prefix="owned-payload-prewriteback-", dir="/dev/shm"))
            try:
                run(
                    mode + "-prewriteback-cases",
                    [str(binary), str(owned)],
                    dict(
                        os.environ,
                        ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                        UBSAN_OPTIONS="halt_on_error=1",
                    ),
                )
            finally:
                assert str(owned) not in Path("/proc/mounts").read_text()
                files = {
                    str(p.relative_to(owned)): dict(bytes=p.stat().st_size, sha256=digest(p))
                    for p in owned.rglob("*")
                    if p.is_file()
                }
                shutil.rmtree(owned)
                report["cleanup"].append(
                    dict(
                        path=str(owned),
                        files=files,
                        removed=not owned.exists(),
                        processes_stopped=True,
                        mount_absent=True,
                    )
                )
            report["binaries"][str(binary)] = digest(binary)
        report["inputs"].update(
            {str(p): digest(p) for p in [helper, Path(__file__), parent, out / "component-build.py"]}
        )
        report["stages"]["payload_prewriteback"] = "PASS"
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save(out / "build.json", report)


if __name__ == "__main__":
    main()
