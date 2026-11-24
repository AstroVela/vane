# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse extent checks and exercise paced scheduling plus retained fault cases."""

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
    here, out = Path(__file__).resolve().parent, Path(sys.argv[1]).resolve()
    parent = here.parent / "versioned_extents/build.py"
    original = code = parent.read_text()
    before = """            support = [
                Path(source["original"]) / "venv/lib/python3.12/site-packages",
                repo / ".venv/lib/python3.12/site-packages",
            ]"""
    changes = [
        (before, '            support = [Path(p) for p in source["dependency_site_paths"]]'),
        ('helper = here / "test_extents.cpp"', f"helper = Path({str(parent.parent / 'test_extents.cpp')!r})"),
    ]
    for before, after in changes:
        assert code.count(before) == 1
        code = code.replace(before, after)
    reverse = code
    for before, after in reversed(changes):
        reverse = reverse.replace(after, before)
    assert reverse == original
    (out / "component-build.py").write_text(code)
    save(
        out / "build-adapter.json",
        dict(parent=str(parent), parent_sha256=digest(parent), exact_reverse=True, changes=changes),
    )
    exec(compile(code, str(parent), "exec"), dict(__name__="__main__", __file__=str(Path(__file__).resolve())))
    stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "all"
    if stage not in ("all", "custom"):
        return
    report = json.loads((out / "build.json").read_text())
    source = json.loads((out / "source.json").read_text())
    assert report["status"] == "PASS"
    report["status"] = "RUNNING"
    target, sdk = Path(source["candidate"]), Path(source["sdk"])
    inherited = here.parent / "payload_range_writeback/test_prewriteback.cpp"
    text = inherited.read_text()
    begin = text.index("int main(int argc, char **argv) {")
    cases = [
        ("Unlocked", None),
        ("Failure", "false"),
        ("Failure", "true"),
        ("Failure", "false, ENOSPC"),
        ("Rollback", None),
        ("Garbage", None),
        ("Forked", None),
        ("Strict", None),
        ("Unsupported", "ENOSYS"),
        ("Unsupported", "EOPNOTSUPP"),
        ("Unsupported", "EINVAL"),
        ("FinalFailure", None),
        ("LateFailure", None),
        ("AsyncError", None),
    ]
    driver = "int main(int argc, char **argv) {\n\tif (argc != 2) return 2;\n\tapplication = true;\n\ttry {\n\t\tauto root = fs::path(argv[1]);\n"
    for i, (function, extra) in enumerate(cases):
        driver += f'\t\t{function}((root / "case-{i}.sqlite").string()' + (", " + extra if extra else "") + ");\n"
    driver += '\t\tRequire(unexpected_sync == 0 && bad_range == 0, "Invalid background syscall");\n\t} catch (const std::exception &error) { hold = false; std::cerr << error.what() << std::endl; return 1; }\n}\n'
    (out / "inherited-cases.cpp").write_text(text[:begin] + driver)
    save(
        out / "inherited-cases-proof.json",
        dict(
            parent=str(inherited),
            parent_sha256=digest(inherited),
            function_definitions_bytes_unchanged=True,
            selected_cases=cases,
            only_replaced_main=True,
        ),
    )

    def run(label, argv, env=None):
        log = out / f"{len(report['commands']):03}-{label}.log"
        started = time.time()
        with log.open("w") as stream:
            r = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=240)
        report["commands"].append(
            dict(
                label=label,
                argv=argv,
                started_epoch=started,
                ended_epoch=time.time(),
                returncode=r.returncode,
                log=str(log),
                log_sha256=digest(log),
            )
        )
        save(out / "build.json", report)
        assert r.returncode == 0, log
        print(label, "PASS", flush=True)

    try:
        for mode, directory in (("release", target / "build/python-release"), ("asan", target / "sanitize")):
            flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if mode == "asan" else []
            for name, helper, wraps in (
                ("pacing", here / "test_pacing.cpp", ["flock", "sync_file_range"]),
                ("inherited", out / "inherited-cases.cpp", ["fdatasync", "flock", "sync_file_range"]),
            ):
                binary = out / f"test-{name}-{mode}"
                run(
                    f"{mode}-{name}-compile",
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
                        *flags,
                        *["-Wl,--wrap=" + w for w in wraps],
                        "-o",
                        str(binary),
                    ],
                )
                owned = Path(tempfile.mkdtemp(prefix="owned-paced-" + name + "-", dir="/dev/shm"))
                try:
                    run(
                        f"{mode}-{name}-cases",
                        [str(binary), str(owned)],
                        dict(
                            os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1"
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
                    save(out / "build.json", report)
                report["binaries"][str(binary)] = digest(binary)
        report["inputs"].update(
            {
                str(p): digest(p)
                for p in [
                    parent,
                    inherited,
                    here / "test_pacing.cpp",
                    out / "component-build.py",
                    out / "inherited-cases.cpp",
                ]
            }
        )
        report["stages"]["paced_range_writeback"] = "PASS"
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save(out / "build.json", report)


if __name__ == "__main__":
    main()
