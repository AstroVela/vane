# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build the reversible two-slot candidate, validate it, and link matched probes."""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from analyze import digest, save, verify


def recorder(original):
    changes = [
        ("#include <time.h>", "#include <time.h>\n#include <sys/stat.h>"),
        (
            "int error = 0, fd = -1, mode = -1, frames = -1, copied = -1;",
            "int error = 0, fd = -1, mode = -1, frames = -1, copied = -1;\n\tint64_t size_before = -1, size_after = -1, allocated_before = -1, allocated_after = -1;",
        ),
        (
            '\\"copied\\"],\\"threads\\":[',
            '\\"copied\\",\\"size_before\\",\\"size_after\\",\\"allocated_before\\",\\"allocated_after\\"],\\"threads\\":[',
        ),
        (
            "<< e.frames << ',' << e.copied << ']';",
            "<< e.frames << ',' << e.copied << ',' << e.size_before << ',' << e.size_after << ',' << e.allocated_before << ',' << e.allocated_after << ']';",
        ),
        (
            "\te.bytes = bytes;\n\te.mode = mode;\n\te.begin = Now(CLOCK_MONOTONIC);",
            """\te.bytes = bytes;
\te.mode = mode;
\tconst bool watched = stage == Pwrite || stage == Fsync || stage == Fdatasync;
\tauto entry_errno = errno;
\tif (watched) {
\t\te.file = FileKind(fd);
\t\tstruct stat st {};
\t\tif (!fstat(fd, &st)) {
\t\t\te.size_before = st.st_size;
\t\t\te.allocated_before = st.st_blocks * 512;
\t\t}
\t}
\terrno = entry_errno;
\te.begin = Now(CLOCK_MONOTONIC);""",
        ),
        (
            "\te.end = Now(CLOCK_MONOTONIC);\n\tspan.Finish();\n\tif (stage == Fsync",
            """\te.end = Now(CLOCK_MONOTONIC);
\tif (watched) {
\t\tstruct stat st {};
\t\tif (!fstat(fd, &st)) {
\t\t\te.size_after = st.st_size;
\t\t\te.allocated_after = st.st_blocks * 512;
\t\t}
\t}
\tspan.Finish();
\tif (stage == Pwrite || stage == Fsync""",
        ),
        (
            "\t\te.file = FileKind(fd);\n\t\te.result = result;",
            "\t\tif (!watched)\n\t\t\te.file = FileKind(fd);\n\t\te.result = result;",
        ),
    ]
    text = original
    for before, after in changes:
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    reverse = text
    for before, after in reversed(changes):
        assert reverse.count(after) == 1
        reverse = reverse.replace(after, before)
    assert reverse == original
    return text, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    out, parent, component = args.artifacts.resolve(), args.parent.resolve(), args.component.resolve()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    prior = json.loads((parent / "build.json").read_text())
    assert prior["status"] == "PASS"
    inputs = prior["inputs"] | prior["sources"] | prior["binaries"]
    for path in [parent / "build.json", *here.glob("*.py"), *here.glob("*.cpp"), here / "materialize.patch"]:
        inputs[str(path)] = digest(path)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    out.mkdir()
    candidate = out / "component"
    candidate.mkdir()
    for name in ("src", "include", "python", "tests", "LICENSES"):
        shutil.copytree(component / name, candidate / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("CMakeLists.txt", "pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(component / name, candidate / name)
    original_sources = {
        str(p): digest(p) for folder in ("src", "include") for p in (component / folder).rglob("*") if p.is_file()
    }
    inputs.update(original_sources)
    original_core = parent / "diagnostic/baseline"
    diagnostic = out / "diagnostic"
    for name in ("src", "include"):
        shutil.copytree(original_core / name, diagnostic / name)
    sdk = Path(prior["sdk"])
    report = dict(
        status="RUNNING",
        started_epoch=time.time(),
        parent=str(parent),
        original_component=str(component),
        component=str(candidate),
        sdk=str(sdk),
        inputs=inputs,
        sources={},
        binaries={},
        commands=[],
        tests=[],
        cleanup=[],
        production_unchanged=True,
        checkpoint_and_full_ordering_unchanged=True,
    )
    if args.preflight:
        preflight = args.preflight.resolve()
        previous = json.loads((preflight / "build.json").read_text())
        assert previous["status"] == "FAIL"
        retained = out / "preflight"
        retained.mkdir()
        for path in [
            *preflight.glob("*.json"),
            *preflight.glob("*.log"),
            *preflight.glob("preflight-sources/*"),
            *preflight.glob("retained_sources/*/*"),
            *[p for p in (preflight / "preflight").rglob("*") if p.is_file()],
        ]:
            target = retained / path.relative_to(preflight)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            inputs[str(target)] = digest(target)
        report["preflight"] = dict(
            path=str(preflight),
            error=previous["error"],
            native_runs=0,
            cleanup=previous["cleanup"] + previous.get("preflight", {}).get("cleanup", []),
            reason="Retain the .orig-backup equality-audit failure and the later missing-pytest launch failure. The test dependency is in the parent's support.pth path; copy that explicit path closure into the isolated environment. No native test case or measurement ran during either preflight.",
        )

    def run(label, argv, cwd=repo, env=None):
        log = out / (label + ".log")
        begin = time.time()
        with log.open("w") as stream:
            process = subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=stream, timeout=600)
        report["commands"].append(
            dict(
                label=label,
                argv=argv,
                cwd=str(cwd),
                started_epoch=begin,
                ended_epoch=time.time(),
                returncode=process.returncode,
                log=str(log),
                sha256=digest(log),
            )
        )
        save(out / "build.json", report)
        assert process.returncode == 0, log
        print(label, "PASS", flush=True)

    def clean(owned):
        assert str(owned) not in Path("/proc/mounts").read_text()
        files = {
            str(p.relative_to(owned)): dict(bytes=p.stat().st_size, sha256=digest(p))
            for p in owned.rglob("*")
            if p.is_file()
        }
        shutil.rmtree(owned)
        report["cleanup"].append(
            dict(path=str(owned), files=files, removed=not owned.exists(), processes_stopped=True, mount_absent=True)
        )
        save(out / "build.json", report)

    try:
        for name, target in (("component", candidate), ("diagnostic", diagnostic)):
            original = (target / "src/staging.hpp").read_bytes()
            run(
                name + "-patch",
                ["patch", "-p1", "--fuzz=0", "--no-backup-if-mismatch", "-i", str(here / "materialize.patch")],
                target,
            )
            modified = (target / "src/staging.hpp").read_bytes()
            run(
                name + "-reverse",
                ["patch", "-p1", "--fuzz=0", "--no-backup-if-mismatch", "-R", "-i", str(here / "materialize.patch")],
                target,
            )
            assert (target / "src/staging.hpp").read_bytes() == original
            (target / "src/staging.hpp").write_bytes(modified)
            report[name + "_exact_reverse"] = True
            retained = out / "retained_sources" / name
            retained.mkdir(parents=True)
            (retained / "original-staging.hpp").write_bytes(original)
            (retained / "materialized-staging.hpp").write_bytes(modified)
        for original in candidate.rglob("*"):
            if original.is_file() and original.relative_to(candidate) != Path("src/staging.hpp"):
                source = component / original.relative_to(candidate)
                assert original.read_bytes() == source.read_bytes()
                inputs[str(source)] = digest(source)
        for original in diagnostic.rglob("*"):
            if original.is_file() and original.relative_to(diagnostic) != Path("src/staging.hpp"):
                assert original.read_bytes() == (original_core / original.relative_to(diagnostic)).read_bytes()
        recorded, changes = recorder((original_core / "src/wait_trace.cpp").read_text())
        (out / "recorder.cpp").write_text(recorded)
        report.update(recorder_changes=changes, recorder_exact_reverse=True)
        python = candidate / "venv/bin/python"
        run("venv", ["uv", "venv", "--python", str(repo / ".venv/bin/python"), str(candidate / "venv")])
        support = candidate / "venv/lib/python3.12/site-packages/support.pth"
        support.write_text(
            str(component / "venv/lib/python3.12/site-packages")
            + "\n"
            + (component / "venv/lib/python3.12/site-packages/support.pth").read_text()
            + str(repo / ".venv/lib/python3.12/site-packages")
            + "\n"
        )
        env = dict(
            os.environ,
            SKBUILD_BUILD_DIR=str(candidate / "build/python-release"),
            SKBUILD_CMAKE_BUILD_TYPE="Release",
            CMAKE_BUILD_PARALLEL_LEVEL="2",
        )
        env.pop("PYTHONPATH", None)
        run(
            "install",
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                ".",
                "--no-build-isolation",
                "-Ccmake.define.unofficial-sqlite3_DIR=" + str(sdk / "share/unofficial-sqlite3"),
                "-Ccmake.define.VANE_FS_BUILD_TESTS=ON",
                "-Ccmake.define.VANE_FS_TEST_SQLITE_SYNC=fsync",
            ],
            candidate,
            env,
        )
        owned = Path(tempfile.mkdtemp(prefix="owned-fence-python-", dir="/dev/shm"))
        try:
            run(
                "python-tests",
                [
                    str(python),
                    "-I",
                    "-m",
                    "pytest",
                    "--rootdir",
                    str(candidate),
                    "--confcutdir",
                    str(candidate),
                    "--basetemp",
                    str(owned),
                    "-q",
                    *[
                        str(candidate / "tests" / name)
                        for name in (
                            "test_workspace.py",
                            "test_recovery.py",
                            "test_staging.py",
                            "test_external.py",
                            "test_fsspec.py",
                        )
                    ],
                ],
                Path("/tmp"),
                dict(env, VANE_FS_PRODUCTION_PYTHON=str(repo / ".venv/bin/python")),
            )
            report["tests"].append(dict(label="python-tests", mode="release", category="component_python"))
        finally:
            clean(owned)
        wraps = [
            "fsync",
            "fdatasync",
            "flock",
            "pread",
            "pread64",
            "pwrite",
            "pwrite64",
            "sqlite3_step",
            "sqlite3_exec",
            "sqlite3_wal_checkpoint_v2",
            "sync_file_range",
        ]
        for mode in ("release", "asan"):
            flags = (
                ["-O2", "-DNDEBUG"]
                if mode == "release"
                else ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
            )
            build = candidate / ("build/python-release" if mode == "release" else "sanitize")
            if mode == "asan":
                run(
                    "asan-configure",
                    [
                        "cmake",
                        "-S",
                        str(candidate),
                        "-B",
                        str(build),
                        "-DCMAKE_BUILD_TYPE=Debug",
                        "-DVANE_FS_BUILD_PYTHON=OFF",
                        "-DVANE_FS_BUILD_TESTS=ON",
                        "-DVANE_FS_TEST_SQLITE_SYNC=fsync",
                        "-Dunofficial-sqlite3_DIR=" + str(sdk / "share/unofficial-sqlite3"),
                        "-DCMAKE_CXX_FLAGS=" + " ".join(flags),
                    ],
                )
                run("asan-build", ["cmake", "--build", str(build), "-j2"])
            owned = Path(tempfile.mkdtemp(prefix="owned-fence-native-", dir="/dev/shm"))
            test_env = dict(
                os.environ,
                TMPDIR=str(owned),
                ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                UBSAN_OPTIONS="halt_on_error=1",
            )
            try:
                run(mode + "-component-tests", ["ctest", "--test-dir", str(build), "--output-on-failure"], env=test_env)
                report["tests"].append(dict(label=mode + "-component-tests", mode=mode, category="component_cpp"))
            finally:
                clean(owned)
                for leftover in build.glob("owned*"):
                    if leftover.is_dir():
                        clean(leftover)
            slot_test = out / (mode + "-test-slots")
            run(
                mode + "-slot-compile",
                [
                    "c++",
                    "-std=c++17",
                    *flags,
                    "-I" + str(candidate / "include"),
                    str(here / "test_slots.cpp"),
                    str(build / "libvane_fs.a"),
                    str(sdk / "lib/libsqlite3.a"),
                    "-pthread",
                    "-ldl",
                    "-Wl,--wrap=pwrite",
                    "-Wl,--wrap=fdatasync",
                    "-o",
                    str(slot_test),
                ],
            )
            owned = Path(tempfile.mkdtemp(prefix="owned-fence-slots-", dir="/dev/shm"))
            try:
                run(mode + "-slot-faults", [str(slot_test), str(owned)], env=test_env)
                report["tests"].append(dict(label=mode + "-slot-faults", mode=mode, category="slot_faults"))
            finally:
                clean(owned)
            report["binaries"][str(slot_test)] = digest(slot_test)
            trace_object = out / (mode + "-recorder.o")
            run(
                mode + "-recorder",
                [
                    "c++",
                    "-std=c++17",
                    *flags,
                    "-I" + str(original_core / "src"),
                    "-I" + str(sdk / "include"),
                    "-c",
                    str(out / "recorder.cpp"),
                    "-o",
                    str(trace_object),
                ],
            )
            contract = out / (mode + "-test-trace")
            run(
                mode + "-trace-compile",
                [
                    "c++",
                    "-std=c++17",
                    *flags,
                    "-I" + str(original_core / "src"),
                    str(here / "test_trace.cpp"),
                    str(trace_object),
                    str(sdk / "lib/libsqlite3.a"),
                    "-pthread",
                    "-ldl",
                    *["-Wl,--wrap=" + symbol for symbol in wraps if symbol != "fdatasync"],
                    "-o",
                    str(contract),
                ],
            )
            owned = Path(tempfile.mkdtemp(prefix="owned-fence-trace-", dir="/dev/shm"))
            try:
                db = owned / "trace.sqlite"
                run(mode + "-trace-contract", [str(contract), str(db)], env=test_env)
                trace = json.loads(Path(str(db) + ".waits.json").read_text())
                events = [e for t in trace["threads"] for e in t["events"]]
                assert len(events) == 5 and events[0][14:16] == [0, 4101]
                assert [e[7] for e in events[2:]] == [5, 4, 9]
                assert events[-1][14:18] == [-1] * 4
                shutil.copy2(str(db) + ".waits.json", out / (mode + "-trace-contract.json"))
                report["tests"].append(dict(label=mode + "-trace-contract", mode=mode, category="recorder"))
            finally:
                clean(owned)
            report["binaries"][str(contract)] = digest(contract)
            for variant in ("original", "materialized"):
                core = original_core if variant == "original" else diagnostic
                library = out / (mode + "-" + variant + "-core.o")
                run(
                    mode + "-" + variant + "-core",
                    [
                        "c++",
                        "-std=c++17",
                        *flags,
                        "-I" + str(core / "src"),
                        "-I" + str(core / "include"),
                        "-I" + str(sdk / "include"),
                        "-c",
                        str(core / "src/workspace.cpp"),
                        "-o",
                        str(library),
                    ],
                )
                binary = out / (mode + "-" + variant + "-probe")
                mapping = out / (binary.name + ".map")
                run(
                    binary.name,
                    [
                        "c++",
                        "-std=c++17",
                        *flags,
                        "-I" + str(core / "src"),
                        "-I" + str(core / "include"),
                        "-I" + str(sdk / "include"),
                        str(here / "probe.cpp"),
                        str(trace_object),
                        str(library),
                        str(sdk / "lib/libsqlite3.a"),
                        "-pthread",
                        "-ldl",
                        *["-Wl,--wrap=" + symbol for symbol in wraps],
                        "-Wl,-Map=" + str(mapping),
                        "-o",
                        str(binary),
                    ],
                )
                assert not re.search(r"libvane_fs\.a\([^)]*wait_trace\.cpp\.o\)", mapping.read_text())
                report["binaries"][str(binary)] = digest(binary)
                for state in ("fresh", "reused", "gc"):
                    owned = Path(tempfile.mkdtemp(prefix="owned-fence-probe-", dir="/dev/shm"))
                    label = binary.name + "-" + state
                    result = out / (label + ".json")
                    try:
                        run(label, [str(binary), str(owned / "db.sqlite"), state, variant, str(result)], env=test_env)
                        trace_path = owned / "db.sqlite.waits.json"
                        trace = json.loads(trace_path.read_text())
                        validation = verify(json.loads(result.read_text()), trace)
                        shutil.copy2(trace_path, out / (label + "-trace.json"))
                        report["tests"].append(dict(label=label, mode=mode, category="probe", validation=validation))
                    finally:
                        if (owned / "db.sqlite.waits.json").exists() and not (out / (label + "-trace.json")).exists():
                            shutil.copy2(owned / "db.sqlite.waits.json", out / (label + "-trace.json"))
                        clean(owned)
        for base in (candidate, diagnostic):
            for folder in ("src", "include", "python", "tests"):
                if (base / folder).exists():
                    report["sources"].update(
                        {
                            str(p): digest(p)
                            for p in (base / folder).rglob("*")
                            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
                        }
                    )
        report["sources"][str(out / "recorder.cpp")] = digest(out / "recorder.cpp")
        report["sources"].update({str(p): digest(p) for p in (out / "retained_sources").rglob("*") if p.is_file()})
        for path in [
            *candidate.glob("build/python-release/libvane_fs.a"),
            *candidate.glob("sanitize/libvane_fs.a"),
            *candidate.glob("venv/lib/python3.12/site-packages/vane_fs/*.so"),
            *candidate.glob("build/python-release/*_test"),
            *candidate.glob("sanitize/*_test"),
        ]:
            report["binaries"][str(path)] = digest(path)
        assert all(digest(Path(p)) == sha for p, sha in inputs.items())
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save(out / "build.json", report)


if __name__ == "__main__":
    main()
