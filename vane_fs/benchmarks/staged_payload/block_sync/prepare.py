"""Link two diagnostic FUSE binaries against an existing fdatasync component."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    parser.add_argument("--fuse-prefix", type=Path, required=True)
    args = parser.parse_args()
    output, component, sqlite, fuse = (
        p.resolve() for p in (args.output, args.component, args.sqlite_prefix, args.fuse_prefix)
    )
    libraries = [
        component / "build/libvane_fs.a",
        sqlite / "lib/libsqlite3.a",
        fuse / "lib/x86_64-linux-gnu/libfuse3.so",
    ]
    for library in libraries:
        assert library.is_file(), library
    symbols = subprocess.check_output(["nm", "-u", str(libraries[1])], text=True)
    assert " U fdatasync\n" in symbols, "Compile SQLite with HAVE_FDATASYNC=1 first"
    original = (component / "src/fuse.cpp").read_text()
    entry = "int main(int argc, char **argv) {"
    assert original.count(entry) == 1
    original = "void timeline_install(const char *);\n" + original.replace(
        entry, entry + "\n if(argc>1)timeline_install(argv[1]);"
    )
    output.mkdir()  # Every experiment has a separate output directory.
    trace = output / "timeline.cpp"
    shutil.copy2(Path(__file__).with_name("timeline.cpp"), trace)
    report = {
        "status": "RUNNING",
        "started_epoch": time.time(),
        "inputs": {str(p): digest(p) for p in [*libraries, component / "src/fuse.cpp", trace]},
        "commands": [],
    }
    try:
        for variant, split in [("ordinary", 0), ("split", 1)]:
            target = output / variant
            (target / "build").mkdir(parents=True)
            if (component / "venv").exists():
                (target / "venv").symlink_to(component / "venv", target_is_directory=True)
            source = target / "fuse.cpp"
            source.write_text(original)
            command = [
                "g++",
                "-std=c++17",
                "-O3",
                "-DNDEBUG",
                f"-DVANE_FS_SPLIT_WAL_SYNC={split}",
                "-I" + str(component / "include"),
                "-I" + str(sqlite / "include"),
                "-I" + str(fuse / "include/fuse3"),
                str(source),
                str(trace),
                *map(str, libraries),
                "-Wl,-rpath," + str(libraries[-1].parent),
                "-ldl",
                "-pthread",
            ]
            wrapped = [
                "sqlite3_step",
                "sqlite3_wal_checkpoint_v2",
                "fsync",
                "fdatasync",
                "pread",
                "pwrite",
                "pread64",
                "pwrite64",
                "ftruncate",
                "ftruncate64",
            ]
            command += ["-Wl,--wrap=" + name for name in wrapped]
            command += ["-o", str(target / "build/vane-fs-mount")]
            start = time.time()
            with (output / "build.log").open("a") as log:
                result = subprocess.run(command, stdout=log, stderr=log)
            report["commands"].append(
                {
                    "argv": command,
                    "started_epoch": start,
                    "ended_epoch": time.time(),
                    "returncode": result.returncode,
                }
            )
            result.check_returncode()
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
