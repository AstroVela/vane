# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse verified native artifacts and related tests with separate GC Python modules."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--recipe-archive", type=Path, required=True)
    args = parser.parse_args()
    out, parent = args.output.resolve(), args.parent.resolve()
    out.mkdir()
    old = json.loads((parent / "build.json").read_text())
    archive = json.loads(args.recipe_archive.read_text())
    assert old["status"] == "PASS"
    inputs = {
        str(Path(__file__).resolve()): digest(Path(__file__)),
        str(parent / "build.json"): digest(parent / "build.json"),
    }
    for path, expected in old["inputs"].items():
        frozen = Path(archive.get(path, path))
        assert digest(frozen) == expected
        inputs[str(frozen)] = expected
    components = {
        "baseline": Path(old["parent"]) / "updated",
        "orphan-control": Path(old["orphan_parent"]) / "updated",
        "updated": parent / "updated",
    }
    for name, component in components.items():
        assert all(digest(component / p) == h for p, h in old["source_inputs"][name].items())
        if name == "updated":
            (out / name).symlink_to(component, target_is_directory=True)
        else:
            (out / name / "build").mkdir(parents=True)
            (out / name / "build/python-release").symlink_to(
                component / "build/python-release", target_is_directory=True
            )
            (out / name / "venv").symlink_to(component / "venv", target_is_directory=True)
            for binary in (parent / name).glob("*-test"):
                (out / name / binary.name).symlink_to(binary)
    assert all(digest(out / p) == h for p, h in old["binaries"].items())
    for name in ("tests.json", "sanitize.json", "index-tests.json"):
        value = json.loads((parent / name).read_text())
        assert value["status"] == "PASS"
        (out / name).write_bytes((parent / name).read_bytes())
        inputs[str(parent / name)] = digest(parent / name)
    report = dict(status="PASS", runs=[], cleanup=[], native_rebuilt=False)
    bindings = {}
    for name in ("baseline", "orphan-control", "updated"):
        component = components[name]
        native = component / "venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so"
        bindings[name] = dict(path=str(native.resolve()), sha256=digest(native))
        inputs[str(native.resolve())] = digest(native)
        work = Path(tempfile.mkdtemp(prefix="owned-vane-index-binding-", dir="/dev/shm"))
        start = time.time()
        try:
            code = "import sys,json,hashlib,sqlite3;import vane_fs._native as n;from vane_fs import Workspace;w=Workspace(sys.argv[1]);w.close();db=sqlite3.connect(sys.argv[1]);present=bool(db.execute(\"SELECT count(*) FROM sqlite_schema WHERE type='index' AND name='block_versions_high'\").fetchone()[0]);assert present==bool(int(sys.argv[2]));print(json.dumps({'module':n.__file__,'sha256':hashlib.sha256(open(n.__file__,'rb').read()).hexdigest(),'index_present':present}));db.close()"
            argv = [
                str(out / name / "venv/bin/python"),
                "-I",
                "-c",
                code,
                str(work / "db.sqlite"),
                "0" if name == "updated" else "1",
            ]
            result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
            (out / (name + "-binding.log")).write_text(result.stdout + result.stderr)
            result.check_returncode()
            observed = json.loads(result.stdout)
            assert observed["sha256"] == bindings[name]["sha256"]
            report["runs"].append(
                dict(
                    argv=argv,
                    returncode=result.returncode,
                    started_epoch=start,
                    ended_epoch=time.time(),
                    observed=observed,
                )
            )
        finally:
            files = [
                dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in work.rglob("*") if p.is_file()
            ]
            shutil.rmtree(work)
            report["cleanup"].append(dict(path=str(work), removed=not work.exists(), files=files))
    (out / "binding-tests.json").write_text(json.dumps(report, indent=2) + "\n")
    build = dict(
        old,
        inputs=inputs,
        python_modules=bindings,
        reused_from=str(parent),
        reuse_reason="Only GC Python bindings and pre-phase schema assertions change; all native/test binaries are identical",
    )
    build["commands"] = old["commands"] + report["runs"]
    (out / "build.json").write_text(json.dumps(build, indent=2) + "\n")
    shutil.copytree(parent / "probe-preflight-sources", out / "probe-preflight-sources")
    print("native/test identity reuse and 3 independent Python module/index checks PASS")


if __name__ == "__main__":
    main()
