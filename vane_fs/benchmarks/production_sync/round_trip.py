# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Check production BLOB databases across independently installed SQLite builds."""

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path


def worker():
    import vane_fs._native as native

    from vane_fs import Workspace

    operation, database, mode, label, state = sys.argv[2:]
    state = json.loads(state)
    original = bytes(range(251)) * 1100
    updated = bytearray(original)
    updated[4095:4102] = b"changed"
    # Python exposes strict mutations. C++ seed binaries exercise both modes.
    with Workspace(database) as workspace:
        live = workspace.checkout()
        if operation == "seed":
            live.mkdir("/dir")
            live.write_file("/dir/file", original)
            live.write_file("/sparse", b"")
            live.write("/sparse", b"end", offset=1024 * 1024 - 3)
            state["snapshot"] = workspace.snapshot()
            child = workspace.fork("main", "child")
            workspace.checkout(child.id).write_file("/child-only", b"child data")
            workspace.merge(workspace.preview_merge(child.id, "main"))
            live.write("/dir/file", b"changed", offset=4095)
        else:
            assert live.read("/dir/file") == updated
            assert live.read("/child-only") == b"child data"
            assert live.read("/sparse") == bytes(1024 * 1024 - 3) + b"end"
            with workspace.open_snapshot(state["snapshot"]) as frozen:
                assert frozen.read("/dir/file") == original
                assert "child-only" not in frozen.listdir()
            for previous in state.get("visitors", []):
                assert live.read("/reader-" + previous) == previous.encode()
            live.write_file("/reader-" + label, label.encode())
            state.setdefault("visitors", []).append(label)
            workspace.collect_garbage()
            assert live.read("/dir/file") == updated
            with workspace.open_snapshot(state["snapshot"]) as frozen:
                assert frozen.read("/dir/file") == original
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as db:
        assert db.execute("SELECT version,block_size FROM format").fetchall() == [(2, 4096)]
        assert db.execute("SELECT count(*) FROM block_payloads WHERE length(data)!=4096").fetchone() == (0,)
        assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    assert not Path(database + ".payload").exists()
    print(
        json.dumps(
            {
                "state": state,
                "module": native.__file__,
                "effective_durability": "strict",
                "content_sha256": hashlib.sha256(updated).hexdigest(),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--python", action="append", required=True, help="Unique label=/absolute/path/to/python")
    parser.add_argument(
        "--seed", action="append", default=[], help="Matching label=/path/to/native/seed for both modes"
    )
    args = parser.parse_args()
    interpreters = dict(v.split("=", 1) for v in args.python)
    assert len(interpreters) == len(args.python) >= 2
    assert all(label.isidentifier() and Path(binary).is_file() for label, binary in interpreters.items())
    seeds = dict(v.split("=", 1) for v in args.seed)
    assert len(seeds) == len(args.seed) and seeds.keys() <= interpreters.keys()
    assert all(Path(binary).is_file() for binary in seeds.values())
    args.output.mkdir()
    result = {"status": "RUNNING", "started_epoch": time.time(), "native_seeds": seeds, "cases": [], "cleanup": []}
    try:
        for writer, binary in interpreters.items():
            for mode in ("strict", "fsync") if writer in seeds else ("strict",):
                work = args.output / f"owned-{writer}-{mode}"
                work.mkdir()
                database = work / "db.sqlite"
                state = {}
                row = {"writer": writer, "durability": mode, "operations": []}
                result["cases"].append(row)
                try:
                    steps = [("seed", writer, binary)] + [
                        ("visit", label, path) for label, path in interpreters.items()
                    ]
                    # Reopen in the writer again after every other library has mutated it.
                    steps.append(("visit", writer + "_reopen", binary))
                    for operation, label, python in steps:
                        command = [
                            python,
                            "-I",
                            str(Path(__file__).resolve()),
                            "--worker",
                            operation,
                            str(database),
                            mode,
                            label,
                            json.dumps(state),
                        ]
                        if operation == "seed" and writer in seeds:
                            command = [seeds[writer], str(database), mode]
                        process = subprocess.run(command, capture_output=True, text=True, timeout=60)
                        entry = {
                            "command": command,
                            "returncode": process.returncode,
                            "stdout": process.stdout,
                            "stderr": process.stderr,
                        }
                        row["operations"].append(entry)
                        assert process.returncode == 0, entry
                        state = json.loads(process.stdout)["state"]
                    print(writer, mode, "PASS", flush=True)
                finally:
                    files = [
                        {
                            "path": str(p),
                            "bytes": p.stat().st_size,
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        }
                        for p in work.rglob("*")
                        if p.is_file()
                    ]
                    shutil.rmtree(work)
                    result["cleanup"].append({"path": str(work), "removed": not work.exists(), "files": files})
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        result["ended_epoch"] = time.time()
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker()
    else:
        main()
