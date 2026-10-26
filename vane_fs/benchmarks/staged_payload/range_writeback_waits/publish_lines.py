# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve raw request bytes with text chunks ending at complete JSON lines."""

import json
import sys
import time
from pathlib import Path

from prepare import digest, save

if __name__ == "__main__":
    here, out = Path(__file__).resolve().parent, Path(sys.argv[1]).resolve()
    audit = json.loads((here / "audit.json").read_text())
    index = json.loads((here / "requests/index.json").read_text())
    data = (out / "requests.json").read_bytes()
    assert b"".join((here / name).read_bytes() for name in index["parts"]) == data
    previous = {name: dict(bytes=(here / name).stat().st_size, sha256=digest(here / name)) for name in index["parts"]}
    for name in index["parts"]:
        (here / name).unlink()
        audit["retained_files"].pop(name)
    parts, start = [], 0
    while start < len(data):
        end = min(start + 300000, len(data))
        if end < len(data):
            end = data.rfind(b"\n", start, end) + 1
        assert end > start
        relative = f"requests/{len(parts):03}.txt"
        (here / relative).write_bytes(data[start:end])
        parts.append(relative)
        start = end
    assert b"".join((here / name).read_bytes() for name in parts) == data
    save(here / "requests/index.json", dict(bytes=len(data), sha256=digest(out / "requests.json"), parts=parts))
    audit["original_files"][str(out / "requests.json")]["retained_as"] = parts
    save(
        here / "validation/retention-adapter.json",
        dict(
            status="PASS",
            recorded_epoch=time.time(),
            original_publisher_sha256=digest(here / "publish.py"),
            reason="Raw byte boundaries split JSON indentation, creating trailing-space EOFs; complete-line boundaries preserve every byte.",
            previous_parts=previous,
            maximum_chunk_bytes=300000,
            original_sha256=digest(out / "requests.json"),
            reassembled_bytes_unchanged=True,
        ),
    )
    for path in here.rglob("*"):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            assert path.stat().st_size < 500 * 1024, path
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    save(here / "audit.json", audit)
    print("PASS: lossless complete-line chunks; unchanged request bytes and measurement inputs")
