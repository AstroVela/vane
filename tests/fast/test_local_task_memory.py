# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace


def test_cgroup_headroom_respects_parent_and_host_limits(monkeypatch, tmp_path):
    import vane.execution.udf_local_resources as memory

    (tmp_path / "proc/self").mkdir(parents=True)
    (tmp_path / "proc/self/cgroup").write_text("0::/parent/child\n")
    root = tmp_path / "sys/fs/cgroup"
    child = root / "parent/child"
    child.mkdir(parents=True)
    for path, limit, used in [(root, "max", 0), (child.parent, 1000, 900), (child, 800, 100)]:
        (path / "memory.max").write_text(str(limit))
        (path / "memory.current").write_text(str(used))
    monkeypatch.setattr(memory, "Path", lambda path: tmp_path / Path(path).relative_to("/"))
    monkeypatch.setattr(memory.sys, "platform", "linux")
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=500))
    assert memory.available_process_memory() == 100
    (child.parent / "memory.max").write_text("max")
    assert memory.available_process_memory() == 500
