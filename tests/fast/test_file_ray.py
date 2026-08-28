# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

import vane


@pytest.mark.usefixtures("ray_local")
def test_default_ray_materializes_scalar_and_nested_file_results(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    vane.teardown_runner()
    vane.set_runner_ray(noop_if_initialized=True)

    value = vane.File("memory://ray", "text/plain", 1, 2, "sha256:ray")
    connection = vane.connect()
    try:
        rows = connection.sql(
            """
            SELECT
                file('memory://ray', 'text/plain', 1, 2, 'sha256:ray') AS scalar_file,
                [file('memory://ray', 'text/plain', 1, 2, 'sha256:ray'), NULL::FILE] AS file_list,
                struct_pack(item := file('memory://ray', 'text/plain', 1, 2, 'sha256:ray')) AS file_struct,
                map(['item'], [file('memory://ray', 'text/plain', 1, 2, 'sha256:ray')]) AS file_map,
                union_value(item := file('memory://ray', 'text/plain', 1, 2, 'sha256:ray')) AS file_union
            FROM range(2)
            """
        ).fetchall()
    finally:
        connection.close()

    expected = (value, [value, None], {"item": value}, {"item": value}, value)
    assert rows == [expected, expected]


@pytest.mark.usefixtures("ray_local")
def test_default_ray_discovers_and_materializes_files(monkeypatch, tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.json"
    first.write_text("a", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("VANE_RUNNER", "ray")
    vane.teardown_runner()
    vane.set_runner_ray(noop_if_initialized=True)

    rows = vane.from_files(str(tmp_path / "*")).fetchall()

    assert rows == [
        (vane.File(str(first), "text/plain", 0, 1),),
        (vane.File(str(second), "application/json", 0, 2),),
    ]


@pytest.mark.usefixtures("ray_local")
def test_default_ray_discovers_connection_registered_filesystem_on_coordinator(monkeypatch):
    fsspec = pytest.importorskip("fsspec", minversion="2022.11.0")
    memory = fsspec.filesystem("memory", skip_instance_cache=True)
    memory.store = {}
    memory.pseudo_dirs = [""]
    memory.pipe("root/a.txt", b"a")
    memory.pipe("root/b.json", b"{}")

    monkeypatch.setenv("VANE_RUNNER", "ray")
    vane.teardown_runner()
    vane.set_runner_ray(noop_if_initialized=True)

    connection = vane.connect()
    connection.register_filesystem(memory)
    try:
        rows = vane.from_files("memory://root/*", connection=connection).fetchall()
    finally:
        connection.unregister_filesystem("memory")
        connection.close()

    assert sorted(rows, key=lambda row: row[0].url) == [
        (vane.File("memory:///root/a.txt", "text/plain"),),
        (vane.File("memory:///root/b.json", "application/json"),),
    ]
