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
