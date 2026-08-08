# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

import vane


class RecordingConnection:
    def __init__(self):
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)


def test_require_loads_present_extension(require, tmp_path, monkeypatch):
    extension_path = tmp_path / "custom" / "fixture_test.duckdb_extension"
    extension_path.parent.mkdir()
    extension_path.touch()
    connection = RecordingConnection()
    connect_calls = []

    def connect(db_name, config):
        connect_calls.append((db_name, config))
        return connection

    monkeypatch.setenv("DUCKDB_PYTHON_TEST_EXTENSION_PATH", str(tmp_path))
    monkeypatch.setattr(vane, "connect", connect)

    assert require("fixture_test", "fixture.db") is connection
    assert connect_calls == [("fixture.db", {"allow_unsigned_extensions": "true"})]
    assert connection.statements == [f"LOAD '{extension_path.resolve()}'"]


def test_require_skips_only_when_extension_is_absent(require, tmp_path, monkeypatch):
    unrelated_extension = tmp_path / "unrelated.duckdb_extension"
    unrelated_extension.touch()
    monkeypatch.setenv("DUCKDB_PYTHON_TEST_EXTENSION_PATH", str(tmp_path))

    def unexpected_connect(*_args, **_kwargs):
        pytest.fail("require should not connect without a matching extension")

    monkeypatch.setattr(vane, "connect", unexpected_connect)

    with pytest.raises(pytest.skip.Exception, match="could not load missing"):
        require("missing")
