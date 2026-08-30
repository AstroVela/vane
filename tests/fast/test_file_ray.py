# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import Any

import pytest

import vane
from vane.ai._media import PromptMedia
from vane.ai.protocols import PrompterDescriptor
from vane.ai.provider import Provider
from vane.ai.typing import UDFOptions


class _RayFilePrompter:
    async def prompt(self, messages: tuple[Any, ...]) -> str:
        rendered = [
            f"{message.content_type}={bytes(message).hex()}" if isinstance(message, PromptMedia) else str(message)
            for message in messages
        ]
        return f"{os.getpid()}:" + ":".join(rendered)


@dataclass
class _RayFilePrompterDescriptor(PrompterDescriptor):
    def get_provider(self) -> str:
        return "ray-file-test"

    def get_model(self) -> str:
        return "ray-file-test"

    def get_options(self) -> dict[str, object]:
        return {}

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=0, batch_size=1, max_retries=0)

    def instantiate(self) -> _RayFilePrompter:
        return _RayFilePrompter()


class _RayFileProvider(Provider):
    @property
    def name(self) -> str:
        return "ray-file-test"

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        return_format: dict[str, Any] | None = None,
        return_raw_response: bool = False,
        *,
        options: dict[str, Any] | None = None,
    ) -> _RayFilePrompterDescriptor:
        return _RayFilePrompterDescriptor()


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
        (vane.File("memory://root/a.txt", "text/plain"),),
        (vane.File("memory://root/b.json", "application/json"),),
    ]


@pytest.mark.usefixtures("ray_local")
def test_default_ray_executes_scalar_and_batch_file_udfs(monkeypatch):
    import pyarrow as pa

    monkeypatch.setenv("VANE_RUNNER", "ray")
    vane.teardown_runner()
    vane.set_runner_ray(noop_if_initialized=True)

    @vane.func(return_dtype=vane.file_type())
    def scalar_identity(value):
        assert isinstance(value, vane.File)
        return value

    @vane.func.batch(return_dtype=vane.file_type(), batch_size=2)
    def batch_identity(values):
        assert isinstance(values, (pa.Array, pa.ChunkedArray))
        return values

    @vane.func(return_dtype=vane.list_type(vane.file_type()))
    def nested_identity(values):
        assert isinstance(values[0], vane.File)
        assert values[1] is None
        return values

    connection = vane.connect()
    try:
        source = connection.sql(
            """
            SELECT
                i,
                file('memory://ray-udf/' || i::VARCHAR, NULL, NULL, NULL, NULL) AS value,
                [file('memory://ray-udf/' || i::VARCHAR, NULL, NULL, NULL, NULL), NULL::FILE] AS values
            FROM range(4) AS t(i)
            """
        )
        scalar_rows = source.select(vane.col("i"), scalar_identity(vane.col("value")).alias("value")).fetchall()
        batch_rows = source.select(vane.col("i"), batch_identity(vane.col("value")).alias("value")).fetchall()
        nested_rows = source.select(vane.col("i"), nested_identity(vane.col("values")).alias("values")).fetchall()
    finally:
        connection.close()

    expected = [(index, vane.File(f"memory://ray-udf/{index}")) for index in range(4)]
    assert sorted(scalar_rows) == expected
    assert sorted(batch_rows) == expected
    assert sorted(nested_rows) == [(index, [vane.File(f"memory://ray-udf/{index}"), None]) for index in range(4)]


@pytest.mark.usefixtures("ray_local")
def test_default_ray_ai_prompt_reads_file_view_on_worker(monkeypatch, tmp_path):
    path = tmp_path / "ray-ai-media.bin"
    path.write_bytes(b"prefix-media-suffix")
    path_sql = str(path).replace("'", "''")

    monkeypatch.setenv("VANE_RUNNER", "ray")
    vane.teardown_runner()
    vane.set_runner_ray(noop_if_initialized=True)

    connection = vane.connect()
    try:
        source = connection.sql(f"""
            SELECT
                'describe'::VARCHAR AS prompt,
                file('{path_sql}', 'audio/wav', 7, 5, NULL) AS media
        """)
        response = (
            vane.ai.prompt(
                source,
                [vane.col("prompt"), vane.col("media")],
                provider=_RayFileProvider(),
            )
            .project("response")
            .fetchone()[0]
        )
    finally:
        connection.close()

    worker_pid, rendered = response.split(":", 1)
    assert int(worker_pid) != os.getpid()
    assert rendered == "describe:audio/wav=6d65646961"
