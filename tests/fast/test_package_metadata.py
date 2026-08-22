# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import platform
import subprocess
import sys
import types
from importlib.metadata import distribution, metadata, requires, version
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

import vane

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_vane_public_exports_are_unique_and_resolvable():
    assert len(vane.__all__) == len(set(vane.__all__))

    expected_vane_exports = {
        "Connection",
        "EnvRegistry",
        "Relation",
        "VaneConfig",
        "__engine_version__",
        "__version__",
        "attach_function",
        "cls",
        "col",
        "configure",
        "current_config",
        "detach_function",
        "env",
        "func",
        "lit",
        "sql_expr",
    }
    assert expected_vane_exports <= set(vane.__all__)
    assert all(hasattr(vane, name) for name in vane.__all__)
    undeclared_public_objects = {
        name
        for name, value in vars(vane).items()
        if not name.startswith("_") and not isinstance(value, types.ModuleType) and name not in vane.__all__
    }
    assert not undeclared_public_objects

    wildcard_namespace: dict[str, object] = {}
    exec("from vane import *", wildcard_namespace)
    assert wildcard_namespace["Connection"] is vane.DuckDBPyConnection
    assert wildcard_namespace["Relation"] is vane.DuckDBPyRelation

    assert not hasattr(vane, "__duckdb_version__")
    assert not hasattr(vane, "__vane_version__")
    assert not hasattr(vane, "vane_runners_cpp")


def test_vane_adbc_public_exports_are_explicit():
    pytest.importorskip("adbc_driver_manager")
    import vane.adbc

    assert vane.adbc.__all__ == ["StatementOptions", "connect", "driver_path"]
    wildcard_namespace: dict[str, object] = {}
    exec("from vane.adbc import *", wildcard_namespace)
    assert {name for name in wildcard_namespace if name != "__builtins__"} == set(vane.adbc.__all__)


def test_vane_lazily_exposes_owned_public_submodules():
    for name in ("ai", "runners", "sqltypes", "udf"):
        assert getattr(vane, name) is importlib.import_module(f"vane.{name}")
        assert name in dir(vane)

    with pytest.raises(AttributeError, match="has no attribute 'not_a_vane_submodule'"):
        getattr(vane, "not_a_vane_submodule")


def test_native_submodules_share_the_public_runtime_identity():
    import vane.sqltypes as vane_sqltypes
    import vane.udf as vane_udf
    from vane import _native

    assert _native._func.FunctionNullHandling is vane_udf.FunctionNullHandling
    assert _native._func.PythonUDFType is vane_udf.PythonUDFType
    assert _native._sqltypes.DuckDBPyType is vane_sqltypes.DuckDBPyType
    assert Path(_native.__file__).name.startswith("_native.")
    assert Path(_native.__file__).suffix in {".pyd", ".so"}


def test_native_enum_members_match_the_public_contract():
    import vane.udf as vane_udf

    expected_members = {
        vane.StatementType: (
            "INVALID",
            "SELECT",
            "INSERT",
            "UPDATE",
            "CREATE",
            "DELETE",
            "PREPARE",
            "EXECUTE",
            "ALTER",
            "TRANSACTION",
            "COPY",
            "ANALYZE",
            "VARIABLE_SET",
            "CREATE_FUNC",
            "EXPLAIN",
            "DROP",
            "EXPORT",
            "PRAGMA",
            "VACUUM",
            "CALL",
            "SET",
            "LOAD",
            "RELATION",
            "EXTENSION",
            "LOGICAL_PLAN",
            "ATTACH",
            "DETACH",
            "MULTI",
            "COPY_DATABASE",
            "MERGE_INTO",
        ),
        vane.ExpectedResultType: ("QUERY_RESULT", "CHANGED_ROWS", "NOTHING"),
        vane.ExplainType: ("STANDARD", "ANALYZE"),
        vane.CSVLineTerminator: ("LINE_FEED", "CARRIAGE_RETURN_LINE_FEED"),
        vane.PythonExceptionHandling: ("DEFAULT", "RETURN_NULL"),
        vane.RenderMode: ("ROWS", "COLUMNS"),
        vane.token_type: ("identifier", "numeric_const", "string_const", "operator", "keyword", "comment"),
        vane_udf.PythonUDFType: ("NATIVE", "ARROW"),
        vane_udf.FunctionNullHandling: ("DEFAULT", "SPECIAL"),
    }

    for enum_type, members in expected_members.items():
        assert tuple(enum_type.__members__) == members


def test_native_runtime_edge_cases_match_the_typing_contract():
    from vane import _native

    for runtime_class in (_native.DuckDBPyConnection, _native.DuckDBPyRelation, _native.Statement):
        with pytest.raises(TypeError, match="No constructor defined"):
            runtime_class()

    with pytest.raises(TypeError):
        _native._sqltypes.DuckDBPyType(type_str="INTEGER", connection=None)
    with pytest.raises(TypeError):
        _native._sqltypes.DuckDBPyType(obj="INTEGER")

    connection = vane.connect()
    assert connection.description is None
    connection.execute("SELECT 1 AS value")
    assert connection.description == [("value", vane.sqltypes.INTEGER, None, None, None, None, None)]

    children_by_type = {
        "array": vane.array_type(vane.sqltypes.INTEGER, 3).children,
        "decimal": vane.decimal_type(10, 2).children,
        "enum": connection.sql("SELECT 'sad'::ENUM('sad', 'ok')").types[0].children,
        "tensor": vane.tensor_type(vane.sqltypes.FLOAT, [2, 3]).children,
    }
    assert children_by_type == {
        "array": [("child", vane.sqltypes.INTEGER), ("size", 3)],
        "decimal": [("precision", 10), ("scale", 2)],
        "enum": [("values", ["sad", "ok"])],
        "tensor": [("dtype", vane.sqltypes.FLOAT), ("shape", (2, 3))],
    }


def _expected_duckdb_source_id(repository_root: Path) -> str:
    source_id_file = repository_root / "DUCKDB_SOURCE_ID"
    if not (repository_root / ".git").exists() and source_id_file.is_file():
        return source_id_file.read_text(encoding="ascii").strip()

    result = subprocess.run(
        [sys.executable, "scripts/sync_duckdb_source_id.py", "--print"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _expected_duckdb_fork_version(repository_root: Path) -> str:
    result = subprocess.run(
        [sys.executable, "scripts/resolve_duckdb_fork_version.py", "--print-version"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _base_requirements():
    base_requirements = {}
    for raw_requirement in requires("vane-ai") or []:
        requirement = Requirement(raw_requirement)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            base_requirements[canonicalize_name(requirement.name)] = requirement

    return base_requirements


def test_base_distribution_installs_expression_runtime_dependencies():
    base_requirements = _base_requirements()

    assert {"numpy", "pyarrow"} <= set(base_requirements)
    assert "duckdb" not in base_requirements


def test_base_distribution_requires_pyarrow_14_or_newer():
    pyarrow_requirement = _base_requirements()["pyarrow"]

    assert pyarrow_requirement.specifier == SpecifierSet(">=14.0.0")


def test_base_distribution_requires_botocore_1_38_or_newer():
    botocore_requirement = _base_requirements()["botocore"]

    assert botocore_requirement.specifier == SpecifierSet(">=1.38.0,<2")


def test_artifact_mode_imports_installed_python_packages():
    if os.environ.get("VANE_FAST_TEST_ARTIFACT_MODE") != "1":
        pytest.skip("only applies to artifact-backed fast-test jobs")

    import vane

    environment_root = Path(sys.prefix).resolve()
    from vane import _native

    for package in (_native, vane):
        package_path = Path(package.__file__).resolve()
        assert package_path.is_relative_to(environment_root), (
            f"{package.__name__} was imported from the checkout: {package_path}"
        )


def test_vane_distribution_declares_inline_types():
    import vane

    assert Path(vane.__file__).with_name("py.typed").is_file()


def test_vane_distribution_owns_only_the_vane_import_namespace():
    files = {str(path).replace("\\", "/") for path in distribution("vane-ai").files or []}
    removed_runner_compatibility_modules = {
        "vane/runners/ray/_fte_compat.py",
        "vane/runners/ray/fte.py",
        "vane/runners/ray/fte_attempts.py",
        "vane/runners/ray/fte_config.py",
        "vane/runners/ray/fte_descriptor.py",
        "vane/runners/ray/fte_events.py",
        "vane/runners/ray/fte_exchange.py",
        "vane/runners/ray/fte_execution.py",
        "vane/runners/ray/fte_failures.py",
        "vane/runners/ray/fte_scheduler.py",
        "vane/runners/ray/fte_split_assigner.py",
        "vane/runners/ray/fte_state.py",
        "vane/runners/ray/fte_types.py",
        "vane/runners/ray/fte_update_batch.py",
        "vane/runners/ray/fte_worker_runtime.py",
    }

    assert any(path.startswith("vane/_native.") and path.endswith((".so", ".pyd")) for path in files)
    assert not any(path == "duckdb" or path.startswith(("duckdb/", "_duckdb", "adbc_driver_duckdb/")) for path in files)
    assert files.isdisjoint(removed_runner_compatibility_modules)


def _requirements_for_extra(extra):
    selected = set()
    for raw_requirement in requires("vane-ai") or []:
        requirement = Requirement(raw_requirement)
        if requirement.marker is not None and requirement.marker.evaluate({"extra": extra}):
            selected.add(canonicalize_name(requirement.name))
    return selected


def _requirement_for_extra(extra, package, environment=None):
    selected = []
    marker_environment = {"extra": extra, **(environment or {})}
    for raw_requirement in requires("vane-ai") or []:
        requirement = Requirement(raw_requirement)
        if canonicalize_name(requirement.name) != canonicalize_name(package):
            continue
        if requirement.marker is not None and requirement.marker.evaluate(marker_environment):
            selected.append(requirement)
    assert len(selected) == 1
    return selected[0]


def test_distribution_declares_canonical_version_and_apache_license_expression():
    package_metadata = metadata("vane-ai")
    distribution_version = version("vane-ai")

    assert str(Version(distribution_version)) == distribution_version
    assert package_metadata["Version"] == distribution_version
    assert vane.__version__ == distribution_version
    assert package_metadata["License-Expression"] == "Apache-2.0"
    assert SpecifierSet(package_metadata["Requires-Python"]) == SpecifierSet(">=3.10,<3.15")


def test_provider_extras_match_provider_import_errors():
    assert _requirements_for_extra("openai") == {"openai", "tiktoken"}
    assert _requirements_for_extra("anthropic") == {"anthropic"}
    assert _requirements_for_extra("google") == {"google-genai"}
    assert {"sentence-transformers", "torch", "transformers"} <= _requirements_for_extra("transformers")
    assert "vllm" in _requirements_for_extra("vllm")
    assert "sglang" in _requirements_for_extra("sglang")


def test_structured_provider_extras_require_supported_sdk_versions():
    openai = _requirement_for_extra("openai", "openai")
    google = _requirement_for_extra("google", "google-genai")
    vllm = _requirement_for_extra(
        "vllm",
        "vllm",
        {"platform_system": "Linux", "platform_machine": "x86_64"},
    )
    sglang = _requirement_for_extra(
        "sglang",
        "sglang",
        {"platform_system": "Linux", "platform_machine": "x86_64"},
    )

    assert openai.specifier == SpecifierSet(">=1.66.0")
    assert google.specifier == SpecifierSet(">=1.22.0")
    assert vllm.specifier == SpecifierSet(">=0.11.0")
    assert sglang.specifier == SpecifierSet(">=0.5.17,<0.6")


def test_wheel_or_install_contains_primary_and_third_party_license_files():
    files = {str(path).replace("\\", "/") for path in distribution("vane-ai").files or []}

    assert any(path.endswith("licenses/LICENSE") for path in files)
    assert any(path.endswith("licenses/NOTICE") for path in files)
    assert any(path.endswith("licenses/LICENSES/DuckDB-MIT.txt") for path in files)
    assert any(path.endswith("licenses/LICENSES/vcpkg-binary-dependencies.txt") for path in files)
    assert any(path.endswith("licenses/vane/experimental/spark/LICENSE") for path in files)
    assert any(path.endswith("compression/alp/algorithm/LICENSE") for path in files)
    assert any(path.endswith("compression/alprd/algorithm/LICENSE") for path in files)


def test_duckdb_version_and_source_id_match_recorded_engine_identities():
    from vane import _native

    fork_version = _expected_duckdb_fork_version(REPOSITORY_ROOT)
    source_tree_id = _expected_duckdb_source_id(REPOSITORY_ROOT)
    embedded_version, embedded_source_id = vane.sql(
        "SELECT library_version, source_id FROM pragma_version()"
    ).fetchone()

    assert embedded_version == fork_version
    assert embedded_source_id == source_tree_id[:10]
    assert _native.__version__ == fork_version.removeprefix("v")
    assert vane.__version__ == version("vane-ai")
    assert vane.__engine_version__ == fork_version.removeprefix("v")
    assert vane.__git_revision__ == embedded_source_id


def test_release_runtime_is_self_contained_by_default():
    connection = vane.connect()
    settings = dict(
        connection.execute(
            """
            SELECT name, value
            FROM duckdb_settings()
            WHERE name IN (
                'allow_unsigned_extensions',
                'autoinstall_known_extensions',
                'autoload_known_extensions'
            )
            """
        ).fetchall()
    )
    static_extensions = {
        name: (installed, loaded, extension_version)
        for name, installed, loaded, extension_version in connection.execute(
            """
            SELECT extension_name, installed, loaded, extension_version
            FROM duckdb_extensions()
            WHERE install_mode = 'STATICALLY_LINKED'
            """
        ).fetchall()
    }

    expected_extensions = {"core_functions", "httpfs", "icu", "json", "parquet"}

    assert settings == {
        "allow_unsigned_extensions": "false",
        "autoinstall_known_extensions": "false",
        "autoload_known_extensions": "false",
    }
    assert set(static_extensions) == expected_extensions
    assert all(installed and loaded for installed, loaded, _ in static_extensions.values())

    source_id = _expected_duckdb_source_id(REPOSITORY_ROOT)[:10]
    in_tree_extensions = expected_extensions - {"httpfs"}
    assert {name: static_extensions[name][2] for name in in_tree_extensions} == {
        name: source_id for name in in_tree_extensions
    }


@pytest.mark.parametrize(
    ("setting_name", "setting_value", "expected"),
    [
        ("http_timeout", "41", 41),
        ("binary_as_string", "true", True),
        ("calendar", "gregorian", "gregorian"),
    ],
)
def test_static_extension_settings_are_available_during_connect(tmp_path, setting_name, setting_value, expected):
    extension_directory = tmp_path / "extensions"
    connection = vane.connect(
        config={
            setting_name: setting_value,
            "extension_directory": str(extension_directory),
            "custom_extension_repository": "http://127.0.0.1:9",
        }
    )

    assert connection.execute(f"SELECT current_setting('{setting_name}')").fetchone()[0] == expected
    assert not extension_directory.exists()


def test_dynamic_extension_settings_are_rejected_during_connect_without_installing(tmp_path):
    extension_directory = tmp_path / "extensions"

    with pytest.raises(vane.InvalidInputException, match="sqlite_all_varchar.*sqlite_scanner.*not statically linked"):
        vane.connect(
            config={
                "sqlite_all_varchar": "true",
                "extension_directory": str(extension_directory),
                "custom_extension_repository": "http://127.0.0.1:9",
            }
        )

    assert not extension_directory.exists()


def test_exported_tree_without_manifest_computes_source_id(tmp_path, monkeypatch):
    expected = "a" * 40

    def run_source_id(command, **kwargs):
        assert command == [sys.executable, "scripts/sync_duckdb_source_id.py", "--print"]
        assert kwargs["cwd"] == tmp_path
        return subprocess.CompletedProcess(command, 0, stdout=expected + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run_source_id)

    assert _expected_duckdb_source_id(tmp_path) == expected


def test_sdist_tree_uses_injected_source_id(tmp_path, monkeypatch):
    expected = "b" * 40
    (tmp_path / "DUCKDB_SOURCE_ID").write_text(expected + "\n", encoding="ascii")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("an sdist must use its injected SourceID"),
    )

    assert _expected_duckdb_source_id(tmp_path) == expected


def test_image_extra_installs_pillow():
    assert _requirements_for_extra("image") == {"pillow"}


def test_video_extra_installs_video_dependencies():
    selected = _requirements_for_extra("video")
    assert {"pillow", "psutil"} <= selected
    supports_decord = platform.system() == "Linux" and platform.machine() == "x86_64"
    assert ("decord" in selected) is supports_decord


def test_base_distribution_keeps_video_dependencies_optional():
    base_requirements = set()
    for raw_requirement in requires("vane-ai") or []:
        requirement = Requirement(raw_requirement)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            base_requirements.add(canonicalize_name(requirement.name))

    assert {"pillow", "psutil", "decord"}.isdisjoint(base_requirements)
