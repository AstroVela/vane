from importlib.metadata import distribution, metadata, requires, version

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def test_base_distribution_installs_expression_runtime_dependencies():
    base_requirements = set()
    for raw_requirement in requires("vane-ai") or []:
        requirement = Requirement(raw_requirement)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            base_requirements.add(canonicalize_name(requirement.name))

    assert {"numpy", "pyarrow"} <= base_requirements


def _requirements_for_extra(extra):
    selected = set()
    for raw_requirement in requires("vane-ai") or []:
        requirement = Requirement(raw_requirement)
        if requirement.marker is not None and requirement.marker.evaluate({"extra": extra}):
            selected.add(canonicalize_name(requirement.name))
    return selected


def test_distribution_declares_alpha_version_and_apache_license_expression():
    package_metadata = metadata("vane-ai")

    assert version("vane-ai") == "0.1.0a1"
    assert package_metadata["License-Expression"] == "Apache-2.0"


def test_provider_extras_match_provider_import_errors():
    assert _requirements_for_extra("openai") == {"openai"}
    assert _requirements_for_extra("anthropic") == {"anthropic"}
    assert _requirements_for_extra("google") == {"google-genai"}
    assert {"sentence-transformers", "torch", "transformers"} <= _requirements_for_extra("transformers")
    assert "vllm" in _requirements_for_extra("vllm")


def test_wheel_or_install_contains_primary_and_third_party_license_files():
    files = {str(path).replace("\\", "/") for path in distribution("vane-ai").files or []}

    assert any(path.endswith("licenses/LICENSE") for path in files)
    assert any(path.endswith("licenses/NOTICE") for path in files)
    assert any(path.endswith("licenses/LICENSES/DuckDB-MIT.txt") for path in files)
    assert any(path.endswith("licenses/LICENSES/vcpkg-binary-dependencies.txt") for path in files)
    assert any(path.endswith("licenses/duckdb/experimental/spark/LICENSE") for path in files)
    assert any(path.endswith("compression/alp/algorithm/LICENSE") for path in files)
    assert any(path.endswith("compression/alprd/algorithm/LICENSE") for path in files)
