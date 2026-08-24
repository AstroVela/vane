# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pickle
import sys
import types
import uuid

import pytest

import vane


def _require_ray_cxx():
    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")
    return ray_cxx


def _create_s3_secret(connection, name, scope, *, value_sentinel):
    connection.execute(
        f"""
        CREATE SECRET {name} (
            TYPE S3,
            PROVIDER CONFIG,
            KEY_ID 'key-{name}',
            SECRET '{value_sentinel}',
            SCOPE '{scope}'
        )
        """
    )


def _new_uuid8():
    identity = bytearray(uuid.uuid4().bytes)
    identity[6] = (identity[6] & 0x0F) | 0x80
    identity[8] = (identity[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(identity)))


def _logical_plan_with_uses(connection, query_id, *, source_uris=(), sink_uris=()):
    ray_cxx = _require_ray_cxx()
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql("SELECT 1 AS value"), query_id)
    ray_cxx._attach_scoped_secret_uses_for_test(plan, source_uris, sink_uris)
    return plan


def _restore_logical_plan(ray_cxx, state):
    restored = ray_cxx.PyLogicalPlan.__new__(ray_cxx.PyLogicalPlan)
    restored.__setstate__(tuple(state))
    return restored


def _capture_write_relation(
    monkeypatch,
    connection,
    target,
    *,
    query="SELECT 1 AS value",
    **write_options,
):
    captured = []

    class FakeRayRunner:
        def run_write(self, relation, **_kwargs):
            captured.append(relation)
            return {"ok": True}

    runners = types.ModuleType("vane.runners")
    runners.set_runner_ray = lambda *_args, **_kwargs: FakeRayRunner()
    monkeypatch.setitem(sys.modules, "vane.runners", runners)
    monkeypatch.setenv("VANE_RUNNER", "ray")

    connection.sql(query).write_parquet(target, **write_options)
    assert len(captured) == 1
    return captured[0]


def test_copy_sink_discovers_only_duckdb_longest_prefix_secret(monkeypatch):
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "broad_sink_secret",
        "s3://scoped-secret-bucket/",
        value_sentinel="unused-broad-secret-value",
    )
    _create_s3_secret(
        connection,
        "narrow_sink_secret",
        "s3://scoped-secret-bucket/narrow/",
        value_sentinel="selected-narrow-secret-value",
    )
    relation = _capture_write_relation(
        monkeypatch,
        connection,
        "s3://scoped-secret-bucket/narrow/result.parquet",
    )

    plan = ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, "scoped-secret-copy-sink")
    references = plan.scoped_secret_refs()

    assert len(references) == 1
    assert references[0]["version"] == 1
    assert uuid.UUID(references[0]["reference_id"]).version == 8
    assert references[0]["owner_query_id"] == "scoped-secret-copy-sink"
    assert references[0]["owner_session_id"] == plan.session_id()
    assert references[0]["type"] == "s3"
    assert references[0]["provider"] == "config"
    assert references[0]["scope"] == "s3://scoped-secret-bucket/narrow/"
    assert references[0]["capabilities"] == ["write"]


def test_two_scopes_are_canonical_and_capabilities_are_aggregated():
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "scope_b_secret",
        "s3://scope-b/",
        value_sentinel="scope-b-value-sentinel",
    )
    _create_s3_secret(
        connection,
        "scope_a_secret",
        "s3://scope-a/",
        value_sentinel="scope-a-value-sentinel",
    )
    plan = _logical_plan_with_uses(
        connection,
        "two-scoped-secret-refs",
        source_uris=("s3://scope-b/input.parquet", "s3://scope-a/input.parquet"),
        sink_uris=("s3://scope-a/output.parquet",),
    )

    references = plan.scoped_secret_refs()
    assert [reference["scope"] for reference in references] == ["s3://scope-a/", "s3://scope-b/"]
    assert references[0]["capabilities"] == ["read", "write"]
    assert references[1]["capabilities"] == ["read"]
    assert len({reference["reference_id"] for reference in references}) == 2

    same_capture = _logical_plan_with_uses(
        connection,
        "two-scoped-secret-refs",
        source_uris=("s3://scope-b/input.parquet", "s3://scope-a/input.parquet"),
        sink_uris=("s3://scope-a/output.parquet",),
    )
    assert same_capture.scoped_secret_refs() == references

    restored = pickle.loads(pickle.dumps(plan))
    assert restored.scoped_secret_refs() == references


def test_object_store_secret_type_precedence_matches_httpfs_reader():
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "s3_broad_secret",
        "s3://",
        value_sentinel="s3-priority-sentinel",
    )
    connection.execute(
        """
        CREATE SECRET r2_narrow_secret (
            TYPE R2,
            PROVIDER CONFIG,
            KEY_ID 'r2-key',
            SECRET 'r2-lower-priority-sentinel',
            ACCOUNT_ID 'account-id',
            SCOPE 's3://priority-bucket/narrow/'
        )
        """
    )
    plan = _logical_plan_with_uses(
        connection,
        "scoped-secret-type-priority",
        source_uris=("s3://priority-bucket/narrow/input.parquet",),
    )

    assert plan.scoped_secret_refs() == [
        {
            "version": 1,
            "reference_id": plan.scoped_secret_refs()[0]["reference_id"],
            "owner_query_id": "scoped-secret-type-priority",
            "owner_session_id": plan.session_id(),
            "type": "s3",
            "provider": "config",
            "scope": "s3://",
            "capabilities": ["read"],
        }
    ]


def test_selected_http_secret_for_object_store_uri_is_rejected_before_ray():
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    connection.execute(
        """
        CREATE SECRET object_store_http_secret (
            TYPE HTTP,
            EXTRA_HTTP_HEADERS MAP {'X-Secret-Sentinel': 'must-not-reach-ray'},
            SCOPE 's3://http-secret-scope/'
        )
        """
    )

    uri_sentinel = "uri-secret-value-must-not-reach-errors"
    with pytest.raises(Exception, match="do not support HTTP secret selections") as exc_info:
        _logical_plan_with_uses(
            connection,
            "unsupported-http-scoped-secret",
            source_uris=(f"s3://http-secret-scope/input.parquet?secret={uri_sentinel}",),
        )
    assert uri_sentinel not in str(exc_info.value)


def test_s3_secret_precedes_a_matching_http_secret_when_merge_is_enabled():
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "s3_precedes_http_secret",
        "s3://s3-http-precedence/",
        value_sentinel="s3-precedence-value",
    )
    connection.execute(
        """
        CREATE SECRET lower_priority_http_secret (
            TYPE HTTP,
            EXTRA_HTTP_HEADERS MAP {'X-Secret-Sentinel': 'lower-priority'},
            SCOPE 's3://s3-http-precedence/'
        )
        """
    )
    connection.execute("SET merge_http_secret_into_s3_request = true")

    plan = _logical_plan_with_uses(
        connection,
        "s3-http-secret-precedence",
        source_uris=("s3://s3-http-precedence/input.parquet",),
    )

    assert len(plan.scoped_secret_refs()) == 1
    assert plan.scoped_secret_refs()[0]["type"] == "s3"


def test_disabled_http_secret_merge_does_not_create_an_unused_reference():
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    connection.execute(
        """
        CREATE SECRET unused_object_store_http_secret (
            TYPE HTTP,
            EXTRA_HTTP_HEADERS MAP {'X-Secret-Sentinel': 'unused'},
            SCOPE 's3://unused-http-secret-scope/'
        )
        """
    )
    connection.execute("SET merge_http_secret_into_s3_request = false")

    plan = _logical_plan_with_uses(
        connection,
        "disabled-http-secret-merge",
        source_uris=("s3://unused-http-secret-scope/input.parquet",),
    )

    assert plan.scoped_secret_refs() == []


def test_previously_unmatched_use_rejects_a_new_secret_before_transport():
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    uri_sentinel = "unmatched-uri-sentinel-must-not-reach-errors"
    plan = _logical_plan_with_uses(
        connection,
        "unmatched-use-becomes-secret-backed",
        source_uris=(f"s3://new-secret-after-capture/input.parquet?token={uri_sentinel}",),
    )
    assert plan.scoped_secret_refs() == []

    _create_s3_secret(
        connection,
        "created_after_plan_capture",
        "s3://new-secret-after-capture/",
        value_sentinel="new-secret-value-must-not-reach-errors",
    )

    with pytest.raises(Exception, match="previously unmatched") as pickle_error:
        pickle.dumps(plan)
    assert uri_sentinel not in str(pickle_error.value)

    with pytest.raises(Exception, match="previously unmatched") as physical_error:
        plan.to_physical_plan(vane.connect())
    assert uri_sentinel not in str(physical_error.value)


def test_unchanged_unmatched_use_can_cross_the_transport_boundary():
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    unmatched_uri = "s3://still-unmatched/source-only-uri-sentinel.parquet"
    plan = _logical_plan_with_uses(
        connection,
        "unchanged-unmatched-use",
        source_uris=(unmatched_uri,),
    )

    payload = pickle.dumps(plan)
    restored = pickle.loads(payload)

    assert unmatched_uri.encode() not in payload
    assert restored.scoped_secret_refs() == []


def test_scoped_secret_uses_reject_oversized_and_nul_uris_without_echoing_them():
    connection = vane.connect()
    connection.execute("LOAD httpfs")

    oversized_sentinel = "oversized-read-uri-sentinel"
    oversized_uri = f"s3://oversized-read/{'x' * 65536}/{oversized_sentinel}"
    with pytest.raises(Exception, match="scoped secret use size limit") as oversized_error:
        _logical_plan_with_uses(
            connection,
            "oversized-read-secret-use",
            source_uris=(oversized_uri,),
        )
    assert oversized_sentinel not in str(oversized_error.value)

    nul_sentinel = "nul-read-uri-sentinel"
    with pytest.raises(Exception, match="invalid NUL byte") as nul_error:
        _logical_plan_with_uses(
            connection,
            "nul-read-secret-use",
            source_uris=(f"s3://nul-read/input\0{nul_sentinel}.parquet",),
        )
    assert nul_sentinel not in str(nul_error.value)


def test_scoped_secret_uses_bound_aggregate_uri_bytes():
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    source_uris = tuple(f"s3://aggregate-use-budget/{'y' * 65400}/{uri_idx:04d}.parquet" for uri_idx in range(257))

    with pytest.raises(Exception, match="maximum scoped secret URI byte size"):
        _logical_plan_with_uses(
            connection,
            "aggregate-scoped-secret-use-budget",
            source_uris=source_uris,
        )


@pytest.mark.parametrize(
    ("query", "write_options", "nested_suffix"),
    [
        pytest.param("SELECT 1 AS value", {}, "reserved/", id="single-file"),
        pytest.param(
            "SELECT 1 AS value",
            {"per_thread_output": True},
            "reserved/",
            id="per-thread",
        ),
        pytest.param(
            "SELECT i AS value FROM range(4) tbl(i)",
            {"file_size_bytes": 1024},
            "reserved/",
            id="rotation",
        ),
        pytest.param(
            "SELECT 'US' AS country, 1 AS value",
            {"partition_by": ["country"]},
            "country=US/",
            id="partitioned",
        ),
    ],
)
def test_generated_copy_modes_reject_a_nested_secret_selection_before_ray(
    monkeypatch,
    query,
    write_options,
    nested_suffix,
):
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "generated_copy_broad_secret",
        "s3://generated-copy/",
        value_sentinel="generated-copy-broad-value",
    )
    nested_name = "generated_copy_nested_secret_sentinel"
    nested_value = "generated-copy-nested-value-sentinel"
    _create_s3_secret(
        connection,
        nested_name,
        f"s3://generated-copy/output/{nested_suffix}",
        value_sentinel=nested_value,
    )
    relation = _capture_write_relation(
        monkeypatch,
        connection,
        "s3://generated-copy/output",
        query=query,
        **write_options,
    )

    with pytest.raises(Exception, match="generated output paths can select different scoped secrets") as exc_info:
        ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, "nested-copy-secret-selection")
    assert nested_name not in str(exc_info.value)
    assert nested_value not in str(exc_info.value)


def test_generated_copy_rejects_a_secret_missing_from_the_commit_namespace(monkeypatch):
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "data_namespace_only_secret",
        "s3://copy-sidecar/output/",
        value_sentinel="data-namespace-only-value",
    )
    relation = _capture_write_relation(
        monkeypatch,
        connection,
        "s3://copy-sidecar/output",
    )

    with pytest.raises(Exception, match="generated output paths can select different scoped secrets"):
        ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, "copy-commit-namespace-mismatch")


def test_generated_copy_rejects_an_unmatched_base_with_a_nested_secret(monkeypatch):
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "nested_without_base_secret",
        "s3://copy-unmatched-base/output/country=US/",
        value_sentinel="nested-without-base-value",
    )
    relation = _capture_write_relation(
        monkeypatch,
        connection,
        "s3://copy-unmatched-base/output",
        query="SELECT 'US' AS country, 1 AS value",
        partition_by=["country"],
    )

    with pytest.raises(Exception, match="generated output paths can select different scoped secrets"):
        ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, "copy-unmatched-base")


def test_generated_copy_rejects_non_prefix_secret_matching_semantics(monkeypatch):
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "non_prefix_copy_broad_secret",
        "s3://non-prefix-copy/",
        value_sentinel="non-prefix-copy-broad-value",
    )
    custom_name = "non_prefix_matching_secret_sentinel"
    ray_cxx._register_non_prefix_scoped_secret_for_test(connection, custom_name)
    relation = _capture_write_relation(
        monkeypatch,
        connection,
        "s3://non-prefix-copy/output",
    )

    with pytest.raises(Exception, match="standard longest-prefix matching semantics") as exc_info:
        ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, "non-prefix-copy-secret")
    assert custom_name not in str(exc_info.value)


def test_generated_copy_rejects_an_empty_nonstandard_lookup_storage(monkeypatch):
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "nonstandard_storage_copy_broad_secret",
        "s3://nonstandard-storage-copy/",
        value_sentinel="nonstandard-storage-copy-broad-value",
    )
    ray_cxx._register_nonstandard_empty_secret_storage_for_test(connection)
    relation = _capture_write_relation(
        monkeypatch,
        connection,
        "s3://nonstandard-storage-copy/output",
    )

    with pytest.raises(Exception, match="standard longest-prefix matching semantics"):
        ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, "nonstandard-empty-secret-storage")


def test_generated_copy_rejects_excessive_secret_scope_boundaries(monkeypatch):
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "copy_boundary_limit_broad_secret",
        "s3://copy-boundary-limit/",
        value_sentinel="copy-boundary-limit-broad-value",
    )
    scope_name_sentinel = "copy_boundary_limit_nested_secret"
    scope_value_sentinel = "copy-boundary-limit-nested-value"
    nested_scopes = ", ".join(
        f"'s3://copy-boundary-limit/output/boundary={boundary_idx}/'" for boundary_idx in range(4097)
    )
    connection.execute(
        f"""
        CREATE SECRET {scope_name_sentinel} (
            TYPE S3,
            PROVIDER CONFIG,
            KEY_ID 'key-{scope_name_sentinel}',
            SECRET '{scope_value_sentinel}',
            SCOPE [{nested_scopes}]
        )
        """
    )
    relation = _capture_write_relation(
        monkeypatch,
        connection,
        "s3://copy-boundary-limit/output",
    )

    with pytest.raises(Exception, match="secret metadata validation limit") as exc_info:
        ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, "copy-secret-boundary-limit")
    assert scope_name_sentinel not in str(exc_info.value)
    assert scope_value_sentinel not in str(exc_info.value)


def test_generated_copy_rejects_an_oversized_namespace_without_echoing_it(monkeypatch):
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    path_sentinel = "oversized-copy-path-sentinel"
    target = f"s3://oversized-copy/{'x' * 65536}/{path_sentinel}"
    relation = _capture_write_relation(monkeypatch, connection, target)

    with pytest.raises(Exception, match="scoped secret path size limit") as exc_info:
        ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, "oversized-copy-namespace")
    assert path_sentinel not in str(exc_info.value)


def test_generated_copy_canonicalization_ignores_siblings_and_lower_priority_secrets(monkeypatch):
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "canonical_copy_s3_secret",
        "s3://copy-canonical/",
        value_sentinel="canonical-copy-s3-value",
    )
    _create_s3_secret(
        connection,
        "sibling_copy_s3_secret",
        "s3://copy-canonical/output2/",
        value_sentinel="sibling-copy-value",
    )
    connection.execute(
        """
        CREATE SECRET nested_lower_priority_r2_secret (
            TYPE R2,
            PROVIDER CONFIG,
            KEY_ID 'r2-key',
            SECRET 'lower-priority-r2-value',
            ACCOUNT_ID 'account-id',
            SCOPE 's3://copy-canonical/output/reserved/'
        )
        """
    )
    relation = _capture_write_relation(
        monkeypatch,
        connection,
        "s3://copy-canonical/output///",
    )

    plan = ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, "canonical-copy-scope")

    assert len(plan.scoped_secret_refs()) == 1
    assert plan.scoped_secret_refs()[0]["type"] == "s3"
    assert plan.scoped_secret_refs()[0]["scope"] == "s3://copy-canonical/"


def test_generated_copy_namespace_rejects_a_new_nested_secret_before_transport(monkeypatch):
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "generated_copy_stable_secret",
        "s3://generated-copy-stale/",
        value_sentinel="generated-copy-stable-value",
    )
    relation = _capture_write_relation(
        monkeypatch,
        connection,
        "s3://generated-copy-stale/output",
        query="SELECT 'US' AS country, 1 AS value",
        partition_by=["country"],
    )
    plan = ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, "stale-generated-copy-namespace")
    assert len(plan.scoped_secret_refs()) == 1

    nested_name = "generated_copy_created_after_capture"
    nested_value = "generated-copy-new-nested-value"
    _create_s3_secret(
        connection,
        nested_name,
        "s3://generated-copy-stale/output/country=US/",
        value_sentinel=nested_value,
    )

    with pytest.raises(Exception, match="generated output paths can select different scoped secrets") as pickle_error:
        pickle.dumps(plan)
    assert nested_name not in str(pickle_error.value)
    assert nested_value not in str(pickle_error.value)

    with pytest.raises(Exception, match="generated output paths can select different scoped secrets") as physical_error:
        plan.to_physical_plan(vane.connect())
    assert nested_name not in str(physical_error.value)
    assert nested_value not in str(physical_error.value)


def test_generated_copy_commit_namespace_rejects_a_new_secret_before_transport(monkeypatch):
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "commit_freshness_broad_secret",
        "s3://copy-commit-freshness/",
        value_sentinel="commit-freshness-broad-value",
    )
    relation = _capture_write_relation(
        monkeypatch,
        connection,
        "s3://copy-commit-freshness/output",
    )
    plan = ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, "stale-copy-commit-namespace")

    _create_s3_secret(
        connection,
        "commit_freshness_nested_secret",
        "s3://copy-commit-freshness/output.duckdb_commit/",
        value_sentinel="commit-freshness-nested-value",
    )

    with pytest.raises(Exception, match="generated output paths can select different scoped secrets"):
        pickle.dumps(plan)
    with pytest.raises(Exception, match="generated output paths can select different scoped secrets"):
        plan.to_physical_plan(vane.connect())


def test_secret_values_and_names_are_absent_from_logical_and_physical_pickles():
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    secret_name = "opaque_name_sentinel_588"
    secret_value = "opaque-value-sentinel-588-keep-out-of-plans"
    _create_s3_secret(
        connection,
        secret_name,
        "s3://sentinel-scope/",
        value_sentinel=secret_value,
    )
    logical_plan = _logical_plan_with_uses(
        connection,
        "scoped-secret-sentinel-plan",
        source_uris=("s3://sentinel-scope/input.parquet",),
    )

    logical_payload = pickle.dumps(logical_plan)
    assert secret_name.encode() not in logical_payload
    assert secret_value.encode() not in logical_payload

    transported_logical = pickle.loads(logical_payload)
    physical_plan = transported_logical.to_physical_plan(vane.connect())
    physical_payload = pickle.dumps(physical_plan)
    assert secret_name.encode() not in physical_payload
    assert secret_value.encode() not in physical_payload
    assert physical_plan.scoped_secret_refs() == logical_plan.scoped_secret_refs()


def test_source_binding_rejects_a_stale_secret_before_transport():
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "stale_source_secret",
        "s3://stale-source/",
        value_sentinel="stale-source-value",
    )
    plan = _logical_plan_with_uses(
        connection,
        "stale-scoped-secret-ref",
        source_uris=("s3://stale-source/input.parquet",),
    )
    connection.execute("DROP SECRET stale_source_secret")

    with pytest.raises(Exception, match="is stale because DuckDB now selects a different secret"):
        pickle.dumps(plan)
    with pytest.raises(Exception, match="is stale because DuckDB now selects a different secret"):
        plan.to_physical_plan(vane.connect())


@pytest.mark.parametrize(
    ("field_index", "replacement", "message"),
    [
        (0, 2, "Unsupported scoped secret reference version"),
        (1, str(uuid.uuid4()), "invalid opaque ID"),
        (2, "different-query", "stale or belongs to a different query"),
        (3, "different-session", "belongs to a different Vane session"),
        (4, "http", "unsupported type/provider"),
        (5, "unsupported-provider", "unsupported type/provider"),
        (7, 0, "invalid capabilities"),
    ],
)
def test_logical_pickle_rejects_invalid_scoped_secret_contract(field_index, replacement, message):
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "invalid_contract_secret",
        "s3://invalid-contract/",
        value_sentinel="invalid-contract-value",
    )
    plan = _logical_plan_with_uses(
        connection,
        "invalid-scoped-secret-contract",
        source_uris=("s3://invalid-contract/input.parquet",),
    )
    state = list(plan.__getstate__())
    references = [list(reference) for reference in state[4]]
    references[0][field_index] = replacement
    state[4] = tuple(tuple(reference) for reference in references)

    with pytest.raises(Exception, match=message):
        _restore_logical_plan(ray_cxx, state)


def test_logical_pickle_rejects_ambiguous_and_noncanonical_references():
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "scope_a_contract_secret",
        "s3://contract-a/",
        value_sentinel="contract-a-value",
    )
    _create_s3_secret(
        connection,
        "scope_b_contract_secret",
        "s3://contract-b/",
        value_sentinel="contract-b-value",
    )
    plan = _logical_plan_with_uses(
        connection,
        "canonical-scoped-secret-contract",
        source_uris=("s3://contract-a/input.parquet", "s3://contract-b/input.parquet"),
    )
    state = list(plan.__getstate__())

    noncanonical_state = list(state)
    noncanonical_state[4] = tuple(reversed(state[4]))
    with pytest.raises(Exception, match="not in canonical order"):
        _restore_logical_plan(ray_cxx, noncanonical_state)

    duplicate = list(state[4][0])
    duplicate[1] = _new_uuid8()
    ambiguous_references = [state[4][0], tuple(duplicate)]
    ambiguous_references.sort(
        key=lambda reference: (reference[4], reference[5], reference[6], reference[1], reference[7])
    )
    ambiguous_state = list(state)
    ambiguous_state[4] = tuple(ambiguous_references)
    with pytest.raises(Exception, match="is ambiguous"):
        _restore_logical_plan(ray_cxx, ambiguous_state)


def test_logical_pickle_bounds_reference_bytes_with_long_common_prefixes():
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "reference_byte_budget_secret",
        "s3://reference-byte-budget/",
        value_sentinel="reference-byte-budget-value",
    )
    plan = _logical_plan_with_uses(
        connection,
        "reference-byte-budget",
        source_uris=("s3://reference-byte-budget/input.parquet",),
    )
    base_state = list(plan.__getstate__())
    reference_template = list(base_state[4][0])

    def state_with_scopes(scopes):
        references = []
        for scope in scopes:
            reference = list(reference_template)
            reference[1] = _new_uuid8()
            reference[6] = scope
            references.append(tuple(reference))
        references.sort(key=lambda ref: (ref[4], ref[5], ref[6], ref[1], ref[7]))
        state = list(base_state)
        state[4] = tuple(references)
        return state

    common_prefix = f"s3://reference-byte-budget/{'x' * 32768}"
    accepted_state = state_with_scopes(f"{common_prefix}/{scope_idx:04d}/" for scope_idx in range(128))
    restored = _restore_logical_plan(ray_cxx, accepted_state)
    assert len(restored.scoped_secret_refs()) == 128

    oversized_prefix = f"s3://reference-byte-budget/{'y' * 65400}"
    oversized_state = state_with_scopes(f"{oversized_prefix}/{scope_idx:04d}/" for scope_idx in range(257))
    with pytest.raises(Exception, match="maximum serialized byte size"):
        _restore_logical_plan(ray_cxx, oversized_state)


def test_refs_survive_physical_clone_query_replay_and_worker_task():
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("LOAD httpfs")
    _create_s3_secret(
        connection,
        "worker_replay_secret",
        "s3://worker-replay/",
        value_sentinel="worker-replay-value",
    )
    query_id = "scoped-secret-worker-replay"
    logical_plan = _logical_plan_with_uses(
        connection,
        query_id,
        source_uris=("s3://worker-replay/input.parquet",),
    )
    expected_references = logical_plan.scoped_secret_refs()
    physical_plan = pickle.loads(pickle.dumps(logical_plan)).to_physical_plan(vane.connect())
    physical_plan = pickle.loads(pickle.dumps(physical_plan))
    cloned_plan = physical_plan.clone(vane.connect())

    assert physical_plan.scoped_secret_refs() == expected_references
    assert cloned_plan.scoped_secret_refs() == expected_references

    try:
        assert ray_cxx._register_query_python_replay_state(query_id, physical_plan) is True
        assert ray_cxx._register_query_python_replay_state(query_id, physical_plan) is False
        assert ray_cxx._lookup_query_scoped_secret_refs(query_id) == physical_plan.__getstate__()[7]

        task = ray_cxx._make_worker_task_from_plan_for_test(
            cloned_plan,
            f"{query_id}:worker-task",
            query_id,
        )
        worker_plan = task.plan()
        assert worker_plan.resource_query_id() == query_id
        assert worker_plan.scoped_secret_refs() == expected_references
        assert worker_plan._validate_serializable_for_submission() is None
    finally:
        ray_cxx._cleanup_query_python_replay_state(query_id)
