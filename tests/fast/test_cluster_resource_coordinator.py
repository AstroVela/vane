# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from duckdb.runners.ray import cluster_resource_coordinator as coordinator_module
from duckdb.runners.ray.cluster_resource_coordinator import (
    ClusterQueryResourceCoordinator,
    NodeCapacity,
    QueryDemand,
    read_ray_node_capacities,
)
from duckdb.runners.ray.query_resource_graph import ResourceVector


def _r(
    *,
    cpu: float = 0,
    gpu: float = 0,
    heap: int = 0,
    store: int = 0,
) -> ResourceVector:
    return ResourceVector(cpu=cpu, gpu=gpu, heap_bytes=heap, object_store_bytes=store)


def _node(
    node_id: str,
    *,
    cpu: float,
    gpu: float = 0,
    heap: int = 0,
    store: int = 0,
) -> NodeCapacity:
    return NodeCapacity(node_id=node_id, resources=_r(cpu=cpu, gpu=gpu, heap=heap, store=store))


def _demand(
    query_id: str,
    *,
    minimum: ResourceVector,
    desired: ResourceVector,
    weight: float = 1,
    priority: int = 0,
) -> QueryDemand:
    return QueryDemand(
        query_id=query_id,
        minimum=minimum,
        desired=desired,
        weight=weight,
        priority=priority,
        task_bundles=() if minimum.is_zero() else (minimum,),
    )


def _inject_failure_once(monkeypatch, target, attribute, *, fail_on_call=1):
    original = getattr(target, attribute)
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_on_call:
            raise RuntimeError(f"injected {attribute} failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, attribute, fail_once)


def test_ray_capacity_uses_alive_node_resources_and_object_store_headroom():
    fake_ray = SimpleNamespace(
        nodes=lambda: [
            {
                "NodeID": "node-a",
                "Alive": True,
                "Resources": {
                    "CPU": 8,
                    "GPU": 1,
                    "memory": 10_000,
                    "object_store_memory": 20_000,
                    "node:10.0.0.1": 1,
                },
                "Labels": {"rack": "r1"},
            },
            {
                "NodeID": "node-dead",
                "Alive": False,
                "Resources": {"CPU": 64, "memory": 99_000, "object_store_memory": 99_000},
            },
            {
                "NodeID": "system-only",
                "Alive": True,
                "Resources": {"memory": 50_000, "object_store_memory": 50_000},
            },
        ]
    )

    capacities = read_ray_node_capacities(
        fake_ray,
        object_store_fraction=0.5,
        heap_reserve_bytes_per_node=1_000,
    )

    assert capacities == (
        NodeCapacity(
            node_id="node-a",
            resources=_r(cpu=8, gpu=1, heap=5_000, store=10_000),
            labels=("node:10.0.0.1", "rack=r1"),
        ),
    )


@pytest.mark.parametrize("fraction", [0, -0.1, 1.01])
def test_ray_capacity_rejects_invalid_object_store_fraction(fraction):
    with pytest.raises(ValueError, match="object_store_fraction"):
        read_ray_node_capacities(SimpleNamespace(nodes=list), object_store_fraction=fraction)


def test_ray_capacity_never_uses_host_memory_or_cpu_fallback(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: (_ for _ in ()).throw(AssertionError("host CPU accessed")))
    fake_ray = SimpleNamespace(
        nodes=lambda: [
            {
                "NodeID": "node-a",
                "Alive": True,
                "Resources": {"CPU": 2, "memory": 300, "object_store_memory": 400},
            }
        ]
    )

    capacities = read_ray_node_capacities(fake_ray)

    assert capacities[0].resources == _r(cpu=2, heap=180, store=200)


def test_soft_only_query_gets_node_allocation_without_native_hard_minimum():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("node-a", cpu=8, heap=1_000, store=400),),
    )

    allocation = coordinator.register_query(
        _demand(
            "native-only",
            minimum=_r(),
            desired=_r(store=400),
        ),
        now=0,
    )

    assert allocation.resources == _r(store=400)
    assert allocation.node_allocations[0].node_id == "node-a"
    assert allocation.node_allocations[0].resources == _r(store=400)
    assert coordinator.query_state("native-only", allocation.generation) == "RUNNING"

    with pytest.raises(ValueError, match="stale allocation generation"):
        coordinator.query_state("native-only", allocation.generation + 1)
    with pytest.raises(KeyError, match="query is not registered"):
        coordinator.query_state("missing", allocation.generation)


@pytest.mark.parametrize(
    ("target_name", "attribute", "fail_on_call"),
    [
        pytest.param("module", "_QueryState", 1, id="state-construction"),
        pytest.param("copy", "deepcopy", 1, id="state-staging"),
        pytest.param("coordinator", "_rebalance_locked", 1, id="rebalance-entry"),
        pytest.param("coordinator", "_place_bundle", 1, id="minimum-placement"),
        pytest.param("coordinator", "_weighted_drf_extras", 1, id="fair-share"),
        pytest.param("coordinator", "_place_divisible", 1, id="divisible-placement"),
        pytest.param("module", "NodeResourceAllocation", 1, id="allocation-publication"),
        pytest.param("module", "_hard_positive_difference", 2, id="debt-publication"),
    ],
)
def test_register_query_failure_is_atomic_across_rebalance_phases(
    monkeypatch,
    target_name,
    attribute,
    fail_on_call,
):
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=8, gpu=1, heap=800, store=800),),
        heartbeat_timeout_s=10,
    )
    coordinator.register_query(
        _demand(
            "existing",
            minimum=_r(cpu=1, heap=100),
            desired=_r(cpu=4, heap=400, store=400),
        ),
        now=0,
    )
    before = coordinator.snapshot()
    previous_state = coordinator._queries["existing"]
    previous_next_sequence = coordinator._next_sequence

    targets = {
        "module": coordinator_module,
        "copy": coordinator_module.copy,
        "coordinator": coordinator,
    }
    _inject_failure_once(
        monkeypatch,
        targets[target_name],
        attribute,
        fail_on_call=fail_on_call,
    )

    actor_bundle = _r(cpu=1, gpu=1, heap=100)
    with pytest.raises(RuntimeError, match=f"injected {attribute} failure"):
        coordinator.register_query(
            _demand(
                "failed",
                minimum=actor_bundle,
                desired=_r(cpu=3, gpu=1, heap=300),
            ),
            now=5,
        )

    assert coordinator.snapshot() == before
    assert coordinator._queries["existing"] is previous_state
    assert coordinator._next_sequence == previous_next_sequence

    coordinator.register_query(
        _demand(
            "recovery",
            minimum=_r(cpu=1, heap=100),
            desired=_r(cpu=3, heap=300, store=300),
        ),
        now=6,
    )
    assert coordinator._queries["recovery"].sequence == previous_next_sequence
    assert coordinator._next_sequence == previous_next_sequence + 1

    registered = coordinator.snapshot()["queries"]
    refreshed = coordinator.refresh_queries(
        observed_usage_by_query={
            "existing": _r(cpu=1, heap=100, store=100),
            "recovery": _r(cpu=1, heap=100, store=100),
        },
        generations={query_id: query["allocation"]["generation"] for query_id, query in registered.items()},
        now=7,
    )
    assert set(refreshed) == {"existing", "recovery"}
    assert coordinator.expire_queries(now=16.9) == ()
    assert coordinator.expire_queries(now=17) == ("existing", "recovery")
    assert coordinator.snapshot()["queries"] == {}


def test_equal_weight_queries_receive_equal_dominant_shares():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=12, heap=1_200, store=1_200),),
        heartbeat_timeout_s=30,
    )
    demand_a = _demand(
        "a",
        minimum=_r(cpu=1, heap=100),
        desired=_r(cpu=12, heap=1_200, store=1_200),
    )
    demand_b = _demand(
        "b",
        minimum=_r(cpu=1, heap=100),
        desired=_r(cpu=12, heap=1_200, store=1_200),
    )

    coordinator.register_query(demand_a, now=0)
    coordinator.register_query(demand_b, now=0)
    snapshot = coordinator.snapshot()

    allocation_a = ResourceVector.from_dict(snapshot["queries"]["a"]["allocation"]["resources"])
    allocation_b = ResourceVector.from_dict(snapshot["queries"]["b"]["allocation"]["resources"])
    total = _r(cpu=12, heap=1_200, store=1_200)
    assert allocation_a.dominant_share(total) == pytest.approx(0.5, abs=0.01)
    assert allocation_b.dominant_share(total) == pytest.approx(0.5, abs=0.01)
    assert allocation_a + allocation_b == total


def test_weighted_drf_accounts_for_unequal_hard_minimum_shares():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=10, heap=1_000),),
        heartbeat_timeout_s=30,
    )
    coordinator.register_query(
        _demand(
            "large-minimum",
            minimum=_r(cpu=8, heap=800),
            desired=_r(cpu=10, heap=1_000),
        ),
        now=0,
    )
    coordinator.register_query(
        _demand(
            "small-minimum",
            minimum=_r(cpu=1, heap=100),
            desired=_r(cpu=10, heap=1_000),
        ),
        now=0,
    )

    queries = coordinator.snapshot()["queries"]
    large = ResourceVector.from_dict(queries["large-minimum"]["allocation"]["resources"])
    small = ResourceVector.from_dict(queries["small-minimum"]["allocation"]["resources"])

    assert large == _r(cpu=8, heap=800)
    assert small == _r(cpu=2, heap=200)


def test_weighted_drf_fills_non_dominant_minimum_plateau_without_changing_fair_share():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=10, heap=1_000),),
        heartbeat_timeout_s=30,
    )
    coordinator.register_query(
        _demand(
            "cpu-heavy",
            minimum=_r(cpu=8),
            desired=_r(cpu=8, heap=1_000),
        ),
        now=0,
    )
    coordinator.register_query(
        _demand(
            "heap-progress",
            minimum=_r(cpu=1),
            desired=_r(cpu=1, heap=1_000),
        ),
        now=0,
    )

    queries = coordinator.snapshot()["queries"]
    cpu_heavy = ResourceVector.from_dict(queries["cpu-heavy"]["allocation"]["resources"])
    heap_progress = ResourceVector.from_dict(queries["heap-progress"]["allocation"]["resources"])

    assert cpu_heavy == _r(cpu=8, heap=200)
    assert heap_progress == _r(cpu=1, heap=800)


def test_weighted_dominant_fairness_gives_double_share_to_weight_two_query():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=12, heap=1_200, store=1_200),),
        heartbeat_timeout_s=30,
    )
    coordinator.register_query(
        _demand(
            "weight-one",
            minimum=_r(cpu=0.1, heap=10),
            desired=_r(cpu=12, heap=1_200, store=1_200),
            weight=1,
        ),
        now=0,
    )
    coordinator.register_query(
        _demand(
            "weight-two",
            minimum=_r(cpu=0.1, heap=10),
            desired=_r(cpu=12, heap=1_200, store=1_200),
            weight=2,
        ),
        now=0,
    )

    queries = coordinator.snapshot()["queries"]
    one = ResourceVector.from_dict(queries["weight-one"]["allocation"]["resources"])
    two = ResourceVector.from_dict(queries["weight-two"]["allocation"]["resources"])

    assert two.cpu / one.cpu == pytest.approx(2.0, rel=0.03)
    assert two.heap_bytes / one.heap_bytes == pytest.approx(2.0, rel=0.03)
    assert two.object_store_bytes / one.object_store_bytes == pytest.approx(2.0, rel=0.03)


def test_object_store_budget_remains_fair_when_hard_minima_exhaust_cpu():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=2, heap=200, store=1_000),),
        heartbeat_timeout_s=30,
    )
    for query_id in ("a", "b"):
        coordinator.register_query(
            _demand(
                query_id,
                minimum=_r(cpu=1, heap=100),
                desired=_r(cpu=1, heap=100, store=1_000),
            ),
            now=0,
        )

    queries = coordinator.snapshot()["queries"]
    allocation_a = ResourceVector.from_dict(queries["a"]["allocation"]["resources"])
    allocation_b = ResourceVector.from_dict(queries["b"]["allocation"]["resources"])

    assert allocation_a == _r(cpu=1, heap=100, store=500)
    assert allocation_b == _r(cpu=1, heap=100, store=500)


def test_query_desired_resources_are_downward_caps_not_capacity_overrides():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=32, gpu=4, heap=32_000, store=32_000),),
    )
    allocation = coordinator.register_query(
        _demand(
            "capped",
            minimum=_r(cpu=1, heap=100),
            desired=_r(cpu=3, heap=300, store=400),
        ),
        now=0,
    )

    assert allocation.resources == _r(cpu=3, heap=300, store=400)


def test_query_demand_rejects_object_store_hard_minimum_and_task_bundles():
    with pytest.raises(ValueError, match="minimum query resources may not hard-reserve object-store bytes"):
        QueryDemand(
            query_id="minimum-store",
            minimum=_r(cpu=1, store=1),
            desired=_r(cpu=1, store=10),
            task_bundles=(_r(cpu=1, store=1),),
        )

    with pytest.raises(ValueError, match="task resource bundles may not hard-reserve object-store bytes"):
        QueryDemand(
            query_id="task-store",
            minimum=_r(cpu=1),
            desired=_r(cpu=1, store=10),
            task_bundles=(_r(cpu=1, store=1),),
        )


def test_query_demand_allocates_gpu_headroom_as_a_soft_divisible_share():
    minimum = _r(cpu=1, gpu=1, heap=100)
    desired = _r(cpu=4, gpu=4, heap=400)
    coordinator = ClusterQueryResourceCoordinator(
        (_node("gpu-node", cpu=4, gpu=4, heap=400),),
    )

    allocation = coordinator.register_query(
        _demand(
            "elastic-gpu",
            minimum=minimum,
            desired=desired,
        ),
        now=0,
    )

    assert allocation.resources == desired


def test_gpu_task_bundle_receives_divisible_soft_headroom():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("gpu-node", cpu=4, gpu=1, heap=400, store=400),),
    )
    fixed_gpu_bundle = _r(cpu=1, gpu=1, heap=100)
    desired = _r(cpu=4, gpu=1, heap=400, store=400)

    allocation = coordinator.register_query(
        _demand(
            "fixed-gpu",
            minimum=fixed_gpu_bundle,
            desired=desired,
        ),
        now=0,
    )

    assert allocation.resources == desired


def test_indivisible_gpu_task_bundle_must_fit_one_node_not_cluster_aggregate():
    coordinator = ClusterQueryResourceCoordinator(
        (
            _node("n1", cpu=4, gpu=0.5, heap=1_000, store=1_000),
            _node("n2", cpu=4, gpu=0.5, heap=1_000, store=1_000),
        )
    )
    bundle = _r(cpu=1, gpu=1, heap=100)

    allocation = coordinator.register_query(
        _demand("gpu-task", minimum=bundle, desired=bundle),
        now=0,
    )

    assert allocation.resources.is_zero()
    assert coordinator.snapshot()["queries"]["gpu-task"]["state"] == "PENDING_RESOURCES"


def test_minimum_task_vector_must_fit_one_node_not_cross_node_dimensions():
    coordinator = ClusterQueryResourceCoordinator(
        (
            _node("cpu-node", cpu=2, heap=1, store=1),
            _node("memory-node", cpu=0, heap=199, store=199),
        )
    )
    minimum = _r(cpu=2, heap=200)

    allocation = coordinator.register_query(
        _demand("coherent-task", minimum=minimum, desired=minimum),
        now=0,
    )

    assert allocation.resources.is_zero()
    assert allocation.node_allocations == ()
    assert coordinator.snapshot()["queries"]["coherent-task"]["state"] == "PENDING_RESOURCES"


def test_gpu_task_minima_use_priority_then_fifo():
    coordinator = ClusterQueryResourceCoordinator(
        (
            _node("n1", cpu=4, gpu=1, heap=1_000, store=1_000),
            _node("n2", cpu=4, gpu=1, heap=1_000, store=1_000),
        )
    )
    bundle = _r(cpu=1, gpu=1, heap=100)
    low = _demand("low", minimum=bundle, desired=bundle, priority=0)
    high_old = _demand("high-old", minimum=bundle, desired=bundle, priority=10)
    high_new = _demand("high-new", minimum=bundle, desired=bundle, priority=10)

    coordinator.register_query(high_old, now=0)
    coordinator.register_query(high_new, now=1)
    coordinator.register_query(low, now=2)
    snapshot = coordinator.snapshot()["queries"]

    assert snapshot["high-old"]["state"] == "RUNNING"
    assert snapshot["high-new"]["state"] == "RUNNING"
    assert snapshot["low"]["state"] == "PENDING_RESOURCES"
    assert sum(snapshot[query_id]["allocation"]["resources"]["gpu"] for query_id in snapshot) == 2


def test_live_usage_becomes_debt_when_priority_reassigns_soft_allocation():
    coordinator = ClusterQueryResourceCoordinator((_node("n1", cpu=4, gpu=1, heap=1_000, store=1_000),))
    bundle = _r(cpu=1, gpu=1, heap=100)
    low = coordinator.register_query(
        _demand(
            "low-running",
            minimum=bundle,
            desired=bundle,
            priority=0,
        ),
        now=0,
    )
    low = coordinator.refresh_query(
        "low-running",
        observed_usage=bundle,
        generation=low.generation,
        now=1,
    )

    high = coordinator.register_query(
        _demand(
            "high-pending",
            minimum=bundle,
            desired=bundle,
            priority=100,
        ),
        now=2,
    )
    queries = coordinator.snapshot()["queries"]

    assert low.resources == bundle
    assert high.resources == bundle
    assert queries["low-running"]["state"] == "ALLOCATION_DEBT"
    assert queries["low-running"]["allocation"]["resources"] == _r().to_dict()
    assert queries["low-running"]["allocation_debt"] == bundle.to_dict()
    assert queries["high-pending"]["state"] == "RUNNING"


def test_capacity_shrink_preserves_observed_usage_as_debt_and_stops_new_admission():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=4, heap=400, store=400),),
    )
    allocation = coordinator.register_query(
        _demand(
            "q",
            minimum=_r(cpu=1, heap=100),
            desired=_r(cpu=4, heap=400, store=400),
        ),
        now=0,
    )
    coordinator.refresh_query(
        "q",
        observed_usage=_r(cpu=3, heap=300, store=300),
        generation=allocation.generation,
        now=1,
    )

    coordinator.update_node_capacities((_node("n1", cpu=2, heap=200, store=200),), now=2)
    query = coordinator.snapshot()["queries"]["q"]

    assert query["allocation"]["resources"] == _r(cpu=2, heap=200, store=200).to_dict()
    assert query["observed_usage"] == _r(cpu=3, heap=300, store=300).to_dict()
    assert query["allocation_debt"] == _r(cpu=1, heap=100).to_dict()
    assert query["soft_object_store_debt_bytes"] == 100
    assert query["can_admit_new_tasks"] is False


def test_object_store_overage_is_soft_debt_and_does_not_close_query_admission():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=4, heap=400, store=400),),
    )
    allocation = coordinator.register_query(
        _demand(
            "q",
            minimum=_r(cpu=1, heap=100),
            desired=_r(cpu=4, heap=400, store=400),
        ),
        now=0,
    )

    coordinator.refresh_query(
        "q",
        observed_usage=_r(cpu=1, heap=100, store=450),
        generation=allocation.generation,
        now=1,
    )
    query = coordinator.snapshot()["queries"]["q"]

    assert query["state"] == "RUNNING"
    assert query["allocation_debt"] == _r().to_dict()
    assert query["soft_object_store_debt_bytes"] == 50
    assert query["can_admit_new_tasks"] is True


def test_stale_generation_cannot_refresh_or_release_newer_query_lease():
    coordinator = ClusterQueryResourceCoordinator((_node("n1", cpu=4, heap=400, store=400),))
    first = coordinator.register_query(
        _demand("q", minimum=_r(cpu=1, heap=100), desired=_r(cpu=4, heap=400)),
        now=0,
    )
    second = coordinator.refresh_query(
        "q",
        observed_usage=_r(cpu=1, heap=100),
        generation=first.generation,
        now=1,
    )

    with pytest.raises(ValueError, match="stale allocation generation"):
        coordinator.refresh_query(
            "q",
            observed_usage=_r(),
            generation=first.generation,
            now=2,
        )
    assert coordinator.release_query("q", first.generation) is False
    assert coordinator.release_query("q", second.generation) is True
    assert coordinator.snapshot()["queries"] == {}


def test_heartbeat_expiry_reclaims_query_allocation_idempotently():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=4, heap=400, store=400),),
        heartbeat_timeout_s=10,
    )
    coordinator.register_query(
        _demand("q", minimum=_r(cpu=1, heap=100), desired=_r(cpu=4, heap=400)),
        now=5,
    )

    assert coordinator.expire_queries(now=14.9) == ()
    assert coordinator.expire_queries(now=15) == ("q",)
    assert coordinator.expire_queries(now=100) == ()
    assert coordinator.snapshot()["queries"] == {}


def test_refresh_queries_updates_all_usage_and_heartbeats_atomically():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=8, heap=800, store=800),),
        heartbeat_timeout_s=10,
    )

    def demand(query_id):
        return _demand(
            query_id,
            minimum=_r(cpu=1, heap=100),
            desired=_r(cpu=8, heap=800, store=800),
        )

    coordinator.register_query(demand("a"), now=0)
    coordinator.register_query(demand("b"), now=0)
    before = coordinator.snapshot()["queries"]

    allocations = coordinator.refresh_queries(
        observed_usage_by_query={
            "a": _r(cpu=2, heap=200, store=150),
            "b": _r(cpu=1, heap=120, store=100),
        },
        generations={query_id: query["allocation"]["generation"] for query_id, query in before.items()},
        now=5,
    )

    after = coordinator.snapshot()["queries"]
    assert set(allocations) == {"a", "b"}
    assert len({allocation.generation for allocation in allocations.values()}) == 1
    assert after["a"]["observed_usage"] == _r(cpu=2, heap=200, store=150).to_dict()
    assert after["b"]["observed_usage"] == _r(cpu=1, heap=120, store=100).to_dict()
    assert coordinator.expire_queries(now=14.9) == ()
    assert coordinator.expire_queries(now=15) == ("a", "b")


def test_refresh_queries_rejects_stale_batch_without_partial_mutation():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=8, heap=800, store=800),),
        heartbeat_timeout_s=10,
    )

    def demand(query_id):
        return _demand(
            query_id,
            minimum=_r(cpu=1, heap=100),
            desired=_r(cpu=8, heap=800),
        )

    coordinator.register_query(demand("a"), now=0)
    coordinator.register_query(demand("b"), now=0)
    before = coordinator.snapshot()["queries"]

    with pytest.raises(ValueError, match="stale allocation generation"):
        coordinator.refresh_queries(
            observed_usage_by_query={"a": _r(cpu=2), "b": _r(cpu=3)},
            generations={
                "a": before["a"]["allocation"]["generation"],
                "b": before["b"]["allocation"]["generation"] - 1,
            },
            now=5,
        )

    after = coordinator.snapshot()["queries"]
    assert after["a"]["observed_usage"] == before["a"]["observed_usage"]
    assert after["b"]["observed_usage"] == before["b"]["observed_usage"]
    assert coordinator.expire_queries(now=10) == ("a", "b")


def test_node_allocations_never_exceed_any_node_capacity():
    coordinator = ClusterQueryResourceCoordinator(
        (
            _node("n1", cpu=2, gpu=1, heap=200, store=300),
            _node("n2", cpu=4, gpu=0, heap=500, store=600),
        )
    )
    bundle = _r(cpu=1, gpu=1, heap=100)
    coordinator.register_query(
        _demand(
            "gpu",
            minimum=bundle,
            desired=_r(cpu=3, gpu=1, heap=300, store=300),
        ),
        now=0,
    )
    coordinator.register_query(
        _demand(
            "cpu",
            minimum=_r(cpu=1, heap=100),
            desired=_r(cpu=4, heap=400, store=600),
        ),
        now=0,
    )

    snapshot = coordinator.snapshot()
    used_by_node = {node_id: _r() for node_id in snapshot["nodes"]}
    for query in snapshot["queries"].values():
        for node_id, payload in query["node_allocations"].items():
            used_by_node[node_id] = used_by_node[node_id] + ResourceVector.from_dict(payload)
    for node_id, payload in snapshot["nodes"].items():
        assert used_by_node[node_id].fits_within(ResourceVector.from_dict(payload["resources"]))
