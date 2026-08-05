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
    desired: ResourceVector,
    weight: float = 1,
    priority: int = 0,
) -> QueryDemand:
    return QueryDemand(
        query_id=query_id,
        desired=desired,
        weight=weight,
        priority=priority,
    )


def _inject_failure_once(monkeypatch, target, attribute):
    original = getattr(target, attribute)
    called = False

    def fail_once(*args, **kwargs):
        nonlocal called
        if not called:
            called = True
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


def test_single_query_gets_aggregate_soft_budget_without_node_placement():
    coordinator = ClusterQueryResourceCoordinator(
        (
            _node("node-a", cpu=4, gpu=1, heap=400, store=300),
            _node("node-b", cpu=2, heap=200, store=100),
        )
    )

    allocation = coordinator.register_query(
        _demand("q", desired=_r(cpu=8, gpu=2, heap=800, store=800)),
        now=0,
    )

    assert allocation.resources == _r(cpu=6, gpu=1, heap=600, store=400)
    assert allocation.to_dict() == {
        "resources": _r(cpu=6, gpu=1, heap=600, store=400).to_dict(),
        "generation": allocation.generation,
    }
    assert coordinator.query_state("q", allocation.generation) == "RUNNING"


def test_zero_cluster_capacity_keeps_query_running_with_zero_soft_budget():
    coordinator = ClusterQueryResourceCoordinator(())

    allocation = coordinator.register_query(
        _demand("q", desired=_r(cpu=1, gpu=1, heap=100, store=100)),
        now=0,
    )

    assert allocation.resources.is_zero()
    assert coordinator.query_state("q", allocation.generation) == "RUNNING"
    assert coordinator.snapshot()["queries"]["q"]["state"] == "RUNNING"


def test_query_shape_is_not_bin_packed_or_pre_admitted():
    coordinator = ClusterQueryResourceCoordinator(
        (
            _node("gpu-a", cpu=4, gpu=1, store=100),
            _node("gpu-b", cpu=4, gpu=1, store=100),
        )
    )

    allocation = coordinator.register_query(
        _demand("needs-two-gpus-per-task", desired=_r(cpu=1, gpu=2, store=200)),
        now=0,
    )

    # No current node can run a 2-GPU task. The coordinator still publishes an
    # aggregate soft budget and lets the real Ray request remain pending so the
    # autoscaler can observe it.
    assert allocation.resources == _r(cpu=1, gpu=2, store=200)
    assert coordinator.query_state("needs-two-gpus-per-task", allocation.generation) == "RUNNING"


def test_equal_weight_queries_share_every_contended_dimension():
    coordinator = ClusterQueryResourceCoordinator((_node("n", cpu=8, gpu=2, heap=800, store=801),))
    first = coordinator.register_query(
        _demand("first", desired=_r(cpu=8, gpu=2, heap=800, store=801)),
        now=0,
    )
    second = coordinator.register_query(
        _demand("second", desired=_r(cpu=8, gpu=2, heap=800, store=801)),
        now=0,
    )

    snapshot = coordinator.snapshot()["queries"]
    first_resources = snapshot["first"]["allocation"]["resources"]
    second_resources = snapshot["second"]["allocation"]["resources"]
    assert first_resources == _r(cpu=4, gpu=1, heap=400, store=401).to_dict()
    assert second_resources == _r(cpu=4, gpu=1, heap=400, store=400).to_dict()
    assert first.generation < second.generation


def test_weighted_queries_receive_weighted_max_min_shares():
    coordinator = ClusterQueryResourceCoordinator((_node("n", cpu=9, gpu=3, heap=900, store=900),))
    coordinator.register_query(
        _demand("one", desired=_r(cpu=9, gpu=3, heap=900, store=900), weight=1),
        now=0,
    )
    coordinator.register_query(
        _demand("two", desired=_r(cpu=9, gpu=3, heap=900, store=900), weight=2),
        now=0,
    )

    allocations = {
        query_id: payload["allocation"]["resources"] for query_id, payload in coordinator.snapshot()["queries"].items()
    }
    assert allocations["one"] == _r(cpu=3, gpu=1, heap=300, store=300).to_dict()
    assert allocations["two"] == _r(cpu=6, gpu=2, heap=600, store=600).to_dict()


def test_noncompeting_resource_demands_each_use_the_full_dimension():
    coordinator = ClusterQueryResourceCoordinator((_node("n", cpu=8, gpu=2, heap=800, store=600),))
    coordinator.register_query(
        _demand("cpu", desired=_r(cpu=8, heap=800)),
        now=0,
    )
    coordinator.register_query(
        _demand("gpu-store", desired=_r(gpu=2, store=600)),
        now=0,
    )

    queries = coordinator.snapshot()["queries"]
    assert queries["cpu"]["allocation"]["resources"] == _r(cpu=8, heap=800).to_dict()
    assert queries["gpu-store"]["allocation"]["resources"] == _r(gpu=2, store=600).to_dict()


def test_capacity_shrink_reduces_budget_but_records_only_soft_debt():
    coordinator = ClusterQueryResourceCoordinator((_node("n", cpu=4, heap=400, store=400),))
    allocation = coordinator.register_query(
        _demand("q", desired=_r(cpu=4, heap=400, store=400)),
        now=0,
    )
    allocation = coordinator.refresh_query(
        "q",
        observed_usage=_r(cpu=3, heap=300, store=250),
        generation=allocation.generation,
        now=1,
    )

    coordinator.update_node_capacities((_node("n", cpu=1, heap=100, store=100),))
    query = coordinator.snapshot()["queries"]["q"]

    assert query["state"] == "RUNNING"
    assert query["allocation"]["resources"] == _r(cpu=1, heap=100, store=100).to_dict()
    assert query["soft_allocation_debt"] == _r(cpu=2, heap=200, store=150).to_dict()
    assert coordinator.query_state("q", query["allocation"]["generation"]) == "RUNNING"


def test_capacity_recovery_expands_zero_budget_without_reregistering():
    coordinator = ClusterQueryResourceCoordinator(())
    allocation = coordinator.register_query(
        _demand("q", desired=_r(cpu=2, heap=200, store=300)),
        now=0,
    )

    coordinator.update_node_capacities((_node("new", cpu=2, heap=200, store=300),))
    query = coordinator.snapshot()["queries"]["q"]

    assert query["allocation"]["generation"] > allocation.generation
    assert query["allocation"]["resources"] == _r(cpu=2, heap=200, store=300).to_dict()


def test_refresh_queries_is_atomic_and_can_update_phase_demand():
    coordinator = ClusterQueryResourceCoordinator((_node("n", cpu=4, heap=400, store=400),))
    first = coordinator.register_query(_demand("first", desired=_r(cpu=4, heap=400)), now=0)
    second = coordinator.register_query(_demand("second", desired=_r(store=400)), now=0)

    before = coordinator.snapshot()
    with pytest.raises(ValueError, match="generation sets must match"):
        coordinator.refresh_queries(
            observed_usage_by_query={"first": _r()},
            generations={"second": second.generation},
            now=1,
        )
    assert coordinator.snapshot() == before

    generations = {query_id: payload["allocation"]["generation"] for query_id, payload in before["queries"].items()}
    refreshed = coordinator.refresh_queries(
        observed_usage_by_query={"first": _r(), "second": _r()},
        generations=generations,
        demands_by_query={
            "first": _demand("first", desired=_r(store=400)),
            "second": _demand("second", desired=_r(cpu=4, heap=400)),
        },
        now=2,
    )

    assert refreshed["first"].resources == _r(store=400)
    assert refreshed["second"].resources == _r(cpu=4, heap=400)
    assert first.generation < refreshed["first"].generation


def test_generation_fencing_heartbeat_expiry_and_release():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n", cpu=1),),
        heartbeat_timeout_s=10,
    )
    allocation = coordinator.register_query(_demand("q", desired=_r(cpu=1)), now=0)

    with pytest.raises(ValueError, match="stale allocation generation"):
        coordinator.heartbeat("q", allocation.generation + 1, now=1)
    current = coordinator.heartbeat("q", allocation.generation, now=5)
    assert coordinator.expire_queries(now=14.9) == ()
    assert coordinator.expire_queries(now=15) == ("q",)
    assert coordinator.release_query("q", current.generation) is False

    replacement = coordinator.register_query(_demand("q", desired=_r(cpu=1)), now=20)
    assert coordinator.release_query("q", replacement.generation + 1) is False
    assert coordinator.release_query("q", replacement.generation) is True


@pytest.mark.parametrize(
    ("target_name", "attribute"),
    [
        pytest.param("module", "_QueryState", id="state-construction"),
        pytest.param("copy", "deepcopy", id="state-staging"),
        pytest.param("coordinator", "_rebalance_locked", id="rebalance"),
    ],
)
def test_register_query_failure_is_atomic(monkeypatch, target_name, attribute):
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n", cpu=8, gpu=1, heap=800, store=800),),
        heartbeat_timeout_s=10,
    )
    coordinator.register_query(
        _demand("existing", desired=_r(cpu=4, heap=400, store=400)),
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
    _inject_failure_once(monkeypatch, targets[target_name], attribute)

    with pytest.raises(RuntimeError, match=f"injected {attribute} failure"):
        coordinator.register_query(
            _demand("failed", desired=_r(cpu=3, gpu=1, heap=300)),
            now=5,
        )

    assert coordinator.snapshot() == before
    assert coordinator._queries["existing"] is previous_state
    assert coordinator._next_sequence == previous_next_sequence
    assert "failed" not in coordinator._queries


def test_query_demand_rejects_removed_hard_admission_fields():
    with pytest.raises(TypeError, match="minimum"):
        QueryDemand(
            query_id="q",
            desired=_r(cpu=1),
            minimum=_r(cpu=1),
        )
