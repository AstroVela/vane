# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vane.runners.ray.cluster_resource_coordinator import (
    ActorResourceBundle,
    ClusterQueryResourceCoordinator,
    NodeCapacity,
    QueryDemand,
    read_ray_node_capacities,
)
from vane.runners.ray.query_execution_graph import ResourceVector


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
    actor_bundles: tuple[ResourceVector, ...] = (),
) -> QueryDemand:
    tagged_actor_bundles = tuple(
        ActorResourceBundle(
            stage_id="stage:actor",
            actor_index=index,
            resources=bundle,
        )
        for index, bundle in enumerate(actor_bundles)
    )
    actor_total = _r()
    for bundle in actor_bundles:
        actor_total = actor_total + bundle
    task_bundles = ()
    if actor_total.fits_within(minimum):
        remainder = minimum - actor_total
        if not remainder.is_zero():
            task_bundles = (remainder,)
    return QueryDemand(
        query_id=query_id,
        minimum=minimum,
        desired=desired,
        weight=weight,
        priority=priority,
        actor_bundles=tagged_actor_bundles,
        task_bundles=task_bundles,
    )


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


def test_equal_weight_queries_receive_equal_dominant_shares():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=12, heap=1_200, store=1_200),),
        heartbeat_timeout_s=30,
    )
    demand_a = _demand(
        "a",
        minimum=_r(cpu=1, heap=100, store=100),
        desired=_r(cpu=12, heap=1_200, store=1_200),
    )
    demand_b = _demand(
        "b",
        minimum=_r(cpu=1, heap=100, store=100),
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


def test_weighted_dominant_fairness_gives_double_share_to_weight_two_query():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=12, heap=1_200, store=1_200),),
        heartbeat_timeout_s=30,
    )
    coordinator.register_query(
        _demand(
            "weight-one",
            minimum=_r(cpu=0.1, heap=10, store=10),
            desired=_r(cpu=12, heap=1_200, store=1_200),
            weight=1,
        ),
        now=0,
    )
    coordinator.register_query(
        _demand(
            "weight-two",
            minimum=_r(cpu=0.1, heap=10, store=10),
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


def test_query_desired_resources_are_downward_caps_not_capacity_overrides():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=32, gpu=4, heap=32_000, store=32_000),),
    )
    allocation = coordinator.register_query(
        _demand(
            "capped",
            minimum=_r(cpu=1, heap=100, store=100),
            desired=_r(cpu=3, heap=300, store=400),
        ),
        now=0,
    )

    assert allocation.resources == _r(cpu=3, heap=300, store=400)


def test_indivisible_gpu_actor_bundle_must_fit_one_node_not_cluster_aggregate():
    coordinator = ClusterQueryResourceCoordinator(
        (
            _node("n1", cpu=4, gpu=0.5, heap=1_000, store=1_000),
            _node("n2", cpu=4, gpu=0.5, heap=1_000, store=1_000),
        )
    )
    bundle = _r(cpu=1, gpu=1, heap=100)

    allocation = coordinator.register_query(
        _demand(
            "gpu",
            minimum=bundle,
            desired=bundle,
            actor_bundles=(bundle,),
        ),
        now=0,
    )

    assert allocation.resources.is_zero()
    assert coordinator.snapshot()["queries"]["gpu"]["state"] == "PENDING_RESOURCES"


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
    minimum = _r(cpu=2, heap=200, store=200)

    allocation = coordinator.register_query(
        _demand("coherent-task", minimum=minimum, desired=minimum),
        now=0,
    )

    assert allocation.resources.is_zero()
    assert allocation.node_allocations == ()
    assert coordinator.snapshot()["queries"]["coherent-task"]["state"] == "PENDING_RESOURCES"


def test_actor_bundle_resources_must_be_part_of_query_minimum():
    bundle = _r(cpu=2, gpu=1, heap=200)

    with pytest.raises(ValueError, match="exactly equal"):
        _demand(
            "gpu",
            minimum=_r(cpu=1, gpu=1, heap=100),
            desired=bundle,
            actor_bundles=(bundle,),
        )


def test_gpu_actor_minima_use_priority_then_fifo_without_partial_bundle_grants():
    coordinator = ClusterQueryResourceCoordinator(
        (
            _node("n1", cpu=4, gpu=1, heap=1_000, store=1_000),
            _node("n2", cpu=4, gpu=1, heap=1_000, store=1_000),
        )
    )
    bundle = _r(cpu=1, gpu=1, heap=100)
    low = _demand("low", minimum=bundle, desired=bundle, priority=0, actor_bundles=(bundle,))
    high_old = _demand("high-old", minimum=bundle, desired=bundle, priority=10, actor_bundles=(bundle,))
    high_new = _demand("high-new", minimum=bundle, desired=bundle, priority=10, actor_bundles=(bundle,))

    coordinator.register_query(high_old, now=0)
    coordinator.register_query(high_new, now=1)
    coordinator.register_query(low, now=2)
    snapshot = coordinator.snapshot()["queries"]

    assert snapshot["high-old"]["state"] == "RUNNING"
    assert snapshot["high-new"]["state"] == "RUNNING"
    assert snapshot["low"]["state"] == "PENDING_RESOURCES"
    assert sum(snapshot[query_id]["allocation"]["resources"]["gpu"] for query_id in snapshot) == 2


def test_running_gpu_actor_minimum_is_not_preempted_by_later_high_priority_query():
    coordinator = ClusterQueryResourceCoordinator((_node("n1", cpu=4, gpu=1, heap=1_000, store=1_000),))
    bundle = _r(cpu=1, gpu=1, heap=100)
    low = coordinator.register_query(
        _demand(
            "low-running",
            minimum=bundle,
            desired=bundle,
            priority=0,
            actor_bundles=(bundle,),
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
            actor_bundles=(bundle,),
        ),
        now=2,
    )
    queries = coordinator.snapshot()["queries"]

    assert low.resources == bundle
    assert high.resources.is_zero()
    assert queries["low-running"]["state"] == "RUNNING"
    assert queries["high-pending"]["state"] == "PENDING_RESOURCES"


def test_capacity_shrink_preserves_observed_usage_as_debt_and_stops_new_admission():
    coordinator = ClusterQueryResourceCoordinator(
        (_node("n1", cpu=4, heap=400, store=400),),
    )
    allocation = coordinator.register_query(
        _demand(
            "q",
            minimum=_r(cpu=1, heap=100, store=100),
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
    assert query["allocation_debt"] == _r(cpu=1, heap=100, store=100).to_dict()
    assert query["can_admit_new_tasks"] is False


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
            minimum=_r(cpu=1, heap=100, store=100),
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
            actor_bundles=(bundle,),
        ),
        now=0,
    )
    coordinator.register_query(
        _demand(
            "cpu",
            minimum=_r(cpu=1, heap=100, store=100),
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


def test_actor_placement_is_never_silently_migrated_after_node_loss():
    actor = _r(cpu=1, gpu=1, heap=100)
    coordinator = ClusterQueryResourceCoordinator(
        (
            _node("n1", cpu=2, gpu=1, heap=200),
            _node("n2", cpu=2, gpu=1, heap=200),
        )
    )
    first = coordinator.register_query(
        _demand(
            "actor-query",
            minimum=actor,
            desired=actor,
            actor_bundles=(actor,),
        )
    )
    assert first.actor_placements[0].node_id == "n1"

    coordinator.update_node_capacities((_node("n2", cpu=2, gpu=1, heap=200),))
    lost = coordinator.snapshot()["queries"]["actor-query"]

    assert lost["state"] == "ACTOR_PLACEMENT_LOST"
    assert lost["allocation"]["resources"] == _r().to_dict()
    assert lost["allocation"]["actor_placements"] == []
    assert lost["can_admit_new_tasks"] is False
    assert "no longer available" in lost["rejection_reason"]

    # Loss is terminal for this query generation. Returning capacity cannot
    # make an already-running actor-backed executor appear migrated.
    coordinator.update_node_capacities(
        (
            _node("n1", cpu=2, gpu=1, heap=200),
            _node("n2", cpu=2, gpu=1, heap=200),
        )
    )
    assert coordinator.snapshot()["queries"]["actor-query"]["state"] == "ACTOR_PLACEMENT_LOST"


def test_pinned_actor_allocation_remains_valid_when_task_minimum_is_temporarily_unavailable():
    actor = _r(cpu=1, gpu=1, heap=100)
    task = _r(cpu=1, heap=100)
    minimum = actor + task
    coordinator = ClusterQueryResourceCoordinator((_node("n1", cpu=2, gpu=1, heap=200),))
    coordinator.register_query(
        _demand(
            "actor-and-task",
            minimum=minimum,
            desired=minimum,
            actor_bundles=(actor,),
        )
    )

    coordinator.update_node_capacities((_node("n1", cpu=1, gpu=1, heap=100),))
    pending = coordinator.snapshot()["queries"]["actor-and-task"]

    assert pending["state"] == "PENDING_RESOURCES"
    assert pending["allocation"]["resources"] == actor.to_dict()
    assert pending["allocation"]["node_allocations"] == [{"node_id": "n1", "resources": actor.to_dict()}]
    assert pending["allocation"]["actor_placements"] == [{"stage_id": "stage:actor", "actor_index": 0, "node_id": "n1"}]
    assert pending["can_admit_new_tasks"] is False
