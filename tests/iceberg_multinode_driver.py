# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Container-side probes using installed Vane/providers and real Ray workers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import socket
import sys
import time
from importlib.metadata import entry_points, version
from pathlib import Path

# Only qualification helpers are copied here; Vane always resolves from the
# installed wheel. The container has no checkout or host dataset mount.
sys.path.insert(0, str(Path(__file__).parent))


def emit(value):
    print("VANE_QUALIFICATION=" + json.dumps(value), flush=True)


def start_node(head):
    address = socket.gethostbyname(socket.gethostname())
    options = ["ray", "start", "--node-ip-address=" + address, "--num-cpus=" + ("0" if head else "2"), "--block"]
    options += ["--head", "--port=6379", "--include-dashboard=false"] if head else ["--address=driver:6379"]
    os.execvp(options[0], options)


def topology(ray):
    return sorted(
        [
            {"id": node["NodeID"], "ip": node["NodeManagerAddress"], "cpu": node["Resources"].get("CPU", 0)}
            for node in ray.nodes()
            if node["Alive"]
        ],
        key=lambda node: node["ip"],
    )


def provider_inventory():
    artifacts = []
    for entry in entry_points(group="vane.dynamic_extension_providers"):
        if entry.name in {"avro", "iceberg"}:
            for artifact in entry.load()()._artifact_by_identity.values():
                descriptor = artifact.descriptor.to_dict()
                assert hashlib.sha256(artifact.path.read_bytes()).hexdigest() == descriptor["sha256"]
                artifacts.append(descriptor)
    return sorted(artifacts, key=lambda item: item["name"])


def connect(request):
    import vane

    return vane.connect(
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "allow_persistent_secrets": "false",
            "secret_directory": "/run/qualification/secrets",
            "s3_endpoint": request["s3_endpoint"],
            "s3_region": "us-east-1",
            "s3_url_style": "path",
            "s3_use_ssl": "false",
            "s3_access_key_id": "",
            "s3_secret_access_key": "",
            "s3_session_token": "",
            "http_retries": "0",
        }
    )


def freeze_queries(request):
    from iceberg_s3_helpers import attach
    from iceberg_scan_helpers import freeze

    directory = Path("/run/qualification/frozen")
    directory.mkdir(exist_ok=True)
    with connect(request) as con:
        attach(con, request["endpoint"])
        for name, sql in request["queries"].items():
            assert name.replace("_", "").isalnum()
            (directory / (name + ".pickle")).write_bytes(freeze(con, sql))
    return {"frozen": len(request["queries"])}


def payload(name):
    assert name.replace("_", "").isalnum()
    return (Path("/run/qualification/frozen") / (name + ".pickle")).read_bytes()


def rows_from_parts(parts):
    return [row for table in parts for row in zip(*(column.to_pylist() for column in table.columns), strict=True)]


def execute_queries(request, *, public):
    from iceberg_s3_helpers import attach

    from vane.runners.ray.runner import RayRunner

    runner = None
    try:
        if public:
            with connect(request) as con:
                catalog = attach(con, request["endpoint"])
                relation = catalog.table(request["namespace"], request["name"])
                assert relation._get_runner_type() == "ray"
                names = relation.columns
                label = "renamed_label" if "renamed_label" in names else "label"
                result = {
                    "rows": relation.order("id").fetchall(),
                    "projection": relation.project("id").order("id").fetchall(),
                    "filtered": relation.filter(f"id % 4 = 1 OR {label} IS NULL").order("id").fetchall(),
                    "aggregate": relation.aggregate("count(*), sum(id)::BIGINT").fetchall(),
                }
                if request.get("snapshot_id"):
                    result["history"] = (
                        catalog.table(request["namespace"], request["name"], snapshot_id=request["snapshot_id"])
                        .order("id")
                        .fetchall()
                    )
                return result
        runner = RayRunner(address=None, max_task_backlog=None)
        return {
            mode: rows_from_parts(list(runner.run_iter_tables(pickle.loads(payload(name)))))
            for mode, name in request["queries"].items()
        }
    finally:
        if runner is not None:
            runner.close()


def physical(name):
    import vane

    # This independent cursor has no storage config. The transported bootstrap
    # resolves and owns the actual planning/execution connection.
    worker = vane.connect(
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "allow_persistent_secrets": "false",
        }
    )
    logical = pickle.loads(payload(name))
    plan = logical.to_physical_plan(worker)
    node, batches = next(iter(plan.scan_split_batch_map().items()))
    return worker, logical, plan, str(node), [bytes(batch) for batch in batches]


def lifecycle(request, ray):
    from iceberg_ray_helpers import drain_query, native_workers, wait_for_open_scan
    from ray_native_scan_helpers import NativeScanWorkerTask, register_scan_query

    import vane
    from vane.runners.ray import worker_handle as handles

    action = request["action"]
    worker_nodes = [node["id"] for node in topology(ray) if node["cpu"] > 0]
    assert len(worker_nodes) == 2
    worker, logical, plan, node, batches = physical(request["query"])
    try:
        with native_workers(2, nodes=worker_nodes) as (query_id, actors, workers):
            if action == "disjoint":
                tasks = [
                    NativeScanWorkerTask(
                        query_id=query_id,
                        node_id=node,
                        fragment_node_id=str(index),
                        plan=plan,
                        split_batch=vane.ray_cxx.merge_scan_split_batches(batches[index::2]),
                    )
                    for index in range(2)
                ]
                register_scan_query(tasks)
                attempts = [
                    attempt
                    for handle, task in zip(workers, tasks, strict=True)
                    for attempt in handle.submit_tasks([task])
                ]
                workers[0].task_input_stream_exhausted_for_query(query_id, [node])
                rows, owners, _selected = drain_query(query_id, workers, attempts)
                assert owners == {handle.worker_id for handle in workers}
            else:
                task = NativeScanWorkerTask(
                    query_id=query_id,
                    node_id=node,
                    plan=plan,
                    split_batch=vane.ray_cxx.merge_scan_split_batches(batches),
                )
                register_scan_query([task])
                attempt = workers[0].submit_tasks([task])[0]
                workers[0].pop_fte_result_handles(query_id)
                wait_for_open_scan(actors[0], attempt)
                if action == "cancel":
                    attempt.cancel()
                    assert (
                        ray.get(actors[0].fte_get_task_status.remote(attempt.task_id.to_dict()))["state"] == "CANCELED"
                    )
                    rows, owners = [], set()
                else:
                    if action == "node_failure":
                        emit({"phase": "stop_node", "node_id": worker_nodes[0]})
                    else:
                        ray.kill(actors[0], no_restart=True)
                    try:
                        asyncio.run(asyncio.wait_for(attempt.get_result(), timeout=120))
                    except Exception:
                        pass
                    else:
                        raise AssertionError("Controlled worker failure must fail its initial attempt")
                    workers[0].wait_fte_worker_failure_reconciliation(timeout_s=120)
                    retries = workers[1].pop_fte_result_handles(query_id)
                    workers[1].task_input_stream_exhausted_for_query(query_id, [node])
                    rows, owners, selected = drain_query(query_id, workers, retries)
                    assert owners == {workers[1].worker_id}
                    assert str(attempt.task_id) not in selected
                    assert str(attempt.task_id).rsplit(".", 1)[0] + ".1" in selected
            for index, (actor, handle) in enumerate(zip(actors, workers, strict=True)):
                handle.fte_drop_query(query_id)
                if action in {"retry", "node_failure"} and index == 0:
                    continue
                assert ray.get(actor.stats_fragments.remote())["fragments_total"] == 0
                ray.get(actor.close_session.remote(str(logical.session_id())))
                assert ray.get(actor.stats_snapshot_databases.remote())["active_databases"] == 0
            assert query_id not in handles.fte_registry_stats()["queries"]
            assert not any(owner[0] == query_id for owner in handles._FTE_PARTITION_OWNERS)
            return {"rows": rows, "owners": sorted(owners), "worker_nodes": worker_nodes, "cleaned": True}
    finally:
        plan = logical = None
        worker.close()


def probe_nodes(request, ray):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=0)
    def probe(canary, execute):
        from pathlib import Path

        import ray
        from iceberg_s3_helpers import execute_frozen

        result = {
            "node_id": ray.get_runtime_context().get_node_id(),
            "hostname": socket.gethostname(),
            "ip": socket.gethostbyname(socket.gethostname()),
            "mount_namespace": os.readlink("/proc/self/ns/mnt"),
            "canary_exists": Path(canary).exists(),
            "vane_version": version("vane-ai"),
            "ray_version": version("ray"),
            "python_version": sys.version.split()[0],
            "writer_credentials_in_env": any(
                os.environ.get(name) for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
            ),
            "providers": provider_inventory(),
        }
        if execute is not None:
            try:
                result["rows"] = execute_frozen(execute, None)[0]
            except Exception as error:
                result["error"] = str(error)
        return result

    data = payload(request["query"]) if request.get("query") else None
    return ray.get(
        [
            probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["id"], soft=False)).remote(
                request.get("canary", "/run/qualification/driver-only-canary"), data
            )
            for node in topology(ray)
            if node["cpu"] > 0
        ]
    )


def main(request):
    if request["action"] == "freeze":
        return freeze_queries(request)
    import ray

    import vane

    ray.init(address="auto", log_to_driver=False)
    try:
        action = request["action"]
        if action == "topology":
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                nodes = topology(ray)
                if (
                    len(nodes) == 3
                    and {node["ip"] for node in nodes} == set(request["ips"])
                    and not {node["id"] for node in nodes}.intersection(request.get("excluded_node_ids", []))
                ):
                    break
                time.sleep(0.2)
            nodes = topology(ray)
            assert len(nodes) == 3, nodes
            assert {node["ip"] for node in nodes} == set(request["ips"]), nodes
            assert not {node["id"] for node in nodes}.intersection(request.get("excluded_node_ids", [])), nodes
            return {"nodes": nodes, "version": version("vane-ai"), "providers": provider_inventory()}
        if action == "probe":
            return probe_nodes(request, ray)
        if action in {"public", "execute"}:
            return execute_queries(request, public=action == "public")
        return lifecycle(request, ray)
    finally:
        vane.teardown_runner()
        ray.shutdown()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"head", "worker"}:
        start_node(sys.argv[1] == "head")
    else:
        emit({"result": main(json.load(sys.stdin))})
