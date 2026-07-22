# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types

from vane.runners.ray import fragment_registry as _fragment_registry
from vane.runners.ray import fragment_worker_client as _fragment_worker_client
from vane.runners.ray import fte_events as _fte_events
from vane.runners.ray import fte_fragment_scheduler as _fte_fragment_scheduler
from vane.runners.ray import worker_pool as _worker_pool

_ORIGINAL_WORKER_POOL_START_RAY_WORKERS = _worker_pool.start_ray_workers
_ORIGINAL_WORKER_POOL_TRY_AUTOSCALE = _worker_pool.try_autoscale


def _export_module_symbols(module: object) -> None:
    names = getattr(module, "__all__", None)
    if names is None:
        module_name = getattr(module, "__name__", None)
        names = [
            name
            for name, value in vars(module).items()
            if not name.startswith("_") and getattr(value, "__module__", None) == module_name
        ]
    for name in names:
        if name.startswith("__"):
            continue
        globals()[name] = getattr(module, name)


for _module in (
    _fragment_registry,
    _fte_events,
    _fte_fragment_scheduler,
    _fragment_worker_client,
    _worker_pool,
):
    _export_module_symbols(_module)


def _sync_worker_pool_overrides() -> None:
    for name in (
        "RayWorkerActor",
        "RayWorkerActorHandle",
        "RayWorkerRuntime",
        "_collect_vane_env_overrides",
        "_is_ray_worker_context",
        "ray",
        "resolve_object_refs_blocking",
    ):
        if name in globals():
            setattr(_worker_pool, name, globals()[name])


def start_ray_workers(existing_worker_ids):
    _sync_worker_pool_overrides()
    return _worker_pool.start_ray_workers(existing_worker_ids)


def try_autoscale(bundles):
    _sync_worker_pool_overrides()
    return _worker_pool.try_autoscale(bundles)


_FACADE_START_RAY_WORKERS = start_ray_workers
_FACADE_TRY_AUTOSCALE = try_autoscale


class _WorkerHandleFacadeModule(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "start_ray_workers" and value is _FACADE_START_RAY_WORKERS:
            setattr(_worker_pool, name, _ORIGINAL_WORKER_POOL_START_RAY_WORKERS)
            return
        if name == "try_autoscale" and value is _FACADE_TRY_AUTOSCALE:
            setattr(_worker_pool, name, _ORIGINAL_WORKER_POOL_TRY_AUTOSCALE)
            return
        for module in (
            _fragment_registry,
            _fte_events,
            _fte_fragment_scheduler,
            _fragment_worker_client,
            _worker_pool,
        ):
            if hasattr(module, name):
                setattr(module, name, value)


sys.modules[__name__].__class__ = _WorkerHandleFacadeModule


__all__ = [
    name
    for name in globals()
    if not name.startswith("__")
    and name
    not in {
        "_module",
        "_export_module_symbols",
        "_sync_worker_pool_overrides",
        "_WorkerHandleFacadeModule",
    }
]
