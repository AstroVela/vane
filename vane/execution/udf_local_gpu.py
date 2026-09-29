# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-device model adapter shared by explicit and native query runtimes.

The provisioned inventory is authoritative, scoped to one model registry and
uses full CUDA GPU UUIDs to avoid aliases. No CUDA import, device discovery,
VRAM enforcement or cross-process arbitration is performed here.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vane import pickle as vane_pickle
from vane.execution.resources import ResourceVector, udf_process_resources
from vane.execution.udf_model_pool import ModelPoolIdentity, ModelPoolRegistry
from vane.execution.udf_worker_metrics import WorkerMetrics

if TYPE_CHECKING:
    from vane.execution.udf_subprocess import LocalSubprocessActorPool

_GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.IGNORECASE)


def _device_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("GPU devices must be a sequence of full GPU UUIDs")
    devices = tuple(values)
    if not devices or any(type(value) is not str or not _GPU_UUID.fullmatch(value) for value in devices):
        raise ValueError("GPU devices require full GPU UUIDs; ordinals, prefixes and MIG are unsupported")
    devices = tuple("GPU-" + value[4:].lower() for value in devices)
    if len(set(devices)) != len(devices):
        raise ValueError("GPU devices must be unique")
    return devices


@dataclass(frozen=True)
class _GpuModelRegistration:
    identity: ModelPoolIdentity
    create: Callable[[], LocalSubprocessActorPool]
    resources: ResourceVector
    devices: tuple[str, ...]

    @property
    def exclusive_resources(self) -> tuple[str, ...]:
        return tuple(f"cuda:{device}" for device in self.devices)


class LocalGpuModelAdapter:
    """Bind explicit devices to fixed replicas using common resident ownership.

    All adapters using an inventory must share the same registry. Its ordinary
    acquire/prewarm/drain/close APIs own workers, exclusive device keys and
    numeric reservations together. This adapter adds no acquisition wait queue.
    """

    def __init__(self, registry: ModelPoolRegistry[LocalSubprocessActorPool], *, devices: Sequence[str]) -> None:
        self._registry = registry
        self._devices = frozenset(_device_ids(devices))

    def register(
        self,
        name: str,
        *,
        version: str,
        session_id: str,
        session_config: Mapping[str, str],
        payload: Mapping[str, Any],
        devices: Sequence[str],
        worker_metrics: WorkerMetrics | None = None,
    ) -> ModelPoolIdentity:
        registration = self.prepare_registration(
            name,
            version=version,
            session_id=session_id,
            session_config=session_config,
            payload=payload,
            devices=devices,
            worker_metrics=worker_metrics,
        )
        self._registry.register(
            registration.identity,
            registration.create,
            resources=registration.resources,
            exclusive_resources=registration.exclusive_resources,
        )
        return registration.identity

    def prepare_registration(
        self,
        name: str,
        *,
        version: str,
        session_id: str,
        session_config: Mapping[str, str],
        payload: Mapping[str, Any],
        devices: Sequence[str],
        worker_metrics: WorkerMetrics | None = None,
    ) -> _GpuModelRegistration:
        """Freeze user payloads before the caller takes its publication lock."""
        from vane.execution.udf_local_model import _model_fingerprint
        from vane.execution.udf_subprocess import LocalSubprocessActorPool, _local_actor_pool_size_from_node

        assignment = _device_ids(devices)
        if not set(assignment) <= self._devices:
            raise ValueError("model requests a GPU outside the provisioned device inventory")
        frozen = vane_pickle.dumps(dict(payload))
        snapshot = vane_pickle.loads(frozen)
        if snapshot.get("execution_backend") != "subprocess_actor":
            raise ValueError("local GPU models require subprocess_actor")
        resources = udf_process_resources(snapshot)
        if type(snapshot.get("gpus")) not in (int, float) or resources.gpu != 1:
            raise ValueError("local GPU models require exactly one GPU per replica")
        if type(snapshot.get("actor_number")) is not int or snapshot["actor_number"] <= 0:
            raise ValueError("local GPU models require a fixed positive integer actor_number")
        pool_size = _local_actor_pool_size_from_node({}, snapshot)
        if pool_size != len(assignment):
            raise ValueError("each fixed model replica requires exactly one assigned GPU")
        config = {str(key): str(value) for key, value in session_config.items()}
        identity = ModelPoolIdentity(
            session_id,
            name,
            version,
            "subprocess_actor",
            _model_fingerprint(snapshot),
            hashlib.sha256(vane_pickle.dumps((assignment, tuple(sorted(config.items()))))).hexdigest(),
        )

        def create() -> LocalSubprocessActorPool:
            return LocalSubprocessActorPool(
                vane_pickle.loads(frozen),
                pool_size,
                name=f"model-{name}-{version}",
                session_config=config,
                worker_metrics=worker_metrics,
                _gpu_devices=assignment,
            )

        return _GpuModelRegistration(
            identity,
            create,
            ResourceVector(cpu=resources.cpu * pool_size, gpu=pool_size, heap_bytes=resources.heap_bytes * pool_size),
            assignment,
        )
