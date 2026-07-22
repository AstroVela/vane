# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vane.execution.ray_stream_adapter import TaskLeaseObjectRefGenerator
from vane.execution.udf_ray_remote_submit import _with_generator_backpressure
from vane.execution.udf_ray_stream_protocol import task_payload_with_lease


def _resolve_ref_bundle_task_refs(
    block_refs: tuple[Any, ...] | list[Any],
) -> list[Any]:
    return list(block_refs)


class RemoteUDFRefBundleMixin:
    def _submit_ref_bundle_one(
        self,
        submit_id,
        block_refs,
        slices,
        metadata,
        names,
    ):
        if not self.actors:
            raise RuntimeError("udf ray actor pool is empty")
        task_block_refs = _resolve_ref_bundle_task_refs(block_refs)

        def _submit_after_lease(task_payload: dict[str, Any]):
            self._wait_for_ready_actor()
            actor_idx, actor = self._pick_ready_actor_on_node(
                task_payload["node_id"],
                task_payload["actor_index"],
            )
            try:
                return _with_generator_backpressure(actor.run_ref_bundle_stream).remote(
                    *task_block_refs,
                    payload=task_payload,
                    slices=list(slices or []),
                    metadata=list(metadata or []),
                    names=list(names or []),
                )
            except Exception as exc:
                self._mark_actor_unavailable(actor_idx, exc)
                raise RuntimeError(f"udf ref bundle submission failed: actor_idx={actor_idx}: {exc}") from exc

        admission = self._take_task_admission()
        return TaskLeaseObjectRefGenerator(
            admission=admission,
            submitter=lambda lease: _submit_after_lease(task_payload_with_lease(self._payload, lease)),
        )

    def submit_ref_bundle_with_id(self, submit_id: int, block_refs, slices, metadata, names):
        return self._submit_ref_bundle_one(
            submit_id,
            block_refs,
            slices,
            metadata,
            names,
        )

    def submit_ref_bundle(self, _block_refs, _slices, _metadata, _names):
        raise RuntimeError("distributed Ray UDF ref-bundle submission requires submit_ref_bundle_with_id()")


__all__ = [name for name in globals() if not name.startswith("__")]
