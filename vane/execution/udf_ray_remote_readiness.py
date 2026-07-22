# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vane.execution.udf_ray_actor_state import (
    format_stateful_actor_loss as _format_stateful_actor_loss,
)


class RemoteUDFActorReadinessMixin:
    def _mark_actor_ready(self, actor_idx: int) -> None:
        if actor_idx in self._ready_actor_set:
            return
        self._ready_actor_set.add(actor_idx)
        self._ready_actor_indices.append(actor_idx)
        self._actor_init_errors.pop(actor_idx, None)
        self._actors_obj._confirmed_ready.add(actor_idx)
        with self._ready_refs_cv:
            self._ready_refs_cv.notify_all()

    def _notify_ready_ref_ready(self, ref: Any) -> None:
        with self._ready_refs_cv:
            if ref in self._pending_ready_refs and ref not in self._ready_probe_ref_set:
                self._ready_probe_refs.append(ref)
                self._ready_probe_ref_set.add(ref)
            self._ready_refs_cv.notify_all()

    def _fail_ready_ref_registration(self, ready_ref: Any, actor_idx: int, exc: Exception) -> None:
        with self._ready_refs_cv:
            if self._pending_ready_refs.get(ready_ref) == actor_idx:
                self._pending_ready_refs.pop(ready_ref, None)
            self._actor_init_errors[actor_idx] = exc
            self._ready_refs_cv.notify_all()

    def _register_ready_ref_wakeup(self, ready_ref: Any, actor_idx: int) -> None:
        try:
            future = ready_ref.future()
        except Exception as exc:
            self._fail_ready_ref_registration(
                ready_ref,
                actor_idx,
                TypeError(f"Ray ready ObjectRef does not support completion callbacks: {exc}"),
            )
            return
        try:
            future.add_done_callback(lambda _, _ref=ready_ref: self._notify_ready_ref_ready(_ref))
        except Exception as exc:
            self._fail_ready_ref_registration(ready_ref, actor_idx, exc)

    def _track_ready_ref(self, actor_idx: int, ready_ref: Any) -> None:
        try:
            hash(ready_ref)
        except Exception as exc:
            self._actor_init_errors[actor_idx] = TypeError(f"Ray ready ObjectRef is not hashable: {exc}")
            with self._ready_refs_cv:
                self._ready_refs_cv.notify_all()
            return
        with self._ready_refs_cv:
            if actor_idx in self._ready_actor_set:
                return
            if actor_idx in self._pending_ready_refs.values():
                return
            self._pending_ready_refs[ready_ref] = actor_idx
        self._register_ready_ref_wakeup(ready_ref, actor_idx)

    def _enqueue_ready_probe(self, actor_idx: int) -> None:
        if actor_idx in self._ready_actor_set:
            return

        init_refs = self._actors_obj._init_refs
        if actor_idx < len(init_refs):
            ready_ref = init_refs[actor_idx]
            if ready_ref is None:
                self._actor_init_errors[actor_idx] = "ray actor init_payload did not return a readiness ObjectRef"
                with self._ready_refs_cv:
                    self._ready_refs_cv.notify_all()
                return
            self._track_ready_ref(actor_idx, ready_ref)
            return

        actor = self.actors[actor_idx]
        ready_method = getattr(actor, "__ray_ready__", None)
        if ready_method is None:
            self._actor_init_errors[actor_idx] = "ray actor does not expose __ray_ready__ readiness probe"
            with self._ready_refs_cv:
                self._ready_refs_cv.notify_all()
            return
        try:
            ready_ref = ready_method.remote()
        except Exception as exc:
            self._actor_init_errors[actor_idx] = exc
            return
        if ready_ref is None:
            self._actor_init_errors[actor_idx] = "__ray_ready__ did not return a readiness ObjectRef"
            with self._ready_refs_cv:
                self._ready_refs_cv.notify_all()
            return
        self._track_ready_ref(actor_idx, ready_ref)

    def _prime_actor_readiness(self) -> None:
        confirmed = self._actors_obj._confirmed_ready
        for actor_idx in range(len(self.actors)):
            if actor_idx in confirmed:
                self._mark_actor_ready(actor_idx)
            else:
                self._enqueue_ready_probe(actor_idx)

    def _refresh_actor_readiness(self) -> None:
        ready_items: list[tuple[Any, int]] = []
        with self._ready_refs_cv:
            while self._ready_probe_refs:
                ready_ref = self._ready_probe_refs.popleft()
                self._ready_probe_ref_set.discard(ready_ref)
                actor_idx = self._pending_ready_refs.pop(ready_ref, None)
                if actor_idx is not None:
                    ready_items.append((ready_ref, actor_idx))
        for ready_ref, actor_idx in ready_items:
            if actor_idx is None:
                continue
            try:
                self._resolve_object_ref(ready_ref)
            except Exception as exc:
                self._actor_init_errors[actor_idx] = exc
                continue
            self._mark_actor_ready(actor_idx)

    def _wait_for_ready_actor(self) -> None:
        self._refresh_actor_readiness()
        if self._ready_actor_indices:
            return

        for error in self._actor_init_errors.values():
            if not isinstance(error, BaseException):
                continue
            formatted = _format_stateful_actor_loss(self.error_context(), error)
            if formatted is not error:
                raise formatted from error

        err_preview = ", ".join(f"{idx}:{msg}" for idx, msg in sorted(self._actor_init_errors.items())[:3])
        if err_preview:
            err_preview = f" init_errors=[{err_preview}]"
        raise RuntimeError(
            "udf ray actor pool has no ready actors: "
            f"total={len(self.actors)} ready=0 pending_ready={len(self._pending_ready_refs)}.{err_preview}"
        )

    def _pick_ready_actor_on_node(self, node_id: str, actor_index: int) -> tuple[int, Any]:
        node_key = str(node_id).strip()
        if not node_key:
            raise RuntimeError("UDF actor invocation requires its leased Ray node_id")
        if not self._actor_node_ids:
            raise RuntimeError("UDF actor pool is missing coordinator-confirmed node identities")
        actor_idx = int(actor_index)
        if actor_idx < 0 or actor_idx >= len(self.actors):
            raise RuntimeError(f"UDF actor lease has invalid actor_index={actor_idx}")
        if self._actor_node_ids[actor_idx] != node_key:
            raise RuntimeError(
                "UDF actor lease slot/node mismatch: "
                f"actor_index={actor_idx} expected_node={self._actor_node_ids[actor_idx]} leased_node={node_key}"
            )
        if actor_idx not in self._ready_actor_set:
            raise RuntimeError(f"UDF actor lease targets actor_index={actor_idx}, but that actor is not ready")
        return actor_idx, self.actors[actor_idx]

    def _mark_actor_unavailable(self, actor_idx: int, exc: Exception) -> None:
        if actor_idx in self._ready_actor_set:
            self._ready_actor_set.remove(actor_idx)
            self._ready_actor_indices = [idx for idx in self._ready_actor_indices if idx != actor_idx]
        self._actors_obj._confirmed_ready.discard(actor_idx)
        self._actor_init_errors[actor_idx] = exc
        with self._ready_refs_cv:
            self._ready_refs_cv.notify_all()


__all__ = [name for name in globals() if not name.startswith("__")]
