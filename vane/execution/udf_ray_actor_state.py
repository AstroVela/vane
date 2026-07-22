# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


def get_actor_handle_id_hex(actor: Any) -> str | None:
    actor_id = getattr(actor, "_actor_id", None)
    if actor_id is None:
        return None
    hex_attr = getattr(actor_id, "hex", None)
    if callable(hex_attr):
        try:
            return str(hex_attr())
        except Exception:
            return None
    if hex_attr is not None:
        return str(hex_attr)
    return None


def build_stateful_actor_error_context(
    payload: dict[str, Any],
    actor_handles: list[Any],
) -> dict[str, Any] | None:
    if not payload.get("stateful"):
        return None
    if len(actor_handles) != 1:
        raise RuntimeError(
            "stateful UDF runtime requires exactly one actor handle; multi-actor state semantics are not defined"
        )
    return {
        "stateful": True,
        "udf_name": str(payload.get("udf_name") or "udf"),
        "actor_id": get_actor_handle_id_hex(actor_handles[0]) or "unknown",
    }


def is_ray_actor_loss_error(exc: BaseException) -> bool:
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending and len(seen) < 12:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        type_names = {cls.__name__ for cls in type(current).__mro__}
        if type_names.intersection({"RayActorError", "ActorDiedError", "ActorUnavailableError"}):
            return True
        for attr in ("__cause__", "__context__", "cause", "_cause"):
            nested = getattr(current, attr, None)
            if isinstance(nested, BaseException) and id(nested) not in seen:
                pending.append(nested)
    return False


def format_stateful_actor_loss(
    error_context: dict[str, Any] | None,
    exc: BaseException,
) -> BaseException:
    if not error_context or not error_context.get("stateful") or not is_ray_actor_loss_error(exc):
        return exc
    udf_name = str(error_context.get("udf_name") or "udf")
    actor_id = str(error_context.get("actor_id") or "unknown")
    wrapped = RuntimeError(
        f"stateful UDF {udf_name!r} lost actor {actor_id}; state was not recoverable; "
        "side effects may already have occurred"
    )
    wrapped.__cause__ = exc
    return wrapped
