# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral ownership recovery when UDF actor preparation fails."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

_Pool = TypeVar("_Pool")


class OwnedActorPoolsError(RuntimeError):
    """Preserve the original failure and pools whose cleanup needs a retry."""

    def __init__(
        self,
        message: str,
        *,
        owned_actor_pools: list[Any],
        creation_error: BaseException,
    ) -> None:
        super().__init__(message)
        self.owned_actor_pools = list(owned_actor_pools)
        self.creation_error = creation_error


def actor_pool_cleanup_pending(pool: Any) -> bool:
    """Retain ownership unless the pool can confirm that cleanup finished."""

    check = getattr(pool, "cleanup_pending", None)
    return not callable(check) or bool(check())


def rollback_actor_pools(
    created: Iterable[_Pool],
    creation_error: BaseException,
    *,
    shutdown: Callable[[_Pool], None],
    cleanup_pending: Callable[[_Pool], bool],
    record_error: Callable[[BaseException], None],
) -> list[_Pool]:
    """Close owned pools in reverse order, retaining incomplete cleanup.

    Only newly created pools and ownership carried by a failed constructor
    belong here; borrowed pools remain owned by their caller. Transport-specific
    shutdown and bounded diagnostics stay with each backend. A failed status
    check cannot discard ownership or prevent cleanup of the remaining pools.
    """

    pools = list(created)
    pools.extend(getattr(creation_error, "owned_actor_pools", ()))
    unique_pools: list[_Pool] = []
    seen: set[int] = set()
    for pool in pools:
        if id(pool) not in seen:
            seen.add(id(pool))
            unique_pools.append(pool)

    remaining_owned: list[_Pool] = []
    for pool in reversed(unique_pools):
        try:
            shutdown(pool)
        except BaseException as cleanup_error:
            record_error(cleanup_error)
            try:
                pending = cleanup_pending(pool)
            except BaseException as status_error:
                record_error(status_error)
                pending = True
            if pending:
                remaining_owned.append(pool)
    return list(reversed(remaining_owned))
