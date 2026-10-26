# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

from cloudpickle import CloudPickler, loads  # type: ignore[import-untyped, unused-ignore]


class _ActorAdapterMeta(type):
    """Mark generated expression adapters that have an explicit constructor recipe."""

    _vane_actor_adapter_recipe: tuple[Callable[..., type], tuple[Any, ...]]


class _VanePickler(CloudPickler):
    def reducer_override(self, obj: Any) -> Any:
        if type(obj) is _ActorAdapterMeta and "_vane_actor_adapter_recipe" in vars(obj):
            # Cloudpickle assigns a new tracker ID to each generated class.
            # Serialize these known adapters by their full constructor recipe
            # so rebuilding an expression does not change its callable bytes.
            # Subclasses without their own recipe retain normal serialization.
            return obj._vane_actor_adapter_recipe
        return super().reducer_override(obj)


def dumps(obj: Any, protocol: int | None = None, buffer_callback: Callable[..., Any] | None = None) -> bytes:
    with io.BytesIO() as file:
        pickler = _VanePickler(file, protocol=protocol, buffer_callback=buffer_callback)
        pickler.dump(obj)
        return file.getvalue()


__all__ = ["dumps", "loads"]
