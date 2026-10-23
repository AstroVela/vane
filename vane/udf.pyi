# SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import typing

from vane._native._func import (
    ARROW as ARROW,
)
from vane._native._func import (
    DEFAULT as DEFAULT,
)
from vane._native._func import (
    NATIVE as NATIVE,
)
from vane._native._func import (
    SPECIAL as SPECIAL,
)
from vane._native._func import (
    FunctionNullHandling as FunctionNullHandling,
)
from vane._native._func import (
    PythonUDFType as PythonUDFType,
)

__all__: list[str] = [
    "ARROW",
    "DEFAULT",
    "NATIVE",
    "SPECIAL",
    "FunctionNullHandling",
    "PythonUDFType",
    "vectorized",
    "no_retries",
    "zip_map_batches",
]

def vectorized(func: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]: ...

_Function = typing.TypeVar("_Function", bound=typing.Callable[..., typing.Any])

def no_retries(function: _Function) -> _Function: ...
def zip_map_batches(
    relation: typing.Any,
    left: typing.Callable[..., typing.Any],
    right: typing.Callable[..., typing.Any],
    **options: typing.Any,
) -> typing.Any: ...
