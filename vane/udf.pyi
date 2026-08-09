# SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import typing

from vane._native._func import (
    ARROW as ARROW,
    DEFAULT as DEFAULT,
    NATIVE as NATIVE,
    SPECIAL as SPECIAL,
    FunctionNullHandling as FunctionNullHandling,
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
]

def vectorized(func: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]: ...
