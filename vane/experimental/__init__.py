# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

from . import (
    ray,
    spark,
)

__all__ = [
    "ray",
    "spark",
]


def connect_driver_session(database=":memory:", read_only=False, config=None):
    """Experiment with unresolved requests bound in a driver-owned Ray session.

    Standard vane.connect() retains client-side binding. This explicit prototype
    initializes Ray when SQL parsing, schema analysis or execution first needs it.
    """
    from vane._ray_connection import RayConnection

    return RayConnection(database, read_only=read_only, config=config)


__all__.append("connect_driver_session")
