# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Vane contributors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Low-level ADBC bindings for Vane."""

import enum
import functools
import importlib.util

import adbc_driver_manager

__all__ = ["StatementOptions", "connect", "driver_path"]


class StatementOptions(enum.Enum):
    """Statement options supported by Vane's DuckDB-derived engine."""

    #: The number of rows per batch. Defaults to 2048.
    BATCH_ROWS = "adbc.duckdb.query.batch_rows"


def connect(path: str | None = None) -> adbc_driver_manager.AdbcDatabase:
    """Create a low-level ADBC connection to Vane."""
    if path is None:
        return adbc_driver_manager.AdbcDatabase(driver=driver_path(), entrypoint="vane_adbc_init")
    return adbc_driver_manager.AdbcDatabase(driver=driver_path(), entrypoint="vane_adbc_init", path=path)


@functools.cache
def driver_path() -> str:
    """Get the path to Vane's bundled ADBC driver."""
    native_module_spec = importlib.util.find_spec("vane._native")
    if native_module_spec is None or native_module_spec.origin is None:
        msg = "Could not find the Vane native library. Did you install vane-ai?"
        raise ImportError(msg)
    return native_module_spec.origin
