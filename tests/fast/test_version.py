# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import sys

import vane


def test_version():
    assert vane.__version__ != "0.0.0"


def test_formatted_python_version():
    formatted_python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert vane.__formatted_python_version__ == formatted_python_version
