# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Import guard for the callable ``vane.func`` API.

``vane.func`` is a top-level callable exported from :mod:`vane`, not a
submodule. This guard prevents ``import vane.func`` from falling through to the
delegated ``vane.func`` package via ``vane.__path__``.
"""

_MESSAGE = "No module named 'vane.func'; use the callable vane.func attribute"
raise ModuleNotFoundError(_MESSAGE)
