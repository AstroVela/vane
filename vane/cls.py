# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Import guard for the callable ``vane.cls`` API.

``vane.cls`` is a top-level callable exported from :mod:`vane`, not a
submodule. This guard prevents ``import vane.cls`` from falling through to the
delegated ``vane.cls`` package via ``vane.__path__``.
"""

_MESSAGE = "No module named 'vane.cls'; use the callable vane.cls attribute"
raise ModuleNotFoundError(_MESSAGE)
