# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from vane.runners.fte import fte_update_batch as _impl
from vane.runners.ray._fte_compat import reexport

reexport(_impl, globals())
