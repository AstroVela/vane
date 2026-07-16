from __future__ import annotations

from duckdb.runners.fte import fte_descriptor as _impl
from duckdb.runners.ray._fte_compat import reexport

reexport(_impl, globals())
