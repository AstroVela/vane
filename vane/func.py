"""Import guard for the callable ``vane.func`` API.

``vane.func`` is a top-level callable exported from :mod:`vane`, not a
submodule. This guard prevents ``import vane.func`` from falling through to the
delegated ``duckdb.func`` package via ``vane.__path__``.
"""

_MESSAGE = "No module named 'vane.func'; use the callable vane.func attribute"
raise ModuleNotFoundError(_MESSAGE)
