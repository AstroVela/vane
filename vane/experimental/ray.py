# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Experimental: Ray / Driver integration helper

This module documents the experimental Python surface for DuckDB's Ray integration.

See the developer-facing documentation for details:
`duckdb-python/docs/ray_integration.md`

Short summary
- This is an experimental module (no stable API guarantee).
- The C++ side (in DuckDB core) expects Python workers to expose an API roughly like:

   def submit_task(task: dict) -> task_handle
   # task_handle.get_result() -> awaitable/coroutine  (resolves to ("Success", parts, stats) etc.)

- See `docs/ray_integration.md` for expected return formats and C++ parsing details.

Minimal example (see `examples/ray_worker_example.py` for runnable demo):

```py
# A toy worker handle that accepts simple tasks and returns a coroutine
class SimpleTaskHandle:
    def __init__(self, result):
        self._result = result

    def get_result(self):
        async def coro():
            return ("Success", self._result, b"")

        return coro()


class SimpleWorker:
    def submit_task(self, task: dict):
        # return a handle whose get_result() returns an awaitable
        return SimpleTaskHandle([task])
```

Notes:
- This module is intentionally small: it only contains documentation and examples to help guide contributors.
- If you'd like, we can expand this into runtime helpers for tests or provide a more complete example that integrates with existing examples in `examples/`.
"""

__all__ = []
