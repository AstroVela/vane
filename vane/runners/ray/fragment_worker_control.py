# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vane.runners.fte import FteTaskAttemptId


def enqueue_ordered_fte_control(
    actor_handle: Any,
    tails_by_task: dict[str, Any],
    method_name: str,
    task_id: str | dict[str, Any],
    *args: Any,
) -> Any:
    task_key = str(FteTaskAttemptId.coerce(task_id))
    previous = tails_by_task.get(task_key)
    method = getattr(actor_handle, method_name)
    if previous is None:
        result_ref = method.remote(task_id, *args)
    else:
        result_ref = method.remote(task_id, *args, previous)
    tails_by_task[task_key] = result_ref
    return result_ref
