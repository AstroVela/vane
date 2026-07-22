# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

from vane.execution._common import ensure_table as _ensure_table
from vane.execution.ray_stream_adapter import TaskLeaseObjectRefGenerator
from vane.execution.udf_ray_actor_state import (
    format_stateful_actor_loss as _format_stateful_actor_loss,
)
from vane.execution.udf_ray_stream_protocol import RAY_UDF_GENERATOR_BACKPRESSURE_OBJECTS, task_payload_with_lease
from vane.execution.udf_row_preserving import row_preserving_arg_count


def _with_generator_backpressure(method: Any) -> Any:
    return method.options(_generator_backpressure_num_objects=RAY_UDF_GENERATOR_BACKPRESSURE_OBJECTS)


class RemoteUDFSubmitMixin:
    def _rename_args(self, args: pa.Table) -> pa.Table:
        if not self._input_names:
            return args

        if str(self._payload.get("call_mode") or "") == "map_batches_rows":
            # The streamed layout is [UDF args..., passthrough...]. input_names
            # describes only the argument prefix; the suffix must remain intact.
            arg_count = row_preserving_arg_count(self._payload)
            if len(self._input_names) != arg_count:
                message = f"UDF input_names count {len(self._input_names)} does not match scalar_arg_count {arg_count}"
                raise ValueError(message)
            if arg_count > args.num_columns:
                message = f"scalar_arg_count {arg_count} exceeds input column count {args.num_columns}"
                raise ValueError(message)
            return args.rename_columns([*self._input_names, *args.column_names[arg_count:]])

        if len(self._input_names) != args.num_columns:
            raise ValueError(
                f"UDF input_names count {len(self._input_names)} does not match input column count {args.num_columns}"
            )
        return args.rename_columns(self._input_names)

    def submit_with_id(self, submit_id: int, args: pa.Table):
        return self._submit(args, submit_id=int(submit_id))

    def submit(self, _args: pa.Table):
        raise RuntimeError("distributed Ray UDF submission requires submit_with_id()")

    def _submit(self, args: pa.Table, *, submit_id: int):
        args = _ensure_table(args)

        # Rename columns before dispatching to actor
        args = self._rename_args(args)

        return self._submit_one(args, submit_id=submit_id)

    def _submit_one(
        self,
        args: pa.Table,
        *,
        submit_id: int,
    ):
        if not self.actors:
            raise RuntimeError("udf ray actor pool is empty")

        def _submit_after_lease(task_payload: dict[str, Any]):
            self._wait_for_ready_actor()
            actor_idx, actor = self._pick_ready_actor_on_node(
                task_payload["node_id"],
                task_payload["actor_index"],
            )
            try:
                return _with_generator_backpressure(actor.run_block_stream).remote(
                    args,
                    payload=task_payload,
                )
            except Exception as exc:
                self._mark_actor_unavailable(actor_idx, exc)
                formatted = _format_stateful_actor_loss(self.error_context(), exc)
                if formatted is not exc:
                    raise formatted from exc
                raise RuntimeError(f"udf ray submission failed: actor_idx={actor_idx}: {exc}") from exc

        admission = self._take_task_admission()
        return TaskLeaseObjectRefGenerator(
            admission=admission,
            submitter=lambda lease: _submit_after_lease(task_payload_with_lease(self._payload, lease)),
        )


__all__ = [name for name in globals() if not name.startswith("__")]
