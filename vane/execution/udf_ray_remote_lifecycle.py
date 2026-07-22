# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class RemoteUDFLifecycleMixin:
    def take_ready_result(self):
        return None

    def finished_submitting(self) -> None:
        self._finished_submitting = True

    def all_tasks_finished(self) -> bool:
        return self._finished_submitting

    def shutdown(self) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True
        shutdown_errors: list[Exception] = []
        try:
            with self._ready_refs_cv:
                self._pending_ready_refs = {}
                self._ready_probe_refs.clear()
                self._ready_probe_ref_set.clear()
            self._ready_actor_indices = []
            self._ready_actor_set = set()
            self._actors_obj.shutdown()
        except Exception as exc:
            shutdown_errors.append(exc)
        if shutdown_errors:
            raise shutdown_errors[0]

    def __del__(self) -> None:
        if getattr(self, "_shutdown_called", True):
            return
        try:
            self.shutdown()
        except Exception as exc:
            import sys
            import types

            sys.unraisablehook(
                types.SimpleNamespace(
                    exc_type=type(exc),
                    exc_value=exc,
                    exc_traceback=exc.__traceback__,
                    err_msg="Exception ignored in RemoteUDFLifecycleMixin.__del__",
                    object=self,
                )
            )


__all__ = [name for name in globals() if not name.startswith("__")]
