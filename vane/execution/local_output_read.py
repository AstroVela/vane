# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Consumer-side query admission for a buffered subprocess result."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vane.execution.local_query_admission import LocalOutputAdmission
from vane.execution.ref_bundle import REF_BUNDLE_RESULT_MARKER, LocalShmBlockRef
from vane.execution.udf_admission import AdmissionLease


class LocalOutputRead:
    """Own unread output identities independently of the producer process.

    Storage is already charged by the SHM transport. A native read starts
    query output admission; until then only the producer's existing pending
    generator estimate is charged to the query, matching the Ray collector.
    """

    def __init__(self) -> None:
        self._pending: list[tuple[tuple[LocalShmBlockRef, ...], LocalOutputAdmission]] = []

    def buffer(self, result: Any, admission: AdmissionLease | None) -> None:
        """Register unread blocks after the executor has taken cleanup ownership."""
        query = admission.lease.get("query_policy") if admission is not None else None
        if query is None or not isinstance(result, tuple) or len(result) != 4 or result[0] != REF_BUNDLE_RESULT_MARKER:
            return
        assert admission is not None
        refs = tuple(result[1])
        if refs:
            self._pending.append(
                (refs, query.buffer_output(admission.lease["query_policy_task"], sum(ref.size for ref in refs)))
            )

    def poll(self, wakeup: Callable[[], None]) -> bool:
        """Start/read admission without waiting for an output grant."""
        while self._pending:
            refs, request = self._pending[0]
            request.request(wakeup)
            # The result is handed off as one group. A partial grant could
            # hold the progress token needed by its unpublished siblings.
            owners = request.take_bundle(tuple(ref.size for ref in refs))
            if owners is None:
                return False
            attached = 0
            try:
                for ref, owner in zip(refs, owners, strict=True):
                    ref.attach_query_output_lease(owner)
                    attached += 1
                    owner.transition_to("unit_queue")
            except BaseException as error:
                cleanup_error = None
                for owner in owners[attached:]:
                    try:
                        owner.release()
                    except BaseException as exc:
                        cleanup_error = exc
                if cleanup_error is not None:
                    raise error from cleanup_error
                raise
            self._pending.pop(0)
        return True

    def close(self) -> None:
        errors = []
        for entry in tuple(self._pending):
            try:
                entry[1].close()
            except BaseException as error:
                errors.append(error)
            else:
                self._pending.remove(entry)
        if errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            raise RuntimeError(f"local output admission cleanup failed: {details}") from errors[0]
