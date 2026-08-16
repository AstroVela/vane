# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class CopyResultUnavailableError(RuntimeError):
    """A COPY committed, but the runner could not return its result."""

    def __init__(
        self,
        operation_id: str,
        detail: str = "",
        cleanup_warnings: tuple[str, ...] = (),
    ) -> None:
        self.operation_id = str(operation_id)
        self.detail = str(detail)
        self.cleanup_warnings = tuple(str(warning) for warning in cleanup_warnings)
        self.write_state = "committed"
        self.safe_to_retry = False
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        message = (
            f"COPY operation {self.operation_id} committed, but its result is unavailable; "
            "refusing to resubmit the committed write"
        )
        if self.detail:
            message += f"; {self.detail}"
        if self.cleanup_warnings:
            message += "; cleanup also failed: " + "; ".join(self.cleanup_warnings)
        return message

    def add_cleanup_warnings(self, *warnings: str) -> None:
        additions = tuple(str(warning) for warning in warnings if str(warning))
        if not additions:
            return
        self.cleanup_warnings += additions
        self.args = (self._build_message(),)

    def __reduce__(
        self,
    ) -> tuple[
        type[CopyResultUnavailableError],
        tuple[str, str, tuple[str, ...]],
    ]:
        return CopyResultUnavailableError, (
            self.operation_id,
            self.detail,
            self.cleanup_warnings,
        )


class CopyOutcomeUnknownError(RuntimeError):
    """The runner cannot prove whether a COPY operation committed."""

    def __init__(
        self,
        operation_id: str,
        base_path: str = "",
        run_id: str = "",
        manifest_path: str = "",
        committed_marker_path: str = "",
        detail: str = "",
        cleanup_warnings: tuple[str, ...] = (),
    ) -> None:
        self.operation_id = str(operation_id)
        self.base_path = str(base_path)
        self.run_id = str(run_id)
        self.manifest_path = str(manifest_path)
        self.committed_marker_path = str(committed_marker_path)
        self.detail = str(detail)
        self.cleanup_warnings = tuple(str(warning) for warning in cleanup_warnings)
        self.safe_to_retry = False
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        message = (
            f"COPY outcome is unknown for operation {self.operation_id}; "
            "refusing to resubmit a potentially committed write"
        )
        if self.base_path or self.run_id:
            message += f"; base_path={self.base_path!r}, run_id={self.run_id!r}"
        if self.detail:
            message += f"; {self.detail}"
        if self.base_path and self.run_id:
            message += "; inspect and explicitly resolve this run before deciding whether to retry"
        if self.cleanup_warnings:
            message += "; cleanup also failed: " + "; ".join(self.cleanup_warnings)
        return message

    def add_cleanup_warnings(self, *warnings: str) -> None:
        additions = tuple(str(warning) for warning in warnings if str(warning))
        if not additions:
            return
        self.cleanup_warnings += additions
        self.args = (self._build_message(),)

    @classmethod
    def from_native_result(
        cls,
        operation_id: str,
        result: Mapping[str, Any],
    ) -> CopyOutcomeUnknownError:
        raw_warnings = result.get("copy_runner_cleanup_warnings", ())
        cleanup_warnings: tuple[str, ...]
        if isinstance(raw_warnings, str):
            cleanup_warnings = (raw_warnings,) if raw_warnings else ()
        elif isinstance(raw_warnings, (list, tuple)):
            cleanup_warnings = tuple(str(warning) for warning in raw_warnings if str(warning))
        else:
            cleanup_warnings = ()
        return cls(
            operation_id,
            str(result.get("copy_output_base_path") or ""),
            str(result.get("copy_output_run_id") or ""),
            str(result.get("copy_output_manifest_path") or ""),
            str(result.get("copy_output_committed_marker_path") or ""),
            str(result.get("copy_output_outcome_error") or ""),
            cleanup_warnings,
        )

    def __reduce__(
        self,
    ) -> tuple[
        type[CopyOutcomeUnknownError],
        tuple[str, str, str, str, str, str, tuple[str, ...]],
    ]:
        return CopyOutcomeUnknownError, (
            self.operation_id,
            self.base_path,
            self.run_id,
            self.manifest_path,
            self.committed_marker_path,
            self.detail,
            self.cleanup_warnings,
        )
