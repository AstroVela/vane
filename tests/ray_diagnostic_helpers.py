# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0


def raise_diagnostic_error(error: BaseException, *, long_traceback: bool = False) -> None:
    """Force an oversized traceback without depending on pytest or checkout paths."""
    if long_traceback:
        code = compile("raise error", "diagnostic-frame-" + "x" * 8192, "exec")
        exec(code, {"error": error})
    raise error
