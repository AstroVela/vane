# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Allocation-bounded helpers for exception diagnostics."""

from __future__ import annotations


def exception_message_from_args(error: BaseException) -> str | None:
    """Return one exact string argument without invoking provider code."""

    try:
        args = BaseException.__getattribute__(error, "args")
    except BaseException:
        return None
    if type(args) is not tuple or len(args) != 1 or type(args[0]) is not str:
        return None
    return args[0]


def bounded_utf8_text(value: str, max_bytes: int, *, strip: bool = True) -> str:
    """Bound an exact string before normalization can allocate a large copy."""

    if type(value) is not str:
        raise TypeError("diagnostic text must be an exact string")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 3:
        raise ValueError("diagnostic byte limit must be an integer of at least three")
    if len(value) > max_bytes:
        value = value[:max_bytes] + "…" + value[-max_bytes:]
    if strip:
        value = value.strip()
    encoded = value.encode("utf-8", "replace")
    if len(encoded) <= max_bytes:
        return encoded.decode("utf-8")
    omission = "…".encode()
    remaining = max_bytes - len(omission)
    prefix_size = remaining // 2
    suffix_size = remaining - prefix_size
    suffix = encoded[-suffix_size:] if suffix_size else b""
    return encoded[:prefix_size].decode("utf-8", "ignore") + omission.decode() + suffix.decode("utf-8", "ignore")


def safe_exception_type_name(error: BaseException, max_bytes: int = 256) -> str:
    """Return a bounded exact exception class name."""

    try:
        error_type = type.__getattribute__(type(error), "__name__")
    except BaseException:
        return "BaseException"
    if type(error_type) is not str or len(error_type) > max_bytes:
        return "BaseException"
    encoded = error_type.encode("utf-8", "replace")
    if len(encoded) > max_bytes:
        return "BaseException"
    return encoded.decode("utf-8")
