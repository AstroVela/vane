# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Require native call identity and a fully contained stable kernel observation."""

NUMBERS = {"fsync": {74}, "fdatasync": {75}, "sync_file_range": {277}, "nanosleep": {35, 230}}


def rejection(point, event, tid):
    marker, after, wait = point["marker_before"], point["marker_after"], point["wait"]
    if marker != after:
        return "marker_changed"
    if (marker["tid"], marker["kind"], marker["begin_ns"]) != (tid, event[0], event[3]):
        return "different_native_call"
    if marker["fd"] != event[8] or [marker[key] for key in ("offset", "bytes", "mode")] != event[9:12]:
        return "different_native_arguments"
    if not event[3] <= wait["begin_ns"] <= wait["end_ns"] <= event[4]:
        return "cross_boundary"
    call = wait["syscall_before"]
    if not wait["stable"] or call != wait["syscall_after"] or "error" in call:
        return "unstable_or_unavailable_syscall"
    if call.get("number") not in NUMBERS[event[0]]:
        return "different_syscall"
    if event[0] != "nanosleep":
        if call.get("fd") != event[8]:
            return "different_fd"
        if wait.get("file", {}).get("kind") != event[1]:
            return "unavailable_or_different_file"
        if event[0] == "sync_file_range" and [call.get(k) for k in ("offset", "bytes", "flags")] != event[9:12]:
            return "different_range"
    return None
