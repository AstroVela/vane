# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reject ambiguous wait attribution and application callback reply tails."""

import copy
import unittest

from analyze import belongs, device_context, rejection


class AttributionTest(unittest.TestCase):
    def test_only_contained_same_call_and_descriptor(self):
        event = ["fdatasync", "payload", 0, 100, 200, 1, 0, 0, 7, -1, 0, -1, -1, -1]
        call = dict(number=75, fd=7)
        wait = dict(
            begin_ns=110, end_ns=120, stable=True, syscall_before=call, syscall_after=call, file=dict(kind="payload")
        )
        self.assertIsNone(rejection(wait, event))
        for field, value, reason in (
            ("end_ns", 201, "cross_boundary"),
            ("begin_ns", 99, "cross_boundary"),
            ("stable", False, "unstable_or_unavailable_syscall"),
            ("file", dict(kind="wal"), "unavailable_or_different_file"),
        ):
            changed = dict(wait, **{field: value})
            self.assertEqual(rejection(changed, event), reason)
        for field, value, reason in (("fd", 8, "different_fd"), ("number", 74, "different_syscall")):
            changed = copy.deepcopy(wait)
            changed["syscall_before"][field] = value
            self.assertEqual(rejection(changed, event), reason)
        changed = copy.deepcopy(wait)
        changed["syscall_after"] = dict(number=75, fd=8)
        self.assertEqual(rejection(changed, event), "unstable_or_unavailable_syscall")

    def test_range_arguments(self):
        event = ["sync_file_range", "payload", 0, 100, 200, 1, 0, 0, 7, 4096, 8192, 2, -1, -1]
        call = dict(number=277, fd=7, offset=4096, bytes=8192, flags=2)
        wait = dict(
            begin_ns=110, end_ns=120, stable=True, syscall_before=call, syscall_after=call, file=dict(kind="payload")
        )
        self.assertIsNone(rejection(wait, event))
        call["flags"] = 3
        self.assertEqual(rejection(wait, event), "different_range")

    def test_previous_reply_tail_is_not_next_write(self):
        root = ["write", 0, 100, 201, 1, 0, 4096, 2, 42, 0, []]
        app = dict(kind="write", begin_ns=200, end_ns=300, offset=4096, bytes=4096)
        self.assertFalse(belongs(root, app, 42))
        root[2], root[3], root[5] = 210, 310, 4096
        self.assertTrue(belongs(root, app, 42))
        self.assertFalse(belongs(root, app, 43))
        root[5] = 0
        self.assertFalse(belongs(root, app, 42))

    def test_device_snapshot_edges_are_excluded(self):
        samples = [
            dict(begin_ns=b, end_ns=e, values=[v] * 17)
            for b, e, v in ((90, 110, 0), (120, 125, 1), (180, 185, 3), (190, 210, 5))
        ]
        value = device_context(samples, [r["begin_ns"] for r in samples], 100, 200)
        self.assertEqual(value["interior_snapshots"], 2)
        self.assertEqual(value["delta"]["flushes"], 2)
        self.assertEqual(value["observed_span_ns"], 65)
        self.assertNotIn("delta", device_context(samples, [r["begin_ns"] for r in samples], 121, 190))


if __name__ == "__main__":
    unittest.main()
