# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reject torn markers, callback boundaries and unrelated kernel syscalls."""

import copy
import struct
import unittest
from unittest.mock import patch

import sample
from match import rejection


class ObserverTest(unittest.TestCase):
    def test_header_contract(self):
        data = bytearray(4096)
        struct.pack_into("<8sQQQ", data, 0, sample.MAGIC, 1, 64, 16)
        self.assertEqual(sample.header(data), dict(version=1, stride=64, slots=16))
        for field, value in ((0, b"badmagic"), (1, 2), (2, 128), (3, 17)):
            fields = [sample.MAGIC, 1, 64, 16]
            fields[field] = value
            struct.pack_into("<8sQQQ", data, 0, *fields)
            with self.assertRaises(AssertionError):
                sample.header(data)

    def test_odd_inactive_and_signed_arguments(self):
        data = bytearray(4096)
        struct.pack_into("<QQQqQqqq", data, 64, 2, 123, 4, -1, 100, 200, 0, -1)
        self.assertEqual(
            sample.marker(data, 0),
            dict(sequence=2, tid=123, kind="nanosleep", fd=-1, begin_ns=100, offset=200, bytes=0, mode=-1),
        )
        struct.pack_into("<Q", data, 64, 3)
        self.assertIsNone(sample.marker(data, 0))
        struct.pack_into("<Q", data, 64, 4)
        struct.pack_into("<Q", data, 80, 0)
        self.assertIsNone(sample.marker(data, 0))

    def test_sequence_change_during_read(self):
        data = bytearray(4096)
        struct.pack_into("<QQQqQqqq", data, 64, 2, 123, 4, -1, 100, 200, 0, -1)
        original, calls = struct.unpack_from, []

        def changed(fmt, mapping, offset=0):
            calls.append(fmt)
            if len(calls) == 3:
                struct.pack_into("<Q", data, 64, 4)
            return original(fmt, mapping, offset)

        with patch.object(sample.struct, "unpack_from", side_effect=changed):
            self.assertIsNone(sample.marker(data, 0))

    def point(self):
        marker = dict(sequence=2, tid=42, kind="fsync", fd=7, begin_ns=100, offset=-1, bytes=0, mode=-1)
        call = dict(number=74, fd=7)
        return dict(
            marker_before=marker,
            marker_after=dict(marker),
            wait=dict(
                begin_ns=120,
                end_ns=130,
                stable=True,
                syscall_before=call,
                syscall_after=dict(call),
                file=dict(kind="wal"),
            ),
        )

    def test_exact_identity_and_bracket(self):
        event = ["fsync", "wal", 0, 100, 200, 1, 0, 0, 7, -1, 0, -1, -1, -1]
        point = self.point()
        self.assertIsNone(rejection(point, event, 42))
        self.assertEqual(rejection(point, event, 43), "different_native_call")
        point["marker_after"]["sequence"] += 2
        self.assertEqual(rejection(point, event, 42), "marker_changed")
        for key, value, reason in (
            ("end_ns", 201, "cross_boundary"),
            ("begin_ns", 99, "cross_boundary"),
            ("stable", False, "unstable_or_unavailable_syscall"),
            ("file", dict(kind="payload"), "unavailable_or_different_file"),
        ):
            point = self.point()
            point["wait"][key] = value
            self.assertEqual(rejection(point, event, 42), reason)

    def test_different_call_descriptor_and_range(self):
        event = ["fsync", "wal", 0, 100, 200, 1, 0, 0, 7, -1, 0, -1, -1, -1]
        for key, value, reason in (("number", 75, "different_syscall"), ("fd", 8, "different_fd")):
            point = self.point()
            point["wait"]["syscall_before"][key] = value
            point["wait"]["syscall_after"][key] = value
            self.assertEqual(rejection(point, event, 42), reason)
        point = self.point()
        point["marker_before"]["offset"] = 1
        point["marker_after"] = copy.deepcopy(point["marker_before"])
        self.assertEqual(rejection(point, event, 42), "different_native_arguments")

    def test_sleep_syscall_is_not_an_io_descriptor(self):
        point = self.point()
        event = ["nanosleep", "process", 0, 100, 200, 1, 0, 0, -1, 1000000, 0, -1, -1, -1]
        point["marker_before"].update(kind="nanosleep", fd=-1, offset=1000000)
        point["marker_after"] = dict(point["marker_before"])
        point["wait"]["syscall_before"] = dict(number=230, fd=0)
        point["wait"]["syscall_after"] = dict(number=230, fd=0)
        self.assertIsNone(rejection(point, event, 42))


if __name__ == "__main__":
    unittest.main()
