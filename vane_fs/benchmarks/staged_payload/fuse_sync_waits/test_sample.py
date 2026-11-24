# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Check syscall identity, access failures, clock brackets and a real blocked child."""

import json
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import sample


class SamplerTest(unittest.TestCase):
    def test_syscall_identity(self):
        self.assertEqual(sample.syscall(dict(value="75 0x7 0x0 0x0 0x0 0x0 0x0 0x100 0x200")), dict(number=75, fd=7))
        self.assertEqual(sample.syscall(dict(value="running")), dict(running=True))
        self.assertEqual(sample.syscall(dict(value="-1 0x100 0x200")), dict(number=-1))
        self.assertEqual(
            sample.syscall(dict(value="18 0x8 0xab 0x1000 0x2000 0 0 0 0")),
            dict(number=18, fd=8, bytes=4096, offset=8192),
        )
        self.assertEqual(
            sample.syscall(dict(value="277 0x9 0x1000 0x2000 0x2 0 0 0 0")),
            dict(number=277, fd=9, offset=4096, bytes=8192, flags=2),
        )

    def test_unavailable_and_malformed(self):
        error = dict(error=dict(errno=13, kind="PermissionError"))
        self.assertEqual(sample.syscall(error), error)
        self.assertEqual(sample.syscall(dict(value="bad"))["error"]["kind"], "MalformedSyscall")
        with patch.object(Path, "read_text", side_effect=PermissionError(13, "denied")):
            self.assertEqual(sample.read(Path("unused"))["error"]["errno"], 13)

    def test_descriptor_classification(self):
        for value, expected in (
            ("db.sqlite-wal", "wal"),
            ("db.sqlite.stage-fence", "fence"),
            ("db.sqlite.payload (deleted)", "payload"),
            ("db.sqlite", "database"),
            ("pipe:[123]", "other"),
        ):
            self.assertEqual(sample.file_kind(value), expected)

    def test_device_errors(self):
        with patch.object(sample, "read", return_value=dict(value="1 2 3")):
            self.assertEqual(sample.disk()["error"]["kind"], "InvalidDeviceCounters")
        with patch.object(sample, "read", return_value=dict(value=" ".join(["0"] * 17))):
            value = sample.disk()
            self.assertEqual(value["values"], [0] * 17)
            self.assertLessEqual(value["begin_ns"], value["end_ns"])

    def test_sampler_failure_is_visible(self):
        rows, stop = [], threading.Event()
        with patch.object(sample, "disk", side_effect=RuntimeError("injected")):
            sample.sample(0, stop, rows)
        self.assertEqual(rows[-1]["fatal"]["kind"], "RuntimeError")
        self.assertFalse(rows[-1]["stop_requested"])

    def test_blocked_owned_child(self):
        process = subprocess.Popen(
            [sys.executable, "-I", "-u", "-c", "import sys; print('ready',flush=True); sys.stdin.buffer.read(1)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        rows, stop = [], threading.Event()
        observer = threading.Thread(target=sample.sample, args=(process.pid, stop, rows))
        try:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            observer.start()
            deadline = time.monotonic() + 2
            while len(rows) < 8 and time.monotonic() < deadline:
                stop.wait(0.005)
        finally:
            stop.set()
            if observer.ident is not None:
                observer.join(timeout=5)
            try:
                process.communicate("x", timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)
        self.assertFalse(observer.is_alive())
        self.assertEqual(process.returncode, 0)
        self.assertTrue(rows[-1]["sampler_finished"] and rows[-1]["stop_requested"])
        self.assertIsNone(rows[-1]["fatal"])
        blocked = [
            t
            for r in rows
            for t in r["threads"]
            if t["wait"]["stable"]
            and t["wait"]["syscall_before"].get("number") == 0
            and t["wait"]["wchan"].get("value") not in (None, "0")
        ]
        self.assertTrue(blocked, rows)
        self.assertTrue(all(t["wait"]["begin_ns"] <= t["wait"]["end_ns"] for t in blocked))
        print(
            json.dumps(
                dict(
                    integration="PASS",
                    samples=len(rows),
                    observed_waits=sorted({t["wait"]["wchan"]["value"] for t in blocked}),
                    stack_access=rows[-1]["stack_access"],
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
