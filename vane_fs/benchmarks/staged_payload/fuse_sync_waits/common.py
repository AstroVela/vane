# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Identity and trace checks shared by the read-only sync-wait diagnostic."""

import hashlib
import importlib.util
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def validate(trace):
    path = Path(__file__).resolve().parent.parent / "range_writeback_waits/prepare.py"
    spec = importlib.util.spec_from_file_location("sync_wait_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate(trace)
