# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse conservative native-marker/kernel matching without relaxing identity checks."""

import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "sqlite_step_waits/match.py"
spec = importlib.util.spec_from_file_location("gate_kernel_match", path)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
rejection = parent.rejection
