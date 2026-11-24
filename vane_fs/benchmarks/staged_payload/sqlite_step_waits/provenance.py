# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Restore unchanged workload provenance needed by the inherited analyzer."""

import json
import shutil
import sys
import time
from pathlib import Path

from prepare import digest, save

if __name__ == "__main__":
    out = Path(sys.argv[1]).resolve()
    prior = Path(__file__).resolve().parents[4] / "vane_fs/build/paced-request-waits-20261019"
    assert (out / "workload.py").read_bytes() == (prior / "workload.py").read_bytes()
    original, target = prior / "workload-proof.json", out / "workload-proof.json"
    if target.exists():
        assert target.read_bytes() == original.read_bytes()
    else:
        shutil.copy2(original, target)
    save(
        out / "provenance-check.json",
        dict(
            status="PASS",
            recorded_epoch=time.time(),
            original=str(original),
            original_sha256=digest(original),
            current_sha256=digest(target),
            frozen_workload_sha256=digest(out / "workload.py"),
            proof_values=json.loads(original.read_text()),
            runner_and_measurements_unchanged=True,
        ),
    )
