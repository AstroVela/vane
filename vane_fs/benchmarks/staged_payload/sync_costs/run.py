# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the unchanged 16-mount ext4 diagnostic after recorder checks pass."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    assert json.loads((out / "sync-tests.json").read_text())["status"] == "PASS"
    identity = json.loads((out / "generator-identity.json").read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in identity["inputs"].items())
    assert all(hashlib.sha256((out / p).read_bytes()).hexdigest() == h for p, h in identity["generated_inputs"].items())
    prior = Path(__file__).resolve().parent.parent / "payload_presync/run_diagnostic.py"
    subprocess.run([sys.executable, str(prior), str(out)], check=True)


if __name__ == "__main__":
    main()
