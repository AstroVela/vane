# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Own the host regression services until the qualification launcher's stdin closes."""

from __future__ import annotations

import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iceberg_multinode_cluster import IcebergMultinodeCluster


def main():
    def terminate(_signum, _frame):
        raise SystemExit("qualification service helper terminated")

    signal.signal(signal.SIGTERM, terminate)
    cluster = IcebergMultinodeCluster(Path(sys.argv[1]), sys.argv[2])
    try:
        cluster.start_services()
        print("VANE_SERVICES_READY", flush=True)
        sys.stdin.readline()
    finally:
        cluster.close()


if __name__ == "__main__":
    main()
