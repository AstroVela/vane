# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only preparation on an explicitly owned diagnostic process."""

import argparse
import hashlib
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=2)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    data = bytes(range(256)) * 1024
    rounds = 0
    started = time.time()
    begin = time.monotonic()
    while time.monotonic() - begin < args.seconds:
        value = hashlib.sha256(data).hexdigest()
        rounds += 1
    print(
        json.dumps(
            dict(
                pid=os.getpid(),
                cpu=args.cpu,
                rounds=rounds,
                sha256=value,
                seconds=time.monotonic() - begin,
                started_epoch=started,
                ended_epoch=time.time(),
            )
        )
    )


if __name__ == "__main__":
    main()
