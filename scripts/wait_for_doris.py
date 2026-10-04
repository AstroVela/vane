#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Wait for the CI Doris backend to accept tablet creation after HTTP health passes."""

from __future__ import annotations

import argparse
import os
import time
import uuid

import pymysql


def probe_table_creation() -> None:
    database = f"vane_ci_readiness_{uuid.uuid4().hex}"
    with pymysql.connect(
        host=os.environ.get("VANE_TEST_DORIS_MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("VANE_TEST_DORIS_MYSQL_PORT", "9030")),
        user=os.environ.get("VANE_TEST_DORIS_USER", "root"),
        password=os.environ.get("VANE_TEST_DORIS_PASSWORD", ""),
        autocommit=True,
        connect_timeout=5,
        read_timeout=10,
        write_timeout=10,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{database}`")
            try:
                # HTTP health can pass before the first BE disk report reaches
                # the FE. Creating a real OLAP tablet exercises that readiness.
                cursor.execute(
                    f"CREATE TABLE `{database}`.`probe` (id INT NOT NULL) "
                    "ENGINE=OLAP DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 1 "
                    'PROPERTIES ("replication_num" = "1")'
                )
            finally:
                cursor.execute(f"DROP DATABASE IF EXISTS `{database}`")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    while True:
        try:
            probe_table_creation()
        except pymysql.MySQLError as error:
            if time.monotonic() >= deadline:
                raise SystemExit(f"Doris could not create a tablet within {args.timeout:g} seconds: {error}") from error
            print(f"Waiting for Doris tablet creation: {error}", flush=True)
            time.sleep(2)
        else:
            print("Doris tablet creation succeeded", flush=True)
            return


if __name__ == "__main__":
    main()
