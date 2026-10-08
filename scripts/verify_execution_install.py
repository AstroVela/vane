# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualify installed local, Runtime and TLS Flight query entry points.

Run with python -I outside the checkout. This script has no repository/test
imports and can be extracted by itself from the release sdist.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@contextmanager
def worker_environment(directory):
    import vane

    previous = os.environ.get("PYTHONPATH")
    cwd = Path.cwd()
    # Ray workers do not inherit python -I. Keep their imports on the installed
    # wheel too, even when the caller has checkout paths in its environment.
    os.environ["PYTHONPATH"] = str(Path(vane.__file__).resolve().parent.parent)
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(cwd)
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous


def identity():
    import vane
    from vane import _native
    from vane._native import execution_plan

    distribution = importlib.metadata.distribution("vane-ai")
    package = Path(vane.__file__).resolve()
    native = Path(_native.__file__).resolve()
    require(package == Path(distribution.locate_file("vane/__init__.py")).resolve(), "imported source checkout")
    require(native.is_relative_to(Path(sys.prefix).resolve()), "native module is outside the installed environment")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    require(not direct.get("dir_info", {}).get("editable", False), "editable installs cannot qualify a release")
    digest = hashlib.sha256()
    with native.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return {
        "version": distribution.version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "engine_identity": execution_plan.engine_identity(),
        "native_sha256": digest.hexdigest(),
    }


def options(mode):
    import vane

    target = vane.RayExecution(mode, vane.FteOptions("shared", 2, 0) if mode == "fte" else None)
    return vane.QueryExecutionOptions(target, admission_timeout=60, execution_timeout=60, delivery_timeout=60)


def check_queries(connection, cases, mode=None):
    completed = []
    for case in cases:
        kwargs = {} if mode is None else {"options": options(mode)}
        with connection.query(case["sql"], **kwargs) as result:
            table = result.collect()
            require(table.schema.names == case["columns"], f"{case['name']}: wrong schema names")
            require(all(str(field.type) == "int64" for field in table.schema), f"{case['name']}: wrong types")
            require(table.to_pylist() == case["rows"], f"{case['name']}: wrong result rows")
        completed.append(case["name"])
    return completed


def client_probe(payload):
    import pyarrow.flight as flight
    import ray

    from vane.client import Client

    require(not ray.is_initialized(), "remote client already connected to Ray")
    certificate = payload["certificate"].encode("ascii")
    try:
        with Client(payload["location"], token="invalid-credential-for-release-smoke", tls_root_certs=certificate):
            raise RuntimeError("server accepted invalid credentials")
    except flight.FlightUnauthenticatedError:
        pass
    checks = {}
    for mode in ("pipelined", "fte"):
        with Client(payload["location"], token=payload["token"], tls_root_certs=certificate) as client:
            checks[mode] = check_queries(client, payload["cases"], mode)
            require(client.resource_snapshot()["active_results"] == 0, "client retained closed results")
    require(not ray.is_initialized(), "remote client connected to Ray")
    return {"identity": identity(), "checks": checks, "authentication": "passed", "ray_initialized": False}


def run(report, save):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import ray

    import vane
    from vane.server import Server

    report["identity"] = identity()
    require(not ray.is_initialized(), "qualification must own its isolated Ray cluster")
    openssl = shutil.which("openssl")
    require(openssl is not None, "OpenSSL CLI is required to qualify both TLS endpoints")
    with tempfile.TemporaryDirectory(prefix="vane-execution-smoke-") as temporary, worker_environment(temporary):
        root = Path(temporary).resolve()
        paths = []
        for index, (keys, values) in enumerate((([1, 2], [10, 20]), ([1, None], [1, 3]))):
            path = root / f"input {index}.parquet"
            pq.write_table(pa.table({"k": pa.array(keys, type=pa.int64()), "v": values}), path)
            paths.append("'" + path.as_posix().replace("'", "''") + "'")
        source = "read_parquet([" + ",".join(paths) + "])"
        cases = [
            {
                "name": "aggregate",
                "sql": f"SELECT k, sum(v)::BIGINT AS total FROM {source} GROUP BY k ORDER BY k",
                "columns": ["k", "total"],
                "rows": [{"k": 1, "total": 11}, {"k": 2, "total": 20}, {"k": None, "total": 3}],
            },
            {
                "name": "topn",
                "sql": f"SELECT k, v FROM {source} ORDER BY v DESC LIMIT 2",
                "columns": ["k", "v"],
                "rows": [{"k": 2, "v": 20}, {"k": 1, "v": 10}],
            },
            {"name": "empty", "sql": f"SELECT k, v FROM {source} WHERE v > 100", "columns": ["k", "v"], "rows": []},
        ]
        with vane.connect() as connection:
            report["checks"]["local"] = check_queries(connection, cases)
        save()
        # Exercise public storage defaults: FTE reserves each output object's
        # maximum size, even when this smoke writes only a few rows.
        store = vane.ExchangeStore("shared", str(root / "store"))
        resources = vane.RayResources(
            worker_count=2, partitions=2, max_active_queries=2, max_results=2, exchange_stores=(store,)
        )
        report["stage"] = "ray-startup"
        save()
        try:
            ray.init(address="local", num_cpus=2, include_dashboard=False, log_to_driver=False)
            report["stage"] = "runtime"
            save()
            with vane.Runtime(resources) as runtime:
                with runtime.connect() as connection:
                    for mode in ("pipelined", "fte"):
                        report["checks"]["runtime-" + mode] = check_queries(connection, cases, mode)
                        save()
                require(runtime.resource_snapshot()["service"]["sessions"] == {}, "Runtime retained a session")
            require(runtime.resource_snapshot()["service"]["closed"], "Runtime did not finish closing")
            certificate, key = root / "certificate.pem", root / "key.pem"
            subprocess.run(
                [
                    openssl,
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    str(key),
                    "-out",
                    str(certificate),
                    "-days",
                    "1",
                    "-subj",
                    "/CN=localhost",
                    "-addext",
                    "subjectAltName=DNS:localhost,IP:127.0.0.1",
                ],
                check=True,
                capture_output=True,
                timeout=20,
            )
            report["stage"] = "flight-tls"
            save()
            token = secrets.token_urlsafe(32)
            with Server(
                token=token,
                port=0,
                resources=resources,
                tls_certificates=[(certificate.read_bytes(), key.read_bytes())],
            ) as server:
                # Credentials go through the child's stdin, never argv or reports.
                payload = {
                    "location": server.location.uri.decode(),
                    "token": token,
                    "certificate": certificate.read_text(),
                    "cases": cases,
                }
                child = subprocess.run(
                    [sys.executable, "-I", str(Path(__file__).resolve()), "--client"],
                    input=json.dumps(payload),
                    text=True,
                    capture_output=True,
                    timeout=150,
                )
                require(child.returncode == 0, "remote client failed:\n" + child.stdout + child.stderr)
                remote = json.loads(child.stdout)
                require(remote["identity"] == report["identity"], "remote client loaded a different installation")
                for mode in ("pipelined", "fte"):
                    report["checks"]["flight-" + mode] = remote["checks"][mode]
                report["remote_authentication"] = remote["authentication"]
                report["remote_ray_initialized"] = remote["ray_initialized"]
                require(server.service.snapshot()["sessions"] == 0, "Server retained a closed session")
                save()
        finally:
            ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--client", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.report is not None:
        args.report = args.report.resolve()
    faulthandler.enable()
    faulthandler.dump_traceback_later(240, exit=True)
    if args.client:
        print(json.dumps(client_probe(json.load(sys.stdin))))
        return
    report = {"status": "running", "stage": "local", "checks": {}}

    def save():
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        run(report, save)
        report["status"] = "passed"
        report["stage"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
        print(json.dumps(report, indent=2))
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
