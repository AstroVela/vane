# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Owned Docker topology for opt-in Iceberg object-storage qualification."""

from __future__ import annotations

import json
import os
import queue
import secrets
import shlex
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

MINIO = "minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e"
MC = "minio/mc@sha256:a7fe349ef4bd8521fb8497f55c6042871b2ae640607cf99d9bede5e9bdf11727"
GRAVITINO = "apache/gravitino:1.3.0@sha256:04c74cee17946865fa20186c032fda2ed7d1722468e2d58aa32114ea3b35b057"
HELPERS = (
    "iceberg_multinode_driver.py",
    "iceberg_s3_helpers.py",
    "iceberg_s3_proxy.py",
    "iceberg_ray_helpers.py",
    "iceberg_scan_helpers.py",
    "ray_native_scan_helpers.py",
    "gravitino_iceberg_proxy.py",
)


class IcebergMultinodeCluster:
    def __init__(self, directory, image):
        if not (image.startswith("sha256:") or "@sha256:" in image):
            raise ValueError("VANE_TEST_ICEBERG_MULTINODE_IMAGE must identify an immutable local image")
        self.directory = directory
        self.image = image
        self.command = shlex.split(os.environ.get("VANE_TEST_ICEBERG_DOCKER_COMMAND", "docker"))
        if not self.command:
            raise ValueError("Docker command cannot be empty")
        self.token = uuid.uuid4().hex[:16]
        self.label = "vane.iceberg-multinode=" + self.token
        self.network = "vane-iceberg-multinode-" + self.token
        self.network_id = None
        self.containers = {}
        self.records = {}
        self.policies = {"deny_metadata": False, "denied": []}
        self.logs = []

    def save(self):
        path = self.directory / "owned-resources.json"
        path.write_text(
            json.dumps(
                {
                    "token": self.token,
                    "network": self.network_id,
                    "containers": self.containers,
                    "topology": self.records,
                },
                indent=2,
            )
        )
        path.chmod(0o600)

    def docker(self, *args, input=None, timeout=120):
        result = subprocess.run(self.command + list(args), input=input, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError("Docker fixture command failed: " + result.stderr)
        return result.stdout.strip()

    def envfile(self, name, values):
        path = self.directory / name
        path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
        path.chmod(0o600)
        return str(path)

    def inspect(self, role):
        return json.loads(self.docker("inspect", self.containers[role]))[0]

    def create(self, role, image, command, *, options=(), helpers=False):
        container = self.docker(
            "create",
            "--pull=never",
            "--name",
            self.network + "-" + role,
            "--label",
            self.label,
            "--network",
            self.network,
            "--network-alias",
            role,
            *options,
            image,
            *command,
        )
        self.containers[role] = container
        self.save()
        if helpers:
            for name in HELPERS:
                self.docker(
                    "cp", str(Path(__file__).with_name(name)), container + ":/opt/qualification/helpers/" + name
                )
        self.docker("start", container)
        details = self.inspect(role)
        assert details["Config"]["Labels"]["vane.iceberg-multinode"] == self.token
        assert not any(mount["Type"] in {"bind", "volume"} for mount in details["Mounts"]), details["Mounts"]
        self.records[role] = {
            "id": container,
            "image": details["Image"],
            "mounts": details["Mounts"],
            "ip": details["NetworkSettings"]["Networks"][self.network]["IPAddress"],
        }
        self.save()
        return container

    def wait_http(self, url):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                with urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            time.sleep(0.2)
        raise AssertionError("Fixture endpoint did not become ready: " + url)

    def published(self, role, port):
        return "http://" + self.docker("port", self.containers[role], str(port) + "/tcp")

    def start_services(self):
        """Start only the owned object store/catalog for host-runner regressions."""
        import pyarrow.fs as pafs

        self.network_id = self.docker("network", "create", "--label", self.label, self.network)
        self.save()
        key, secret = "vane-fixture-" + self.token, "fixture-" + secrets.token_hex(16)
        bucket = "vane-multinode-" + self.token
        self.create(
            "storage",
            MINIO,
            ["server", "/data", "--address", ":9000", "--quiet"],
            options=[
                "--memory=512m",
                "--tmpfs",
                "/data:rw,size=268435456,mode=0777",
                "-p",
                "127.0.0.1::9000",
                "--env-file",
                self.envfile(
                    "minio.env", {"MINIO_ROOT_USER": key, "MINIO_ROOT_PASSWORD": secret, "MINIO_BROWSER": "off"}
                ),
            ],
        )
        storage = self.published("storage", 9000)
        self.wait_http(storage + "/minio/health/ready")
        fs = pafs.S3FileSystem(
            access_key=key,
            secret_key=secret,
            region="us-east-1",
            scheme="http",
            endpoint_override=storage.removeprefix("http://"),
            allow_bucket_creation=True,
        )
        fs.create_dir(bucket)
        self.docker(
            "run",
            "--rm",
            "--pull=never",
            "--label",
            self.label,
            "--network",
            self.network,
            "--env-file",
            self.envfile("mc.env", {"MC_HOST_fixture": f"http://{key}:{secret}@storage:9000"}),
            MC,
            "anonymous",
            "set",
            "download",
            "fixture/" + bucket,
        )
        self.create(
            "catalog",
            GRAVITINO,
            [],
            options=[
                "--memory=1536m",
                "-p",
                "127.0.0.1::9001",
                "--env-file",
                self.envfile(
                    "catalog.env",
                    {
                        "GRAVITINO_AUX_SERVICE_NAMES": "iceberg-rest",
                        "GRAVITINO_ICEBERG_REST_CATALOG_BACKEND": "memory",
                        "GRAVITINO_ICEBERG_REST_IO_IMPL": "org.apache.iceberg.io.ResolvingFileIO",
                        "GRAVITINO_ICEBERG_REST_WAREHOUSE": f"s3://{bucket}/warehouse",
                        "GRAVITINO_ICEBERG_REST_S3_ENDPOINT": "http://storage:9000",
                        "GRAVITINO_ICEBERG_REST_S3_REGION": "us-east-1",
                        "GRAVITINO_ICEBERG_REST_S3_PATH_STYLE_ACCESS": "true",
                        "GRAVITINO_ICEBERG_REST_S3_ACCESS_KEY": key,
                        "GRAVITINO_ICEBERG_REST_S3_SECRET_KEY": secret,
                        "GRAVITINO_MEM": "-Xms128m -Xmx768m -XX:MaxMetaspaceSize=384m",
                    },
                ),
            ],
        )
        endpoint = self.published("catalog", 9001) + "/iceberg"
        self.wait_http(endpoint + "/v1/config")
        self.config = {
            "endpoint": endpoint,
            "s3_endpoint": storage,
            "s3_region": "us-east-1",
            "bucket": bucket,
            "admin_key": key,
            "admin_secret": secret,
        }
        config_path = self.directory / "sdk-config.json"
        config_path.write_text(json.dumps(self.config))
        config_path.chmod(0o600)
        self.sdk_config = config_path

    def start(self):
        self.start_services()
        proxy_code = (
            "from pathlib import Path; import sys; sys.path.insert(0,'/opt/qualification/helpers'); "
            "from iceberg_s3_proxy import main; "
            "Path('/run/qualification/control.json').write_text('{}'); "
            "main('http://storage:9000','/run/qualification/audit.jsonl','/run/qualification/control.json','0.0.0.0','9000')"
        )
        self.create(
            "objects",
            self.image,
            ["python", "-I", "-c", proxy_code],
            options=["--memory=256m", "-p", "127.0.0.1::9000"],
            helpers=True,
        )
        self.wait_http(self.published("objects", 9000) + "/minio/health/ready")
        self.create(
            "rest",
            self.image,
            [
                "python",
                "-I",
                "/opt/qualification/helpers/gravitino_iceberg_proxy.py",
                "http://catalog:9001/iceberg",
                "0.0.0.0",
                "9001",
            ],
            options=["--memory=256m", "-p", "127.0.0.1::9001"],
            helpers=True,
        )
        self.wait_http(self.published("rest", 9001) + "/iceberg/v1/config")
        for role in ("driver", "worker-a", "worker-b"):
            self.create(
                role,
                self.image,
                [
                    "python",
                    "-I",
                    "/opt/qualification/helpers/iceberg_multinode_driver.py",
                    "head" if role == "driver" else "worker",
                ],
                options=[
                    "--memory=4g",
                    "--cpus=2",
                    "--shm-size=2g",
                    "-e",
                    "VANE_DISTRIBUTED_NODE_COUNT=2",
                    "-e",
                    "VANE_DISTRIBUTED_WORKER_SLOTS=4",
                    "-e",
                    "VANE_FTE_RETRY_INITIAL_DELAY_S=0",
                    "-e",
                    "VANE_FTE_STATUS_WAIT_TIMEOUT_S=5",
                    "-e",
                    "VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S=0",
                    "-e",
                    "VANE_FTE_SPLIT_QUEUE_SPACE_WAIT_TIMEOUT_S=0.1",
                ],
                helpers=True,
            )
        self.wait_ray()

    def wait_ray(self, *, excluded_node_ids=()):
        # A CLI bootstrap runs as PID 1; keep readiness independent from log text.
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                self.docker(
                    "exec",
                    self.containers["driver"],
                    "python",
                    "-I",
                    "-c",
                    "import socket; socket.create_connection(('driver',6379),timeout=1).close()",
                    timeout=5,
                )
                own_ips = {self.records[name]["ip"] for name in ("driver", "worker-a", "worker-b")}
                self.topology = self.request(
                    {"action": "topology", "ips": sorted(own_ips), "excluded_node_ids": list(excluded_node_ids)}
                )
                assert {node["ip"] for node in self.topology["nodes"]} == own_ips
                return
            except RuntimeError:
                time.sleep(0.5)
        raise AssertionError("Owned Ray cluster did not become ready")

    def request(self, request, *, stop_node=False, timeout=240):
        command = self.command + [
            "exec",
            "-i",
            self.containers["driver"],
            "python",
            "-I",
            "/opt/qualification/helpers/iceberg_multinode_driver.py",
        ]
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        process.stdin.write(json.dumps(request))
        process.stdin.close()
        output = queue.Queue()

        def read():
            for line in process.stdout:
                output.put(line)
            output.put(None)

        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        lines, result, stopped, stopped_node_id = [], None, None, None
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                try:
                    line = output.get(timeout=min(1, max(0.01, deadline - time.monotonic())))
                except queue.Empty:
                    continue
                if line is None:
                    break
                lines.append(line)
                if line.startswith("VANE_QUALIFICATION="):
                    value = json.loads(line.split("=", 1)[1])
                    if value.get("phase") == "stop_node":
                        assert stop_node
                        stopped_node_id = value["node_id"]
                        node = next(node for node in self.topology["nodes"] if node["id"] == value["node_id"])
                        stopped = next(
                            role for role in ("worker-a", "worker-b") if self.records[role]["ip"] == node["ip"]
                        )
                        self.docker("kill", self.containers[stopped])
                    if "result" in value:
                        result = value["result"]
            else:
                process.kill()
                raise AssertionError("Container probe timed out: " + "".join(lines))
            process.wait(timeout=5)
            self.logs.append({"action": request["action"], "returncode": process.returncode, "output": "".join(lines)})
            assert process.returncode == 0 and result is not None, "".join(lines)
            return result
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            thread.join(timeout=5)
            process.stdout.close()
            if stopped is not None:
                self.docker("start", self.containers[stopped])
                self.records[stopped]["ip"] = self.inspect(stopped)["NetworkSettings"]["Networks"][self.network][
                    "IPAddress"
                ]
                self.save()
                self.wait_ray(excluded_node_ids=[stopped_node_id])

    def policy(self, *, deny_metadata=False, denied=()):
        self.policies = {"deny_metadata": deny_metadata, "denied": list(denied)}
        self.docker(
            "exec",
            "-i",
            self.containers["objects"],
            "python",
            "-I",
            "-c",
            "import sys; from pathlib import Path; p=Path('/run/qualification/control.pending'); "
            "p.write_text(sys.stdin.read()); p.replace('/run/qualification/control.json')",
            input=json.dumps(self.policies),
        )

    @contextmanager
    def blocked_metadata(self, *, denied=()):
        self.policy(deny_metadata=True, denied=denied)
        try:
            yield
        finally:
            self.policy()

    def events(self):
        text = self.docker(
            "exec",
            self.containers["objects"],
            "python",
            "-I",
            "-c",
            "from pathlib import Path; print(Path('/run/qualification/audit.jsonl').read_text())",
        )
        return [json.loads(line) for line in text.splitlines() if line]

    def stop_rest_forwarder(self):
        self.docker("stop", "--time=5", self.containers["rest"])

    def provider_installed(self, role, installed):
        assert role in {"worker-a", "worker-b"}
        previous_node = next(node["id"] for node in self.topology["nodes"] if node["ip"] == self.records[role]["ip"])
        if installed:
            self.docker(
                "exec",
                self.containers[role],
                "python",
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                *json.loads(
                    self.docker(
                        "exec",
                        self.containers[role],
                        "python",
                        "-I",
                        "-c",
                        "import glob,json; print(json.dumps(glob.glob('/opt/qualification/wheels/vane_extension_iceberg-*.whl')))",
                    )
                ),
            )
        else:
            self.docker(
                "exec", self.containers[role], "python", "-m", "pip", "uninstall", "-y", "vane-extension-iceberg"
            )
        self.docker("restart", "--time=5", self.containers[role])
        self.records[role]["ip"] = self.inspect(role)["NetworkSettings"]["Networks"][self.network]["IPAddress"]
        self.save()
        self.wait_ray(excluded_node_ids=[previous_node])

    def close(self):
        errors = []
        diagnostics = Path(
            os.environ.get("VANE_TEST_ICEBERG_MULTINODE_DIAGNOSTICS", str(self.directory / "diagnostics"))
        )
        diagnostics.mkdir(parents=True, exist_ok=True)
        for role, container in reversed(list(self.containers.items())):
            try:
                details = self.inspect(role)
                assert details["Id"] == container
                assert details["Config"]["Labels"]["vane.iceberg-multinode"] == self.token
                assert self.network in details["NetworkSettings"]["Networks"]
                captured = subprocess.run(
                    self.command + ["logs", container], capture_output=True, text=True, timeout=30
                )
                log = captured.stdout + captured.stderr
                assert captured.returncode == 0, log
                path = diagnostics / (role + ".log")
                path.write_text(log)
                path.chmod(0o600)
                self.docker("rm", "-f", container)
            except Exception as error:
                errors.append(str(error))
        if self.network_id:
            try:
                network = json.loads(self.docker("network", "inspect", self.network_id))[0]
                assert network["Id"] == self.network_id and network["Labels"]["vane.iceberg-multinode"] == self.token
                assert not network["Containers"]
                self.docker("network", "rm", self.network_id)
            except Exception as error:
                errors.append(str(error))
        (diagnostics / "topology.json").write_text(json.dumps(self.records, indent=2) + "\n")
        (diagnostics / "driver-probes.json").write_text(json.dumps(self.logs, indent=2) + "\n")
        if hasattr(self, "object_audit"):
            (diagnostics / "object-audit.json").write_text(json.dumps(self.object_audit, indent=2) + "\n")
        (diagnostics / "cleanup.json").write_text(
            json.dumps(
                {"containers": list(self.containers.values()), "network": self.network_id, "errors": errors}, indent=2
            )
            + "\n"
        )
        if errors:
            raise AssertionError("Owned cluster cleanup failed: " + "; ".join(errors))
