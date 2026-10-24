#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build or prepare matched test-only wheels, then qualify isolated Iceberg reads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ORIGINAL_SYS_PATH = sys.path.copy()
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from build_iceberg_fixture import PROJECT, build_inputs, source_profile
finally:
    sys.path[:] = _ORIGINAL_SYS_PATH

RUNTIME = ROOT / "tests/iceberg_multinode"
TRUST = "vane-ci-test-key"
LICENSE_EXPRESSION = (
    "Apache-2.0 AND MIT AND BSD-2-Clause AND BSD-3-Clause AND ISC AND 0BSD "
    "AND curl AND Zlib AND MPL-2.0 AND Unicode-3.0"
)
CONTRACTS = ("test_iceberg_qualification.py", "test_ray_worker_lifecycle.py", "test_ray_fragment_submission.py")
CONTRACT_SELECTION = (
    "test_iceberg_qualification or test_ray_worker_lifecycle or worker_failure or quiescence or stale_worker_shutdown"
)
PROBE = """
import hashlib, importlib.metadata as metadata, json, os
from pathlib import Path
import ray, vane
providers = {}
with vane.connect(config={'autoinstall_known_extensions':'false', 'autoload_known_extensions':'false'}) as con:
    vane.load_installed_extension('iceberg', connection=con)
    for entry in metadata.entry_points(group='vane.dynamic_extension_providers'):
        if entry.name in ('avro', 'iceberg'):
            for artifact in entry.load()()._artifact_by_identity.values():
                descriptor = artifact.descriptor.to_dict()
                assert hashlib.sha256(artifact.path.read_bytes()).hexdigest() == descriptor['sha256']
                providers[descriptor['name']] = descriptor
assert set(providers) == {'avro', 'iceberg'}
assert os.access(Path(ray.__file__).parent/'core/src/ray/gcs/gcs_server', os.X_OK)
print('VANE_QUALIFICATION=' + json.dumps({'version': metadata.version('vane-ai'),
    'engine': vane.__engine_version__, 'providers': providers}))
"""


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    path.chmod(0o600)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def clean_environment():
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("AWS_", "PYTHON", "VANE_", "RAY_")) or key in {"VIRTUAL_ENV", "__PYVENV_LAUNCHER__"}:
            env.pop(key)
    env.update(AWS_EC2_METADATA_DISABLED="true", RAY_USAGE_STATS_ENABLED="0", VANE_RUNNER="local")
    env["PIP_CONFIG_FILE"] = os.devnull
    return env


class Commands:
    def __init__(self, directory, *, docker="docker"):
        self.logs = directory / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.environment = clean_environment()
        self.docker = shlex.split(docker)
        if not self.docker:
            raise ValueError("Docker command cannot be empty")
        self.count = 0

    def __call__(self, command, *, environment=None, cwd=ROOT):
        self.count += 1
        log = self.logs / f"{self.count:03d}-{Path(str(command[0])).name}.log"
        print(f"[{self.count}] {Path(str(command[0])).name}: {log}", flush=True)
        with log.open("w") as stream:
            log.chmod(0o600)
            subprocess.run(
                command, cwd=cwd, env=environment or self.environment, stdout=stream, stderr=stream, check=True
            )
        return log.read_text()


def make_environment(directory, context, run, *, tests=False):
    run([sys.executable, "-I", "-m", "venv", str(directory)])
    python = directory / "bin/python"
    pip = [str(python), "-I", "-m", "pip", "--isolated", "--disable-pip-version-check"]
    links = ["--no-index", "--find-links", str(context / "runtime-wheels")]
    run([*pip, "install", *links, "-r", str(context / "runtime-requirements.txt")])
    wheels = sorted((context / "wheels").glob("*.whl"))
    run([*pip, "install", "--no-deps", *map(str, wheels)])
    if tests:
        run(
            [
                *pip,
                "install",
                *links,
                "--find-links",
                str(context.parent / "test-wheels"),
                "-c",
                str(context / "runtime-requirements.txt"),
                "-r",
                str(context.parent / "test-wheels/requirements-qualification.txt"),
            ]
        )
    run([*pip, "check"])
    return python


def probe(python, run):
    output = run([str(python), "-I", "-c", PROBE])
    return parse_probe(output)


def parse_probe(output):
    records = [
        line.removeprefix("VANE_QUALIFICATION=")
        for line in output.splitlines()
        if line.startswith("VANE_QUALIFICATION=")
    ]
    if len(records) != 1:
        raise ValueError("runtime did not emit exactly one provider identity record")
    return json.loads(records[0])


def runtime_wheels(wheelhouse, destination):
    """Copy only compatible wheels matching the complete fixed runtime lock."""
    from packaging.tags import sys_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename

    tags = set(sys_tags())
    for line in (RUNTIME / "runtime-requirements.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        matches = []
        for wheel in wheelhouse.glob("*.whl"):
            distribution, actual, _build, supported = parse_wheel_filename(wheel.name)
            if distribution == canonicalize_name(name) and str(actual) == version and tags & supported:
                matches.append(wheel)
        if len(matches) != 1:
            raise ValueError(f"wheelhouse must contain one compatible wheel for {line}; found {len(matches)}")
        shutil.copy2(matches[0], destination / matches[0].name)


def prepare(directory, inputs, args, run):
    import tomllib

    context = directory / "context"
    context.mkdir()
    for name in ("runtime-wheels", "wheels"):
        (context / name).mkdir()
    for name in ("Dockerfile", "runtime-requirements.txt"):
        shutil.copy2(RUNTIME / name, context / name)
    base = Path(inputs["base_wheel"]).resolve(strict=True)
    shutil.copy2(base, context / "wheels" / base.name)
    if args.wheelhouse:
        runtime_wheels(args.wheelhouse.resolve(strict=True), context / "runtime-wheels")
    else:
        run(
            [
                sys.executable,
                "-I",
                "-m",
                "pip",
                "--isolated",
                "--disable-pip-version-check",
                "download",
                "--only-binary=:all:",
                "--no-deps",
                "-r",
                str(context / "runtime-requirements.txt"),
                "--dest",
                str(context / "runtime-wheels"),
            ]
        )
    tests = directory / "test-wheels"
    tests.mkdir()
    for name in ("requirements-test.txt", "requirements-qualification.txt"):
        shutil.copy2(PROJECT / name, tests / name)
    if args.test_wheelhouse:
        for wheel in args.test_wheelhouse.resolve(strict=True).glob("*.whl"):
            shutil.copy2(wheel, tests / wheel.name)
    else:
        run(
            [
                sys.executable,
                "-I",
                "-m",
                "pip",
                "--isolated",
                "--disable-pip-version-check",
                "download",
                "--only-binary=:all:",
                "-c",
                str(context / "runtime-requirements.txt"),
                "-r",
                str(tests / "requirements-qualification.txt"),
                "--dest",
                str(tests),
            ]
        )
    notices = set()
    for pattern in tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["license-files"]:
        notices.update(ROOT.glob(pattern))
    license_args = ["--license-expression", LICENSE_EXPRESSION]
    for path in sorted(notices):
        license_args.extend(("--license-file", str(path)))
    with tempfile.TemporaryDirectory(prefix="packaging-", dir=directory) as temporary:
        python = make_environment(Path(temporary) / "env", context, run)
        platform_tag = run(
            [
                str(python),
                "-I",
                "-c",
                "from packaging.tags import sys_tags; print(next(t.platform for t in sys_tags() if t.platform.startswith('manylinux_')))",
            ]
        ).strip()
        artifacts = Path(temporary) / "artifacts"
        dependency = None
        for name in ("avro", "iceberg"):
            artifact = artifacts / (name + ".duckdb_extension")
            run(
                [
                    str(python),
                    "-I",
                    str(ROOT / "scripts/sign_test_dynamic_extension.py"),
                    "--private-key",
                    str(ROOT / "external/duckdb/test/mbedtls/private.pem"),
                    str(Path(inputs[name + "_artifact"]).resolve(strict=True)),
                    str(artifact),
                ]
            )
            command = [
                str(python),
                "-I",
                str(ROOT / "scripts/build_extension_wheel.py"),
                "--artifact",
                str(artifact),
                "--extension-name",
                name,
                "--test-only",
                "--output-directory",
                str(context / "wheels"),
                "--platform-tag",
                platform_tag,
                "--trust-identity",
                TRUST,
                *license_args,
            ]
            if dependency:
                command.extend(("--dependency-wheel", str(dependency), "--dependency-trust-identity", TRUST))
            run(command)
            matches = list((context / "wheels").glob(f"vane_extension_{name}-*.whl"))
            if len(matches) != 1:
                raise ValueError(f"expected exactly one {name} provider wheel")
            dependency = matches[0]
        pip = [str(python), "-I", "-m", "pip", "--isolated", "install", "--no-index", "--no-deps"]
        run([*pip, *map(str, (context / "wheels").glob("vane_extension_*.whl"))])
        run([str(python), "-I", "-m", "pip", "--isolated", "check"])
        identity = probe(python, run)
        # Also resolve and load the exact closure with the offline driver SDK installed.
        make_environment(Path(temporary) / "driver", context, run, tests=True)
    run([*run.docker, "build", "--pull=false", "--iidfile", str(directory / "image-id.txt"), str(context)])
    image = (directory / "image-id.txt").read_text().strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Docker did not return an immutable image ID")
    observed = parse_probe(
        run([*run.docker, "run", "--rm", "--pull=never", "--network=none", image, "python", "-I", "-c", PROBE])
    )
    if observed != identity:
        raise ValueError("image providers differ from the prepared wheel closure")
    sys.path.insert(0, str(ROOT / "tests"))
    from iceberg_multinode_cluster import GRAVITINO, MC, MINIO

    for service in (MINIO, MC, GRAVITINO):
        if args.pull_services:
            run([*run.docker, "pull", service])
        run([*run.docker, "image", "inspect", service])
    files = [path for folder in (context, tests) for path in folder.rglob("*") if path.is_file()]
    native_inputs = {key: {"file": Path(value).name, "sha256": digest(Path(value))} for key, value in inputs.items()}
    provenance = None
    if (directory / "build-provenance.json").exists():
        provenance = json.loads((directory / "build-provenance.json").read_text())
        if provenance["profile"] != source_profile() or any(
            provenance["inputs"][key]["sha256"] != value["sha256"] for key, value in native_inputs.items()
        ):
            raise ValueError("supplied inputs differ from the recorded source build")
    bundle = {
        "format_version": 1,
        "test_only": True,
        "checkout": checkout(),
        "profile": source_profile(),
        "identity": identity,
        "image": image,
        "service_images": [MINIO, MC, GRAVITINO],
        "files": [{"path": str(path.relative_to(directory)), "sha256": digest(path)} for path in sorted(files)],
        "native_inputs": native_inputs,
        "input_mode": "source_build" if provenance else "prebuilt",
        "native_build": provenance,
    }
    write_json(directory / "bundle.json", bundle)
    return bundle


def checkout():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT))
    return {"head": head, "dirty": dirty}


def verify_bundle(directory):
    bundle = json.loads((directory / "bundle.json").read_text())
    if bundle.get("format_version") != 1 or bundle.get("test_only") is not True:
        raise ValueError("expected a version-1 test-only qualification bundle")
    if bundle["profile"] != source_profile():
        raise ValueError("bundle source pins/patches differ from this checkout")
    paths = set()
    for record in bundle["files"]:
        path = directory / record["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(directory) or record["path"] in paths:
            raise ValueError("bundle contains an invalid or duplicate file path")
        if digest(path) != record["sha256"]:
            raise ValueError(f"bundle input changed: {record['path']}")
        paths.add(record["path"])
    actual = {
        str(p.relative_to(directory))
        for folder in ("context", "test-wheels")
        for p in (directory / folder).rglob("*")
        if p.is_file()
    }
    if paths != actual:
        raise ValueError("bundle file inventory changed")
    for name in ("Dockerfile", "runtime-requirements.txt"):
        if (directory / "context" / name).read_bytes() != (RUNTIME / name).read_bytes():
            raise ValueError("bundle runtime recipe differs from this checkout")
    for name in ("requirements-test.txt", "requirements-qualification.txt"):
        if (directory / "test-wheels" / name).read_bytes() != (PROJECT / name).read_bytes():
            raise ValueError("bundle test-driver recipe differs from this checkout")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", bundle["image"]):
        raise ValueError("bundle must use an immutable runtime image ID")
    return bundle


def junit_result(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    if not cases:
        raise ValueError("qualification shard produced no test cases")
    failed = [
        case.get("name") for case in cases if any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
    ]
    if failed:
        raise ValueError(f"qualification requires all cases to pass without skips: {failed}")
    identities = [case.get("classname", "") + "::" + case.get("name", "") for case in cases]
    if len(set(identities)) != len(identities):
        raise ValueError("qualification shard contains duplicate test identities")
    return {"passed": len(cases), "junit": str(path), "cases": identities}


@contextmanager
def host_services(python, directory, image, environment):
    log = directory / "helper.log"
    with log.open("w") as errors:
        log.chmod(0o600)
        process = subprocess.Popen(
            [str(python), "-I", str(ROOT / "tests/iceberg_qualification_services.py"), str(directory), image],
            env=dict(environment, VANE_TEST_ICEBERG_MULTINODE_DIAGNOSTICS=str(directory / "diagnostics")),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
        )
        try:
            timer = threading.Timer(120, process.terminate)
            timer.start()
            try:
                line = process.stdout.readline()
            finally:
                timer.cancel()
                timer.join()
            if line.strip() != "VANE_SERVICES_READY":
                raise RuntimeError(f"host regression services failed; see {log}")
            yield directory / "sdk-config.json"
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
                raise RuntimeError(
                    f"host regression services did not stop; see {directory / 'owned-resources.json'}"
                ) from None
            finally:
                process.stdout.close()
            if process.returncode:
                raise RuntimeError(f"host regression service startup/cleanup failed; see {log}")


def qualify(directory, args):
    bundle = verify_bundle(directory)
    output = directory / "runs" / uuid.uuid4().hex
    output.mkdir(parents=True, mode=0o700)
    run = Commands(output, docker=args.docker_command)
    reports = output / "reports"
    reports.mkdir()
    record = {
        "status": "running",
        "checkout": checkout(),
        "bundle": digest(directory / "bundle.json"),
        "identity": bundle["identity"],
        "image": bundle["image"],
        "input_mode": bundle["input_mode"],
        "scope": "Linux x86-64; one Docker host; isolated Ray nodes; anonymous S3",
        "shards": {},
    }
    write_json(output / "qualification.json", record)
    write_json(directory / "latest-run.json", {"path": str(output.relative_to(directory))})
    try:
        with tempfile.TemporaryDirectory(prefix="driver-", dir=output) as temporary:
            python = make_environment(Path(temporary) / "env", directory / "context", run, tests=True)
            if probe(python, run) != bundle["identity"]:
                raise ValueError("test driver providers differ from the bundle")
            env = dict(
                run.environment,
                PATH=str(python.parent) + os.pathsep + os.environ["PATH"],
                VANE_TEST_ICEBERG_DOCKER_COMMAND=args.docker_command,
                VANE_TEST_ICEBERG_MULTINODE_IMAGE=bundle["image"],
                VANE_TEST_ICEBERG_MULTINODE_DIAGNOSTICS=str(output / "multinode"),
            )

            def shard(name, arguments):
                report = reports / (name + ".xml")
                run(
                    [
                        "bash",
                        str(ROOT / "scripts/run_installed_pytest.sh"),
                        *arguments,
                        "-q",
                        "--junitxml=" + str(report),
                    ],
                    environment=env,
                )
                result = junit_result(report)
                previous = {case for shard in record["shards"].values() for case in shard["cases"]}
                if previous.intersection(result["cases"]):
                    raise ValueError("qualification shards contain duplicate test identities")
                record["shards"][name] = result
                write_json(output / "qualification.json", record)

            shard("contracts", [*("tests/fast/" + name for name in CONTRACTS), "-k", CONTRACT_SELECTION])
            shard("multinode", ["tests/fast/test_iceberg_s3_multinode_live.py", "-m", "external_service"])
            cleanup = json.loads((output / "multinode/cleanup.json").read_text())
            if cleanup["errors"]:
                raise ValueError("isolated-node cleanup reported failures")
            record["multinode_cleanup"] = cleanup
            # Reuse the same owned service fixture for the existing host regressions.
            # No driver/worker containers are started in this second topology.
            service_dir = output / "host-services"
            service_dir.mkdir(mode=0o700)
            # SDK imports execute in the isolated driver environment, not the launcher.
            with host_services(python, service_dir, bundle["image"], env) as config:
                env["VANE_TEST_GRAVITINO_ICEBERG_S3_CONFIG"] = str(config)
                shard(
                    "ray-regression",
                    [
                        "tests/fast/test_iceberg_s3_ray_live.py",
                        "-m",
                        "external_service",
                        "-k",
                        "two_ray_workers or replacement_worker or cancel_remote",
                    ],
                )
                shard(
                    "native-regression",
                    [
                        "tests/fast/test_iceberg_s3_scan_live.py::test_frozen_s3_replay_needs_neither_rest_nor_metadata[positions]",
                        "tests/fast/test_iceberg_s3_scan_live.py::test_frozen_s3_replay_needs_neither_rest_nor_metadata[equality]",
                        "-m",
                        "external_service",
                    ],
                )
            record["host_cleanup"] = json.loads((service_dir / "diagnostics/cleanup.json").read_text())
            if record["host_cleanup"]["errors"]:
                raise ValueError("host regression service cleanup reported failures")
            record["status"] = "passed"
            record["passed"] = sum(shard["passed"] for shard in record["shards"].values())
    finally:
        if record["status"] != "passed":
            record["status"] = "failed"
        write_json(output / "qualification.json", record)
    print(f"Qualification passed: {record['passed']} cases; report: {output / 'qualification.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "build", "prepare", "run"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--docker-command", default=os.environ.get("VANE_TEST_ICEBERG_DOCKER_COMMAND", "docker"))
    parser.add_argument("--base-wheel", type=Path)
    parser.add_argument("--avro-artifact", type=Path)
    parser.add_argument("--iceberg-artifact", type=Path)
    parser.add_argument(
        "--wheelhouse", type=Path, help="Existing exact runtime wheels; otherwise download binary wheels"
    )
    parser.add_argument(
        "--test-wheelhouse", type=Path, help="Existing driver SDK/test wheels; otherwise download binary wheels"
    )
    parser.add_argument(
        "--pull-services", action="store_true", help="Fetch the three digest-pinned service images during preparation"
    )
    parser.add_argument("--vcpkg-root", type=Path)
    parser.add_argument("--base-sdk", type=Path)
    parser.add_argument("--iceberg-sdk", type=Path)
    parser.add_argument("--native-build-directory", type=Path)
    parser.add_argument("--triplet", default="x64-linux-release")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux" or platform.machine() != "x86_64":
        parser.error("this qualification profile requires Linux x86-64 and Python 3.12")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    if args.command == "run":
        pipeline = {"status": "running", "command": "run", "phase": "qualification", "checkout": checkout()}
        write_json(directory / "pipeline.json", pipeline)
        try:
            qualify(directory, args)
            pipeline["status"] = "complete"
        finally:
            if pipeline["status"] != "complete":
                pipeline["status"] = "failed"
            write_json(directory / "pipeline.json", pipeline)
        return
    if (directory / "bundle.json").exists() or (directory / "context").exists():
        parser.error("use a fresh directory for preparation; use run to repeat an existing bundle")
    run = Commands(directory, docker=args.docker_command)
    supplied = (args.base_wheel, args.avro_artifact, args.iceberg_artifact)
    if any(supplied) and not all(supplied):
        parser.error("supply the complete base wheel + Avro/Iceberg artifact set")
    if args.command == "build" and any(supplied):
        parser.error("build creates native inputs; select prepare or all to use prebuilt inputs")
    pipeline = {"status": "running", "command": args.command, "checkout": checkout(), "phase": "inputs"}
    write_json(directory / "pipeline.json", pipeline)
    try:
        if args.command == "prepare":
            if not all(supplied):
                parser.error("prepare requires --base-wheel, --avro-artifact and --iceberg-artifact")
            inputs = dict(zip(("base_wheel", "avro_artifact", "iceberg_artifact"), supplied, strict=True))
        else:
            if args.command == "all" and all(supplied):
                inputs = dict(zip(("base_wheel", "avro_artifact", "iceberg_artifact"), supplied, strict=True))
            else:
                args.vcpkg_root = args.vcpkg_root or directory / "vcpkg"
                inputs = build_inputs(directory, args, run, run.environment)
        write_json(directory / "inputs.json", {key: str(value) for key, value in inputs.items()})
        pipeline["input_mode"] = "prebuilt" if all(supplied) else "source_build"
        if args.command != "build":
            pipeline["phase"] = "prepare"
            write_json(directory / "pipeline.json", pipeline)
            prepare(directory, inputs, args, run)
        if args.command == "all":
            pipeline["phase"] = "qualification"
            write_json(directory / "pipeline.json", pipeline)
            qualify(directory, args)
        pipeline["status"] = "complete"
    finally:
        if pipeline["status"] != "complete":
            pipeline["status"] = "failed"
        write_json(directory / "pipeline.json", pipeline)


if __name__ == "__main__":
    main()
