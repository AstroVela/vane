#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${VANE_ICEBERG_MINIO_PYTHON:-python3}"
minio_image="${VANE_TEST_MINIO_IMAGE:-minio/minio@sha256:1dce27c494a16bae114774f1cec295493f3613142713130c2d22dd5696be6ad3}"
container_name="vane-iceberg-minio-${UID}-$$"
access_key="vaneiceberg"
secret_key="vane-iceberg-test-secret"
region="us-east-1"
bucket="vane-iceberg-test"
container_id=""

if [[ ! "$minio_image" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "VANE_TEST_MINIO_IMAGE must be pinned by sha256 digest: $minio_image" >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for the hermetic Iceberg MinIO gate" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "The Docker daemon is not available" >&2
  exit 1
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python interpreter not found: $python_bin" >&2
  exit 1
fi
if ! "$python_bin" -c 'import botocore' >/dev/null 2>&1; then
  echo "botocore is required in $python_bin" >&2
  exit 1
fi

cleanup_minio() {
  if [[ -n "$container_id" ]]; then
    docker stop --time 10 "$container_id" >/dev/null 2>&1 || true
  fi
}
trap cleanup_minio EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

container_id="$(
  docker run \
    --detach \
    --rm \
    --name "$container_name" \
    --publish 127.0.0.1::9000 \
    --env "MINIO_ROOT_USER=$access_key" \
    --env "MINIO_ROOT_PASSWORD=$secret_key" \
    "$minio_image" \
    server /data --address :9000
)"

published_address="$(docker port "$container_id" 9000/tcp)"
published_port="${published_address##*:}"
if [[ ! "$published_port" =~ ^[1-9][0-9]*$ ]]; then
  echo "Could not resolve the published MinIO port from: $published_address" >&2
  exit 1
fi
endpoint="http://127.0.0.1:$published_port"

export TEST_MINIO_ENDPOINT="$endpoint"
export TEST_MINIO_ACCESS_KEY="$access_key"
export TEST_MINIO_SECRET_KEY="$secret_key"
export TEST_MINIO_REGION="$region"
export TEST_MINIO_BUCKET="$bucket"
export VANE_REQUIRE_ICEBERG_MINIO_TEST=1

if ! "$python_bin" - <<'PY'; then
import os
import time

from botocore.config import Config
from botocore.session import get_session

client = get_session().create_client(
    "s3",
    endpoint_url=os.environ["TEST_MINIO_ENDPOINT"],
    region_name=os.environ["TEST_MINIO_REGION"],
    aws_access_key_id=os.environ["TEST_MINIO_ACCESS_KEY"],
    aws_secret_access_key=os.environ["TEST_MINIO_SECRET_KEY"],
    config=Config(
        signature_version="s3v4",
        connect_timeout=1,
        read_timeout=2,
        retries={"max_attempts": 1},
        s3={"addressing_style": "path"},
    ),
)

last_error = None
for _ in range(80):
    try:
        client.list_buckets()
        client.create_bucket(Bucket=os.environ["TEST_MINIO_BUCKET"])
        break
    except Exception as exc:
        last_error = exc
        time.sleep(0.25)
else:
    raise RuntimeError("MinIO did not become ready within 20 seconds") from last_error
PY
  docker logs "$container_id" >&2 || true
  exit 1
fi

cd "$project_root"
"$project_root/scripts/run_installed_pytest.sh" \
  -m external_service \
  tests/fast/test_distributed_iceberg.py
