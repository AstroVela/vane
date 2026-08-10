#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${VANE_ICEBERG_REST_PYTHON:-python3}"
minio_image="${VANE_TEST_MINIO_IMAGE:-minio/minio@sha256:1dce27c494a16bae114774f1cec295493f3613142713130c2d22dd5696be6ad3}"
rest_image="${VANE_TEST_ICEBERG_REST_IMAGE:-apache/iceberg-rest-fixture@sha256:39e1c38a10d1b380dfb22f5c60685a2aa82975651018dd1a87f12045047821d9}"
resource_suffix="${UID}-$$"
network_name="vane-iceberg-rest-${resource_suffix}"
minio_name="vane-iceberg-rest-minio-${resource_suffix}"
rest_name="vane-iceberg-rest-catalog-${resource_suffix}"
access_key="admin"
secret_key="password"
marker_fault_access_key="vane-marker-fault"
marker_fault_secret_key="marker-fault-password"
marker_fault_policy="vane-marker-fault"
region="us-east-1"
bucket="warehouse"
network_created=0
minio_container_id=""
rest_container_id=""

if [[ ! "$minio_image" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "VANE_TEST_MINIO_IMAGE must be pinned by sha256 digest: $minio_image" >&2
  exit 2
fi
if [[ ! "$rest_image" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "VANE_TEST_ICEBERG_REST_IMAGE must be pinned by sha256 digest: $rest_image" >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for the hermetic Iceberg REST Catalog gate" >&2
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

cleanup_services() {
  if [[ -n "$rest_container_id" ]]; then
    docker rm --force "$rest_container_id" >/dev/null 2>&1 || true
  fi
  if [[ -n "$minio_container_id" ]]; then
    docker rm --force "$minio_container_id" >/dev/null 2>&1 || true
  fi
  if ((network_created)); then
    docker network rm "$network_name" >/dev/null 2>&1 || true
  fi
}
trap cleanup_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

docker network create "$network_name" >/dev/null
network_created=1

minio_container_id="$(
  docker run \
    --detach \
    --name "$minio_name" \
    --network "$network_name" \
    --network-alias minio \
    --network-alias warehouse.minio \
    --publish 127.0.0.1::9000 \
    --env "MINIO_ROOT_USER=$access_key" \
    --env "MINIO_ROOT_PASSWORD=$secret_key" \
    --env MINIO_DOMAIN=minio \
    "$minio_image" \
    server /data --address :9000
)"

minio_published_address="$(docker port "$minio_container_id" 9000/tcp)"
minio_published_port="${minio_published_address##*:}"
if [[ ! "$minio_published_port" =~ ^[1-9][0-9]*$ ]]; then
  echo "Could not resolve the published MinIO port from: $minio_published_address" >&2
  exit 1
fi
minio_endpoint="http://127.0.0.1:$minio_published_port"

export TEST_MINIO_ENDPOINT="$minio_endpoint"
export TEST_MINIO_ACCESS_KEY="$access_key"
export TEST_MINIO_SECRET_KEY="$secret_key"
export TEST_MINIO_REGION="$region"
export TEST_MINIO_BUCKET="$bucket"

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
  docker logs "$minio_container_id" >&2 || true
  exit 1
fi

# The root MinIO identity bypasses bucket-policy denies. Provision a dedicated
# writer identity whose only denied operation is publishing Vane's final
# committed marker. The REST catalog keeps using the root identity so a test can
# prove that the Iceberg catalog commit succeeded before marker publication
# failed.
docker exec \
  --env "VANE_TEST_ROOT_ACCESS_KEY=$access_key" \
  --env "VANE_TEST_ROOT_SECRET_KEY=$secret_key" \
  --env "VANE_TEST_MARKER_FAULT_ACCESS_KEY=$marker_fault_access_key" \
  --env "VANE_TEST_MARKER_FAULT_SECRET_KEY=$marker_fault_secret_key" \
  --env "VANE_TEST_MARKER_FAULT_POLICY=$marker_fault_policy" \
  --env "VANE_TEST_BUCKET=$bucket" \
  --interactive \
  "$minio_container_id" \
  sh <<'SH'
set -eu

mc alias set vane-test http://127.0.0.1:9000 "$VANE_TEST_ROOT_ACCESS_KEY" "$VANE_TEST_ROOT_SECRET_KEY" >/dev/null
mc admin user add \
  vane-test \
  "$VANE_TEST_MARKER_FAULT_ACCESS_KEY" \
  "$VANE_TEST_MARKER_FAULT_SECRET_KEY" >/dev/null

policy_file=/tmp/vane-marker-fault-policy.json
cat >"$policy_file" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:*"],
      "Resource": [
        "arn:aws:s3:::$VANE_TEST_BUCKET",
        "arn:aws:s3:::$VANE_TEST_BUCKET/*"
      ]
    },
    {
      "Effect": "Deny",
      "Action": ["s3:PutObject"],
      "Resource": ["arn:aws:s3:::$VANE_TEST_BUCKET/*data.duckdb_commit/*/committed"]
    }
  ]
}
EOF
mc admin policy create vane-test "$VANE_TEST_MARKER_FAULT_POLICY" "$policy_file" >/dev/null
mc admin policy attach \
  vane-test \
  "$VANE_TEST_MARKER_FAULT_POLICY" \
  --user "$VANE_TEST_MARKER_FAULT_ACCESS_KEY" >/dev/null
rm -f "$policy_file"
SH

export TEST_MINIO_MARKER_FAULT_ACCESS_KEY="$marker_fault_access_key"
export TEST_MINIO_MARKER_FAULT_SECRET_KEY="$marker_fault_secret_key"

rest_container_id="$(
  docker run \
    --detach \
    --name "$rest_name" \
    --label ai.astrovela.vane.test=iceberg-rest-catalog \
    --network "$network_name" \
    --publish 127.0.0.1::8181 \
    --env "AWS_ACCESS_KEY_ID=$access_key" \
    --env "AWS_SECRET_ACCESS_KEY=$secret_key" \
    --env "AWS_REGION=$region" \
    --env "CATALOG_WAREHOUSE=s3://$bucket/wh/" \
    --env CATALOG_IO__IMPL=org.apache.iceberg.aws.s3.S3FileIO \
    --env CATALOG_S3_ENDPOINT=http://minio:9000 \
    "$rest_image"
)"

rest_published_address="$(docker port "$rest_container_id" 8181/tcp)"
rest_published_port="${rest_published_address##*:}"
if [[ ! "$rest_published_port" =~ ^[1-9][0-9]*$ ]]; then
  echo "Could not resolve the published Iceberg REST port from: $rest_published_address" >&2
  exit 1
fi
rest_endpoint="http://127.0.0.1:$rest_published_port"

export TEST_ICEBERG_REST_ENDPOINT="$rest_endpoint"
export TEST_ICEBERG_REST_CONTAINER_ID="$rest_container_id"
export VANE_REQUIRE_ICEBERG_REST_TEST=1

if ! "$python_bin" - <<'PY'; then
import os
import time
import urllib.request

url = os.environ["TEST_ICEBERG_REST_ENDPOINT"].rstrip("/") + "/v1/config"
last_error = None
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                break
    except Exception as exc:
        last_error = exc
        time.sleep(0.25)
else:
    raise RuntimeError("Iceberg REST Catalog did not become ready within 30 seconds") from last_error
PY
  docker logs "$rest_container_id" >&2 || true
  exit 1
fi

cd "$project_root"
if ! "$project_root/scripts/run_installed_pytest.sh" \
  -m "external_service and iceberg_rest" \
  tests/fast/test_distributed_iceberg_rest.py; then
  docker logs "$rest_container_id" >&2 || true
  docker logs "$minio_container_id" >&2 || true
  exit 1
fi
