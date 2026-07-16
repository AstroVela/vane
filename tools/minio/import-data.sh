#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

export_defaults
resolve_docker

DATA_DIR="${1:-.}"
MC_MIRROR_ARGS="${MC_MIRROR_ARGS:---overwrite}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Data directory does not exist: ${DATA_DIR}" >&2
  exit 1
fi

require_command realpath

DATA_REALPATH="$(realpath "${DATA_DIR}")"
STORAGE_REALPATH="$(realpath -m "${MINIO_STORAGE_DIR}")"

if [[ "${DATA_REALPATH}" == "${STORAGE_REALPATH}" ]]; then
  echo "Refusing to mirror MinIO's own storage directory back into MinIO: ${DATA_REALPATH}" >&2
  exit 1
fi

wait_for_minio

"${DOCKER[@]}" run --rm --network host \
  -v "${DATA_REALPATH}:/src:ro" \
  -e MINIO_ALIAS="${MINIO_ALIAS}" \
  -e MINIO_BUCKET="${MINIO_BUCKET}" \
  -e MINIO_ENDPOINT="${MINIO_ENDPOINT}" \
  -e MINIO_ROOT_USER="${MINIO_ROOT_USER}" \
  -e MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD}" \
  -e MC_MIRROR_ARGS="${MC_MIRROR_ARGS}" \
  --entrypoint /bin/sh \
  minio/mc:latest \
  -eu -c '
    mc alias set "$MINIO_ALIAS" "$MINIO_ENDPOINT" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
    mc mb -p "$MINIO_ALIAS/$MINIO_BUCKET"
    mc mirror $MC_MIRROR_ARGS /src "$MINIO_ALIAS/$MINIO_BUCKET"
  '
