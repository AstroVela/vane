#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

require_command() {
  local cmd

  for cmd in "$@"; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      echo "Required command not found: ${cmd}" >&2
      exit 1
    fi
  done
}

export_defaults() {
  export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-duckdb-minio}"
  : "${MINIO_ROOT_USER:?Set MINIO_ROOT_USER for the local development server}"
  : "${MINIO_ROOT_PASSWORD:?Set MINIO_ROOT_PASSWORD for the local development server}"
  export MINIO_ROOT_USER
  export MINIO_ROOT_PASSWORD
  export MINIO_STORAGE_DIR="${MINIO_STORAGE_DIR:-${HOME}/minio-data}"
  export MINIO_API_PORT="${MINIO_API_PORT:-9000}"
  export MINIO_CONSOLE_PORT="${MINIO_CONSOLE_PORT:-9001}"
  export MINIO_BUCKET="${MINIO_BUCKET:-datasets}"
  export MINIO_ALIAS="${MINIO_ALIAS:-local}"
  export MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://127.0.0.1:${MINIO_API_PORT}}"
}

resolve_docker() {
  require_command docker

  if docker info >/dev/null 2>&1; then
    DOCKER=(docker)
    return
  fi

  if command -v sudo >/dev/null 2>&1; then
    DOCKER=(sudo -E docker)
    return
  fi

  echo "Docker daemon is not accessible, and sudo is unavailable." >&2
  exit 1
}

wait_for_minio() {
  local max_attempts="${1:-30}"
  local attempt=1

  require_command curl

  while ((attempt <= max_attempts)); do
    if curl -fsS "${MINIO_ENDPOINT}/minio/health/live" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    ((attempt++))
  done

  echo "MinIO did not become ready at ${MINIO_ENDPOINT} within ${max_attempts}s." >&2
  exit 1
}

print_connection_info() {
  cat <<EOF
MinIO is ready.
API endpoint: ${MINIO_ENDPOINT}
Console: http://127.0.0.1:${MINIO_CONSOLE_PORT}
Access key: ${MINIO_ROOT_USER}
Secret key: ${MINIO_ROOT_PASSWORD}
Bucket: ${MINIO_BUCKET}
EOF
}
