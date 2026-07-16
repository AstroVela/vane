#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

export_defaults
resolve_docker

mkdir -p "${MINIO_STORAGE_DIR}"

"${DOCKER[@]}" compose -f "${COMPOSE_FILE}" up -d
wait_for_minio
print_connection_info
