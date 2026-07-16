#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

export_defaults
resolve_docker

"${DOCKER[@]}" compose -f "${COMPOSE_FILE}" down
