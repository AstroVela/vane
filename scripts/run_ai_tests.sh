#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shard="${1:-all}"
if (($#)); then
  shift
fi

marker='not external_service and not real_ray and not gpu'
targets=("$project_root/tests/ai")
expression_targets=("$project_root"/tests/ai/test_expression*.py)
case "$shard" in
  all)
    for selected in core expressions ray; do
      "$project_root/scripts/run_ai_tests.sh" "$selected" "$@"
    done
    exit 0
    ;;
  core)
    for path in "${expression_targets[@]}"; do
      targets+=(--ignore="$path")
    done
    ;;
  expressions)
    targets=("${expression_targets[@]}")
    ;;
  ray)
    marker='not external_service and real_ray and not gpu'
    ;;
  *)
    printf '%s\n' "Usage: $0 [all|core|expressions|ray] [PYTEST_ARG ...]" >&2
    exit 2
    ;;
esac

exec "$project_root/scripts/run_installed_pytest.sh" \
  --quiet --maxfail=1 -m "$marker" "${targets[@]}" "$@"
