#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if (($# == 0)); then
  printf '%s\n' "Usage: scripts/run_installed_pytest.sh PYTEST_ARG [...]" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
site_packages="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
test_workdir="$(mktemp -d "${TMPDIR:-/tmp}/vane-pytest.XXXXXX")"
cleanup_test_workdir() {
  rm -rf -- "$test_workdir"
}
trap cleanup_test_workdir EXIT

pytest_args=()
for argument in "$@"; do
  case "$argument" in
    tests/*)
      pytest_args+=("$project_root/$argument")
      ;;
    ./tests/*)
      pytest_args+=("$project_root/${argument#./}")
      ;;
    *)
      pytest_args+=("$argument")
      ;;
  esac
done

cd "$test_workdir"
export PYTHONSAFEPATH=1
export PYTHONPATH="${site_packages}:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export VANE_FAST_TEST_ARTIFACT_MODE=1

python -m pytest \
  -c "$project_root/pyproject.toml" \
  --rootdir="$project_root" \
  --import-mode=importlib \
  -o "pythonpath=$project_root/tests" \
  "${pytest_args[@]}"
