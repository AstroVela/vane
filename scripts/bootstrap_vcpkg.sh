#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

project_root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
project_root="$(cd "$project_root" && pwd)"
triplet="${VCPKG_TARGET_TRIPLET:-x64-linux}"
vcpkg_root="${VCPKG_ROOT:-${RUNNER_TEMP:-$project_root/.cache}/vcpkg}"
install_root="${VCPKG_INSTALLED_DIR:-$project_root/vcpkg_installed}"

python_cmd=""
for candidate in \
  "${VANE_BOOTSTRAP_PYTHON:-}" \
  /opt/python/cp310-cp310/bin/python \
  "$(command -v python3 2>/dev/null || true)" \
  "$(command -v python 2>/dev/null || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]] \
    && "$candidate" -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info < (3, 15)))'; then
    python_cmd="$candidate"
    break
  fi
done

if [[ -z "$python_cmd" ]]; then
  echo "Python 3.10 through 3.14 is required to bootstrap Vane dependencies." >&2
  exit 1
fi

baseline="$(
  "$python_cmd" - "$project_root/vcpkg.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as manifest:
    print(json.load(manifest)["builtin-baseline"])
PY
)"

if [[ -e "$vcpkg_root" && ! -d "$vcpkg_root/.git" ]]; then
  echo "VCPKG_ROOT exists but is not a Git checkout: $vcpkg_root" >&2
  exit 1
fi

if [[ ! -d "$vcpkg_root/.git" ]]; then
  mkdir -p "$(dirname "$vcpkg_root")"
  git init "$vcpkg_root"
  git -C "$vcpkg_root" remote add origin https://github.com/microsoft/vcpkg.git
fi

if ! git -C "$vcpkg_root" cat-file -e "$baseline^{commit}" 2>/dev/null; then
  git -C "$vcpkg_root" fetch --depth 1 origin "$baseline"
fi
git -C "$vcpkg_root" checkout --detach "$baseline"

export VCPKG_DISABLE_METRICS=1
export VCPKG_DEFAULT_BINARY_CACHE="${VCPKG_DEFAULT_BINARY_CACHE:-${RUNNER_TEMP:-$project_root/.cache}/vcpkg-archives}"
mkdir -p "$VCPKG_DEFAULT_BINARY_CACHE" "$install_root"

"$vcpkg_root/bootstrap-vcpkg.sh" -disableMetrics
"$vcpkg_root/vcpkg" install \
  --x-manifest-root="$project_root" \
  --x-install-root="$install_root" \
  --triplet="$triplet"

"$python_cmd" "$project_root/scripts/sync_vcpkg_licenses.py" \
  --share-dir "$install_root/$triplet/share" \
  --output "$project_root/LICENSES/vcpkg-binary-dependencies.txt" \
  --check
