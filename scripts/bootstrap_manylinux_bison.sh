#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

bison_version="3.8.2"
bison_sha256="06c9e13bdf7eb24d4ceb6b59205a4f67c2c7e7213119644430fe82fbd14a0abb"
bison_urls=(
  "https://ftp.gnu.org/gnu/bison/bison-${bison_version}.tar.gz"
  "https://mirrors.kernel.org/gnu/bison/bison-${bison_version}.tar.gz"
)
install_prefix="${VANE_BISON_INSTALL_PREFIX:-/usr/local}"
build_root="${VANE_BUILD_TOOLS_DIR:-/tmp/vane-build-tools}"
archive="${build_root}/bison-${bison_version}.tar.gz"
source_dir="${build_root}/bison-${bison_version}"
bison_bin="${install_prefix}/bin/bison"

if [[ -x "$bison_bin" ]] \
  && [[ "$($bison_bin --version | sed -n '1s/.* //p')" == "$bison_version" ]]; then
  echo "Using cached GNU Bison ${bison_version} from ${bison_bin}"
  exit 0
fi

mkdir -p "$build_root"
if [[ -f "$archive" ]] \
  && ! echo "${bison_sha256}  ${archive}" | sha256sum --check --status; then
  rm -f "$archive"
fi

if [[ ! -f "$archive" ]]; then
  for bison_url in "${bison_urls[@]}"; do
    # manylinux_2_28 ships curl 7.61; use flags supported by that baseline.
    if curl --fail --location --retry 2 --retry-delay 2 \
      --connect-timeout 15 --max-time 120 --output "${archive}.part" "$bison_url" \
      && echo "${bison_sha256}  ${archive}.part" | sha256sum --check; then
      mv "${archive}.part" "$archive"
      break
    fi
    rm -f "${archive}.part"
  done
  if [[ ! -f "$archive" ]]; then
    echo "Failed to download verified GNU Bison ${bison_version} from all mirrors" >&2
    exit 1
  fi
fi
echo "${bison_sha256}  ${archive}" | sha256sum --check

rm -rf "$source_dir"
tar -xzf "$archive" -C "$build_root"

jobs="${VANE_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
if ((jobs > 16)); then
  jobs=16
fi

(
  cd "$source_dir"
  ./configure --prefix="$install_prefix"
  make -j"$jobs"
  make install
)

if [[ "$($bison_bin --version | sed -n '1s/.* //p')" != "$bison_version" ]]; then
  echo "Failed to install GNU Bison ${bison_version} at ${bison_bin}" >&2
  exit 1
fi

echo "Installed GNU Bison ${bison_version} at ${bison_bin}"
