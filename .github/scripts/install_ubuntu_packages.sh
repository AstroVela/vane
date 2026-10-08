#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

# The Ubuntu 24.04 hosted runner uses this mirror list. Its Azure HTTP mirror
# can stall on package indexes even after InRelease succeeds on another mirror.
set -euo pipefail

if (($# == 0)); then
  echo "Usage: install_ubuntu_packages.sh PACKAGE [...]" >&2
  exit 2
fi

printf '%s\n' 'https://archive.ubuntu.com/ubuntu/' \
  | sudo tee /etc/apt/apt-mirrors.txt >/dev/null
sudo tee /etc/apt/apt.conf.d/80vane-ci >/dev/null <<'APT'
Acquire::Retries "2";
Acquire::http::Timeout "30";
Acquire::https::Timeout "30";
APT

# Bound both stalled connections and the whole command. Do not silently use
# stale package indexes if any repository failed to update.
timeout --kill-after=10s 3m sudo apt-get -o APT::Update::Error-Mode=any update
timeout --kill-after=10s 7m sudo apt-get install -y "$@"
