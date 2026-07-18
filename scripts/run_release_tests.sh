#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Keep this gate limited to tests that exercise the supported base installation.
# Optional provider, benchmark, compatibility, and external-service suites run
# separately because they need additional dependencies or infrastructure.
python -m pytest \
  --import-mode=importlib \
  -o pythonpath=tests \
  tests/fast/test_package_metadata.py \
  tests/fast/test_transformers_provider_security.py \
  tests/fast/test_vane_config.py \
  tests/fast/test_expression_udf_contracts.py \
  tests/fast/test_local_e2e.py \
  tests/fast/test_ray_cpp_bindings.py \
  tests/fast/test_ray_result_contract.py \
  tests/fast/test_fte_production_readiness.py
