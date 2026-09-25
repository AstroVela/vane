// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

void InitializeDynamicExtensionBindings(pybind11::module_ &module);

} // namespace duckdb
