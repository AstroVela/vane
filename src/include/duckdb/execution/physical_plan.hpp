// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

// Compatibility header: re-export PhysicalPlan from physical_plan_generator.hpp
#pragma once

#include "duckdb/execution/physical_plan_generator.hpp"

namespace duckdb {
// PhysicalPlan is defined in physical_plan_generator.hpp
using PhysicalPlan = ::duckdb::PhysicalPlan;
} // namespace duckdb
