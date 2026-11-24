// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "ray_binding_internal.hpp"

namespace duckdb {

void register_ray_bindings(py::module_ &mod) {
	auto m = mod.def_submodule("ray_cxx");
	m.doc() = "C++ Ray execution bindings (experimental)";
	ray_bindings::RegisterRayCoreBindings(m);
	ray_bindings::RegisterRayPlansBindings(m);
	ray_bindings::RegisterRayRunnerBindings(m);
	ray_bindings::RegisterRayExecutionTestsBindings(m);
	ray_bindings::RegisterRayExchangeTestsBindings(m);
	ray_bindings::RegisterRayShuffleTestsBindings(m);
	ray_bindings::RegisterRayRecoveryTestsBindings(m);
	ray_bindings::RegisterRayDescriptorTestsBindings(m);
	ray_bindings::RegisterRayMiscTestsBindings(m);
}

} // namespace duckdb
