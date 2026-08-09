// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// vane_python/jupyter_progress_bar_display.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"
#include "duckdb/common/progress_bar/progress_bar_display.hpp"
#include "duckdb/common/helper.hpp"

namespace duckdb {

class JupyterProgressBarDisplay : public ProgressBarDisplay {
public:
	JupyterProgressBarDisplay();
	virtual ~JupyterProgressBarDisplay() {
	}

	static unique_ptr<ProgressBarDisplay> Create();

public:
	void Update(double progress);
	void Finish();

private:
	void Initialize();

private:
	py::object progress_bar;
};

} // namespace duckdb
