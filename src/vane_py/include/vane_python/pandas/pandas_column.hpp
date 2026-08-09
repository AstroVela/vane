// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#pragma once

namespace duckdb {

enum class PandasColumnBackend { NUMPY };

class PandasColumn {
public:
	PandasColumn(PandasColumnBackend backend) : backend(backend) {
	}
	virtual ~PandasColumn() {
	}

public:
	PandasColumnBackend Backend() const {
		return backend;
	}

protected:
	PandasColumnBackend backend;
};

} // namespace duckdb
