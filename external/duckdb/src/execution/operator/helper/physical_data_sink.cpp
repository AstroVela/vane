// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/helper/physical_data_sink.hpp"

#include "duckdb/common/serializer/serializer.hpp"

namespace duckdb {

void PhysicalDataSink::SerializeOperatorData(Serializer &serializer) const {
	serializer.WriteProperty<string>(103, "operation_id", operation_id);
}

} // namespace duckdb
