// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/pipeline_node/data_sink_finish.hpp"
#include "duckdb/execution/operator/helper/physical_data_sink.hpp"

namespace duckdb {
namespace distributed {

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateDataSink(
    const PhysicalDataSink &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	if (children.size() != 1 || !children[0] || !children[0]->inner()) {
		throw NotImplementedException("Distributed DataSink requires exactly one child operator");
	}
	return std::make_shared<DataSinkFinishNode>(get_next_pipeline_node_id(), children[0]->inner(), op.operation_id);
}

} // namespace distributed
} // namespace duckdb
