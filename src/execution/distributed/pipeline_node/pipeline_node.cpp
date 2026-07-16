// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/exchange/exchange.hpp"
#include "duckdb/execution/distributed/pipeline_node/scan_source.hpp"

namespace duckdb {
namespace distributed {

void RecordRemoteExchangeFinishedSinks(Exchange &exchange, const std::vector<MaterializedOutput> &outputs,
                                       const char *mismatch_context) {
	for (idx_t i = 0; i < outputs.size(); i++) {
		if (!outputs[i].has_exchange_sink_instance()) {
			throw InternalException("%s exchange sink output is missing exchange sink instance metadata at index %llu",
			                        mismatch_context ? mismatch_context : "remote", i);
		}
		std::string node_id;
		auto worker_id = outputs[i].worker_id();
		if (worker_id) {
			node_id = *worker_id;
		}
		const auto &sink_instance = outputs[i].exchange_sink_instance();
		exchange.SinkFinished(sink_instance.sink_handle, sink_instance.attempt_id, node_id, outputs[i].flight_port());
	}
}

size_t DistributedPipelineNode::num_partitions() const {
	return op_->config().clustering_spec()->num_partitions();
}

bool DistributedPipelineNode::try_get_scan_tasks(std::vector<ScanTaskDescriptor> &out) const {
	if (!op_) {
		return false;
	}
	auto scan_node = std::dynamic_pointer_cast<ScanSourceNode>(op_);
	if (!scan_node) {
		return false;
	}
	const auto &tasks = scan_node->scan_tasks();
	if (tasks.empty()) {
		return false;
	}
	out = tasks;
	return true;
}

std::vector<std::shared_ptr<DistributedPipelineNode>> DistributedPipelineNode::arc_children() const {
	return children_;
}

} // namespace distributed
} // namespace duckdb
