// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/copy_to_file.hpp"
#include "duckdb/execution/distributed/data_sink.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

namespace duckdb {
namespace distributed {

struct FinalizedSinkResult {
	enum Tag { COPY, DATA_SINK };
	Tag tag = COPY;
	DistributedCopyResult copy_result;
	DistributedDataSinkResult data_sink_result;

	static FinalizedSinkResult MakeCopy(DistributedCopyResult result) {
		FinalizedSinkResult finalized;
		finalized.tag = COPY;
		finalized.copy_result = std::move(result);
		return finalized;
	}

	static FinalizedSinkResult MakeDataSink(DistributedDataSinkResult result) {
		FinalizedSinkResult finalized;
		finalized.tag = DATA_SINK;
		finalized.data_sink_result = std::move(result);
		return finalized;
	}
};

class FinalizableSinkNode : public PipelineNodeImpl {
public:
	bool is_sink() const final {
		return true;
	}

	virtual NodeID result_node_id() const = 0;
	virtual DuckDBResult<FinalizedSinkResult> finalize(const std::vector<ResultPartitionRef> &partitions,
	                                                   ClientContext &context) = 0;
};

} // namespace distributed
} // namespace duckdb
