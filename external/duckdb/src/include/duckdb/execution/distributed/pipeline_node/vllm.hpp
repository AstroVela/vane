// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/operator/projection/physical_vllm.hpp"
#include "yyjson.hpp"

#include <cctype>

namespace duckdb {
namespace distributed {

// ── vLLM pool-name helpers (used by both VLLMProjectNode and collect_vllm_nodes) ──

inline std::string SanitizePoolComponent(const std::string &value) {
	std::string result;
	result.reserve(value.size());
	for (auto ch : value) {
		if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-') {
			result.push_back(ch);
		} else {
			result.push_back('_');
		}
	}
	if (result.empty()) {
		result = "unknown";
	}
	return result;
}

inline std::string BuildPoolName(const PipelineNodeContext &ctx, NodeID node_id) {
	auto query_id = ctx.query_id();
	if (query_id.empty()) {
		query_id = std::to_string(ctx.query_idx());
	}
	auto safe_query = SanitizePoolComponent(query_id);
	return "duckdb_vllm_" + safe_query + "_" + std::to_string(node_id);
}

inline bool TryMergeJsonOptions(const std::string &json, const std::string &pool_name, std::string &out) {
	using namespace duckdb_yyjson;
	if (json.empty()) {
		return false;
	}
	constexpr auto READ_FLAG = YYJSON_READ_ALLOW_INVALID_UNICODE;
	yyjson_read_err err;
	auto *doc = yyjson_read_opts(const_cast<char *>(json.data()), json.size(), READ_FLAG, nullptr, &err); // NOLINT
	if (!doc) {
		return false;
	}
	auto *root = yyjson_doc_get_root(doc);
	if (!root || yyjson_get_type(root) != YYJSON_TYPE_OBJ) {
		yyjson_doc_free(doc);
		return false;
	}
	auto *mut_doc = yyjson_doc_mut_copy(doc, nullptr);
	yyjson_doc_free(doc);
	if (!mut_doc) {
		return false;
	}
	auto *mut_root = yyjson_mut_doc_get_root(mut_doc);
	if (!mut_root || !yyjson_mut_is_obj(mut_root)) {
		yyjson_mut_doc_free(mut_doc);
		return false;
	}

	auto put_bool = [&](const char *key, bool value) {
		auto *k = yyjson_mut_str(mut_doc, key);
		auto *v = value ? yyjson_mut_true(mut_doc) : yyjson_mut_false(mut_doc);
		yyjson_mut_obj_put(mut_root, k, v);
	};
	auto put_str = [&](const char *key, const std::string &value) {
		auto *k = yyjson_mut_str(mut_doc, key);
		auto *v = yyjson_mut_strncpy(mut_doc, value.c_str(), value.size());
		yyjson_mut_obj_put(mut_root, k, v);
	};

	put_bool("use_ray", true);
	put_bool("ray_worker_only", true);
	put_str("ray_actor_pool_name", pool_name);

	yyjson_write_err write_err;
	size_t len = 0;
	constexpr auto WRITE_FLAG = YYJSON_WRITE_ALLOW_INVALID_UNICODE;
	char *buf = yyjson_mut_val_write_opts(mut_root, WRITE_FLAG, nullptr, &len, &write_err);
	if (!buf) {
		yyjson_mut_doc_free(mut_doc);
		return false;
	}
	out.assign(buf, len);
	free(buf);
	yyjson_mut_doc_free(mut_doc);
	return true;
}

inline Value InjectDistributedOptions(const Value &options, const std::string &pool_name) {
	if (options.IsNull()) {
		child_list_t<Value> child_list = {
		    {"use_ray", Value::BOOLEAN(true)},
		    {"ray_worker_only", Value::BOOLEAN(true)},
		    {"ray_actor_pool_name", Value(pool_name)},
		};
		return Value::STRUCT(std::move(child_list));
	}

	const auto type_id = options.type().id();
	if (type_id == LogicalTypeId::STRUCT) {
		child_list_t<Value> child_list;
		const auto &values = StructValue::GetChildren(options);
		const auto child_count = StructType::GetChildCount(options.type());
		child_list.reserve(child_count + 3);
		for (idx_t i = 0; i < child_count; i++) {
			child_list.emplace_back(StructType::GetChildName(options.type(), i), values[i]);
		}
		auto upsert = [&](const std::string &name, Value value) {
			for (auto &entry : child_list) {
				if (entry.first == name) {
					entry.second = std::move(value);
					return;
				}
			}
			child_list.emplace_back(name, std::move(value));
		};
		upsert("use_ray", Value::BOOLEAN(true));
		upsert("ray_worker_only", Value::BOOLEAN(true));
		upsert("ray_actor_pool_name", Value(pool_name));
		return Value::STRUCT(std::move(child_list));
	}

	if (type_id == LogicalTypeId::VARCHAR || options.type().IsJSONType()) {
		const auto &json = StringValue::Get(options);
		std::string merged;
		if (TryMergeJsonOptions(json, pool_name, merged)) {
			return Value(merged);
		}
	}

	return options;
}

// Extract a pre-injected ray_actor_pool_name from options (if any).
// Returns empty string when the field is absent or the type is unrecognised.
inline std::string ExtractPoolNameFromOptions(const Value &options) {
	if (options.IsNull()) {
		return "";
	}
	const auto type_id = options.type().id();
	if (type_id == LogicalTypeId::STRUCT) {
		const auto child_count = StructType::GetChildCount(options.type());
		for (idx_t i = 0; i < child_count; i++) {
			if (StructType::GetChildName(options.type(), i) == "ray_actor_pool_name") {
				const auto &child = StructValue::GetChildren(options)[i];
				if (!child.IsNull() && child.type().id() == LogicalTypeId::VARCHAR) {
					auto val = StringValue::Get(child);
					if (!val.empty()) {
						return val;
					}
				}
				break;
			}
		}
		return "";
	}
	if (type_id == LogicalTypeId::VARCHAR || options.type().IsJSONType()) {
		// Quick JSON extraction via yyjson.
		using namespace duckdb_yyjson;
		const auto &json = StringValue::Get(options);
		if (json.empty()) {
			return "";
		}
		constexpr auto READ_FLAG = YYJSON_READ_ALLOW_INVALID_UNICODE;
		yyjson_read_err err;
		auto *doc = yyjson_read_opts(const_cast<char *>(json.data()), json.size(), READ_FLAG, nullptr, &err); // NOLINT
		if (!doc) {
			return "";
		}
		auto *root = yyjson_doc_get_root(doc);
		std::string result;
		if (root && yyjson_get_type(root) == YYJSON_TYPE_OBJ) {
			auto *val = yyjson_obj_get(root, "ray_actor_pool_name");
			if (val && yyjson_is_str(val)) {
				result = yyjson_get_str(val);
			}
		}
		yyjson_doc_free(doc);
		return result;
	}
	return "";
}

class VLLMProjectNode : public PipelineNodeImpl, public std::enable_shared_from_this<VLLMProjectNode> {
public:
	VLLMProjectNode(NodeID node_id, PipelineNodeRef child, ExpressionRef prompt_expr, std::string model, Value options,
	                std::string output_column_name, duckdb::vector<LogicalType> output_types)
	    : ctx_(InheritPipelineNodeContext(child, node_id, "VLLMProject")),
	      config_(BuildSchema(output_types), child ? child->config().execution_config() : DuckDBExecutionConfigRef(),
	              child ? child->config().clustering_spec() : ClusteringSpec::unknown_with_num_partitions(1)),
	      child_(std::move(child)), prompt_expr_(std::move(prompt_expr)), model_(std::move(model)),
	      options_(std::move(options)), output_column_name_(std::move(output_column_name)),
	      output_types_(std::move(output_types)) {
	}

	std::string name() const override {
		return "VLLMProject";
	}
	NodeID node_id() const override {
		return ctx_.node_id();
	}
	const PipelineNodeContext &context() const override {
		return ctx_;
	}
	const PipelineNodeConfig &config() const override {
		return config_;
	}

	std::vector<PipelineNodeRef> children() const override {
		return {child_};
	}

	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override {
		auto input_stream = child_->produce_tasks(plan_context);

		auto prompt_expr = prompt_expr_;
		auto model = model_;
		auto options = options_;
		auto output_name = output_column_name_;
		auto output_types = output_types_;
		auto this_node_id = this->node_id();
		// Prefer a pool name pre-injected by collect_vllm_nodes(); fall back
		// to computing one from the pipeline-node context when absent.
		auto existing_pool = ExtractPoolNameFromOptions(options);
		auto pool_name = existing_pool.empty() ? BuildPoolName(this->ctx_, this_node_id) : existing_pool;

		return input_stream.pipeline_instruction(
		    shared_from_this(),
		    [prompt_expr, model, options, output_name, output_types,
		     pool_name](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
			    if (!prompt_expr) {
				    return input_plan;
			    }
			    auto expr_copy = prompt_expr->Copy();
			    auto out_types = output_types;
			    if (out_types.empty()) {
				    out_types = input_plan->Root().GetTypes();
			    }
			    idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;
			    auto &old_root = input_plan->Root();
			    auto model_copy = model;
			    auto options_copy = InjectDistributedOptions(options, pool_name);
			    auto &vllm = input_plan->Make<::duckdb::PhysicalVLLM>(
			        out_types, duckdb::unique_ptr<duckdb::Expression>(expr_copy.release()), std::move(model_copy),
			        std::move(options_copy), output_name, estimated_cardinality);
			    vllm.children.push_back(old_root);
			    input_plan->SetRoot(vllm);
			    return input_plan;
		    },
		    plan_context.client_context());
	}

	std::vector<std::string> multiline_display(bool verbose) const override {
		std::string expr_name = prompt_expr_ ? prompt_expr_->GetName() : std::string("<none>");
		std::string output_name = output_column_name_.empty() ? std::string("<unnamed>") : output_column_name_;
		return {std::string("VLLM: ") + expr_name + " -> " + output_name};
	}

private:
	static SchemaRef BuildSchema(const duckdb::vector<LogicalType> &output_types) {
		if (output_types.empty()) {
			return nullptr;
		}
		return std::make_shared<duckdb::LogicalType>(output_types[0]);
	}

	PipelineNodeContext ctx_;
	PipelineNodeConfig config_;
	PipelineNodeRef child_;
	ExpressionRef prompt_expr_;
	std::string model_;
	Value options_;
	std::string output_column_name_;
	duckdb::vector<LogicalType> output_types_;
};

} // namespace distributed
} // namespace duckdb
