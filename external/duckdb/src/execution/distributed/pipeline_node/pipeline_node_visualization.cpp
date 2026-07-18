// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

#include <sstream>

namespace duckdb {
namespace distributed {
namespace {

std::vector<std::string> SplitLinesForPipeline(const std::string &text) {
	std::vector<std::string> lines;
	if (text.empty()) {
		return lines;
	}
	size_t start = 0;
	while (true) {
		auto pos = text.find('\n', start);
		if (pos == std::string::npos) {
			lines.push_back(text.substr(start));
			break;
		}
		lines.push_back(text.substr(start, pos - start));
		start = pos + 1;
	}
	return lines;
}

std::vector<std::string> NodeDisplayLines(const DistributedPipelineNodeRef &node, DisplayLevel level) {
	std::string display;
	try {
		display = node->display_as(level);
	} catch (...) {
		display.clear();
	}
	auto lines = SplitLinesForPipeline(display);
	if (lines.empty()) {
		lines.push_back(node->name());
	}
	lines[0] += " [partitions=" + std::to_string(node->num_partitions()) + "]";
	return lines;
}

void AppendTree(const DistributedPipelineNodeRef &node, DisplayLevel level, const std::string &prefix, bool is_last,
                bool is_root, std::vector<std::string> &out) {
	auto lines = NodeDisplayLines(node, level);
	if (is_root) {
		out.push_back(lines[0]);
		for (size_t i = 1; i < lines.size(); ++i) {
			out.push_back("   " + lines[i]);
		}
	} else {
		out.push_back(prefix + "+- " + lines[0]);
		std::string continuation_prefix = prefix + (is_last ? "   " : "|  ");
		for (size_t i = 1; i < lines.size(); ++i) {
			out.push_back(continuation_prefix + lines[i]);
		}
	}

	auto children = node->arc_children();
	std::string child_prefix = is_root ? "" : prefix + (is_last ? "   " : "|  ");
	for (size_t idx = 0; idx < children.size(); ++idx) {
		bool child_last = (idx + 1) == children.size();
		AppendTree(children[idx], level, child_prefix, child_last, false, out);
	}
}

} // namespace

std::string viz_distributed_pipeline_mermaid(const DistributedPipelineNodeRef &root, DisplayLevel display_type,
                                             bool bottom_up, const std::string & /*subgraph_options*/) {
	std::ostringstream ss;
	ss << "graph TD;\n";
	ss << "%% Node: " << root->name() << "\n";
	ss << "%% partitions: " << root->num_partitions() << "\n";
	ss << "%% display: " << static_cast<int>(display_type) << "\n";
	ss << "%% bottom_up: " << (bottom_up ? "true" : "false") << "\n";
	return ss.str();
}

std::string viz_distributed_pipeline_ascii(const DistributedPipelineNodeRef &root, bool simple) {
	std::vector<std::string> lines;
	if (root) {
		auto level = simple ? DisplayLevel::Compact : DisplayLevel::Default;
		AppendTree(root, level, "", true, true, lines);
	}
	std::ostringstream ss;
	for (size_t i = 0; i < lines.size(); ++i) {
		ss << lines[i];
		if (i + 1 < lines.size()) {
			ss << "\n";
		}
	}
	return ss.str();
}

} // namespace distributed
} // namespace duckdb
