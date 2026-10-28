// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "json_scan.hpp"
#include "json_multi_file_info.hpp"
#include "duckdb/common/file_open_flags.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/function/distributed_table_function.hpp"

#include <algorithm>

namespace duckdb {

bool JSONScanRange::TryGet(const OpenFileInfo &file, JSONScanRange &range) {
	if (!file.extended_info) {
		return false;
	}
	const auto &options = file.extended_info->options;
	auto start = options.find(START_OPTION);
	auto end = options.find(END_OPTION);
	if (start == options.end() && end == options.end()) {
		return false;
	}
	if (start == options.end() || end == options.end() || start->second.type() != LogicalType::UBIGINT ||
	    end->second.type() != LogicalType::UBIGINT || start->second.IsNull() || end->second.IsNull()) {
		throw InvalidInputException("Invalid NDJSON byte range metadata");
	}
	range.start = start->second.GetValue<idx_t>();
	range.end = end->second.GetValue<idx_t>();
	if (range.start >= range.end) {
		throw InvalidInputException("Invalid NDJSON byte range offsets");
	}
	return true;
}

OpenFileInfo JSONScanRange::Strip(const OpenFileInfo &file) {
	auto result = JSONFileSnapshot(0, file).ToOpenFileInfo();
	result.extended_info->options.erase(START_OPTION);
	result.extended_info->options.erase(END_OPTION);
	return result;
}

OpenFileInfo JSONScanRange::Set(const OpenFileInfo &file, idx_t start, idx_t end) {
	auto result = Strip(file);
	result.extended_info->options[START_OPTION] = Value::UBIGINT(start);
	result.extended_info->options[END_OPTION] = Value::UBIGINT(end);
	JSONScanRange range;
	TryGet(result, range);
	return result;
}

static string JSONSplitId(const JSONFileSnapshot &file) {
	JSONScanRange range;
	return "json:" + std::to_string(file.ordinal) +
	       (JSONScanRange::TryGet(file.ToOpenFileInfo(), range)
	            ? ":" + std::to_string(range.start) + ":" + std::to_string(range.end)
	            : ":file");
}

static void ValidateJSONAssignments(const vector<JSONFileSnapshot> &files) {
	map<idx_t, vector<pair<idx_t, idx_t>>> ranges;
	set<idx_t> whole;
	for (const auto &file : files) {
		if (file.path.empty() || file.options.count(JSONFileSnapshot::ORDINAL_OPTION)) {
			throw InvalidInputException("Invalid distributed JSON file snapshot");
		}
		JSONScanRange range;
		if (JSONScanRange::TryGet(file.ToOpenFileInfo(), range)) {
			if (whole.count(file.ordinal)) {
				throw InvalidInputException("Distributed JSON assignment mixes whole-file and range work");
			}
			ranges[file.ordinal].emplace_back(range.start, range.end);
		} else if (!whole.insert(file.ordinal).second || ranges.count(file.ordinal)) {
			throw InvalidInputException("Distributed JSON assignment repeats whole-file work");
		}
	}
	for (auto &entry : ranges) {
		auto &file_ranges = entry.second;
		std::sort(file_ranges.begin(), file_ranges.end());
		for (idx_t i = 1; i < file_ranges.size(); i++) {
			if (file_ranges[i].first < file_ranges[i - 1].second) {
				throw InvalidInputException("Distributed JSON assignment has overlapping ranges");
			}
		}
	}
}

void ValidateJSONDistributedState(const JSONScanData &data, const vector<OpenFileInfo> &files) {
	map<idx_t, JSONFileSnapshot> allowed;
	for (const auto &file : data.distributed_allowed_files) {
		JSONScanRange range;
		if (file.path.empty() || file.options.count(JSONFileSnapshot::ORDINAL_OPTION) ||
		    JSONScanRange::TryGet(file.ToOpenFileInfo(), range) || !allowed.emplace(file.ordinal, file).second) {
			throw InvalidInputException("Invalid distributed JSON file authorization");
		}
	}
	if (!data.distributed_worker && (!allowed.empty() || data.distributed_splits_applied ||
	                                 data.distributed_assignment_restricted || !data.distributed_split_ids.empty())) {
		throw InvalidInputException("Coordinator JSON bind contains worker state");
	}
	if (data.distributed_splits_applied && !data.distributed_assignment_restricted) {
		throw InvalidInputException("Applied JSON worker lacks assignment authorization");
	}
	if (!data.distributed_assignment_restricted && !data.distributed_split_ids.empty()) {
		throw InvalidInputException("Unrestricted JSON worker contains split identities");
	}
	if (data.distributed_worker && !data.distributed_splits_applied && !files.empty()) {
		throw InvalidInputException("Detached JSON worker contains active files");
	}
	vector<JSONFileSnapshot> active;
	vector<string> ids;
	for (idx_t i = 0; i < files.size(); i++) {
		JSONFileSnapshot snapshot(i, files[i]);
		JSONScanRange range;
		const bool has_range = JSONScanRange::TryGet(files[i], range);
		if (has_range && (!data.distributed_worker || data.options.format != JSONFormat::NEWLINE_DELIMITED)) {
			throw InvalidInputException("JSON byte ranges require a distributed NDJSON worker");
		}
		if (data.distributed_worker) {
			idx_t ordinal;
			if (!JSONFileSnapshot::TryGetOrdinal(files[i], ordinal)) {
				throw InvalidInputException("Distributed JSON worker file has no ordinal");
			}
			auto found = allowed.find(ordinal);
			auto stripped = JSONFileSnapshot(ordinal, JSONScanRange::Strip(files[i]));
			if (found == allowed.end() || found->second.path != stripped.path ||
			    found->second.options != stripped.options) {
				throw InvalidInputException("Distributed JSON file is outside its worker bind");
			}
		}
		ids.push_back(JSONSplitId(snapshot));
		active.push_back(std::move(snapshot));
	}
	ValidateJSONAssignments(active);
	std::sort(ids.begin(), ids.end());
	if (data.distributed_worker && data.distributed_splits_applied && ids != data.distributed_split_ids) {
		throw InvalidInputException("Distributed JSON assignment does not match its authorized splits");
	}
}

static DistributedScanSplit MakeJSONSplit(const JSONFileSnapshot &file, optional_idx bytes = optional_idx()) {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	serializer.WriteProperty<idx_t>(1, "version", 1);
	serializer.WriteProperty(2, "file", file);
	serializer.End();
	DistributedScanSplit result;
	result.split_id = JSONSplitId(file);
	result.payload = string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
	result.estimated_bytes = bytes;
	return result;
}

static JSONFileSnapshot DecodeJSONSplit(const DistributedScanSplit &split) {
	split.Validate();
	MemoryStream stream(reinterpret_cast<data_ptr_t>(const_cast<char *>(split.payload.data())), split.payload.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	if (deserializer.ReadProperty<idx_t>(1, "version") != 1) {
		throw InvalidInputException("Unsupported distributed JSON split version");
	}
	auto result = deserializer.ReadProperty<JSONFileSnapshot>(2, "file");
	deserializer.End();
	if (stream.GetPosition() != split.payload.size() || split.split_id != JSONSplitId(result)) {
		throw InvalidInputException("Distributed JSON split payload does not match its identity");
	}
	return result;
}

static const MultiFileBindData &GetJSONBind(const TableFunctionDistributedScanInput &input) {
	if (!input.bind_data) {
		throw InvalidInputException("Distributed JSON scan requires bind data");
	}
	auto &bind = input.bind_data->Cast<MultiFileBindData>();
	ValidateJSONDistributedState(bind.bind_data->Cast<JSONScanData>(), bind.file_list->GetAllFiles());
	return bind;
}

static constexpr idx_t JSON_MINIMUM_SPLIT_BYTES = 1024 * 1024;

static idx_t JSONMaximumSplitBytes() {
	const auto raw = FileSystem::GetEnvVariable("VANE_NDJSON_MAX_SPLIT_BYTES");
	if (raw.empty()) {
		return 256 * 1024 * 1024;
	}
	idx_t bytes = 0;
	for (const auto digit : raw) {
		if (digit < '0' || digit > '9' || bytes > (NumericLimits<idx_t>::Maximum() - (digit - '0')) / 10) {
			throw InvalidInputException("VANE_NDJSON_MAX_SPLIT_BYTES must be an integer of at least 2097152 bytes");
		}
		bytes = bytes * 10 + (digit - '0');
	}
	// Twice the minimum allows balanced ranges to satisfy both size bounds.
	if (bytes < 2 * JSON_MINIMUM_SPLIT_BYTES) {
		throw InvalidInputException("VANE_NDJSON_MAX_SPLIT_BYTES must be an integer of at least 2097152 bytes");
	}
	return bytes;
}

static idx_t JSONCeilDivide(idx_t size, idx_t divisor) {
	return size / divisor + (size % divisor != 0);
}

struct JSONPlanningFile {
	optional_idx bytes;
	bool can_split = false;
};

static vector<DistributedScanSplit> PlanJSONSplits(const TableFunctionDistributedScanPlanningInput &input) {
	auto &bind = GetJSONBind(input);
	auto &data = bind.bind_data->Cast<JSONScanData>();
	auto files = bind.file_list->GetAllFiles();
	vector<DistributedScanSplit> result;
	// Replay the exact authorized ranges, independent of current planning settings.
	if (data.distributed_worker && data.distributed_splits_applied) {
		for (idx_t i = 0; i < files.size(); i++) {
			JSONScanRange existing;
			const auto bytes = JSONScanRange::TryGet(files[i], existing) ? optional_idx(existing.end - existing.start)
			                                                             : optional_idx();
			result.push_back(MakeJSONSplit(JSONFileSnapshot(i, files[i]), bytes));
		}
		return result;
	}
	if (files.empty()) {
		return result;
	}
	const auto maximum_bytes = JSONMaximumSplitBytes();
	vector<JSONPlanningFile> properties(files.size());
	idx_t total_bytes = 0;
	for (idx_t i = 0; i < files.size(); i++) {
		if (input.file_system.IsPipe(files[i].path)) {
			throw InvalidInputException("Distributed JSON scanning requires replayable input, not a pipe");
		}
		unique_ptr<FileHandle> handle;
		try {
			handle = input.file_system.OpenFile(files[i], FileFlags::FILE_FLAGS_READ | data.options.compression);
		} catch (const InterruptException &) {
			throw;
		} catch (const FatalException &) {
			throw;
		} catch (const InternalException &) {
			throw;
		} catch (const OutOfMemoryException &) {
			throw;
		} catch (const Exception &) {
			// Metadata is optional: execution can reopen with worker credentials.
			continue;
		}
		if (handle->IsPipe()) {
			throw InvalidInputException("Distributed JSON scanning does not support pipes");
		}
		const auto size = handle->GetFileSize();
		auto &file = properties[i];
		if (size != DConstants::INVALID_INDEX) {
			file.bytes = optional_idx(size);
			file.can_split = data.options.format == JSONFormat::NEWLINE_DELIMITED && handle->CanSeek() &&
			                 handle->GetFileCompressionType() == FileCompressionType::UNCOMPRESSED;
			if (file.can_split) {
				if (size > NumericLimits<idx_t>::Maximum() - total_bytes) {
					throw InvalidInputException("Distributed NDJSON total input size exceeds the supported byte count");
				}
				total_bytes += size;
			}
		}
	}
	// File counts do not consume a large file's parallelism budget. The byte cap
	// can produce more elementary splits than worker slots; the scheduler groups them.
	const auto target_bytes = MinValue<idx_t>(
	    maximum_bytes, MaxValue<idx_t>(JSON_MINIMUM_SPLIT_BYTES,
	                                   JSONCeilDivide(total_bytes, MaxValue<idx_t>(1, input.target_split_count))));
	for (idx_t i = 0; i < files.size(); i++) {
		JSONFileSnapshot snapshot(i, files[i]);
		const auto &file = properties[i];
		idx_t count = 1;
		if (file.can_split) {
			const auto size = file.bytes.GetIndex();
			count = MinValue<idx_t>(MaxValue<idx_t>(1, JSONCeilDivide(size, target_bytes)),
			                        MaxValue<idx_t>(1, size / JSON_MINIMUM_SPLIT_BYTES));
		}
		if (count == 1) {
			result.push_back(MakeJSONSplit(snapshot, file.bytes));
			continue;
		}
		const auto size = file.bytes.GetIndex();
		idx_t start = 0;
		for (idx_t part = 0; part < count; part++) {
			const auto end = start + size / count + (part < size % count ? 1 : 0);
			result.push_back(MakeJSONSplit(
			    JSONFileSnapshot(snapshot.ordinal, JSONScanRange::Set(snapshot.ToOpenFileInfo(), start, end)),
			    optional_idx(end - start)));
			start = end;
		}
	}
	return result;
}

static unique_ptr<FunctionData> CreateJSONWorker(const TableFunctionDistributedScanInput &input) {
	auto &source = GetJSONBind(input);
	auto &data = source.bind_data->Cast<JSONScanData>();
	auto result = make_uniq<MultiFileBindData>();
	result->column_ids = source.column_ids;
	result->bind_data = unique_ptr_cast<FunctionData, TableFunctionData>(data.Copy());
	result->file_list = make_shared_ptr<SimpleMultiFileList>(vector<OpenFileInfo> {});
	result->multi_file_reader = source.multi_file_reader->Copy();
	result->interface = source.interface->Copy();
	result->columns = source.columns;
	result->reader_bind = source.reader_bind;
	result->file_options = source.file_options;
	result->types = source.types;
	result->names = source.names;
	result->virtual_columns = source.virtual_columns;
	result->table_columns = source.table_columns;
	auto &worker = result->bind_data->Cast<JSONScanData>();
	if (!data.distributed_worker) {
		auto files = source.file_list->GetAllFiles();
		for (idx_t i = 0; i < files.size(); i++) {
			worker.distributed_allowed_files.emplace_back(i, files[i]);
		}
	}
	worker.distributed_worker = true;
	worker.distributed_splits_applied = false;
	return std::move(result);
}

static void ApplyJSONSplits(optional_ptr<FunctionData> worker_bind, const vector<DistributedScanSplit> &splits) {
	if (!worker_bind) {
		throw InvalidInputException("Distributed JSON assignment requires worker bind");
	}
	auto &bind = worker_bind->Cast<MultiFileBindData>();
	auto &data = bind.bind_data->Cast<JSONScanData>();
	if (!data.distributed_worker) {
		throw InvalidInputException("JSON splits require a detached worker bind");
	}
	ValidateJSONDistributedState(data, bind.file_list->GetAllFiles());
	vector<OpenFileInfo> files;
	vector<string> ids;
	for (const auto &split : splits) {
		files.push_back(DecodeJSONSplit(split).ToOpenFileInfo());
		ids.push_back(split.split_id);
	}
	std::sort(ids.begin(), ids.end());
	if (data.distributed_assignment_restricted && ids != data.distributed_split_ids) {
		throw InvalidInputException("Cannot replace an existing distributed JSON assignment");
	}
	auto checked = data.Copy();
	auto &candidate = checked->Cast<JSONScanData>();
	candidate.distributed_splits_applied = true;
	candidate.distributed_assignment_restricted = true;
	candidate.distributed_split_ids = ids;
	ValidateJSONDistributedState(candidate, files);
	bind.file_list = make_shared_ptr<SimpleMultiFileList>(std::move(files));
	bind.initial_reader.reset();
	bind.union_readers.clear();
	data.distributed_splits_applied = true;
	data.distributed_assignment_restricted = true;
	data.distributed_split_ids = std::move(ids);
}

TableFunctionDistributedScanCallbacks JSONDistributedScanCallbacks() {
	TableFunctionDistributedScanCallbacks callbacks;
	callbacks.protocol_version = 1;
	callbacks.split_codec = {"vane.json.scan-split", 1};
	callbacks.plan_splits = PlanJSONSplits;
	callbacks.create_worker_bind = CreateJSONWorker;
	callbacks.apply_splits = ApplyJSONSplits;
	return callbacks;
}

} // namespace duckdb
