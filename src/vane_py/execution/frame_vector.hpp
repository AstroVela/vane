// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector_buffer.hpp"

#include <cstring>

namespace duckdb {
namespace vane_execution {
namespace {

idx_t AddSize(idx_t left, idx_t right) {
	if (right > NumericLimits<idx_t>::Maximum() - left) {
		throw InvalidInputException("direct frame size overflow");
	}
	return left + right;
}

idx_t Align(idx_t size) {
	return AddSize(size, 7) & ~idx_t(7);
}

idx_t FrameWidth(const LogicalType &type) {
	auto physical = type.InternalType();
	return physical == PhysicalType::STRUCT || physical == PhysicalType::ARRAY ? 0 : GetTypeIdSize(physical);
}

struct FrameColumn {
	idx_t validity = 0;
	idx_t values = 0;
	idx_t count = 0;
	vector<FrameColumn> children;
};

// Measurement and copying traverse precisely the same selected values. Lists
// are compacted: unused elements and borrowed operator buffers never enter a
// frame. The limit bounds even the temporary selection used for nested values.
bool VisitFrameVector(Vector &column, idx_t source_count, const vector<idx_t> &rows, FrameColumn &layout, idx_t &size,
                      idx_t limit, data_ptr_t buffer = nullptr) {
	const auto &type = column.GetType();
	auto width = FrameWidth(type);
	auto count = rows.size();
	if (count > limit / MaxValue<idx_t>(1, width)) {
		return false;
	}
	layout.count = count;
	layout.validity = Align(size);
	layout.values = AddSize(layout.validity, ValidityMask::ValidityMaskSize(count));
	size = AddSize(layout.values, count * width);
	if (size > limit) {
		return false;
	}
	UnifiedVectorFormat data;
	column.ToUnifiedFormat(source_count, data);
	auto index_at = [&](idx_t row) {
		return data.sel->get_index(rows[row]);
	};
	if (buffer) {
		ValidityMask validity(reinterpret_cast<validity_t *>(buffer + layout.validity), count);
		for (idx_t row = 0; row < count; row++) {
			if (data.validity.RowIsValid(index_at(row))) {
				validity.SetValid(row);
			}
		}
	}
	auto physical = type.InternalType();
	if (physical == PhysicalType::STRUCT) {
		auto &children = StructVector::GetEntries(column);
		layout.children.resize(children.size());
		vector<idx_t> selected;
		selected.reserve(count);
		idx_t child_count = 0;
		for (idx_t row = 0; row < count; row++) {
			auto index = index_at(row);
			selected.push_back(index);
			child_count = MaxValue(child_count, index + 1);
		}
		for (idx_t i = 0; i < children.size(); i++) {
			if (!VisitFrameVector(*children[i], child_count, selected, layout.children[i], size, limit, buffer)) {
				return false;
			}
		}
	} else if (physical == PhysicalType::LIST || physical == PhysicalType::ARRAY) {
		auto &child = physical == PhysicalType::LIST ? ListVector::GetEntry(column) : ArrayVector::GetEntry(column);
		auto child_capacity =
		    physical == PhysicalType::LIST ? ListVector::GetListSize(column) : ArrayVector::GetTotalSize(column);
		vector<idx_t> child_rows;
		auto entries = physical == PhysicalType::LIST ? UnifiedVectorFormat::GetData<list_entry_t>(data) : nullptr;
		auto array_size = physical == PhysicalType::ARRAY ? ArrayType::GetSize(type) : 0;
		idx_t child_count = 0;
		for (idx_t row = 0; row < count; row++) {
			auto index = index_at(row);
			auto length = entries ? (data.validity.RowIsValid(index) ? entries[index].length : 0) : array_size;
			child_count = AddSize(child_count, length);
			if (child_count > limit / MaxValue<idx_t>(1, FrameWidth(child.GetType()))) {
				return false;
			}
		}
		child_rows.reserve(child_count);
		for (idx_t row = 0; row < count; row++) {
			auto index = index_at(row);
			auto start = entries ? (data.validity.RowIsValid(index) ? entries[index].offset : 0) : index * array_size;
			auto length = entries ? (data.validity.RowIsValid(index) ? entries[index].length : 0) : array_size;
			if (start > child_capacity || length > child_capacity - start) {
				throw InvalidInputException(
				    "invalid nested direct frame selection for %s: offset %llu, length %llu, size %llu",
				    type.ToString(), start, length, child_capacity);
			}
			if (buffer && entries) {
				reinterpret_cast<list_entry_t *>(buffer + layout.values)[row] = {child_rows.size(), length};
			}
			for (idx_t i = 0; i < length; i++) {
				child_rows.push_back(start + i);
			}
		}
		layout.children.resize(1);
		if (!VisitFrameVector(child, child_capacity, child_rows, layout.children[0], size, limit, buffer)) {
			return false;
		}
	} else {
		for (idx_t row = 0; row < count; row++) {
			auto index = index_at(row);
			if (!data.validity.RowIsValid(index)) {
				continue;
			}
			if (physical == PhysicalType::VARCHAR) {
				auto value = UnifiedVectorFormat::GetData<string_t>(data)[index];
				if (!value.IsInlined()) {
					auto start = size;
					size = AddSize(size, value.GetSize());
					if (size > limit) {
						return false;
					}
					if (buffer) {
						memcpy(buffer + start, value.GetData(), value.GetSize());
						value = string_t(reinterpret_cast<const char *>(buffer + start), value.GetSize());
					}
				}
				if (buffer) {
					reinterpret_cast<string_t *>(buffer + layout.values)[row] = value;
				}
			} else if (buffer) {
				memcpy(buffer + layout.values + width * row, data.data + width * index, width);
			}
		}
	}
	return true;
}

idx_t Measure(DataChunk &input, const vector<idx_t> &rows, idx_t offset, idx_t count, idx_t limit,
              vector<FrameColumn> *columns = nullptr, data_ptr_t buffer = nullptr) {
	if (!count || offset > rows.size() || count > rows.size() - offset) {
		throw InvalidInputException("invalid direct frame row selection");
	}
	vector<idx_t> selected;
	for (idx_t i = offset; i < offset + count; i++) {
		if (rows[i] >= input.size()) {
			throw InvalidInputException("direct frame row selection is out of bounds");
		}
		selected.push_back(rows[i]);
	}
	idx_t size = 0;
	for (auto &column : input.data) {
		FrameColumn layout;
		if (!VisitFrameVector(column, input.size(), selected, layout, size, limit, buffer)) {
			return AddSize(limit, 1);
		}
		if (columns) {
			columns->push_back(std::move(layout));
		}
	}
	return size;
}

template <class BUFFER>
class LeasedNestedBuffer : public BUFFER {
public:
	template <class... ARGS>
	LeasedNestedBuffer(shared_ptr<DirectBatch> batch, ARGS &&...args)
	    : BUFFER(std::forward<ARGS>(args)...), batch(std::move(batch)) {
	}
	shared_ptr<DirectBatch> batch;
};

class LeaseBuffer : public VectorBuffer {
public:
	explicit LeaseBuffer(shared_ptr<DirectBatch> batch)
	    : VectorBuffer(VectorBufferType::OPAQUE_BUFFER), batch(std::move(batch)) {
	}
	shared_ptr<DirectBatch> batch;
};

unique_ptr<Vector> ReferenceFrameVector(const LogicalType &type, const FrameColumn &layout, data_ptr_t buffer,
                                        shared_ptr<DirectBatch> batch) {
	auto view = make_uniq<Vector>(type, buffer + layout.values);
	FlatVector::SetValidity(*view,
	                        ValidityMask(reinterpret_cast<validity_t *>(buffer + layout.validity), layout.count));
	switch (type.InternalType()) {
	case PhysicalType::STRUCT: {
		auto children = make_buffer<LeasedNestedBuffer<VectorStructBuffer>>(batch);
		auto &types = StructType::GetChildTypes(type);
		for (idx_t i = 0; i < types.size(); i++) {
			children->GetChildren().push_back(ReferenceFrameVector(types[i].second, layout.children[i], buffer, batch));
		}
		view->SetAuxiliary(std::move(children));
		break;
	}
	case PhysicalType::LIST: {
		auto child = ReferenceFrameVector(ListType::GetChildType(type), layout.children[0], buffer, batch);
		auto children =
		    make_buffer<LeasedNestedBuffer<VectorListBuffer>>(batch, std::move(child), layout.children[0].count);
		children->SetSize(layout.children[0].count);
		view->SetAuxiliary(std::move(children));
		break;
	}
	case PhysicalType::ARRAY: {
		auto child = ReferenceFrameVector(ArrayType::GetChildType(type), layout.children[0], buffer, batch);
		auto children = make_buffer<LeasedNestedBuffer<VectorArrayBuffer>>(batch, std::move(child),
		                                                                   ArrayType::GetSize(type), layout.count);
		view->SetAuxiliary(std::move(children));
		break;
	}
	case PhysicalType::VARCHAR: {
		auto strings = make_buffer<VectorStringBuffer>();
		strings->AddHeapReference(make_buffer<LeaseBuffer>(batch));
		view->SetAuxiliary(std::move(strings));
		break;
	}
	default:
		view->SetAuxiliary(make_buffer<LeaseBuffer>(batch));
	}
	return view;
}

} // namespace
} // namespace vane_execution
} // namespace duckdb
