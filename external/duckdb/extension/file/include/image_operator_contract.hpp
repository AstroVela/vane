// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/image.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

//! Shared validation and buffer access; pixel algorithms belong to their backend.
struct ImageOperatorContract {
	static constexpr idx_t MIB = 1024 * 1024;
	static constexpr idx_t MAX_BYTES = 256 * MIB;
	static constexpr idx_t MAX_PIXELS = 100000000;
	static constexpr idx_t COPY_BYTES = MIB;

	static void Interrupt(ClientContext &context) {
		if (context.IsInterrupted()) {
			throw InterruptException();
		}
	}

	static idx_t CheckSize(uint64_t width, uint64_t height, uint16_t channels, idx_t remaining) {
		if (channels < 1 || channels > 4) {
			throw InvalidInputException("Image operators require one to four UInt8 channels");
		}
		if (!width || !height || width > NumericLimits<uint32_t>::Maximum() ||
		    height > NumericLimits<uint32_t>::Maximum()) {
			throw InvalidInputException("Image dimensions must be positive UINTEGER values");
		}
		if (width > MAX_PIXELS / height || width * height > remaining / channels) {
			throw OutOfRangeException("Image operator exceeds its pixel or byte limit");
		}
		return idx_t(width * height * channels);
	}

	static LogicalType BindImage(ScalarFunction &function, vector<unique_ptr<Expression>> &arguments) {
		auto type = arguments[0]->return_type;
		if (type.id() == LogicalTypeId::UNKNOWN) {
			throw ParameterNotResolvedException();
		}
		if (type.id() == LogicalTypeId::SQLNULL) {
			type = ImageLogicalType::Create();
		}
		if (!ImageLogicalType::IsImage(type)) {
			throw BinderException("%s requires IMAGE, not %s", function.name, type);
		}
		function.arguments[0] = type;
		return type;
	}

	static unique_ptr<FunctionData> BindCrop(ClientContext &, ScalarFunction &function,
	                                         vector<unique_ptr<Expression>> &arguments) {
		auto type = BindImage(function, arguments);
		auto mode = ImageLogicalType::GetMode(type);
		function.return_type = mode.empty() ? ImageLogicalType::Create() : ImageLogicalType::Create(mode);
		auto bbox = arguments[1]->return_type;
		if (bbox.id() == LogicalTypeId::UNKNOWN || bbox.id() == LogicalTypeId::SQLNULL) {
			function.arguments[1] = LogicalType::LIST(LogicalType::BIGINT);
		} else if (bbox.id() == LogicalTypeId::LIST && ListType::GetChildType(bbox).IsIntegral()) {
			function.arguments[1] = LogicalType::LIST(LogicalType::BIGINT);
		} else if (bbox.id() == LogicalTypeId::ARRAY && ArrayType::GetSize(bbox) == 4 &&
		           ArrayType::GetChildType(bbox).IsIntegral()) {
			function.arguments[1] = LogicalType::ARRAY(LogicalType::BIGINT, 4);
		} else {
			throw BinderException("crop bbox must be an integer LIST or a four-element integer ARRAY");
		}
		return nullptr;
	}

	static unique_ptr<FunctionData> BindEncode(ClientContext &, ScalarFunction &function,
	                                           vector<unique_ptr<Expression>> &arguments) {
		BindImage(function, arguments);
		return nullptr;
	}

	static bool ReadPNGFormat(Vector &input, idx_t row) {
		UnifiedVectorFormat values;
		input.ToUnifiedFormat(row + 1, values);
		auto selected = values.sel->get_index(row);
		if (!values.validity.RowIsValid(selected)) {
			return false;
		}
		auto value = UnifiedVectorFormat::GetData<string_t>(values)[selected];
		auto data = value.GetData();
		if (value.GetSize() != 3 || (data[0] != 'P' && data[0] != 'p') || (data[1] != 'N' && data[1] != 'n') ||
		    (data[2] != 'G' && data[2] != 'g')) {
			throw NotImplementedException("encode_image currently supports PNG only");
		}
		return true;
	}
};

struct ImagePixelView {
	ImageLayout layout;
	const_data_ptr_t data;
};

//! Read selected metadata and the existing contiguous pixel payload. In
//! particular, a constant HD Image is never broadcast into a full input batch.
class ImageOperatorInput {
public:
	ImageOperatorInput(Vector &input, idx_t count)
	    : input(input), fixed(ImageLogicalType::IsFixedShape(input.GetType())), pixels(LogicalType::UTINYINT, nullptr) {
		input.ToUnifiedFormat(count, rows);
		if (fixed) {
			pixel_count = ArrayVector::GetTotalSize(input);
			pixels.Reference(ArrayVector::GetEntry(input));
		} else {
			auto &children = StructVector::GetEntries(input);
			for (idx_t i = 0; i < 5; i++) {
				children[i]->ToUnifiedFormat(count, fields[i]);
			}
			auto &data = *children[ImageLogicalType::DATA];
			pixel_count = ListVector::GetListSize(data);
			pixels.Reference(ListVector::GetEntry(data));
		}
		pixels.Flatten(pixel_count);
	}

	bool IsNull(idx_t row) const {
		return !rows.validity.RowIsValid(rows.sel->get_index(row));
	}

	bool Read(idx_t row, ImagePixelView &view) {
		auto selected = rows.sel->get_index(row);
		if (IsNull(row)) {
			return false;
		}
		list_entry_t range;
		if (fixed) {
			auto &type = input.GetType();
			auto mode = ImageLogicalType::GetMode(type);
			view.layout = {ImageLogicalType::GetWidth(type), ImageLogicalType::GetHeight(type),
			               ImageLogicalType::ChannelsForMode(mode), ImageLogicalType::ModeCode(mode)};
			auto size = ArrayType::GetSize(type);
			range = list_entry_t(selected * size, size);
		} else {
			// STRUCT children already carry dictionary selections. Index them
			// with the logical row, without applying the parent's selection twice.
			view.layout = {
			    Field<uint32_t>(ImageLogicalType::WIDTH, row), Field<uint32_t>(ImageLogicalType::HEIGHT, row),
			    Field<uint16_t>(ImageLogicalType::CHANNELS, row), Field<uint8_t>(ImageLogicalType::MODE, row)};
			range = Field<list_entry_t>(ImageLogicalType::DATA, row);
		}
		auto mode = ImageLogicalType::ModeName(view.layout.mode);
		auto channels = ImageLogicalType::ChannelsForMode(mode);
		ImageOperatorContract::CheckSize(view.layout.width, view.layout.height, channels,
		                                 ImageOperatorContract::MAX_BYTES);
		ImageLogicalType::ValidateFields(range.length, view.layout.width, view.layout.height, view.layout.channels,
		                                 mode, "Image operator");
		ImageLogicalType::ValidateShape(input.GetType(), view.layout.width, view.layout.height, mode, "Image operator");
		if (range.offset > pixel_count || range.length > pixel_count - range.offset) {
			throw InvalidInputException("Image operator received invalid pixel offsets");
		}
		auto &validity = FlatVector::Validity(pixels);
		if (!validity.AllValid()) {
			for (idx_t i = range.offset; i < range.offset + range.length; i++) {
				if (!validity.RowIsValid(i)) {
					throw InvalidInputException("Image operator pixels cannot contain NULL");
				}
			}
		}
		view.data = FlatVector::GetData<uint8_t>(pixels) + range.offset;
		return true;
	}

private:
	template <class T>
	T Field(idx_t index, idx_t row) {
		auto &field = fields[index];
		auto selected = field.sel->get_index(row);
		if (!field.validity.RowIsValid(selected)) {
			throw InvalidInputException("Non-NULL IMAGE values cannot contain NULL fields");
		}
		return UnifiedVectorFormat::GetData<T>(field)[selected];
	}

	Vector &input;
	bool fixed;
	Vector pixels;
	idx_t pixel_count;
	UnifiedVectorFormat rows;
	UnifiedVectorFormat fields[5];
};

struct ImageCropBox {
	int64_t x;
	int64_t y;
	uint32_t width;
	uint32_t height;

	static bool Read(Vector &input, idx_t row, ImageCropBox &box) {
		UnifiedVectorFormat rows;
		input.ToUnifiedFormat(row + 1, rows);
		auto selected = rows.sel->get_index(row);
		if (!rows.validity.RowIsValid(selected)) {
			return false;
		}
		auto list = input.GetType().id() == LogicalTypeId::LIST;
		auto range = list ? UnifiedVectorFormat::GetData<list_entry_t>(rows)[selected] : list_entry_t(selected * 4, 4);
		if (range.length != 4) {
			throw InvalidInputException("crop bbox must contain exactly four integers: x, y, width, height");
		}
		auto &child = list ? ListVector::GetEntry(input) : ArrayVector::GetEntry(input);
		auto child_size = list ? ListVector::GetListSize(input) : ArrayVector::GetTotalSize(input);
		if (range.offset > child_size || range.length > child_size - range.offset) {
			throw InvalidInputException("crop bbox contains invalid list offsets");
		}
		int64_t values[4];
		for (idx_t i = 0; i < 4; i++) {
			auto value = child.GetValue(range.offset + i);
			if (value.IsNull()) {
				throw InvalidInputException("crop bbox coordinates cannot be NULL");
			}
			values[i] = value.GetValue<int64_t>();
		}
		if (values[2] <= 0 || values[3] <= 0 || values[2] > NumericLimits<uint32_t>::Maximum() ||
		    values[3] > NumericLimits<uint32_t>::Maximum()) {
			throw InvalidInputException("crop width and height must be positive UINTEGER values");
		}
		box = {values[0], values[1], uint32_t(values[2]), uint32_t(values[3])};
		return true;
	}
};

} // namespace duckdb
