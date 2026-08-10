// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/function/scalar/image_functions.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression.hpp"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace duckdb {
namespace {

constexpr std::uint64_t PNG_MAX_DIMENSION = std::numeric_limits<std::uint32_t>::max();
constexpr std::size_t PILLOW_MAX_BLOCK = 65536;
constexpr std::size_t RGB_BYTES_PER_PIXEL = 3;
constexpr int PILLOW_COMPRESSION_LEVEL = 2;

struct RgbRegion {
	const std::uint8_t *frame_data;
	const SelectionVector *frame_selection;
	const ValidityMask *frame_validity;
	const char *function_name;
	idx_t frame_offset;
	std::size_t source_row_stride;
	std::int64_t source_width;
	std::int64_t source_height;
	std::int64_t left;
	std::int64_t top;
	std::int64_t right;
	std::uint32_t width;
	std::uint32_t height;
};

struct ImageCropBindData : FunctionData {
	ImageCropBindData(idx_t height_p, idx_t width_p, idx_t frame_size_p, string function_name_p)
	    : height(height_p), width(width_p), frame_size(frame_size_p), function_name(std::move(function_name_p)) {
	}

	idx_t height;
	idx_t width;
	idx_t frame_size;
	string function_name;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ImageCropBindData>(height, width, frame_size, function_name);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ImageCropBindData>();
		return height == other.height && width == other.width && frame_size == other.frame_size &&
		       function_name == other.function_name;
	}
};

enum class NativeImageFormat : std::uint8_t { PNG };

struct ImageEncodeBindData : FunctionData {
	explicit ImageEncodeBindData(NativeImageFormat format_p) : format(format_p) {
	}

	NativeImageFormat format;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ImageEncodeBindData>(format);
	}

	bool Equals(const FunctionData &other_p) const override {
		return format == other_p.Cast<ImageEncodeBindData>().format;
	}
};

class DeflateStream {
public:
	explicit DeflateStream(int compression_level) {
		stream.zalloc = nullptr;
		stream.zfree = nullptr;
		stream.opaque = nullptr;
		const auto result = deflateInit2(&stream, compression_level, Z_DEFLATED, 15, 9, Z_FILTERED);
		if (result != Z_OK) {
			throw InternalException("failed to initialize Pillow-compatible PNG compression");
		}
		initialized = true;
	}

	DeflateStream(const DeflateStream &) = delete;
	DeflateStream &operator=(const DeflateStream &) = delete;

	~DeflateStream() {
		if (initialized) {
			deflateEnd(&stream);
		}
	}

	z_stream stream {};

private:
	bool initialized = false;
};

void AppendBigEndian32(std::vector<std::uint8_t> &output, std::uint32_t value) {
	output.push_back(static_cast<std::uint8_t>(value >> 24));
	output.push_back(static_cast<std::uint8_t>(value >> 16));
	output.push_back(static_cast<std::uint8_t>(value >> 8));
	output.push_back(static_cast<std::uint8_t>(value));
}

void AppendPngChunk(std::vector<std::uint8_t> &output, const char (&chunk_type)[5], const std::uint8_t *data,
                    std::size_t size) {
	if (size > std::numeric_limits<std::uint32_t>::max()) {
		throw OutOfRangeException("Pillow-compatible PNG chunk exceeds the format limit");
	}
	AppendBigEndian32(output, static_cast<std::uint32_t>(size));
	output.insert(output.end(), chunk_type, chunk_type + 4);
	if (size > 0) {
		output.insert(output.end(), data, data + size);
	}
	auto checksum = crc32(0, Z_NULL, 0);
	checksum = crc32(checksum, reinterpret_cast<const Bytef *>(chunk_type), 4);
	if (size > 0) {
		checksum = crc32(checksum, data, static_cast<uInt>(size));
	}
	AppendBigEndian32(output, static_cast<std::uint32_t>(checksum));
}

std::uint64_t PillowFilterScore(const std::uint8_t *values, std::size_t size) {
	std::uint64_t score = 0;
	for (std::size_t index = 0; index < size; ++index) {
		const auto value = values[index];
		score += value < 128 ? value : 256 - value;
	}
	return score;
}

std::uint8_t PillowPaethPredictor(std::uint8_t left, std::uint8_t above, std::uint8_t upper_left) {
	const auto left_distance = std::abs(static_cast<int>(above) - static_cast<int>(upper_left));
	const auto above_distance = std::abs(static_cast<int>(left) - static_cast<int>(upper_left));
	const auto upper_left_distance =
	    std::abs(static_cast<int>(left) + static_cast<int>(above) - 2 * static_cast<int>(upper_left));
	if (left_distance <= above_distance && left_distance <= upper_left_distance) {
		return left;
	}
	return above_distance <= upper_left_distance ? above : upper_left;
}

void ReadRgbRow(const RgbRegion &region, std::uint32_t output_row, std::vector<std::uint8_t> &row) {
	const auto source_y = region.top + static_cast<std::int64_t>(output_row);
	if (source_y < 0 || source_y >= region.source_height) {
		std::fill(row.begin(), row.end(), 0);
		return;
	}

	const auto source_left = std::max<std::int64_t>(region.left, 0);
	const auto source_right = std::min(region.right, region.source_width);
	if (source_right <= source_left) {
		std::fill(row.begin(), row.end(), 0);
		return;
	}
	if (source_left != region.left || source_right != region.right) {
		std::fill(row.begin(), row.end(), 0);
	}
	const auto destination_left = static_cast<std::size_t>(source_left - region.left);
	const auto copy_pixels = static_cast<std::size_t>(source_right - source_left);
	const auto source_offset = region.frame_offset + static_cast<idx_t>(source_y) * region.source_row_stride +
	                           static_cast<idx_t>(source_left) * RGB_BYTES_PER_PIXEL;
	const auto copy_bytes = copy_pixels * RGB_BYTES_PER_PIXEL;
	if ((!region.frame_selection || !region.frame_selection->IsSet()) &&
	    (!region.frame_validity || region.frame_validity->AllValid())) {
		std::memcpy(row.data() + destination_left * RGB_BYTES_PER_PIXEL, region.frame_data + source_offset, copy_bytes);
		return;
	}

	for (std::size_t index = 0; index < copy_bytes; ++index) {
		const auto source_index =
		    region.frame_selection ? region.frame_selection->get_index(source_offset + index) : source_offset + index;
		if (region.frame_validity && !region.frame_validity->RowIsValid(source_index)) {
			throw InvalidInputException("%s frame tensors cannot contain NULL pixels", region.function_name);
		}
		row[destination_left * RGB_BYTES_PER_PIXEL + index] = region.frame_data[source_index];
	}
}

const std::vector<std::uint8_t> &PillowFilterRow(const std::vector<std::uint8_t> &row,
                                                 const std::vector<std::uint8_t> &previous,
                                                 std::vector<std::uint8_t> &none, std::vector<std::uint8_t> &up,
                                                 std::vector<std::uint8_t> &prior, std::vector<std::uint8_t> &paeth) {
	const auto row_bytes = row.size();
	none[0] = 0;
	std::copy(row.begin(), row.end(), none.begin() + 1);
	const std::vector<std::uint8_t> *selected = &none;
	auto best_score = PillowFilterScore(row.data(), row_bytes);

	// This order, the strict comparisons, and the signed-byte distance metric
	// intentionally mirror Pillow 11.3.0's ImagingZipEncode implementation.
	if (best_score > 0) {
		up[0] = 2;
		for (std::size_t index = 0; index < row_bytes; ++index) {
			up[index + 1] = static_cast<std::uint8_t>(row[index] - previous[index]);
		}
		const auto score = PillowFilterScore(up.data() + 1, row_bytes);
		if (score < best_score) {
			selected = &up;
			best_score = score;
		}
	}

	if (best_score > 0) {
		prior[0] = 1;
		for (std::size_t index = 0; index < std::min(RGB_BYTES_PER_PIXEL, row_bytes); ++index) {
			prior[index + 1] = row[index];
		}
		for (std::size_t index = RGB_BYTES_PER_PIXEL; index < row_bytes; ++index) {
			prior[index + 1] = static_cast<std::uint8_t>(row[index] - row[index - RGB_BYTES_PER_PIXEL]);
		}
		const auto score = PillowFilterScore(prior.data() + 1, row_bytes);
		if (score < best_score) {
			selected = &prior;
			best_score = score;
		}
	}

	if (best_score > 0) {
		paeth[0] = 4;
		for (std::size_t index = 0; index < std::min(RGB_BYTES_PER_PIXEL, row_bytes); ++index) {
			paeth[index + 1] = static_cast<std::uint8_t>(row[index] - previous[index]);
		}
		for (std::size_t index = RGB_BYTES_PER_PIXEL; index < row_bytes; ++index) {
			const auto prediction = PillowPaethPredictor(row[index - RGB_BYTES_PER_PIXEL], previous[index],
			                                             previous[index - RGB_BYTES_PER_PIXEL]);
			paeth[index + 1] = static_cast<std::uint8_t>(row[index] - prediction);
		}
		const auto score = PillowFilterScore(paeth.data() + 1, row_bytes);
		if (score < best_score) {
			selected = &paeth;
		}
	}
	return *selected;
}

void DeflateBytes(DeflateStream &deflater, const std::uint8_t *input, std::size_t input_size, int flush,
                  std::vector<std::uint8_t> &compressed, std::vector<std::uint8_t> &buffer) {
	if (input_size > std::numeric_limits<uInt>::max()) {
		throw OutOfRangeException("Pillow-compatible PNG row exceeds zlib's input limit");
	}
	deflater.stream.next_in = const_cast<Bytef *>(input);
	deflater.stream.avail_in = static_cast<uInt>(input_size);
	for (;;) {
		deflater.stream.next_out = buffer.data();
		deflater.stream.avail_out = static_cast<uInt>(buffer.size());
		const auto result = deflate(&deflater.stream, flush);
		if (result != Z_OK && result != Z_STREAM_END) {
			throw InternalException("zlib failed to produce a Pillow-compatible PNG stream");
		}
		const auto produced = buffer.size() - deflater.stream.avail_out;
		compressed.insert(compressed.end(), buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(produced));
		if (result == Z_STREAM_END || (deflater.stream.avail_in == 0 && deflater.stream.avail_out > 0)) {
			return;
		}
	}
}

std::vector<std::uint8_t> EncodePng(const RgbRegion &region, int compression_level) {
	const auto row_bytes = static_cast<std::size_t>(region.width) * RGB_BYTES_PER_PIXEL;
	std::vector<std::uint8_t> row(row_bytes);
	std::vector<std::uint8_t> previous(row_bytes, 0);
	std::vector<std::uint8_t> none(row_bytes + 1);
	std::vector<std::uint8_t> up(row_bytes + 1);
	std::vector<std::uint8_t> prior(row_bytes + 1);
	std::vector<std::uint8_t> paeth(row_bytes + 1);
	std::vector<std::uint8_t> compressed;
	std::vector<std::uint8_t> deflate_buffer(PILLOW_MAX_BLOCK);
	DeflateStream deflater(compression_level);

	for (std::uint32_t output_row = 0; output_row < region.height; ++output_row) {
		ReadRgbRow(region, output_row, row);
		const auto &filtered = PillowFilterRow(row, previous, none, up, prior, paeth);
		DeflateBytes(deflater, filtered.data(), filtered.size(), Z_NO_FLUSH, compressed, deflate_buffer);
		previous.swap(row);
	}
	DeflateBytes(deflater, nullptr, 0, Z_FINISH, compressed, deflate_buffer);

	static constexpr std::uint8_t signature[] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
	std::vector<std::uint8_t> output(signature, signature + sizeof(signature));
	std::vector<std::uint8_t> header;
	header.reserve(13);
	AppendBigEndian32(header, region.width);
	AppendBigEndian32(header, region.height);
	header.insert(header.end(), {8, 2, 0, 0, 0});
	AppendPngChunk(output, "IHDR", header.data(), header.size());

	const auto pillow_buffer_size = std::max<std::size_t>(PILLOW_MAX_BLOCK, static_cast<std::size_t>(region.width) * 4);
	for (std::size_t offset = 0; offset < compressed.size(); offset += pillow_buffer_size) {
		const auto size = std::min(pillow_buffer_size, compressed.size() - offset);
		AppendPngChunk(output, "IDAT", compressed.data() + offset, size);
	}
	AppendPngChunk(output, "IEND", nullptr, 0);
	return output;
}

std::int64_t TruncateLikePythonInt(double value) {
	if (std::isnan(value)) {
		throw InvalidInputException("cannot convert float NaN to integer");
	}
	if (std::isinf(value)) {
		throw OutOfRangeException("cannot convert float infinity to integer");
	}
	const auto truncated = std::trunc(value);
	if (truncated >= static_cast<double>(std::numeric_limits<long>::max()) ||
	    truncated < static_cast<double>(std::numeric_limits<long>::min())) {
		throw OutOfRangeException("Python int too large to convert to C long");
	}
	return static_cast<std::int64_t>(truncated);
}

[[noreturn]] void RaisePillowEmptyTileError() {
	throw InvalidInputException("tile cannot extend outside image");
}

std::size_t GetRgbByteSize(std::uint32_t width, std::uint32_t height, const string &function_name) {
	const auto pixel_count = static_cast<std::uint64_t>(width) * height;
	if (pixel_count > string_t::MAX_STRING_SIZE / RGB_BYTES_PER_PIXEL) {
		throw OutOfRangeException("%s RGB pixels exceed DuckDB's BLOB size limit", function_name);
	}
	return static_cast<std::size_t>(pixel_count * RGB_BYTES_PER_PIXEL);
}

std::vector<std::uint8_t> CropPixels(const RgbRegion &crop, std::size_t output_size) {
	const auto row_bytes = static_cast<std::size_t>(crop.width) * RGB_BYTES_PER_PIXEL;
	std::vector<std::uint8_t> pixels(output_size);
	std::vector<std::uint8_t> row(row_bytes);
	for (std::uint32_t output_row = 0; output_row < crop.height; ++output_row) {
		ReadRgbRow(crop, output_row, row);
		if (row_bytes > 0) {
			std::memcpy(pixels.data() + static_cast<std::size_t>(output_row) * row_bytes, row.data(), row_bytes);
		}
	}
	return pixels;
}

unique_ptr<FunctionData> BindImageCrop(ClientContext &, ScalarFunction &bound_function,
                                       vector<unique_ptr<Expression>> &arguments) {
	auto &function_name = bound_function.name;
	if (arguments.size() != 2) {
		throw BinderException("%s expects a frame tensor and one bounding box", function_name);
	}
	if (arguments[0]->HasParameter() || arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}

	const auto &frame_type = arguments[0]->return_type;
	if (!TensorType::IsTensor(frame_type)) {
		throw BinderException("%s frame must be a TENSOR(UTINYINT, [height, width, 3])", function_name);
	}
	if (TensorType::GetChildType(frame_type) != LogicalType::UTINYINT) {
		throw BinderException("%s frame tensor elements must be UTINYINT", function_name);
	}
	const auto shape = TensorType::GetShape(frame_type);
	if (shape.size() != 3 || shape[2] != RGB_BYTES_PER_PIXEL) {
		throw BinderException("%s frame must have shape [height, width, 3]", function_name);
	}
	const auto frame_size = TensorType::GetFlattenedSize(frame_type);
	bound_function.arguments[0] = frame_type;

	const auto &bbox_type = arguments[1]->return_type;
	if (bbox_type.id() == LogicalTypeId::LIST) {
		if (!ListType::GetChildType(bbox_type).IsNumeric()) {
			throw BinderException("%s bbox must contain numeric coordinates", function_name);
		}
		bound_function.arguments[1] = LogicalType::LIST(LogicalType::DOUBLE);
	} else if (bbox_type.id() == LogicalTypeId::ARRAY) {
		if (ArrayType::GetSize(bbox_type) != 4) {
			throw BinderException("%s bbox array must contain exactly four coordinates", function_name);
		}
		if (!ArrayType::GetChildType(bbox_type).IsNumeric()) {
			throw BinderException("%s bbox must contain numeric coordinates", function_name);
		}
		bound_function.arguments[1] = LogicalType::ARRAY(LogicalType::DOUBLE, 4);
	} else {
		throw BinderException("%s bbox must be a numeric LIST or four-element ARRAY", function_name);
	}

	return make_uniq<ImageCropBindData>(shape[0], shape[1], frame_size, function_name);
}

void ExecuteImageCrop(DataChunk &arguments, ExpressionState &state, Vector &result) {
	auto &bind_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<ImageCropBindData>();
	const auto count = arguments.size();
	auto &frame_vector = arguments.data[0];
	auto &bbox_vector = arguments.data[1];

	UnifiedVectorFormat frame_rows;
	frame_vector.ToUnifiedFormat(count, frame_rows);
	auto &frame_children = ArrayVector::GetEntry(frame_vector);
	UnifiedVectorFormat frame_pixels;
	frame_children.ToUnifiedFormat(ArrayVector::GetTotalSize(frame_vector), frame_pixels);
	const auto *frame_data = frame_pixels.GetData<std::uint8_t>();

	UnifiedVectorFormat bbox_rows;
	bbox_vector.ToUnifiedFormat(count, bbox_rows);
	const auto bbox_is_list = bbox_vector.GetType().id() == LogicalTypeId::LIST;
	auto &bbox_children = bbox_is_list ? ListVector::GetEntry(bbox_vector) : ArrayVector::GetEntry(bbox_vector);
	const auto bbox_child_count =
	    bbox_is_list ? ListVector::GetListSize(bbox_vector) : ArrayVector::GetTotalSize(bbox_vector);
	UnifiedVectorFormat bbox_values;
	bbox_children.ToUnifiedFormat(bbox_child_count, bbox_values);
	const auto *bbox_data = bbox_values.GetData<double>();
	const auto *bbox_entries = bbox_is_list ? bbox_rows.GetData<list_entry_t>() : nullptr;

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &output_validity = FlatVector::Validity(result);
	output_validity.SetAllValid(count);
	auto &image_children = StructVector::GetEntries(result);
	for (auto &child : image_children) {
		child->SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::Validity(*child).SetAllValid(count);
	}
	auto *width_output = FlatVector::GetData<std::uint32_t>(*image_children[0]);
	auto *height_output = FlatVector::GetData<std::uint32_t>(*image_children[1]);
	auto &pixels_output = *image_children[2];
	auto *pixels_data = FlatVector::GetData<string_t>(pixels_output);
	for (idx_t row = 0; row < count; ++row) {
		const auto frame_index = frame_rows.sel->get_index(row);
		const auto bbox_index = bbox_rows.sel->get_index(row);
		if (!frame_rows.validity.RowIsValid(frame_index) || !bbox_rows.validity.RowIsValid(bbox_index)) {
			output_validity.SetInvalid(row);
			continue;
		}

		idx_t bbox_offset;
		if (bbox_is_list) {
			const auto entry = bbox_entries[bbox_index];
			if (entry.length != 4) {
				throw InvalidInputException("%s bbox must contain exactly four coordinates", bind_data.function_name);
			}
			bbox_offset = entry.offset;
		} else {
			bbox_offset = bbox_index * 4;
		}

		double coordinates[4];
		for (idx_t coordinate = 0; coordinate < 4; ++coordinate) {
			const auto value_index = bbox_values.sel->get_index(bbox_offset + coordinate);
			if (!bbox_values.validity.RowIsValid(value_index)) {
				throw InvalidInputException("%s bbox coordinates cannot be NULL", bind_data.function_name);
			}
			coordinates[coordinate] = bbox_data[value_index];
		}

		const auto left = TruncateLikePythonInt(coordinates[0]);
		const auto top = TruncateLikePythonInt(coordinates[1]);
		const auto right = TruncateLikePythonInt(coordinates[2]);
		const auto bottom = TruncateLikePythonInt(coordinates[3]);
		if (right < left) {
			throw InvalidInputException("Coordinate 'right' is less than 'left'");
		}
		if (bottom < top) {
			throw InvalidInputException("Coordinate 'lower' is less than 'upper'");
		}
		const auto crop_width = static_cast<std::uint64_t>(right) - static_cast<std::uint64_t>(left);
		const auto crop_height = static_cast<std::uint64_t>(bottom) - static_cast<std::uint64_t>(top);
		if (crop_width > PNG_MAX_DIMENSION || crop_height > PNG_MAX_DIMENSION) {
			throw OutOfRangeException("crop dimensions exceed the PNG format limit");
		}
		const auto crop_byte_size = GetRgbByteSize(static_cast<std::uint32_t>(crop_width),
		                                           static_cast<std::uint32_t>(crop_height), bind_data.function_name);

		RgbRegion crop {frame_data,
		                frame_pixels.sel,
		                &frame_pixels.validity,
		                bind_data.function_name.c_str(),
		                frame_index * bind_data.frame_size,
		                bind_data.width * RGB_BYTES_PER_PIXEL,
		                static_cast<std::int64_t>(bind_data.width),
		                static_cast<std::int64_t>(bind_data.height),
		                left,
		                top,
		                right,
		                static_cast<std::uint32_t>(crop_width),
		                static_cast<std::uint32_t>(crop_height)};
		auto pixels = CropPixels(crop, crop_byte_size);
		width_output[row] = crop.width;
		height_output[row] = crop.height;
		const auto *pixel_data = pixels.empty() ? "" : reinterpret_cast<const char *>(pixels.data());
		pixels_data[row] = StringVector::AddStringOrBlob(pixels_output, pixel_data, pixels.size());
	}
}

unique_ptr<FunctionData> BindImageEncode(ClientContext &context, ScalarFunction &,
                                         vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 2) {
		throw BinderException("image_encode expects an RGB image and a format");
	}
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("image_encode format must be a constant string");
	}
	const auto value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	if (value.IsNull()) {
		throw BinderException("image_encode format cannot be NULL");
	}
	const auto format = StringUtil::Lower(value.GetValue<string>());
	if (format == "png") {
		return make_uniq<ImageEncodeBindData>(NativeImageFormat::PNG);
	}
	throw NotImplementedException("image_encode format '%s' is not supported; supported formats: png", format);
}

void ExecuteImageEncode(DataChunk &arguments, ExpressionState &state, Vector &result) {
	auto &bind_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<ImageEncodeBindData>();
	if (bind_data.format != NativeImageFormat::PNG) {
		throw InternalException("image_encode bound an unknown native image format");
	}
	const auto count = arguments.size();
	auto &images = arguments.data[0];
	images.Flatten(count);
	auto &image_validity = FlatVector::Validity(images);
	auto &image_children = StructVector::GetEntries(images);
	for (auto &child : image_children) {
		child->Flatten(count);
	}
	const auto *widths = FlatVector::GetData<std::uint32_t>(*image_children[0]);
	const auto *heights = FlatVector::GetData<std::uint32_t>(*image_children[1]);
	const auto *pixels = FlatVector::GetData<string_t>(*image_children[2]);
	const auto &width_validity = FlatVector::Validity(*image_children[0]);
	const auto &height_validity = FlatVector::Validity(*image_children[1]);
	const auto &pixels_validity = FlatVector::Validity(*image_children[2]);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto *output = FlatVector::GetData<string_t>(result);
	auto &output_validity = FlatVector::Validity(result);
	output_validity.SetAllValid(count);
	for (idx_t row = 0; row < count; ++row) {
		if (!image_validity.RowIsValid(row)) {
			output_validity.SetInvalid(row);
			continue;
		}
		if (!width_validity.RowIsValid(row) || !height_validity.RowIsValid(row) || !pixels_validity.RowIsValid(row)) {
			throw InvalidInputException("image_encode RGB image fields cannot be NULL");
		}
		const auto width = widths[row];
		const auto height = heights[row];
		if (width == 0 || height == 0) {
			RaisePillowEmptyTileError();
		}
		const auto expected_size = GetRgbByteSize(width, height, "image_encode");
		if (expected_size != pixels[row].GetSize()) {
			throw InvalidInputException("image_encode RGB pixel size does not match width and height");
		}

		RgbRegion image {reinterpret_cast<const std::uint8_t *>(pixels[row].GetData()),
		                 nullptr,
		                 nullptr,
		                 "image_encode",
		                 0,
		                 static_cast<std::size_t>(width) * RGB_BYTES_PER_PIXEL,
		                 width,
		                 height,
		                 0,
		                 0,
		                 width,
		                 width,
		                 height};
		auto encoded = EncodePng(image, PILLOW_COMPRESSION_LEVEL);
		if (encoded.size() > std::numeric_limits<std::uint32_t>::max()) {
			throw OutOfRangeException("encoded PNG exceeds DuckDB's BLOB size limit");
		}
		output[row] =
		    StringVector::AddStringOrBlob(result, reinterpret_cast<const char *>(encoded.data()), encoded.size());
	}
}

} // namespace

ScalarFunction ImageCropFunction::GetFunction() {
	ScalarFunction function("image_crop", {LogicalType::ANY, LogicalType::ANY}, ImageType::Create(ImageMode::RGB8),
	                        ExecuteImageCrop, BindImageCrop);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	return function;
}

ScalarFunction ImageEncodeFunction::GetFunction() {
	ScalarFunction function("image_encode", {ImageType::Create(ImageMode::RGB8), LogicalType::VARCHAR},
	                        LogicalType::BLOB, ExecuteImageEncode, BindImageEncode);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	return function;
}

} // namespace duckdb
