// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "image_extension.hpp"
#include "image_crop.hpp"
#include "image_operator_contract.hpp"
#include "image_codec_contract.hpp"
#include "image_codec.hpp"
#include "image_transform.hpp"
#include "image_transform_contract.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/expression_executor_state.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <zlib.h>

namespace duckdb {
namespace {

//! One bounded zlib stream, carried in PNG IDAT chunks. Encoding preserves
//! every UInt8 channel and does not perform color conversion or filesystem I/O.
class PNGEncoder {
public:
	PNGEncoder(ClientContext &context, idx_t limit) : context(context), limit(limit) {
		stream = {};
		auto code = deflateInit(&stream, Z_DEFAULT_COMPRESSION);
		if (code == Z_MEM_ERROR) {
			throw OutOfMemoryException("PNG compression initialization ran out of memory");
		}
		if (code != Z_OK) {
			throw InternalException("PNG compression initialization failed");
		}
	}

	~PNGEncoder() {
		deflateEnd(&stream);
	}

	PNGEncoder(const PNGEncoder &) = delete;
	PNGEncoder &operator=(const PNGEncoder &) = delete;

	string Encode(const ImagePixelView &image) {
		Append("\x89PNG\r\n\x1a\n", 8);
		uint8_t header[13] = {};
		BigEndian(header, image.layout.width);
		BigEndian(header + 4, image.layout.height);
		auto element_size = ImageLogicalType::ElementSize(ImageLogicalType::ModeName(image.layout.mode));
		header[8] = uint8_t(element_size * 8);
		const uint8_t colors[] = {0, 4, 2, 6};
		header[9] = colors[image.layout.channels - 1];
		Chunk("IHDR", header, sizeof(header));
		auto row_bytes = idx_t(image.layout.width) * image.layout.channels * element_size;
		auto copy = [element_size](data_ptr_t target, const_data_ptr_t source, idx_t size) {
			if (element_size == 1) {
				memcpy(target, source, size);
				return;
			}
			for (idx_t i = 0; i < size; i += 2) {
				uint16_t value;
				memcpy(&value, source + i, 2);
				target[i] = uint8_t(value >> 8);
				target[i + 1] = uint8_t(value);
			}
		};
		const uint8_t filter = 0; // PNG's lossless None filter.
		uint8_t scanlines[64 * 1024];
		if (row_bytes + 1 <= sizeof(scanlines)) {
			// Batch short scanlines so tall, narrow images do not make two
			// zlib calls per pixel row. Keep staging and cancellation bounded.
			auto stride = row_bytes + 1;
			auto rows_per_block = sizeof(scanlines) / stride;
			for (idx_t row = 0; row < image.layout.height;) {
				auto count = MinValue(idx_t(image.layout.height) - row, idx_t(rows_per_block));
				for (idx_t i = 0; i < count; i++) {
					scanlines[i * stride] = filter;
					copy(scanlines + i * stride + 1, image.data + (row + i) * row_bytes, row_bytes);
				}
				Deflate(scanlines, count * stride, Z_NO_FLUSH);
				row += count;
			}
		} else {
			for (idx_t row = 0; row < image.layout.height; row++) {
				Deflate(&filter, 1, Z_NO_FLUSH);
				for (idx_t offset = 0; offset < row_bytes;) {
					auto size = MinValue(row_bytes - offset, ImageOperatorContract::COPY_BYTES);
					if (element_size == 1) {
						Deflate(image.data + row * row_bytes + offset, size, Z_NO_FLUSH);
					} else {
						string swapped(size, '\0');
						copy(data_ptr_cast(&swapped[0]), image.data + row * row_bytes + offset, size);
						Deflate(const_data_ptr_cast(swapped.data()), size, Z_NO_FLUSH);
					}
					offset += size;
				}
			}
		}
		Deflate(nullptr, 0, Z_FINISH);
		Chunk("IEND", nullptr, 0);
		return std::move(output);
	}

private:
	static void BigEndian(data_ptr_t target, uint32_t value) {
		for (idx_t i = 0; i < 4; i++) {
			target[i] = uint8_t(value >> (24 - i * 8));
		}
	}

	void Append(const char *data, idx_t size) {
		if (size > limit - output.size()) {
			throw OutOfRangeException("PNG encoding exceeds the Image operator batch byte limit");
		}
		if (size) {
			output.append(data, size);
		}
	}

	void Chunk(const char *name, const_data_ptr_t data, idx_t size) {
		uint8_t length[4];
		BigEndian(length, uint32_t(size));
		Append(reinterpret_cast<const char *>(length), 4);
		Append(name, 4);
		Append(reinterpret_cast<const char *>(data), size);
		auto crc = crc32(0, reinterpret_cast<const Bytef *>(name), 4);
		if (size) {
			crc = crc32(crc, data, uInt(size));
		}
		BigEndian(length, uint32_t(crc));
		Append(reinterpret_cast<const char *>(length), 4);
	}

	void Deflate(const_data_ptr_t input, idx_t size, int flush) {
		stream.next_in = const_cast<Bytef *>(input);
		stream.avail_in = uInt(size);
		int code;
		do {
			ImageOperatorContract::Interrupt(context);
			stream.next_out = buffer;
			stream.avail_out = sizeof(buffer);
			code = deflate(&stream, flush);
			if (code == Z_MEM_ERROR) {
				throw OutOfMemoryException("PNG compression ran out of memory");
			}
			if (code != Z_OK && code != Z_STREAM_END) {
				throw InternalException("PNG compression failed");
			}
			auto produced = sizeof(buffer) - stream.avail_out;
			if (produced) {
				Chunk("IDAT", buffer, produced);
			}
		} while (stream.avail_in || (flush == Z_FINISH && code != Z_STREAM_END));
	}

	ClientContext &context;
	idx_t limit;
	z_stream stream;
	uint8_t buffer[64 * 1024];
	string output;
};

static void CropImage(DataChunk &args, ExpressionState &state, Vector &result) {
	auto constant = args.AllConstant();
	auto count = constant && args.size() ? idx_t(1) : args.size();
	ImageOperatorInput images(args.data[0], count, &state.GetContext());
	result.SetVectorType(VectorType::FLAT_VECTOR);
	idx_t bytes = 0;
	for (idx_t row = 0; row < count; row++) {
		auto &context = state.GetContext();
		ImageOperatorContract::Interrupt(context);
		ImagePixelView image;
		ImageCropBox box;
		if (images.IsNull(row) || !ImageCropBox::Read(args.data[1], row, box) || !images.Read(row, image)) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}
		auto size = ImageOperatorContract::CheckSize(
		    box.width, box.height, image.layout.channels, ImageOperatorContract::MAX_BYTES - bytes,
		    GetTypeIdSize(ImageLogicalType::StorageType(result.GetType()).InternalType()));
		bytes += size;
		auto layout = image.layout;
		layout.width = box.width;
		layout.height = box.height;
		ImageOperatorOutput output_pixels(result, row, layout);
		auto target = output_pixels.Data();
		CropImagePixels(image, box, target, layout.Bytes(),
		                [&context]() { ImageOperatorContract::Interrupt(context); });
		output_pixels.Finish(context);
	}
	if (constant && count) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void ResizeImage(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &context = state.GetContext();
	ImageTransformContract::Execute(
	    args, context, result, ImageTransform::RESIZE,
	    [&context](const ImagePixelView &image, const ImageLayout &layout, data_ptr_t target, bool antialias) {
		    ResizeImagePixels(
		        image, layout, target, [&context]() { ImageOperatorContract::Interrupt(context); }, antialias);
	    });
}

static void ConvertImage(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &context = state.GetContext();
	ImageTransformContract::Execute(
	    args, context, result, ImageTransform::CONVERT,
	    [&context](const ImagePixelView &image, const ImageLayout &layout, data_ptr_t target, bool) {
		    ConvertImagePixels(image, layout, target, [&context]() { ImageOperatorContract::Interrupt(context); });
	    });
}

static void EncodeImage(DataChunk &args, ExpressionState &state, Vector &result) {
	auto constant = args.AllConstant();
	auto count = constant && args.size() ? idx_t(1) : args.size();
	ImageOperatorInput images(args.data[0], count, &state.GetContext());
	result.SetVectorType(VectorType::FLAT_VECTOR);
	idx_t bytes = 0;
	for (idx_t row = 0; row < count; row++) {
		auto &context = state.GetContext();
		ImageOperatorContract::Interrupt(context);
		ImagePixelView image;
		auto format_value = args.data[1].GetValue(row);
		if (images.IsNull(row) || format_value.IsNull() || !images.Read(row, image)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto format = ImageCodecContract::Format(format_value.GetValue<string>());
		ImageCodecContract::CheckEncoding(format, image.layout);
		string encoded;
		if (format == "PNG") {
			PNGEncoder encoder(context, ImageOperatorContract::MAX_BYTES - bytes);
			encoded = encoder.Encode(image);
		} else {
			encoded = NativeImageCodec::Encode(context, image, format, ImageOperatorContract::MAX_BYTES - bytes);
		}
		bytes += encoded.size();
		FlatVector::GetData<string_t>(result)[row] =
		    StringVector::AddStringOrBlob(result, encoded.data(), encoded.size());
		FlatVector::SetNull(result, row, false);
	}
	if (constant && count) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

void RegisterImagePixelFunctions(ExtensionLoader &loader) {
	ScalarFunction crop("native_crop", {LogicalType::ANY, LogicalType::ANY}, ImageLogicalType::Create(), CropImage,
	                    ImageOperatorContract::BindCrop);
	ScalarFunction encode("native_encode_image", {LogicalType::ANY, LogicalType::VARCHAR}, LogicalType::BLOB,
	                      EncodeImage, ImageOperatorContract::BindEncode);
	ScalarFunction resize("native_resize", {LogicalType::ANY, LogicalType::ANY, LogicalType::ANY},
	                      ImageLogicalType::Create(), ResizeImage, ImageTransformContract::BindResize);
	ScalarFunction convert("native_convert_image", {LogicalType::ANY, LogicalType::ANY}, ImageLogicalType::Create(),
	                       ConvertImage, ImageTransformContract::BindConvert);
	ScalarFunction resize_antialias("native_resize",
	                                {LogicalType::ANY, LogicalType::ANY, LogicalType::ANY, LogicalType::ANY},
	                                ImageLogicalType::Create(), ResizeImage, ImageTransformContract::BindResize);
	ScalarFunctionSet resizes("native_resize");
	for (auto &function : {&resize, &resize_antialias}) {
		function->SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
		function->SetFallible();
		resizes.AddFunction(*function);
	}
	loader.RegisterFunction(resizes);
	for (auto &function : {&crop, &encode, &convert}) {
		function->SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
		function->SetFallible();
		loader.RegisterFunction(*function);
	}
}

} // namespace duckdb
