// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "image_operator_contract.hpp"

#include <cmath>

namespace duckdb {

template <class INTERRUPT>
void CopyTransformPixels(const ImagePixelView &source, data_ptr_t target, INTERRUPT check_interrupted) {
	for (idx_t offset = 0; offset < source.layout.Bytes();) {
		check_interrupted();
		auto size = MinValue(source.layout.Bytes() - offset, ImageOperatorContract::COPY_BYTES);
		memcpy(target + offset, source.data + offset, size);
		offset += size;
	}
}

inline double ImageSample(const_data_ptr_t data, uint8_t mode, idx_t index) {
	if (mode <= 4) {
		return data[index];
	}
	if (mode <= 8) {
		uint16_t value;
		memcpy(&value, data + index * sizeof(value), sizeof(value));
		return value;
	}
	float value;
	memcpy(&value, data + index * sizeof(value), sizeof(value));
	return value;
}

inline double ImageRange(uint8_t mode) {
	return mode <= 4 ? 255.0 : mode <= 8 ? 65535.0 : 1.0;
}

inline void ImageStore(data_ptr_t data, uint8_t mode, idx_t index, double value) {
	if (mode <= 8) {
		value = std::floor(MaxValue(0.0, MinValue(ImageRange(mode), value)) + 0.5);
		if (mode <= 4) {
			data[index] = uint8_t(value);
		} else {
			auto pixel = uint16_t(value);
			memcpy(data + index * sizeof(pixel), &pixel, sizeof(pixel));
		}
	} else {
		auto pixel = float(value);
		if (!std::isfinite(pixel)) {
			throw OutOfRangeException("Image transform produced a non-finite Float32 pixel");
		}
		memcpy(data + index * sizeof(pixel), &pixel, sizeof(pixel));
	}
}

//! Half-pixel bilinear sampling with edge clamping. Alpha-bearing inputs are
//! filtered in premultiplied form and returned with straight alpha. This does
//! not apply an antialiasing prefilter or a transfer-function conversion.
template <class INTERRUPT>
void ResizeImagePixels(const ImagePixelView &source, const ImageLayout &layout, data_ptr_t target,
                       INTERRUPT check_interrupted) {
	if (source.layout.width == layout.width && source.layout.height == layout.height) {
		CopyTransformPixels(source, target, check_interrupted);
		return;
	}
	auto channels = layout.channels;
	auto alpha = channels == 2 || channels == 4;
	auto colors = alpha ? channels - 1 : channels;
	auto pixels = idx_t(layout.width) * layout.height;
	const idx_t block = 16384;
	for (idx_t begin = 0; begin < pixels; begin += block) {
		check_interrupted();
		auto end = MinValue(pixels, begin + block);
		for (idx_t pixel = begin; pixel < end; pixel++) {
			auto x = MinValue(
			    double(source.layout.width - 1),
			    MaxValue(0.0, (double(pixel % layout.width) + 0.5) * source.layout.width / layout.width - 0.5));
			auto y = MinValue(
			    double(source.layout.height - 1),
			    MaxValue(0.0, (double(pixel / layout.width) + 0.5) * source.layout.height / layout.height - 0.5));
			auto x0 = uint32_t(x);
			auto y0 = uint32_t(y);
			auto x1 = MinValue(x0 + 1, source.layout.width - 1);
			auto y1 = MinValue(y0 + 1, source.layout.height - 1);
			auto fx = x - x0;
			auto fy = y - y0;
			double weights[] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
			idx_t samples[] = {
			    (idx_t(y0) * source.layout.width + x0) * channels, (idx_t(y0) * source.layout.width + x1) * channels,
			    (idx_t(y1) * source.layout.width + x0) * channels, (idx_t(y1) * source.layout.width + x1) * channels};
			auto output = pixel * channels;
			double opacity = 0;
			if (alpha) {
				for (idx_t i = 0; i < 4; i++) {
					weights[i] *= ImageSample(source.data, source.layout.mode, samples[i] + colors);
					opacity += weights[i];
				}
				ImageStore(target, layout.mode, output + colors, opacity);
			}
			for (idx_t channel = 0; channel < idx_t(colors); channel++) {
				double value = 0;
				for (idx_t i = 0; i < 4; i++) {
					value += ImageSample(source.data, source.layout.mode, samples[i] + channel) * weights[i];
				}
				ImageStore(target, layout.mode, output + channel, alpha ? (opacity > 0 ? value / opacity : 0) : value);
			}
		}
	}
}

//! Full-range RGB luma uses (299 R + 587 G + 114 B + 500) / 1000.
//! Alpha is copied, added as opaque, or dropped without compositing.
template <class INTERRUPT>
void ConvertImagePixels(const ImagePixelView &source, const ImageLayout &layout, data_ptr_t target,
                        INTERRUPT check_interrupted) {
	if (source.layout.mode == layout.mode) {
		CopyTransformPixels(source, target, check_interrupted);
		return;
	}
	auto source_channels = source.layout.channels;
	auto target_channels = layout.channels;
	auto pixels = idx_t(layout.width) * layout.height;
	const idx_t block = 16384;
	for (idx_t begin = 0; begin < pixels; begin += block) {
		check_interrupted();
		auto end = MinValue(pixels, begin + block);
		for (idx_t pixel = begin; pixel < end; pixel++) {
			auto input = pixel * source_channels;
			auto output = pixel * target_channels;
			auto red = ImageSample(source.data, source.layout.mode, input);
			auto green = source_channels < 3 ? red : ImageSample(source.data, source.layout.mode, input + 1);
			auto blue = source_channels < 3 ? red : ImageSample(source.data, source.layout.mode, input + 2);
			auto scale = ImageRange(layout.mode) / ImageRange(source.layout.mode);
			if (target_channels < 3) {
				ImageStore(target, layout.mode, output, ((299 * red + 587 * green + 114 * blue) / 1000) * scale);
			} else {
				ImageStore(target, layout.mode, output, red * scale);
				ImageStore(target, layout.mode, output + 1, green * scale);
				ImageStore(target, layout.mode, output + 2, blue * scale);
			}
			if (target_channels == 2 || target_channels == 4) {
				auto opacity = source_channels == 2 || source_channels == 4
				                   ? ImageSample(source.data, source.layout.mode, input + source_channels - 1) * scale
				                   : ImageRange(layout.mode);
				ImageStore(target, layout.mode, output + target_channels - 1, opacity);
			}
		}
	}
}

} // namespace duckdb
