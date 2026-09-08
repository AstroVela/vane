// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "image_operator_contract.hpp"

#include <cmath>

namespace duckdb {

template <class INTERRUPT>
void CopyTransformPixels(const ImagePixelView &source, data_ptr_t target, INTERRUPT check_interrupted) {
	for (idx_t offset = 0; offset < source.layout.Size();) {
		check_interrupted();
		auto size = MinValue(source.layout.Size() - offset, ImageOperatorContract::COPY_BYTES);
		memcpy(target + offset, source.data + offset, size);
		offset += size;
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
	auto byte = [](double value) {
		return uint8_t(MinValue(255.0, std::floor(value + 0.5)));
	};
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
			const_data_ptr_t samples[] = {source.data + (idx_t(y0) * source.layout.width + x0) * channels,
			                              source.data + (idx_t(y0) * source.layout.width + x1) * channels,
			                              source.data + (idx_t(y1) * source.layout.width + x0) * channels,
			                              source.data + (idx_t(y1) * source.layout.width + x1) * channels};
			auto output = target + pixel * channels;
			double opacity = 0;
			if (alpha) {
				for (idx_t i = 0; i < 4; i++) {
					weights[i] *= samples[i][colors];
					opacity += weights[i];
				}
				output[colors] = byte(opacity);
			}
			for (idx_t channel = 0; channel < idx_t(colors); channel++) {
				double value = 0;
				for (idx_t i = 0; i < 4; i++) {
					value += samples[i][channel] * weights[i];
				}
				output[channel] = byte(alpha ? (opacity > 0 ? value / opacity : 0) : value);
			}
		}
	}
}

//! Full-range RGB luma uses (299 R + 587 G + 114 B + 500) / 1000.
//! Alpha is copied, added as opaque, or dropped without compositing.
template <class INTERRUPT>
void ConvertImagePixels(const ImagePixelView &source, const ImageLayout &layout, data_ptr_t target,
                        INTERRUPT check_interrupted) {
	if (source.layout.channels == layout.channels) {
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
			auto input = source.data + pixel * source_channels;
			auto output = target + pixel * target_channels;
			auto red = input[0];
			auto green = source_channels < 3 ? red : input[1];
			auto blue = source_channels < 3 ? red : input[2];
			if (target_channels < 3) {
				output[0] = uint8_t((299 * red + 587 * green + 114 * blue + 500) / 1000);
			} else {
				output[0] = red;
				output[1] = green;
				output[2] = blue;
			}
			if (target_channels == 2 || target_channels == 4) {
				output[target_channels - 1] =
				    source_channels == 2 || source_channels == 4 ? input[source_channels - 1] : 255;
			}
		}
	}
}

} // namespace duckdb
