// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "media_reader.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
extern "C" {
#include <libswresample/swresample.h>
}

namespace duckdb {
namespace {

static void ValidateAudio(const AVCodecParameters &parameters) {
	if (parameters.sample_rate <= 0 || parameters.sample_rate > 384000 || parameters.ch_layout.nb_channels <= 0 ||
	    parameters.ch_layout.nb_channels > 64) {
		throw MediaFormatException("native audio requires 1..64 channels and a sample rate in 1..384000 Hz");
	}
}

static void AudioMetadata(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		auto value = args.data[0].GetValue(row);
		if (value.IsNull() || (args.ColumnCount() == 2 && args.data[1].GetValue(row).IsNull())) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}
		auto budget = args.ColumnCount() == 2 ? MediaPositive(args.data[1].GetValue(row), "max_bytes", 64 * MEDIA_MIB)
		                                      : MEDIA_METADATA_BYTES;
		MediaReader reader(state.GetContext(), FileReference::FromValue(value, "native_audio_metadata"),
		                   AVMEDIA_TYPE_AUDIO, INT64_MAX, budget);
		auto &stream = reader.Stream();
		auto &parameters = *stream.codecpar;
		ValidateAudio(parameters);
		Value frames(LogicalType::BIGINT), duration(LogicalType::DOUBLE);
		if (stream.duration != AV_NOPTS_VALUE && stream.duration >= 0 && stream.time_base.num > 0 &&
		    stream.time_base.den > 0) {
			duration = Value::DOUBLE(stream.duration * av_q2d(stream.time_base));
			// Encoded container duration need not establish an exact sample count.
			if (parameters.codec_id >= AV_CODEC_ID_PCM_S16LE && parameters.codec_id <= AV_CODEC_ID_PCM_SGA) {
				frames = Value::BIGINT(
				    av_rescale_q(stream.duration, stream.time_base, AVRational {1, parameters.sample_rate}));
			}
		}
		result.SetValue(row, Value::STRUCT(result.GetType(), {Value::BIGINT(parameters.sample_rate),
		                                                      Value::BIGINT(parameters.ch_layout.nb_channels), frames,
		                                                      duration, Value(reader.Format().iformat->name),
		                                                      Value(avcodec_get_name(parameters.codec_id))}));
	}
}

struct Resampler {
	SwrContext *context = nullptr;
	AVChannelLayout layout {};
	~Resampler() {
		swr_free(&context);
		av_channel_layout_uninit(&layout);
	}
};

static void AudioResample(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	for (auto &child : children) {
		child->SetVectorType(VectorType::FLAT_VECTOR);
	}
	auto &samples = *children[0];
	ListVector::SetListSize(samples, 0);
	for (idx_t row = 0; row < args.size(); row++) {
		bool null = false;
		for (idx_t col = 0; col < args.ColumnCount(); col++) {
			null = null || args.data[col].GetValue(row).IsNull();
		}
		if (null) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto sample_rate = MediaPositive(args.data[1].GetValue(row), "sample_rate", 384000);
		uint64_t limits[] = {512 * MEDIA_MIB, 100000000, 512 * MEDIA_MIB, 100000000, 512 * MEDIA_MIB};
		if (args.ColumnCount() == 7) {
			const char *names[] = {"max_input_bytes", "max_frames", "max_decoded_bytes", "max_output_frames",
			                       "max_output_bytes"};
			uint64_t maxima[] = {4 * 1024 * MEDIA_MIB, 100000000, 512 * MEDIA_MIB, 100000000, 512 * MEDIA_MIB};
			for (idx_t index = 0; index < 5; index++) {
				limits[index] = MediaPositive(args.data[index + 2].GetValue(row), names[index], maxima[index]);
			}
		}
		auto &context = state.GetContext();
		MediaReader reader(context, FileReference::FromValue(args.data[0].GetValue(row), "native_audio_resample"),
		                   AVMEDIA_TYPE_AUDIO, limits[0], limits[0] * 4, MEDIA_MAX_PIXELS,
		                   MinValue<uint64_t>(limits[2], 64 * MEDIA_MIB));
		auto &parameters = *reader.Stream().codecpar;
		ValidateAudio(parameters);
		auto channels = uint64_t(parameters.ch_layout.nb_channels);
		Resampler resampler;
		int source_format = -1;
		uint64_t decoded_frames = 0, output_frames = 0;
		auto start = ListVector::GetListSize(samples);
		auto convert = [&](const uint8_t **input, int count) {
			MediaInterrupt(context);
			auto upper_bound = swr_get_out_samples(resampler.context, count);
			MediaCheck(upper_bound, "estimate resampled output");
			if (!upper_bound) {
				return 0;
			}
			auto offset = ListVector::GetListSize(samples);
			auto row_limit = MinValue<uint64_t>(limits[3], limits[4] / (channels * sizeof(double)));
			auto output_capacity = MinValue<uint64_t>(upper_bound, row_limit - output_frames + 1);
			auto batch_capacity = (MEDIA_BATCH_BYTES / sizeof(double) - offset) / channels;
			double overflow_probe[64];
			uint8_t *target;
			if (!batch_capacity) {
				output_capacity = 1;
				target = reinterpret_cast<uint8_t *>(overflow_probe);
			} else {
				output_capacity = MinValue<uint64_t>(output_capacity, batch_capacity);
				ListVector::Reserve(samples, NumericCast<idx_t>(offset + output_capacity * channels));
				target =
				    reinterpret_cast<uint8_t *>(FlatVector::GetData<double>(ListVector::GetEntry(samples)) + offset);
			}
			auto written = swr_convert(resampler.context, &target, NumericCast<int>(output_capacity), input, count);
			MediaCheck(written, "resample audio");
			if (written && !batch_capacity) {
				throw OutOfRangeException("native audio exceeds its batch byte limit");
			}
			MediaInterrupt(context);
			if (uint64_t(written) > limits[3] - output_frames) {
				throw OutOfRangeException("native audio exceeds max_output_frames");
			}
			output_frames += uint64_t(written);
			MediaProduct(output_frames, channels * sizeof(double), limits[4], "audio output bytes");
			ListVector::SetListSize(samples, offset + uint64_t(written) * channels);
			return written;
		};
		while (reader.NextFrame()) {
			auto &frame = reader.Frame();
			if (frame.sample_rate != parameters.sample_rate ||
			    frame.ch_layout.nb_channels != parameters.ch_layout.nb_channels || frame.nb_samples < 0) {
				throw MediaFormatException("audio stream changed sample rate or channel layout");
			}
			if (uint64_t(frame.nb_samples) > limits[1] - decoded_frames) {
				throw OutOfRangeException("native audio exceeds max_frames");
			}
			decoded_frames += uint64_t(frame.nb_samples);
			MediaProduct(decoded_frames, channels * sizeof(double), limits[2], "decoded audio bytes");
			if (!resampler.context) {
				source_format = frame.format;
				MediaCheck(av_channel_layout_copy(&resampler.layout, &frame.ch_layout), "retain channel layout");
				MediaCheck(swr_alloc_set_opts2(&resampler.context, &frame.ch_layout, AV_SAMPLE_FMT_DBL,
				                               NumericCast<int>(sample_rate), &frame.ch_layout,
				                               AVSampleFormat(frame.format), frame.sample_rate, 0, nullptr),
				           "configure resampler");
				if (!resampler.context) {
					throw OutOfMemoryException("Cannot allocate native resampler");
				}
				MediaCheck(swr_init(resampler.context), "initialize resampler");
			} else if (source_format != frame.format ||
			           av_channel_layout_compare(&frame.ch_layout, &resampler.layout)) {
				throw MediaFormatException("audio stream changed sample format");
			}
			vector<const uint8_t *> input(av_sample_fmt_is_planar(AVSampleFormat(frame.format)) ? channels : 1);
			for (idx_t channel = 0; channel < input.size(); channel++) {
				input[channel] = frame.extended_data[channel];
			}
			convert(input.data(), frame.nb_samples);
		}
		if (resampler.context) {
			while (convert(nullptr, 0)) {
			}
		}
		auto &entry = FlatVector::GetData<list_entry_t>(samples)[row];
		entry = list_entry_t(start, ListVector::GetListSize(samples) - start);
		FlatVector::GetData<int64_t>(*children[1])[row] = NumericCast<int64_t>(sample_rate);
		FlatVector::GetData<int64_t>(*children[2])[row] = NumericCast<int64_t>(output_frames);
		FlatVector::GetData<int64_t>(*children[3])[row] = NumericCast<int64_t>(channels);
		FlatVector::SetNull(result, row, false);
		for (auto &child : children) {
			FlatVector::SetNull(*child, row, false);
		}
	}
}
} // namespace

void RegisterMediaAudio(ExtensionLoader &loader) {
	ScalarFunctionSet metadata("native_audio_metadata");
	metadata.AddFunction(MediaScalar("audio_metadata", {LogicalType::ANY}, MediaAudioMetadataType(), AudioMetadata));
	metadata.AddFunction(MediaScalar("audio_metadata", {LogicalType::ANY, LogicalType::UBIGINT},
	                                 MediaAudioMetadataType(), AudioMetadata));
	loader.RegisterFunction(metadata);
	ScalarFunctionSet resample("native_audio_resample");
	resample.AddFunction(
	    MediaScalar("audio_resample", {LogicalType::ANY, LogicalType::BIGINT}, MediaAudioResultType(), AudioResample));
	resample.AddFunction(MediaScalar("audio_resample",
	                                 {LogicalType::ANY, LogicalType::BIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT,
	                                  LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT},
	                                 MediaAudioResultType(), AudioResample));
	loader.RegisterFunction(resample);
}
} // namespace duckdb
