// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "media_reader.hpp"
#include "audio_decoder.hpp"
#include "audio_content_type.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include <algorithm>
#include <cmath>
#include <soxr.h>
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

//! Public identifiers match Python SoundFile, rather than libsndfile's display
//! descriptions or FFmpeg's codec names. Only shared decoder formats reach here.
static const char *SoundFileFormatName(int format) {
	switch (format) {
	case SF_FORMAT_WAV:
		return "WAV";
	case SF_FORMAT_WAVEX:
		return "WAVEX";
	case SF_FORMAT_RF64:
		return "RF64";
	case SF_FORMAT_AIFF:
		return "AIFF";
	case SF_FORMAT_FLAC:
		return "FLAC";
	case SF_FORMAT_OGG:
		return "OGG";
	case SF_FORMAT_MPEG:
		return "MP3";
	case SF_FORMAT_PCM_S8:
		return "PCM_S8";
	case SF_FORMAT_PCM_16:
		return "PCM_16";
	case SF_FORMAT_PCM_24:
		return "PCM_24";
	case SF_FORMAT_PCM_32:
		return "PCM_32";
	case SF_FORMAT_PCM_U8:
		return "PCM_U8";
	case SF_FORMAT_FLOAT:
		return "FLOAT";
	case SF_FORMAT_DOUBLE:
		return "DOUBLE";
	case SF_FORMAT_ULAW:
		return "ULAW";
	case SF_FORMAT_ALAW:
		return "ALAW";
	case SF_FORMAT_VORBIS:
		return "VORBIS";
	case SF_FORMAT_OPUS:
		return "OPUS";
	case SF_FORMAT_MPEG_LAYER_I:
		return "MPEG_LAYER_I";
	case SF_FORMAT_MPEG_LAYER_II:
		return "MPEG_LAYER_II";
	case SF_FORMAT_MPEG_LAYER_III:
		return "MPEG_LAYER_III";
	default:
		throw MediaFormatException("audio decoder reported an unsupported metadata format");
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
		auto file = FileReference::FromValue(value, "native_audio_metadata");
		MediaReader reader(state.GetContext(), file, AVMEDIA_TYPE_AUDIO, INT64_MAX, budget, MEDIA_MAX_PIXELS,
		                   MEDIA_MAX_FRAME_BYTES, budget);
		auto &stream = reader.Stream();
		auto &parameters = *stream.codecpar;
		ValidateAudio(parameters);
		AudioContentType::Validate(file, reader);
		Value frames(LogicalType::BIGINT), duration(LogicalType::DOUBLE);
		if (NativeSoundFile::Supports(reader)) {
			// Opening the decoder supplies bounded metadata without reading the
			// full waveform. Its frames/rate use the same encoder-delay and tail
			// rules as resample, including Opus rate hints and empty inputs.
			NativeSoundFile decoder(reader);
			auto info = decoder.Info();
			decoder.Close();
			if (info.frames != INT64_MAX) {
				frames = Value::BIGINT(info.frames);
				duration = Value::DOUBLE(double(info.frames) / info.samplerate);
			}
			result.SetValue(row, Value::STRUCT(result.GetType(),
			                                   {Value::BIGINT(info.samplerate), Value::BIGINT(info.channels), frames,
			                                    duration, Value(SoundFileFormatName(info.format & SF_FORMAT_TYPEMASK)),
			                                    Value(SoundFileFormatName(info.format & SF_FORMAT_SUBMASK))}));
			continue;
		}
		// Additional FFmpeg codecs retain their native format/codec identifiers.
		// A container duration is not necessarily a decoded sample count. Keep
		// frames and duration unknown together unless PCM establishes the count.
		if (parameters.codec_id >= AV_CODEC_ID_PCM_S16LE && parameters.codec_id <= AV_CODEC_ID_PCM_SGA &&
		    stream.duration != AV_NOPTS_VALUE && stream.duration >= 0 && stream.time_base.num > 0 &&
		    stream.time_base.den > 0) {
			auto count = av_rescale_q(stream.duration, stream.time_base, AVRational {1, parameters.sample_rate});
			if (count < 0) {
				throw MediaFormatException("audio decoder reported an invalid frame count");
			}
			frames = Value::BIGINT(count);
			duration = Value::DOUBLE(double(count) / parameters.sample_rate);
		}
		result.SetValue(row, Value::STRUCT(result.GetType(), {Value::BIGINT(parameters.sample_rate),
		                                                      Value::BIGINT(parameters.ch_layout.nb_channels), frames,
		                                                      duration, Value(reader.Format().iformat->name),
		                                                      Value(avcodec_get_name(parameters.codec_id))}));
	}
}

struct SampleConverter {
	SwrContext *context = nullptr;
	AVChannelLayout layout {};
	~SampleConverter() {
		swr_free(&context);
		av_channel_layout_uninit(&layout);
	}
};

struct Resampler {
	soxr_t context = nullptr;
	~Resampler() {
		soxr_delete(context);
	}
};

struct AudioProfile {
	double setup_seconds = 0;
	double decode_seconds = 0;
	double resample_seconds = 0;
	double allocation_seconds = 0;
	uint64_t buffer_growths = 0;
	MediaReadProfile reads;
};

static LogicalType AudioProfileType() {
	return LogicalType::STRUCT({{"setup_seconds", LogicalType::DOUBLE},
	                            {"decode_seconds", LogicalType::DOUBLE},
	                            {"resample_seconds", LogicalType::DOUBLE},
	                            {"allocation_seconds", LogicalType::DOUBLE},
	                            {"file_read_seconds", LogicalType::DOUBLE},
	                            {"file_read_calls", LogicalType::UBIGINT},
	                            {"file_bytes_read", LogicalType::UBIGINT},
	                            {"decoded_frames", LogicalType::UBIGINT},
	                            {"output_frames", LogicalType::UBIGINT},
	                            {"output_bytes", LogicalType::UBIGINT},
	                            {"buffer_growths", LogicalType::UBIGINT},
	                            {"buffer_capacity_bytes", LogicalType::UBIGINT},
	                            {"codec_version", LogicalType::UINTEGER},
	                            {"resampler_version", LogicalType::UINTEGER},
	                            {"decoder_library", LogicalType::VARCHAR},
	                            {"decoder_version", LogicalType::VARCHAR},
	                            {"resampler_library", LogicalType::VARCHAR},
	                            {"resampler_version_string", LogicalType::VARCHAR},
	                            {"source_sample_rate", LogicalType::UINTEGER}});
}

//! The diagnostic instantiation executes the same decoder/resampler and allocates
//! the same batch of waveforms, but returns its costs instead of those waveforms.
//! Ordinary execution does not enable diagnostic timers.
template <bool PROFILE>
static void AudioResample(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	unique_ptr<Vector> profiled_output;
	if (PROFILE) {
		profiled_output = make_uniq<Vector>(MediaAudioResultType());
	}
	auto &waveforms = PROFILE ? *profiled_output : result;
	auto &children = StructVector::GetEntries(waveforms);
	for (auto &child : children) {
		child->SetVectorType(VectorType::FLAT_VECTOR);
	}
	auto &samples = *children[0];
	auto &shape = *children[1];
	auto &dimensions = ArrayVector::GetEntry(shape);
	dimensions.SetVectorType(VectorType::FLAT_VECTOR);
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
		FileReference file;
		AudioProfile profile;
		auto reader = [&]() {
			MediaProfileTimer timer(PROFILE ? &profile.setup_seconds : nullptr);
			file = FileReference::FromValue(args.data[0].GetValue(row), "native_audio_resample");
			return MediaReader(context, file, AVMEDIA_TYPE_AUDIO, limits[0], limits[0] * 4, MEDIA_MAX_PIXELS,
			                   MinValue<uint64_t>(limits[2], 64 * MEDIA_MIB), MEDIA_METADATA_BYTES,
			                   PROFILE ? &profile.reads : nullptr);
		}();
		auto &parameters = *reader.Stream().codecpar;
		ValidateAudio(parameters);
		AudioContentType::Validate(file, reader);
		auto channels = uint64_t(parameters.ch_layout.nb_channels);
		auto source_rate = uint64_t(parameters.sample_rate);
		const bool soundfile_decoder = NativeSoundFile::Supports(reader);
		unique_ptr<NativeSoundFile> decoder;
		bool have_first_frame = false;
		if (soundfile_decoder) {
			MediaProfileTimer timer(PROFILE ? &profile.setup_seconds : nullptr);
			decoder = make_uniq<NativeSoundFile>(reader);
			source_rate = uint64_t(decoder->Info().samplerate);
			channels = uint64_t(decoder->Info().channels);
		} else {
			MediaProfileTimer timer(PROFILE ? &profile.decode_seconds : nullptr);
			have_first_frame = reader.NextFrame();
			if (have_first_frame) {
				// Container parameters can be hints: WebM may advertise the Opus
				// encoder's 8 kHz input while its decoder emits 48 kHz samples.
				// Establish the waveform contract from the first decoded frame.
				auto &frame = reader.Frame();
				if (frame.sample_rate <= 0 || frame.sample_rate > 384000 || frame.ch_layout.nb_channels <= 0 ||
				    frame.ch_layout.nb_channels > 64) {
					throw MediaFormatException("decoded audio requires 1..64 channels and 1..384000 Hz");
				}
				source_rate = uint64_t(frame.sample_rate);
				channels = uint64_t(frame.ch_layout.nb_channels);
			}
		}
		if (MaxValue<uint64_t>(sample_rate, source_rate) > MinValue<uint64_t>(sample_rate, source_rate) * 64) {
			throw OutOfRangeException("native audio resample ratio exceeds the safe 64:1 limit");
		}
		const uint64_t frame_bytes = channels * sizeof(double);
		const uint64_t chunk_frames = MinValue<uint64_t>(65536, MEDIA_MIB / frame_bytes);
		const uint64_t resample_input_frames =
		    MinValue<uint64_t>(chunk_frames, chunk_frames * source_rate / sample_rate);
		Resampler resampler;
		uint64_t decoded_frames = 0, output_frames = 0;
		auto start = ListVector::GetListSize(samples);
		auto row_limit = MinValue<uint64_t>(limits[3], limits[4] / frame_bytes);
		auto batch_limit = (MEDIA_BATCH_BYTES / sizeof(double) - start) / channels;
		auto expected_frames = [&]() {
			return (decoded_frames * sample_rate + source_rate - 1) / source_rate;
		};
		auto check_output = [&](uint64_t frames) {
			if (frames > limits[3]) {
				throw OutOfRangeException("native audio exceeds max_output_frames");
			}
			MediaProduct(frames, frame_bytes, limits[4], "audio output bytes");
			if (frames > batch_limit) {
				throw OutOfRangeException("native audio exceeds its batch byte limit");
			}
		};
		auto account_input = [&](uint64_t frames) {
			if (frames > limits[1] - decoded_frames) {
				throw OutOfRangeException("native audio exceeds max_frames");
			}
			decoded_frames += frames;
			MediaProduct(decoded_frames, frame_bytes, limits[2], "decoded audio bytes");
			check_output(expected_frames());
		};
		auto reserve = [&](uint64_t frames) {
			MediaProfileTimer timer(PROFILE ? &profile.allocation_seconds : nullptr);
			auto required = NumericCast<idx_t>(start + (output_frames + frames) * channels);
			if (PROFILE && required > ListVector::GetListCapacity(samples)) {
				profile.buffer_growths++;
			}
			ListVector::Reserve(samples, required);
			return FlatVector::GetData<double>(ListVector::GetEntry(samples)) + start + output_frames * channels;
		};
		auto advance = [&](uint64_t frames) {
			check_output(output_frames + frames);
			output_frames += frames;
			ListVector::SetListSize(samples, start + output_frames * channels);
			MediaInterrupt(context);
		};
		auto append = [&](const double *input, uint64_t frames) {
			check_output(output_frames + frames);
			if (frames) {
				auto target = reserve(frames);
				MediaProfileTimer timer(PROFILE ? &profile.resample_seconds : nullptr);
				if (input) {
					std::copy_n(input, frames * channels, target);
				} else {
					std::fill_n(target, frames * channels, 0.0);
				}
			}
			advance(frames);
		};
		// Bound each SoXR output allocation by both the row and batch budgets.
		// A one-frame stack probe distinguishes an exhausted budget from EOF.
		auto convert = [&](const double *input, uint64_t frames, size_t &consumed) {
			MediaInterrupt(context);
			auto delay = soxr_delay(resampler.context);
			if (!std::isfinite(delay) || delay < 0 || delay > 100000000) {
				throw InternalException("native audio resampler returned an invalid delay");
			}
			auto bound = uint64_t(std::ceil(delay)) + (frames * sample_rate + source_rate - 1) / source_rate + 1;
			auto capacity = MinValue<uint64_t>(MinValue<uint64_t>(chunk_frames, bound),
			                                   MinValue<uint64_t>(row_limit, batch_limit) - output_frames);
			double overflow_probe[64];
			auto target = capacity ? reserve(capacity) : overflow_probe;
			size_t written = 0;
			soxr_error_t error;
			{
				MediaProfileTimer timer(PROFILE ? &profile.resample_seconds : nullptr);
				error = soxr_process(resampler.context, input, frames, &consumed, target, capacity ? capacity : 1,
				                     &written);
			}
			if (error) {
				throw MediaFormatException(string("cannot resample audio: ") + error);
			}
			advance(written);
			return written;
		};
		auto process_input = [&](const double *input, uint64_t frames) {
			account_input(frames);
			if (sample_rate == source_rate) {
				append(input, frames);
				return;
			}
			if (!frames) {
				return;
			}
			if (!resampler.context) {
				MediaProfileTimer timer(PROFILE ? &profile.resample_seconds : nullptr);
				auto io = soxr_io_spec(SOXR_FLOAT64_I, SOXR_FLOAT64_I);
				auto quality = soxr_quality_spec(SOXR_HQ, 0);
				soxr_error_t error = nullptr;
				resampler.context = soxr_create(source_rate, sample_rate, NumericCast<unsigned>(channels), &error, &io,
				                                &quality, nullptr);
				if (!resampler.context) {
					throw OutOfMemoryException("Cannot allocate native SoXR resampler: %s", soxr_strerror(error));
				}
				if (error) {
					throw MediaFormatException(string("cannot configure audio resampler: ") + error);
				}
			}
			uint64_t offset = 0;
			while (offset < frames) {
				size_t consumed = 0;
				auto written = convert(input + offset * channels,
				                       MinValue<uint64_t>(frames - offset, resample_input_frames), consumed);
				if (!consumed && !written) {
					throw InternalException("native audio resampler made no progress");
				}
				offset += consumed;
			}
		};
		if (soundfile_decoder) {
			auto known_frames = decoder->Info().frames != INT64_MAX;
			auto total_frames = uint64_t(decoder->Info().frames);
			if (known_frames) {
				if (total_frames > limits[1]) {
					throw OutOfRangeException("native audio exceeds max_frames");
				}
				MediaProduct(total_frames, frame_bytes, limits[2], "decoded audio bytes");
			}
			auto input_limit = MinValue<uint64_t>(limits[1], limits[2] / frame_bytes);
			auto frame_limit = known_frames ? total_frames : input_limit;
			vector<double> decoded(MinValue<uint64_t>(chunk_frames, frame_limit) * channels);
			while (decoded_frames < frame_limit) {
				auto requested = MinValue<uint64_t>(chunk_frames, frame_limit - decoded_frames);
				uint64_t returned;
				{
					MediaProfileTimer timer(PROFILE ? &profile.decode_seconds : nullptr);
					returned = decoder->ReadFrames(decoded.data(), requested);
				}
				process_input(decoded.data(), returned);
				if (returned != requested) {
					if (known_frames) {
						throw MediaFormatException("audio decoder returned fewer frames than its header reports");
					}
					break;
				}
			}
			if (!known_frames && decoded_frames == input_limit) {
				double probe[64];
				if (decoder->ReadFrames(probe, 1)) {
					account_input(1);
				}
			}
			decoder->Close();
		} else {
			// FFmpeg retains the additional containers/codecs supported by native.
			// libswresample only converts sample layout/dtype at the original rate;
			// every actual sample-rate conversion goes through the SoXR path above.
			SampleConverter converter;
			int source_format = -1;
			bool have_frame = have_first_frame;
			while (have_frame) {
				auto &frame = reader.Frame();
				if (frame.sample_rate != NumericCast<int>(source_rate) ||
				    frame.ch_layout.nb_channels != NumericCast<int>(channels) || frame.nb_samples < 0) {
					throw MediaFormatException("audio stream changed sample rate or channel layout");
				}
				if (uint64_t(frame.nb_samples) > limits[1] - decoded_frames) {
					throw OutOfRangeException("native audio exceeds max_frames");
				}
				MediaProduct(decoded_frames + frame.nb_samples, frame_bytes, limits[2], "decoded audio bytes");
				if (!converter.context) {
					MediaProfileTimer timer(PROFILE ? &profile.resample_seconds : nullptr);
					source_format = frame.format;
					MediaCheck(av_channel_layout_copy(&converter.layout, &frame.ch_layout), "retain channel layout");
					MediaCheck(swr_alloc_set_opts2(&converter.context, &frame.ch_layout, AV_SAMPLE_FMT_DBL,
					                               frame.sample_rate, &frame.ch_layout, AVSampleFormat(frame.format),
					                               frame.sample_rate, 0, nullptr),
					           "configure sample conversion");
					if (!converter.context) {
						throw OutOfMemoryException("Cannot allocate native sample converter");
					}
					MediaCheck(swr_init(converter.context), "initialize sample conversion");
				} else if (source_format != frame.format ||
				           av_channel_layout_compare(&frame.ch_layout, &converter.layout)) {
					throw MediaFormatException("audio stream changed sample format");
				}
				vector<const uint8_t *> input(av_sample_fmt_is_planar(AVSampleFormat(frame.format)) ? channels : 1);
				for (idx_t channel = 0; channel < input.size(); channel++) {
					input[channel] = frame.extended_data[channel];
				}
				vector<double> decoded(uint64_t(frame.nb_samples) * channels);
				auto target = reinterpret_cast<uint8_t *>(decoded.data());
				int count;
				{
					MediaProfileTimer timer(PROFILE ? &profile.resample_seconds : nullptr);
					count = swr_convert(converter.context, &target, frame.nb_samples, input.data(), frame.nb_samples);
				}
				MediaCheck(count, "convert audio samples");
				if (count != frame.nb_samples) {
					throw InternalException("native audio sample conversion changed the frame count");
				}
				process_input(decoded.data(), count);
				{
					MediaProfileTimer timer(PROFILE ? &profile.decode_seconds : nullptr);
					have_frame = reader.NextFrame();
				}
			}
		}
		if (resampler.context) {
			size_t consumed = 0;
			while (convert(nullptr, 0, consumed)) {
			}
		}
		// Normalize the complete stream once, after decoder padding has already
		// been removed. Do not round each packet or synthesize a short waveform.
		auto expected = expected_frames();
		check_output(expected);
		if (output_frames > expected) {
			output_frames = expected;
			ListVector::SetListSize(samples, start + output_frames * channels);
		}
		while (output_frames < expected) {
			append(nullptr, MinValue<uint64_t>(chunk_frames, expected - output_frames));
		}
		auto &entry = FlatVector::GetData<list_entry_t>(samples)[row];
		entry = list_entry_t(start, ListVector::GetListSize(samples) - start);
		auto shape_data = FlatVector::GetData<int32_t>(dimensions);
		shape_data[row * 2] = NumericCast<int32_t>(output_frames);
		shape_data[row * 2 + 1] = NumericCast<int32_t>(channels);
		FlatVector::Validity(dimensions).SetValid(row * 2);
		FlatVector::Validity(dimensions).SetValid(row * 2 + 1);
		FlatVector::Validity(ListVector::GetEntry(samples)).SetAllValid(ListVector::GetListSize(samples));
		FlatVector::SetNull(waveforms, row, false);
		for (auto &child : children) {
			FlatVector::SetNull(*child, row, false);
		}
		if (PROFILE) {
			result.SetValue(
			    row,
			    Value::STRUCT(result.GetType(),
			                  {Value::DOUBLE(profile.setup_seconds), Value::DOUBLE(profile.decode_seconds),
			                   Value::DOUBLE(profile.resample_seconds), Value::DOUBLE(profile.allocation_seconds),
			                   Value::DOUBLE(profile.reads.seconds), Value::UBIGINT(profile.reads.calls),
			                   Value::UBIGINT(reader.BytesRead()), Value::UBIGINT(decoded_frames),
			                   Value::UBIGINT(output_frames), Value::UBIGINT(output_frames * channels * sizeof(double)),
			                   Value::UBIGINT(profile.buffer_growths),
			                   Value::UBIGINT(ListVector::GetListCapacity(samples) * sizeof(double)),
			                   Value::UINTEGER(avcodec_version()), Value::UINTEGER(SOXR_THIS_VERSION),
			                   Value(soundfile_decoder ? "libsndfile" : "ffmpeg"),
			                   Value(soundfile_decoder ? sf_version_string() : av_version_info()), Value("soxr_hq"),
			                   Value(soxr_version()), Value::UINTEGER(NumericCast<uint32_t>(source_rate))}));
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
	resample.AddFunction(MediaScalar("audio_resample", {LogicalType::ANY, LogicalType::BIGINT}, MediaAudioResultType(),
	                                 AudioResample<false>));
	resample.AddFunction(MediaScalar("audio_resample",
	                                 {LogicalType::ANY, LogicalType::BIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT,
	                                  LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT},
	                                 MediaAudioResultType(), AudioResample<false>));
	loader.RegisterFunction(resample);
	ScalarFunctionSet profile("native_audio_resample_profile");
	profile.AddFunction(MediaScalar("audio_resample_profile", {LogicalType::ANY, LogicalType::BIGINT},
	                                AudioProfileType(), AudioResample<true>));
	profile.AddFunction(MediaScalar("audio_resample_profile",
	                                {LogicalType::ANY, LogicalType::BIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT,
	                                 LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT},
	                                AudioProfileType(), AudioResample<true>));
	loader.RegisterFunction(profile);
}
} // namespace duckdb
