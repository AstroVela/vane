// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "media_reader.hpp"

#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <limits>

namespace duckdb {

void MediaInterrupt(ClientContext &context) {
	if (context.IsInterrupted()) {
		throw InterruptException();
	}
}

uint64_t MediaProduct(uint64_t left, uint64_t right, uint64_t limit, const char *description) {
	if (right && left > limit / right) {
		throw OutOfRangeException("native media %s exceeds its limit of %llu", description, limit);
	}
	return left * right;
}

uint64_t MediaPositive(const Value &value, const char *name, uint64_t maximum) {
	if (value.IsNull()) {
		throw InvalidInputException("native media %s cannot be NULL", name);
	}
	auto result = value.GetValue<uint64_t>();
	if (!result || result > maximum) {
		throw InvalidInputException("native media %s must be between 1 and %llu", name, maximum);
	}
	return result;
}

void MediaCheck(int code, const char *operation) {
	if (code >= 0) {
		return;
	}
	if (code == AVERROR(ENOMEM)) {
		throw OutOfMemoryException("native media %s ran out of memory", operation);
	}
	char message[AV_ERROR_MAX_STRING_SIZE];
	av_strerror(code, message, sizeof(message));
	throw MediaFormatException(string(operation) + ": " + message);
}

// Match PyAV's narrow content-error allowlist. Argument, internal and OS
// failures must remain observable even under on_error=null/skip.
void VideoCheck(int code, const char *operation) {
	if (code >= 0) {
		return;
	}
	if (code == AVERROR(ENOMEM) || code == AVERROR_INVALIDDATA || code == AVERROR_DECODER_NOT_FOUND ||
	    code == AVERROR_DEMUXER_NOT_FOUND || code == AVERROR_EOF) {
		MediaCheck(code, operation);
	}
	char message[AV_ERROR_MAX_STRING_SIZE];
	av_strerror(code, message, sizeof(message));
	if (code == AVERROR_EXIT) {
		throw OutOfRangeException("video %s exceeded its I/O timeout", operation);
	}
	if (code > -4096 && code != AVERROR(EINVAL) && code != AVERROR(ERANGE) && code != AVERROR(ENOSYS)) {
		throw IOException("video %s: %s", operation, message);
	}
	throw InternalException("video %s: %s", operation, message);
}

void MediaReader::CheckCode(int code, const char *operation) {
	if (video_policy) {
		VideoCheck(code, operation);
	} else {
		MediaCheck(code, operation);
	}
}

static string CanonicalMIME(string value) {
	value = value.substr(0, value.find(';'));
	StringUtil::Trim(value);
	value = StringUtil::Lower(value);
	if (value == "image/jpg" || value == "image/pjpeg") {
		return "image/jpeg";
	}
	if (value == "image/x-png") {
		return "image/png";
	}
	if (value == "audio/x-wav" || value == "audio/wave" || value == "audio/vnd.wave") {
		return "audio/wav";
	}
	if (value == "audio/x-flac") {
		return "audio/flac";
	}
	if (value == "audio/x-aiff" || value == "audio/aif") {
		return "audio/aiff";
	}
	if (value == "audio/mp3" || value == "audio/x-mp3") {
		return "audio/mpeg";
	}
	if (value == "video/avi") {
		return "video/x-msvideo";
	}
	if (value == "video/x-matroska" || value == "video/mkv") {
		return "video/webm";
	}
	if (value == "audio/x-matroska") {
		return "audio/webm";
	}
	if (value == "video/quicktime" || value == "video/x-m4v") {
		return "video/mp4";
	}
	if (value == "audio/x-m4a") {
		return "audio/mp4";
	}
	return value;
}

void MediaValidateMIME(const FileReference &file, const string &detected) {
	if (!file.has_content_type) {
		return;
	}
	auto declared = CanonicalMIME(file.content_type);
	if (declared == "application/octet-stream" || declared == "binary/octet-stream") {
		return;
	}
	if (declared == "application/ogg" && (detected == "audio/ogg" || detected == "video/ogg")) {
		return;
	}
	auto separator = detected.find('/');
	if (separator != string::npos && declared == detected.substr(0, separator) + "/*") {
		return;
	}
	if (declared != detected) {
		throw MediaFormatException("content_type '" + file.content_type + "' does not match '" + detected + "'");
	}
}

static void ValidateVideoMIME(const FileReference &file, const AVInputFormat &format) {
	vector<string> allowed;
	for (auto &name : StringUtil::Split(format.name, ',')) {
		if (name == "mov" || name == "mp4") {
			allowed.emplace_back("video/mp4");
		} else if (name == "matroska" || name == "webm") {
			allowed.emplace_back("video/webm");
		} else if (name == "asf") {
			allowed.emplace_back("video/x-ms-asf");
			allowed.emplace_back("video/x-ms-wmv");
		} else if (name == "avi") {
			allowed.emplace_back("video/x-msvideo");
		} else if (name == "flv") {
			allowed.emplace_back("video/x-flv");
		} else if (name == "3g2") {
			allowed.emplace_back("video/3gpp2");
		} else if (name == "3gp") {
			allowed.emplace_back("video/3gpp");
		} else if (name == "mj2") {
			allowed.emplace_back("video/mj2");
		} else if (name == "mpeg") {
			allowed.emplace_back("video/mpeg");
		} else if (name == "mpegts") {
			allowed.emplace_back("video/mp2t");
		} else if (name == "ogg") {
			allowed.emplace_back("video/ogg");
		}
	}
	if (allowed.empty()) {
		throw MediaFormatException("unsupported video container format");
	}
	if (!file.has_content_type) {
		return;
	}
	auto declared = CanonicalMIME(file.content_type);
	if (declared == "application/ogg") {
		declared = "video/ogg";
	}
	if (declared == "application/octet-stream" || declared == "binary/octet-stream" || declared == "video/*") {
		return;
	}
	if (std::find(allowed.begin(), allowed.end(), declared) == allowed.end()) {
		throw MediaFormatException("VIDEOFILE content_type contradicts the video content");
	}
}

unique_ptr<FunctionData> BindMediaFile(ClientContext &, ScalarFunction &function,
                                       vector<unique_ptr<Expression>> &arguments) {
	FileMediaType media_type;
	if (function.name.find("image") != string::npos) {
		media_type = FileMediaType::IMAGE;
	} else if (function.name.find("audio") != string::npos) {
		media_type = FileMediaType::AUDIO;
	} else {
		media_type = FileMediaType::VIDEO;
	}
	auto type = arguments[0]->return_type;
	if (type.id() == LogicalTypeId::SQLNULL || type.id() == LogicalTypeId::UNKNOWN) {
		type = FileLogicalType::Create(media_type);
	}
	if (!FileLogicalType::IsFile(type) || FileLogicalType::GetMediaType(type) != media_type) {
		throw BinderException("%s requires %s", function.name, FileLogicalType::GetTypeName(media_type));
	}
	function.arguments[0] = type;
	return nullptr;
}

ScalarFunction MediaScalar(const string &name, vector<LogicalType> arguments, LogicalType result,
                           scalar_function_t implementation) {
	ScalarFunction function(
	    "native_" + name, std::move(arguments), std::move(result),
	    [implementation](DataChunk &args, ExpressionState &state, Vector &output) {
		    try {
			    implementation(args, state, output);
		    } catch (...) {
			    MediaInterrupt(state.GetContext());
			    throw;
		    }
	    },
	    BindMediaFile);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetStability(FunctionStability::VOLATILE);
	function.SetFallible();
	return function;
}

LogicalType MediaImageMetadataType() {
	return LogicalType::STRUCT({{"width", LogicalType::UINTEGER},
	                            {"height", LogicalType::UINTEGER},
	                            {"format", LogicalType::VARCHAR},
	                            {"mode", LogicalType::VARCHAR}});
}

LogicalType MediaAudioMetadataType() {
	return LogicalType::STRUCT({{"sample_rate", LogicalType::BIGINT},
	                            {"channels", LogicalType::BIGINT},
	                            {"frames", LogicalType::BIGINT},
	                            {"duration", LogicalType::DOUBLE},
	                            {"format", LogicalType::VARCHAR},
	                            {"subtype", LogicalType::VARCHAR}});
}

LogicalType MediaAudioResultType() {
	return TensorType::Create(LogicalType::DOUBLE, {TensorType::VARIABLE_DIMENSION, TensorType::VARIABLE_DIMENSION});
}

LogicalType MediaVideoMetadataType() {
	return LogicalType::STRUCT({{"width", LogicalType::UINTEGER},
	                            {"height", LogicalType::UINTEGER},
	                            {"fps", LogicalType::DOUBLE},
	                            {"duration", LogicalType::DOUBLE},
	                            {"container_duration", LogicalType::DOUBLE},
	                            {"frame_count", LogicalType::BIGINT},
	                            {"time_base", LogicalType::STRUCT({{"numerator", LogicalType::BIGINT},
	                                                               {"denominator", LogicalType::BIGINT}})}});
}

LogicalType MediaVideoFrameType() {
	return LogicalType::STRUCT({{"frame_index", LogicalType::BIGINT},
	                            {"frame_time", LogicalType::DOUBLE},
	                            {"frame_time_base_numerator", LogicalType::BIGINT},
	                            {"frame_time_base_denominator", LogicalType::BIGINT},
	                            {"frame_pts", LogicalType::BIGINT},
	                            {"frame_dts", LogicalType::BIGINT},
	                            {"frame_duration", LogicalType::BIGINT},
	                            {"is_key_frame", LogicalType::BOOLEAN},
	                            {"data", ImageLogicalType::Create()}});
}

static string StreamMIME(const AVInputFormat &format, AVMediaType kind) {
	string name(format.name);
	if (name.find("mov") != string::npos) {
		return kind == AVMEDIA_TYPE_AUDIO ? "audio/mp4" : "video/mp4";
	}
	if (name.find("matroska") != string::npos) {
		return kind == AVMEDIA_TYPE_AUDIO ? "audio/webm" : "video/webm";
	}
	if (name == "ogg") {
		return kind == AVMEDIA_TYPE_AUDIO ? "audio/ogg" : "video/ogg";
	}
	if (name == "wav") {
		return "audio/wav";
	}
	if (name == "aiff") {
		return "audio/aiff";
	}
	if (name == "flac") {
		return "audio/flac";
	}
	if (name == "mp3") {
		return "audio/mpeg";
	}
	if (name == "aac") {
		return "audio/aac";
	}
	if (name == "avi") {
		return "video/x-msvideo";
	}
	if (name == "mpegts") {
		return "video/mp2t";
	}
	if (name == "mpeg") {
		return "video/mpeg";
	}
	if (name == "png_pipe") {
		return "image/png";
	}
	if (name == "jpeg_pipe") {
		return "image/jpeg";
	}
	throw MediaFormatException("unsupported container '" + name + "'");
}

MediaReader::MediaReader(ClientContext &context_p, const FileReference &reference, AVMediaType kind,
                         uint64_t input_limit, uint64_t read_limit_p, uint64_t max_pixels_p, uint64_t frame_bytes_p,
                         uint64_t probe_limit, MediaReadProfile *profile_p)
    : MediaReader(context_p, reference, ResolvedFile::Open(context_p, reference), kind, input_limit, read_limit_p,
                  max_pixels_p, frame_bytes_p, probe_limit, nullptr, profile_p) {
}

MediaReader::MediaReader(ClientContext &context_p, const FileReference &reference, unique_ptr<ResolvedFile> resolved,
                         AVMediaType kind, uint64_t input_limit, uint64_t read_limit_p, uint64_t max_pixels_p,
                         uint64_t frame_bytes_p, uint64_t probe_limit, unique_ptr<MediaReadVerifier> verifier_p,
                         MediaReadProfile *profile_p)
    : context(context_p), video_policy(reference.media_type == FileMediaType::VIDEO), file(std::move(resolved)),
      verifier(std::move(verifier_p)), profile(profile_p), read_limit(read_limit_p),
      max_pixels(MinValue<uint64_t>(max_pixels_p, frame_bytes_p / 8)), frame_bytes(frame_bytes_p),
      probe_deadline(std::chrono::steady_clock::now() +
                     std::chrono::seconds(reference.media_type == FileMediaType::VIDEO ? 5 : 30)) {
	if (!file->LogicalSize()) {
		throw MediaFormatException("empty FILE view");
	}
	if (file->LogicalSize() > input_limit || file->LogicalSize() > uint64_t(INT64_MAX)) {
		throw OutOfRangeException("native media input exceeds max_input_bytes");
	}
	try {
		format = avformat_alloc_context();
		if (!format) {
			throw OutOfMemoryException("Cannot allocate native media format context");
		}
		auto buffer = static_cast<uint8_t *>(av_malloc(64 * 1024));
		if (!buffer) {
			throw OutOfMemoryException("Cannot allocate native media read buffer");
		}
		io = avio_alloc_context(buffer, 64 * 1024, 0, this, Read, nullptr, SeekIO);
		if (!io) {
			av_free(buffer);
			throw OutOfMemoryException("Cannot allocate native media I/O context");
		}
		format->pb = io;
		format->opaque = this;
		format->flags |= AVFMT_FLAG_CUSTOM_IO;
		if (video_policy) {
			// PyAV enables this at container construction. AVI in particular
			// needs demuxer lookahead to recover missing presentation times.
			format->flags |= AVFMT_FLAG_GENPTS;
		}
		format->io_open = DenyNestedIO;
		format->interrupt_callback = {Interrupt, this};
		auto probe_bytes = MinValue<uint64_t>(read_limit, probe_limit);
		if (video_policy) {
			probe_bytes = MinValue<uint64_t>(file->LogicalSize(), probe_bytes);
		}
		format->probesize = NumericCast<int64_t>(video_policy ? MaxValue<uint64_t>(32, probe_bytes) : probe_bytes);
		// FFmpeg requires at least 2048 for format detection. Read still enforces
		// smaller caller budgets and preserves their resource-limit exception.
		format->format_probesize = NumericCast<int>(MaxValue<uint64_t>(probe_bytes, 2048));
		format->max_analyze_duration = video_policy ? 5 * AV_TIME_BASE : AV_TIME_BASE;
		format->max_streams = video_policy ? 64 : 16;
		if (video_policy) {
			format->fps_probe_size = 32;
			format->max_index_size = 256 * 1024;
			format->skip_estimate_duration_from_pts = 1;
		}
		format->max_probe_packets = 256;
		AVDictionary *options = nullptr;
		struct DictionaryGuard {
			AVDictionary **values;
			unsigned count;
			~DictionaryGuard() {
				for (unsigned i = 0; i < count; i++) {
					av_dict_free(&values[i]);
				}
			}
		};
		DictionaryGuard options_guard {&options, 1};
		// Video validates the detected container after bounded probing, just
		// as PyAV does; custom AVIO and DenyNestedIO govern all source access.
		if (!video_policy) {
			CheckCode(av_dict_set(&options, "format_whitelist",
			                      "wav,aiff,flac,mp3,aac,ogg,mov,matroska,webm,avi,mpegts,mpeg,png_pipe,jpeg_pipe", 0),
			          "set container allowlist");
		}
		CheckCode(av_dict_set(&options, "protocol_whitelist", "", 0), "set protocol policy");
		if (video_policy) {
			probe_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
		}
		auto code = avformat_open_input(&format, nullptr, nullptr, &options);
		av_dict_free(&options);
		CheckIO();
		CheckCode(code, "open container");
		AVDictionary *stream_options[64] = {};
		DictionaryGuard stream_options_guard {stream_options, 64};
		auto stream_count = format->nb_streams;
		if (stream_count > unsigned(format->max_streams)) {
			throw OutOfRangeException("native media container exceeds the stream limit");
		}
		for (unsigned index = 0; index < stream_count; index++) {
			CheckCode(av_dict_set_int(&stream_options[index], "max_pixels",
			                          NumericCast<int64_t>(video_policy ? 64 * MEDIA_MIB : max_pixels), 0),
			          "set probe pixel limit");
			CheckCode(av_dict_set_int(&stream_options[index], "threads", 1, 0), "set probe thread limit");
			CheckCode(av_dict_set_int(&stream_options[index], "max_samples",
			                          int64_t(video_policy ? MEDIA_MIB : frame_bytes / sizeof(double)), 0),
			          "set probe sample limit");
			if (video_policy) {
				CheckCode(av_dict_set(&stream_options[index], "skip_frame", "all", 0), "set video probe policy");
			}
		}
		bool have_parameters = reference.media_type == FileMediaType::IMAGE;
		for (unsigned index = 0; index < stream_count; index++) {
			auto &parameters = *format->streams[index]->codecpar;
			if (parameters.codec_type != kind || parameters.codec_id == AV_CODEC_ID_NONE ||
			    (kind == AVMEDIA_TYPE_VIDEO && (format->streams[index]->disposition & AV_DISPOSITION_ATTACHED_PIC))) {
				continue;
			}
			have_parameters =
			    have_parameters ||
			    (kind == AVMEDIA_TYPE_AUDIO ? parameters.sample_rate > 0 && parameters.ch_layout.nb_channels > 0
			                                : parameters.width > 0 && parameters.height > 0);
		}
		// Video matches PyAV: probing establishes packet DTS, frame rate and
		// stream origin even when the header already contains dimensions.
		// skip_frame=all prevents the probe from decoding output pixels.
		if (video_policy && !stream_count) {
			throw MediaFormatException("video container cannot be inspected safely within metadata resource limits");
		}
		if (video_policy || !have_parameters) {
			if (video_policy) {
				probe_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
			}
			code = avformat_find_stream_info(format, stream_options);
			CheckIO();
			CheckCode(code, "inspect container");
		}
		for (unsigned index = 0; index < format->nb_streams; index++) {
			auto stream = format->streams[index];
			if (stream->codecpar->codec_type == kind && !(reference.media_type == FileMediaType::VIDEO &&
			                                              (stream->disposition & AV_DISPOSITION_ATTACHED_PIC))) {
				stream_index = NumericCast<int>(index);
				break;
			}
		}
		if (stream_index < 0) {
			throw MediaFormatException("container does not contain the requested media stream");
		}
		if (video_policy) {
			ValidateVideoMIME(reference, *format->iformat);
			auto &stream = Stream();
			auto &parameters = *stream.codecpar;
			if (!avcodec_find_decoder(parameters.codec_id) || parameters.width <= 0 || parameters.height <= 0 ||
			    stream.time_base.num <= 0 || stream.time_base.den <= 0 || stream.nb_frames < 0 ||
			    (stream.duration != AV_NOPTS_VALUE && stream.duration < 0) ||
			    (format->duration != AV_NOPTS_VALUE && format->duration < 0)) {
				throw MediaFormatException("invalid video stream metadata");
			}
			auto rate = stream.avg_frame_rate;
			if (!rate.num || !rate.den) {
				rate = av_guess_frame_rate(format, &stream, nullptr);
			}
			if ((rate.num < 0) != (rate.den < 0)) {
				throw MediaFormatException("video parser reported an out-of-range frame rate");
			}
			if (uint64_t(parameters.width) * parameters.height > max_pixels) {
				throw OutOfRangeException("video dimensions exceed max_pixels");
			}
			av_reduce(&stream.time_base.num, &stream.time_base.den, stream.time_base.num, stream.time_base.den,
			          INT_MAX);
		} else {
			auto mime = StreamMIME(*format->iformat, kind);
			if (reference.media_type == FileMediaType::AUDIO && !StringUtil::StartsWith(mime, "audio/")) {
				throw MediaFormatException("container does not belong to the requested FILE media type");
			}
			MediaValidateMIME(reference, mime);
		}
		probing = false;
	} catch (...) {
		Close();
		throw;
	}
}

MediaReader::~MediaReader() {
	Close();
}

void MediaReader::Close() noexcept {
	av_frame_free(&frame);
	av_packet_free(&packet);
	avcodec_free_context(&decoder);
	avformat_close_input(&format);
	if (io) {
		av_freep(&io->buffer);
		avio_context_free(&io);
	}
}

int MediaReader::Read(void *opaque, uint8_t *target, int size) noexcept {
	auto &self = *static_cast<MediaReader *>(opaque);
	try {
		self.CheckIO();
		if (size <= 0) {
			return AVERROR(EINVAL);
		}
		if (self.position == self.file->LogicalSize()) {
			return AVERROR_EOF;
		}
		auto count = self.ReadAt(target, uint64_t(size), self.position);
		self.position += count;
		return NumericCast<int>(count);
	} catch (...) {
		self.io_error = std::current_exception();
		return AVERROR_EXTERNAL;
	}
}

int64_t MediaReader::SeekIO(void *opaque, int64_t offset, int whence) noexcept {
	auto &self = *static_cast<MediaReader *>(opaque);
	try {
		self.CheckIO();
		whence &= ~AVSEEK_FORCE;
		if (whence == AVSEEK_SIZE) {
			return NumericCast<int64_t>(self.file->LogicalSize());
		}
		uint64_t origin;
		switch (whence) {
		case SEEK_SET:
			origin = 0;
			break;
		case SEEK_CUR:
			origin = self.position;
			break;
		case SEEK_END:
			origin = self.file->LogicalSize();
			break;
		default:
			return AVERROR(EINVAL);
		}
		if ((offset < 0 && uint64_t(-(offset + 1)) + 1 > origin) ||
		    (offset >= 0 && uint64_t(offset) > self.file->LogicalSize() - origin)) {
			return AVERROR(EINVAL);
		}
		self.position = offset < 0 ? origin - (uint64_t(-(offset + 1)) + 1) : origin + uint64_t(offset);
		return NumericCast<int64_t>(self.position);
	} catch (...) {
		self.io_error = std::current_exception();
		return AVERROR_EXTERNAL;
	}
}

int MediaReader::Interrupt(void *opaque) noexcept {
	auto &self = *static_cast<MediaReader *>(opaque);
	return self.context.IsInterrupted() || bool(self.io_error) ||
	       (self.probing && std::chrono::steady_clock::now() >= self.probe_deadline);
}

int MediaReader::DenyNestedIO(AVFormatContext *format, AVIOContext **, const char *, int, AVDictionary **) noexcept {
	if (!format || !format->opaque) {
		return AVERROR(EACCES);
	}
	auto &self = *static_cast<MediaReader *>(format->opaque);
	try {
		if (self.video_policy) {
			throw MediaFormatException("VideoFile does not permit nested external resources");
		}
		throw PermissionException("container requested an external resource outside its FILE view");
	} catch (...) {
		self.io_error = std::current_exception();
	}
	return AVERROR_EXTERNAL;
}

void MediaReader::CheckIO(bool check_probe_deadline) {
	MediaInterrupt(context);
	if (io_error) {
		std::rethrow_exception(io_error);
	}
	if ((probing || check_probe_deadline) && std::chrono::steady_clock::now() >= probe_deadline) {
		throw OutOfRangeException("native media metadata probe exceeded its time budget");
	}
}

AVStream &MediaReader::Stream() {
	return *format->streams[stream_index];
}

AVFormatContext &MediaReader::Format() {
	return *format;
}

AVFrame &MediaReader::Frame() {
	return *frame;
}

uint64_t MediaReader::BytesRead() const {
	return verifier ? verifier->BytesRead() : bytes_read;
}

uint64_t MediaReader::FrameBytes() const {
	return frame_bytes;
}

uint64_t MediaReader::LogicalSize() const {
	return file->LogicalSize();
}

uint64_t MediaReader::ReadAt(data_ptr_t target, uint64_t size, uint64_t offset) {
	CheckIO();
	if (offset > file->LogicalSize()) {
		throw OutOfRangeException("native media read is outside its FILE view");
	}
	auto count = MinValue<uint64_t>(size, file->LogicalSize() - offset);
	if (!count) {
		return 0;
	}
	if (bytes_read >= read_limit) {
		throw OutOfRangeException("native media exceeded its read/probe byte budget");
	}
	count = MinValue<uint64_t>(count, read_limit - bytes_read);
	{
		MediaProfileTimer timer(profile ? &profile->seconds : nullptr);
		if (verifier) {
			verifier->Read(*file, target, count, offset);
		} else {
			file->ReadExact(target, count, offset);
		}
	}
	if (profile) {
		profile->calls++;
	}
	bytes_read += count;
	return count;
}

int MediaReader::GetBuffer(AVCodecContext *decoder, AVFrame *frame, int flags) noexcept {
	auto &self = *static_cast<MediaReader *>(decoder->opaque);
	try {
		self.CheckIO();
		if (frame->width > 0 && frame->height > 0) {
			auto pixels = MediaProduct(frame->width, frame->height,
			                           self.video_policy ? 64 * MEDIA_MIB : self.max_pixels, "decoded pixels");
			MediaProduct(pixels, 8, self.frame_bytes, "decoded frame bytes");
			int width = frame->width, height = frame->height;
			int alignments[AV_NUM_DATA_POINTERS] = {};
			avcodec_align_dimensions2(decoder, &width, &height, alignments);
			int alignment = 1;
			for (auto value : alignments) {
				alignment = MaxValue(alignment, value);
			}
			auto padded_size = av_image_get_buffer_size(AVPixelFormat(frame->format), width, height, alignment);
			self.CheckCode(padded_size, "calculate decoder buffer size");
			MediaProduct(1, uint64_t(padded_size) + 4 * AV_INPUT_BUFFER_PADDING_SIZE, self.frame_bytes,
			             "padded decoder frame bytes");
		}
		if (frame->nb_samples > 0) {
			MediaProduct(frame->nb_samples, uint64_t(frame->ch_layout.nb_channels) * sizeof(double), self.frame_bytes,
			             "decoded audio frame bytes");
		}
		return avcodec_default_get_buffer2(decoder, frame, flags);
	} catch (...) {
		self.io_error = std::current_exception();
		return AVERROR_EXTERNAL;
	}
}

void MediaReader::OpenDecoder() {
	if (decoder) {
		return;
	}
	auto &parameters = *Stream().codecpar;
	if (parameters.width > 0 && parameters.height > 0) {
		MediaProduct(parameters.width, parameters.height, max_pixels, "decoded pixels");
		MediaProduct(uint64_t(parameters.width) * parameters.height, 8, frame_bytes, "decoded frame bytes");
	}
	const auto codec = avcodec_find_decoder(parameters.codec_id);
	if (!codec) {
		throw MediaFormatException("decoder is unavailable for the selected codec");
	}
	decoder = avcodec_alloc_context3(codec);
	packet = av_packet_alloc();
	frame = av_frame_alloc();
	if (!decoder || !packet || !frame) {
		throw OutOfMemoryException("Cannot allocate native media decoder state");
	}
	CheckCode(avcodec_parameters_to_context(decoder, &parameters), "configure decoder");
	decoder->thread_count = 1;
	decoder->opaque = this;
	decoder->get_buffer2 = GetBuffer;
	decoder->max_pixels = NumericCast<int64_t>(video_policy ? 64 * MEDIA_MIB : max_pixels);
	decoder->pkt_timebase = Stream().time_base;
	decoder->max_samples = NumericCast<int64_t>(frame_bytes / sizeof(double));
	auto code = avcodec_open2(decoder, codec, nullptr);
	CheckIO();
	CheckCode(code, "open decoder");
}

bool MediaReader::NextFrame() {
	OpenDecoder();
	av_frame_unref(frame);
	for (;;) {
		CheckIO();
		auto code = avcodec_receive_frame(decoder, frame);
		CheckIO();
		if (code >= 0) {
			if (video_policy && frame->duration < 0) {
				throw MediaFormatException("video parser reported a negative frame duration");
			}
			if (frame->width > 0 && frame->height > 0) {
				MediaProduct(frame->width, frame->height, max_pixels, "decoded pixels");
				MediaProduct(uint64_t(frame->width) * frame->height, 8, frame_bytes, "decoded frame bytes");
			}
			if (frame->nb_samples > 0) {
				MediaProduct(frame->nb_samples, uint64_t(frame->ch_layout.nb_channels) * sizeof(double), frame_bytes,
				             "decoded audio frame bytes");
			}
			return true;
		}
		if (code == AVERROR_EOF) {
			return false;
		}
		if (code != AVERROR(EAGAIN)) {
			CheckCode(code, "decode frame");
		}
		if (source_eof) {
			if (decoder_flushed) {
				throw MediaFormatException("decoder requested packets after end of stream");
			}
			auto flush_code = avcodec_send_packet(decoder, nullptr);
			CheckIO();
			CheckCode(flush_code, "flush decoder");
			decoder_flushed = true;
			continue;
		}
		for (;;) {
			MediaInterrupt(context);
			if (video_policy) {
				probing = true;
				probe_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
			}
			code = av_read_frame(format, packet);
			CheckIO();
			probing = false;
			if (code == AVERROR_EOF) {
				source_eof = true;
				break;
			}
			CheckCode(code, "read packet");
			if (packet->stream_index == stream_index) {
				code = avcodec_send_packet(decoder, packet);
				CheckIO();
				av_packet_unref(packet);
				CheckCode(code, "submit packet");
				break;
			}
			av_packet_unref(packet);
		}
	}
}

void MediaReader::Seek(int64_t timestamp) {
	OpenDecoder();
	MediaInterrupt(context);
	const auto code = av_seek_frame(format, stream_index, timestamp, AVSEEK_FLAG_BACKWARD);
	CheckIO();
	CheckCode(code, "seek to keyframe");
	avcodec_flush_buffers(decoder);
	av_packet_unref(packet);
	av_frame_unref(frame);
	source_eof = false;
	decoder_flushed = false;
}

} // namespace duckdb
