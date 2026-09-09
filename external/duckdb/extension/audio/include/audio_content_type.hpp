// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "media_reader.hpp"
#include "duckdb/common/string_util.hpp"
#include <cstring>
#include <map>

namespace duckdb {

//! Validate codec declarations without giving the decoder an ungoverned URL or
//! requiring Python. Container MIME aliases remain governed by MediaReader.
class AudioContentType {
private:
	struct Part {
		string value;
		bool encoded;
		bool continued;
	};

	class Parameters {
	public:
		explicit Parameters(const string &text) : text(text), position(text.find(';')) {
		}

		std::map<string, string> Read() {
			std::map<string, std::map<idx_t, Part>> groups;
			while (position != string::npos && position < text.size()) {
				if (text[position++] != ';') {
					Invalid();
				}
				Space();
				if (position == text.size()) {
					break;
				}
				auto name = StringUtil::Lower(Token());
				Space();
				if (position == text.size() || text[position++] != '=') {
					Invalid();
				}
				Space();
				auto value = position == text.size() || text[position] == ';' ? string()
				             : text[position] == '"'                          ? Quoted()
				                                                              : Token();
				Space();
				idx_t index = 0;
				bool encoded = false, continued = false;
				auto star = name.find('*');
				if (star != string::npos) {
					auto suffix = name.substr(star + 1);
					name.resize(star);
					if (name.empty()) {
						Invalid();
					}
					encoded = suffix.empty() || suffix.back() == '*';
					if (!suffix.empty() && encoded) {
						suffix.pop_back();
						if (suffix.empty()) {
							Invalid();
						}
					}
					continued = !suffix.empty();
					if (suffix.size() > 1 && suffix[0] == '0') {
						Invalid();
					}
					for (auto c : suffix) {
						if (c < '0' || c > '9' || index > text.size() / 10) {
							Invalid();
						}
						index = index * 10 + idx_t(c - '0');
						if (index > text.size()) {
							Invalid();
						}
					}
				}
				auto &parts = groups[name];
				if (!parts.emplace(index, Part {std::move(value), encoded, continued}).second) {
					Invalid();
				}
			}
			std::map<string, string> result;
			for (auto &group : groups) {
				string value;
				idx_t next = 0;
				for (auto &entry : group.second) {
					auto &part = entry.second;
					if (entry.first != next++ || (group.second.size() > 1 && !part.continued)) {
						Invalid();
					}
					if (part.encoded) {
						if (entry.first == 0) {
							// RFC 2231: charset'language'percent-encoded-value.
							// Codec identifiers use ASCII-compatible MIME charsets.
							auto first = part.value.find('\'');
							auto second = first == string::npos ? string::npos : part.value.find('\'', first + 1);
							if (second == string::npos) {
								Invalid();
							}
							auto charset = StringUtil::Lower(part.value.substr(0, first));
							if ((group.first == "codec" || group.first == "codecs") && !charset.empty() &&
							    charset != "utf-8" && charset != "utf8" && charset != "us-ascii" &&
							    charset != "ascii" && charset != "iso-8859-1" && charset != "iso8859-1" &&
							    charset != "latin1" && charset != "latin-1") {
								throw MediaFormatException(
								    "AUDIOFILE codec parameter uses an unsupported MIME charset");
							}
							part.value.erase(0, second + 1);
						}
						value += Decode(part.value);
					} else {
						value += part.value;
					}
				}
				result.emplace(group.first, std::move(value));
			}
			return result;
		}

	private:
		[[noreturn]] static void Invalid() {
			throw MediaFormatException("AUDIOFILE content_type has invalid codec parameter syntax");
		}

		static bool IsToken(char c) {
			return c > 32 && c < 127 && !std::strchr("()<>@,;:\\\"/[]?=", c);
		}

		void Space() {
			while (position < text.size()) {
				if (text[position] == ' ' || text[position] == '\t') {
					position++;
				} else if (text[position] == '(') {
					idx_t depth = 1;
					position++;
					while (depth && position < text.size()) {
						auto c = text[position++];
						if (c == '\\') {
							if (position == text.size()) {
								Invalid();
							}
							position++;
						} else if (c == '(') {
							depth++;
						} else if (c == ')') {
							depth--;
						}
					}
					if (depth) {
						Invalid();
					}
				} else {
					break;
				}
			}
		}

		string Token() {
			auto start = position;
			while (position < text.size() && IsToken(text[position])) {
				position++;
			}
			if (position == start) {
				Invalid();
			}
			return text.substr(start, position - start);
		}

		string Quoted() {
			position++;
			string result;
			while (position < text.size()) {
				auto c = text[position++];
				if (c == '"') {
					return result;
				}
				if (c == '\\') {
					if (position == text.size()) {
						Invalid();
					}
					c = text[position++];
				}
				result += c;
			}
			Invalid();
		}

		static string Decode(const string &value) {
			auto hex = [](char c) -> int {
				return c >= '0' && c <= '9'   ? c - '0'
				       : c >= 'a' && c <= 'f' ? c - 'a' + 10
				       : c >= 'A' && c <= 'F' ? c - 'A' + 10
				                              : -1;
			};
			string result;
			for (idx_t i = 0; i < value.size(); i++) {
				if (value[i] == '%' && i + 2 < value.size() && hex(value[i + 1]) >= 0 && hex(value[i + 2]) >= 0) {
					result += char(hex(value[i + 1]) * 16 + hex(value[i + 2]));
					i += 2;
				} else {
					result += value[i];
				}
			}
			return result;
		}

		const string &text;
		idx_t position;
	};

	static string Normalize(string value) {
		StringUtil::Trim(value);
		return StringUtil::Lower(value);
	}

	static const char *WaveCodec(AVCodecID codec) {
		switch (codec) {
		case AV_CODEC_ID_PCM_U8:
		case AV_CODEC_ID_PCM_S16LE:
		case AV_CODEC_ID_PCM_S16BE:
		case AV_CODEC_ID_PCM_S24LE:
		case AV_CODEC_ID_PCM_S24BE:
		case AV_CODEC_ID_PCM_S32LE:
		case AV_CODEC_ID_PCM_S32BE:
			return "1";
		case AV_CODEC_ID_PCM_F32LE:
		case AV_CODEC_ID_PCM_F32BE:
		case AV_CODEC_ID_PCM_F64LE:
		case AV_CODEC_ID_PCM_F64BE:
			return "3";
		case AV_CODEC_ID_PCM_ALAW:
			return "6";
		case AV_CODEC_ID_PCM_MULAW:
			return "7";
		case AV_CODEC_ID_ADPCM_MS:
			return "2";
		case AV_CODEC_ID_ADPCM_IMA_WAV:
			return "11";
		case AV_CODEC_ID_GSM_MS:
			return "31";
		case AV_CODEC_ID_ADPCM_G726:
		case AV_CODEC_ID_ADPCM_G726LE:
			return "40";
		case AV_CODEC_ID_MP3:
			return "55";
		default:
			return "";
		}
	}

public:
	static void Validate(const FileReference &file, MediaReader &reader) {
		if (!file.has_content_type) {
			return;
		}
		for (auto c : file.content_type) {
			if ((static_cast<unsigned char>(c) < 32 && c != '\t') || c == 127) {
				throw MediaFormatException("AUDIOFILE content_type has invalid codec parameter syntax");
			}
		}
		auto parameters = Parameters(file.content_type).Read();
		auto codec = parameters.find("codec"), codecs = parameters.find("codecs");
		if (codec == parameters.end() && codecs == parameters.end()) {
			return;
		}
		auto invalid = [&]() {
			throw MediaFormatException("AUDIOFILE content_type '" + file.content_type +
			                           "' contradicts the detected audio codec");
		};
		const string format = reader.Format().iformat->name;
		auto detected = reader.Stream().codecpar->codec_id;
		if (codec != parameters.end()) {
			auto declared = Normalize(file.content_type.substr(0, file.content_type.find(';')));
			bool wave = declared == "audio/wav" || declared == "audio/x-wav" || declared == "audio/vnd.wave" ||
			            declared == "audio/wave";
			auto value = Normalize(codec->second);
			if (codecs != parameters.end() || !wave || format != "wav" || value.empty() ||
			    value != WaveCodec(detected)) {
				invalid();
			}
		} else {
			const string expected = format != "ogg"                  ? ""
			                        : detected == AV_CODEC_ID_OPUS   ? "opus"
			                        : detected == AV_CODEC_ID_VORBIS ? "vorbis"
			                                                         : "";
			idx_t start = 0;
			do {
				auto end = codecs->second.find(',', start);
				auto value = Normalize(codecs->second.substr(start, end == string::npos ? end : end - start));
				if (expected.empty() || value != expected) {
					invalid();
				}
				if (end == string::npos) {
					break;
				}
				start = end + 1;
			} while (true);
		}
	}
};

} // namespace duckdb
