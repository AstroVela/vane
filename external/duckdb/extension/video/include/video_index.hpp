// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "media_reader.hpp"
#include "video_frame_contract.hpp"

namespace duckdb {

//! One execution-owned decoder. Index values contain no handles or credentials.
//! Indexed and sequential scans share selection, limits, and output metadata.
class VideoFrameCursor {
public:
	VideoFrameCursor(ClientContext &context, const Value &file, const VideoFrameOptions &options,
	                 VideoFrameOperation operation, const Value &index);
	~VideoFrameCursor();
	bool Next();
	MediaReader &Reader();
	uint64_t FrameIndex() const;
	Value FrameTime() const;
	Value Statistics() const;

private:
	struct State;
	unique_ptr<State> state;
};

void RegisterVideoIndexFunctions(ExtensionLoader &loader);

} // namespace duckdb
