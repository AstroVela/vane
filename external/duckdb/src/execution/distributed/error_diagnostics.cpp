// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/error_diagnostics.hpp"

namespace duckdb {
namespace distributed {

const size_t ErrorDiagnostic::MAX_TYPE_BYTES;
const size_t ErrorDiagnostic::MAX_MESSAGE_BYTES;
const size_t ErrorDiagnostic::MAX_TRACEBACK_BYTES;
const size_t ErrorDiagnostic::MAX_CAUSE_BYTES;
const size_t ErrorDiagnostic::MAX_TRACEBACK_FRAMES;
const size_t ErrorDiagnostic::MAX_CAUSES;

const size_t ErrorDiagnostics::MAX_DETAILS;
const size_t ErrorDiagnostics::MAX_DETAIL_BYTES;
const size_t ErrorDiagnostics::MAX_LABEL_BYTES;
const size_t ErrorDiagnostics::MAX_TOTAL_BYTES;

} // namespace distributed
} // namespace duckdb
