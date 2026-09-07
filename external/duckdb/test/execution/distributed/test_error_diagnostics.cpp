// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "utf8proc_wrapper.hpp"

using namespace duckdb::distributed;

static bool HasCompleteDiagnosticUnits(std::string_view text, std::string_view unit) {
	while (!text.empty()) {
		if (text.substr(0, unit.size()) == unit) {
			text.remove_prefix(unit.size());
		} else if (text[0] == '.') {
			// Repeated limiting can join existing and new omission markers.
			text.remove_prefix(1);
		} else {
			return false;
		}
	}
	return true;
}

TEST_CASE("Distributed diagnostics preserve UTF-8 at byte boundaries", "[distributed][diagnostics]") {
	for (size_t limit = 0; limit < 12; limit++) {
		const auto bounded = BoundDiagnosticText("界🙂界🙂", limit);
		REQUIRE(bounded.size() <= limit);
		REQUIRE(duckdb::Utf8Proc::IsValid(bounded.data(), bounded.size()));
		const auto edges = ErrorDiagnostics::BoundDetailText(std::string("界\0🙂\xff", 9), limit);
		REQUIRE(edges.size() <= limit);
		REQUIRE(duckdb::Utf8Proc::IsValid(edges.data(), edges.size()));
	}
	REQUIRE(BoundDiagnosticText(std::string(4096, 'a'), 4096) == std::string(4096, 'a'));
	REQUIRE(BoundDiagnosticText(std::string(4097, 'a'), 4096) == std::string(4093, 'a') + "...");
	const auto invalid = BoundDiagnosticText(std::string("a\0b\xff", 4), 32);
	REQUIRE(invalid.find("a\\x00b") == 0);
	REQUIRE(duckdb::Utf8Proc::IsValid(invalid.data(), invalid.size()));
}

TEST_CASE("Distributed diagnostic edges use normalized byte sizes", "[distributed][diagnostics]") {
	for (const size_t limit : {size_t(64), ErrorDiagnostic::MAX_MESSAGE_BYTES, ErrorDiagnostics::MAX_DETAIL_BYTES}) {
		for (const char byte : {'\0', char(0xFF)}) {
			const auto raw = "head:" + std::string(limit / 2, byte) + ":reason-tail";
			REQUIRE(raw.size() < limit);
			const auto bounded = ErrorDiagnostics::BoundDetailText(raw, limit);
			REQUIRE(bounded.size() <= limit);
			REQUIRE(bounded.find("head:") == 0);
			REQUIRE(bounded.find(":reason-tail") != std::string::npos);
			REQUIRE(duckdb::Utf8Proc::IsValid(bounded.data(), bounded.size()));
		}
	}
	const auto raw = "head:" + std::string(1000, '\0') + ":reason-tail";
	REQUIRE(ErrorDiagnostics::FromText(raw).AppendTo().find(":reason-tail") != std::string::npos);
	REQUIRE(std::string(DuckDBError::external_error(raw).what()).find(":reason-tail") != std::string::npos);
}

TEST_CASE("Distributed diagnostic cuts retain complete escape tokens", "[distributed][diagnostics]") {
	std::vector<size_t> limits;
	for (size_t limit = 0; limit <= 32; limit++) {
		limits.push_back(limit);
	}
	for (const auto limit :
	     {size_t(63), size_t(64), size_t(65), ErrorDiagnostic::MAX_MESSAGE_BYTES, ErrorDiagnostics::MAX_DETAIL_BYTES}) {
		limits.push_back(limit);
	}
	for (const std::string unit :
	     {std::string(1, '\0'), std::string("\\x00"), std::string("\\xff"), std::string("\\ud800"),
	      std::string("\\U0001f642"), std::string("界"), std::string("🙂")}) {
		const auto normalized = unit == std::string(1, '\0') ? "\\x00" : unit;
		std::string raw;
		for (size_t i = 0; i < 2000; i++) {
			raw += unit;
		}
		for (const auto limit : limits) {
			INFO("unit=" << normalized << ", limit=" << limit);
			const auto prefix = BoundDiagnosticText(raw, limit);
			const auto edges = ErrorDiagnostics::BoundDetailText(raw, limit);
			REQUIRE(prefix.size() <= limit);
			REQUIRE(edges.size() <= limit);
			REQUIRE(HasCompleteDiagnosticUnits(prefix, normalized));
			REQUIRE(HasCompleteDiagnosticUnits(edges, normalized));
			REQUIRE(duckdb::Utf8Proc::IsValid(edges.data(), edges.size()));
			if (unit != std::string(1, '\0')) {
				REQUIRE(BoundDiagnosticCString(raw.c_str(), limit) == prefix);
			}
			if (limit >= 12) {
				const auto rebound = ErrorDiagnostics::BoundDetailText(edges, limit / 2);
				REQUIRE(rebound.size() <= limit / 2);
				REQUIRE(HasCompleteDiagnosticUnits(rebound, normalized));
			}
		}
		REQUIRE(BoundDiagnosticText(unit, normalized.size()) == normalized);
		REQUIRE(ErrorDiagnostics::BoundDetailText(unit, normalized.size()) == normalized);
	}
	const auto bounded = ErrorDiagnostics::BoundDetailText("head:" + std::string(1000, '\0') + ":tail",
	                                                       ErrorDiagnostic::MAX_MESSAGE_BYTES);
	REQUIRE(bounded.substr(0, 5) == "head:");
	REQUIRE(bounded.substr(bounded.size() - 5) == ":tail");
	REQUIRE(HasCompleteDiagnosticUnits(std::string_view(bounded).substr(5, bounded.size() - 10), "\\x00"));
	REQUIRE(ErrorDiagnostics::BoundDetailText(std::string(1000, '\0'), 7) == "...\\x00");
	REQUIRE(ErrorDiagnostics::BoundDetailText("\\ud800\\ud800", 9) == "...\\ud800");
}

TEST_CASE("Distributed diagnostics retain primary summaries through nested aggregation", "[distributed][diagnostics]") {
	ErrorDiagnostics cleanup;
	for (size_t i = 0; i < 20; i++) {
		cleanup.Add(
		    "release[" + std::to_string(i) + "]",
		    ErrorDiagnostics::FromDiagnostic(ErrorDiagnostic(
		        "RuntimeError", "cleanup-" + std::to_string(i) + std::string(8000, 'c'), std::string(16000, 't'))));
	}
	auto primary = ErrorDiagnostics::FromDiagnostic(
	    ErrorDiagnostic("RuntimeError", "primary status failure", std::string(16000, 'p')));
	cleanup.AddPrimary("status", primary);
	auto propagated = DuckDBError::external_error(cleanup.WithContext(std::string(10000, 'w')));
	for (size_t i = 0; i < 30; i++) {
		propagated = DuckDBError::external_error(propagated.Diagnostics().WithContext("outer wrapper"));
	}
	const auto message = propagated.Diagnostics().AppendTo();
	REQUIRE(propagated.Diagnostics().Count() == 21);
	REQUIRE(message.find("primary status failure") < message.find("cleanup-0"));
	REQUIRE(message.find("cleanup-14") < message.find("Traceback ["));
	REQUIRE(message.find("cleanup-15") == std::string::npos);
	REQUIRE(message.find("additional 5 error(s) omitted") != std::string::npos);
	REQUIRE(message.size() <= ErrorDiagnostics::MAX_TOTAL_BYTES);
	REQUIRE(duckdb::Utf8Proc::IsValid(message.data(), message.size()));
}

TEST_CASE("Distributed diagnostics bound summary fields independently", "[distributed][diagnostics]") {
	const auto diagnostic = ErrorDiagnostics::FromDiagnostic(
	    ErrorDiagnostic(std::string(10000, 'x'), "useful message", std::string(10000, 't'), std::string(10000, 'c')));
	const auto rendered = diagnostic.WithContext(std::string(10000, 'l')).AppendTo();
	REQUIRE(rendered.find("useful message") != std::string::npos);
	REQUIRE(rendered.size() <= ErrorDiagnostics::MAX_DETAIL_BYTES);
	REQUIRE(rendered.find(std::string(129, 'x')) == std::string::npos);
}

TEST_CASE("Distributed status diagnostics retain both ends when aggregated", "[distributed][diagnostics]") {
	const auto status = ErrorDiagnostics::BoundDetailText("status-head:" + std::string(10000, 'x') + ":status-tail");
	const auto rendered = ErrorDiagnostics::FromText(status).WithContext("status").AppendTo();
	REQUIRE(rendered.find("status-head:") != std::string::npos);
	REQUIRE(rendered.find(":status-tail") != std::string::npos);
	REQUIRE(rendered.size() <= ErrorDiagnostics::MAX_DETAIL_BYTES);
}

TEST_CASE("Distributed result copies retain structured error diagnostics", "[distributed][diagnostics]") {
	auto diagnostic = ErrorDiagnostics::FromDiagnostic(ErrorDiagnostic("ValueError", "original", "source.py:1"));
	auto result = DuckDBResult<int>::err(DuckDBError::external_error(diagnostic.WithContext("operation")));
	auto copy = result;
	auto moved = std::move(copy);
	REQUIRE(moved.error().type() == DuckDBError::Type::ExternalError);
	REQUIRE(moved.error().Diagnostics().AppendTo() == diagnostic.WithContext("operation").AppendTo());
	REQUIRE(DuckDBResult<void>::ok().is_ok());
}

TEST_CASE("Distributed errors render their original type exactly once", "[distributed][diagnostics]") {
	const auto original = DuckDBError::value_error("bad input");
	REQUIRE(std::string(original.what()) == "DuckDBError::ValueError: bad input");
	REQUIRE(original.Diagnostics().AppendTo() == original.what());
	const auto wrapped = DuckDBError::external_error(original.Diagnostics().WithContext("operation"));
	REQUIRE(std::string(wrapped.what()) == "operation: DuckDBError::ValueError: bad input");
	REQUIRE(wrapped.type() == DuckDBError::Type::ExternalError);
	const auto external = DuckDBError::external_error("remote failure");
	REQUIRE(std::string(DuckDBError::external_error(external.Diagnostics()).what()) == external.what());
	const auto oversized =
	    DuckDBError::value_error("native-head:" + std::string(10000, 'x') + ":planned provider timeout");
	REQUIRE(std::string(oversized.what()).size() <= ErrorDiagnostics::MAX_DETAIL_BYTES);
	REQUIRE(std::string(oversized.what()).find("native-head:") != std::string::npos);
	REQUIRE(std::string(oversized.what()).find(":planned provider timeout") != std::string::npos);
	const auto propagated = DuckDBError::external_error(oversized.Diagnostics().WithContext("status"));
	REQUIRE(std::string(propagated.what()).find(":planned provider timeout") != std::string::npos);
}
