// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/exchange/flight_client.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/distributed/exchange/flight_ticket.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"

#include <arrow/c/bridge.h>
#include <arrow/flight/api.h>
#include <arrow/ipc/api.h>

namespace duckdb {
namespace distributed {

namespace {

DuckDBError FlightClientArrowToError(const arrow::Status &status, const std::string &context) {
	return DuckDBError::external_error(context + ": " + status.ToString());
}

DuckDBResult<std::unique_ptr<arrow::flight::FlightClient>> ConnectClient(const std::string &location) {
	auto location_res = arrow::flight::Location::Parse(location);
	if (!location_res.ok()) {
		return DuckDBResult<std::unique_ptr<arrow::flight::FlightClient>>::err(
		    FlightClientArrowToError(location_res.status(), "parse flight location"));
	}
	auto client_res = arrow::flight::FlightClient::Connect(std::move(location_res).ValueOrDie());
	if (!client_res.ok()) {
		return DuckDBResult<std::unique_ptr<arrow::flight::FlightClient>>::err(
		    FlightClientArrowToError(client_res.status(), "connect flight client"));
	}
	return DuckDBResult<std::unique_ptr<arrow::flight::FlightClient>>::ok(std::move(client_res).ValueOrDie());
}

DuckDBResult<std::unique_ptr<ColumnDataCollection>>
BatchesToCollection(ClientContext &context, const std::shared_ptr<arrow::Schema> &schema,
                    const std::vector<std::shared_ptr<arrow::RecordBatch>> &batches,
                    const std::vector<LogicalType> &expected_types) {
	ArrowSchema c_schema;
	c_schema.Init();
	auto export_status = arrow::ExportSchema(*schema, &c_schema);
	if (!export_status.ok()) {
		return DuckDBResult<std::unique_ptr<ColumnDataCollection>>::err(
		    FlightClientArrowToError(export_status, "export schema"));
	}

	ArrowTableSchema arrow_table;
	ArrowTableFunction::PopulateArrowTableSchema(context, arrow_table, c_schema);
	if (c_schema.release) {
		c_schema.release(&c_schema);
	}

	auto &types = arrow_table.GetTypes();
	duckdb::vector<LogicalType> output_types;
	bool needs_cast = false;
	if (!expected_types.empty()) {
		output_types.reserve(expected_types.size());
		for (auto &type : expected_types) {
			output_types.push_back(type);
		}
		if (types.size() != expected_types.size()) {
			return DuckDBResult<std::unique_ptr<ColumnDataCollection>>::err(
			    DuckDBError::value_error("flight partition types mismatch"));
		}
		for (idx_t idx = 0; idx < types.size(); idx++) {
			if (types[idx] != expected_types[idx]) {
				if (expected_types[idx].id() != LogicalTypeId::AGGREGATE_STATE ||
				    types[idx].id() != LogicalTypeId::BLOB) {
					return DuckDBResult<std::unique_ptr<ColumnDataCollection>>::err(
					    DuckDBError::value_error("flight partition types mismatch"));
				}
				needs_cast = true;
			}
		}
	} else {
		output_types = types;
	}

	std::unique_ptr<ColumnDataCollection> collection(
	    new ColumnDataCollection(Allocator::DefaultAllocator(), output_types));
	ColumnDataAppendState append_state;
	collection->InitializeAppend(append_state);

	for (const auto &batch : batches) {
		if (!batch || batch->num_rows() == 0) {
			continue;
		}
		ArrowArray c_array;
		c_array.Init();
		auto export_array_status = arrow::ExportRecordBatch(*batch, &c_array);
		if (!export_array_status.ok()) {
			return DuckDBResult<std::unique_ptr<ColumnDataCollection>>::err(
			    FlightClientArrowToError(export_array_status, "export record batch"));
		}

		auto array_wrapper = make_uniq<ArrowArrayWrapper>();
		array_wrapper->arrow_array = c_array;
		ArrowScanLocalState scan_state(std::move(array_wrapper), context);
		scan_state.chunk_offset = 0;

		DataChunk output;
		output.Initialize(Allocator::DefaultAllocator(), types);
		output.SetCardinality(batch->num_rows());
		ArrowTableFunction::ArrowToDuckDB(scan_state, arrow_table.GetColumns(), output, 0);
		if (needs_cast) {
			DataChunk casted;
			casted.Initialize(Allocator::DefaultAllocator(), output_types);
			casted.SetCardinality(output.size());
			for (idx_t col = 0; col < output_types.size(); col++) {
				if (output.data[col].GetType() == output_types[col]) {
					casted.data[col].Reference(output.data[col]);
				} else {
					VectorOperations::Cast(context, output.data[col], casted.data[col], output.size());
				}
			}
			collection->Append(append_state, casted);
		} else {
			collection->Append(append_state, output);
		}
	}

	return DuckDBResult<std::unique_ptr<ColumnDataCollection>>::ok(std::move(collection));
}

} // namespace

FlightClient::FlightClient(FlightClientConfig config) : config_(std::move(config)) {
}

const FlightClientConfig &FlightClient::config() const {
	return config_;
}

DuckDBResult<void> FlightClient::Validate() const {
	if (config_.location.empty()) {
		return DuckDBResult<void>::err(DuckDBError::value_error("flight client location is empty"));
	}
	auto client_res = ConnectClient(config_.location);
	if (client_res.is_err()) {
		return DuckDBResult<void>::err(client_res.error());
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<std::unique_ptr<ColumnDataCollection>>
FlightClient::FetchPartition(ClientContext &context, const FlightExchangeTicket &ticket,
                             const std::vector<LogicalType> &expected_types) const {
	if (config_.location.empty()) {
		return DuckDBResult<std::unique_ptr<ColumnDataCollection>>::err(
		    DuckDBError::value_error("flight client location is empty"));
	}

	auto client_res = ConnectClient(config_.location);
	if (client_res.is_err()) {
		return DuckDBResult<std::unique_ptr<ColumnDataCollection>>::err(client_res.error());
	}
	auto client = std::move(client_res.value());

	arrow::flight::Ticket flight_ticket;
	flight_ticket.ticket = ticket.Serialize();

	arrow::flight::FlightCallOptions call_options;
	if (config_.timeout_seconds > 0.0) {
		call_options.timeout = arrow::flight::TimeoutDuration(config_.timeout_seconds);
	}

	auto reader_res = client->DoGet(call_options, flight_ticket);
	if (!reader_res.ok()) {
		return DuckDBResult<std::unique_ptr<ColumnDataCollection>>::err(
		    FlightClientArrowToError(reader_res.status(), "flight do_get"));
	}
	auto reader = std::move(reader_res).ValueOrDie();
	auto schema_res = reader->GetSchema();
	if (!schema_res.ok()) {
		return DuckDBResult<std::unique_ptr<ColumnDataCollection>>::err(
		    FlightClientArrowToError(schema_res.status(), "flight get schema"));
	}
	auto schema = std::move(schema_res).ValueOrDie();

	auto batches_res = reader->ToRecordBatches(call_options.stop_token);
	if (!batches_res.ok()) {
		return DuckDBResult<std::unique_ptr<ColumnDataCollection>>::err(
		    FlightClientArrowToError(batches_res.status(), "flight read batches"));
	}
	auto batches = std::move(batches_res).ValueOrDie();

	return BatchesToCollection(context, schema, batches, expected_types);
}

} // namespace distributed
} // namespace duckdb
