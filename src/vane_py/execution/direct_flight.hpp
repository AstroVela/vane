// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "direct_exchange.hpp"

struct ArrowSchema;
struct ArrowArray;

namespace duckdb {
namespace vane_execution {

// One query's finite transport reservation. Tickets include both worker epochs,
// query/attempt, routing/schema identity and a random access capability. Exact
// ticket matching fences every RPC; ticket contents are never included in errors.
class DirectFlight {
public:
	DirectFlight(const string &bind_host, const string &advertise_host, idx_t max_links, idx_t staging_bytes,
	             idx_t frame_bytes, int port = 0, const string &certificate = "", const string &private_key = "");
	~DirectFlight();
	string Location() const;
	void Publish(const string &ticket, shared_ptr<DirectChannel> channel, const string &consumer);
	void Subscribe(const string &location, const string &ticket, shared_ptr<DirectChannel> channel,
	               const string &producer, double timeout, const string &root_certificates = "");
	void Revoke(const string &ticket);
	bool Delivered(const string &ticket) const;
	void Cancel(const string &reason);
	void Close();
	string Error() const;
	idx_t ActiveLinks() const;
	bool Ready() const;
	static idx_t StagingPerLink(idx_t frame_bytes);
	static void ExportSchema(const vector<LogicalType> &types, const vector<string> &names, ArrowSchema *out);
	static void ExportBatch(DataChunk &chunk, const vector<string> &names, ArrowArray *array, ArrowSchema *schema);

private:
	struct Impl;
	unique_ptr<Impl> impl;
};

} // namespace vane_execution
} // namespace duckdb
