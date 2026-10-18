// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Included after Payload() inside the isolated format-1005 implementation.
std::string ReadExtentBytes(sqlite3 *db, int64_t inode, int64_t offset, int64_t length, const View &view) {
	std::string result(size_t(length), '\0');
	if (!length)
		return result;
	Statement rows(db, "SELECT b.block,b.count,b.payload_offset,p.data FROM block_versions b "
	                   "LEFT JOIN block_payloads p ON p.id=b.payload WHERE b.inode=?1 "
	                   "AND b.block>=max(0,?2-63) AND b.block<=?3 AND b.block+b.count>?2 "
	                   "AND b.low<=?4 AND b.high>?4 AND b.deleted=0 ORDER BY b.block");
	rows.Bind(1, inode);
	rows.Bind(2, offset / BLOCK_SIZE);
	rows.Bind(3, (offset + length - 1) / BLOCK_SIZE);
	rows.Bind(4, view.point);
	int64_t previous_end = -1, pending_offset = 0, pending_target = 0, pending_size = 0;
	auto flush = [&] {
		if (pending_size)
			PayloadFile::Get(db).Read(&result[size_t(pending_target)], size_t(pending_size), pending_offset);
		pending_size = 0;
	};
	while (rows.Step()) {
		auto block = rows.Integer(0), count = rows.Integer(1), displacement = rows.Integer(2);
		auto bytes = rows.BytesData(3);
		if (!bytes)
			Fail(ErrorCode::Storage, "Missing block payload");
		auto stored = rows.BytesSize(3);
		if ((stored != 8 && stored != BLOCK_SIZE) || count < 1 || count > 64 || block < 0 ||
		    block > std::numeric_limits<int64_t>::max() / BLOCK_SIZE - (count - 1) || displacement < 0 ||
		    displacement % BLOCK_SIZE || displacement > 64 * BLOCK_SIZE - count * BLOCK_SIZE ||
		    (stored == BLOCK_SIZE && (count != 1 || displacement)))
			Fail(ErrorCode::Storage, "Invalid extent payload range");
		if (block < previous_end)
			Fail(ErrorCode::Storage, "Overlapping visible versions");
		previous_end = block + count;
		auto start = std::max(offset, block * BLOCK_SIZE);
		auto source = start - block * BLOCK_SIZE, target = start - offset;
		auto amount = std::min(count * BLOCK_SIZE - source, length - target);
		if (stored == 8) {
			auto physical = ExtentPhysical(rows.Bytes(3), displacement + source);
			if (pending_size && (physical != pending_offset + pending_size || target != pending_target + pending_size))
				flush();
			if (!pending_size) {
				pending_offset = physical;
				pending_target = target;
			}
			pending_size += amount;
		} else {
			flush();
			std::memcpy(&result[size_t(target)], bytes + source, size_t(amount));
		}
	}
	flush();
	return result;
}

void InsertExtentBatch(sqlite3 *db, int64_t inode, const std::vector<std::pair<int64_t, std::string_view>> &blocks,
                       int64_t &last_payload, int64_t physical, const View &view) {
	Statement payload(db, "INSERT INTO block_payloads(id,data) VALUES(?,?)");
	payload.Bind(1, ++last_payload);
	payload.BindBytes(2, PayloadFile::Token(physical));
	payload.Step();
	// A physical batch can contain holes in its logical keys. Preserve those
	// holes by making a separate extent for every contiguous logical run.
	std::vector<std::pair<size_t, size_t>> runs;
	for (size_t i = 0; i < blocks.size();) {
		auto begin = i++;
		while (i < blocks.size() && blocks[i].first == blocks[i - 1].first + 1)
			++i;
		runs.emplace_back(begin, i - begin);
	}
	std::string sql =
	    "INSERT INTO block_versions(inode,block,low,high,writer,deleted,payload,count,payload_offset) VALUES ";
	for (size_t i = 0; i < runs.size(); ++i) {
		if (i)
			sql += ',';
		sql += "(?1,?" + std::to_string(6 + 3 * i) + ",?2,?3,?4,0,?5,?" + std::to_string(7 + 3 * i) + ",?" +
		       std::to_string(8 + 3 * i) + ")";
	}
	Statement versions(db, sql);
	versions.Bind(1, inode);
	versions.Bind(2, view.point);
	versions.Bind(3, view.high);
	versions.Bind(4, view.writer);
	versions.Bind(5, last_payload);
	for (size_t i = 0; i < runs.size(); ++i) {
		versions.Bind(int(6 + 3 * i), blocks[runs[i].first].first);
		versions.Bind(int(7 + 3 * i), int64_t(runs[i].second));
		versions.Bind(int(8 + 3 * i), int64_t(runs[i].first) * BLOCK_SIZE);
	}
	versions.Step();
}
