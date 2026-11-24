// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Included inside workspace.cpp's record-layer namespace in format 1005 only.
// Spatial ranges contain at most 64 blocks. Coordinate intervals keep their
// original half-open meaning; mutations split both dimensions explicitly.
int64_t ExtentPhysical(const std::string &token, int64_t displacement) {
	auto base = PayloadFile::Offset(token);
	if (displacement < 0 || displacement >= 64 * BLOCK_SIZE ||
	    base > std::numeric_limits<int64_t>::max() - 64 * BLOCK_SIZE)
		Fail(ErrorCode::Storage, "Invalid extent payload range");
	return base + displacement;
}

bool GetExtent(sqlite3 *db, const Key &key, const Coordinate &point, Record &record) {
	Statement rows(db, Select(BLOCKS) + " WHERE inode=?1 AND block<=?2 AND block>=max(0,?2-63) AND block+count>?2 "
	                                    "AND low<=?3 AND high>?3 AND deleted=0");
	rows.Bind(1, key.inode);
	rows.Bind(2, key.block);
	rows.Bind(3, point);
	if (!rows.Step())
		return false;
	record = ReadRecord(rows, BLOCKS);
	record.fields[2] += (key.block - record.key.block) * BLOCK_SIZE;
	record.fields[1] = 1;
	record.key = key;
	if (rows.Step())
		Fail(ErrorCode::Storage, "Overlapping visible versions");
	return true;
}

std::vector<Record> VisibleExtents(sqlite3 *db, const Coordinate &point, int64_t inode) {
	Statement rows(db, Select(BLOCKS) + " WHERE low<=?1 AND high>?1 AND deleted=0" +
	                       (inode < 0 ? std::string() : " AND inode=?2"));
	rows.Bind(1, point);
	if (inode >= 0)
		rows.Bind(2, inode);
	std::vector<Record> result;
	while (rows.Step()) {
		auto extent = ReadRecord(rows, BLOCKS);
		for (int64_t i = 0; i < extent.fields[1]; ++i) {
			auto block = extent;
			block.key.block += i;
			block.fields[1] = 1;
			block.fields[2] += i * BLOCK_SIZE;
			result.push_back(std::move(block));
		}
	}
	return result;
}

void PutExtent(sqlite3 *db, const Key &key, const std::vector<int64_t> &fields, const View &view, bool deleted) {
	std::vector<Record> overlaps;
	{
		Statement rows(db, Select(BLOCKS) + " WHERE inode=?1 AND block<=?2 AND block>=max(0,?2-63) AND block+count>?2 "
		                                    "AND low<?3 AND high>?4");
		rows.Bind(1, key.inode);
		rows.Bind(2, key.block);
		rows.Bind(3, view.high);
		rows.Bind(4, view.point);
		while (rows.Step())
			overlaps.push_back(ReadRecord(rows, BLOCKS));
	}
	for (auto extent : overlaps) {
		Statement remove(db, "DELETE FROM block_versions WHERE inode=? AND block=? AND low=?");
		remove.Bind(1, key.inode);
		remove.Bind(2, extent.key.block);
		remove.Bind(3, extent.low);
		remove.Step();
		auto index = key.block - extent.key.block;
		if (index) {
			auto left = extent;
			left.fields[1] = index;
			Insert(db, BLOCKS, left);
		}
		if (index + 1 < extent.fields[1]) {
			auto right = extent;
			right.key.block = key.block + 1;
			right.fields[1] -= index + 1;
			right.fields[2] += (index + 1) * BLOCK_SIZE;
			Insert(db, BLOCKS, right);
		}
		extent.key = key;
		extent.fields[1] = 1;
		extent.fields[2] += index * BLOCK_SIZE;
		if (extent.low < view.point) {
			auto before = extent;
			before.high = view.point;
			Insert(db, BLOCKS, before);
		}
		if (view.high < extent.high) {
			extent.low = view.high;
			Insert(db, BLOCKS, extent);
		}
	}
	if (fields.size() != 1 && (fields.size() != 3 || fields[1] != 1))
		Fail(ErrorCode::Storage, "Scalar mutation has a non-scalar extent reference");
	Insert(db, BLOCKS, {key, view.point, view.high, view.writer, deleted, fields});
}

std::vector<int64_t> LiveExtentBlocks(sqlite3 *db) {
	// Count referenced slices, not entire original payload allocations. A
	// snapshot or branch can keep just one fragment after the others disappear.
	Statement rows(db, "SELECT p.data,b.count,b.payload_offset FROM block_versions b "
	                   "JOIN block_payloads p ON p.id=b.payload WHERE b.deleted=0 AND length(p.data)=8");
	std::vector<int64_t> live;
	while (rows.Step()) {
		auto count = rows.Integer(1), displacement = rows.Integer(2);
		if (count < 1 || count > 64 || displacement < 0 || displacement % BLOCK_SIZE ||
		    displacement > 64 * BLOCK_SIZE - count * BLOCK_SIZE)
			Fail(ErrorCode::Storage, "Invalid extent range during collection");
		for (int64_t i = 0; i < count; ++i)
			live.push_back(ExtentPhysical(rows.Bytes(0), displacement + i * BLOCK_SIZE));
	}
	return live;
}
