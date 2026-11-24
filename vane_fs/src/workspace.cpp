// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include "owner_lock.hpp"
#include "checkpoint.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>

#if SQLITE_VERSION_NUMBER < 3051003
#error VaneFS requires SQLite 3.51.3 or later with the WAL-reset fix
#endif

namespace vane_fs {
namespace {
constexpr int64_t BLOCK_SIZE = 4096;
constexpr int WAL_CHECKPOINT_PAGES = 4096;
constexpr int APPLICATION_ID = 0x56465331;

[[noreturn]] void Fail(ErrorCode code, const std::string &message) {
	throw Error(code, message);
}

void Check(sqlite3 *db, int code) {
	if (code == SQLITE_OK || code == SQLITE_ROW || code == SQLITE_DONE) {
		return;
	}
	auto primary = code & 0xff;
	Fail(primary == SQLITE_BUSY || primary == SQLITE_LOCKED ? ErrorCode::Busy : ErrorCode::Storage, sqlite3_errmsg(db));
}

void Exec(sqlite3 *db, const std::string &sql) {
	Check(db, sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr));
}

// Fixed-width unsigned big-endian coordinates have the same ordering as SQLite BLOBs.
struct Coordinate {
	std::array<unsigned char, 32> bytes {};
	static Coordinate Maximum() {
		Coordinate value;
		value.bytes.fill(255);
		return value;
	}
	bool operator<(const Coordinate &other) const {
		return bytes < other.bytes;
	}
	bool operator==(const Coordinate &other) const {
		return bytes == other.bytes;
	}
	Coordinate Add(const Coordinate &other) const {
		Coordinate result;
		unsigned carry = 0;
		for (int i = 31; i >= 0; --i) {
			unsigned sum = bytes[i] + other.bytes[i] + carry;
			result.bytes[i] = sum & 255;
			carry = sum >> 8;
		}
		if (carry) {
			Fail(ErrorCode::Capacity, "Interval coordinate overflow");
		}
		return result;
	}
	Coordinate AddOne() const {
		Coordinate one;
		one.bytes[31] = 1;
		return Add(one);
	}
	Coordinate Subtract(const Coordinate &other) const {
		if (*this < other) {
			Fail(ErrorCode::Capacity, "Interval coordinate underflow");
		}
		Coordinate result;
		int borrow = 0;
		for (int i = 31; i >= 0; --i) {
			int difference = int(bytes[i]) - int(other.bytes[i]) - borrow;
			borrow = difference < 0;
			result.bytes[i] = static_cast<unsigned char>(difference);
		}
		return result;
	}
	Coordinate Divide(unsigned divisor) const {
		Coordinate result;
		unsigned remainder = 0;
		for (size_t i = 0; i < bytes.size(); ++i) {
			unsigned current = remainder * 256 + bytes[i];
			result.bytes[i] = current / divisor;
			remainder = current % divisor;
		}
		return result;
	}
};

// Access is serialized by Database::mutex. A slot holds at most one idle
// statement; nested uses of the same SQL prepare their own active statement.
struct StatementCache {
	std::unordered_map<std::string, sqlite3_stmt *> idle;
	~StatementCache() {
		Clear();
	}
	void Clear() {
		for (auto &entry : idle) {
			sqlite3_finalize(entry.second);
		}
		idle.clear();
	}
};
constexpr const char *STATEMENT_CACHE = "vane_fs.statement_cache";

class Statement {
public:
	Statement(sqlite3 *db, const std::string &sql) : db(db) {
		if (auto cache = static_cast<StatementCache *>(sqlite3_get_clientdata(db, STATEMENT_CACHE))) {
			// References to unordered_map elements survive rehashing.
			slot = &cache->idle[sql];
			statement = std::exchange(*slot, nullptr);
		}
		if (!statement) {
			Check(db, sqlite3_prepare_v2(db, sql.c_str(), -1, &statement, nullptr));
		}
	}
	~Statement() {
		if (slot && !*slot && sqlite3_reset(statement) == SQLITE_OK && sqlite3_clear_bindings(statement) == SQLITE_OK) {
			*slot = statement;
		} else {
			sqlite3_finalize(statement);
		}
	}
	Statement(const Statement &) = delete;
	Statement &operator=(const Statement &) = delete;
	void Bind(int index, int64_t value) {
		Check(db, sqlite3_bind_int64(statement, index, value));
	}
	void Bind(int index, const std::string &value) {
		Check(db, sqlite3_bind_text64(statement, index, value.data(), value.size(), SQLITE_TRANSIENT, SQLITE_UTF8));
	}
	void Bind(int index, const Coordinate &value) {
		Check(db, sqlite3_bind_blob(statement, index, value.bytes.data(), 32, SQLITE_TRANSIENT));
	}
	void BindBytes(int index, const std::string &value) {
		Check(db, sqlite3_bind_blob64(statement, index, value.data(), value.size(), SQLITE_TRANSIENT));
	}
	void BindNull(int index) {
		Check(db, sqlite3_bind_null(statement, index));
	}
	bool Step() {
		int code = sqlite3_step(statement);
		Check(db, code);
		return code == SQLITE_ROW;
	}
	int64_t Integer(int index) const {
		return sqlite3_column_int64(statement, index);
	}
	bool IsNull(int index) const {
		return sqlite3_column_type(statement, index) == SQLITE_NULL;
	}
	std::string Text(int index) const {
		auto value = sqlite3_column_text(statement, index);
		return value ? std::string(reinterpret_cast<const char *>(value), sqlite3_column_bytes(statement, index)) : "";
	}
	std::string Bytes(int index) const {
		auto value = BytesData(index);
		return value ? std::string(value, BytesSize(index)) : "";
	}
	const char *BytesData(int index) const {
		return static_cast<const char *>(sqlite3_column_blob(statement, index));
	}
	int BytesSize(int index) const {
		return sqlite3_column_bytes(statement, index);
	}
	Coordinate Point(int index) const {
		if (sqlite3_column_type(statement, index) != SQLITE_BLOB || sqlite3_column_bytes(statement, index) != 32) {
			Fail(ErrorCode::Storage, "Invalid interval coordinate in workspace");
		}
		Coordinate result;
		std::memcpy(result.bytes.data(), sqlite3_column_blob(statement, index), 32);
		return result;
	}

private:
	sqlite3 *db;
	sqlite3_stmt *statement = nullptr;
	sqlite3_stmt **slot = nullptr;
};

std::string NewId() {
	std::array<unsigned char, 16> bytes;
	// SQLite's userspace PRNG state is copied by fork. Fresh system entropy
	// prevents parent/child connections from generating identical durable IDs.
	std::random_device entropy;
	for (auto &byte : bytes)
		byte = static_cast<unsigned char>(entropy());
	std::string result;
	for (auto byte : bytes) {
		result += "0123456789abcdef"[byte >> 4];
		result += "0123456789abcdef"[byte & 15];
	}
	return result;
}

int64_t Now() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
	    .count();
}

struct View {
	Coordinate low, point, high;
	std::string branch, writer;
	BranchInfo info;
};

enum Table { INODES, DIRENTS, BLOCKS };
const char *TableName(Table table) {
	return table == INODES ? "inode_versions" : table == DIRENTS ? "dirent_versions" : "block_versions";
}
std::string KeyColumns(Table table) {
	return table == INODES ? "inode" : table == DIRENTS ? "parent, name" : "inode, block";
}
std::string FieldColumns(Table table) {
	return table == INODES ? "kind, size, mode, mtime_ns" : table == DIRENTS ? "inode" : "payload";
}
std::string KeyPredicate(Table table) {
	return table == INODES ? "inode=?" : table == DIRENTS ? "parent=? AND name=?" : "inode=? AND block=?";
}
int KeyCount(Table table) {
	return table == INODES ? 1 : 2;
}
int FieldCount(Table table) {
	return table == INODES ? 4 : 1;
}

struct Key {
	int64_t inode = 0;
	std::string name {};
	int64_t block = 0;
	bool operator<(const Key &other) const {
		return std::tie(inode, name, block) < std::tie(other.inode, other.name, other.block);
	}
	int Bind(Statement &statement, Table table, int index = 1) const {
		statement.Bind(index++, inode);
		if (table == DIRENTS) {
			statement.Bind(index++, name);
		}
		if (table == BLOCKS) {
			statement.Bind(index++, block);
		}
		return index;
	}
};

struct Record {
	Key key;
	Coordinate low, high;
	std::string writer;
	bool deleted = false;
	std::vector<int64_t> fields;
};

Record ReadRecord(Statement &statement, Table table) {
	Record record;
	int column = 0;
	record.key.inode = statement.Integer(column++);
	if (table == DIRENTS) {
		record.key.name = statement.Text(column++);
	}
	if (table == BLOCKS) {
		record.key.block = statement.Integer(column++);
	}
	record.low = statement.Point(column++);
	record.high = statement.Point(column++);
	record.writer = statement.Text(column++);
	record.deleted = statement.Integer(column++);
	for (int i = 0; i < FieldCount(table); ++i) {
		record.fields.push_back(statement.Integer(column++));
	}
	return record;
}

std::string Select(Table table) {
	return "SELECT " + KeyColumns(table) + ", low, high, writer, deleted, " + FieldColumns(table) + " FROM " +
	       TableName(table);
}

bool Get(sqlite3 *db, Table table, const Key &key, const Coordinate &point, Record &record) {
	Statement statement(db, Select(table) + " WHERE " + KeyPredicate(table) + " AND low<=? AND high>? AND deleted=0");
	int index = key.Bind(statement, table);
	statement.Bind(index++, point);
	statement.Bind(index, point);
	if (!statement.Step()) {
		return false;
	}
	record = ReadRecord(statement, table);
	if (statement.Step()) {
		Fail(ErrorCode::Storage, "Overlapping visible versions");
	}
	return true;
}

std::vector<Record> Visible(sqlite3 *db, Table table, const Coordinate &point, int64_t inode = -1) {
	std::string predicate = " WHERE low<=? AND high>? AND deleted=0";
	if (inode >= 0) {
		predicate += table == DIRENTS ? " AND parent=?" : " AND inode=?";
	}
	Statement statement(db, Select(table) + predicate);
	statement.Bind(1, point);
	statement.Bind(2, point);
	if (inode >= 0) {
		statement.Bind(3, inode);
	}
	std::vector<Record> records;
	while (statement.Step()) {
		records.push_back(ReadRecord(statement, table));
	}
	return records;
}

void Insert(sqlite3 *db, Table table, const Record &record) {
	std::string placeholders;
	for (int i = 0; i < KeyCount(table) + 4 + FieldCount(table); ++i) {
		if (i) {
			placeholders += ',';
		}
		placeholders += '?';
	}
	Statement statement(db, "INSERT INTO " + std::string(TableName(table)) + " (" + KeyColumns(table) +
	                            ", low, high, writer, deleted, " + FieldColumns(table) + ") VALUES (" + placeholders +
	                            ")");
	int index = record.key.Bind(statement, table);
	statement.Bind(index++, record.low);
	statement.Bind(index++, record.high);
	statement.Bind(index++, record.writer);
	statement.Bind(index++, int64_t(record.deleted));
	for (auto field : record.fields) {
		if (table == BLOCKS && record.deleted) {
			statement.BindNull(index++);
		} else {
			statement.Bind(index++, field);
		}
	}
	statement.Step();
}

void Put(sqlite3 *db, Table table, const Key &key, const std::vector<int64_t> &fields, const View &view,
         bool deleted = false) {
	std::vector<Record> overlaps;
	{
		Statement statement(db, Select(table) + " WHERE " + KeyPredicate(table) + " AND low<? AND high>?");
		int index = key.Bind(statement, table);
		statement.Bind(index++, view.high);
		statement.Bind(index, view.point);
		while (statement.Step()) {
			overlaps.push_back(ReadRecord(statement, table));
		}
	}
	for (auto record : overlaps) {
		Statement remove(db, "DELETE FROM " + std::string(TableName(table)) + " WHERE " + KeyPredicate(table) +
		                         " AND low=?");
		int index = key.Bind(remove, table);
		remove.Bind(index, record.low);
		remove.Step();
		if (record.low < view.point) {
			auto left = record;
			left.high = view.point;
			Insert(db, table, left);
		}
		if (view.high < record.high) {
			record.low = view.high;
			Insert(db, table, record);
		}
	}
	Insert(db, table, {key, view.point, view.high, view.writer, deleted, fields});
}

void Changed(sqlite3 *db, const View &view) {
	Statement statement(db, "UPDATE branches SET generation=generation+1 WHERE id=?");
	statement.Bind(1, view.branch);
	statement.Step();
}

std::vector<std::string> Parts(const std::string &path) {
	if (path.find('\0') != std::string::npos || path.size() > 4096) {
		Fail(ErrorCode::Invalid, "Invalid virtual path");
	}
	std::vector<std::string> parts;
	size_t begin = 0;
	while (begin < path.size()) {
		auto end = path.find('/', begin);
		if (end == std::string::npos) {
			end = path.size();
		}
		auto part = path.substr(begin, end - begin);
		begin = end + 1;
		if (part == "..") {
			Fail(ErrorCode::Invalid, "Parent traversal is not supported");
		}
		if (part.size() > 255) {
			Fail(ErrorCode::Invalid, "Directory entry name exceeds 255 bytes");
		}
		if (!part.empty() && part != ".") {
			parts.push_back(std::move(part));
		}
	}
	return parts;
}

Record Inode(sqlite3 *db, int64_t inode, const View &view) {
	Record record;
	if (!Get(db, INODES, {inode}, view.point, record)) {
		Fail(ErrorCode::NotFound, "Inode does not exist");
	}
	return record;
}

Record Resolve(sqlite3 *db, const std::vector<std::string> &parts, const View &view) {
	auto node = Inode(db, 1, view);
	for (const auto &part : parts) {
		if (!node.fields[0]) {
			Fail(ErrorCode::NotDirectory, "Path component is not a directory");
		}
		Record entry;
		if (!Get(db, DIRENTS, {node.key.inode, part}, view.point, entry)) {
			Fail(ErrorCode::NotFound, "Path does not exist: " + part);
		}
		node = Inode(db, entry.fields[0], view);
	}
	return node;
}

Record Resolve(sqlite3 *db, const std::string &path, const View &view) {
	auto node = Resolve(db, Parts(path), view);
	if (!path.empty() && path.back() == '/' && !node.fields[0]) {
		Fail(ErrorCode::NotDirectory, "Trailing slash requires a directory");
	}
	return node;
}

Key Parent(sqlite3 *db, const std::string &path, const View &view) {
	auto parts = Parts(path);
	if (parts.empty()) {
		Fail(ErrorCode::Invalid, "Operation cannot replace the root directory");
	}
	auto name = parts.back();
	parts.pop_back();
	auto parent = Resolve(db, parts, view);
	if (!parent.fields[0]) {
		Fail(ErrorCode::NotDirectory, "Parent is not a directory");
	}
	return {parent.key.inode, name};
}

void Touch(sqlite3 *db, int64_t inode, const View &view) {
	auto node = Inode(db, inode, view);
	node.fields[3] = Now();
	Put(db, INODES, node.key, node.fields, view);
}

std::string Payload(sqlite3 *db, int64_t id) {
	Statement statement(db, "SELECT data FROM block_payloads WHERE id=?");
	statement.Bind(1, id);
	if (!statement.Step()) {
		Fail(ErrorCode::Storage, "Missing block payload");
	}
	return statement.Bytes(0);
}

std::string Block(sqlite3 *db, int64_t inode, int64_t block, const View &view) {
	Record record;
	return Get(db, BLOCKS, {inode, {}, block}, view.point, record) ? Payload(db, record.fields[0])
	                                                               : std::string(BLOCK_SIZE, '\0');
}

std::string ReadBytes(sqlite3 *db, int64_t inode, int64_t offset, int64_t length, const View &view) {
	std::string result(size_t(length), '\0');
	if (!length) {
		return result;
	}
	// Keep absent blocks as zeroes. A LEFT JOIN must expose a missing payload
	// as corruption, rather than silently turning a dangling reference into a hole.
	Statement blocks(db, "SELECT b.block,p.data FROM block_versions b "
	                     "LEFT JOIN block_payloads p ON p.id=b.payload "
	                     "WHERE b.inode=? AND b.block>=? AND b.block<=? "
	                     "AND b.low<=? AND b.high>? AND b.deleted=0 ORDER BY b.block");
	blocks.Bind(1, inode);
	blocks.Bind(2, offset / BLOCK_SIZE);
	blocks.Bind(3, (offset + length - 1) / BLOCK_SIZE);
	blocks.Bind(4, view.point);
	blocks.Bind(5, view.point);
	int64_t previous = -1;
	while (blocks.Step()) {
		auto block = blocks.Integer(0);
		if (block == previous) {
			Fail(ErrorCode::Storage, "Overlapping visible versions");
		}
		previous = block;
		auto bytes = blocks.BytesData(1);
		if (!bytes) {
			Fail(ErrorCode::Storage, "Missing block payload");
		}
		if (blocks.BytesSize(1) != BLOCK_SIZE) {
			Fail(ErrorCode::Storage, "Invalid block payload size");
		}
		auto start = std::max(offset, block * BLOCK_SIZE);
		auto source_offset = start - block * BLOCK_SIZE;
		auto target_offset = start - offset;
		auto amount = std::min(BLOCK_SIZE - source_offset, length - target_offset);
		std::memcpy(&result[size_t(target_offset)], bytes + source_offset, size_t(amount));
	}
	return result;
}

void StoreBlock(sqlite3 *db, int64_t inode, int64_t block, const std::string &data, const View &view) {
	Key key {inode, {}, block};
	Record old;
	bool exists = Get(db, BLOCKS, key, view.point, old);
	if (std::all_of(data.begin(), data.end(), [](char byte) { return byte == 0; })) {
		if (exists) {
			Put(db, BLOCKS, key, {0}, view, true);
		}
		return;
	}
	if (exists && Payload(db, old.fields[0]) == data) {
		return;
	}
	Statement statement(db, "INSERT INTO block_payloads(data) VALUES(?)");
	statement.BindBytes(1, data);
	statement.Step();
	Put(db, BLOCKS, key, {sqlite3_last_insert_rowid(db)}, view);
}

void InsertNewBlocks(sqlite3 *db, int64_t inode, std::vector<std::pair<int64_t, std::string>> &blocks,
                     int64_t &last_payload, const View &view) {
	if (blocks.empty()) {
		return;
	}
	std::string payload_sql = "INSERT INTO block_payloads(id,data) VALUES ";
	std::string version_sql = "INSERT INTO block_versions(inode,block,low,high,writer,deleted,payload) VALUES ";
	for (size_t i = 0; i < blocks.size(); ++i) {
		if (i) {
			payload_sql += ',';
			version_sql += ',';
		}
		payload_sql += "(?,?)";
		version_sql += "(?1,?" + std::to_string(5 + 2 * i) + ",?2,?3,?4,0,?" + std::to_string(6 + 2 * i) + ")";
	}
	Statement payloads(db, payload_sql), versions(db, version_sql);
	versions.Bind(1, inode);
	versions.Bind(2, view.point);
	versions.Bind(3, view.high);
	versions.Bind(4, view.writer);
	for (size_t i = 0; i < blocks.size(); ++i) {
		auto payload = ++last_payload;
		payloads.Bind(int(1 + 2 * i), payload);
		payloads.BindBytes(int(2 + 2 * i), blocks[i].second);
		versions.Bind(int(5 + 2 * i), blocks[i].first);
		versions.Bind(int(6 + 2 * i), payload);
	}
	payloads.Step();
	versions.Step();
	blocks.clear();
}

void RequireFile(const Record &node) {
	if (node.fields[0]) {
		Fail(ErrorCode::IsDirectory, "Path is a directory");
	}
}

void Resize(sqlite3 *db, Record &node, int64_t size, const View &view) {
	RequireFile(node);
	if (size < 0) {
		Fail(ErrorCode::Invalid, "Negative file size");
	}
	if (size < node.fields[1]) {
		for (const auto &block : Visible(db, BLOCKS, view.point, node.key.inode)) {
			if (size == 0 || block.key.block > (size - 1) / BLOCK_SIZE) {
				Put(db, BLOCKS, block.key, {0}, view, true);
			}
		}
		if (size % BLOCK_SIZE) {
			auto bytes = Block(db, node.key.inode, size / BLOCK_SIZE, view);
			std::fill(bytes.begin() + size % BLOCK_SIZE, bytes.end(), '\0');
			StoreBlock(db, node.key.inode, size / BLOCK_SIZE, bytes, view);
		}
	}
	node.fields[1] = size;
	node.fields[3] = Now();
	Put(db, INODES, node.key, node.fields, view);
}

void WriteBytes(sqlite3 *db, Record &node, const std::string &data, int64_t offset, const View &view) {
	RequireFile(node);
	if (offset < 0 || data.size() > uint64_t(std::numeric_limits<int64_t>::max() - offset)) {
		Fail(ErrorCode::Invalid, "File range overflow");
	}
	if (data.empty()) {
		return;
	}
	bool vacant = false;
	int64_t last_payload = 0;
	if (data.size() >= 2 * BLOCK_SIZE) {
		// Check the entire replacement interval, including tombstones and
		// versions beyond the current view point, before bypassing Put.
		Statement occupied(db, "SELECT 1 FROM block_versions WHERE inode=? AND block>=? AND block<=? "
		                       "AND low<? AND high>? LIMIT 1");
		auto first = offset / BLOCK_SIZE;
		auto last = (offset + int64_t(data.size()) - 1) / BLOCK_SIZE;
		occupied.Bind(1, node.key.inode);
		occupied.Bind(2, first);
		occupied.Bind(3, last);
		occupied.Bind(4, view.high);
		occupied.Bind(5, view.point);
		vacant = !occupied.Step();
		if (vacant) {
			// The enclosing IMMEDIATE transaction serializes payload allocation.
			// Explicit IDs keep batches mapped without relying on RETURNING order.
			Statement maximum(db, "SELECT coalesce(max(id),0) FROM block_payloads");
			maximum.Step();
			last_payload = std::max<int64_t>(0, maximum.Integer(0));
			// Preserve SQLite's automatic rowid allocation at the integer limit.
			vacant = last - first + 1 <= std::numeric_limits<int64_t>::max() - last_payload;
		}
	}
	constexpr size_t BATCH_BLOCKS = 64;
	std::vector<std::pair<int64_t, std::string>> new_blocks;
	if (vacant) {
		new_blocks.reserve(BATCH_BLOCKS);
	}
	size_t consumed = 0;
	while (consumed < data.size()) {
		int64_t position = offset + int64_t(consumed);
		size_t amount = std::min<size_t>(BLOCK_SIZE - position % BLOCK_SIZE, data.size() - consumed);
		std::string bytes;
		if (amount == BLOCK_SIZE) {
			bytes.assign(data.data() + consumed, amount);
		} else {
			bytes = vacant ? std::string(BLOCK_SIZE, 0) : Block(db, node.key.inode, position / BLOCK_SIZE, view);
			bytes.replace(position % BLOCK_SIZE, amount, data.data() + consumed, amount);
		}
		if (vacant) {
			if (!std::all_of(bytes.begin(), bytes.end(), [](char byte) { return byte == 0; })) {
				new_blocks.emplace_back(position / BLOCK_SIZE, std::move(bytes));
				if (new_blocks.size() == BATCH_BLOCKS) {
					InsertNewBlocks(db, node.key.inode, new_blocks, last_payload, view);
				}
			}
		} else {
			StoreBlock(db, node.key.inode, position / BLOCK_SIZE, bytes, view);
		}
		consumed += amount;
	}
	InsertNewBlocks(db, node.key.inode, new_blocks, last_payload, view);
	node.fields[1] = std::max<int64_t>(node.fields[1], offset + data.size());
	node.fields[3] = Now();
	Put(db, INODES, node.key, node.fields, view);
}

Record Create(sqlite3 *db, const Key &entry, bool directory, int64_t mode, const View &view) {
	if (mode < 0 || mode > 0777) {
		Fail(ErrorCode::Invalid, "Mode must be between 0 and 0777");
	}
	Record existing;
	if (Get(db, DIRENTS, entry, view.point, existing)) {
		Fail(ErrorCode::Exists, "Path already exists");
	}
	Exec(db, "INSERT INTO inode_ids DEFAULT VALUES");
	Key key {sqlite3_last_insert_rowid(db)};
	std::vector<int64_t> fields {directory, 0, mode, Now()};
	Put(db, INODES, key, fields, view);
	Put(db, DIRENTS, entry, {key.inode}, view);
	Touch(db, entry.inode, view);
	return Inode(db, key.inode, view);
}

void Erase(sqlite3 *db, const Key &entry, const Record &node, const View &view) {
	if (node.fields[0] && !Visible(db, DIRENTS, view.point, node.key.inode).empty()) {
		Fail(ErrorCode::NotEmpty, "Directory is not empty");
	}
	Statement opened(db, "SELECT 1 FROM open_inodes WHERE branch=? AND inode=?");
	opened.Bind(1, view.branch);
	opened.Bind(2, node.key.inode);
	if (opened.Step()) {
		Statement orphan(db, "INSERT INTO orphans VALUES(?,?)");
		orphan.Bind(1, view.branch);
		orphan.Bind(2, node.key.inode);
		orphan.Step();
	} else {
		for (const auto &block : Visible(db, BLOCKS, view.point, node.key.inode)) {
			Put(db, BLOCKS, block.key, {0}, view, true);
		}
		Put(db, INODES, node.key, node.fields, view, true);
	}
	Put(db, DIRENTS, entry, {node.key.inode}, view, true);
	Touch(db, entry.inode, view);
}
} // namespace

struct SnapshotPin {
	std::string id = NewId();
	std::string path, workspace, owner;
	int64_t process = OwnerLock::Process();
};

namespace {
// Register before publishing a pin. If a destructor cannot delete its SQLite
// row, the registry keeps the token without allocating during error cleanup.
// A token owned only by this registry no longer protects a live session.
struct PinRegistry {
	std::mutex mutex;
	std::vector<std::shared_ptr<SnapshotPin>> pins;

	void Forget(const std::shared_ptr<SnapshotPin> &pin) {
		std::lock_guard<std::mutex> lock(mutex);
		pins.erase(std::remove(pins.begin(), pins.end(), pin), pins.end());
	}
};

PinRegistry &Pins() {
	static PinRegistry registry;
	return registry;
}
} // namespace

class Database {
public:
	sqlite3 *db = nullptr;
	StatementCache statements;
	std::mutex mutex;
	std::string uuid, owner = NewId();
	OwnerLock owner_lock;
	int64_t process = OwnerLock::Process();
	const Durability durability;
	std::unique_ptr<Checkpointer> checkpointer;

	explicit Database(const std::string &path, int timeout_ms, Durability durability) : durability(durability) {
		if (path.empty() || path == ":memory:" || path.find('\0') != std::string::npos || timeout_ms < 0) {
			Fail(ErrorCode::Invalid, "A durable database path and nonnegative timeout are required");
		}
		if (sqlite3_libversion_number() < 3051003) {
			Fail(ErrorCode::Storage, "SQLite 3.51.3 or later is required");
		}
		int code = sqlite3_open_v2(
		    path.c_str(), &db,
		    SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX | SQLITE_OPEN_PRIVATECACHE, nullptr);
		bool owner_committed = false;
		try {
			Check(db, code);
			Check(db, sqlite3_busy_timeout(db, timeout_ms));
			Check(db, sqlite3_extended_result_codes(db, 1));
			Check(db, sqlite3_set_clientdata(db, STATEMENT_CACHE, &statements, nullptr));
			Check(db, sqlite3_create_function_v2(
			              db, "vane_fs_owner", 0, SQLITE_UTF8, &owner,
			              [](sqlite3_context *context, int, sqlite3_value **) {
				              auto &id = *static_cast<std::string *>(sqlite3_user_data(context));
				              sqlite3_result_text(context, id.c_str(), -1, SQLITE_TRANSIENT);
			              },
			              nullptr, nullptr, nullptr));
			ValidateIdentity();
			Exec(db, "PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL; PRAGMA temp_store=MEMORY");
			{
				Statement mode(db, "PRAGMA journal_mode=WAL");
				if (!mode.Step() || mode.Text(0) != "wal") {
					Fail(ErrorCode::Storage, "Could not enable SQLite WAL");
				}
			}
			// Amortize checkpoint syncs over 16 MiB with the default 4 KiB pages.
			// Strict mode also synchronizes every WAL commit before returning.
			Check(db, sqlite3_wal_autocheckpoint(db, WAL_CHECKPOINT_PAGES));
			Exec(db, "BEGIN IMMEDIATE");
			ValidateIdentity();
			Initialize();
			Exec(db, R"SQL(
CREATE TABLE IF NOT EXISTS owners(id TEXT PRIMARY KEY, lock_path TEXT NOT NULL, device INTEGER NOT NULL, inode INTEGER NOT NULL) STRICT;
CREATE TABLE IF NOT EXISTS mounts(branch TEXT PRIMARY KEY REFERENCES branches(id), owner TEXT NOT NULL REFERENCES owners(id)) STRICT;
CREATE TABLE IF NOT EXISTS open_inodes(owner TEXT NOT NULL REFERENCES owners(id), branch TEXT NOT NULL REFERENCES branches(id),
 inode INTEGER NOT NULL REFERENCES inode_ids(id), refs INTEGER NOT NULL CHECK(refs>0), PRIMARY KEY(owner,branch,inode)) STRICT;
CREATE INDEX IF NOT EXISTS opened_inodes ON open_inodes(branch,inode);
CREATE TEMP TABLE inode_references(branch TEXT NOT NULL, inode INTEGER NOT NULL,
 refs INTEGER NOT NULL CHECK(refs>0), PRIMARY KEY(branch,inode)) STRICT, WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS orphans(branch TEXT NOT NULL REFERENCES branches(id), inode INTEGER NOT NULL REFERENCES inode_ids(id),
 PRIMARY KEY(branch,inode)) STRICT;
CREATE INDEX IF NOT EXISTS dirent_targets ON dirent_versions(inode,low,high);
CREATE INDEX IF NOT EXISTS block_payload_references ON block_versions(payload);
CREATE TABLE IF NOT EXISTS sync_barrier(id INTEGER PRIMARY KEY CHECK(id=1), value INTEGER NOT NULL CHECK(value IN(0,1))) STRICT;
INSERT OR IGNORE INTO sync_barrier VALUES(1,0);
UPDATE format SET version=2;
)SQL");
			owner_lock.Create(sqlite3_db_filename(db, "main"), owner);
			{
				Statement insert(db, "INSERT INTO owners VALUES(?,?,?,?)");
				insert.Bind(1, owner);
				insert.Bind(2, owner_lock.path);
				insert.Bind(3, owner_lock.device);
				insert.Bind(4, owner_lock.inode);
				insert.Step();
			}
			Exec(db, "COMMIT");
			owner_committed = true;
			if (durability == Durability::Fsync) {
				Exec(db, "PRAGMA synchronous=NORMAL");
				// Reuse a modest WAL allocation after restart, including after a
				// single transaction temporarily exceeds the admission budget.
				Exec(db, "PRAGMA journal_size_limit=" + std::to_string(Checkpointer::START_BYTES));
				Statement pages(db, "PRAGMA page_size");
				pages.Step();
				checkpointer = std::make_unique<Checkpointer>(db, int(pages.Integer(0)), timeout_ms);
				sqlite3_wal_hook(db, Checkpointer::OnCommit, checkpointer.get());
			}
		} catch (...) {
			checkpointer.reset();
			if (db) {
				statements.Clear();
				sqlite3_close_v2(db);
				db = nullptr;
			}
			// Worker creation can fail after registration committed. Preserve
			// its unlocked file so RecoverOwners can retire that durable row.
			if (owner_committed)
				owner_lock.Close();
			else
				owner_lock.Remove();
			throw;
		}
	}
	~Database() {
		try {
			Close();
		} catch (...) {
			if (db)
				sqlite3_wal_hook(db, nullptr, nullptr);
			checkpointer.reset();
			if (db) {
				statements.Clear();
				sqlite3_close_v2(db);
			}
		}
	}
	void ValidateIdentity() {
		Statement app(db, "PRAGMA application_id");
		app.Step();
		if (app.Integer(0) == APPLICATION_ID) {
			return;
		}
		if (app.Integer(0) != 0) {
			Fail(ErrorCode::Invalid, "Not a VaneFS database");
		}
		Statement objects(db, "SELECT count(*) FROM sqlite_master WHERE name NOT GLOB 'sqlite_*'");
		objects.Step();
		if (objects.Integer(0)) {
			Fail(ErrorCode::Invalid, "Refusing to initialize an existing non-VaneFS database");
		}
	}
	void Initialize();
	void Close();
};

namespace {
class Transaction {
public:
	Transaction(Database &database, bool write, bool synchronous = false, bool barrier = false)
	    : database(database), lock(database.mutex, std::defer_lock),
	      restore_normal(write && synchronous && database.durability == Durability::Fsync) {
		if (database.process != OwnerLock::Process()) {
			Fail(ErrorCode::Closed, "Reopen the workspace after fork");
		}
		lock.lock();
		if (!database.db) {
			Fail(ErrorCode::Closed, "Workspace is closed");
		}
		if (write && database.checkpointer) {
			if (barrier)
				database.checkpointer->CheckError();
			else
				database.checkpointer->BeforeWrite(database.db);
		}
		// SQLite requires changing synchronous outside a transaction. On failure
		// leave FULL enabled until a subsequent successful synchronous commit.
		if (restore_normal)
			Exec(database.db, "PRAGMA synchronous=FULL");
		Exec(database.db, write ? "BEGIN IMMEDIATE" : "BEGIN");
		try {
			if (write) {
				{
					auto &registry = Pins();
					std::lock_guard<std::mutex> guard(registry.mutex);
					for (const auto &pin : registry.pins) {
						if (pin.use_count() == 1 && pin->process == database.process &&
						    pin->workspace == database.uuid && pin->path == sqlite3_db_filename(database.db, "main")) {
							retired_pins.push_back(pin);
						}
					}
				}
				for (const auto &pin : retired_pins) {
					Statement remove(database.db, "DELETE FROM pins WHERE id=? AND owner=?");
					remove.Bind(1, pin->id);
					remove.Bind(2, pin->owner);
					remove.Step();
				}
			}
		} catch (...) {
			sqlite3_exec(database.db, "ROLLBACK", nullptr, nullptr, nullptr);
			throw;
		}
	}
	~Transaction() {
		if (!committed) {
			sqlite3_exec(database.db, "ROLLBACK", nullptr, nullptr, nullptr);
		}
	}
	void Commit() {
		Exec(database.db, "COMMIT");
		committed = true;
		for (const auto &pin : retired_pins) {
			Pins().Forget(pin);
		}
		if (restore_normal)
			Exec(database.db, "PRAGMA synchronous=NORMAL");
	}

private:
	Database &database;
	std::unique_lock<std::mutex> lock;
	std::vector<std::shared_ptr<SnapshotPin>> retired_pins;
	bool committed = false;
	bool restore_normal;
};

void CheckMount(sqlite3 *db, const std::string &branch, bool require_unmounted = false) {
	Statement mount(db, "SELECT owner,owner=vane_fs_owner() FROM mounts WHERE branch=?");
	mount.Bind(1, branch);
	if (mount.Step() && (require_unmounted || !mount.Integer(1))) {
		Fail(ErrorCode::Busy, "Branch has an exclusive writable mount");
	}
}

View Branch(sqlite3 *db, const std::string &id, bool write = false, bool allow_name = false) {
	Statement statement(
	    db, "SELECT id,name,parent,fork_base,state,generation,low,frontier,high,writer FROM branches WHERE " +
	            std::string(allow_name ? "(id=? OR name=?)" : "id=?") + " AND state!='deleted'");
	statement.Bind(1, id);
	if (allow_name) {
		statement.Bind(2, id);
	}
	if (!statement.Step()) {
		Fail(ErrorCode::NotFound, "Branch does not exist: " + id);
	}
	View view;
	view.branch = statement.Text(0);
	view.info = {view.branch,       statement.Text(1), statement.Text(2),
	             statement.Text(3), statement.Text(4), statement.Integer(5)};
	view.low = statement.Point(6);
	view.point = statement.Point(7);
	view.high = statement.Point(8);
	view.writer = statement.Text(9);
	if (write && view.info.state != "writable") {
		Fail(ErrorCode::ReadOnly, "Branch is sealed");
	}
	if (write) {
		CheckMount(db, view.branch);
	}
	return view;
}

Coordinate SnapshotPoint(sqlite3 *db, const std::string &id) {
	Statement statement(db, "SELECT point FROM snapshots WHERE id=?");
	statement.Bind(1, id);
	if (!statement.Step()) {
		Fail(ErrorCode::NotFound, "Snapshot does not exist: " + id);
	}
	return statement.Point(0);
}

View SessionView(sqlite3 *db, const std::string &id, bool snapshot, bool closed, bool write = false) {
	if (closed) {
		Fail(ErrorCode::Closed, "Session is closed");
	}
	if (!snapshot) {
		return Branch(db, id, write);
	}
	if (write) {
		Fail(ErrorCode::ReadOnly, "Snapshot is read-only");
	}
	View view;
	view.point = SnapshotPoint(db, id);
	return view;
}

std::string SaveSnapshot(sqlite3 *db, const View &view, bool retained) {
	auto id = NewId();
	Statement statement(db, "INSERT INTO snapshots(id,point,retained) VALUES(?,?,?)");
	statement.Bind(1, id);
	statement.Bind(2, view.point);
	statement.Bind(3, int64_t(retained));
	statement.Step();
	return id;
}

void Advance(sqlite3 *db, const View &view, const Coordinate &point) {
	if (!(view.point < point) || !(point < view.high)) {
		Fail(ErrorCode::Capacity, "Branch interval is exhausted");
	}
	Statement statement(db, "UPDATE branches SET frontier=?,writer=?,generation=generation+1 WHERE id=?");
	statement.Bind(1, point);
	statement.Bind(2, NewId());
	statement.Bind(3, view.branch);
	statement.Step();
}
} // namespace

void Database::Initialize() {
	Statement app(db, "PRAGMA application_id");
	app.Step();
	if (app.Integer(0) == APPLICATION_ID) {
		Statement meta(db, "SELECT uuid,version,block_size,coordinate_bytes FROM format");
		if (!meta.Step() || (meta.Integer(1) != 1 && meta.Integer(1) != 2) || meta.Integer(2) != BLOCK_SIZE ||
		    meta.Integer(3) != 32) {
			Fail(ErrorCode::Storage, "Unsupported VaneFS format");
		}
		uuid = meta.Text(0);
		return;
	}
	Exec(db, R"SQL(
CREATE TABLE format(uuid TEXT NOT NULL, version INTEGER NOT NULL, block_size INTEGER NOT NULL, coordinate_bytes INTEGER NOT NULL) STRICT;
CREATE TABLE branches(id TEXT PRIMARY KEY, name TEXT NOT NULL, parent TEXT NOT NULL, fork_base TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('writable','sealed','deleted')), generation INTEGER NOT NULL,
 low BLOB NOT NULL CHECK(length(low)=32), frontier BLOB NOT NULL CHECK(length(frontier)=32), high BLOB NOT NULL CHECK(length(high)=32),
 writer TEXT NOT NULL, CHECK(low<=frontier AND frontier<high)) STRICT;
CREATE UNIQUE INDEX branch_names ON branches(name) WHERE state!='deleted';
CREATE INDEX branch_parents ON branches(parent);
CREATE TABLE snapshots(id TEXT PRIMARY KEY, point BLOB NOT NULL CHECK(length(point)=32), retained INTEGER NOT NULL) STRICT;
CREATE TABLE pins(id TEXT PRIMARY KEY, snapshot TEXT NOT NULL REFERENCES snapshots(id), owner TEXT NOT NULL) STRICT;
CREATE INDEX snapshot_pins ON pins(snapshot);
CREATE TABLE inode_ids(id INTEGER PRIMARY KEY AUTOINCREMENT) STRICT;
CREATE TABLE block_payloads(id INTEGER PRIMARY KEY, data BLOB NOT NULL CHECK(length(data)=4096)) STRICT;
)SQL");
	for (auto table : {INODES, DIRENTS, BLOCKS}) {
		std::string keys =
		    table == INODES ? "inode INTEGER NOT NULL REFERENCES inode_ids(id),"
		    : table == DIRENTS
		        ? "parent INTEGER NOT NULL REFERENCES inode_ids(id),name TEXT NOT NULL,"
		        : "inode INTEGER NOT NULL REFERENCES inode_ids(id),block INTEGER NOT NULL CHECK(block>=0),";
		std::string fields = table == INODES    ? "kind INTEGER NOT NULL CHECK(kind IN (0,1)),size INTEGER NOT NULL "
		                                          "CHECK(size>=0),mode INTEGER NOT NULL,mtime_ns INTEGER NOT NULL,"
		                     : table == DIRENTS ? "inode INTEGER NOT NULL REFERENCES inode_ids(id),"
		                                        : "payload INTEGER REFERENCES block_payloads(id),CHECK((deleted=1 AND "
		                                          "payload IS NULL) OR (deleted=0 AND payload IS NOT NULL)),";
		Exec(db, "CREATE TABLE " + std::string(TableName(table)) + "(" + keys +
		             "low BLOB NOT NULL CHECK(length(low)=32),high BLOB NOT NULL CHECK(length(high)=32),writer TEXT "
		             "NOT NULL,deleted INTEGER NOT NULL CHECK(deleted IN (0,1))," +
		             fields + "CHECK(low<high),PRIMARY KEY(" + KeyColumns(table) + ",low)) STRICT, WITHOUT ROWID");
		Exec(db, "CREATE INDEX " + std::string(TableName(table)) + "_high ON " + TableName(table) + "(" +
		             KeyColumns(table) + ",high)");
	}
	uuid = NewId();
	{
		Statement meta(db, "INSERT INTO format VALUES(?,1,4096,32)");
		meta.Bind(1, uuid);
		meta.Step();
	}
	View root;
	root.low = {};
	root.point = Coordinate().AddOne();
	root.high = Coordinate::Maximum();
	root.writer = NewId();
	root.branch = NewId();
	{
		Statement branch(db, "INSERT INTO branches VALUES(?,'main','','','writable',0,?,?,?,?)");
		branch.Bind(1, root.branch);
		branch.Bind(2, root.low);
		branch.Bind(3, root.point);
		branch.Bind(4, root.high);
		branch.Bind(5, root.writer);
		branch.Step();
	}
	Exec(db, "INSERT INTO inode_ids DEFAULT VALUES");
	Put(db, INODES, {1}, {1, 0, 0755, Now()}, root);
	Exec(db, "PRAGMA application_id=" + std::to_string(APPLICATION_ID));
}

Workspace::Workspace(const std::string &path, int timeout_ms, Durability durability)
    : database(std::make_shared<Database>(path, timeout_ms, durability)) {
}
Workspace::~Workspace() = default;
void Workspace::Close() {
	database->Close();
}
void Workspace::Sync() {
	// Even under WAL backpressure, callers must be able to persist writes
	// that have already committed. Only this small barrier bypasses admission.
	Transaction tx(*database, true, true, true);
	// An empty transaction need not write or synchronize the WAL. Changing
	// this page forces a FULL commit, covering every earlier WAL commit, even
	// when another connection used NORMAL and this connection uses Strict.
	Exec(database->db, "UPDATE sync_barrier SET value=1-value WHERE id=1");
	tx.Commit();
}
std::string Workspace::Id() const {
	return database->uuid;
}
std::string Workspace::SQLiteVersion() {
	return sqlite3_libversion();
}

Session::Session(std::shared_ptr<Database> database, std::string id, bool snapshot)
    : database(std::move(database)), id(std::move(id)), snapshot(snapshot) {
}
Session::~Session() {
	try {
		Close();
	} catch (...) {
		// Releasing our token leaves it in the registry for the next write
		// transaction on this database, including through another connection.
	}
}
void Session::Close() {
	if (database->process != OwnerLock::Process()) {
		closed = true;
		return;
	}
	std::lock_guard<std::mutex> lock(database->mutex);
	if (closed) {
		return;
	}
	if (database->db && pin) {
		if (database->checkpointer)
			database->checkpointer->BeforeWrite(database->db);
		Statement remove(database->db, "DELETE FROM pins WHERE id=?");
		remove.Bind(1, pin->id);
		remove.Step();
	}
	if (pin) {
		Pins().Forget(pin);
		pin.reset();
	}
	closed = true;
}

FileStat Session::Stat(const std::string &path) {
	Transaction tx(*database, false);
	auto view = SessionView(database->db, id, snapshot, closed);
	auto node = Resolve(database->db, path, view);
	tx.Commit();
	return {node.key.inode, bool(node.fields[0]), node.fields[1],
	        node.fields[2], node.fields[3],       node.fields[0] ? 2 : 1};
}

std::vector<std::string> Session::ListDirectory(const std::string &path) {
	Transaction tx(*database, false);
	auto view = SessionView(database->db, id, snapshot, closed);
	auto node = Resolve(database->db, path, view);
	if (!node.fields[0]) {
		Fail(ErrorCode::NotDirectory, "Path is not a directory");
	}
	std::vector<std::string> names;
	for (const auto &entry : Visible(database->db, DIRENTS, view.point, node.key.inode)) {
		names.push_back(entry.key.name);
	}
	std::sort(names.begin(), names.end());
	tx.Commit();
	return names;
}

std::string Session::Read(const std::string &path, int64_t offset, int64_t size) {
	if (offset < 0 || size < -1) {
		Fail(ErrorCode::Invalid, "Invalid read range");
	}
	Transaction tx(*database, false);
	auto view = SessionView(database->db, id, snapshot, closed);
	auto node = Resolve(database->db, path, view);
	RequireFile(node);
	int64_t available = std::max<int64_t>(0, node.fields[1] - offset);
	int64_t length = size < 0 ? available : std::min(size, available);
	auto result = ReadBytes(database->db, node.key.inode, offset, length, view);
	tx.Commit();
	return result;
}

void Session::MakeDirectory(const std::string &path, int64_t mode) {
	Transaction tx(*database, true);
	auto view = SessionView(database->db, id, snapshot, closed, true);
	Create(database->db, Parent(database->db, path, view), true, mode, view);
	Changed(database->db, view);
	tx.Commit();
}

void Session::WriteFile(const std::string &path, const std::string &data) {
	Transaction tx(*database, true);
	auto view = SessionView(database->db, id, snapshot, closed, true);
	if (!path.empty() && path.back() == '/') {
		Fail(ErrorCode::IsDirectory, "Cannot write a file with a trailing slash");
	}
	auto entry = Parent(database->db, path, view);
	Record record;
	auto node = Get(database->db, DIRENTS, entry, view.point, record) ? Inode(database->db, record.fields[0], view)
	                                                                  : Create(database->db, entry, false, 0644, view);
	WriteBytes(database->db, node, data, 0, view);
	Resize(database->db, node, int64_t(data.size()), view);
	Changed(database->db, view);
	tx.Commit();
}

void Session::Write(const std::string &path, const std::string &data, int64_t offset) {
	Transaction tx(*database, true);
	auto view = SessionView(database->db, id, snapshot, closed, true);
	auto node = Resolve(database->db, path, view);
	WriteBytes(database->db, node, data, offset, view);
	Changed(database->db, view);
	tx.Commit();
}

void Session::Truncate(const std::string &path, int64_t size) {
	Transaction tx(*database, true);
	auto view = SessionView(database->db, id, snapshot, closed, true);
	auto node = Resolve(database->db, path, view);
	Resize(database->db, node, size, view);
	Changed(database->db, view);
	tx.Commit();
}

void Session::Unlink(const std::string &path) {
	Transaction tx(*database, true);
	auto view = SessionView(database->db, id, snapshot, closed, true);
	auto node = Resolve(database->db, path, view);
	RequireFile(node);
	Erase(database->db, Parent(database->db, path, view), node, view);
	Changed(database->db, view);
	tx.Commit();
}

void Session::RemoveDirectory(const std::string &path) {
	Transaction tx(*database, true);
	auto view = SessionView(database->db, id, snapshot, closed, true);
	auto node = Resolve(database->db, path, view);
	if (!node.fields[0]) {
		Fail(ErrorCode::NotDirectory, "Path is not a directory");
	}
	Erase(database->db, Parent(database->db, path, view), node, view);
	Changed(database->db, view);
	tx.Commit();
}

void Session::Rename(const std::string &source, const std::string &target, bool no_replace) {
	Transaction tx(*database, true);
	auto view = SessionView(database->db, id, snapshot, closed, true);
	auto from = Parent(database->db, source, view);
	auto to = Parent(database->db, target, view);
	auto node = Resolve(database->db, source, view);
	if (!target.empty() && target.back() == '/' && !node.fields[0]) {
		Fail(ErrorCode::NotDirectory, "Trailing slash requires a directory");
	}
	if (from.inode == to.inode && from.name == to.name) {
		if (no_replace) {
			Fail(ErrorCode::Exists, "Rename destination exists");
		}
		tx.Commit();
		return;
	}
	auto parts = Parts(target);
	parts.pop_back();
	for (size_t i = 0; i <= parts.size(); ++i) {
		std::vector<std::string> prefix(parts.begin(), parts.begin() + i);
		if (Resolve(database->db, prefix, view).key.inode == node.key.inode) {
			Fail(ErrorCode::Invalid, "Cannot move a directory into itself");
		}
	}
	Record destination;
	if (Get(database->db, DIRENTS, to, view.point, destination)) {
		if (no_replace) {
			Fail(ErrorCode::Exists, "Rename destination exists");
		}
		auto replaced = Inode(database->db, destination.fields[0], view);
		if (node.fields[0] != replaced.fields[0]) {
			Fail(replaced.fields[0] ? ErrorCode::IsDirectory : ErrorCode::NotDirectory, "Rename type mismatch");
		}
		Erase(database->db, to, replaced, view);
	}
	Put(database->db, DIRENTS, from, {node.key.inode}, view, true);
	Put(database->db, DIRENTS, to, {node.key.inode}, view);
	Touch(database->db, from.inode, view);
	Touch(database->db, to.inode, view);
	Changed(database->db, view);
	tx.Commit();
}

BranchInfo Workspace::GetBranch(const std::string &branch) {
	Transaction tx(*database, false);
	auto result = Branch(database->db, branch, false, true).info;
	tx.Commit();
	return result;
}

std::vector<BranchInfo> Workspace::ListBranches() {
	Transaction tx(*database, false);
	Statement statement(database->db, "SELECT id FROM branches WHERE state!='deleted' ORDER BY name");
	std::vector<BranchInfo> result;
	while (statement.Step()) {
		result.push_back(Branch(database->db, statement.Text(0)).info);
	}
	tx.Commit();
	return result;
}

std::shared_ptr<Session> Workspace::Checkout(const std::string &branch) {
	std::string id;
	{
		Transaction tx(*database, false);
		id = Branch(database->db, branch, false, true).branch;
		tx.Commit();
	}
	return std::shared_ptr<Session>(new Session(database, id, false));
}

BranchInfo Workspace::Fork(const std::string &source, const std::string &name, bool terminal) {
	if (name.empty() || name.size() > 255 || name.find('\0') != std::string::npos ||
	    name.find('/') != std::string::npos) {
		Fail(ErrorCode::Invalid, "Invalid branch name");
	}
	Transaction tx(*database, true);
	auto view = Branch(database->db, source, true, true);
	CheckMount(database->db, view.branch, true);
	{
		Statement exists(database->db, "SELECT 1 FROM branches WHERE (name=? AND state!='deleted') OR id=?");
		exists.Bind(1, name);
		exists.Bind(2, name);
		if (exists.Step()) {
			Fail(ErrorCode::Exists, "Branch name already exists");
		}
	}
	auto one = Coordinate().AddOne();
	auto child_low = view.point.AddOne();
	if (!(child_low.AddOne() < view.high)) {
		Fail(ErrorCode::Capacity, "Branch interval is exhausted");
	}
	auto width = terminal ? one : view.high.Subtract(child_low.AddOne()).Divide(8);
	if (width < one) {
		width = one;
	}
	auto split = child_low.Add(width);
	auto base = SaveSnapshot(database->db, view, false);
	auto id = NewId();
	Statement insert(database->db, "INSERT INTO branches VALUES(?,?,?,?,'writable',0,?,?,?,?)");
	insert.Bind(1, id);
	insert.Bind(2, name);
	insert.Bind(3, view.branch);
	insert.Bind(4, base);
	insert.Bind(5, child_low);
	insert.Bind(6, child_low);
	insert.Bind(7, split);
	insert.Bind(8, NewId());
	insert.Step();
	Advance(database->db, view, split);
	auto result = Branch(database->db, id).info;
	tx.Commit();
	return result;
}

std::string Workspace::Snapshot(const std::string &branch) {
	Transaction tx(*database, true);
	auto view = Branch(database->db, branch, false, true);
	CheckMount(database->db, view.branch, true);
	auto id = SaveSnapshot(database->db, view, true);
	// A sealed branch cannot change, so retaining its frontier needs no allocation.
	if (view.info.state == "writable") {
		Advance(database->db, view, view.point.AddOne());
	}
	tx.Commit();
	return id;
}

std::shared_ptr<Session> Workspace::OpenSnapshot(const std::string &snapshot) {
	// Allocate the owner before locking so exception cleanup never re-enters
	// the database mutex while a failed transaction still holds it.
	auto result = std::shared_ptr<Session>(new Session(database, snapshot, true));
	Transaction tx(*database, true);
	SnapshotPoint(database->db, snapshot);
	auto pin = std::make_shared<SnapshotPin>();
	pin->path = sqlite3_db_filename(database->db, "main");
	pin->workspace = database->uuid;
	pin->owner = database->owner;
	{
		auto &registry = Pins();
		std::lock_guard<std::mutex> lock(registry.mutex);
		registry.pins.push_back(pin);
	}
	result->pin = pin;
	Statement insert(database->db, "INSERT INTO pins VALUES(?,?,?)");
	insert.Bind(1, pin->id);
	insert.Bind(2, snapshot);
	insert.Bind(3, database->owner);
	insert.Step();
	tx.Commit();
	return result;
}

void Workspace::DropSnapshot(const std::string &snapshot) {
	Transaction tx(*database, true);
	SnapshotPoint(database->db, snapshot);
	Statement used(
	    database->db,
	    "SELECT 1 FROM pins WHERE snapshot=? UNION ALL SELECT 1 FROM branches WHERE fork_base=? AND state!='deleted'");
	used.Bind(1, snapshot);
	used.Bind(2, snapshot);
	if (used.Step()) {
		Fail(ErrorCode::Busy, "Snapshot is pinned or retained as a fork base");
	}
	Statement remove(database->db, "DELETE FROM snapshots WHERE id=?");
	remove.Bind(1, snapshot);
	remove.Step();
	tx.Commit();
}

// Diff, merge and collection operate below on the same native record layer.
namespace {
struct TreeEntry {
	Record node;
	std::map<int64_t, int64_t> blocks;
	int64_t parent;
};
using Tree = std::map<std::string, TreeEntry>;

std::string ParentPath(const std::string &path) {
	auto slash = path.rfind('/');
	return slash == 0 ? "/" : path.substr(0, slash);
}

Tree ReadTree(sqlite3 *db, const Coordinate &point) {
	std::map<int64_t, Record> nodes;
	for (auto &record : Visible(db, INODES, point)) {
		if (!nodes.emplace(record.key.inode, record).second) {
			Fail(ErrorCode::Storage, "Duplicate visible inode");
		}
	}
	if (!nodes.count(1) || !nodes.at(1).fields[0]) {
		Fail(ErrorCode::Storage, "Missing workspace root");
	}
	std::map<int64_t, std::vector<Record>> children;
	for (auto &record : Visible(db, DIRENTS, point)) {
		children[record.key.inode].push_back(record);
	}
	Tree tree;
	std::map<int64_t, std::string> paths;
	std::vector<std::tuple<std::string, int64_t, int64_t>> queue {{"/", 1, 0}};
	for (size_t i = 0; i < queue.size(); ++i) {
		auto path = std::get<0>(queue[i]);
		auto id = std::get<1>(queue[i]);
		if (!nodes.count(id) || !paths.emplace(id, path).second) {
			Fail(ErrorCode::Storage, "Invalid inode reference, cycle or hard link");
		}
		auto node = nodes.at(id);
		if (!tree.emplace(path, TreeEntry {node, {}, std::get<2>(queue[i])}).second) {
			Fail(ErrorCode::Storage, "Duplicate directory entry");
		}
		if (!children[id].empty() && !node.fields[0]) {
			Fail(ErrorCode::Storage, "File has directory entries");
		}
		for (const auto &entry : children[id]) {
			if (entry.key.name.empty() || entry.key.name == "." || entry.key.name == ".." ||
			    entry.key.name.find('/') != std::string::npos) {
				Fail(ErrorCode::Storage, "Invalid directory entry");
			}
			queue.emplace_back(path == "/" ? path + entry.key.name : path + "/" + entry.key.name, entry.fields[0], id);
		}
	}
	if (paths.size() != nodes.size()) {
		Fail(ErrorCode::Storage, "Unreachable visible inode");
	}
	for (const auto &entries : children) {
		if (!entries.second.empty() && !paths.count(entries.first)) {
			Fail(ErrorCode::Storage, "Unreachable directory entries");
		}
	}
	for (const auto &block : Visible(db, BLOCKS, point)) {
		if (!paths.count(block.key.inode)) {
			Fail(ErrorCode::Storage, "Block has no visible inode");
		}
		auto &entry = tree.at(paths.at(block.key.inode));
		if (entry.node.fields[0] || !entry.node.fields[1] ||
		    block.key.block > (entry.node.fields[1] - 1) / BLOCK_SIZE) {
			Fail(ErrorCode::Storage, "Block lies outside file size");
		}
		if (!entry.blocks.emplace(block.key.block, block.fields[0]).second) {
			Fail(ErrorCode::Storage, "Duplicate visible block");
		}
	}
	return tree;
}

const TreeEntry *Entry(const Tree &tree, const std::string &path) {
	auto entry = tree.find(path);
	return entry == tree.end() ? nullptr : &entry->second;
}

bool Equal(sqlite3 *db, const TreeEntry *left, const TreeEntry *right) {
	if (!left || !right) {
		return left == right;
	}
	if (left->node.key.inode != right->node.key.inode || left->parent != right->parent) {
		return false;
	}
	// Modification times describe an operation, not a semantic merge conflict.
	for (size_t i = 0; i < 3; ++i) {
		if (left->node.fields[i] != right->node.fields[i]) {
			return false;
		}
	}
	std::set<int64_t> blocks;
	for (const auto &block : left->blocks) {
		blocks.insert(block.first);
	}
	for (const auto &block : right->blocks) {
		blocks.insert(block.first);
	}
	for (auto block : blocks) {
		auto a = left->blocks.find(block), b = right->blocks.find(block);
		int64_t aid = a == left->blocks.end() ? 0 : a->second, bid = b == right->blocks.end() ? 0 : b->second;
		if (aid == bid) {
			continue;
		}
		if ((aid ? Payload(db, aid) : std::string(BLOCK_SIZE, '\0')) !=
		    (bid ? Payload(db, bid) : std::string(BLOCK_SIZE, '\0'))) {
			return false;
		}
	}
	return true;
}

std::set<std::string> Paths(const Tree &a, const Tree &b) {
	std::set<std::string> paths;
	for (const auto &entry : a) {
		paths.insert(entry.first);
	}
	for (const auto &entry : b) {
		paths.insert(entry.first);
	}
	return paths;
}

std::vector<Change> Compare(sqlite3 *db, const Tree &source, const Tree &target) {
	std::vector<Change> changes;
	for (const auto &path : Paths(source, target)) {
		auto a = Entry(source, path), b = Entry(target, path);
		if (!Equal(db, a, b)) {
			changes.push_back({path, !a ? "deleted" : !b ? "added" : "modified"});
		}
	}
	return changes;
}

std::set<std::string> NamespaceConflicts(const Tree &tree) {
	std::set<std::string> conflicts;
	std::map<int64_t, std::string> inodes;
	for (const auto &entry : tree) {
		auto previous = inodes.emplace(entry.second.node.key.inode, entry.first);
		if (!previous.second) {
			conflicts.insert(previous.first->second);
			conflicts.insert(entry.first);
		}
		if (entry.first != "/") {
			auto parent_path = ParentPath(entry.first);
			auto parent = Entry(tree, parent_path);
			if (!parent || !parent->node.fields[0] || parent->node.key.inode != entry.second.parent) {
				conflicts.insert(parent_path);
			}
		}
	}
	return conflicts;
}

struct Comparison {
	Tree result;
	std::set<std::string> conflicts;
};

Comparison CompareMerge(sqlite3 *db, const View &source, const View &target,
                        const std::map<std::string, std::string> &resolutions) {
	if (source.info.parent_id != target.branch || source.info.fork_base.empty()) {
		Fail(ErrorCode::Invalid, "Merge requires a direct child and its parent");
	}
	{
		Statement children(db, "SELECT 1 FROM branches WHERE parent=? AND state!='deleted'");
		children.Bind(1, source.branch);
		if (children.Step()) {
			Fail(ErrorCode::Invalid, "Merge source must be a leaf branch");
		}
	}
	auto base = ReadTree(db, SnapshotPoint(db, source.info.fork_base));
	auto left = ReadTree(db, source.point), right = ReadTree(db, target.point);
	auto paths = Paths(left, right);
	for (const auto &entry : base) {
		paths.insert(entry.first);
	}
	for (const auto &resolution : resolutions) {
		if (!paths.count(resolution.first) || (resolution.second != "source" && resolution.second != "target") ||
		    Equal(db, Entry(left, resolution.first), Entry(right, resolution.first))) {
			Fail(ErrorCode::Invalid, "Invalid merge resolution: " + resolution.first);
		}
	}
	Comparison comparison;
	for (const auto &path : paths) {
		auto b = Entry(base, path), s = Entry(left, path), t = Entry(right, path);
		const TreeEntry *selected = t;
		auto resolution = resolutions.find(path);
		if (resolution != resolutions.end()) {
			selected = resolution->second == "source" ? s : t;
		} else if (!Equal(db, s, b)) {
			if (Equal(db, t, b) || Equal(db, s, t)) {
				selected = s;
			} else {
				comparison.conflicts.insert(path);
			}
		}
		if (selected) {
			comparison.result.emplace(path, *selected);
		}
	}
	auto structural = NamespaceConflicts(comparison.result);
	comparison.conflicts.insert(structural.begin(), structural.end());
	return comparison;
}

void ApplyTree(sqlite3 *db, const Tree &tree, const View &target) {
	std::map<Key, std::vector<int64_t>> desired[3];
	for (const auto &entry : tree) {
		const auto &node = entry.second.node;
		desired[INODES][node.key] = node.fields;
		if (entry.first != "/") {
			auto parent = tree.at(ParentPath(entry.first)).node.key.inode;
			desired[DIRENTS][{parent, entry.first.substr(entry.first.rfind('/') + 1)}] = {node.key.inode};
		}
		for (const auto &block : entry.second.blocks) {
			desired[BLOCKS][{node.key.inode, {}, block.first}] = {block.second};
		}
	}
	std::set<int64_t> changed_directories;
	for (auto table : {INODES, DIRENTS, BLOCKS}) {
		std::map<Key, std::vector<int64_t>> current;
		for (const auto &record : Visible(db, table, target.point)) {
			current[record.key] = record.fields;
		}
		for (const auto &entry : current) {
			if (!desired[table].count(entry.first)) {
				Put(db, table, entry.first, entry.second, target, true);
				if (table == DIRENTS) {
					changed_directories.insert(entry.first.inode);
				}
			}
		}
		for (const auto &entry : desired[table]) {
			auto existing = current.find(entry.first);
			if (existing == current.end() || existing->second != entry.second) {
				Put(db, table, entry.first, entry.second, target);
				if (table == DIRENTS) {
					changed_directories.insert(entry.first.inode);
				}
			}
		}
	}
	for (auto inode : changed_directories) {
		if (desired[INODES].count({inode})) {
			Touch(db, inode, target);
		}
	}
}
} // namespace

std::vector<Change> Workspace::Diff(const std::string &source_snapshot, const std::string &target_snapshot) {
	Transaction tx(*database, false);
	auto source = ReadTree(database->db, SnapshotPoint(database->db, source_snapshot));
	auto target = ReadTree(database->db, SnapshotPoint(database->db, target_snapshot));
	auto changes = Compare(database->db, source, target);
	tx.Commit();
	return changes;
}

MergePreview Workspace::PreviewMerge(const std::string &source_id, const std::string &target_id) {
	Transaction tx(*database, false);
	auto source = Branch(database->db, source_id, true, true), target = Branch(database->db, target_id, true, true);
	CheckMount(database->db, source.branch, true);
	CheckMount(database->db, target.branch, true);
	auto comparison = CompareMerge(database->db, source, target, {});
	MergePreview preview;
	preview.workspace_id = database->uuid;
	preview.source = source.branch;
	preview.target = target.branch;
	preview.source_generation = source.info.generation;
	preview.target_generation = target.info.generation;
	preview.changes = Compare(database->db, comparison.result, ReadTree(database->db, target.point));
	preview.conflicts.assign(comparison.conflicts.begin(), comparison.conflicts.end());
	tx.Commit();
	return preview;
}

void Workspace::Merge(const MergePreview &preview, const std::map<std::string, std::string> &resolutions) {
	Transaction tx(*database, true);
	if (preview.workspace_id != database->uuid) {
		Fail(ErrorCode::Invalid, "Merge preview belongs to another workspace");
	}
	auto source = Branch(database->db, preview.source, true), target = Branch(database->db, preview.target, true);
	CheckMount(database->db, source.branch, true);
	CheckMount(database->db, target.branch, true);
	if (source.info.generation != preview.source_generation || target.info.generation != preview.target_generation) {
		Fail(ErrorCode::Stale, "Merge preview is stale; recompute it before publication");
	}
	auto comparison = CompareMerge(database->db, source, target, resolutions);
	if (!comparison.conflicts.empty()) {
		std::string message = "Unresolved merge conflicts:";
		for (const auto &path : comparison.conflicts) {
			message += " " + path;
		}
		Fail(ErrorCode::Conflict, message);
	}
	ApplyTree(database->db, comparison.result, target);
	Changed(database->db, target);
	Statement seal(database->db, "UPDATE branches SET state='sealed',generation=generation+1 WHERE id=?");
	seal.Bind(1, source.branch);
	seal.Step();
	tx.Commit();
}

void Workspace::DeleteBranch(const std::string &branch_id, bool recursive) {
	Transaction tx(*database, true);
	auto branch = Branch(database->db, branch_id, false, true);
	if (branch.info.parent_id.empty()) {
		Fail(ErrorCode::Invalid, "Cannot delete the root branch");
	}
	std::vector<std::string> branches {branch.branch};
	for (size_t i = 0; i < branches.size(); ++i) {
		Statement children(database->db, "SELECT id FROM branches WHERE parent=? AND state!='deleted'");
		children.Bind(1, branches[i]);
		while (children.Step()) {
			if (!recursive) {
				Fail(ErrorCode::NotEmpty, "Branch has descendants; use recursive deletion");
			}
			branches.push_back(children.Text(0));
		}
	}
	for (const auto &id : branches) {
		CheckMount(database->db, id, true);
		Statement remove(database->db, "UPDATE branches SET state='deleted',generation=generation+1 WHERE id=?");
		remove.Bind(1, id);
		remove.Step();
	}
	tx.Commit();
}

CollectionResult Workspace::CollectGarbage() {
	Transaction tx(*database, true);
	CollectionResult result;
	Exec(database->db,
	     "DELETE FROM snapshots WHERE retained=0 AND NOT EXISTS(SELECT 1 FROM branches WHERE state!='deleted' AND "
	     "fork_base=snapshots.id) AND NOT EXISTS(SELECT 1 FROM pins WHERE snapshot=snapshots.id)");
	result.snapshots = sqlite3_changes64(database->db);
	for (auto table : {INODES, DIRENTS, BLOCKS}) {
		Exec(
		    database->db,
		    "DELETE FROM " + std::string(TableName(table)) +
		        " AS v WHERE NOT EXISTS(SELECT 1 FROM branches b WHERE b.state!='deleted' AND v.low<=b.frontier AND "
		        "b.frontier<v.high) AND NOT EXISTS(SELECT 1 FROM snapshots s WHERE v.low<=s.point AND s.point<v.high)");
		result.versions += sqlite3_changes64(database->db);
	}
	Exec(database->db,
	     "DELETE FROM block_payloads WHERE NOT EXISTS(SELECT 1 FROM block_versions WHERE payload=block_payloads.id)");
	result.payloads = sqlite3_changes64(database->db);
	tx.Commit();
	return result;
}

namespace {
void CleanupOrphans(sqlite3 *db) {
	std::vector<std::pair<std::string, int64_t>> orphans;
	{
		Statement rows(db, "SELECT branch,inode FROM orphans o WHERE NOT EXISTS(SELECT 1 FROM open_inodes f "
		                   "WHERE f.branch=o.branch AND f.inode=o.inode)");
		while (rows.Step()) {
			orphans.emplace_back(rows.Text(0), rows.Integer(1));
		}
	}
	for (const auto &entry : orphans) {
		auto view = Branch(db, entry.first);
		auto node = Inode(db, entry.second, view);
		for (const auto &block : Visible(db, BLOCKS, view.point, entry.second)) {
			Put(db, BLOCKS, block.key, {0}, view, true);
		}
		Put(db, INODES, node.key, node.fields, view, true);
		Statement remove(db, "DELETE FROM orphans WHERE branch=? AND inode=?");
		remove.Bind(1, entry.first);
		remove.Bind(2, entry.second);
		remove.Step();
		Changed(db, view);
	}
}

RecoveryResult RetireOwner(sqlite3 *db, const std::string &owner) {
	RecoveryResult result;
	for (const auto *table : {"pins", "open_inodes", "mounts"}) {
		Statement remove(db, "DELETE FROM " + std::string(table) + " WHERE owner=?");
		remove.Bind(1, owner);
		remove.Step();
		if (std::string(table) == "pins") {
			result.pins = sqlite3_changes64(db);
		} else if (std::string(table) == "mounts") {
			result.mounts = sqlite3_changes64(db);
		}
	}
	CleanupOrphans(db);
	Statement remove(db, "DELETE FROM owners WHERE id=?");
	remove.Bind(1, owner);
	remove.Step();
	result.owners = sqlite3_changes64(db);
	return result;
}

void RequireOwnMount(sqlite3 *db, const View &view) {
	Statement mount(db, "SELECT 1 FROM mounts WHERE branch=? AND owner=vane_fs_owner()");
	mount.Bind(1, view.branch);
	if (!mount.Step()) {
		Fail(ErrorCode::Busy, "Inode handles require an exclusive mount lease");
	}
}

void PinInode(sqlite3 *db, int64_t inode, const View &view, int64_t references = 1) {
	if (view.branch.empty()) {
		return;
	}
	RequireOwnMount(db, view);
	if (references < 1)
		Fail(ErrorCode::Invalid, "Invalid inode reference count");
	// The durable row protects the inode across connections and crash recovery.
	// Exact counts are connection-local and roll back with the same transaction;
	// additional opens/lookups need no WAL writes while this pin remains live.
	Statement pin(db, "INSERT INTO open_inodes VALUES(vane_fs_owner(),?,?,1) "
	                  "ON CONFLICT(owner,branch,inode) DO NOTHING");
	pin.Bind(1, view.branch);
	pin.Bind(2, inode);
	pin.Step();
	Statement count(db, "INSERT INTO temp.inode_references VALUES(?,?,?) "
	                    "ON CONFLICT(branch,inode) DO UPDATE SET refs=refs+excluded.refs");
	count.Bind(1, view.branch);
	count.Bind(2, inode);
	count.Bind(3, references);
	count.Step();
}

Record OpenedInode(sqlite3 *db, int64_t inode, const View &view) {
	if (!view.branch.empty()) {
		Statement opened(db, "SELECT 1 FROM open_inodes WHERE owner=vane_fs_owner() AND branch=? AND inode=?");
		opened.Bind(1, view.branch);
		opened.Bind(2, inode);
		if (!opened.Step()) {
			Fail(ErrorCode::Closed, "Inode handle is closed");
		}
	}
	return Inode(db, inode, view);
}

FileStat Describe(const Record &node) {
	return {node.key.inode, bool(node.fields[0]), node.fields[1],
	        node.fields[2], node.fields[3],       node.fields[0] ? 2 : 1};
}
} // namespace

void Database::Close() {
	if (process != OwnerLock::Process()) {
		// SQLite connections and C++ mutexes must not be reused across fork.
		statements.idle.clear(); // Abandon inherited statements without calling SQLite.
		db = nullptr;
		// The worker and its locks belong to vanished parent threads. As with
		// inherited SQLite handles, abandon them without joining or destroying.
		checkpointer.release();
		owner_lock.Close();
		return;
	}
	std::lock_guard<std::mutex> lock(mutex);
	if (!db) {
		return;
	}
	if (checkpointer)
		checkpointer->Stop();
	try {
		if (checkpointer)
			checkpointer->CheckError();
		if (durability == Durability::Fsync)
			Exec(db, "PRAGMA synchronous=FULL");
		Exec(db, "BEGIN IMMEDIATE");
		try {
			RetireOwner(db, owner);
			Exec(db, "COMMIT");
		} catch (...) {
			sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
			throw;
		}
		{
			auto &registry = Pins();
			std::lock_guard<std::mutex> guard(registry.mutex);
			registry.pins.erase(std::remove_if(registry.pins.begin(), registry.pins.end(),
			                                   [&](const auto &pin) { return pin->owner == owner; }),
			                    registry.pins.end());
		}
		statements.Clear();
		sqlite3_wal_hook(db, nullptr, nullptr);
		checkpointer.reset();
		Check(db, sqlite3_close(db));
		db = nullptr;
		owner_lock.Remove();
	} catch (...) {
		if (checkpointer)
			checkpointer->Start();
		throw;
	}
}

RecoveryResult Workspace::RecoverOwners() {
	RecoveryResult result;
	std::vector<std::unique_ptr<OwnerLock>> recovered;
	{
		Transaction tx(*database, true);
		struct Owner {
			std::string id, path;
			int64_t device, inode;
		};
		std::vector<Owner> owners;
		{
			Statement rows(database->db, "SELECT id,lock_path,device,inode FROM owners WHERE id<>vane_fs_owner()");
			while (rows.Step()) {
				owners.push_back({rows.Text(0), rows.Text(1), rows.Integer(2), rows.Integer(3)});
			}
		}
		for (const auto &owner : owners) {
			auto lock = std::make_unique<OwnerLock>();
			if (!lock->TryRecover(owner.path, owner.device, owner.inode)) {
				continue;
			}
			auto retired = RetireOwner(database->db, owner.id);
			result.owners += retired.owners;
			result.pins += retired.pins;
			result.mounts += retired.mounts;
			recovered.push_back(std::move(lock));
		}
		tx.Commit();
	}
	for (auto &lock : recovered) {
		lock->Remove();
	}
	return result;
}

BranchInfo Workspace::AcquireMount(const std::string &branch) {
	Transaction tx(*database, true);
	auto view = Branch(database->db, branch, true, true);
	CheckMount(database->db, view.branch, true);
	Statement mount(database->db, "INSERT INTO mounts VALUES(?,vane_fs_owner())");
	mount.Bind(1, view.branch);
	mount.Step();
	tx.Commit();
	return view.info;
}

void Workspace::ReleaseMount(const std::string &branch) {
	Transaction tx(*database, true);
	auto view = Branch(database->db, branch, false, true);
	RequireOwnMount(database->db, view);
	Statement references(database->db, "DELETE FROM temp.inode_references WHERE branch=?");
	references.Bind(1, view.branch);
	references.Step();
	for (const auto *table : {"open_inodes", "mounts"}) {
		Statement remove(database->db,
		                 "DELETE FROM " + std::string(table) + " WHERE owner=vane_fs_owner() AND branch=?");
		remove.Bind(1, view.branch);
		remove.Step();
	}
	CleanupOrphans(database->db);
	tx.Commit();
}

FileStat Session::OpenFile(const std::string &path, bool create, bool exclusive, bool truncate, int64_t mode) {
	Transaction tx(*database, !snapshot);
	auto view = SessionView(database->db, id, snapshot, closed, create || truncate);
	Record node;
	bool exists = true;
	try {
		node = Resolve(database->db, path, view);
	} catch (const Error &error) {
		if (error.code != ErrorCode::NotFound || !create) {
			throw;
		}
		exists = false;
		if (!path.empty() && path.back() == '/') {
			Fail(ErrorCode::IsDirectory, "Cannot create a file with a trailing slash");
		}
		node = Create(database->db, Parent(database->db, path, view), false, mode, view);
	}
	RequireFile(node);
	if (create && exclusive && exists) {
		Fail(ErrorCode::Exists, "File already exists");
	}
	if (truncate) {
		Resize(database->db, node, 0, view);
	}
	if (!exists || truncate) {
		Changed(database->db, view);
	}
	PinInode(database->db, node.key.inode, view);
	tx.Commit();
	return Describe(node);
}

FileStat Session::OpenDirectory(const std::string &path) {
	Transaction tx(*database, !snapshot);
	auto view = SessionView(database->db, id, snapshot, closed);
	auto node = Resolve(database->db, path, view);
	if (!node.fields[0]) {
		Fail(ErrorCode::NotDirectory, "Path is not a directory");
	}
	PinInode(database->db, node.key.inode, view);
	tx.Commit();
	return Describe(node);
}

void Session::CloseFile(int64_t inode, int64_t references) {
	if (references < 1)
		Fail(ErrorCode::Invalid, "Invalid inode reference count");
	Transaction tx(*database, !snapshot);
	auto view = SessionView(database->db, id, snapshot, closed);
	if (!snapshot) {
		OpenedInode(database->db, inode, view);
		Statement count(database->db, "SELECT refs FROM temp.inode_references WHERE branch=? AND inode=?");
		count.Bind(1, id);
		count.Bind(2, inode);
		if (!count.Step())
			Fail(ErrorCode::Closed, "Inode handle is closed");
		if (count.Integer(0) < references)
			Fail(ErrorCode::Invalid, "Too many inode references released");
		if (count.Integer(0) == references) {
			Statement remove_refs(database->db, "DELETE FROM temp.inode_references WHERE branch=? AND inode=?");
			remove_refs.Bind(1, id);
			remove_refs.Bind(2, inode);
			remove_refs.Step();
			Statement remove(database->db,
			                 "DELETE FROM open_inodes WHERE owner=vane_fs_owner() AND branch=? AND inode=?");
			remove.Bind(1, id);
			remove.Bind(2, inode);
			remove.Step();
			CleanupOrphans(database->db);
		} else {
			Statement decrement(database->db,
			                    "UPDATE temp.inode_references SET refs=refs-? WHERE branch=? AND inode=?");
			decrement.Bind(1, references);
			decrement.Bind(2, id);
			decrement.Bind(3, inode);
			decrement.Step();
		}
	}
	tx.Commit();
}

FileStat Session::StatInode(int64_t inode) {
	Transaction tx(*database, false);
	auto view = SessionView(database->db, id, snapshot, closed);
	auto node = OpenedInode(database->db, inode, view);
	auto result = Describe(node);
	Statement orphan(database->db, "SELECT 1 FROM orphans WHERE branch=? AND inode=?");
	orphan.Bind(1, view.branch);
	orphan.Bind(2, inode);
	if (orphan.Step()) {
		result.links = 0;
	}
	tx.Commit();
	return result;
}

std::vector<std::string> Session::ListDirectoryInode(int64_t inode) {
	Transaction tx(*database, false);
	auto view = SessionView(database->db, id, snapshot, closed);
	auto node = OpenedInode(database->db, inode, view);
	if (!node.fields[0]) {
		Fail(ErrorCode::NotDirectory, "Inode is not a directory");
	}
	std::vector<std::string> result;
	for (const auto &entry : Visible(database->db, DIRENTS, view.point, inode)) {
		result.push_back(entry.key.name);
	}
	std::sort(result.begin(), result.end());
	tx.Commit();
	return result;
}

std::string Session::ReadInode(int64_t inode, int64_t offset, int64_t size) {
	if (offset < 0 || size < 0) {
		Fail(ErrorCode::Invalid, "Invalid read range");
	}
	Transaction tx(*database, false);
	auto view = SessionView(database->db, id, snapshot, closed);
	auto node = OpenedInode(database->db, inode, view);
	RequireFile(node);
	int64_t length = std::min(size, std::max<int64_t>(0, node.fields[1] - offset));
	auto result = ReadBytes(database->db, inode, offset, length, view);
	tx.Commit();
	return result;
}

void Session::WriteInode(int64_t inode, const std::string &data, int64_t offset, bool append, bool synchronous) {
	Transaction tx(*database, true, synchronous);
	auto view = SessionView(database->db, id, snapshot, closed, true);
	auto node = OpenedInode(database->db, inode, view);
	WriteBytes(database->db, node, data, append ? node.fields[1] : offset, view);
	Changed(database->db, view);
	tx.Commit();
}

void Session::TruncateInode(int64_t inode, int64_t size) {
	Transaction tx(*database, true);
	auto view = SessionView(database->db, id, snapshot, closed, true);
	auto node = OpenedInode(database->db, inode, view);
	Resize(database->db, node, size, view);
	Changed(database->db, view);
	tx.Commit();
}

void Session::SetAttributes(const std::string &path, std::optional<int64_t> mode, std::optional<int64_t> mtime_ns,
                            int64_t inode, std::optional<int64_t> size) {
	if (mode && (*mode < 0 || *mode > 0777)) {
		Fail(ErrorCode::Invalid, "Mode must be between 0 and 0777");
	}
	Transaction tx(*database, true);
	auto view = SessionView(database->db, id, snapshot, closed, true);
	auto node = inode ? OpenedInode(database->db, inode, view) : Resolve(database->db, path, view);
	if (size)
		Resize(database->db, node, *size, view);
	if (mode) {
		node.fields[2] = *mode;
	}
	if (mtime_ns) {
		node.fields[3] = *mtime_ns;
	}
	Put(database->db, INODES, node.key, node.fields, view);
	Changed(database->db, view);
	tx.Commit();
}

namespace {
Key ChildKey(sqlite3 *db, int64_t parent, const std::string &name, const View &view) {
	if (name.empty() || name == "." || name == ".." || name.size() > 255 || name.find('/') != std::string::npos ||
	    name.find('\0') != std::string::npos) {
		Fail(ErrorCode::Invalid, "Invalid directory entry name");
	}
	auto directory = OpenedInode(db, parent, view);
	if (!directory.fields[0])
		Fail(ErrorCode::NotDirectory, "Parent is not a directory");
	Statement orphan(db, "SELECT 1 FROM orphans WHERE branch=? AND inode=?");
	orphan.Bind(1, view.branch);
	orphan.Bind(2, parent);
	if (orphan.Step())
		Fail(ErrorCode::NotFound, "Parent directory has been removed");
	return {parent, name};
}

int64_t ParentInode(sqlite3 *db, int64_t inode, const View &view) {
	if (inode == 1)
		return 1;
	Statement parent(db, "SELECT parent FROM dirent_versions WHERE inode=? AND low<=? AND high>? AND deleted=0");
	parent.Bind(1, inode);
	parent.Bind(2, view.point);
	parent.Bind(3, view.point);
	if (!parent.Step())
		Fail(ErrorCode::NotFound, "Directory has been removed");
	auto result = parent.Integer(0);
	if (parent.Step())
		Fail(ErrorCode::Storage, "Directory has multiple parents");
	return result;
}
} // namespace

FileStat Session::OpenInode(int64_t inode, bool directory, bool truncate, bool synchronous) {
	Transaction tx(*database, !snapshot, synchronous);
	auto view = SessionView(database->db, id, snapshot, closed, truncate);
	auto node = OpenedInode(database->db, inode, view);
	if (bool(node.fields[0]) != directory)
		Fail(directory ? ErrorCode::NotDirectory : ErrorCode::IsDirectory, "Open type mismatch");
	if (truncate) {
		Resize(database->db, node, 0, view);
		Changed(database->db, view);
	}
	PinInode(database->db, inode, view);
	tx.Commit();
	return Describe(node);
}

FileStat Session::LookupInode(int64_t parent, const std::string &name) {
	Transaction tx(*database, !snapshot);
	auto view = SessionView(database->db, id, snapshot, closed);
	Record node;
	if (name == "." || name == "..") {
		auto directory = OpenedInode(database->db, parent, view);
		if (!directory.fields[0])
			Fail(ErrorCode::NotDirectory, "Parent is not a directory");
		node = Inode(database->db, name == "." ? parent : ParentInode(database->db, parent, view), view);
	} else {
		auto key = ChildKey(database->db, parent, name, view);
		Record entry;
		if (!Get(database->db, DIRENTS, key, view.point, entry))
			Fail(ErrorCode::NotFound, "Entry does not exist");
		node = Inode(database->db, entry.fields[0], view);
	}
	PinInode(database->db, node.key.inode, view);
	tx.Commit();
	return Describe(node);
}

FileStat Session::CreateNode(int64_t parent, const std::string &name, bool directory, int64_t mode, bool exclusive,
                             bool truncate, int64_t references, bool synchronous) {
	Transaction tx(*database, true, synchronous);
	auto view = SessionView(database->db, id, snapshot, closed, true);
	auto key = ChildKey(database->db, parent, name, view);
	Record entry, node;
	if (Get(database->db, DIRENTS, key, view.point, entry)) {
		if (exclusive || directory)
			Fail(ErrorCode::Exists, "Entry already exists");
		node = Inode(database->db, entry.fields[0], view);
		RequireFile(node);
	} else {
		node = Create(database->db, key, directory, mode, view);
	}
	if (truncate)
		Resize(database->db, node, 0, view);
	PinInode(database->db, node.key.inode, view, references);
	Changed(database->db, view);
	tx.Commit();
	return Describe(node);
}

void Session::RemoveNode(int64_t parent, const std::string &name, bool directory) {
	Transaction tx(*database, true);
	auto view = SessionView(database->db, id, snapshot, closed, true);
	auto key = ChildKey(database->db, parent, name, view);
	Record entry;
	if (!Get(database->db, DIRENTS, key, view.point, entry))
		Fail(ErrorCode::NotFound, "Entry does not exist");
	auto node = Inode(database->db, entry.fields[0], view);
	if (bool(node.fields[0]) != directory)
		Fail(directory ? ErrorCode::NotDirectory : ErrorCode::IsDirectory, "Removal type mismatch");
	Erase(database->db, key, node, view);
	Changed(database->db, view);
	tx.Commit();
}

void Session::RenameNode(int64_t parent, const std::string &name, int64_t new_parent, const std::string &new_name,
                         bool no_replace) {
	Transaction tx(*database, true);
	auto view = SessionView(database->db, id, snapshot, closed, true);
	auto from = ChildKey(database->db, parent, name, view), to = ChildKey(database->db, new_parent, new_name, view);
	Record entry;
	if (!Get(database->db, DIRENTS, from, view.point, entry))
		Fail(ErrorCode::NotFound, "Entry does not exist");
	auto node = Inode(database->db, entry.fields[0], view);
	if (parent == new_parent && name == new_name) {
		if (no_replace)
			Fail(ErrorCode::Exists, "Rename destination exists");
		tx.Commit();
		return;
	}
	if (node.fields[0]) {
		for (auto ancestor = new_parent;; ancestor = ParentInode(database->db, ancestor, view)) {
			if (ancestor == node.key.inode)
				Fail(ErrorCode::Invalid, "Cannot move a directory into itself");
			if (ancestor == 1)
				break;
		}
	}
	Record destination;
	if (Get(database->db, DIRENTS, to, view.point, destination)) {
		if (no_replace)
			Fail(ErrorCode::Exists, "Rename destination exists");
		auto replaced = Inode(database->db, destination.fields[0], view);
		if (node.fields[0] != replaced.fields[0])
			Fail(replaced.fields[0] ? ErrorCode::IsDirectory : ErrorCode::NotDirectory, "Rename type mismatch");
		Erase(database->db, to, replaced, view);
	}
	Put(database->db, DIRENTS, from, {node.key.inode}, view, true);
	Put(database->db, DIRENTS, to, {node.key.inode}, view);
	Touch(database->db, parent, view);
	Touch(database->db, new_parent, view);
	Changed(database->db, view);
	tx.Commit();
}

std::vector<std::pair<std::string, FileStat>> Session::DirectoryEntries(int64_t inode) {
	Transaction tx(*database, false);
	auto view = SessionView(database->db, id, snapshot, closed);
	auto node = OpenedInode(database->db, inode, view);
	if (!node.fields[0])
		Fail(ErrorCode::NotDirectory, "Inode is not a directory");
	std::vector<std::pair<std::string, FileStat>> result;
	try {
		result.emplace_back("..", Describe(Inode(database->db, ParentInode(database->db, inode, view), view)));
	} catch (const Error &error) {
		if (error.code != ErrorCode::NotFound)
			throw;
	}
	result.emplace_back(".", Describe(node));
	// Resolve every child's inode at the same view point in one query. Keep a
	// missing inode visible to the caller instead of silently dropping its name.
	Statement entries(database->db,
	                  "SELECT d.name,i.inode,i.kind,i.size,i.mode,i.mtime_ns FROM dirent_versions d "
	                  "LEFT JOIN inode_versions i ON i.inode=d.inode AND i.low<=?1 AND i.high>?1 AND i.deleted=0 "
	                  "WHERE d.parent=?2 AND d.low<=?1 AND d.high>?1 AND d.deleted=0 ORDER BY d.name");
	entries.Bind(1, view.point);
	entries.Bind(2, inode);
	while (entries.Step()) {
		if (entries.IsNull(1))
			Fail(ErrorCode::NotFound, "Inode does not exist");
		auto name = entries.Text(0);
		if (result.back().first == name)
			Fail(ErrorCode::Storage, "Overlapping visible versions");
		bool directory = entries.Integer(2) != 0;
		result.emplace_back(std::move(name), FileStat {entries.Integer(1), directory, entries.Integer(3),
		                                               entries.Integer(4), entries.Integer(5), directory ? 2 : 1});
	}
	std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
	tx.Commit();
	return result;
}

} // namespace vane_fs
