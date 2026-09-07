#include "catch.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/extension_type_info.hpp"
#include "duckdb/common/operator/cast_operators.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/image.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/cast/cast_function_set.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/execution/expression_executor_state.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/array_stats.hpp"
#include "duckdb/storage/statistics/list_stats.hpp"
#include "duckdb/storage/statistics/struct_stats.hpp"

using namespace duckdb; // NOLINT
using namespace std;    // NOLINT

TEST_CASE("Image scalar pixels remain compact through vectors and serialization", "[image]") {
	string bytes(1920 * 1080 * 3, '\xff');
	bytes.front() = '\0';
	for (auto type :
	     {ImageLogicalType::Create(), ImageLogicalType::Create("RGB"), ImageLogicalType::Create("RGB", 1080, 1920)}) {
		auto value = ImageVector::FromPixels(const_data_ptr_cast(bytes.data()), bytes.size(), 1920, 1080, "RGB", type);
		REQUIRE(ByteSequenceValue::TryGet(ImageVector::PixelValues(value)));
		Vector constant(value);
		REQUIRE(constant.GetVectorType() == VectorType::CONSTANT_VECTOR);
		if (ImageLogicalType::IsFixedShape(type)) {
			REQUIRE(ArrayVector::GetTotalSize(constant) == bytes.size());
		}
		auto extracted = constant.GetValue(STANDARD_VECTOR_SIZE - 1);
		REQUIRE(ByteSequenceValue::TryGet(ImageVector::PixelValues(extracted)));
		REQUIRE(value == extracted);
		auto stats = BaseStatistics::FromConstant(value);
		auto &pixel_stats = ImageLogicalType::IsFixedShape(type)
		                        ? ArrayStats::GetChildStats(stats)
		                        : ListStats::GetChildStats(StructStats::GetChildStats(stats, ImageLogicalType::DATA));
		REQUIRE(NumericStats::GetMin<uint8_t>(pixel_stats) == 0);
		REQUIRE(NumericStats::GetMax<uint8_t>(pixel_stats) == 255);
		REQUIRE_FALSE(pixel_stats.CanHaveNull());
		REQUIRE(pixel_stats.CanHaveNoNull());
		REQUIRE(value.Hash() == extracted.Hash());
		MemoryStream stream;
		BinarySerializer::Serialize(value, stream);
		REQUIRE(stream.GetPosition() < bytes.size() + 2048);
		stream.Rewind();
		BinaryDeserializer deserializer(stream);
		deserializer.Begin();
		auto restored = Value::Deserialize(deserializer);
		deserializer.End();
		REQUIRE(ByteSequenceValue::TryGet(ImageVector::PixelValues(restored)));
		REQUIRE(restored == value);
		REQUIRE(*ByteSequenceValue::TryGet(ImageVector::PixelValues(restored)) == bytes);
	}
}

TEST_CASE("Image constructors retain a single constant pixel payload for full batches", "[image][file]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_FALSE(con.Query("SELECT image('a'::BLOB, 1, 1, 1, 'L')")->HasError());
	con.BeginTransaction();
	auto &entry = Catalog::GetEntry<ScalarFunctionCatalogEntry>(*con.context, INVALID_CATALOG, DEFAULT_SCHEMA, "image");
	duckdb::vector<LogicalType> types {LogicalType::BLOB, LogicalType::UINTEGER, LogicalType::UINTEGER,
	                                   LogicalType::UTINYINT, LogicalType::VARCHAR};
	auto function = entry.functions.GetFunctionByArguments(*con.context, types);
	DataChunk args;
	args.Initialize(Allocator::DefaultAllocator(), types);
	string pixels(96 * 128 * 3, 'x');
	duckdb::vector<Value> values {Value::BLOB_RAW(pixels), Value::UINTEGER(128), Value::UINTEGER(96),
	                              Value::UTINYINT(3), Value("RGB")};
	for (idx_t i = 0; i < values.size(); i++) {
		args.data[i].Reference(values[i]);
	}
	args.SetCardinality(STANDARD_VECTOR_SIZE);
	BoundConstantExpression expression(Value::INTEGER(1));
	ExpressionExecutorState root;
	ExpressionState state(expression, root);
	Vector result(ImageLogicalType::Create());
	function.function(args, state, result);
	REQUIRE(result.GetVectorType() == VectorType::CONSTANT_VECTOR);
	auto &data = *StructVector::GetEntries(result)[ImageLogicalType::DATA];
	REQUIRE(ListVector::GetListSize(data) == pixels.size());
	REQUIRE(result.GetValue(STANDARD_VECTOR_SIZE - 1) == result.GetValue(0));
	args.data[0].Reference(Value(LogicalType::BLOB));
	function.function(args, state, result);
	REQUIRE(result.GetVectorType() == VectorType::CONSTANT_VECTOR);
	REQUIRE(result.GetValue(STANDARD_VECTOR_SIZE - 1).IsNull());
	// Attribute access on a constant fixed Image must preserve its one-row
	// pixel vector even when a different property is requested on each row.
	auto fixed = ImageLogicalType::Create("RGB", 96, 128);
	DataChunk properties;
	properties.Initialize(Allocator::DefaultAllocator(), {fixed, LogicalType::VARCHAR}, {false, true});
	properties.data[0].Reference(
	    ImageVector::FromPixels(const_data_ptr_cast(pixels.data()), pixels.size(), 128, 96, "RGB", fixed));
	for (idx_t i = 0; i < STANDARD_VECTOR_SIZE; i++) {
		properties.data[1].SetValue(i, Value(i % 2 ? "width" : "height"));
	}
	properties.SetCardinality(STANDARD_VECTOR_SIZE);
	auto &attribute =
	    Catalog::GetEntry<ScalarFunctionCatalogEntry>(*con.context, INVALID_CATALOG, DEFAULT_SCHEMA, "image_attribute");
	auto attribute_function = attribute.functions.GetFunctionByArguments(*con.context, {fixed, LogicalType::VARCHAR});
	Vector attributes(LogicalType::UINTEGER);
	attribute_function.function(properties, state, attributes);
	REQUIRE(properties.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR);
	REQUIRE(ArrayVector::GetTotalSize(properties.data[0]) == pixels.size());
	REQUIRE(attributes.GetValue(0) == Value::UINTEGER(96));
	REQUIRE(attributes.GetValue(STANDARD_VECTOR_SIZE - 1) == Value::UINTEGER(128));
	CastFunctionSet casts;
	GetCastFunctionInput cast_input;
	cast_input.file_cast_mode = FileCastMode::EXPLICIT_IMAGE_LAYOUT;
	Vector cast_result(ImageLogicalType::Create(), 1);
	REQUIRE(
	    VectorOperations::TryCast(casts, cast_input, properties.data[0], cast_result, STANDARD_VECTOR_SIZE, nullptr));
	REQUIRE(cast_result.GetVectorType() == VectorType::CONSTANT_VECTOR);
	REQUIRE(ListVector::GetListSize(*StructVector::GetEntries(cast_result)[ImageLogicalType::DATA]) == pixels.size());
	REQUIRE(properties.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR);
	REQUIRE(properties.data[0].GetValue(STANDARD_VECTOR_SIZE - 1) == properties.data[0].GetValue(0));
	con.Rollback();
}

TEST_CASE("Compact Image scalar extraction applies dictionary selections once", "[image]") {
	for (auto type :
	     {ImageLogicalType::Create(), ImageLogicalType::Create("RGB"), ImageLogicalType::Create("RGB", 1, 2)}) {
		string first_bytes("abcdef"), second_bytes("uvwxyz");
		auto first = ImageVector::FromPixels(const_data_ptr_cast(first_bytes.data()), 6, 2, 1, "RGB", type);
		auto second = ImageVector::FromPixels(const_data_ptr_cast(second_bytes.data()), 6, 2, 1, "RGB", type);
		Vector source(type, 3);
		source.SetValue(0, first);
		source.SetValue(1, second);
		source.SetValue(2, Value(type));
		SelectionVector selection(3);
		selection.set_index(0, 1);
		selection.set_index(1, 0);
		selection.set_index(2, 2);
		Vector selected(source, selection, 3);
		REQUIRE(selected.GetValue(0) == second);
		REQUIRE(selected.GetValue(1) == first);
		REQUIRE(selected.GetValue(2).IsNull());
		SelectionVector again(2);
		again.set_index(0, 2);
		again.set_index(1, 0);
		selected.Slice(again, 2);
		REQUIRE(selected.GetValue(0).IsNull());
		REQUIRE(selected.GetValue(1) == second);
	}
}

TEST_CASE("IMAGE casts ignore inactive UNION payloads without changing their source", "[cast][image]") {
	auto image_type = ImageLogicalType::Create();
	auto fixed_image_type = ImageLogicalType::Create("RGB", 1, 2);
	child_list_t<LogicalType> source_members {{"image", image_type}, {"number", LogicalType::INTEGER}};
	child_list_t<LogicalType> target_members {{"image", fixed_image_type}, {"number", LogicalType::INTEGER}};
	auto source_type = LogicalType::UNION(source_members);
	auto target_type = LogicalType::UNION(target_members);
	duckdb::vector<Value> pixels;
	for (auto byte : string("abcdef")) {
		pixels.push_back(Value::UTINYINT(uint8_t(byte)));
	}
	auto good = ImageVector::FromPixels(pixels, 2, 1, "RGB", image_type);
	auto bad = ImageVector::FromPixels(pixels, 1, 2, "RGB", image_type);
	for (bool try_cast : {false, true}) {
		INFO("TRY_CAST=" << try_cast);
		Vector source(source_type, 4);
		source.SetValue(0, Value::UNION(source_members, 1, Value::INTEGER(7)));
		source.SetValue(1, Value::UNION(source_members, 0, good));
		source.SetValue(2, Value::UNION(source_members, 0, bad));
		source.SetValue(3, Value::UNION(source_members, 0, bad));
		// Retain a mismatched IMAGE beneath both an inactive member and a NULL
		// parent. Python UDF boundaries deliberately reject governed UNION types,
		// so exercise these raw vector slots directly.
		auto &images = UnionVector::GetMember(source, 0);
		images.SetValue(0, bad);
		FlatVector::Validity(source).SetInvalid(2);
		FlatVector::Validity(UnionVector::GetTags(source)).SetInvalid(2);

		CastFunctionSet casts;
		GetCastFunctionInput input;
		input.file_cast_mode = FileCastMode::EXPLICIT_IMAGE_LAYOUT;
		string error;
		Vector result(target_type, 4);
		REQUIRE(VectorOperations::TryCast(casts, input, source, result, 3, try_cast ? &error : nullptr));
		REQUIRE(error.empty());
		REQUIRE(UnionVector::GetMember(result, 1).GetValue(0) == Value::INTEGER(7));
		REQUIRE(FlatVector::IsNull(UnionVector::GetMember(result, 0), 0));
		REQUIRE(UnionVector::GetMember(result, 0).GetValue(1) ==
		        ImageVector::FromPixels(pixels, 2, 1, "RGB", fixed_image_type));
		REQUIRE(result.GetValue(2).IsNull());
		REQUIRE(images.GetValue(0) == bad);
		REQUIRE(images.GetValue(2) == bad);
		REQUIRE(FlatVector::IsNull(source, 2));
		// The same payload must fail once it belongs to a visible selected member.
		Vector with_active_failure(target_type, 4);
		if (try_cast) {
			REQUIRE_FALSE(VectorOperations::TryCast(casts, input, source, with_active_failure, 4, &error));
			REQUIRE_FALSE(error.empty());
			REQUIRE(UnionVector::GetTags(with_active_failure).GetValue(3) == Value::UTINYINT(0));
			REQUIRE(FlatVector::IsNull(UnionVector::GetMember(with_active_failure, 0), 3));
		} else {
			REQUIRE_THROWS_AS(VectorOperations::TryCast(casts, input, source, with_active_failure, 4, nullptr),
			                  InvalidInputException);
		}
	}
}

TEST_CASE("Arrow alias restoration preserves already governed siblings", "[cast][image]") {
	auto file = FileLogicalType::Create(FileMediaType::AUDIO);
	auto storage = file.DeepCopy();
	storage.SetAlias(string());
	storage.SetExtensionInfo(nullptr);
	CastFunctionSet casts;
	GetCastFunctionInput input;
	input.file_cast_mode = FileCastMode::INTERNAL_ALIAS_RESTORATION;
	for (auto image :
	     {ImageLogicalType::Create(), ImageLogicalType::Create("RGB"), ImageLogicalType::Create("RGB", 1, 2)}) {
		auto source = LogicalType::STRUCT({{"files", LogicalType::LIST(storage)}, {"image", image}});
		auto target = LogicalType::STRUCT({{"files", LogicalType::LIST(file)}, {"image", image}});
		REQUIRE_NOTHROW(casts.GetCastFunction(source, target, input));
		// An equally sized but differently shaped Image must not be retagged.
		auto changed =
		    LogicalType::STRUCT({{"files", LogicalType::LIST(file)}, {"image", ImageLogicalType::Create("RGB", 2, 1)}});
		REQUIRE_THROWS_AS(casts.GetCastFunction(source, changed, input), BinderException);
		auto erased_image = image.DeepCopy();
		erased_image.SetAlias(string());
		erased_image.SetExtensionInfo(nullptr);
		auto erased = LogicalType::STRUCT({{"files", LogicalType::LIST(file)}, {"image", erased_image}});
		REQUIRE_THROWS_AS(casts.GetCastFunction(source, erased, input), BinderException);
		REQUIRE_THROWS_AS(
		    casts.GetCastFunction(source, LogicalType::STRUCT({{"files", LogicalType::LIST(file)}}), input),
		    BinderException);
	}
}

template <class SRC, class DST>
struct ExpectedNumericCast {
	static inline DST Operation(SRC value) {
		return (DST)value;
	}
};

template <class DST>
struct ExpectedNumericCast<double, DST> {
	static inline DST Operation(double value) {
		return (DST)nearbyint(value);
	}
};

template <class DST>
struct ExpectedNumericCast<float, DST> {
	static inline DST Operation(float value) {
		return (DST)nearbyintf(value);
	}
};

template <class SRC, class DST>
static void TestNumericCast(duckdb::vector<SRC> &working_values, duckdb::vector<SRC> &broken_values) {
	DST result;
	for (auto value : working_values) {
		REQUIRE_NOTHROW(Cast::Operation<SRC, DST>(value) == (DST)value);
		REQUIRE(TryCast::Operation<SRC, DST>(value, result));
		REQUIRE(result == ExpectedNumericCast<SRC, DST>::Operation(value));
	}
	for (auto value : broken_values) {
		REQUIRE_THROWS(Cast::Operation<SRC, DST>(value));
		REQUIRE(!TryCast::Operation<SRC, DST>(value, result));
	}
}

template <class DST>
static void TestStringCast(duckdb::vector<string> &working_values, duckdb::vector<DST> &expected_values,
                           duckdb::vector<string> &broken_values) {
	DST result;
	for (idx_t i = 0; i < working_values.size(); i++) {
		auto &value = working_values[i];
		auto expected_value = expected_values[i];
		REQUIRE_NOTHROW(Cast::Operation<string_t, DST>(string_t(value)) == expected_value);
		REQUIRE(TryCast::Operation<string_t, DST>(string_t(value), result));
		REQUIRE(result == expected_value);

		StringUtil::Trim(value);
		duckdb::vector<string> splits;
		splits = StringUtil::Split(value, 'e');
		if (splits.size() > 1 || value[0] == '+') {
			continue;
		}
		splits = StringUtil::Split(value, '.');
		REQUIRE(ConvertToString::Operation<DST>(result) == splits[0]);
	}
	for (auto &value : broken_values) {
		REQUIRE_THROWS(Cast::Operation<string_t, DST>(string_t(value)));
		REQUIRE(!TryCast::Operation<string_t, DST>(string_t(value), result));
	}
}

template <class T>
static void TestExponent() {
	T parse_result;
	string str;
	double value = 1;
	T expected_value = 1;
	for (idx_t exponent = 0; exponent < 100; exponent++) {
		if (value < (double)NumericLimits<T>::Maximum()) {
			// expect success
			str = "1e" + to_string(exponent);
			REQUIRE(TryCast::Operation<string_t, T>(string_t(str), parse_result));
			REQUIRE(parse_result == expected_value);
			str = "-1e" + to_string(exponent);
			REQUIRE(TryCast::Operation<string_t, T>(string_t(str), parse_result));
			REQUIRE(parse_result == -expected_value);
			value *= 10;
			// check again because otherwise this overflows
			if (value < (double)NumericLimits<T>::Maximum()) {
				expected_value *= 10;
			}
		} else {
			// expect failure
			str = "1e" + to_string(exponent);
			REQUIRE(!TryCast::Operation<string_t, T>(string_t(str), parse_result));
			str = "-1e" + to_string(exponent);
			REQUIRE(!TryCast::Operation<string_t, T>(string_t(str), parse_result));
		}
	}
}

TEST_CASE("Test casting to boolean", "[cast]") {
	duckdb::vector<string> working_values = {"true", "false", "TRUE", "FALSE", "T", "F", "1", "0", "False", "True"};
	duckdb::vector<bool> expected_values = {true, false, true, false, true, false, true, false, false, true};
	duckdb::vector<string> broken_values = {"304", "1002", "blabla", "", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaa"};

	bool result;
	for (idx_t i = 0; i < working_values.size(); i++) {
		auto &value = working_values[i];
		auto expected_value = expected_values[i];
		REQUIRE_NOTHROW(Cast::Operation<string_t, bool>(value) == expected_value);
		REQUIRE(TryCast::Operation<string_t, bool>(value, result));
		REQUIRE(result == expected_value);
	}
	for (auto &value : broken_values) {
		REQUIRE_THROWS(Cast::Operation<string_t, bool>(value));
		REQUIRE(!TryCast::Operation<string_t, bool>(value, result));
	}
}

TEST_CASE("Test casting to int8_t", "[cast]") {
	// int16_t -> int8_t
	duckdb::vector<int16_t> working_values_int16 = {10, -10, 127, -128};
	duckdb::vector<int16_t> broken_values_int16 = {128, -129, 1000, -1000};
	TestNumericCast<int16_t, int8_t>(working_values_int16, broken_values_int16);
	// int32_t -> int8_t
	duckdb::vector<int32_t> working_values_int32 = {10, -10, 127, -128};
	duckdb::vector<int32_t> broken_values_int32 = {128, -129, 1000000, -1000000};
	TestNumericCast<int32_t, int8_t>(working_values_int32, broken_values_int32);
	// int64_t -> int8_t
	duckdb::vector<int64_t> working_values_int64 = {10, -10, 127, -128};
	duckdb::vector<int64_t> broken_values_int64 = {128, -129, 10000000000LL, -10000000000LL};
	TestNumericCast<int64_t, int8_t>(working_values_int64, broken_values_int64);
	// float -> int8_t
	duckdb::vector<float> working_values_float = {10, -10, 127, -128, 1.3f, -2.7f};
	duckdb::vector<float> broken_values_float = {128, -129, 10000000000.0f, -10000000000.0f, 1e30f, -1e30f};
	TestNumericCast<float, int8_t>(working_values_float, broken_values_float);
	// double -> int8_t
	duckdb::vector<double> working_values_double = {10, -10, 127, -128, 1.3, -2.7};
	duckdb::vector<double> broken_values_double = {128, -129, 10000000000.0, -10000000000.0, 1e100, -1e100};
	TestNumericCast<double, int8_t>(working_values_double, broken_values_double);
	// string -> int8_t
	duckdb::vector<string> working_values_str = {"10",  "+10", "-10",   "127", "-128", "1.3",   "1e2",
	                                             "2e1", "2e0", "20e-1", "1.",  "  3",  " 3   ", "\t3 \t \n"};
	duckdb::vector<int8_t> expected_values_str = {10, 10, -10, 127, -128, 1, 100, 20, 2, 2, 1, 3, 3, 3};
	duckdb::vector<string> broken_values_str = {"128",
	                                            "-129",
	                                            "10000000000000000000000000000000000000000000000000000000000000",
	                                            "aaaa",
	                                            "19A",
	                                            "",
	                                            "1e3",
	                                            "1e",
	                                            "1e-",
	                                            "1e100",
	                                            "1e100000000",
	                                            "10000e-1",
	                                            " 3 2",
	                                            "+"};
	TestStringCast<int8_t>(working_values_str, expected_values_str, broken_values_str);
	TestExponent<int8_t>();
}

TEST_CASE("Test casting to int16_t", "[cast]") {
	// int32_t -> int16_t
	duckdb::vector<int32_t> working_values_int32 = {10, -10, 127, -127, 32767, -32768};
	duckdb::vector<int32_t> broken_values_int32 = {32768, -32769, 1000000, -1000000};
	TestNumericCast<int32_t, int16_t>(working_values_int32, broken_values_int32);
	// int64_t -> int16_t
	duckdb::vector<int64_t> working_values_int64 = {10, -10, 127, -127, 32767, -32768};
	duckdb::vector<int64_t> broken_values_int64 = {32768, -32769, 10000000000LL, -10000000000LL};
	TestNumericCast<int64_t, int16_t>(working_values_int64, broken_values_int64);
	// float -> int16_t
	duckdb::vector<float> working_values_float = {10.0f, -10.0f, 32767.0f, -32768.0f, 1.3f, -2.7f};
	duckdb::vector<float> broken_values_float = {32768.0f, -32769.0f, 10000000000.0f, -10000000000.0f, 1e30f, -1e30f};
	TestNumericCast<float, int16_t>(working_values_float, broken_values_float);
	// double -> int16_t
	duckdb::vector<double> working_values_double = {10, -10, 32767, -32768, 1.3, -2.7};
	duckdb::vector<double> broken_values_double = {32768, -32769, 10000000000.0, -10000000000.0, 1e100, -1e100};
	TestNumericCast<double, int16_t>(working_values_double, broken_values_double);
	// string -> int16_t
	duckdb::vector<string> working_values_str = {"10",  "-10",   "32767", "-32768", "1.3",
	                                             "3e4", "250e2", "3e+4",  "3e0",    "30e-1"};
	duckdb::vector<int16_t> expected_values_str = {10, -10, 32767, -32768, 1, 30000, 25000, 30000, 3, 3};
	duckdb::vector<string> broken_values_str = {
	    "32768", "-32769",      "10000000000000000000000000000000000000000000000000000000000000",
	    "aaaa",  "19A",         "",
	    "1.A",   "1e",          "1e-",
	    "1e100", "1e100000000", "+"};
	TestStringCast<int16_t>(working_values_str, expected_values_str, broken_values_str);
	TestExponent<int16_t>();
}

TEST_CASE("Test casting to int32_t", "[cast]") {
	// int64_t -> int32_t
	duckdb::vector<int64_t> working_values_int64 = {10, -10, 127, -127, 32767, -32768, 2147483647LL, -2147483648LL};
	duckdb::vector<int64_t> broken_values_int64 = {2147483648LL, -2147483649LL, 10000000000LL, -10000000000LL};
	TestNumericCast<int64_t, int32_t>(working_values_int64, broken_values_int64);
	// float -> int32_t
	duckdb::vector<float> working_values_float = {10.0f, -10.0f, 2000000000.0f, -2000000000.0f, 1.3f, -2.7f};
	duckdb::vector<float> broken_values_float = {3000000000.0f,   -3000000000.0f, 10000000000.0f,
	                                             -10000000000.0f, 1e30f,          -1e30f};
	TestNumericCast<float, int32_t>(working_values_float, broken_values_float);
	// double -> int32_t
	duckdb::vector<double> working_values_double = {10, -10, 32767.0, -32768.0, 1.3, -2.7, 2147483647.0, -2147483648.0};
	duckdb::vector<double> broken_values_double = {2147483648.0,   -2147483649.0, 10000000000.0,
	                                               -10000000000.0, 1e100,         -1e100};
	TestNumericCast<double, int32_t>(working_values_double, broken_values_double);
	// string -> int32_t
	duckdb::vector<string> working_values_str = {"10", "-10", "2147483647", "-2147483647", "1.3", "-1.3", "1e6"};
	duckdb::vector<int32_t> expected_values_str = {10, -10, 2147483647, -2147483647, 1, -1, 1000000};
	duckdb::vector<string> broken_values_str = {
	    "2147483648", "-2147483649", "10000000000000000000000000000000000000000000000000000000000000",
	    "aaaa",       "19A",         "",
	    "1.A",        "1e1e1e1"};
	TestStringCast<int32_t>(working_values_str, expected_values_str, broken_values_str);
	TestExponent<int32_t>();
}

TEST_CASE("Test casting to int64_t", "[cast]") {
	// float -> int64_t
	duckdb::vector<float> working_values_float = {10.0f,
	                                              -10.0f,
	                                              32767.0f,
	                                              -32768.0f,
	                                              1.3f,
	                                              -2.7f,
	                                              2000000000.0f,
	                                              -2000000000.0f,
	                                              4000000000000000000.0f,
	                                              -4000000000000000000.0f};
	duckdb::vector<float> broken_values_float = {20000000000000000000.0f, -20000000000000000000.0f, 1e30f, -1e30f};
	TestNumericCast<float, int64_t>(working_values_float, broken_values_float);
	// double -> int64_t
	duckdb::vector<double> working_values_double = {
	    10, -10, 32767, -32768, 1.3, -2.7, 2147483647, -2147483648.0, 4611686018427387904.0, -4611686018427387904.0};
	duckdb::vector<double> broken_values_double = {18446744073709551616.0, -18446744073709551617.0, 1e100, -1e100};
	TestNumericCast<double, int64_t>(working_values_double, broken_values_double);
	// string -> int64_t
	duckdb::vector<string> working_values_str = {
	    "10",    "-10", "9223372036854775807", "-9223372036854775807", "1.3", "-9223372036854775807.1293813", "1e18",
	    "1e+18", "1."};
	duckdb::vector<int64_t> expected_values_str = {10,
	                                               -10,
	                                               9223372036854775807LL,
	                                               -9223372036854775807LL,
	                                               1,
	                                               -9223372036854775807LL,
	                                               1000000000000000000LL,
	                                               1000000000000000000LL,
	                                               1};
	duckdb::vector<string> broken_values_str = {"9223372036854775808",
	                                            "-9223372036854775809",
	                                            "10000000000000000000000000000000000000000000000000000000000000",
	                                            "aaaa",
	                                            "19A",
	                                            "",
	                                            "1.A",
	                                            "1.2382398723A",
	                                            "1e++1",
	                                            "1e+1+1",
	                                            "1e+1-1",
	                                            "+"};
	TestStringCast<int64_t>(working_values_str, expected_values_str, broken_values_str);
	TestExponent<int64_t>();
}

template <class DST>
static void TestStringCastDouble(duckdb::vector<string> &working_values, duckdb::vector<DST> &expected_values,
                                 duckdb::vector<string> &broken_values) {
	DST result;
	for (idx_t i = 0; i < working_values.size(); i++) {
		auto &value = working_values[i];
		auto expected_value = expected_values[i];
		REQUIRE_NOTHROW(Cast::Operation<string_t, DST>(string_t(value)) == expected_value);
		REQUIRE(TryCast::Operation<string_t, DST>(string_t(value), result));
		REQUIRE(ApproxEqual(result, expected_value));

		auto to_str_and_back =
		    Cast::Operation<string_t, DST>(string_t(ConvertToString::Operation<DST>(expected_value)));
		REQUIRE(ApproxEqual(to_str_and_back, expected_value));
	}
	for (auto &value : broken_values) {
		REQUIRE_THROWS(Cast::Operation<string_t, DST>(string_t(value)));
		REQUIRE(!TryCast::Operation<string_t, DST>(string_t(value), result));
	}
}

TEST_CASE("Test casting to float", "[cast]") {
	// string -> float
	duckdb::vector<string> working_values = {
	    "1.3",         "1.34514", "1e10", "1e-2", "-1e-1", "1.1781237378938173987123987123981723981723981723987123",
	    "1.123456789", "1."};
	duckdb::vector<float> expected_values = {
	    1.3f,         1.34514f, 1e10f, 1e-2f, -1e-1f, 1.1781237378938173987123987123981723981723981723987123f,
	    1.123456789f, 1.0f};
	duckdb::vector<string> broken_values = {
	    "-",     "",        "aaa",
	    "12aaa", "1e10e10", "1e",
	    "1e-",   "1e10a",   "1.1781237378938173987123987123981723981723981723934834583490587123w",
	    "1.2.3"};
	TestStringCastDouble<float>(working_values, expected_values, broken_values);
}

TEST_CASE("Test casting to double", "[cast]") {
	// string -> double
	duckdb::vector<string> working_values = {"1.3",
	                                         "+1.3",
	                                         "1.34514",
	                                         "1e10",
	                                         "1e-2",
	                                         "-1e-1",
	                                         "1.1781237378938173987123987123981723981723981723987123",
	                                         "1.123456789",
	                                         "1.",
	                                         "-1.2",
	                                         "-1.2e1",
	                                         " 1.2 ",
	                                         "  1.2e2  ",
	                                         " \t 1.2e2 \t"};
	duckdb::vector<double> expected_values = {
	    1.3,         1.3, 1.34514, 1e10, 1e-2, -1e-1, 1.1781237378938173987123987123981723981723981723987123,
	    1.123456789, 1.0, -1.2,    -12,  1.2,  120,   120};
	duckdb::vector<string> broken_values = {
	    "-",     "",        "aaa",
	    "12aaa", "1e10e10", "1e",
	    "1e-",   "1e10a",   "1.1781237378938173987123987123981723981723981723934834583490587123w",
	    "1.2.3", "1.222.",  "1..",
	    "1 . 2", "1. 2",    "1.2 e20",
	    "+"};
	TestStringCastDouble<double>(working_values, expected_values, broken_values);
}
