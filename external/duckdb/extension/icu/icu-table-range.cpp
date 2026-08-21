// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/common/exception.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/types/interval.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/table/distributed_sequence.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "include/icu-datefunc.hpp"
#include "tz_calendar.hpp"
#include "unicode/calendar.h"

namespace duckdb {

struct ICUTableRange {
	using CalendarPtr = unique_ptr<icu::Calendar>;

	static CalendarPtr CreateCalendar(const string &tz_setting, const string &cal_setting) {
		auto timezone = icu::TimeZone::createTimeZone(icu::UnicodeString::fromUTF8(icu::StringPiece(tz_setting)));
		const auto calendar_name = cal_setting.empty() ? "gregorian" : cal_setting;
		const string calendar_locale = "@calendar=" + calendar_name;
		icu::Locale locale(calendar_locale.c_str());
		UErrorCode status = U_ZERO_ERROR;
		CalendarPtr result(icu::Calendar::createInstance(timezone, locale, status));
		if (U_FAILURE(status) || !result) {
			throw InternalException("Unable to create ICU calendar");
		}
		return result;
	}

	struct ICURangeBindData : public TableFunctionData {
		ICURangeBindData() = default;

		ICURangeBindData(const ICURangeBindData &other)
		    : TableFunctionData(other), tz_setting(other.tz_setting), cal_setting(other.cal_setting),
		      calendar(other.calendar ? other.calendar->clone() : nullptr), has_parameters(other.has_parameters),
		      is_null(other.is_null), generate_series(other.generate_series), distributed(other.distributed),
		      exact_count(other.exact_count), increasing(other.increasing), empty_range(other.empty_range),
		      start(other.start), end(other.end), increment(other.increment), cardinality(other.cardinality),
		      shards(other.shards) {
		}

		ICURangeBindData(ClientContext &context, const vector<Value> &inputs, bool generate_series_p)
		    : generate_series(generate_series_p) {
			Value setting;
			if (context.TryGetCurrentSetting("TimeZone", setting)) {
				tz_setting = setting.ToString();
			} else {
				tz_setting = "UTC";
			}
			if (context.TryGetCurrentSetting("Calendar", setting)) {
				cal_setting = setting.ToString();
			} else {
				cal_setting = "gregorian";
			}
			calendar = CreateCalendar(tz_setting, cal_setting);
			if (inputs.empty()) {
				return;
			}
			has_parameters = true;
			for (const auto &value : inputs) {
				if (value.IsNull()) {
					is_null = true;
					exact_count = true;
					empty_range = true;
					return;
				}
			}
			start = inputs[0].GetValue<timestamp_tz_t>();
			end = inputs[1].GetValue<timestamp_tz_t>();
			increment = inputs[2].GetValue<interval_t>();
			InitializeSequence();
		}

		void InitializeSequence() {
			if (!Timestamp::IsFinite(start) || !Timestamp::IsFinite(end)) {
				throw BinderException("RANGE with infinite bounds is not supported");
			}
			const bool has_positive = increment.months > 0 || increment.days > 0 || increment.micros > 0;
			const bool has_negative = increment.months < 0 || increment.days < 0 || increment.micros < 0;
			if (!has_positive && !has_negative) {
				throw BinderException("interval cannot be 0!");
			}
			if (has_positive && has_negative) {
				throw BinderException("RANGE with composite interval that has mixed signs is not supported");
			}
			increasing = has_positive;
			empty_range = increasing ? start > end : start < end;
			if (empty_range) {
				exact_count = true;
				cardinality = 0;
				return;
			}
			if (start == end) {
				exact_count = true;
				cardinality = generate_series ? 1 : 0;
				empty_range = !generate_series;
				return;
			}
			if (increment.months == 0 && increment.days == 0) {
				exact_count = true;
				cardinality =
				    ComputeDistributedSequenceCardinality(start.value, end.value, increment.micros, generate_series);
				empty_range = cardinality == 0;
			}
		}

		string tz_setting;
		string cal_setting;
		CalendarPtr calendar;
		bool has_parameters = false;
		bool is_null = false;
		bool generate_series = false;
		bool distributed = false;
		bool exact_count = false;
		bool increasing = true;
		bool empty_range = false;
		timestamp_t start = timestamp_t(0);
		timestamp_t end = timestamp_t(0);
		interval_t increment {0, 0, 0};
		idx_t cardinality = 0;
		vector<DistributedSequenceShard> shards;

		unique_ptr<FunctionData> Copy() const override {
			return make_uniq<ICURangeBindData>(*this);
		}

		bool Equals(const FunctionData &other_p) const override {
			auto other = dynamic_cast<const ICURangeBindData *>(&other_p);
			if (!other || column_ids != other->column_ids || tz_setting != other->tz_setting ||
			    cal_setting != other->cal_setting || has_parameters != other->has_parameters ||
			    is_null != other->is_null || generate_series != other->generate_series ||
			    distributed != other->distributed || exact_count != other->exact_count ||
			    increasing != other->increasing || empty_range != other->empty_range || start != other->start ||
			    end != other->end || increment != other->increment || cardinality != other->cardinality ||
			    shards.size() != other->shards.size()) {
				return false;
			}
			for (idx_t shard_index = 0; shard_index < shards.size(); shard_index++) {
				const auto &left = shards[shard_index];
				const auto &right = other->shards[shard_index];
				if (left.ordinal != right.ordinal || left.row_offset != right.row_offset ||
				    left.row_count != right.row_count || left.exact_count != right.exact_count) {
					return false;
				}
			}
			return true;
		}
	};

	struct ICURangeLocalState : public LocalTableFunctionState {
		bool initialized_row = false;
		idx_t current_input_row = 0;
		timestamp_t current_state;

		timestamp_t start;
		timestamp_t end;
		interval_t increment {0, 0, 0};
		bool inclusive_bound = false;
		bool greater_than_check = true;
		bool empty_range = false;

		idx_t shard_index = 0;
		idx_t shard_row = 0;

		bool Finished(timestamp_t current_value) const {
			if (greater_than_check) {
				return inclusive_bound ? current_value > end : current_value >= end;
			}
			return inclusive_bound ? current_value < end : current_value <= end;
		}
	};

	template <bool GENERATE_SERIES>
	static void GenerateRangeDateTimeParameters(DataChunk &input, idx_t row_id, ICURangeLocalState &result) {
		input.Flatten();
		result.empty_range = false;
		for (idx_t column_index = 0; column_index < input.ColumnCount(); column_index++) {
			if (FlatVector::IsNull(input.data[column_index], row_id)) {
				result.start = timestamp_t(0);
				result.end = timestamp_t(0);
				result.increment = interval_t {0, 0, 0};
				result.greater_than_check = true;
				result.inclusive_bound = false;
				result.empty_range = true;
				return;
			}
		}

		result.start = FlatVector::GetValue<timestamp_t>(input.data[0], row_id);
		result.end = FlatVector::GetValue<timestamp_t>(input.data[1], row_id);
		result.increment = FlatVector::GetValue<interval_t>(input.data[2], row_id);
		if (!Timestamp::IsFinite(result.start) || !Timestamp::IsFinite(result.end)) {
			throw BinderException("RANGE with infinite bounds is not supported");
		}
		const bool has_positive =
		    result.increment.months > 0 || result.increment.days > 0 || result.increment.micros > 0;
		const bool has_negative =
		    result.increment.months < 0 || result.increment.days < 0 || result.increment.micros < 0;
		if (!has_positive && !has_negative) {
			throw BinderException("interval cannot be 0!");
		}
		if (has_positive && has_negative) {
			throw BinderException("RANGE with composite interval that has mixed signs is not supported");
		}
		result.greater_than_check = has_positive;
		if ((has_positive && result.start > result.end) || (has_negative && result.start < result.end)) {
			result.empty_range = true;
		}
		result.inclusive_bound = GENERATE_SERIES;
	}

	template <bool GENERATE_SERIES>
	static unique_ptr<FunctionData> Bind(ClientContext &context, TableFunctionBindInput &input,
	                                     vector<LogicalType> &return_types, vector<string> &names) {
		return_types.push_back(LogicalType::TIMESTAMP_TZ);
		names.emplace_back(GENERATE_SERIES ? "generate_series" : "range");
		if (!input.inputs.empty() && input.inputs.size() != 3) {
			return nullptr;
		}
		return make_uniq<ICURangeBindData>(context, input.inputs, GENERATE_SERIES);
	}

	static unique_ptr<LocalTableFunctionState> LocalInit(ExecutionContext &, TableFunctionInitInput &,
	                                                     GlobalTableFunctionState *) {
		return make_uniq<ICURangeLocalState>();
	}

	static unique_ptr<NodeStatistics> Cardinality(ClientContext &, const FunctionData *bind_data_p) {
		if (!bind_data_p) {
			return nullptr;
		}
		auto &bind_data = bind_data_p->Cast<ICURangeBindData>();
		if (!bind_data.has_parameters || !bind_data.exact_count) {
			return nullptr;
		}
		return make_uniq<NodeStatistics>(bind_data.cardinality, bind_data.cardinality);
	}

	static OperatorResultType DistributedFunction(const ICURangeBindData &bind_data, ICURangeLocalState &state,
	                                              DataChunk &output) {
		TZCalendar calendar(*bind_data.calendar, bind_data.cal_setting);
		while (state.shard_index < bind_data.shards.size()) {
			const auto &shard = bind_data.shards[state.shard_index];
			if (!state.initialized_row) {
				state.current_state = shard.exact_count
				                          ? timestamp_t(GetDistributedSequenceValue(
				                                bind_data.start.value, bind_data.increment.micros, shard.row_offset))
				                          : bind_data.start;
				state.end = bind_data.end;
				state.increment = bind_data.increment;
				state.greater_than_check = bind_data.increasing;
				state.inclusive_bound = bind_data.generate_series;
				state.shard_row = 0;
				state.initialized_row = true;
			}

			idx_t size = 0;
			auto output_data = FlatVector::GetData<timestamp_t>(output.data[0]);
			if (shard.exact_count) {
				while (state.shard_row < shard.row_count && size < STANDARD_VECTOR_SIZE) {
					output_data[size++] = timestamp_t(GetDistributedSequenceValue(
					    bind_data.start.value, bind_data.increment.micros, shard.row_offset + state.shard_row));
					state.shard_row++;
				}
			} else {
				while (!state.Finished(state.current_state) && size < STANDARD_VECTOR_SIZE) {
					output_data[size++] = state.current_state;
					state.current_state = ICUDateFunc::Add(calendar, state.current_state, state.increment);
				}
			}

			const bool finished =
			    shard.exact_count ? state.shard_row == shard.row_count : state.Finished(state.current_state);
			if (finished) {
				state.shard_index++;
				state.initialized_row = false;
			}
			if (size > 0) {
				output.SetCardinality(size);
				return OperatorResultType::HAVE_MORE_OUTPUT;
			}
		}
		output.SetCardinality(0);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	template <bool GENERATE_SERIES>
	static OperatorResultType Function(ExecutionContext &, TableFunctionInput &data_p, DataChunk &input,
	                                   DataChunk &output) {
		auto &bind_data = data_p.bind_data->Cast<ICURangeBindData>();
		auto &state = data_p.local_state->Cast<ICURangeLocalState>();
		if (bind_data.distributed) {
			return DistributedFunction(bind_data, state, output);
		}
		TZCalendar calendar(*bind_data.calendar, bind_data.cal_setting);
		while (true) {
			if (!state.initialized_row) {
				if (state.current_input_row >= input.size()) {
					state.current_input_row = 0;
					state.initialized_row = false;
					return OperatorResultType::NEED_MORE_INPUT;
				}
				GenerateRangeDateTimeParameters<GENERATE_SERIES>(input, state.current_input_row, state);
				state.initialized_row = true;
				state.current_state = state.start;
			}
			if (state.empty_range) {
				output.SetCardinality(0);
				state.current_input_row++;
				state.initialized_row = false;
				return OperatorResultType::HAVE_MORE_OUTPUT;
			}
			idx_t size = 0;
			auto output_data = FlatVector::GetData<timestamp_t>(output.data[0]);
			while (!state.Finished(state.current_state) && size < STANDARD_VECTOR_SIZE) {
				output_data[size++] = state.current_state;
				state.current_state = ICUDateFunc::Add(calendar, state.current_state, state.increment);
			}
			if (size == 0) {
				state.current_input_row++;
				state.initialized_row = false;
				continue;
			}
			output.SetCardinality(size);
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
	}

	static void Serialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p, const TableFunction &) {
		if (!bind_data_p) {
			throw SerializationException("ICU sequence function requires bind data");
		}
		auto &bind_data = bind_data_p->Cast<ICURangeBindData>();
		serializer.WriteProperty(101, "timezone", bind_data.tz_setting);
		serializer.WriteProperty(102, "calendar", bind_data.cal_setting);
		serializer.WriteProperty(103, "has_parameters", bind_data.has_parameters);
		serializer.WriteProperty(104, "is_null", bind_data.is_null);
		serializer.WriteProperty(105, "generate_series", bind_data.generate_series);
		serializer.WriteProperty(106, "distributed", bind_data.distributed);
		serializer.WriteProperty(107, "start", bind_data.start.value);
		serializer.WriteProperty(108, "end", bind_data.end.value);
		serializer.WriteProperty(109, "increment_months", bind_data.increment.months);
		serializer.WriteProperty(110, "increment_days", bind_data.increment.days);
		serializer.WriteProperty(111, "increment_micros", bind_data.increment.micros);
	}

	static bool IsGenerateSeries(const TableFunction &function) {
		if (function.name == "generate_series") {
			return true;
		}
		if (function.name == "range") {
			return false;
		}
		throw SerializationException("unexpected ICU sequence table function '%s'", function.name);
	}

	static unique_ptr<FunctionData> Deserialize(Deserializer &deserializer, TableFunction &function) {
		auto result = make_uniq<ICURangeBindData>();
		result->tz_setting = deserializer.ReadProperty<string>(101, "timezone");
		result->cal_setting = deserializer.ReadProperty<string>(102, "calendar");
		result->has_parameters = deserializer.ReadProperty<bool>(103, "has_parameters");
		result->is_null = deserializer.ReadProperty<bool>(104, "is_null");
		result->generate_series = deserializer.ReadProperty<bool>(105, "generate_series");
		result->distributed = deserializer.ReadProperty<bool>(106, "distributed");
		result->start = timestamp_t(deserializer.ReadProperty<int64_t>(107, "start"));
		result->end = timestamp_t(deserializer.ReadProperty<int64_t>(108, "end"));
		result->increment.months = deserializer.ReadProperty<int32_t>(109, "increment_months");
		result->increment.days = deserializer.ReadProperty<int32_t>(110, "increment_days");
		result->increment.micros = deserializer.ReadProperty<int64_t>(111, "increment_micros");
		if (result->generate_series != IsGenerateSeries(function)) {
			throw SerializationException("serialized ICU sequence bind does not match function '%s'", function.name);
		}
		if (result->is_null && !result->has_parameters) {
			throw SerializationException("serialized ICU sequence has null state without bound parameters");
		}
		if (result->distributed && !result->has_parameters) {
			throw SerializationException("serialized distributed ICU sequence has no bound parameters");
		}
		result->calendar = CreateCalendar(result->tz_setting, result->cal_setting);
		if (result->has_parameters) {
			if (result->is_null) {
				result->exact_count = true;
				result->empty_range = true;
			} else {
				result->InitializeSequence();
			}
		}
		return std::move(result);
	}

	static vector<DistributedScanTask> Plan(const TableFunctionDistributedScanInput &input) {
		auto &bind_data = input.bind_data.Cast<ICURangeBindData>();
		if (!bind_data.has_parameters) {
			throw InvalidInputException("distributed ICU range requires scalar bound parameters");
		}
		return PlanDistributedSequenceTasks(bind_data.cardinality, bind_data.exact_count, input.target_task_count);
	}

	static unique_ptr<FunctionData> CreateWorkerBind(const TableFunctionDistributedScanInput &input) {
		auto &source_bind = input.bind_data.Cast<ICURangeBindData>();
		if (!source_bind.has_parameters) {
			throw InvalidInputException("distributed ICU range requires scalar bound parameters");
		}
		auto result = make_uniq<ICURangeBindData>(source_bind);
		result->distributed = true;
		result->shards.clear();
		return std::move(result);
	}

	static void ApplyTasks(FunctionData &worker_bind_data, const vector<DistributedScanTask> &tasks) {
		auto &bind_data = worker_bind_data.Cast<ICURangeBindData>();
		if (!bind_data.distributed || !bind_data.has_parameters) {
			throw InvalidInputException("distributed ICU range tasks require a worker bind");
		}
		bind_data.shards = DecodeDistributedSequenceTasks(tasks, bind_data.cardinality, bind_data.exact_count);
	}

	static TableFunctionDistributedScanCallbacks DistributedCallbacks() {
		TableFunctionDistributedScanCallbacks callbacks;
		callbacks.protocol_version = DISTRIBUTED_SEQUENCE_PROTOCOL_VERSION;
		callbacks.task_codec = {DISTRIBUTED_SEQUENCE_TASK_CODEC, DISTRIBUTED_SEQUENCE_TASK_CODEC_VERSION};
		callbacks.plan = Plan;
		callbacks.create_worker_bind = CreateWorkerBind;
		callbacks.apply_tasks = ApplyTasks;
		return callbacks;
	}

	template <bool GENERATE_SERIES>
	static TableFunction CreateFunction() {
		const string name = GENERATE_SERIES ? "generate_series" : "range";
		TableFunction result(name, {LogicalType::TIMESTAMP_TZ, LogicalType::TIMESTAMP_TZ, LogicalType::INTERVAL},
		                     nullptr, Bind<GENERATE_SERIES>, nullptr, LocalInit);
		result.in_out_function = Function<GENERATE_SERIES>;
		result.cardinality = Cardinality;
		result.serialize = Serialize;
		result.deserialize = Deserialize;
		result.SetDistributedScanCallbacks(DistributedCallbacks());
		return result;
	}

	static void AddICUTableRangeFunction(ExtensionLoader &loader) {
		TableFunctionSet range("range");
		range.AddFunction(CreateFunction<false>());
		loader.RegisterFunction(range);

		TableFunctionSet generate_series("generate_series");
		generate_series.AddFunction(CreateFunction<true>());
		loader.RegisterFunction(generate_series);
	}
};

void RegisterICUTableRangeFunctions(ExtensionLoader &loader) {
	ICUTableRange::AddICUTableRangeFunction(loader);
}

} // namespace duckdb
