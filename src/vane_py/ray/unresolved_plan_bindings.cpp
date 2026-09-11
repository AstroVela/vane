// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Included by ray_module.cpp after the bound-plan transport definitions.

struct UnresolvedResultSchema {
	vector<string> names;
	vector<LogicalType> types;
	ClientProperties properties;

	void Serialize(Serializer &serializer) const {
		serializer.WriteProperty(100, "names", names);
		serializer.WriteProperty(101, "types", types);
		serializer.WriteProperty(102, "time_zone", properties.time_zone);
		serializer.WriteProperty(103, "arrow_offset_size", properties.arrow_offset_size);
		serializer.WriteProperty(104, "arrow_use_list_view", properties.arrow_use_list_view);
		serializer.WriteProperty(105, "produce_arrow_string_view", properties.produce_arrow_string_view);
		serializer.WriteProperty(106, "arrow_lossless_conversion", properties.arrow_lossless_conversion);
		serializer.WriteProperty(107, "arrow_output_version", properties.arrow_output_version);
	}
	static UnresolvedResultSchema Deserialize(Deserializer &deserializer) {
		UnresolvedResultSchema result;
		deserializer.ReadProperty(100, "names", result.names);
		deserializer.ReadProperty(101, "types", result.types);
		deserializer.ReadProperty(102, "time_zone", result.properties.time_zone);
		deserializer.ReadProperty(103, "arrow_offset_size", result.properties.arrow_offset_size);
		deserializer.ReadProperty(104, "arrow_use_list_view", result.properties.arrow_use_list_view);
		deserializer.ReadProperty(105, "produce_arrow_string_view", result.properties.produce_arrow_string_view);
		deserializer.ReadProperty(106, "arrow_lossless_conversion", result.properties.arrow_lossless_conversion);
		deserializer.ReadProperty(107, "arrow_output_version", result.properties.arrow_output_version);
		if (result.names.size() != result.types.size()) {
			throw SerializationException("Unresolved result schema has mismatched names and types");
		}
		return result;
	}
};

template <class T>
static py::bytes SerializeUnresolvedValue(const T &value) {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer::Serialize(value, stream);
	return py::bytes(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

template <class T>
static auto DeserializeUnresolvedValue(const py::bytes &payload)
    -> decltype(T::Deserialize(std::declval<Deserializer &>())) {
	string bytes = payload;
	MemoryStream stream(Allocator::DefaultAllocator());
	stream.WriteData(reinterpret_cast<const_data_ptr_t>(bytes.data()), bytes.size());
	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto result = T::Deserialize(deserializer);
	deserializer.End();
	if (stream.GetPosition() != bytes.size()) {
		throw SerializationException("Unresolved request contains trailing bytes");
	}
	return result;
}

class DriverContextExpressions : public LogicalOperatorVisitor {
public:
	explicit DriverContextExpressions(ClientContext &context) : context(context) {
	}
	bool context_read = false;
	bool data_read = false;
	bool write = false;

	void VisitOperator(LogicalOperator &op) override {
		if (op.type == LogicalOperatorType::LOGICAL_GET) {
			auto &get = op.Cast<LogicalGet>();
			context_read |= get.function.RequiresClientContext();
			data_read |= !get.function.RequiresClientContext();
		}
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_INSERT:
		case LogicalOperatorType::LOGICAL_UPDATE:
		case LogicalOperatorType::LOGICAL_DELETE:
		case LogicalOperatorType::LOGICAL_MERGE_INTO:
		case LogicalOperatorType::LOGICAL_CREATE_TABLE:
		case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
		case LogicalOperatorType::LOGICAL_DATA_SINK:
			write = true;
			break;
		default:
			break;
		}
		for (auto &child : op.children) {
			VisitOperator(*child);
		}
		VisitOperatorExpressions(op);
	}

private:
	ClientContext &context;

	unique_ptr<Expression> VisitReplace(BoundFunctionExpression &expression, unique_ptr<Expression> *) override {
		auto lambda = dynamic_cast<ListLambdaBindData *>(expression.bind_info.get());
		if (lambda && lambda->lambda_expr) {
			VisitExpression(&lambda->lambda_expr);
		}
		if (!expression.function.RequiresClientContext()) {
			return nullptr;
		}
		context_read = true;
		static const case_insensitive_set_t snapshot_functions {"current_query",
		                                                        "current_schema",
		                                                        "current_database",
		                                                        "current_catalog",
		                                                        "current_schemas",
		                                                        "in_search_path",
		                                                        "txid_current",
		                                                        "current_connection_id",
		                                                        "current_query_id",
		                                                        "current_transaction_id",
		                                                        "now",
		                                                        "current_timestamp",
		                                                        "transaction_timestamp",
		                                                        "current_date",
		                                                        "current_time",
		                                                        "today",
		                                                        "localtime",
		                                                        "localtimestamp"};
		if (!snapshot_functions.count(expression.function.name)) {
			return nullptr;
		}
		for (auto &child : expression.children) {
			if (!child->IsFoldable()) {
				return nullptr;
			}
		}
		auto value = ExpressionExecutor::EvaluateScalar(context, expression, true);
		auto result = make_uniq<BoundConstantExpression>(std::move(value));
		result->alias = expression.alias;
		result->query_location = expression.query_location;
		return std::move(result);
	}
};

static py::dict PrepareUnresolvedExecution(py::object connection, py::object input, py::object parameters,
                                           const string &query_id, const string &session_id,
                                           const py::dict &session_config) {
	auto &wrapper = ExtractPyConnectionWrapper(connection);
	auto context = wrapper.con.GetConnection().context;
	if (!context->vane_driver_session || context->vane_runner_type != "ray") {
		throw InvalidInputException("Unresolved execution requires a driver-owned Ray connection");
	}
	unique_ptr<SQLStatement> statement;
	shared_ptr<Relation> relation;
	if (py::isinstance<DuckDBPyRelation>(input)) {
		relation = input.cast<DuckDBPyRelation &>().GetRelation();
		if (relation->context->GetContext() != context) {
			throw InvalidInputException("Unresolved Relation must belong to the driver connection");
		}
	} else {
		statement = input.cast<DuckDBPyStatement &>().GetStatement();
		ValidateRunnerStatement(*statement);
	}
	case_insensitive_map_t<BoundParameterData> converted;
	if (!parameters.is_none()) {
		if (py::is_dict_like(parameters)) {
			converted = DuckDBPyConnection::TransformPythonParamDict(parameters.cast<py::dict>());
		} else if (py::is_list_like(parameters)) {
			auto values = DuckDBPyConnection::TransformPythonParamList(parameters);
			for (idx_t i = 0; i < values.size(); i++) {
				converted[std::to_string(i + 1)] = BoundParameterData(std::move(values[i]));
			}
		} else {
			throw InvalidInputException("Prepared parameters can only be passed as a list or a dictionary");
		}
	}
	unique_ptr<RunnerBoundPlan> bound;
	UnresolvedResultSchema schema;
	PendingQueryParameters options;
	options.parameters = &converted;
	options.bound_plan_handler = [&](Planner &planner, unique_ptr<LogicalOperator> &plan,
	                                 PreparedStatementData &prepared) {
		schema = {prepared.names, prepared.types, context->GetClientProperties()};
		DriverContextExpressions expressions(*context);
		expressions.VisitOperator(*plan);
		if (prepared.statement_type != StatementType::CALL_STATEMENT && expressions.context_read &&
		    !expressions.data_read && !expressions.write) {
			return false;
		}
		bound = AdmitRunnerBoundPlan(planner, plan, prepared, converted, RunnerPlanAdmission::DRIVER);
		return bool(bound);
	};
	py::dict result;
	unique_ptr<PendingQueryResult> pending;
	try {
		{
			py::gil_scoped_release release;
			pending = statement ? context->PendingQuery(std::move(statement), options)
			                    : context->PendingQuery(relation, options);
			if (pending && pending->HasError()) {
				pending->ThrowError();
			}
		}
		if (pending) {
			unique_ptr<QueryResult> native;
			{
				py::gil_scoped_release release;
				native = pending->Execute();
				if (native->HasError()) {
					native->ThrowError();
				}
			}
			schema = {native->names, native->types, native->client_properties};
			result["kind"] = "native";
			result["result"] =
			    py::cast(make_uniq<DuckDBPyRelation>(make_shared_ptr<DuckDBPyResult>(std::move(native))));
		} else {
			if (!bound) {
				throw InternalException("Unresolved query produced neither a result nor a bound plan");
			}
			auto transport = SerializeRunnerBoundPlan(*bound, connection);
			auto &plan = transport.cast<PyLogicalPlan &>();
			plan.query_id_ = query_id;
			auto snapshot = plan.connection_snapshot_.cast<py::dict>();
			py::dict session;
			session["id"] = session_id;
			session["config"] = session_config;
			snapshot["vane_session"] = std::move(session);
			result["kind"] = bound->kind == RunnerPlanKind::READ        ? "read"
			                 : bound->kind == RunnerPlanKind::DATA_SINK ? "datasink"
			                                                            : "write";
			result["plan"] = std::move(transport);
			{
				py::gil_scoped_release release;
				context->CancelBoundPlan(bound->query_number);
			}
			bound.reset();
		}
		result["schema"] = SerializeUnresolvedValue(schema);
		return result;
	} catch (...) {
		if (bound) {
			py::gil_scoped_release release;
			try {
				context->CancelBoundPlan(bound->query_number);
			} catch (...) {
			}
		}
		throw;
	}
}

static void RegisterUnresolvedBindings(py::module_ &m) {
	m.def("_unresolved_update_relation", &DuckDBPyRelation::BuildUpdate, py::arg("relation"), py::arg("set"),
	      py::kw_only(), py::arg("condition") = py::none());
	m.def("_unresolved_delete_relation", &DuckDBPyRelation::BuildDelete, py::arg("relation"), py::kw_only(),
	      py::arg("condition") = py::none());
	m.def("_unresolved_merge_relation", &DuckDBPyRelation::BuildMergeInto, py::arg("relation"), py::arg("target_table"),
	      py::arg("condition"), py::arg("when_clauses"), py::kw_only(), py::arg("target_alias") = "target",
	      py::arg("source_alias") = "source");
	m.def("_unresolved_insert_relation", &DuckDBPyRelation::BuildInsertInto, py::arg("relation"),
	      py::arg("table_name"));
	m.def("_unresolved_create_relation", &DuckDBPyRelation::BuildCreate, py::arg("relation"), py::arg("table_name"),
	      py::kw_only(), py::arg("properties") = py::none(), py::arg("partition_by") = py::none());
	m.def("_unresolved_parquet_relation", &DuckDBPyRelation::BuildToParquet,
	      "Write the relation object to a Parquet file in 'file_name'", py::arg("relation"), py::arg("file_name"),
	      py::kw_only(), py::arg("compression") = py::none(), py::arg("field_ids") = py::none(),
	      py::arg("row_group_size_bytes") = py::none(), py::arg("row_group_size") = py::none(),
	      py::arg("overwrite") = py::none(), py::arg("per_thread_output") = py::none(),
	      py::arg("use_tmp_file") = py::none(), py::arg("partition_by") = py::none(),
	      py::arg("write_partition_columns") = py::none(), py::arg("append") = py::none(),
	      py::arg("filename_pattern") = py::none(), py::arg("file_size_bytes") = py::none());

	m.def("_unresolved_csv_relation", &DuckDBPyRelation::BuildToCSV,
	      "Write the relation object to a CSV file in 'file_name'", py::arg("relation"), py::arg("file_name"),
	      py::kw_only(), py::arg("sep") = py::none(), py::arg("na_rep") = py::none(), py::arg("header") = py::none(),
	      py::arg("quotechar") = py::none(), py::arg("escapechar") = py::none(), py::arg("date_format") = py::none(),
	      py::arg("timestamp_format") = py::none(), py::arg("quoting") = py::none(), py::arg("encoding") = py::none(),
	      py::arg("compression") = py::none(), py::arg("overwrite") = py::none(),
	      py::arg("per_thread_output") = py::none(), py::arg("use_tmp_file") = py::none(),
	      py::arg("partition_by") = py::none(), py::arg("write_partition_columns") = py::none());

	m.def("_unresolved_file_relation", &DuckDBPyRelation::BuildToFile,
	      "Write the relation object with a registered DuckDB COPY format", py::arg("relation"), py::arg("file_name"),
	      py::kw_only(), py::arg("format"));

	m.def("_serialize_unresolved_expression", [](const DuckDBPyExpression &expression) {
		return py::make_tuple(SerializeUnresolvedValue(expression.GetExpression()), int(expression.order_type),
		                      int(expression.null_order));
	});
	m.def("_deserialize_unresolved_expression", [](const py::bytes &payload, int order, int null_order) {
		return make_shared_ptr<DuckDBPyExpression>(DeserializeUnresolvedValue<ParsedExpression>(payload),
		                                           OrderType(order), OrderByNullType(null_order));
	});
	m.def("_serialize_unresolved_type", [](const DuckDBPyType &type) { return SerializeUnresolvedValue(type.Type()); });
	m.def("_deserialize_unresolved_type", [](const py::bytes &payload) {
		return make_shared_ptr<DuckDBPyType>(DeserializeUnresolvedValue<LogicalType>(payload));
	});
	m.def("_unresolved_relation_schema", [](DuckDBPyRelation &relation) {
		UnresolvedResultSchema schema;
		schema.properties = relation.GetRelation()->context->GetContext()->GetClientProperties();
		for (auto &column : relation.GetRelation()->Columns()) {
			schema.names.push_back(column.Name());
			schema.types.push_back(column.Type());
		}
		return SerializeUnresolvedValue(schema);
	});
	m.def("_prepare_unresolved_execution", &PrepareUnresolvedExecution);
	m.def("_make_unresolved_result", [](py::object connection, py::object iterator, const py::bytes &payload) {
		auto &wrapper = ExtractPyConnectionWrapper(connection);
		auto schema = DeserializeUnresolvedValue<UnresolvedResultSchema>(payload);
		auto result = make_shared_ptr<DuckDBPyResult>(MakeDistributedArrowPyResultSource(
		    std::move(iterator), py::none(), false, false, std::move(schema.names), std::move(schema.types),
		    wrapper.con.GetConnection().context, connection, &schema.properties));
		return make_uniq<DuckDBPyRelation>(std::move(result));
	});
}
