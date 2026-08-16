// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Included by ray_module.cpp inside namespace duckdb.

static void AssignDataSourceQueryOwner(duckdb::PhysicalOperator &op, const string &query_id,
                                       bool allow_unacquired_rebind = false) {
	if (op.type == duckdb::PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<duckdb::PhysicalTableScan>();
		auto *bind_data = dynamic_cast<duckdb::DataSourceScanBindData *>(scan.bind_data.get());
		if (bind_data) {
			if (query_id.empty()) {
				throw duckdb::InternalException("Datasource scan requires a non-empty resource query owner");
			}
			if (!bind_data->query_id.empty() && bind_data->query_id != query_id) {
				if (!allow_unacquired_rebind) {
					throw duckdb::InvalidInputException(
					    "Datasource scan query ownership mismatch: existing='%s', requested='%s'", bind_data->query_id,
					    query_id);
				}
			}
			bind_data->query_id = query_id;
		}
	}
	for (auto &child : op.children) {
		AssignDataSourceQueryOwner(child.get(), query_id, allow_unacquired_rebind);
	}
}

static std::shared_ptr<duckdb::distributed::FlightExchangeManager>
FindFlightExchangeSinkManager(const duckdb::PhysicalOperator &op) {
	if (op.type == duckdb::PhysicalOperatorType::EXCHANGE_SINK) {
		auto *sink = dynamic_cast<const duckdb::PhysicalRemoteExchangeSink *>(&op);
		if (sink) {
			return std::dynamic_pointer_cast<duckdb::distributed::FlightExchangeManager>(sink->GetExchangeManager());
		}
	}
	for (auto &child : op.children) {
		auto manager = FindFlightExchangeSinkManager(child.get());
		if (manager) {
			return manager;
		}
	}
	return nullptr;
}

static const duckdb::PhysicalRemoteExchangeSink *FindRemoteExchangeSinkOperator(const duckdb::PhysicalOperator &op) {
	if (op.type == duckdb::PhysicalOperatorType::EXCHANGE_SINK) {
		return dynamic_cast<const duckdb::PhysicalRemoteExchangeSink *>(&op);
	}
	for (auto &child : op.children) {
		if (auto *sink = FindRemoteExchangeSinkOperator(child.get())) {
			return sink;
		}
	}
	return nullptr;
}

struct PyPhysicalPlanWrapper {
	static constexpr uint64_t INIT_MAGIC = 0x445046504C414E31ULL;
	uint64_t init_magic_;
	// CRITICAL: Field order matters for destruction!
	// In C++, members are destroyed in REVERSE order of declaration.
	// worker_connection_ MUST be declared FIRST so it is destroyed LAST,
	// after plan_ is destroyed. This ensures the allocator stays valid.
	py::object worker_connection_; // Keep DuckDB connection alive for worker-side deserialized plans
	duckdb::shared_ptr<duckdb::ClientContext>
	    client_context_; // Keep driver-side context alive when built from relation
	string query_id_;
	string resource_query_id_;
	std::shared_ptr<duckdb::distributed::DistributedPhysicalPlan> plan_;
	py::object arrow_schema_;        // Arrow schema for type information
	py::object udf_registrations_;   // Connection-local Python UDF registrations captured from the source relation
	py::object udf_actor_handles_;   // Node-id keyed actor handles for UDF worker execution
	py::object connection_snapshot_; // Connection settings/extensions snapshot captured from the source relation
	vector<ScopedSecretRef> scoped_secret_refs_; // Opaque scoped secret references; never secret values
	string serialized_root_;                     // Deferred serialized PhysicalOperator bytes (for pickle round-trips)

	bool has_root() const {
		return plan_ && plan_->physical_plan() && plan_->physical_plan()->HasRoot();
	}

	void ensure_connection_snapshot(py::object conn_obj, bool apply_session_config = true) const {
		if (connection_snapshot_.is_none()) {
			return;
		}
		ConnectionSnapshotApplyOptions options;
		options.apply_session_config = apply_session_config;
		ApplyConnectionSnapshot(conn_obj, connection_snapshot_, options);
	}

	py::object resolve_execution_connection(py::object conn_obj) const {
		if (!worker_connection_.is_none()) {
			auto &worker_connection = ExtractPyConnectionWrapper(worker_connection_);
			if (worker_connection.con.ConnectionIsClosed()) {
				throw py::value_error("DistributedPhysicalPlan binding connection is closed");
			}
			return worker_connection_;
		}
		return ResolveConnectionForSnapshot(conn_obj, connection_snapshot_);
	}

	void apply_udf_actor_handles() {
		if (udf_actor_handles_.is_none() || !has_root()) {
			return;
		}
		if (!py::isinstance<py::dict>(udf_actor_handles_)) {
			throw duckdb::InternalException("udf_actor_handles must be a dict keyed by node_id");
		}
		auto handles_map = udf_actor_handles_.cast<py::dict>();
		if (handles_map.empty()) {
			return;
		}

		auto physical_plan = plan_->physical_plan();
		idx_t node_counter = 0;
		std::function<void(PhysicalOperator &)> inject = [&](PhysicalOperator &op) -> void {
			vector<UDFFunctionData *> bind_nodes;
			CollectMutableUDFBindData(op, bind_nodes);
			for (auto *bind_data : bind_nodes) {
				py::str key(std::to_string(node_counter++));
				if (handles_map.contains(key)) {
					py::object handles_obj = handles_map[key];
					bind_data->actor_handles = WrapPyObjectForUDFActorHandles(handles_obj);
				}
			}
			for (auto &child : op.children) {
				inject(child.get());
			}
		};
		inject(physical_plan->Root());
	}

	void ensure_plan_identity() {
		auto effective_query_id = query_id_;
		if (effective_query_id.empty() && plan_) {
			effective_query_id = plan_->query_id();
		}
		if (effective_query_id.empty()) {
			return;
		}

		if (plan_ && plan_->query_id() == effective_query_id) {
			return;
		}

		uint16_t idx = duckdb::distributed::get_query_idx_counter().fetch_add(1);
		auto cfg = std::make_shared<duckdb::distributed::DuckDBExecutionConfig>(
		    duckdb::distributed::DuckDBExecutionConfig::from_env());
		duckdb::Allocator &alloc = duckdb::Allocator::DefaultAllocator();
		auto physical_plan = std::make_shared<duckdb::PhysicalPlan>(alloc);

		if (plan_) {
			idx = plan_->idx();
			if (plan_->execution_config()) {
				cfg = plan_->execution_config();
			}
			if (plan_->physical_plan()) {
				physical_plan = plan_->physical_plan();
			}
		}

		plan_ =
		    std::make_shared<duckdb::distributed::DistributedPhysicalPlan>(idx, effective_query_id, physical_plan, cfg);
		if (physical_plan && physical_plan->HasRoot()) {
			AssignDataSourceQueryOwner(physical_plan->Root(), resource_query_id_);
		}
	}

	// Materialize deferred serialized_root_ into the plan's PhysicalPlan.
	// Requires a DuckDB connection for deserialization context.
	void materialize_deferred_root(py::object conn_obj, bool apply_session_config = true) {
		if (serialized_root_.empty())
			return;
		if (has_root())
			return; // Already has a root

		ensure_plan_identity();
		ensure_connection_snapshot(conn_obj, apply_session_config);

		auto &py_conn = ExtractPyConnectionWrapper(conn_obj);
		auto &db_conn = py_conn.con.GetConnection();
		auto &context = *db_conn.context;
		auto &db = duckdb::DatabaseInstance::GetDatabase(context);

		auto physical_plan = plan_->physical_plan();
		if (!physical_plan) {
			physical_plan = std::make_shared<duckdb::PhysicalPlan>(duckdb::Allocator::Get(context));
		}

		duckdb::MemoryStream stream(reinterpret_cast<duckdb::data_ptr_t>(const_cast<char *>(serialized_root_.data())),
		                            serialized_root_.size());
		duckdb::BinaryDeserializer deserializer(stream);
		deserializer.Set<duckdb::DatabaseInstance &>(db);
		deserializer.Set<duckdb::ClientContext &>(context);

		duckdb::unique_ptr<duckdb::PhysicalOperator> root_op;
		context.RunFunctionInTransaction([&]() {
			deserializer.Begin();
			root_op = physical_plan->Deserialize(deserializer);
			deserializer.End();
		});

		if (root_op) {
			auto *root_ptr = root_op.get();
			physical_plan->TakeOwnership(std::move(root_op));
			physical_plan->SetRoot(*root_ptr);
			// The root was just deserialized and cannot have acquired a process-
			// local datasource factory yet, so replacing its serialized source
			// plan owner with the explicit resource query owner is safe.
			AssignDataSourceQueryOwner(*root_ptr, resource_query_id_, true);
		}
		// Keep both the Python connection and its context alive to ensure allocator validity.
		// The parent connection may close this cursor while the physical plan is still referenced.
		worker_connection_ = conn_obj;
		client_context_ = db_conn.context;
		serialized_root_.clear();
	}

	string serialize_root_for_clone() const {
		if (!has_root()) {
			if (!serialized_root_.empty()) {
				return serialized_root_;
			}
			throw duckdb::InternalException("DistributedPhysicalPlan has no root to clone");
		}
		auto physical_plan = plan_->physical_plan();
		if (!physical_plan || !physical_plan->HasRoot()) {
			throw duckdb::InternalException("DistributedPhysicalPlan physical plan has no root to clone");
		}
		return serialize_operator_root(physical_plan->Root());
	}

	void validate_serializable_for_submission() const {
		string scoped_secret_session_id;
		if (!scoped_secret_refs_.empty()) {
			scoped_secret_session_id = VaneSessionIdFromSnapshot(connection_snapshot_);
		}
		ValidateScopedSecretRefs(scoped_secret_refs_, resource_query_id_, scoped_secret_session_id,
		                         "physical plan submission");
		if (!has_root()) {
			(void)serialize_root_for_clone();
			return;
		}

		auto physical_plan = plan_->physical_plan();
		if (!physical_plan || !physical_plan->HasRoot()) {
			throw duckdb::InternalException("DistributedPhysicalPlan physical plan has no root to submit");
		}
		auto &root = physical_plan->Root();
		if (!root.GetExtensionWriteTaskProvider()) {
			(void)serialize_operator_root(root);
			return;
		}
		if (root.type != duckdb::PhysicalOperatorType::EXTENSION) {
			throw duckdb::InvalidInputException(
			    "Distributed extension write provider must be exposed by an EXTENSION physical root");
		}
		if (root.children.size() != 1) {
			throw duckdb::InvalidInputException(
			    "Distributed extension write requires exactly one worker-executable child");
		}

		// The extension root owns coordinator-only catalog state and never crosses
		// the worker boundary. Validate exactly the subtree that task production
		// can submit; requiring the extension root itself to serialize would make
		// the generic protocol depend on an irrelevant extension capability.
		(void)serialize_operator_root(root.children[0].get());
	}

	static string serialize_operator_root(const duckdb::PhysicalOperator &root) {
		duckdb::MemoryStream stream;
		duckdb::SerializationOptions options;
		options.serialization_compatibility = duckdb::SerializationCompatibility::Latest();
		options.serialize_default_values = true;
		duckdb::BinarySerializer serializer(stream, options);
		serializer.Begin();
		root.Serialize(serializer);
		serializer.End();
		auto data_size = stream.GetPosition();
		if (data_size == 0) {
			throw duckdb::InternalException("DistributedPhysicalPlan serialized a physical root to an empty payload");
		}
		auto data_ptr = stream.GetData();
		return string(reinterpret_cast<const char *>(data_ptr), data_size);
	}

	PyPhysicalPlanWrapper clone(py::object conn_obj) const {
		PyPhysicalPlanWrapper result;
		result.query_id_ = idx();
		result.resource_query_id_ = resource_query_id_;
		result.udf_registrations_ = udf_registrations_;
		result.udf_actor_handles_ = udf_actor_handles_;
		result.connection_snapshot_ = connection_snapshot_;
		result.scoped_secret_refs_ = scoped_secret_refs_;
		result.serialized_root_ = serialize_root_for_clone();
		result.ensure_plan_identity();
		if (!conn_obj.is_none()) {
			result.materialize_deferred_root(conn_obj);
		}
		return result;
	}

	PyPhysicalPlanWrapper()
	    : init_magic_(INIT_MAGIC), worker_connection_(py::none()), client_context_(nullptr), query_id_(),
	      resource_query_id_(), plan_(nullptr), arrow_schema_(py::none()), udf_registrations_(py::none()),
	      udf_actor_handles_(py::none()), connection_snapshot_(py::none()), scoped_secret_refs_(), serialized_root_() {
		// Create a minimal placeholder DistributedPhysicalPlan directly (avoid using from_logical_plan_builder).
		try {
			uint16_t idx = duckdb::distributed::get_query_idx_counter().fetch_add(1);
			// Default execution config from environment
			auto cfg = std::make_shared<duckdb::distributed::DuckDBExecutionConfig>(
			    duckdb::distributed::DuckDBExecutionConfig::from_env());
			duckdb::Allocator &alloc = duckdb::Allocator::DefaultAllocator();
			auto physical_plan = std::make_shared<duckdb::PhysicalPlan>(alloc);
			plan_ =
			    std::make_shared<duckdb::distributed::DistributedPhysicalPlan>(idx, string("py"), physical_plan, cfg);
		} catch (const std::exception &ex) {
			throw duckdb::InternalException(string("Failed to construct default DistributedPhysicalPlan: ") +
			                                ex.what());
		}
	}
	explicit PyPhysicalPlanWrapper(std::shared_ptr<duckdb::distributed::DistributedPhysicalPlan> p)
	    : init_magic_(INIT_MAGIC), worker_connection_(py::none()), client_context_(nullptr),
	      query_id_(p ? p->query_id() : string()), resource_query_id_(query_id_), plan_(std::move(p)),
	      arrow_schema_(py::none()), udf_registrations_(py::none()), udf_actor_handles_(py::none()),
	      connection_snapshot_(py::none()), scoped_secret_refs_(), serialized_root_() {
		if (plan_ && plan_->physical_plan() && plan_->physical_plan()->HasRoot()) {
			AssignDataSourceQueryOwner(plan_->physical_plan()->Root(), resource_query_id_);
		}
	}

	PyPhysicalPlanWrapper(std::shared_ptr<duckdb::distributed::DistributedPhysicalPlan> p, string resource_query_id)
	    : init_magic_(INIT_MAGIC), worker_connection_(py::none()), client_context_(nullptr),
	      query_id_(p ? p->query_id() : string()), resource_query_id_(std::move(resource_query_id)),
	      plan_(std::move(p)), arrow_schema_(py::none()), udf_registrations_(py::none()),
	      udf_actor_handles_(py::none()), connection_snapshot_(py::none()), scoped_secret_refs_(), serialized_root_() {
		if (resource_query_id_.empty()) {
			throw duckdb::InternalException("DistributedPhysicalPlan requires a non-empty resource_query_id");
		}
		if (plan_ && plan_->physical_plan() && plan_->physical_plan()->HasRoot()) {
			// WorkerTask plans are unexecuted copies. Their serialized source
			// plan ID may differ from the resource query that owns teardown.
			AssignDataSourceQueryOwner(plan_->physical_plan()->Root(), resource_query_id_, true);
		}
	}

	bool IsInitialized() const {
		return init_magic_ == INIT_MAGIC;
	}

	string idx() const {
		if (!IsInitialized()) {
			return string();
		}
		if (!query_id_.empty()) {
			return query_id_;
		}
		if (plan_) {
			auto qid = plan_->query_id();
			if (!qid.empty()) {
				return qid;
			}
		}
		return query_id_;
	}

	string resource_query_id() const {
		if (!IsInitialized()) {
			return string();
		}
		return resource_query_id_;
	}

	string session_id() const {
		return VaneSessionIdFromSnapshot(connection_snapshot_);
	}

	py::dict session_config() const {
		return VaneSessionConfigFromSnapshot(connection_snapshot_);
	}

	bool has_explicit_s3_credentials() const {
		return HasExplicitS3CredentialsFromSnapshot(connection_snapshot_);
	}

	py::list scoped_secret_refs() const {
		return DescribeScopedSecretRefs(scoped_secret_refs_);
	}

	size_t num_partitions() const {
		if (!IsInitialized()) {
			throw duckdb::InternalException("DistributedPhysicalPlan is uninitialized");
		}
		if (!plan_) {
			throw duckdb::InternalException("DistributedPhysicalPlan is not initialized");
		}
		auto pipeline_node = BuildDistributedPipelineNode(plan_, client_context_.get());
		return pipeline_node->num_partitions();
	}

	static std::pair<bool, string> UDFPayloadStringField(const Value &payload, const string &name) {
		if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
			return std::make_pair(false, string());
		}
		auto &children = StructValue::GetChildren(payload);
		auto child_count = StructType::GetChildCount(payload.type());
		for (idx_t i = 0; i < child_count; i++) {
			if (StructType::GetChildName(payload.type(), i) != name) {
				continue;
			}
			if (i >= children.size() || children[i].IsNull()) {
				return std::make_pair(false, string());
			}
			auto &child_type = StructType::GetChildType(payload.type(), i);
			if (child_type.id() == LogicalTypeId::VARCHAR) {
				return std::make_pair(true, StringValue::Get(children[i]));
			}
			return std::make_pair(true, children[i].ToString());
		}
		return std::make_pair(false, string());
	}

	static string UDFPayloadStringOrDefault(const Value &payload, const string &name,
	                                        const string &default_value = string()) {
		auto field = UDFPayloadStringField(payload, name);
		if (!field.first || field.second.empty()) {
			return default_value;
		}
		return field.second;
	}

	static bool UDFPayloadStringEquals(const Value &payload, const string &name, const string &expected) {
		auto field = UDFPayloadStringField(payload, name);
		return field.first && field.second == expected;
	}

	static string UDFOperatorIdentityOrEmpty(const Value &payload) {
		static constexpr const char *OPERATOR_ID_FIELD = "_vane_udf_operator_id";
		if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
			throw duckdb::InvalidInputException("UDF payload must be a STRUCT");
		}
		auto &children = StructValue::GetChildren(payload);
		auto child_count = StructType::GetChildCount(payload.type());
		for (idx_t i = 0; i < child_count; i++) {
			if (StructType::GetChildName(payload.type(), i) != OPERATOR_ID_FIELD) {
				continue;
			}
			if (i >= children.size() || children[i].IsNull() ||
			    StructType::GetChildType(payload.type(), i).id() != LogicalTypeId::VARCHAR) {
				throw duckdb::InvalidInputException("UDF payload field %s must be a non-empty VARCHAR",
				                                    OPERATOR_ID_FIELD);
			}
			auto identity = StringValue::Get(children[i]);
			if (identity.empty()) {
				throw duckdb::InvalidInputException("UDF payload field %s must be a non-empty VARCHAR",
				                                    OPERATOR_ID_FIELD);
			}
			return identity;
		}
		return string();
	}

	static Value UDFPayloadWithStringField(const Value &payload, const string &name, const string &value) {
		if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
			throw duckdb::InvalidInputException("UDF payload must be a STRUCT");
		}
		auto &children = StructValue::GetChildren(payload);
		auto &payload_type = payload.type();
		child_list_t<Value> new_children;
		bool found = false;
		auto child_count = StructType::GetChildCount(payload_type);
		for (idx_t i = 0; i < child_count; i++) {
			auto child_name = StructType::GetChildName(payload_type, i);
			if (child_name == name) {
				new_children.emplace_back(child_name, Value(value));
				found = true;
			} else {
				new_children.emplace_back(child_name, children[i]);
			}
		}
		if (!found) {
			new_children.emplace_back(name, Value(value));
		}
		return Value::STRUCT(std::move(new_children));
	}

	static vector<Value> EnsureStableUDFOperatorIdentities(const vector<UDFFunctionData *> &physical_udfs,
	                                                       const string &query_id) {
		static constexpr const char *OPERATOR_ID_FIELD = "_vane_udf_operator_id";
		auto identity_prefix = "query:" + query_id + ":physical_udf:";
		std::unordered_set<string> identities;
		idx_t existing_identity_count = 0;
		for (auto *bind_data : physical_udfs) {
			if (!bind_data) {
				throw duckdb::InternalException("physical UDF collection returned null bind data");
			}
			auto identity = UDFOperatorIdentityOrEmpty(bind_data->payload);
			if (identity.empty()) {
				continue;
			}
			if (identity.compare(0, identity_prefix.size(), identity_prefix) != 0) {
				throw duckdb::InvalidInputException(
				    "physical UDF operator identity query ownership mismatch: operator=%s plan=%s", identity, query_id);
			}
			existing_identity_count++;
			if (!identities.insert(identity).second) {
				throw duckdb::InvalidInputException("duplicate physical UDF operator identity: %s", identity);
			}
		}
		if (existing_identity_count != 0 && existing_identity_count != physical_udfs.size()) {
			throw duckdb::InvalidInputException(
			    "physical UDF operator identities are incomplete: identified=%llu total=%llu",
			    static_cast<unsigned long long>(existing_identity_count),
			    static_cast<unsigned long long>(physical_udfs.size()));
		}
		if (existing_identity_count == physical_udfs.size()) {
			return {};
		}

		vector<Value> identified_payloads;
		identified_payloads.reserve(physical_udfs.size());
		for (idx_t i = 0; i < physical_udfs.size(); i++) {
			auto identity = identity_prefix + std::to_string(i);
			identified_payloads.push_back(
			    UDFPayloadWithStringField(physical_udfs[i]->payload, OPERATOR_ID_FIELD, identity));
		}
		vector<Value> original_payloads;
		original_payloads.reserve(physical_udfs.size());
		for (auto *bind_data : physical_udfs) {
			original_payloads.push_back(std::move(bind_data->payload));
		}
		for (idx_t i = 0; i < physical_udfs.size(); i++) {
			physical_udfs[i]->payload = std::move(identified_payloads[i]);
		}
		return original_payloads;
	}

	py::dict collect_query_resource_graph_metadata(py::object conn_obj) {
		if (!IsInitialized() || !plan_ || !has_root()) {
			throw duckdb::InternalException(
			    "DistributedPhysicalPlan must have a root before query resource graph metadata collection");
		}
		auto query_id = idx();
		if (query_id.empty()) {
			throw duckdb::InternalException(
			    "DistributedPhysicalPlan query resource graph metadata collection requires a query_id");
		}
		if (plan_->query_id() != query_id) {
			throw duckdb::InvalidInputException("DistributedPhysicalPlan query ownership mismatch: wrapper=%s plan=%s",
			                                    query_id, plan_->query_id());
		}

		vector<UDFFunctionData *> physical_udfs;
		auto physical_plan = plan_->physical_plan();
		std::function<void(PhysicalOperator &)> collect_physical_udfs = [&](PhysicalOperator &op) -> void {
			CollectMutableUDFBindData(op, physical_udfs);
			for (auto &child : op.children) {
				collect_physical_udfs(child.get());
			}
		};
		collect_physical_udfs(physical_plan->Root());
		auto unidentified_payloads = EnsureStableUDFOperatorIdentities(physical_udfs, query_id);
		struct UDFIdentityRollback {
			const vector<UDFFunctionData *> &bind_data;
			vector<Value> &payloads;
			bool active = true;

			~UDFIdentityRollback() {
				if (!active) {
					return;
				}
				for (idx_t i = 0; i < payloads.size(); i++) {
					bind_data[i]->payload = std::move(payloads[i]);
				}
			}

			void Commit() {
				active = false;
			}
		} identity_rollback {physical_udfs, unidentified_payloads};

		auto pipeline_root = BuildDistributedPipelineNode(plan_, client_context_.get());
		vector<duckdb::distributed::DistributedPipelineNodeRef> pipeline_nodes;
		std::unordered_set<duckdb::distributed::NodeID> visited;
		std::function<void(const duckdb::distributed::DistributedPipelineNodeRef &)> collect_pipeline =
		    [&](const duckdb::distributed::DistributedPipelineNodeRef &node) -> void {
			if (!node || !visited.insert(node->node_id()).second) {
				return;
			}
			pipeline_nodes.push_back(node);
			for (auto &child : node->arc_children()) {
				collect_pipeline(child);
			}
		};
		collect_pipeline(pipeline_root);

		struct PipelineUDFBinding {
			duckdb::distributed::NodeID node_id;
			const UDFFunctionData *bind_data;
			duckdb::distributed::DistributedPipelineNodeRef node;
		};
		std::unordered_map<string, PipelineUDFBinding> pipeline_udfs;
		for (auto &node : pipeline_nodes) {
			const UDFFunctionData *bind_data = nullptr;
			auto implementation = node->implementation();
			if (auto table_inout = std::dynamic_pointer_cast<duckdb::distributed::TableInOutNode>(implementation)) {
				bind_data = TryGetUDFBindData(table_inout->bind_data());
			} else if (auto streaming = std::dynamic_pointer_cast<duckdb::distributed::StreamingUDFPassthroughNode>(
			               implementation)) {
				bind_data = TryGetUDFBindData(streaming->bind_data());
			}
			if (bind_data) {
				auto identity = UDFOperatorIdentityOrEmpty(bind_data->payload);
				if (identity.empty()) {
					throw duckdb::InvalidInputException("pipeline UDF node %llu is missing a stable operator identity",
					                                    static_cast<unsigned long long>(node->node_id()));
				}
				if (!pipeline_udfs.emplace(identity, PipelineUDFBinding {node->node_id(), bind_data, node}).second) {
					throw duckdb::InvalidInputException("duplicate pipeline UDF operator identity: %s", identity);
				}
			}
		}
		std::sort(pipeline_nodes.begin(), pipeline_nodes.end(),
		          [](const auto &left, const auto &right) { return left->node_id() < right->node_id(); });

		if (physical_udfs.size() != pipeline_udfs.size()) {
			throw duckdb::InternalException(
			    "physical/pipeline UDF resource-unit count mismatch: physical=%llu pipeline=%llu",
			    static_cast<unsigned long long>(physical_udfs.size()),
			    static_cast<unsigned long long>(pipeline_udfs.size()));
		}

		struct PendingUDFResourceUnitAnnotation {
			UDFFunctionData *bind_data;
			Value payload;
		};
		vector<PendingUDFResourceUnitAnnotation> pending_annotations;
		pending_annotations.reserve(physical_udfs.size());
		std::unordered_map<duckdb::distributed::NodeID, idx_t> udf_by_node;
		for (auto *bind_data : physical_udfs) {
			auto identity = UDFOperatorIdentityOrEmpty(bind_data->payload);
			if (identity.empty()) {
				throw duckdb::InvalidInputException("physical UDF is missing a stable operator identity");
			}
			auto pipeline_entry = pipeline_udfs.find(identity);
			if (pipeline_entry == pipeline_udfs.end()) {
				throw duckdb::InvalidInputException("physical UDF operator identity is absent from pipeline DAG: %s",
				                                    identity);
			}
			auto &pipeline_binding = pipeline_entry->second;
			auto node_id = pipeline_binding.node_id;
			auto *pipeline_bind_data = pipeline_binding.bind_data;
			auto pipeline_query_id = pipeline_binding.node->context().query_id();
			if (pipeline_query_id != query_id) {
				throw duckdb::InvalidInputException(
				    "pipeline UDF query ownership mismatch for operator %s at node %llu: pipeline=%s plan=%s", identity,
				    static_cast<unsigned long long>(node_id), pipeline_query_id, query_id);
			}
			auto pipeline_backend = UDFPayloadStringOrDefault(pipeline_bind_data->payload, "execution_backend");
			auto physical_backend = UDFPayloadStringOrDefault(bind_data->payload, "execution_backend");
			if (pipeline_backend != physical_backend) {
				throw duckdb::InternalException(
				    "physical/pipeline UDF backend mismatch for operator %s at node %llu: physical=%s pipeline=%s",
				    identity, static_cast<unsigned long long>(node_id), physical_backend, pipeline_backend);
			}
			auto resource_unit_id = "resource:" + query_id + ":udf:node:" + std::to_string(node_id);
			auto existing_query_id = UDFPayloadStringOrDefault(bind_data->payload, "query_id");
			if (!existing_query_id.empty() && existing_query_id != query_id) {
				throw duckdb::InvalidInputException(
				    "physical UDF query ownership mismatch for operator %s at node %llu: payload=%s plan=%s", identity,
				    static_cast<unsigned long long>(node_id), existing_query_id, query_id);
			}
			auto pipeline_payload_query_id = UDFPayloadStringOrDefault(pipeline_bind_data->payload, "query_id");
			if (!pipeline_payload_query_id.empty() && pipeline_payload_query_id != query_id) {
				throw duckdb::InvalidInputException(
				    "pipeline UDF payload query ownership mismatch for operator %s at node %llu: payload=%s plan=%s",
				    identity, static_cast<unsigned long long>(node_id), pipeline_payload_query_id, query_id);
			}
			auto existing_resource_unit_id = UDFPayloadStringOrDefault(bind_data->payload, "resource_unit_id");
			if (!existing_resource_unit_id.empty() && existing_resource_unit_id != resource_unit_id) {
				throw duckdb::InvalidInputException(
				    "physical UDF resource-unit identity mismatch for operator %s at node %llu: payload=%s expected=%s",
				    identity, static_cast<unsigned long long>(node_id), existing_resource_unit_id, resource_unit_id);
			}
			auto pipeline_resource_unit_id = UDFPayloadStringOrDefault(pipeline_bind_data->payload, "resource_unit_id");
			if (!pipeline_resource_unit_id.empty() && pipeline_resource_unit_id != resource_unit_id) {
				throw duckdb::InvalidInputException(
				    "pipeline UDF resource-unit identity mismatch for operator %s at node %llu: payload=%s expected=%s",
				    identity, static_cast<unsigned long long>(node_id), pipeline_resource_unit_id, resource_unit_id);
			}
			if (bind_data->return_type != pipeline_bind_data->return_type ||
			    bind_data->payload != pipeline_bind_data->payload ||
			    bind_data->actor_handles != pipeline_bind_data->actor_handles) {
				throw duckdb::InternalException(
				    "physical/pipeline UDF descriptor mismatch for operator %s at node %llu", identity,
				    static_cast<unsigned long long>(node_id));
			}
			auto annotated_payload = UDFPayloadWithStringField(bind_data->payload, "query_id", query_id);
			annotated_payload = UDFPayloadWithStringField(annotated_payload, "resource_unit_id", resource_unit_id);
			auto annotation_index = pending_annotations.size();
			pending_annotations.push_back(PendingUDFResourceUnitAnnotation {bind_data, std::move(annotated_payload)});
			if (!udf_by_node.emplace(node_id, annotation_index).second) {
				throw duckdb::InternalException("duplicate pipeline UDF node identity: %llu",
				                                static_cast<unsigned long long>(node_id));
			}
		}

		duckdb::ClientProperties client_properties;
		if (!conn_obj.is_none()) {
			auto &conn_wrapper = ExtractPyConnectionWrapper(conn_obj);
			client_properties = conn_wrapper.con.GetConnection().context->GetClientProperties();
		}

		py::list nodes;
		for (auto &node : pipeline_nodes) {
			py::dict metadata;
			metadata[py::str("node_id")] = py::str(std::to_string(node->node_id()));
			metadata[py::str("node_name")] = py::str(node->name());
			py::list input_node_ids;
			auto children = node->arc_children();
			std::sort(children.begin(), children.end(),
			          [](const auto &left, const auto &right) { return left->node_id() < right->node_id(); });
			for (auto &child : children) {
				input_node_ids.append(py::str(std::to_string(child->node_id())));
			}
			metadata[py::str("input_node_ids")] = std::move(input_node_ids);
			metadata[py::str("is_sink")] = py::bool_(node->is_sink());
			metadata[py::str("is_materialization_barrier")] = py::bool_(node->is_materialization_barrier());
			py::list materialized_input_node_ids;
			auto materialized_inputs = node->materialized_input_node_ids();
			std::sort(materialized_inputs.begin(), materialized_inputs.end());
			for (auto input_node_id : materialized_inputs) {
				materialized_input_node_ids.append(py::str(std::to_string(input_node_id)));
			}
			metadata[py::str("materialized_input_node_ids")] = std::move(materialized_input_node_ids);
			metadata[py::str("num_partitions")] = py::int_(std::max<size_t>(1, node->num_partitions()));
			auto udf_entry = udf_by_node.find(node->node_id());
			if (udf_entry == udf_by_node.end()) {
				metadata[py::str("udf_payload")] = py::none();
			} else {
				auto &payload = pending_annotations[udf_entry->second].payload;
				metadata[py::str("udf_payload")] = PythonObject::FromValue(payload, payload.type(), client_properties);
			}
			nodes.append(std::move(metadata));
		}

		py::dict result;
		result[py::str("query_id")] = py::str(query_id);
		result[py::str("nodes")] = std::move(nodes);
		py::list terminals;
		terminals.append(py::str(std::to_string(pipeline_root->node_id())));
		result[py::str("terminal_node_ids")] = std::move(terminals);
		for (auto &pending : pending_annotations) {
			pending.bind_data->payload = std::move(pending.payload);
		}
		identity_rollback.Commit();
		return result;
	}

	string repr_ascii(bool simple) const {
		if (!IsInitialized()) {
			throw duckdb::InternalException("DistributedPhysicalPlan is uninitialized");
		}
		if (!plan_) {
			throw duckdb::InternalException("DistributedPhysicalPlan is not initialized");
		}
		auto pipeline_node = BuildDistributedPipelineNode(plan_, client_context_.get());
		return duckdb::distributed::viz_distributed_pipeline_ascii(pipeline_node, simple);
	}
	string repr_mermaid(bool simple, bool bottom_up) const {
		if (!IsInitialized()) {
			throw duckdb::InternalException("DistributedPhysicalPlan is uninitialized");
		}
		if (!plan_) {
			throw duckdb::InternalException("DistributedPhysicalPlan is not initialized");
		}
		auto pipeline_node = BuildDistributedPipelineNode(plan_, client_context_.get());
		auto level = simple ? duckdb::distributed::DisplayLevel::Compact : duckdb::distributed::DisplayLevel::Default;
		return duckdb::distributed::viz_distributed_pipeline_mermaid(pipeline_node, level, bottom_up, "");
	}
	py::dict scan_task_descriptor_map() const {
		py::dict out;
		if (!IsInitialized()) {
			throw duckdb::InternalException("DistributedPhysicalPlan is uninitialized");
		}
		if (!plan_) {
			throw duckdb::InternalException("DistributedPhysicalPlan is not initialized");
		}
		auto physical_plan = plan_->physical_plan();
		if (!physical_plan) {
			throw duckdb::InternalException("DistributedPhysicalPlan has no physical plan");
		}
		if (!physical_plan->HasRoot()) {
			throw duckdb::InternalException("DistributedPhysicalPlan physical plan has no root");
		}
		std::unordered_map<idx_t, std::vector<duckdb::distributed::ScanTaskDescriptor>> task_map;
		duckdb::shared_ptr<duckdb::DatabaseInstance> db;
		if (client_context_) {
			db = client_context_->db;
		}
		task_map = duckdb::distributed::physical_plan_scan_task_map_wrapper(physical_plan, plan_->execution_config(),
		                                                                    std::move(db));
		for (auto &kv : task_map) {
			py::list lst;
			for (auto &task : kv.second) {
				lst.append(py::bytes(task.SerializeToBytes()));
			}
			out[py::str(std::to_string(kv.first))] = std::move(lst);
		}

		if (plan_ && plan_->physical_plan() && plan_->physical_plan()->HasRoot()) {
			idx_t max_id = 0;
			std::function<void(PhysicalOperator &)> update_max = [&](PhysicalOperator &op) -> void {
				if (op.type == PhysicalOperatorType::TABLE_SCAN) {
					auto &scan = op.Cast<PhysicalTableScan>();
					if (scan.extra_info.scan_node_id.IsValid()) {
						auto id = scan.extra_info.scan_node_id.GetIndex();
						if (id > max_id) {
							max_id = id;
						}
					}
				}
				for (auto &child : op.children) {
					update_max(child.get());
				}
			};
			update_max(plan_->physical_plan()->Root());
			for (auto item : out) {
				auto key = py::str(item.first);
				auto id = static_cast<idx_t>(std::stoull(key.cast<string>()));
				if (id > max_id) {
					max_id = id;
				}
			}
			idx_t last_id = max_id;
			unordered_map<idx_t, idx_t> base_node_for_group;
			unordered_map<idx_t, idx_t> group_for_node;
			unordered_map<idx_t, idx_t> alias_to_parent;
			auto allocate_node_id = [&]() {
				if (last_id == NumericLimits<idx_t>::Maximum()) {
					throw InvalidInputException("cannot allocate a unique distributed scan node identity");
				}
				return ++last_id;
			};
			std::function<void(PhysicalOperator &)> normalize = [&](PhysicalOperator &op) -> void {
				if (op.type == PhysicalOperatorType::TABLE_SCAN) {
					auto &scan = op.Cast<PhysicalTableScan>();
					if (!scan.extra_info.scan_group_id.IsValid()) {
						if (scan.extra_info.scan_node_id.IsValid()) {
							scan.extra_info.scan_group_id = scan.extra_info.scan_node_id;
						} else {
							scan.extra_info.scan_group_id = optional_idx(allocate_node_id());
						}
					}
					if (!scan.extra_info.scan_node_id.IsValid()) {
						scan.extra_info.scan_node_id = optional_idx(allocate_node_id());
					}
					const auto group_id = scan.extra_info.scan_group_id.GetIndex();
					auto node_id = scan.extra_info.scan_node_id.GetIndex();
					const auto original_node_id = node_id;
					auto existing_node = group_for_node.find(node_id);
					if (existing_node != group_for_node.end()) {
						if (existing_node->second != group_id) {
							throw InvalidInputException(
							    "distributed scan node identity %llu is reused by scan groups %llu and %llu",
							    static_cast<unsigned long long>(node_id),
							    static_cast<unsigned long long>(existing_node->second),
							    static_cast<unsigned long long>(group_id));
						}
						node_id = allocate_node_id();
						scan.extra_info.scan_node_id = optional_idx(node_id);
						group_for_node.emplace(node_id, group_id);
						alias_to_parent.emplace(node_id, original_node_id);
					} else {
						group_for_node.emplace(node_id, group_id);
						auto group_entry = base_node_for_group.find(group_id);
						if (group_entry == base_node_for_group.end()) {
							base_node_for_group.emplace(group_id, node_id);
						} else if (node_id != group_entry->second) {
							alias_to_parent.emplace(node_id, group_entry->second);
						}
					}
				}
				for (auto &child : op.children) {
					normalize(child.get());
				}
			};
			normalize(plan_->physical_plan()->Root());

			for (const auto &kv : alias_to_parent) {
				auto dup_key = py::str(std::to_string(kv.first));
				if (out.contains(dup_key)) {
					continue;
				}
				set<idx_t> visited;
				auto assignment_id = kv.first;
				while (!out.contains(py::str(std::to_string(assignment_id)))) {
					if (!visited.insert(assignment_id).second) {
						throw InternalException("distributed scan node alias cycle detected at node %llu",
						                        static_cast<unsigned long long>(assignment_id));
					}
					auto alias = alias_to_parent.find(assignment_id);
					if (alias == alias_to_parent.end()) {
						break;
					}
					assignment_id = alias->second;
				}
				auto assignment_key = py::str(std::to_string(assignment_id));
				if (out.contains(assignment_key)) {
					out[dup_key] = out[assignment_key];
				}
			}
		}
		return out;
	}
	// ── UDF node collection ──────────────────────────────────────
	// Walk the physical plan and return metadata for each UDF node.
	// Used by RayQueryDriverActor to pre-create actor pools at the driver level.
	py::list collect_udf_nodes(py::object conn_obj) {
		py::list result;
		if (!IsInitialized() || !plan_)
			return result;
		auto physical_plan = plan_->physical_plan();
		if (!physical_plan || !physical_plan->HasRoot())
			return result;

		// We need ClientProperties to convert Value→py::object for the payload.
		duckdb::ClientProperties options;
		if (!conn_obj.is_none()) {
			auto &conn_wrapper = ExtractPyConnectionWrapper(conn_obj);
			options = conn_wrapper.con.GetConnection().context->GetClientProperties();
		}

		idx_t node_counter = 0;
		std::function<void(PhysicalOperator &)> collect = [&](PhysicalOperator &op) -> void {
			vector<UDFFunctionData *> bind_nodes;
			CollectMutableUDFBindData(op, bind_nodes);
			for (auto *bind_data : bind_nodes) {
				idx_t node_id = node_counter++;
				py::dict meta;
				meta[py::str("node_id")] = py::int_(node_id);
				// Convert the payload Value to a Python dict
				py::object payload_obj =
				    PythonObject::FromValue(bind_data->payload, bind_data->payload.type(), options);
				meta[py::str("payload")] = payload_obj;
				// Extract key fields
				if (py::isinstance<py::dict>(payload_obj)) {
					py::object execution_backend = payload_obj.attr("get")("execution_backend");
					py::object actor_pool_size = payload_obj.attr("get")("actor_pool_size");
					py::object cpus = payload_obj.attr("get")("cpus");
					py::object gpus = payload_obj.attr("get")("gpus");
					meta[py::str("execution_backend")] = execution_backend;
					meta[py::str("actor_pool_size")] = actor_pool_size;
					meta[py::str("cpus")] = cpus.is_none() ? py::float_(1.0) : cpus;
					meta[py::str("gpus")] = gpus.is_none() ? py::float_(0.0) : gpus;
				}
				result.append(meta);
			}
			for (auto &child : op.children) {
				collect(child.get());
			}
		};
		collect(physical_plan->Root());
		return result;
	}

	// ── Collect vLLM nodes and pre-inject pool names ─────────────────
	// Walks the physical plan, finds PhysicalVLLM operators, computes a
	// deterministic pool name for each, injects it into the operator's
	// options (so that VLLMProjectNode::produce_tasks reads it back),
	// and returns metadata for driver-side actor pre-creation.
	py::list collect_vllm_nodes(py::object conn_obj) {
		py::list result;
		if (!IsInitialized() || !plan_)
			return result;
		auto physical_plan = plan_->physical_plan();
		if (!physical_plan || !physical_plan->HasRoot())
			return result;

		// The pool identity must include the connection session. Query IDs are
		// normally unique, but they are caller-provided at this binding boundary;
		// allowing the same query ID from another session to attach to an
		// existing named actor would reuse that actor's session environment.
		auto session_id = VaneSessionIdFromSnapshot(connection_snapshot_);
		auto query_id = plan_->query_id();
		if (query_id.empty()) {
			query_id = std::to_string(plan_->idx());
		}

		// ClientProperties for Value→py::object conversion.
		duckdb::ClientProperties client_props;
		if (!conn_obj.is_none()) {
			try {
				auto &conn_wrapper = ExtractPyConnectionWrapper(conn_obj);
				client_props = conn_wrapper.con.GetConnection().context->GetClientProperties();
			} catch (...) {
			}
		}

		idx_t vllm_counter = 0;
		std::function<void(PhysicalOperator &)> collect = [&](PhysicalOperator &op) -> void {
			if (op.type == PhysicalOperatorType::VLLM_PROJECT) {
				auto &vllm_op = op.Cast<PhysicalVLLM>();
				idx_t node_id = vllm_counter++;

				// Keep distributed vLLM pools scoped to one connection session
				// and query. Cross-query reuse requires immutable configuration
				// validation and explicit lifecycle ownership.
				auto safe_session = duckdb::distributed::BuildUniquePoolComponent(session_id);
				auto safe_query = duckdb::distributed::BuildUniquePoolComponent(query_id);
				auto pool_name = "duckdb_vllm_" + safe_session + "_" + safe_query + "_" + std::to_string(node_id);

				// Inject pool name into operator options so the translator
				// propagates it to VLLMProjectNode → produce_tasks.
				vllm_op.options = duckdb::distributed::InjectDistributedOptions(vllm_op.options, pool_name);

				py::dict meta;
				meta[py::str("node_id")] = py::int_(node_id);
				meta[py::str("model")] = py::str(vllm_op.model);
				meta[py::str("pool_name")] = py::str(pool_name);

				// Convert options to Python for downstream parsing.
				py::object options_obj;
				try {
					options_obj = PythonObject::FromValue(vllm_op.options, vllm_op.options.type(), client_props);
				} catch (...) {
					options_obj = py::none();
				}
				meta[py::str("options")] = options_obj;

				result.append(meta);
			}
			for (auto &child : op.children) {
				collect(child.get());
			}
		};
		collect(physical_plan->Root());
		return result;
	}

	// ── Inject actor handles into UDF nodes ─────────────────────
	// Takes a dict {node_id: executor_options} and injects actor handles into
	// the corresponding UDF bind data for worker execution.
	void set_udf_actor_handles(py::dict handles_map, py::object) {
		if (!IsInitialized() || !plan_)
			return;
		if (handles_map.empty())
			return;

		if (!handles_map.empty()) {
			udf_actor_handles_ = handles_map;
		}

		auto physical_plan = plan_->physical_plan();
		if (!physical_plan || !physical_plan->HasRoot())
			return;

		idx_t node_counter = 0;
		std::function<void(PhysicalOperator &)> inject = [&](PhysicalOperator &op) -> void {
			vector<UDFFunctionData *> bind_nodes;
			CollectMutableUDFBindData(op, bind_nodes);
			for (auto *bind_data : bind_nodes) {
				idx_t node_id = node_counter++;
				py::str key(std::to_string(node_id));

				if (handles_map.contains(key)) {
					py::object handles_obj = handles_map[key];
					bind_data->actor_handles = WrapPyObjectForUDFActorHandles(handles_obj);
				}
			}
			for (auto &child : op.children) {
				inject(child.get());
			}
		};
		inject(physical_plan->Root());
	}
};

PyPhysicalPlanWrapper PyLogicalPlan::to_physical_plan(py::object conn_obj, py::object effective_session_config) const {
	if (conn_obj.is_none()) {
		throw duckdb::InternalException("Connection is required for to_physical_plan");
	}

	if (serialized_logical_plan_.empty()) {
		throw duckdb::InternalException("PyLogicalPlan missing serialized logical plan");
	}
	string scoped_secret_session_id;
	if (!scoped_secret_refs_.empty()) {
		scoped_secret_session_id = VaneSessionIdFromSnapshot(connection_snapshot_);
	}
	ValidateScopedSecretRefs(scoped_secret_refs_, query_id_, scoped_secret_session_id, "logical to physical planning");
	if (!source_scoped_secret_bindings_.empty() || !source_unmatched_scoped_secret_uses_.empty()) {
		if (source_connection_.is_none()) {
			throw duckdb::InternalException("Source scoped secret validation requires the source connection");
		}
		auto &source_wrapper = ExtractPyConnectionWrapper(source_connection_);
		auto source_context = source_wrapper.con.GetConnection().context;
		source_context->RunFunctionInTransaction([&]() {
			ValidateSourceScopedSecretBindings(*source_context, scoped_secret_refs_, source_scoped_secret_bindings_,
			                                   source_unmatched_scoped_secret_uses_);
		});
	}
	auto logical_payload = DecodeLogicalPlanEnvelope(serialized_logical_plan_);

	py::object planning_conn = ResolvePlanningConnectionForSnapshot(conn_obj, source_connection_, connection_snapshot_);
	auto &conn_wrapper = ExtractPyConnectionWrapper(planning_conn);
	const bool shares_source_database = ConnectionsShareDatabaseInstance(planning_conn, source_connection_);
	ConnectionSnapshotApplyOptions snapshot_options;
	snapshot_options.apply_session_config = effective_session_config.is_none();
	snapshot_options.enforce_extension_security = !shares_source_database;
	snapshot_options.apply_attached_databases = !shares_source_database;
	// Validate and load the exact static extension set before refreshed AWS
	// settings are allowed to execute LOAD httpfs. A snapshot that does not
	// declare httpfs must leave the planning DatabaseInstance uncontaminated.
	ValidateConnectionSnapshotExtensions(planning_conn, connection_snapshot_,
	                                     snapshot_options.enforce_extension_security);
	if (ConnectionSnapshotDeclaresStaticExtension(connection_snapshot_, "httpfs")) {
		// Resolved environment/profile credentials are the session baseline.
		// Replay the source connection below so explicit source SET values retain
		// DuckDB's normal precedence.
		ApplyEffectiveVaneSessionConfig(conn_wrapper, effective_session_config);
	}
	ApplyConnectionSnapshot(planning_conn, connection_snapshot_, snapshot_options);
	if (!udf_registrations_.is_none()) {
		conn_wrapper.ApplyDistributedPythonUDFRegistrations(udf_registrations_);
	}

	std::shared_ptr<duckdb::PhysicalPlan> physical_plan;
	auto &context = *conn_wrapper.con.GetConnection().context;
	context.RunFunctionInTransaction([&]() {
		duckdb::MemoryStream stream(duckdb::Allocator::Get(context));
		stream.WriteData(reinterpret_cast<const uint8_t *>(logical_payload.data()), logical_payload.size());
		stream.Rewind();

		duckdb::bound_parameter_map_t parameters;
		auto logical_plan =
		    duckdb::BinaryDeserializer::Deserialize<duckdb::LogicalOperator>(stream, context, parameters);
		// Re-enter the standard LogicalPlanStatement -> Planner path so the fresh
		// binder seeds bound_tables from the deserialized plan before optimizer
		// rewriters allocate any new table indexes.
		logical_plan = RebindAndOptimizeDeserializedLogicalPlan(context, std::move(logical_plan));

		duckdb::PhysicalPlanGenerator physical_planner(context);
		auto physical_plan_uptr = physical_planner.Plan(std::move(logical_plan));
		physical_plan = std::shared_ptr<duckdb::PhysicalPlan>(physical_plan_uptr.release());
	});

	uint16_t idx = duckdb::distributed::get_query_idx_counter().fetch_add(1);
	auto cfg = std::make_shared<duckdb::distributed::DuckDBExecutionConfig>(
	    duckdb::distributed::DuckDBExecutionConfig::from_env());
	if (cfg->shuffle_algorithm().empty()) {
		cfg->set_shuffle_algorithm("flight_shuffle");
	}
	auto distributed_plan =
	    std::make_shared<duckdb::distributed::DistributedPhysicalPlan>(idx, query_id_, physical_plan, cfg);

	auto plan_wrapper = PyPhysicalPlanWrapper(distributed_plan);
	plan_wrapper.query_id_ = query_id_;
	plan_wrapper.resource_query_id_ = query_id_;
	plan_wrapper.worker_connection_ = planning_conn;
	plan_wrapper.client_context_ = conn_wrapper.con.GetConnection().context;
	plan_wrapper.udf_registrations_ = udf_registrations_;
	plan_wrapper.connection_snapshot_ = PrepareWorkerConnectionSnapshot(connection_snapshot_);
	plan_wrapper.scoped_secret_refs_ = scoped_secret_refs_;
	auto validate_serialization =
	    py::module_::import("vane._ray_cxx").attr("validate_plan_serialization_for_submission");
	validate_serialization(py::cast(plan_wrapper));
	return plan_wrapper;
}

class PyBackendResultOperationGuard {
public:
	explicit PyBackendResultOperationGuard(std::function<void()> finish) : finish_(std::move(finish)) {
	}

	PyBackendResultOperationGuard(const PyBackendResultOperationGuard &) = delete;
	PyBackendResultOperationGuard &operator=(const PyBackendResultOperationGuard &) = delete;

	~PyBackendResultOperationGuard() {
		finish_();
	}

private:
	std::function<void()> finish_;
};

class PyBackendWorkerManager : public duckdb::distributed::WorkerManager {
public:
	template <class T>
	using DuckDBResult = duckdb::distributed::DuckDBResult<T>;
	using DuckDBError = duckdb::distributed::DuckDBError;
	using QueryCleanup = std::function<void(const string &)>;

	explicit PyBackendWorkerManager(py::object backend, QueryCleanup query_cleanup)
	    : backend_(std::move(backend)), query_cleanup_(std::move(query_cleanup)) {
	}

	void register_query_owner(const string &query_id, const string &owner_query_id,
	                          const std::function<void(const py::error_already_set &)> &publish_registration_error = {},
	                          bool *lifecycle_published = nullptr) {
		if (lifecycle_published) {
			*lifecycle_published = false;
		}
		if (query_id.empty() || owner_query_id.empty()) {
			throw std::invalid_argument("FTE query ownership requires non-empty query and owner IDs");
		}
		{
			lock_guard<mutex> guard(mutex_);
			auto existing_owner = result_handle_owner_by_query_.find(query_id);
			if (existing_owner != result_handle_owner_by_query_.end() && existing_owner->second != owner_query_id) {
				throw std::runtime_error("FTE query result owner changed while active: query=" + query_id +
				                         " existing=" + existing_owner->second + " requested=" + owner_query_id);
			}
			if (all_result_handle_ingress_closed_ ||
			    closed_result_handle_queries_.find(query_id) != closed_result_handle_queries_.end() ||
			    closed_result_handle_query_owners_.find(owner_query_id) != closed_result_handle_query_owners_.end() ||
			    dropping_result_handle_query_owners_.find(owner_query_id) !=
			        dropping_result_handle_query_owners_.end()) {
				throw std::runtime_error("cannot register closing FTE query lifecycle: " + query_id);
			}
			if (registering_result_handle_owner_by_query_.find(query_id) !=
			    registering_result_handle_owner_by_query_.end()) {
				throw std::runtime_error("FTE query lifecycle registration is already in progress: " + query_id);
			}
			if (existing_owner != result_handle_owner_by_query_.end()) {
				return;
			}
			// Publish the ownership edge before invoking Python. Abort and shutdown
			// can resolve the resource owner immediately, but wait on the
			// registration transition below before starting their sweep.
			result_handle_owner_by_query_[query_id] = owner_query_id;
			registering_result_handle_owner_by_query_[query_id] = owner_query_id;
			if (lifecycle_published) {
				*lifecycle_published = true;
			}
		}
		try {
			duckdb::PythonGILWrapper gil;
			auto backend = backend_.get();
			backend.attr("register_query_owner")(query_id, owner_query_id);
		} catch (const py::error_already_set &ex) {
			std::exception_ptr publication_error;
			if (publish_registration_error) {
				try {
					publish_registration_error(ex);
				} catch (...) {
					publication_error = std::current_exception();
				}
			}
			{
				lock_guard<mutex> guard(mutex_);
				closed_result_handle_query_owners_.insert(owner_query_id);
				registering_result_handle_owner_by_query_.erase(query_id);
			}
			result_handle_condition_.notify_all();
			if (publication_error) {
				std::rethrow_exception(publication_error);
			}
			throw;
		} catch (...) {
			{
				lock_guard<mutex> guard(mutex_);
				closed_result_handle_query_owners_.insert(owner_query_id);
				registering_result_handle_owner_by_query_.erase(query_id);
			}
			result_handle_condition_.notify_all();
			throw;
		}
		{
			lock_guard<mutex> guard(mutex_);
			registering_result_handle_owner_by_query_.erase(query_id);
		}
		result_handle_condition_.notify_all();
	}

	DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>> worker_snapshots() const override {
		duckdb::PythonGILWrapper gil;
		try {
			auto backend = backend_.get();
			py::object raw_snapshots = backend.attr("worker_snapshots")();
			std::vector<duckdb::distributed::WorkerSnapshot> snapshots;
			for (auto item : raw_snapshots) {
				py::object obj = py::reinterpret_borrow<py::object>(item);
				if (!py::isinstance<py::dict>(obj)) {
					return DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>::err(
					    DuckDBError("Python backend worker_snapshots entries must be dicts"));
				}
				auto snapshot = obj.cast<py::dict>();
				auto worker_id = PyStringField(snapshot, "worker_id", "python-backend");
				double num_cpus = PyDoubleField(snapshot, "num_cpus", PyDoubleField(snapshot, "CPU", 1.0));
				double num_gpus = PyDoubleField(snapshot, "num_gpus", PyDoubleField(snapshot, "GPU", 0.0));
				size_t memory_bytes = static_cast<size_t>(
				    PyUInt64Field(snapshot, "total_memory_bytes", PyUInt64Field(snapshot, "memory", 4ULL << 30)));
				snapshots.emplace_back(duckdb::distributed::make_worker_id(worker_id), num_cpus, num_gpus,
				                       memory_bytes);
			}
			return DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>::ok(std::move(snapshots));
		} catch (const std::exception &e) {
			return DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>::err(
			    DuckDBError(string("Python backend worker_snapshots failed: ") + e.what()));
		}
	}

	DuckDBResult<void> try_autoscale(const std::vector<duckdb::distributed::TaskResourceRequest> &) override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> shutdown() override {
		if (!BeginResultHandleShutdown()) {
			return DuckDBResult<void>::ok();
		}
		bool shutdown_succeeded = false;
		PyBackendResultOperationGuard shutdown_guard(
		    [this, &shutdown_succeeded]() { EndResultHandleShutdown(shutdown_succeeded); });
		auto clear_handles = [&](const char *phase) -> std::optional<string> {
			try {
				ClearAllResultHandles();
				return std::nullopt;
			} catch (const std::exception &ex) {
				return string(phase) + " result cleanup: " + ex.what();
			} catch (...) {
				return string(phase) + " result cleanup: unknown error";
			}
		};
		std::optional<string> backend_shutdown_error;
		try {
			duckdb::PythonGILWrapper gil;
			auto backend = backend_.get();
			backend.attr("shutdown")();
		} catch (const std::exception &e) {
			backend_shutdown_error = string("backend shutdown: ") + e.what();
		} catch (...) {
			backend_shutdown_error = "backend shutdown: unknown error";
		}
		WaitForAllResultHandleOperations();
		auto initial_cleanup_error = clear_handles("post-quiescence");
		std::optional<string> final_cleanup_error;
		if (initial_cleanup_error) {
			final_cleanup_error = clear_handles("cleanup retry");
		}
		std::vector<string> owner_cleanup_errors;
		std::vector<string> ordered_owner_query_ids;
		if (!backend_shutdown_error && !final_cleanup_error) {
			std::unordered_set<string> owner_query_ids;
			{
				lock_guard<mutex> guard(mutex_);
				for (const auto &entry : result_handle_owner_by_query_) {
					owner_query_ids.insert(entry.second);
				}
			}
			ordered_owner_query_ids.assign(owner_query_ids.begin(), owner_query_ids.end());
			std::sort(ordered_owner_query_ids.begin(), ordered_owner_query_ids.end());
			for (const auto &owner_query_id : ordered_owner_query_ids) {
				try {
					query_cleanup_(owner_query_id);
				} catch (const std::exception &ex) {
					owner_cleanup_errors.push_back(owner_query_id + ": " + ex.what());
				} catch (...) {
					owner_cleanup_errors.push_back(owner_query_id + ": unknown cleanup error");
				}
			}
		}
		std::vector<string> errors;
		if (backend_shutdown_error) {
			errors.push_back(*backend_shutdown_error);
		}
		if (initial_cleanup_error && final_cleanup_error) {
			errors.push_back(*initial_cleanup_error);
			errors.push_back(*final_cleanup_error);
		}
		for (const auto &owner_cleanup_error : owner_cleanup_errors) {
			errors.push_back("query owner cleanup: " + owner_cleanup_error);
		}
		if (!errors.empty()) {
			string message = "Python backend shutdown failed";
			for (const auto &error : errors) {
				message += "; " + error;
			}
			return DuckDBResult<void>::err(DuckDBError(std::move(message)));
		}
		for (const auto &owner_query_id : ordered_owner_query_ids) {
			submission_errors_.Discard(owner_query_id);
		}
		FinishResultHandleShutdown();
		shutdown_succeeded = true;
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> submit_fte_task_events(std::vector<duckdb::distributed::WorkerTask> tasks) override {
		string query_id;
		string submission_error_owner;
		try {
			query_id = QueryIdFromTaskEvents(tasks);
			submission_error_owner = duckdb::distributed::python::ray::SubmissionErrorOwnerQueryId(tasks, query_id);
			if (!tasks.empty() && query_id.empty()) {
				return DuckDBResult<void>::err(DuckDBError::value_error("FTE task events require non-empty query_id"));
			}
			if (tasks.empty()) {
				return DuckDBResult<void>::ok();
			}
			// Execution query IDs created by operators (for example ORDER BY
			// sampling/range stages) join the outer resource lifecycle before
			// any task can reach the backend. Registration failures publish their
			// Python exception before the admission transition becomes quiescent.
			register_query_owner(query_id, submission_error_owner, [&](const py::error_already_set &error) {
				submission_errors_.Store(submission_error_owner, error);
			});
			auto active_owner = BeginResultHandleOperation(query_id, submission_error_owner);
			if (!active_owner) {
				return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
				    "FTE task submission rejected because its resource query is closing: " + submission_error_owner));
			}
			PyBackendResultOperationGuard operation(
			    [this, owner = *active_owner]() { EndResultHandleOperation(owner); });
			try {
				duckdb::PythonGILWrapper gil;
				py::list py_tasks;
				for (auto &task : tasks) {
					duckdb::distributed::python::ray::RayWorkerTask py_task_wrapper(std::move(task));
					py_tasks.append(py::cast(std::move(py_task_wrapper), py::return_value_policy::move));
				}
				auto backend = backend_.get();
				py::object raw_handles = backend.attr("submit_tasks")(py_tasks);
				StorePythonResultHandles(query_id, *active_owner, std::move(raw_handles));
				return DuckDBResult<void>::ok();
			} catch (const py::error_already_set &e) {
				// Keep exception publication inside the active-operation boundary so
				// shutdown cannot clear the owner and then receive a late exception.
				CloseResultHandleOwnerIngress(*active_owner);
				submission_errors_.Store(*active_owner, e);
				return DuckDBResult<void>::err(
				    DuckDBError(string("Python backend submit_fte_task_events failed: ") + e.what()));
			} catch (const std::exception &e) {
				CloseResultHandleOwnerIngress(*active_owner);
				return DuckDBResult<void>::err(DuckDBError(string("submit_fte_task_events failed: ") + e.what()));
			}
		} catch (const py::error_already_set &e) {
			return DuckDBResult<void>::err(
			    DuckDBError(string("Python backend submit_fte_task_events failed: ") + e.what()));
		} catch (const std::exception &e) {
			return DuckDBResult<void>::err(DuckDBError(string("submit_fte_task_events failed: ") + e.what()));
		}
	}

	DuckDBResult<void> task_input_stream_exhausted_for_query(
	    const string &query_id, const std::unordered_set<duckdb::distributed::SourceNodeId> &source_node_ids) override {
		if (query_id.empty()) {
			return DuckDBResult<void>::err(
			    DuckDBError::value_error("FTE task input exhaustion requires non-empty query_id"));
		}
		try {
			auto active_owner = BeginResultHandleOperation(query_id);
			if (!active_owner) {
				return DuckDBResult<void>::err(
				    DuckDBError::invalid_state_error("Python backend FTE query input stream is closing: " + query_id));
			}
			PyBackendResultOperationGuard operation(
			    [this, owner = *active_owner]() { EndResultHandleOperation(owner); });
			duckdb::PythonGILWrapper gil;
			py::list py_source_node_ids;
			for (auto source_node_id : source_node_ids) {
				py_source_node_ids.append(std::to_string(source_node_id));
			}
			auto backend = backend_.get();
			py::object raw_handles = backend.attr("task_input_stream_exhausted")(query_id, py_source_node_ids);
			StorePythonResultHandles(query_id, *active_owner, std::move(raw_handles));
			return DuckDBResult<void>::ok();
		} catch (const std::exception &e) {
			return DuckDBResult<void>::err(
			    DuckDBError(string("Python backend task_input_stream_exhausted failed: ") + e.what()));
		}
	}

	DuckDBResult<void> materialization_barrier_completed(const string &query_id,
	                                                     duckdb::distributed::NodeID node_id) override {
		if (query_id.empty()) {
			return DuckDBResult<void>::err(
			    DuckDBError::value_error("materialization barrier completion requires non-empty query_id"));
		}
		try {
			auto active_owner = BeginResultHandleOperation(query_id);
			if (!active_owner) {
				return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
				    "Python backend FTE query materialization barrier is closing: " + query_id));
			}
			PyBackendResultOperationGuard operation(
			    [this, owner = *active_owner]() { EndResultHandleOperation(owner); });
			duckdb::PythonGILWrapper gil;
			auto backend = backend_.get();
			backend.attr("materialization_barrier_completed")(query_id, std::to_string(node_id));
			return DuckDBResult<void>::ok();
		} catch (const std::exception &e) {
			return DuckDBResult<void>::err(
			    DuckDBError(string("Python backend materialization_barrier_completed failed: ") + e.what()));
		}
	}

	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> wait_fte_query(const string &query_id,
	                                                                                  double timeout_s) override {
		return wait_fte_query(query_id, timeout_s, {});
	}

	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>
	wait_fte_query(const string &query_id, double timeout_s,
	               duckdb::distributed::MaterializedOutputCallback on_output) override {
		const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash> empty_contexts;
		return wait_fte_query(query_id, timeout_s, empty_contexts, std::move(on_output));
	}

	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> wait_fte_query(
	    const string &query_id, double timeout_s,
	    const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash> &task_contexts,
	    duckdb::distributed::MaterializedOutputCallback on_output) override {
		if (query_id.empty()) {
			return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
			    DuckDBError::value_error("query_id must be non-empty"));
		}
		auto active_owner = BeginResultHandleOperation(query_id);
		if (!active_owner) {
			return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
			    DuckDBError::external_error("Python backend FTE query is closing: " + query_id));
		}
		PyBackendResultOperationGuard operation([this, owner = *active_owner]() { EndResultHandleOperation(owner); });

		std::unordered_set<string> selected_attempt_task_ids;
		const bool has_deadline = timeout_s > 0.0;
		const auto deadline = has_deadline ? std::chrono::steady_clock::now() +
		                                         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
		                                             std::chrono::duration<double>(timeout_s))
		                                   : std::chrono::steady_clock::time_point::max();
		while (true) {
			bool failed = false;
			bool finished = false;
			bool canceled = false;
			bool matched = true;
			bool registration_pending = false;
			string status_message;
			try {
				duckdb::PythonGILWrapper gil;
				auto backend = backend_.get();
				py::object status_obj;
				if (task_contexts.empty()) {
					status_obj = backend.attr("fte_query_status")(query_id);
				} else {
					py::list py_task_contexts;
					for (const auto &task_context : task_contexts) {
						py::dict info;
						info["query_idx"] = task_context.query_idx();
						info["last_node_id"] = task_context.last_node_id();
						info["task_id"] = task_context.task_id();
						py::list node_ids;
						for (auto node_id : task_context.node_ids()) {
							node_ids.append(node_id);
						}
						info["node_ids"] = std::move(node_ids);
						py_task_contexts.append(std::move(info));
					}
					status_obj = backend.attr("fte_query_status")(query_id, std::move(py_task_contexts));
				}
				status_message = PyStatusMessage(status_obj);
				failed = RequiredStatusBool(status_obj, "failed");
				finished = RequiredStatusBool(status_obj, "finished");
				canceled = OptionalStatusBool(status_obj, "canceled", false);
				if (!task_contexts.empty()) {
					matched = RequiredStatusBool(status_obj, "matched");
					registration_pending = OptionalStatusBool(status_obj, "registration_pending", false);
				}
				selected_attempt_task_ids = SelectedAttemptTaskIds(status_obj);
			} catch (const std::exception &e) {
				ClearResultHandles(query_id);
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError(string("Python backend fte_query_status failed: ") + e.what()));
			}
			if (failed) {
				ClearResultHandles(query_id);
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError::external_error("Python backend FTE query failed: " + status_message));
			}
			if (canceled) {
				ClearResultHandles(query_id);
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError::external_error("Python backend FTE query canceled: " + status_message));
			}
			if (!task_contexts.empty() && !matched) {
				if (!registration_pending) {
					return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
					    DuckDBError::external_error(
					        "Python backend FTE query scope did not match any registered fragment: " + status_message));
				}
			} else if (finished) {
				break;
			}
			if (has_deadline && std::chrono::steady_clock::now() >= deadline) {
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError::external_error("timed out waiting for Python backend FTE query: " + status_message));
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}

		const double remaining_timeout_s =
		    has_deadline
		        ? std::max(0.0, std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count())
		        : -1.0;
		auto drain_res = DrainResultHandles(query_id, remaining_timeout_s, selected_attempt_task_ids,
		                                    task_contexts.empty() ? nullptr : &task_contexts, false);
		if (drain_res.is_err()) {
			return drain_res;
		}
		std::vector<duckdb::distributed::MaterializedOutput> outputs;
		for (auto &output : drain_res.value()) {
			if (on_output) {
				auto callback_res = on_output(output);
				if (callback_res.is_err()) {
					ClearResultHandles(query_id);
					return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
					    callback_res.error());
				}
			}
			outputs.push_back(std::move(output));
		}
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::ok(std::move(outputs));
	}

	DuckDBResult<void> abort_and_quiesce_query(const string &query_id) override {
		if (query_id.empty()) {
			return DuckDBResult<void>::err(DuckDBError::value_error("FTE query abort requires non-empty query_id"));
		}
		std::optional<ResultHandleQuiesce> active_quiesce;
		try {
			active_quiesce = BeginResultHandleQuiesce(query_id);
		} catch (const std::exception &ex) {
			return DuckDBResult<void>::err(DuckDBError::external_error(ex.what()));
		} catch (...) {
			return DuckDBResult<void>::err(
			    DuckDBError::external_error("unknown error while starting Python backend resource query abort"));
		}
		if (!active_quiesce) {
			return DuckDBResult<void>::ok();
		}

		bool succeeded = false;
		PyBackendResultOperationGuard quiesce_guard(
		    [&]() { EndResultHandleQuiesce(active_quiesce->owner_query_id, active_quiesce->token, succeeded); });
		try {
			std::vector<std::pair<string, std::exception_ptr>> backend_drop_errors;
			auto drop_execution_queries = [&]() {
				try {
					duckdb::PythonGILWrapper gil;
					auto backend = backend_.get();
					auto drop_query = backend.attr("drop_query");
					for (const auto &execution_query_id : active_quiesce->execution_query_ids) {
						try {
							drop_query(execution_query_id);
						} catch (...) {
							backend_drop_errors.emplace_back(execution_query_id, std::current_exception());
						}
					}
				} catch (...) {
					backend_drop_errors.emplace_back(active_quiesce->owner_query_id, std::current_exception());
				}
			};
			drop_execution_queries();
			if (backend_drop_errors.empty()) {
				WaitForResultHandleOperations(active_quiesce->owner_query_id);
				if (active_quiesce->had_active_operations) {
					drop_execution_queries();
				}
			}
			succeeded = backend_drop_errors.empty();
			if (succeeded) {
				return DuckDBResult<void>::ok();
			}

			auto exception_message = [](const std::exception_ptr &error) {
				try {
					std::rethrow_exception(error);
				} catch (const std::exception &ex) {
					return string(ex.what());
				} catch (...) {
					return string("unknown exception");
				}
			};
			string message = "Python backend resource query abort barrier failed";
			for (const auto &error : backend_drop_errors) {
				message += "; " + error.first + "=" + exception_message(error.second);
			}
			return DuckDBResult<void>::err(DuckDBError::external_error(std::move(message)));
		} catch (const std::exception &ex) {
			return DuckDBResult<void>::err(DuckDBError::external_error(
			    string("Python backend resource query abort orchestration failed: ") + ex.what()));
		} catch (...) {
			return DuckDBResult<void>::err(
			    DuckDBError::external_error("Python backend resource query abort orchestration failed: unknown error"));
		}
	}

	void drop_query_fragments(const string &query_id) {
		if (query_id.empty()) {
			return;
		}
		auto quiesce_res = abort_and_quiesce_query(query_id);
		if (quiesce_res.is_err()) {
			throw std::runtime_error(quiesce_res.error().what());
		}
		auto active_drop_owner = BeginResultHandleDrop(query_id);
		if (!active_drop_owner) {
			return;
		}
		const auto owner_query_id = active_drop_owner->owner_query_id;
		const auto drop_token = active_drop_owner->token;
		PyBackendResultOperationGuard drop_guard(
		    [this, owner_query_id, drop_token]() { EndResultHandleDrop(owner_query_id, drop_token); });
		std::exception_ptr initial_cleanup_error;
		std::exception_ptr final_cleanup_error;
		std::exception_ptr owner_cleanup_error;
		WaitForResultHandleOperations(owner_query_id);
		try {
			ClearResultHandlesForOwner(owner_query_id);
		} catch (...) {
			initial_cleanup_error = std::current_exception();
		}
		if (initial_cleanup_error) {
			try {
				ClearResultHandlesForOwner(owner_query_id);
			} catch (...) {
				final_cleanup_error = std::current_exception();
			}
		}
		if (!final_cleanup_error) {
			try {
				query_cleanup_(owner_query_id);
			} catch (...) {
				owner_cleanup_error = std::current_exception();
			}
		}
		if (!final_cleanup_error && !owner_cleanup_error) {
			submission_errors_.Discard(owner_query_id);
			FinishResultHandleIngress(owner_query_id, drop_token);
			return;
		}
		std::vector<std::pair<const char *, std::exception_ptr>> errors;
		if (initial_cleanup_error && final_cleanup_error) {
			errors.emplace_back("initial result cleanup", initial_cleanup_error);
		}
		if (final_cleanup_error) {
			errors.emplace_back("final result cleanup", final_cleanup_error);
		}
		if (owner_cleanup_error) {
			errors.emplace_back("query owner cleanup", owner_cleanup_error);
		}
		if (errors.size() == 1) {
			std::rethrow_exception(errors.front().second);
		}
		auto exception_message = [](const std::exception_ptr &error) {
			try {
				std::rethrow_exception(error);
			} catch (const std::exception &e) {
				return string(e.what());
			} catch (...) {
				return string("unknown exception");
			}
		};
		string message = "Python backend query teardown failed";
		for (const auto &error : errors) {
			message += string("; ") + error.first + "=" + exception_message(error.second);
		}
		throw std::runtime_error(std::move(message));
	}

	void rethrow_submission_error(const string &query_id, const string &details = string()) {
		string owner_query_id;
		{
			lock_guard<mutex> guard(mutex_);
			owner_query_id = ResultHandleOwnerForQueryLocked(query_id);
		}
		auto message = string("distributed worker task submission failed for query_id=") + query_id;
		if (!details.empty()) {
			message += "; execution error: " + details;
		}
		submission_errors_.RethrowAsCause(owner_query_id, std::move(message));
	}

	std::unordered_map<string, std::unordered_map<string, idx_t>> fragment_stats_by_worker() const {
		duckdb::PythonGILWrapper gil;
		std::unordered_map<string, std::unordered_map<string, idx_t>> out;
		auto backend = backend_.get();
		if (!py::hasattr(backend, "fragment_stats_by_worker")) {
			return out;
		}
		py::object raw_stats = backend.attr("fragment_stats_by_worker")();
		if (raw_stats.is_none()) {
			return out;
		}
		if (!py::isinstance<py::dict>(raw_stats)) {
			throw py::type_error("Python backend fragment_stats_by_worker() must return a dict");
		}
		auto stats_by_worker = raw_stats.cast<py::dict>();
		for (auto worker_item : stats_by_worker) {
			auto worker_id = py::str(worker_item.first).cast<string>();
			auto worker_stats_obj = py::reinterpret_borrow<py::object>(worker_item.second);
			if (!py::isinstance<py::dict>(worker_stats_obj)) {
				throw py::type_error("Python backend fragment stats entries must be dicts");
			}
			auto worker_stats = worker_stats_obj.cast<py::dict>();
			auto &target = out[worker_id];
			for (auto stat_item : worker_stats) {
				auto stat_name = py::str(stat_item.first).cast<string>();
				auto stat_value_obj = py::reinterpret_borrow<py::object>(stat_item.second);
				int64_t value = 0;
				try {
					value = py::int_(stat_value_obj).cast<int64_t>();
				} catch (const std::exception &e) {
					throw py::type_error("Python backend fragment stats values must be integers");
				}
				target[stat_name] = value < 0 ? 0 : static_cast<idx_t>(value);
			}
		}
		return out;
	}

private:
	struct ResultHandleDrop {
		string owner_query_id;
		uint64_t token;
	};
	struct ResultHandleQuiesce {
		string owner_query_id;
		std::vector<string> execution_query_ids;
		uint64_t token;
		bool had_active_operations;
	};

	mutable mutex mutex_;
	std::condition_variable result_handle_condition_;
	duckdb::distributed::python::ray::SafePyObject backend_;
	QueryCleanup query_cleanup_;
	duckdb::distributed::python::ray::PythonExceptionStore submission_errors_;
	std::unordered_map<string, std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>>>
	    result_handles_by_query_;
	std::unordered_map<string, std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>>>
	    retained_result_handles_by_query_;
	std::unordered_map<string, std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>>>
	    cleanup_retry_result_handles_by_query_;
	std::unordered_map<string, string> result_handle_owner_by_query_;
	std::unordered_map<string, string> registering_result_handle_owner_by_query_;
	std::unordered_map<string, idx_t> active_result_handle_operations_by_owner_;
	std::unordered_set<string> closed_result_handle_queries_;
	std::unordered_set<string> closed_result_handle_query_owners_;
	std::unordered_set<string> quiesced_result_handle_query_owners_;
	std::unordered_map<string, uint64_t> quiescing_result_handle_query_owners_;
	std::unordered_map<string, idx_t> result_handle_quiesce_waiters_by_owner_;
	std::unordered_map<string, uint64_t> dropping_result_handle_query_owners_;
	uint64_t next_result_handle_quiesce_token_ = 1;
	uint64_t next_result_handle_drop_token_ = 1;
	bool all_result_handle_ingress_closed_ = false;
	bool result_handle_shutdown_running_ = false;
	bool result_handle_shutdown_finished_ = false;

	static string QueryIdFromTaskEvents(const std::vector<duckdb::distributed::WorkerTask> &tasks) {
		std::string query_id;
		for (const auto &task : tasks) {
			const auto &context = task.context();
			auto it = context.find("query_id");
			if (it == context.end() || it->second.empty()) {
				continue;
			}
			if (query_id.empty()) {
				query_id = it->second;
				continue;
			}
			if (query_id != it->second) {
				throw std::runtime_error("FTE submit batch contains multiple query_id values");
			}
		}
		return query_id;
	}

	string ResultHandleOwnerForQueryLocked(const string &query_id) const {
		auto owner = result_handle_owner_by_query_.find(query_id);
		return owner == result_handle_owner_by_query_.end() ? query_id : owner->second;
	}

	void CloseResultHandleOwnerIngress(const string &owner_query_id) {
		{
			lock_guard<mutex> guard(mutex_);
			closed_result_handle_query_owners_.insert(owner_query_id);
		}
		result_handle_condition_.notify_all();
	}

	std::optional<ResultHandleQuiesce> BeginResultHandleQuiesceWithoutGIL(const string &query_id) {
		std::unique_lock<mutex> guard(mutex_);
		auto mapped_owner = result_handle_owner_by_query_.find(query_id);
		auto registering_owner = registering_result_handle_owner_by_query_.find(query_id);
		string owner_query_id;
		if (mapped_owner != result_handle_owner_by_query_.end()) {
			owner_query_id = mapped_owner->second;
		} else if (registering_owner != registering_result_handle_owner_by_query_.end()) {
			owner_query_id = registering_owner->second;
		} else {
			const bool known_owner =
			    std::any_of(result_handle_owner_by_query_.begin(), result_handle_owner_by_query_.end(),
			                [&](const auto &entry) { return entry.second == query_id; }) ||
			    std::any_of(registering_result_handle_owner_by_query_.begin(),
			                registering_result_handle_owner_by_query_.end(),
			                [&](const auto &entry) { return entry.second == query_id; });
			if (!known_owner) {
				return std::nullopt;
			}
			owner_query_id = query_id;
		}

		while (true) {
			auto active_quiesce = quiescing_result_handle_query_owners_.find(owner_query_id);
			if (active_quiesce == quiescing_result_handle_query_owners_.end()) {
				break;
			}
			const auto active_token = active_quiesce->second;
			result_handle_quiesce_waiters_by_owner_[owner_query_id]++;
			result_handle_condition_.wait(guard, [&]() {
				auto current = quiescing_result_handle_query_owners_.find(owner_query_id);
				return current == quiescing_result_handle_query_owners_.end() || current->second != active_token;
			});
			auto waiter = result_handle_quiesce_waiters_by_owner_.find(owner_query_id);
			D_ASSERT(waiter != result_handle_quiesce_waiters_by_owner_.end());
			D_ASSERT(waiter->second > 0);
			if (--waiter->second == 0) {
				result_handle_quiesce_waiters_by_owner_.erase(waiter);
			}
			result_handle_condition_.notify_all();
			if (quiesced_result_handle_query_owners_.find(owner_query_id) !=
			    quiesced_result_handle_query_owners_.end()) {
				return std::nullopt;
			}
		}
		if (quiesced_result_handle_query_owners_.find(owner_query_id) != quiesced_result_handle_query_owners_.end()) {
			return std::nullopt;
		}
		if (all_result_handle_ingress_closed_) {
			while (result_handle_shutdown_running_) {
				result_handle_condition_.wait(guard);
			}
			if (result_handle_shutdown_finished_) {
				return std::nullopt;
			}
			throw std::runtime_error("cannot abort FTE query after Python backend shutdown failed: " + query_id);
		}

		closed_result_handle_queries_.insert(query_id);
		closed_result_handle_query_owners_.insert(owner_query_id);
		const auto token = next_result_handle_quiesce_token_++;
		if (next_result_handle_quiesce_token_ == 0) {
			next_result_handle_quiesce_token_ = 1;
		}
		quiescing_result_handle_query_owners_[owner_query_id] = token;
		auto owner_registration_in_progress = [&]() {
			return std::any_of(
			    registering_result_handle_owner_by_query_.begin(), registering_result_handle_owner_by_query_.end(),
			    [&](const auto &entry) { return entry.first == owner_query_id || entry.second == owner_query_id; });
		};
		result_handle_condition_.wait(guard, [&]() { return !owner_registration_in_progress(); });
		const bool had_active_operations = active_result_handle_operations_by_owner_.find(owner_query_id) !=
		                                   active_result_handle_operations_by_owner_.end();
		auto query_ids = ResultHandleQueryIdsForOwnerLocked(owner_query_id);
		std::vector<string> ordered_query_ids(query_ids.begin(), query_ids.end());
		std::sort(ordered_query_ids.begin(), ordered_query_ids.end(), [&](const string &lhs, const string &rhs) {
			if ((lhs == owner_query_id) != (rhs == owner_query_id)) {
				return lhs != owner_query_id;
			}
			return lhs < rhs;
		});
		return ResultHandleQuiesce {owner_query_id, std::move(ordered_query_ids), token, had_active_operations};
	}

	std::optional<ResultHandleQuiesce> BeginResultHandleQuiesce(const string &query_id) {
		if (Py_IsInitialized() && !duckdb::PythonIsFinalizing() && PyGILState_Check()) {
			py::gil_scoped_release release;
			return BeginResultHandleQuiesceWithoutGIL(query_id);
		}
		return BeginResultHandleQuiesceWithoutGIL(query_id);
	}

	void EndResultHandleQuiesce(const string &owner_query_id, uint64_t token, bool succeeded) {
		{
			lock_guard<mutex> guard(mutex_);
			auto current = quiescing_result_handle_query_owners_.find(owner_query_id);
			if (current != quiescing_result_handle_query_owners_.end() && current->second == token) {
				quiescing_result_handle_query_owners_.erase(current);
				if (succeeded) {
					quiesced_result_handle_query_owners_.insert(owner_query_id);
				}
			}
		}
		result_handle_condition_.notify_all();
	}

	std::optional<string> BeginResultHandleOperation(const string &query_id,
	                                                 const string &requested_owner_query_id = string()) {
		if (query_id.empty()) {
			throw std::invalid_argument("result-handle operation requires non-empty query_id");
		}
		lock_guard<mutex> guard(mutex_);
		auto existing_owner = result_handle_owner_by_query_.find(query_id);
		if (existing_owner == result_handle_owner_by_query_.end()) {
			return std::nullopt;
		}
		const auto owner_query_id =
		    requested_owner_query_id.empty() ? existing_owner->second : requested_owner_query_id;
		if (owner_query_id.empty()) {
			throw std::invalid_argument("result-handle operation requires non-empty owner query_id");
		}
		if (existing_owner->second != owner_query_id) {
			throw std::runtime_error("FTE query result owner changed while active: query=" + query_id +
			                         " existing=" + existing_owner->second + " requested=" + owner_query_id);
		}
		if (all_result_handle_ingress_closed_ ||
		    closed_result_handle_queries_.find(query_id) != closed_result_handle_queries_.end() ||
		    closed_result_handle_query_owners_.find(owner_query_id) != closed_result_handle_query_owners_.end() ||
		    dropping_result_handle_query_owners_.find(owner_query_id) != dropping_result_handle_query_owners_.end()) {
			return std::nullopt;
		}
		result_handle_owner_by_query_[query_id] = owner_query_id;
		active_result_handle_operations_by_owner_[owner_query_id]++;
		return owner_query_id;
	}

	void EndResultHandleOperation(const string &owner_query_id) {
		{
			lock_guard<mutex> guard(mutex_);
			auto active = active_result_handle_operations_by_owner_.find(owner_query_id);
			D_ASSERT(active != active_result_handle_operations_by_owner_.end());
			D_ASSERT(active->second > 0);
			if (--active->second == 0) {
				active_result_handle_operations_by_owner_.erase(active);
			}
		}
		result_handle_condition_.notify_all();
	}

	void WaitForResultHandleOperationsWithoutGIL(const string &owner_query_id) {
		std::unique_lock<mutex> guard(mutex_);
		result_handle_condition_.wait(guard, [&]() {
			return active_result_handle_operations_by_owner_.find(owner_query_id) ==
			       active_result_handle_operations_by_owner_.end();
		});
	}

	void WaitForResultHandleOperations(const string &owner_query_id) {
		if (Py_IsInitialized() && !duckdb::PythonIsFinalizing() && PyGILState_Check()) {
			py::gil_scoped_release release;
			WaitForResultHandleOperationsWithoutGIL(owner_query_id);
			return;
		}
		WaitForResultHandleOperationsWithoutGIL(owner_query_id);
	}

	void WaitForAllResultHandleOperationsWithoutGIL() {
		std::unique_lock<mutex> guard(mutex_);
		result_handle_condition_.wait(guard, [&]() { return active_result_handle_operations_by_owner_.empty(); });
	}

	void WaitForAllResultHandleOperations() {
		if (Py_IsInitialized() && !duckdb::PythonIsFinalizing() && PyGILState_Check()) {
			py::gil_scoped_release release;
			WaitForAllResultHandleOperationsWithoutGIL();
			return;
		}
		WaitForAllResultHandleOperationsWithoutGIL();
	}

	std::unordered_set<string> ResultHandleQueryIdsForOwnerLocked(const string &owner_query_id) const {
		std::unordered_set<string> query_ids {owner_query_id};
		for (const auto &entry : result_handle_owner_by_query_) {
			if (entry.second == owner_query_id) {
				query_ids.insert(entry.first);
			}
		}
		auto collect_owned_keys = [&](const auto &handles_by_query) {
			for (const auto &entry : handles_by_query) {
				if (ResultHandleOwnerForQueryLocked(entry.first) == owner_query_id) {
					query_ids.insert(entry.first);
				}
			}
		};
		collect_owned_keys(result_handles_by_query_);
		collect_owned_keys(retained_result_handles_by_query_);
		collect_owned_keys(cleanup_retry_result_handles_by_query_);
		return query_ids;
	}

	bool ResultHandleOwnerHasHandlesLocked(const string &owner_query_id) const {
		auto has_owned_handles = [&](const auto &handles_by_query) {
			for (const auto &entry : handles_by_query) {
				if (!entry.second.empty() && ResultHandleOwnerForQueryLocked(entry.first) == owner_query_id) {
					return true;
				}
			}
			return false;
		};
		return has_owned_handles(result_handles_by_query_) || has_owned_handles(retained_result_handles_by_query_) ||
		       has_owned_handles(cleanup_retry_result_handles_by_query_);
	}

	std::optional<ResultHandleDrop> BeginResultHandleDropWithoutGIL(const string &query_id) {
		std::unique_lock<mutex> guard(mutex_);
		auto registration_in_progress = [&]() {
			return std::any_of(registering_result_handle_owner_by_query_.begin(),
			                   registering_result_handle_owner_by_query_.end(),
			                   [&](const auto &entry) { return entry.first == query_id || entry.second == query_id; });
		};
		result_handle_condition_.wait(guard, [&]() { return !registration_in_progress(); });
		auto mapped_owner = result_handle_owner_by_query_.find(query_id);
		string owner_query_id;
		if (mapped_owner != result_handle_owner_by_query_.end()) {
			owner_query_id = mapped_owner->second;
		} else {
			const bool known_owner =
			    std::any_of(result_handle_owner_by_query_.begin(), result_handle_owner_by_query_.end(),
			                [&](const auto &entry) { return entry.second == query_id; });
			if (!known_owner &&
			    dropping_result_handle_query_owners_.find(query_id) == dropping_result_handle_query_owners_.end()) {
				return std::nullopt;
			}
			owner_query_id = query_id;
		}
		auto owner_registration_in_progress = [&]() {
			return std::any_of(
			    registering_result_handle_owner_by_query_.begin(), registering_result_handle_owner_by_query_.end(),
			    [&](const auto &entry) { return entry.first == owner_query_id || entry.second == owner_query_id; });
		};
		result_handle_condition_.wait(guard, [&]() { return !owner_registration_in_progress(); });
		result_handle_condition_.wait(guard, [&]() {
			return result_handle_quiesce_waiters_by_owner_.find(owner_query_id) ==
			       result_handle_quiesce_waiters_by_owner_.end();
		});
		auto joined_drop = dropping_result_handle_query_owners_.find(owner_query_id);
		if (joined_drop != dropping_result_handle_query_owners_.end()) {
			const auto joined_drop_token = joined_drop->second;
			result_handle_condition_.wait(guard, [&]() {
				auto current = dropping_result_handle_query_owners_.find(owner_query_id);
				return current == dropping_result_handle_query_owners_.end() || current->second != joined_drop_token;
			});
			if (closed_result_handle_query_owners_.find(owner_query_id) == closed_result_handle_query_owners_.end()) {
				return std::nullopt;
			}
			if (dropping_result_handle_query_owners_.find(owner_query_id) !=
			    dropping_result_handle_query_owners_.end()) {
				return std::nullopt;
			}
		}
		if (all_result_handle_ingress_closed_) {
			while (result_handle_shutdown_running_) {
				result_handle_condition_.wait(guard);
			}
			if (result_handle_shutdown_finished_) {
				return std::nullopt;
			}
			throw std::runtime_error("cannot drop FTE query after Python backend shutdown failed: " + query_id);
		}
		if (quiesced_result_handle_query_owners_.find(owner_query_id) == quiesced_result_handle_query_owners_.end()) {
			throw std::runtime_error("cannot drop FTE query before its abort barrier: " + query_id);
		}
		closed_result_handle_queries_.insert(query_id);
		closed_result_handle_query_owners_.insert(owner_query_id);
		const auto drop_token = next_result_handle_drop_token_++;
		if (next_result_handle_drop_token_ == 0) {
			next_result_handle_drop_token_ = 1;
		}
		dropping_result_handle_query_owners_[owner_query_id] = drop_token;
		return ResultHandleDrop {owner_query_id, drop_token};
	}

	std::optional<ResultHandleDrop> BeginResultHandleDrop(const string &query_id) {
		if (Py_IsInitialized() && !duckdb::PythonIsFinalizing() && PyGILState_Check()) {
			py::gil_scoped_release release;
			return BeginResultHandleDropWithoutGIL(query_id);
		}
		return BeginResultHandleDropWithoutGIL(query_id);
	}

	void EndResultHandleDrop(const string &owner_query_id, uint64_t drop_token) {
		{
			lock_guard<mutex> guard(mutex_);
			auto current = dropping_result_handle_query_owners_.find(owner_query_id);
			if (current != dropping_result_handle_query_owners_.end() && current->second == drop_token) {
				dropping_result_handle_query_owners_.erase(current);
			}
		}
		result_handle_condition_.notify_all();
	}

	bool BeginResultHandleShutdownWithoutGIL() {
		std::unique_lock<mutex> guard(mutex_);
		while (result_handle_shutdown_running_) {
			result_handle_condition_.wait(guard);
		}
		if (result_handle_shutdown_finished_) {
			return false;
		}
		all_result_handle_ingress_closed_ = true;
		result_handle_shutdown_running_ = true;
		result_handle_condition_.wait(guard, [&]() {
			return registering_result_handle_owner_by_query_.empty() && quiescing_result_handle_query_owners_.empty() &&
			       result_handle_quiesce_waiters_by_owner_.empty() && dropping_result_handle_query_owners_.empty();
		});
		return true;
	}

	bool BeginResultHandleShutdown() {
		if (Py_IsInitialized() && !duckdb::PythonIsFinalizing() && PyGILState_Check()) {
			py::gil_scoped_release release;
			return BeginResultHandleShutdownWithoutGIL();
		}
		return BeginResultHandleShutdownWithoutGIL();
	}

	void EndResultHandleShutdown(bool succeeded) {
		{
			lock_guard<mutex> guard(mutex_);
			result_handle_shutdown_finished_ = succeeded;
			result_handle_shutdown_running_ = false;
		}
		result_handle_condition_.notify_all();
	}

	void FinishResultHandleIngress(const string &owner_query_id, uint64_t drop_token) {
		{
			lock_guard<mutex> guard(mutex_);
			auto current_drop = dropping_result_handle_query_owners_.find(owner_query_id);
			if (current_drop == dropping_result_handle_query_owners_.end() || current_drop->second != drop_token) {
				throw std::runtime_error("cannot finish stale FTE query result drop: " + owner_query_id);
			}
			if (active_result_handle_operations_by_owner_.find(owner_query_id) !=
			    active_result_handle_operations_by_owner_.end()) {
				throw std::runtime_error("cannot finish active FTE query result lifecycle: " + owner_query_id);
			}
			if (ResultHandleOwnerHasHandlesLocked(owner_query_id)) {
				throw std::runtime_error("cannot finish FTE query lifecycle with pending result cleanup: " +
				                         owner_query_id);
			}
			if (quiesced_result_handle_query_owners_.find(owner_query_id) ==
			    quiesced_result_handle_query_owners_.end()) {
				throw std::runtime_error("cannot finish FTE query lifecycle before quiescence: " + owner_query_id);
			}
			if (result_handle_quiesce_waiters_by_owner_.find(owner_query_id) !=
			    result_handle_quiesce_waiters_by_owner_.end()) {
				throw std::runtime_error("cannot finish FTE query lifecycle with active abort waiters: " +
				                         owner_query_id);
			}
			std::unordered_set<string> owned_query_ids {owner_query_id};
			for (auto entry = result_handle_owner_by_query_.begin(); entry != result_handle_owner_by_query_.end();) {
				if (entry->second == owner_query_id) {
					owned_query_ids.insert(entry->first);
					entry = result_handle_owner_by_query_.erase(entry);
				} else {
					++entry;
				}
			}
			for (const auto &query_id : owned_query_ids) {
				closed_result_handle_queries_.erase(query_id);
			}
			closed_result_handle_query_owners_.erase(owner_query_id);
			quiesced_result_handle_query_owners_.erase(owner_query_id);
			dropping_result_handle_query_owners_.erase(current_drop);
		}
		result_handle_condition_.notify_all();
	}

	void FinishResultHandleShutdown() {
		lock_guard<mutex> guard(mutex_);
		if (!registering_result_handle_owner_by_query_.empty() || !quiescing_result_handle_query_owners_.empty() ||
		    !result_handle_quiesce_waiters_by_owner_.empty() || !dropping_result_handle_query_owners_.empty()) {
			throw std::runtime_error("cannot finish Python backend shutdown with active lifecycle transitions");
		}
		if (!active_result_handle_operations_by_owner_.empty()) {
			throw std::runtime_error("cannot finish Python backend shutdown with active result operations");
		}
		if (!result_handles_by_query_.empty() || !retained_result_handles_by_query_.empty() ||
		    !cleanup_retry_result_handles_by_query_.empty()) {
			throw std::runtime_error("cannot finish Python backend shutdown with pending result cleanup");
		}
		result_handle_owner_by_query_.clear();
		closed_result_handle_queries_.clear();
		closed_result_handle_query_owners_.clear();
		quiesced_result_handle_query_owners_.clear();
	}

	static string PyStringField(const py::dict &dict, const char *field_name, const string &default_value) {
		auto key = py::str(field_name);
		if (!dict.contains(key) || py::reinterpret_borrow<py::object>(dict[key]).is_none()) {
			return default_value;
		}
		auto value = py::str(py::reinterpret_borrow<py::object>(dict[key])).cast<string>();
		return value.empty() ? default_value : value;
	}

	static double PyDoubleField(const py::dict &dict, const char *field_name, double default_value) {
		auto key = py::str(field_name);
		if (!dict.contains(key) || py::reinterpret_borrow<py::object>(dict[key]).is_none()) {
			return default_value;
		}
		return py::float_(py::reinterpret_borrow<py::object>(dict[key])).cast<double>();
	}

	static uint64_t PyUInt64Field(const py::dict &dict, const char *field_name, uint64_t default_value) {
		auto key = py::str(field_name);
		if (!dict.contains(key) || py::reinterpret_borrow<py::object>(dict[key]).is_none()) {
			return default_value;
		}
		return py::int_(py::reinterpret_borrow<py::object>(dict[key])).cast<uint64_t>();
	}

	static string PyStatusMessage(const py::object &status_obj) {
		try {
			return py::str(status_obj).cast<string>();
		} catch (...) {
			return "<unprintable status>";
		}
	}

	static bool RequiredStatusBool(const py::object &status_obj, const char *field_name) {
		if (!py::isinstance<py::dict>(status_obj)) {
			throw duckdb::InternalException("FTE query status must be a dict");
		}
		auto status = status_obj.cast<py::dict>();
		auto key = py::str(field_name);
		if (!status.contains(key)) {
			throw duckdb::InternalException("FTE query status must include boolean '%s'", field_name);
		}
		auto value = py::reinterpret_borrow<py::object>(status[key]);
		if (!py::isinstance<py::bool_>(value)) {
			throw duckdb::InternalException("FTE query status field '%s' must be boolean", field_name);
		}
		return value.cast<bool>();
	}

	static bool OptionalStatusBool(const py::object &status_obj, const char *field_name, bool default_value) {
		if (!py::isinstance<py::dict>(status_obj)) {
			throw duckdb::InternalException("FTE query status must be a dict");
		}
		auto status = status_obj.cast<py::dict>();
		auto key = py::str(field_name);
		if (!status.contains(key) || py::reinterpret_borrow<py::object>(status[key]).is_none()) {
			return default_value;
		}
		auto value = py::reinterpret_borrow<py::object>(status[key]);
		if (!py::isinstance<py::bool_>(value)) {
			throw duckdb::InternalException("FTE query status field '%s' must be boolean", field_name);
		}
		return value.cast<bool>();
	}

	static std::unordered_set<string> SelectedAttemptTaskIds(const py::object &status_obj) {
		std::unordered_set<string> selected;
		if (!py::isinstance<py::dict>(status_obj)) {
			return selected;
		}
		auto status = status_obj.cast<py::dict>();
		auto key = py::str("selected_attempt_task_ids");
		if (!status.contains(key)) {
			return selected;
		}
		auto selected_obj = py::reinterpret_borrow<py::object>(status[key]);
		if (selected_obj.is_none()) {
			return selected;
		}
		for (auto item : selected_obj) {
			auto value = py::str(py::reinterpret_borrow<py::object>(item)).cast<string>();
			if (!value.empty()) {
				selected.insert(std::move(value));
			}
		}
		return selected;
	}

	void StorePythonResultHandles(const string &query_id, const string &owner_query_id, py::object raw_handles) {
		if (query_id.empty() || raw_handles.is_none()) {
			return;
		}
		std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> wrapped;
		for (auto item : raw_handles) {
			auto py_handle = py::reinterpret_borrow<py::object>(item);
			auto handle = duckdb::distributed::python::ray::MakePythonTaskResultHandle(std::move(py_handle));
			wrapped.push_back(make_uniq<duckdb::distributed::python::ray::PythonTaskResultHandle>(std::move(handle)));
		}
		if (wrapped.empty()) {
			return;
		}
		bool query_closed = false;
		{
			lock_guard<mutex> guard(mutex_);
			query_closed =
			    all_result_handle_ingress_closed_ ||
			    closed_result_handle_query_owners_.find(owner_query_id) != closed_result_handle_query_owners_.end();
			if (!query_closed) {
				auto &stored = result_handles_by_query_[query_id];
				stored.reserve(stored.size() + wrapped.size());
				for (auto &handle : wrapped) {
					stored.push_back(std::move(handle));
				}
			}
		}
		if (query_closed) {
			ReleaseLatePythonTaskResultHandles(query_id, std::move(wrapped));
		}
	}

	void StorePythonTaskResultHandles(
	    const string &query_id,
	    std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> handles) {
		if (query_id.empty() || handles.empty()) {
			return;
		}
		bool query_closed = false;
		{
			lock_guard<mutex> guard(mutex_);
			const auto owner_query_id = ResultHandleOwnerForQueryLocked(query_id);
			query_closed =
			    all_result_handle_ingress_closed_ ||
			    closed_result_handle_query_owners_.find(owner_query_id) != closed_result_handle_query_owners_.end();
			if (!query_closed) {
				auto &stored = result_handles_by_query_[query_id];
				stored.reserve(stored.size() + handles.size());
				for (auto &handle : handles) {
					stored.push_back(std::move(handle));
				}
			}
		}
		if (query_closed) {
			ReleaseLatePythonTaskResultHandles(query_id, std::move(handles));
		}
	}

	void RetainPythonTaskResultHandles(
	    const string &query_id,
	    std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> handles) {
		if (query_id.empty() || handles.empty()) {
			return;
		}
		// These handles back outputs that the current operation has already
		// materialized and is about to hand downstream. A concurrent drop may
		// close ingress, but it must wait for that operation before releasing
		// this ownership. Late, not-yet-consumed ingress uses
		// StorePythonResultHandles/StorePythonTaskResultHandles instead.
		lock_guard<mutex> guard(mutex_);
		auto &retained = retained_result_handles_by_query_[query_id];
		retained.reserve(retained.size() + handles.size());
		for (auto &handle : handles) {
			retained.push_back(std::move(handle));
		}
	}

	void StoreCleanupRetryResultHandles(
	    const string &query_id,
	    std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> handles) {
		if (handles.empty()) {
			return;
		}
		lock_guard<mutex> guard(mutex_);
		auto &retry = cleanup_retry_result_handles_by_query_[query_id];
		retry.reserve(retry.size() + handles.size());
		for (auto &handle : handles) {
			retry.push_back(std::move(handle));
		}
	}

	void ReleaseLatePythonTaskResultHandles(
	    const string &query_id,
	    std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> handles) {
		std::vector<string> errors;
		std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> retry_handles;
		for (size_t index = 0; index < handles.size(); index++) {
			try {
				handles[index]->ReleasePollResult();
			} catch (const std::exception &ex) {
				errors.push_back("late[" + std::to_string(index) + "]: " + ex.what());
				retry_handles.push_back(std::move(handles[index]));
			} catch (...) {
				errors.push_back("late[" + std::to_string(index) + "]: unknown release error");
				retry_handles.push_back(std::move(handles[index]));
			}
		}
		StoreCleanupRetryResultHandles(query_id, std::move(retry_handles));
		if (!errors.empty()) {
			string message = "failed to release late Python backend result handle(s) for closed query " + query_id;
			for (const auto &error : errors) {
				message += "; " + error;
			}
			throw std::runtime_error(message);
		}
	}

	void ClearResultHandles(const string &query_id) {
		std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> handles;
		std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> retained_handles;
		std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> cleanup_retry_handles;
		bool query_closed = false;
		{
			lock_guard<mutex> guard(mutex_);
			const auto owner_query_id = ResultHandleOwnerForQueryLocked(query_id);
			query_closed =
			    all_result_handle_ingress_closed_ ||
			    closed_result_handle_query_owners_.find(owner_query_id) != closed_result_handle_query_owners_.end();
			auto it = result_handles_by_query_.find(query_id);
			if (it != result_handles_by_query_.end()) {
				handles = std::move(it->second);
				result_handles_by_query_.erase(it);
			}
			auto retained_it = retained_result_handles_by_query_.find(query_id);
			if (retained_it != retained_result_handles_by_query_.end()) {
				retained_handles = std::move(retained_it->second);
				retained_result_handles_by_query_.erase(retained_it);
			}
			auto cleanup_retry_it = cleanup_retry_result_handles_by_query_.find(query_id);
			if (cleanup_retry_it != cleanup_retry_result_handles_by_query_.end()) {
				cleanup_retry_handles = std::move(cleanup_retry_it->second);
				cleanup_retry_result_handles_by_query_.erase(cleanup_retry_it);
			}
		}
		std::vector<string> errors;
		std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> retry_handles;
		auto release_all = [&](auto &owned_handles, const char *kind) {
			for (size_t index = 0; index < owned_handles.size(); index++) {
				try {
					owned_handles[index]->ReleasePollResult();
				} catch (const std::exception &ex) {
					errors.push_back(string(kind) + "[" + std::to_string(index) + "]: " + ex.what());
					retry_handles.push_back(std::move(owned_handles[index]));
				} catch (...) {
					errors.push_back(string(kind) + "[" + std::to_string(index) + "]: unknown release error");
					retry_handles.push_back(std::move(owned_handles[index]));
				}
			}
		};
		release_all(handles, "pending");
		release_all(retained_handles, "retained");
		release_all(cleanup_retry_handles, "cleanup-retry");
		if (query_closed) {
			StoreCleanupRetryResultHandles(query_id, std::move(retry_handles));
		} else {
			StorePythonTaskResultHandles(query_id, std::move(retry_handles));
		}
		if (!errors.empty()) {
			string message = "failed to release " + std::to_string(errors.size()) + " backend result handle(s)";
			for (const auto &error : errors) {
				message += "; " + error;
			}
			throw std::runtime_error(message);
		}
	}

	void ClearResultHandlesForOwner(const string &owner_query_id) {
		std::unordered_set<string> query_ids;
		{
			lock_guard<mutex> guard(mutex_);
			query_ids = ResultHandleQueryIdsForOwnerLocked(owner_query_id);
		}
		std::vector<string> errors;
		for (const auto &query_id : query_ids) {
			try {
				ClearResultHandles(query_id);
			} catch (const std::exception &ex) {
				errors.push_back(query_id + ": " + ex.what());
			} catch (...) {
				errors.push_back(query_id + ": unknown cleanup error");
			}
		}
		if (!errors.empty()) {
			string message = "failed to clear Python backend result handles for owner " + owner_query_id;
			for (const auto &error : errors) {
				message += "; " + error;
			}
			throw std::runtime_error(message);
		}
	}

	void ClearAllResultHandles() {
		std::unordered_set<string> query_ids;
		{
			lock_guard<mutex> guard(mutex_);
			for (const auto &entry : result_handles_by_query_) {
				query_ids.insert(entry.first);
			}
			for (const auto &entry : retained_result_handles_by_query_) {
				query_ids.insert(entry.first);
			}
			for (const auto &entry : cleanup_retry_result_handles_by_query_) {
				query_ids.insert(entry.first);
			}
		}
		std::vector<string> errors;
		for (const auto &query_id : query_ids) {
			try {
				ClearResultHandles(query_id);
			} catch (const std::exception &ex) {
				errors.push_back(query_id + ": " + ex.what());
			} catch (...) {
				errors.push_back(query_id + ": unknown cleanup error");
			}
		}
		if (!errors.empty()) {
			string message = "failed to clear Python backend result handles";
			for (const auto &error : errors) {
				message += "; " + error;
			}
			throw std::runtime_error(message);
		}
	}

	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>
	DrainResultHandles(const string &query_id, double timeout_s,
	                   const std::unordered_set<string> &selected_attempt_task_ids,
	                   const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash>
	                       *task_context_filter,
	                   bool release_payloads) {
		std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> handles;
		{
			lock_guard<mutex> guard(mutex_);
			auto it = result_handles_by_query_.find(query_id);
			if (it != result_handles_by_query_.end()) {
				auto stored_handles = std::move(it->second);
				result_handles_by_query_.erase(it);
				if (task_context_filter && !task_context_filter->empty()) {
					std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>>
					    retained_handles;
					for (auto &handle : stored_handles) {
						if (task_context_filter->find(handle->GetTaskContext()) == task_context_filter->end()) {
							retained_handles.push_back(std::move(handle));
						} else {
							handles.push_back(std::move(handle));
						}
					}
					if (!retained_handles.empty()) {
						auto &retained = result_handles_by_query_[query_id];
						for (auto &handle : retained_handles) {
							retained.push_back(std::move(handle));
						}
					}
				} else {
					handles = std::move(stored_handles);
				}
			}
		}

		std::vector<duckdb::distributed::MaterializedOutput> outputs;
		std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> retained_handles;
		std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> pending =
		    std::move(handles);
		if (!selected_attempt_task_ids.empty()) {
			std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> selected_pending;
			std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> retry_handles;
			std::vector<string> release_errors;
			selected_pending.reserve(pending.size());
			for (size_t index = 0; index < pending.size(); index++) {
				auto &handle = pending[index];
				if (selected_attempt_task_ids.find(handle->GetFteTaskId()) == selected_attempt_task_ids.end()) {
					try {
						handle->AckPollResult();
						handle->ReleasePollResult();
					} catch (const std::exception &ex) {
						release_errors.push_back("unselected[" + std::to_string(index) + "]: " + ex.what());
						retry_handles.push_back(std::move(handle));
					} catch (...) {
						release_errors.push_back("unselected[" + std::to_string(index) + "]: unknown release error");
						retry_handles.push_back(std::move(handle));
					}
					continue;
				}
				selected_pending.push_back(std::move(handle));
			}
			pending = std::move(selected_pending);
			StorePythonTaskResultHandles(query_id, std::move(retry_handles));
			if (!release_errors.empty()) {
				StorePythonTaskResultHandles(query_id, std::move(pending));
				string message = "failed to release unselected Python backend result handle(s)";
				for (const auto &error : release_errors) {
					message += "; " + error;
				}
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(DuckDBError(message));
			}
		}
		const bool has_deadline = timeout_s > 0.0;
		const auto deadline = has_deadline ? std::chrono::steady_clock::now() +
		                                         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
		                                             std::chrono::duration<double>(timeout_s))
		                                   : std::chrono::steady_clock::time_point::max();
		while (!pending.empty()) {
			std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> still_pending;
			std::vector<string> drain_errors;
			for (size_t index = 0; index < pending.size(); index++) {
				auto &handle = pending[index];
				auto polled = handle->poll();
				if (!polled.first) {
					still_pending.push_back(std::move(handle));
					continue;
				}
				if (polled.second.is_err()) {
					drain_errors.push_back(polled.second.error().what());
					still_pending.push_back(std::move(handle));
					continue;
				}
				auto payload = std::move(polled.second).value();
				const bool produced_output = payload.first;
				bool retain_payload_until_query_cleanup = false;
				if (produced_output) {
					retain_payload_until_query_cleanup =
					    !release_payloads && !payload.second.has_exchange_sink_instance();
					outputs.push_back(std::move(payload.second));
				}
				try {
					handle->AckPollResult();
					if (retain_payload_until_query_cleanup) {
						retained_handles.push_back(std::move(handle));
					} else {
						handle->ReleasePollResult();
					}
				} catch (const std::exception &ex) {
					still_pending.push_back(std::move(handle));
					drain_errors.push_back(string("failed to finalize Python backend result handle: ") + ex.what());
				} catch (...) {
					still_pending.push_back(std::move(handle));
					drain_errors.push_back("failed to finalize Python backend result handle: unknown error");
				}
			}
			if (!drain_errors.empty()) {
				StorePythonTaskResultHandles(query_id, std::move(still_pending));
				RetainPythonTaskResultHandles(query_id, std::move(retained_handles));
				string message = "failed to drain Python backend result handle(s)";
				for (const auto &error : drain_errors) {
					message += "; " + error;
				}
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(DuckDBError(message));
			}
			if (still_pending.empty()) {
				break;
			}
			if (has_deadline && std::chrono::steady_clock::now() >= deadline) {
				StorePythonTaskResultHandles(query_id, std::move(still_pending));
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError("timed out draining Python backend FTE result handles"));
			}
			pending = std::move(still_pending);
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		RetainPythonTaskResultHandles(query_id, std::move(retained_handles));
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::ok(std::move(outputs));
	}
};

static void CleanupQueryOwnedPythonState(const string &query_id) {
	std::exception_ptr datasource_cleanup_error;
	try {
		ReleaseDataSourceFactoriesForQuery(query_id);
	} catch (...) {
		datasource_cleanup_error = std::current_exception();
	}
	CleanupQueryPythonReplayState(query_id);
	if (datasource_cleanup_error) {
		std::rethrow_exception(datasource_cleanup_error);
	}
}

struct PyPhysicalPlanWrapperRunner {
	std::shared_ptr<duckdb::distributed::WorkerManager> worker_manager_;
	std::shared_ptr<duckdb::distributed::python::ray::RayWorkerManager> ray_worker_manager_;
	std::shared_ptr<PyBackendWorkerManager> py_backend_worker_manager_;
	// Stored copy plan streaming context for two-phase COPY
	struct StoredCopyContext {
		duckdb::distributed::DistributedCopySpec copy_spec;
		std::string staging_root;
	};
	std::unordered_map<string, std::shared_ptr<StoredCopyContext>> streaming_results_;
	std::atomic<bool> fail_next_extension_result_marshalling_for_test_ {false};

	PyPhysicalPlanWrapperRunner()
	    : ray_worker_manager_(
	          std::make_shared<duckdb::distributed::python::ray::RayWorkerManager>(CleanupQueryOwnedPythonState)) {
		worker_manager_ = ray_worker_manager_;
	}

	explicit PyPhysicalPlanWrapperRunner(py::object backend) {
		if (backend.is_none()) {
			ray_worker_manager_ =
			    std::make_shared<duckdb::distributed::python::ray::RayWorkerManager>(CleanupQueryOwnedPythonState);
			worker_manager_ = ray_worker_manager_;
			return;
		}
		py_backend_worker_manager_ =
		    std::make_shared<PyBackendWorkerManager>(std::move(backend), CleanupQueryOwnedPythonState);
		worker_manager_ = py_backend_worker_manager_;
	}

	void register_query_owner(const string &query_id, const string &owner_query_id,
	                          bool *lifecycle_published = nullptr) {
		if (lifecycle_published) {
			*lifecycle_published = false;
		}
		if (ray_worker_manager_) {
			ray_worker_manager_->register_query_owner(query_id, owner_query_id);
			if (lifecycle_published) {
				*lifecycle_published = true;
			}
		} else if (py_backend_worker_manager_) {
			py_backend_worker_manager_->register_query_owner(query_id, owner_query_id, {}, lifecycle_published);
		}
	}

	void drop_query_fragments(const string &query_id) {
		if (query_id.empty()) {
			return;
		}
		std::exception_ptr teardown_error;
		try {
			if (ray_worker_manager_) {
				ray_worker_manager_->drop_query_fragments(query_id);
			} else if (py_backend_worker_manager_) {
				py_backend_worker_manager_->drop_query_fragments(query_id);
			}
		} catch (...) {
			teardown_error = std::current_exception();
		}
		if (teardown_error) {
			// Teardown did not finish. Its lifecycle mapping and any owner state not
			// already released remain available to a successful retry or shutdown.
			std::rethrow_exception(teardown_error);
		}
	}

	void warm_up() {
		worker_manager_->worker_snapshots();
	}

	void shutdown() {
		auto result = worker_manager_->shutdown();
		if (result.is_err()) {
			throw std::runtime_error(result.error().what());
		}
	}

	void fail_next_extension_result_marshalling_for_test() {
		fail_next_extension_result_marshalling_for_test_.store(true, std::memory_order_relaxed);
	}

	void close_session(const string &session_id) {
		if (!ray_worker_manager_) {
			throw std::runtime_error("Vane session lifecycle requires the Ray worker manager");
		}
		auto result = ray_worker_manager_->close_session(session_id);
		if (result.is_err()) {
			throw std::runtime_error(result.error().what());
		}
	}

	std::unordered_map<string, std::unordered_map<string, idx_t>> fragment_stats_by_worker() const {
		if (ray_worker_manager_) {
			return ray_worker_manager_->fragment_stats_by_worker();
		}
		if (py_backend_worker_manager_) {
			return py_backend_worker_manager_->fragment_stats_by_worker();
		}
		return {};
	}

	void rethrow_submission_error(const string &query_id, const string &details = string()) {
		if (ray_worker_manager_) {
			ray_worker_manager_->rethrow_submission_error(query_id, details);
		} else if (py_backend_worker_manager_) {
			py_backend_worker_manager_->rethrow_submission_error(query_id, details);
		}
	}

	std::shared_ptr<ResultPartitionStream> run_plan(const PyPhysicalPlanWrapper &plan,
	                                                duckdb::shared_ptr<duckdb::ClientContext> client_context = nullptr,
	                                                duckdb::distributed::python::ray::SafePyObject py_conn_keepalive =
	                                                    duckdb::distributed::python::ray::SafePyObject()) {
		using namespace duckdb::distributed;
		auto run_state = std::make_shared<PlanRunState>();
		run_state->client_context = client_context;
		run_state->py_conn_keepalive = std::move(py_conn_keepalive);
		auto exception_message = [](const std::exception_ptr &error) {
			try {
				std::rethrow_exception(error);
			} catch (const std::exception &e) {
				return string(e.what());
			} catch (...) {
				return string("unknown exception");
			}
		};
		bool query_owner_registered = false;
		auto cleanup_failed_start = [&](const std::exception_ptr &execution_error) {
			if (!query_owner_registered) {
				return;
			}
			std::exception_ptr cleanup_error;
			try {
				drop_query_fragments(plan.idx());
			} catch (...) {
				cleanup_error = std::current_exception();
			}
			if (cleanup_error) {
				throw std::runtime_error("distributed streaming plan startup and teardown both failed: startup=" +
				                         exception_message(execution_error) +
				                         "; teardown=" + exception_message(cleanup_error));
			}
		};

		// Call PlanRunner::run_plan
		try {
			const bool replay_state_created = RegisterQueryPythonReplayState(
			    plan.resource_query_id_, plan.udf_registrations_, plan.udf_actor_handles_, plan.connection_snapshot_,
			    plan.scoped_secret_refs_);
			try {
				register_query_owner(plan.idx(), plan.resource_query_id_, &query_owner_registered);
			} catch (...) {
				if (!query_owner_registered && replay_state_created) {
					CleanupQueryPythonReplayState(plan.resource_query_id_);
				}
				throw;
			}
			query_owner_registered = true;
			// Defensive check: ensure the distributed physical plan includes a valid physical plan
			if (!plan.plan_) {
				string msg =
				    string("DistributedPhysicalPlan missing underlying physical plan for query_id=") + plan.idx();
				throw py::value_error(msg);
			}

			// Check if physical plan has a root
			bool has_root = plan.plan_->physical_plan() && plan.plan_->physical_plan()->HasRoot();

			if (!has_root) {
				// Plan has no root - this is an empty placeholder from serialization
				// This is not supported yet - need to implement actual plan serialization
				string msg =
				    string("DistributedPhysicalPlan has no root (empty placeholder) for query_id=") + plan.idx() +
				    ". Physical plan serialization is not yet implemented. "
				    "This likely means the plan was serialized to a Ray worker, which is not currently supported.";
				throw py::value_error(msg);
			}

			auto runner = std::make_shared<duckdb::distributed::PlanRunner>(worker_manager_, client_context);
			run_state->runner = runner;
			DuckDBResult<duckdb::distributed::PlanRunner::PlanResult> res;
			{
				py::gil_scoped_release release;
				res = runner->run_plan(plan.plan_);
			}

			if (!res.is_ok()) {
				rethrow_submission_error(plan.idx(), res.error().what());
				throw py::value_error(res.error().what());
			}
			rethrow_submission_error(plan.idx());

			auto &plan_result = res.value();
			if (plan_result.tag != duckdb::distributed::PlanRunner::PlanResult::STREAMING) {
				throw py::value_error("run_plan returned copy result instead of stream for a streaming plan");
			}
			auto stream = std::make_shared<duckdb::distributed::PlanResultStream>(std::move(plan_result.stream));
			auto py_stream = std::make_shared<ResultPartitionStream>(stream);
			auto query_id = plan.idx();
			if (ray_worker_manager_) {
				auto worker_manager = ray_worker_manager_;
				py_stream->rethrow_pending_error_ = [worker_manager, query_id]() {
					worker_manager->rethrow_submission_error(query_id);
				};
			} else if (py_backend_worker_manager_) {
				auto worker_manager = py_backend_worker_manager_;
				py_stream->rethrow_pending_error_ = [worker_manager, query_id]() {
					worker_manager->rethrow_submission_error(query_id);
				};
			}
			py_stream->keepalive_ = run_state;
			return py_stream;
		} catch (const py::error_already_set &ex) {
			auto execution_error = std::current_exception();
			cleanup_failed_start(execution_error);
			throw;
		} catch (const std::exception &ex) {
			auto execution_error = std::current_exception();
			cleanup_failed_start(execution_error);
			throw py::value_error(string("C++ run_plan failed: ") + ex.what());
		} catch (...) {
			auto execution_error = std::current_exception();
			cleanup_failed_start(execution_error);
			throw;
		}
	}

	py::object run_copy_plan(const PyPhysicalPlanWrapper &plan,
	                         duckdb::shared_ptr<duckdb::ClientContext> client_context = nullptr,
	                         duckdb::distributed::python::ray::SafePyObject py_conn_keepalive =
	                             duckdb::distributed::python::ray::SafePyObject(),
	                         duckdb::distributed::python::ray::SafePyObject on_execution_started =
	                             duckdb::distributed::python::ray::SafePyObject()) {
		using namespace duckdb::distributed;
		(void)py_conn_keepalive;
		auto copy_started = std::chrono::steady_clock::now();
		idx_t cleanup_ms = 0;
		auto elapsed_ms = [](std::chrono::steady_clock::time_point started) -> idx_t {
			auto elapsed =
			    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started);
			if (elapsed.count() <= 0) {
				return 0;
			}
			auto max_value = static_cast<unsigned long long>(std::numeric_limits<idx_t>::max());
			auto value = static_cast<unsigned long long>(elapsed.count());
			return static_cast<idx_t>(value > max_value ? max_value : value);
		};
		bool cleanup_done = false;
		bool body_succeeded = false;
		bool query_owner_registered = false;
		std::exception_ptr cleanup_error;
		enum class ExtensionWriteTerminalState : uint8_t { NONE, COMMITTED, OUTCOME_UNKNOWN };
		ExtensionWriteTerminalState extension_terminal_state = ExtensionWriteTerminalState::NONE;
		string extension_terminal_detail;
		string extension_terminal_base_path;
		string extension_terminal_run_id;
		string extension_terminal_manifest_path;
		string extension_terminal_committed_marker_path;
		vector<string> post_commit_warnings;
		auto exception_message = [](const std::exception_ptr &error) {
			try {
				std::rethrow_exception(error);
			} catch (const std::exception &e) {
				return string(e.what());
			} catch (...) {
				return string("unknown exception");
			}
		};
		auto cleanup = [&]() {
			if (cleanup_done) {
				return;
			}
			cleanup_done = true;
			if (!query_owner_registered) {
				return;
			}
			auto cleanup_started = std::chrono::steady_clock::now();
			try {
				drop_query_fragments(plan.idx());
			} catch (...) {
				cleanup_error = std::current_exception();
			}
			cleanup_ms = elapsed_ms(cleanup_started);
		};
		auto raise_extension_terminal_error = [&](const std::exception_ptr &execution_error) -> void {
			vector<string> cleanup_warnings(post_commit_warnings.begin(), post_commit_warnings.end());
			if (cleanup_error) {
				cleanup_warnings.push_back("native COPY runner cleanup failed: " + exception_message(cleanup_error));
			}
			py::tuple cleanup_warnings_py(cleanup_warnings.size());
			for (idx_t index = 0; index < cleanup_warnings.size(); index++) {
				cleanup_warnings_py[index] = py::str(cleanup_warnings[index]);
			}

			auto result_error = exception_message(execution_error);
			py::object error_type;
			py::object error_value;
			if (extension_terminal_state == ExtensionWriteTerminalState::COMMITTED) {
				error_type = py::module_::import("vane.runners.copy_outcome").attr("CopyResultUnavailableError");
				error_value = error_type(
				    plan.idx(), "extension write result handling failed after the catalog commit: " + result_error,
				    cleanup_warnings_py);
			} else {
				auto detail = extension_terminal_detail;
				if (!detail.empty()) {
					detail += "; ";
				}
				detail += "extension write result handling failed after finalization: " + result_error;
				error_type = py::module_::import("vane.runners.copy_outcome").attr("CopyOutcomeUnknownError");
				error_value = error_type(plan.idx(), extension_terminal_base_path, extension_terminal_run_id,
				                         extension_terminal_manifest_path, extension_terminal_committed_marker_path,
				                         detail, cleanup_warnings_py);
			}
			PyErr_SetObject(error_type.ptr(), error_value.ptr());
			throw py::error_already_set();
		};

		try {
			const bool replay_state_created = RegisterQueryPythonReplayState(
			    plan.resource_query_id_, plan.udf_registrations_, plan.udf_actor_handles_, plan.connection_snapshot_,
			    plan.scoped_secret_refs_);
			try {
				register_query_owner(plan.idx(), plan.resource_query_id_, &query_owner_registered);
			} catch (...) {
				if (!query_owner_registered && replay_state_created) {
					CleanupQueryPythonReplayState(plan.resource_query_id_);
				}
				throw;
			}
			query_owner_registered = true;
			if (!plan.plan_) {
				throw py::value_error("DistributedPhysicalPlan missing underlying physical plan");
			}
			bool has_root = plan.plan_->physical_plan() && plan.plan_->physical_plan()->HasRoot();
			if (!has_root) {
				throw py::value_error("DistributedPhysicalPlan has no root (empty placeholder)");
			}

			auto runner = std::make_shared<duckdb::distributed::PlanRunner>(worker_manager_, client_context);

			DuckDBResult<DistributedCopyResult> res;
			bool extension_write = false;
			DistributedExtensionWriteResult extension_result;
			idx_t run_plan_ms = 0;
			{
				auto run_plan_started = std::chrono::steady_clock::now();
				py::gil_scoped_release release;
				DuckDBResult<duckdb::distributed::PlanRunner::PlanResult> plan_res;
				auto publish_execution_started = [&]() {
					if (!on_execution_started.has_value()) {
						return;
					}
					py::gil_scoped_acquire acquire;
					on_execution_started.get()();
				};
				std::exception_ptr extension_catalog_commit_error;
				if (!client_context) {
					plan_res = runner->run_plan(plan.plan_, {}, publish_execution_started);
				} else {
					class PlanResultRollbackException final : public std::exception {};
					try {
						client_context->RunFunctionInTransaction([&]() {
							plan_res = runner->run_plan(plan.plan_, {}, publish_execution_started);
							if (plan_res.is_err() ||
							    (plan_res.value().tag == duckdb::distributed::PlanRunner::PlanResult::EXTENSION_WRITE &&
							     plan_res.value().extension_write_result.outcome_unknown)) {
								throw PlanResultRollbackException();
							}
						});
					} catch (const PlanResultRollbackException &) {
					} catch (...) {
						if (plan_res.is_ok() &&
						    plan_res.value().tag == duckdb::distributed::PlanRunner::PlanResult::EXTENSION_WRITE) {
							// Provider finalization returned, but the coordinator transaction
							// did not report a definitive outcome. The extension catalog may
							// already reference the selected artifacts.
							extension_catalog_commit_error = std::current_exception();
						} else {
							throw;
						}
					}
				}
				if (!plan_res.is_ok()) {
					res = DuckDBResult<DistributedCopyResult>::err(plan_res.error());
				} else {
					auto &plan_result = plan_res.value();
					if (plan_result.tag == duckdb::distributed::PlanRunner::PlanResult::COPY) {
						res = DuckDBResult<DistributedCopyResult>::ok(std::move(plan_result.copy_result));
					} else if (plan_result.tag == duckdb::distributed::PlanRunner::PlanResult::EXTENSION_WRITE) {
						extension_write = true;
						extension_result = std::move(plan_result.extension_write_result);
						if (extension_catalog_commit_error) {
							extension_result.catalog_committed = false;
							extension_result.outcome_unknown = true;
							extension_result.outcome_error =
							    "extension catalog commit outcome is unknown; selected artifacts were retained: " +
							    exception_message(extension_catalog_commit_error);
							if (extension_result.info.mode == DistributedWriteMode::FILE_ARTIFACT) {
								extension_result.file_result.output_outcome_unknown = true;
								extension_result.file_result.output_outcome_error = extension_result.outcome_error;
							}
						} else if (!extension_result.outcome_unknown && !extension_result.catalog_committed) {
							extension_result.catalog_committed = true;
						}
						if (extension_result.catalog_committed) {
							extension_terminal_state = ExtensionWriteTerminalState::COMMITTED;
						} else if (extension_result.outcome_unknown) {
							extension_terminal_state = ExtensionWriteTerminalState::OUTCOME_UNKNOWN;
						}
						extension_terminal_detail = extension_result.outcome_error;
						if (extension_result.info.mode == DistributedWriteMode::FILE_ARTIFACT) {
							extension_terminal_base_path = extension_result.file_result.output_base_path;
							extension_terminal_run_id = extension_result.file_result.output_run_id;
							extension_terminal_manifest_path = extension_result.file_result.output_manifest_path;
							extension_terminal_committed_marker_path =
							    extension_result.file_result.output_committed_marker_path;
						}
						if (extension_result.info.mode == DistributedWriteMode::FILE_ARTIFACT) {
							res = DuckDBResult<DistributedCopyResult>::ok(extension_result.file_result);
						} else {
							DistributedCopyResult callback_result;
							callback_result.rows_copied = extension_result.rows_written;
							callback_result.output_committed = extension_result.catalog_committed;
							callback_result.output_outcome_unknown = extension_result.outcome_unknown;
							callback_result.output_outcome_error = extension_result.outcome_error;
							res = DuckDBResult<DistributedCopyResult>::ok(std::move(callback_result));
						}
					} else {
						res = DuckDBResult<DistributedCopyResult>::err(
						    DuckDBError("run_plan returned stream instead of a write result for a write plan"));
					}
				}
				run_plan_ms = elapsed_ms(run_plan_started);
			}
			if (extension_terminal_state != ExtensionWriteTerminalState::NONE &&
			    fail_next_extension_result_marshalling_for_test_.exchange(false, std::memory_order_relaxed)) {
				throw std::runtime_error("planned extension result marshalling failure");
			}

			if (!res.is_ok()) {
				rethrow_submission_error(plan.idx(), res.error().what());
				throw py::value_error(res.error().what());
			}

			auto result = std::move(res).value();
			if (extension_write && !extension_result.catalog_committed && !extension_result.outcome_unknown) {
				rethrow_submission_error(plan.idx());
				throw py::value_error("distributed extension write completed without a confirmed catalog commit");
			}
			if ((!extension_write || extension_result.info.mode == DistributedWriteMode::FILE_ARTIFACT) &&
			    !result.output_committed && !result.output_outcome_unknown) {
				rethrow_submission_error(plan.idx());
				throw py::value_error("distributed COPY completed without a committed output marker");
			}
			try {
				rethrow_submission_error(plan.idx());
			} catch (...) {
				post_commit_warnings.push_back("native COPY post-commit submission state failed: " +
				                               exception_message(std::current_exception()));
			}
			py::list files;
			for (auto &info : result.files) {
				py::dict entry;
				entry["staging_path"] = info.staging_path;
				entry["worker_output_path"] = info.staging_path;
				entry["final_path"] = info.final_path;
				entry["row_count"] = info.row_count;
				entry["file_size_bytes"] = info.file_size_bytes;
				entry["footer_size_bytes"] = info.footer_size_bytes.IsNull()
				                                 ? py::object(py::none())
				                                 : py::object(py::str(info.footer_size_bytes.ToString()));
				entry["column_statistics"] = info.column_statistics.IsNull()
				                                 ? py::object(py::none())
				                                 : py::object(py::str(info.column_statistics.ToString()));
				entry["partition_keys"] = info.partition_keys.IsNull()
				                              ? py::object(py::none())
				                              : py::object(py::str(info.partition_keys.ToString()));
				files.append(entry);
			}
			py::dict out;
			out["rows_copied"] = result.rows_copied;
			out["files"] = files;
			AppendDistributedCopyResultMetadata(out, result);
			out["extension_write"] = py::bool_(extension_write);
			out["extension_write_name"] = py::str(extension_write ? extension_result.info.Name() : string());
			out["extension_catalog_committed"] = py::bool_(extension_write && extension_result.catalog_committed);
			out["extension_write_mode"] =
			    py::str(!extension_write                                                    ? string()
			            : extension_result.info.mode == DistributedWriteMode::FILE_ARTIFACT ? "file_artifact"
			                                                                                : "callback");
			idx_t extension_fragment_count = 0;
			idx_t extension_artifact_count = 0;
			if (extension_write) {
				for (const auto &task_result : extension_result.selected_task_results) {
					extension_fragment_count += task_result.fragments.size();
					for (const auto &fragment : task_result.fragments) {
						extension_artifact_count += fragment.artifacts.size();
					}
				}
			}
			out["extension_task_result_count"] =
			    py::int_(extension_write ? extension_result.selected_task_results.size() : 0);
			out["extension_fragment_count"] = py::int_(extension_fragment_count);
			out["extension_artifact_count"] = py::int_(extension_artifact_count);
			out["extension_bytes_written"] = py::int_(extension_write ? extension_result.bytes_written : 0);
			body_succeeded = true;
			cleanup();
			out["copy_total_ms"] = py::int_(elapsed_ms(copy_started));
			out["copy_run_plan_ms"] = py::int_(run_plan_ms);
			out["copy_runner_cleanup_ms"] = py::int_(cleanup_ms);
			out["copy_runner_cleanup_pending"] = py::bool_(cleanup_error != nullptr);
			py::list cleanup_warnings;
			for (const auto &warning : post_commit_warnings) {
				cleanup_warnings.append(warning);
			}
			if (cleanup_error) {
				cleanup_warnings.append("native COPY runner cleanup failed: " + exception_message(cleanup_error));
			}
			out["copy_runner_cleanup_warnings"] = cleanup_warnings;
			out["copy_selected_file_count"] = py::int_(result.files.size());
			out["copy_duplicate_file_count"] = py::int_(0);
			return out;
		} catch (...) {
			auto execution_error = std::current_exception();
			if (extension_terminal_state != ExtensionWriteTerminalState::NONE) {
				cleanup();
				raise_extension_terminal_error(execution_error);
			}
			if (body_succeeded) {
				std::rethrow_exception(execution_error);
			}
			cleanup();
			if (cleanup_error) {
				throw std::runtime_error("distributed COPY execution and teardown both failed: execution=" +
				                         exception_message(execution_error) +
				                         "; teardown=" + exception_message(cleanup_error));
			}
			std::rethrow_exception(execution_error);
		}
	}

	py::object finalize_copy_impl(py::list file_infos_py, py::str copy_spec_key, py::str,
	                              duckdb::shared_ptr<duckdb::ClientContext> client_context = nullptr) {
		using namespace duckdb::distributed;

		string key = copy_spec_key.cast<string>();
		auto it = streaming_results_.find(key);
		if (it == streaming_results_.end()) {
			throw py::value_error("No stored copy context for key: " + key);
		}
		auto ctx = it->second;
		streaming_results_.erase(it);

		// Convert Python file info dicts to C++ DistributedCopyFileInfo
		vector<DistributedCopyFileInfo> files;
		for (auto item : file_infos_py) {
			auto d = item.cast<py::dict>();
			DistributedCopyFileInfo info;
			if (d.contains("worker_output_path") && !d["worker_output_path"].is_none()) {
				info.staging_path = d["worker_output_path"].cast<string>();
			} else if (d.contains("staging_path") && !d["staging_path"].is_none()) {
				info.staging_path = d["staging_path"].cast<string>();
			} else {
				throw py::value_error("copy file info is missing worker_output_path/staging_path");
			}
			if (d.contains("final_path") && !d["final_path"].is_none()) {
				info.final_path = d["final_path"].cast<string>();
			}
			info.row_count = d["row_count"].cast<idx_t>();
			info.file_size_bytes = d["file_size_bytes"].cast<idx_t>();
			files.push_back(std::move(info));
		}

		auto runner = std::make_shared<PlanRunner>(worker_manager_, client_context);

		DuckDBResult<DistributedCopyResult> res;
		idx_t finalize_ms = 0;
		{
			auto finalize_started = std::chrono::steady_clock::now();
			py::gil_scoped_release release;
			if (!client_context) {
				res = runner->finalize_copy(ctx->copy_spec, ctx->staging_root, std::move(files));
			} else {
				client_context->RunFunctionInTransaction(
				    [&]() { res = runner->finalize_copy(ctx->copy_spec, ctx->staging_root, std::move(files)); });
			}
			auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
			                                                                     finalize_started);
			if (elapsed.count() > 0) {
				auto max_value = static_cast<unsigned long long>(std::numeric_limits<idx_t>::max());
				auto value = static_cast<unsigned long long>(elapsed.count());
				finalize_ms = static_cast<idx_t>(value > max_value ? max_value : value);
			}
		}

		if (!res.is_ok()) {
			throw py::value_error(res.error().what());
		}

		auto result = std::move(res).value();
		py::dict out;
		out["rows_copied"] = result.rows_copied;
		py::list out_files;
		for (auto &f : result.files) {
			py::dict entry;
			entry["staging_path"] = f.staging_path;
			entry["worker_output_path"] = f.staging_path;
			entry["final_path"] = f.final_path;
			entry["row_count"] = f.row_count;
			entry["file_size_bytes"] = f.file_size_bytes;
			out_files.append(entry);
		}
		out["files"] = out_files;
		AppendDistributedCopyResultMetadata(out, result);
		out["copy_total_ms"] = py::int_(finalize_ms);
		out["copy_finalize_call_ms"] = py::int_(finalize_ms);
		out["copy_selected_file_count"] = py::int_(result.files.size());
		out["copy_duplicate_file_count"] = py::int_(0);
		return out;
	}

	// Core implementation of execute_native that works with a raw PhysicalPlan pointer
	py::object execute_native_impl(
	    py::object conn_obj, std::shared_ptr<duckdb::PhysicalPlan> physical_plan, const string &plan_id,
	    const string &resource_query_id,
	    const std::unordered_map<idx_t, duckdb::distributed::ScanTaskDescriptor> *scan_task_map,
	    const std::unordered_map<idx_t, duckdb::distributed::ExchangeSourceTaskDescriptor> *exchange_source_task_map =
	        nullptr,
	    const duckdb::distributed::ExchangeSinkInstanceTaskDescriptor *exchange_sink_instance_task = nullptr,
	    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>>
	        *fte_scan_source_queue_map = nullptr,
	    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>>
	        *fte_exchange_source_queue_map = nullptr,
	    const CopyOutputInfo *copy_output_info = nullptr, py::object dynamic_filter_domains_obj = py::none(),
	    py::object native_progress_callback = py::none(), py::object runtime_context_obj = py::none()) {
		using namespace duckdb;

		// Extract DuckDB connection from Python object
		auto &conn_wrapper = [&]() -> DuckDBPyConnection & {
			if (py::hasattr(conn_obj, "c")) {
				return conn_obj.attr("c").cast<DuckDBPyConnection &>();
			}
			if (py::isinstance<DuckDBPyConnection>(conn_obj)) {
				return conn_obj.cast<DuckDBPyConnection &>();
			}
			throw py::value_error("Connection object must have 'c' attribute or be a DuckDBPyConnection");
		}();

		// Get ClientContext from connection
		auto &context = *conn_wrapper.con.GetConnection().context;

		// Validate that we have a physical plan with a root
		if (!physical_plan || !physical_plan->HasRoot()) {
			throw py::value_error("Physical plan is missing or has no root operator");
		}
		string distributed_write_task_attempt_id;
		if (!runtime_context_obj.is_none() && !py::isinstance<py::dict>(runtime_context_obj)) {
			throw py::value_error("runtime_context must be a dict");
		}
		if (!runtime_context_obj.is_none()) {
			auto runtime_context = runtime_context_obj.cast<py::dict>();
			if (runtime_context.contains("task_id") && !runtime_context["task_id"].is_none()) {
				if (!py::isinstance<py::str>(runtime_context["task_id"])) {
					throw py::value_error("runtime_context task_id must be a string");
				}
				distributed_write_task_attempt_id = runtime_context["task_id"].cast<string>();
				if (distributed_write_task_attempt_id.empty()) {
					throw py::value_error("runtime_context task_id must not be empty");
				}
			}
		}
		duckdb::DistributedWriteTaskContext distributed_write_task_context;
		distributed_write_task_context.query_id = plan_id;
		distributed_write_task_context.task_attempt_id = distributed_write_task_attempt_id;
		duckdb::ValidateDistributedWriteTaskContextAssignment(*physical_plan, distributed_write_task_context);
		const bool worker_task_execution = !distributed_write_task_attempt_id.empty() || scan_task_map != nullptr ||
		                                   exchange_source_task_map != nullptr ||
		                                   exchange_sink_instance_task != nullptr ||
		                                   fte_scan_source_queue_map != nullptr ||
		                                   fte_exchange_source_queue_map != nullptr || copy_output_info != nullptr;
		if (scan_task_map && fte_scan_source_queue_map) {
			for (const auto &entry : *scan_task_map) {
				if (fte_scan_source_queue_map->find(entry.first) != fte_scan_source_queue_map->end()) {
					throw py::value_error("scan node_id=" + std::to_string(entry.first) +
					                      " has both a static task descriptor and an FTE split queue");
				}
			}
		}
		if (exchange_source_task_map && fte_exchange_source_queue_map) {
			for (const auto &entry : *exchange_source_task_map) {
				if (fte_exchange_source_queue_map->find(entry.first) != fte_exchange_source_queue_map->end()) {
					throw py::value_error("exchange source node_id=" + std::to_string(entry.first) +
					                      " has both a static task descriptor and an FTE split queue");
				}
			}
		}
		set<idx_t> exchange_source_assignment_node_ids;
		if (exchange_source_task_map) {
			for (const auto &entry : *exchange_source_task_map) {
				exchange_source_assignment_node_ids.insert(entry.first);
			}
		}
		if (fte_exchange_source_queue_map) {
			for (const auto &entry : *fte_exchange_source_queue_map) {
				exchange_source_assignment_node_ids.insert(entry.first);
			}
		}
		set<idx_t> scan_assignment_node_ids;
		if (scan_task_map) {
			for (const auto &entry : *scan_task_map) {
				scan_assignment_node_ids.insert(entry.first);
			}
		}
		if (fte_scan_source_queue_map) {
			for (const auto &entry : *fte_scan_source_queue_map) {
				scan_assignment_node_ids.insert(entry.first);
			}
		}
		if (worker_task_execution || duckdb::distributed::HasDistributedScanTaskTargets(*physical_plan)) {
			string error;
			if (!duckdb::distributed::ValidateScanTaskAssignments(*physical_plan, scan_assignment_node_ids, &error)) {
				throw py::value_error("Scan task assignment validation failed: " + error);
			}
		}
		{
			string error;
			if (!duckdb::distributed::ValidateExchangeSourceAssignments(*physical_plan,
			                                                            exchange_source_assignment_node_ids, &error)) {
				throw py::value_error("Exchange source assignment validation failed: " + error);
			}
		}
		AssignDataSourceQueryOwner(physical_plan->Root(), resource_query_id);

		if (scan_task_map && !scan_task_map->empty()) {
			string error;
			if (!duckdb::distributed::ApplyScanTasksToPlan(*physical_plan, *scan_task_map, &error)) {
				throw py::value_error("Failed to apply scan tasks to plan: " + error);
			}
		}

		if (fte_scan_source_queue_map && !fte_scan_source_queue_map->empty()) {
			string error;
			bool applied;
			{
				// Extension callbacks materialize opaque task envelopes here and may
				// wait until no_more_splits. The control thread that seals the queue
				// must be able to acquire the GIL while that wait is in progress.
				py::gil_scoped_release release;
				applied = duckdb::distributed::ApplyFteScanSourceQueuesToPlan(*physical_plan,
				                                                              *fte_scan_source_queue_map, &error);
			}
			if (!applied) {
				throw py::value_error("Failed to apply FTE scan source queues to plan: " + error);
			}
		}
		if (worker_task_execution || duckdb::distributed::HasDistributedScanTaskTargets(*physical_plan)) {
			string error;
			if (!duckdb::distributed::ValidateDistributedScanTasksApplied(*physical_plan, &error)) {
				throw py::value_error("Distributed scan task validation failed: " + error);
			}
		}

		if (exchange_source_task_map && !exchange_source_task_map->empty()) {
			string error;
			if (!duckdb::distributed::ApplyExchangeSourceTasksToPlan(*physical_plan, *exchange_source_task_map,
			                                                         &error)) {
				throw py::value_error("Failed to apply exchange source tasks to plan: " + error);
			}
		}

		if (fte_exchange_source_queue_map && !fte_exchange_source_queue_map->empty()) {
			string error;
			if (!duckdb::distributed::ApplyFteExchangeSourceQueuesToPlan(*physical_plan, *fte_exchange_source_queue_map,
			                                                             &error)) {
				throw py::value_error("Failed to apply FTE exchange source queues to plan: " + error);
			}
		}

		if (exchange_sink_instance_task) {
			string error;
			if (!duckdb::distributed::ApplyExchangeSinkInstanceToPlan(*physical_plan, *exchange_sink_instance_task,
			                                                          &error)) {
				throw py::value_error("Failed to apply exchange sink instance to plan: " + error);
			}
		}

		ApplyDynamicFilterDomainsToPlan(*physical_plan, dynamic_filter_domains_obj);

		ApplyTaskLocalCopyOutput(*physical_plan, copy_output_info, &context);

		duckdb::ApplyDistributedWriteTaskContext(*physical_plan, distributed_write_task_context);

		auto &root_op = physical_plan->Root();
		auto task_flight_manager = FindFlightExchangeSinkManager(root_op);
		auto task_flight_port = [&]() {
			return task_flight_manager ? task_flight_manager->GetPublishedFlightServerPort() : 0;
		};
		auto task_flight_host = [&]() {
			return task_flight_manager ? task_flight_manager->GetPublishedFlightServerHost() : string();
		};
		auto task_flight_server_epoch = [&]() {
			return task_flight_manager ? task_flight_manager->GetPublishedFlightServerEpoch() : string();
		};
		auto task_exchange_sink_instance_obj = [&]() -> py::object {
			if (!exchange_sink_instance_task) {
				return py::none();
			}
			auto completed_task = *exchange_sink_instance_task;
			if (auto *sink = FindRemoteExchangeSinkOperator(root_op)) {
				completed_task.sink_instance = sink->SinkHandle();
			}
			completed_task.sink_instance.flight_host = task_flight_host();
			completed_task.sink_instance.flight_server_epoch = task_flight_server_epoch();
			return py::bytes(completed_task.SerializeToBytes());
		};

		py::list results;
		auto build_result_schema = [&](const duckdb::vector<string> &names,
		                               const duckdb::vector<duckdb::LogicalType> &result_types) -> py::dict {
			return BuildNativeResultSchema(names, result_types);
		};
		auto build_empty_result = [&](const duckdb::vector<string> &names,
		                              const duckdb::vector<duckdb::LogicalType> &result_types, const string &status,
		                              py::object task_stats) -> py::object {
			py::list payloads;
			py::list metadatas;
			return BuildNativeTaskResult(payloads, metadatas, build_result_schema(names, result_types), py::list(),
			                             std::move(task_stats), status, task_flight_port(),
			                             task_exchange_sink_instance_obj());
		};
		auto build_table_result = [&](const py::object &table, idx_t row_count, const duckdb::vector<string> &names,
		                              const duckdb::vector<duckdb::LogicalType> &result_types,
		                              py::object task_stats) -> py::object {
			if (row_count == 0) {
				return build_empty_result(names, result_types, "empty", std::move(task_stats));
			}
			py::list payloads;
			payloads.append(table);
			py::list metadatas;
			metadatas.append(
			    py::cast(NativePartitionMetadata(static_cast<size_t>(row_count), GetPyPayloadSizeBytes(table))));
			return BuildNativeTaskResult(payloads, metadatas, build_result_schema(names, result_types), py::list(),
			                             std::move(task_stats), "ok", task_flight_port(),
			                             task_exchange_sink_instance_obj());
		};
		auto build_executed_result = [&](py::object task_stats) -> py::object {
			duckdb::vector<string> names;
			names.reserve(root_op.types.size());
			for (idx_t i = 0; i < root_op.types.size(); i++) {
				names.push_back("col_" + std::to_string(i));
			}
			return build_empty_result(names, root_op.types, "executed", std::move(task_stats));
		};

		// Materialize operators that return rows to the caller. A terminal remote
		// exchange sink is different: its source side is a zero-row completion
		// interface and its data output is already carried by Arrow Flight.
		// Terminal exchange sinks still need a managed DuckDB query. Operators
		// below the sink (notably hash joins) can schedule follow-up work through
		// ClientContext::GetExecutor(), which requires the active-query lifecycle
		// installed by PendingQueryPreparedStatementNoRebind. The collector only
		// consumes the sink's zero-row completion source; data remains on Flight.
		const bool terminal_exchange_sink = root_op.type == PhysicalOperatorType::EXCHANGE_SINK;
		const bool needs_result_collector = terminal_exchange_sink || NativePlanNeedsResultCollector(root_op);
		if (needs_result_collector) {

			auto prepared_data = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
			prepared_data->types = root_op.types;
			for (idx_t i = 0; i < root_op.types.size(); i++) {
				prepared_data->names.push_back("col_" + std::to_string(i));
			}
			prepared_data->properties.return_type = StatementReturnType::QUERY_RESULT;
			prepared_data->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
			prepared_data->memory_type = QueryResultMemoryType::IN_MEMORY;

			// Create unique_ptr from raw pointer for PreparedStatementData
			// IMPORTANT: We must release() the unique_ptr before prepared_data is destroyed
			// to avoid double-free, since shared_ptr manages the actual lifetime
			// Use RAII to ensure release() is always called, even on exceptions
			PhysicalPlan *raw_plan = physical_plan.get();
			prepared_data->physical_plan = unique_ptr<PhysicalPlan>(raw_plan);
			// Keep shared_ptr in scope - it will manage the pointer's lifetime
			// RAII guard to ensure release() is called on all exit paths
			struct PhysicalPlanGuard {
				unique_ptr<PhysicalPlan> &plan;
				bool released = false;
				PhysicalPlanGuard(unique_ptr<PhysicalPlan> &p) : plan(p) {
				}
				~PhysicalPlanGuard() {
					if (!released) {
						plan.release();
					}
				}
				void release() {
					if (!released) {
						plan.release();
						released = true;
					}
				}
			} plan_guard(prepared_data->physical_plan);

			PendingQueryParameters parameters;
			// execute_native always returns a materialized result. Keep that collector contract query-local,
			// while only allowing parallel collection when the plan does not require order preservation.
			parameters.get_result_collector =
			    [](duckdb::ClientContext &context,
			       duckdb::PreparedStatementData &data) -> duckdb::unique_ptr<duckdb::PhysicalOperator> {
				auto &physical_plan = *data.physical_plan;
				const bool preserve_order =
				    duckdb::PhysicalPlanGenerator::PreserveInsertionOrder(context, physical_plan.Root());
				return duckdb::make_uniq<duckdb::PhysicalMaterializedCollector>(physical_plan, data, !preserve_order);
			};
			std::unique_ptr<QueryResult> query_result;
			vector<PipelineProgressSnapshot> stable_pipeline_snapshots;
			try {
				const bool has_native_progress_callback = !native_progress_callback.is_none();
				const auto native_progress_env_ms = DuckdbGetEnvIntMs("VANE_NATIVE_PROGRESS_INTERVAL_MS");
				const auto native_progress_ms =
				    has_native_progress_callback ? (native_progress_env_ms > 0 ? native_progress_env_ms : 500) : 0;

				py::gil_scoped_release release;
				auto pending = context.PendingQueryPreparedStatementNoRebind(string("external_plan:") + plan_id,
				                                                             prepared_data, parameters);
				if (!pending || pending->HasError()) {
					throw InvalidInputException(pending ? pending->GetError() : "Failed to start pending query");
				}

				auto &executor = context.GetExecutor();
				stable_pipeline_snapshots = executor.GetPipelinesProgressSnapshots();
				if (stable_pipeline_snapshots.empty()) {
					throw InternalException("native executor produced no pipeline topology after initialization");
				}
				auto last_native_progress = std::chrono::steady_clock::now();
				bool native_progress_callback_failed = false;
				auto emit_native_progress = [&](bool force) {
					auto snapshots = executor.GetPipelinesProgressSnapshots();
					if (!snapshots.empty()) {
						stable_pipeline_snapshots = snapshots;
					}
					if (!has_native_progress_callback || native_progress_callback_failed) {
						return;
					}
					auto now = std::chrono::steady_clock::now();
					auto elapsed_ms =
					    std::chrono::duration_cast<std::chrono::milliseconds>(now - last_native_progress).count();
					if (!force && elapsed_ms < native_progress_ms) {
						return;
					}
					py::gil_scoped_acquire acquire;
					try {
						auto stats = BuildNativeTaskStatsDict(stable_pipeline_snapshots, scan_task_map,
						                                      exchange_source_task_map, fte_scan_source_queue_map,
						                                      fte_exchange_source_queue_map, terminal_exchange_sink);
						native_progress_callback(stats);
						last_native_progress = now;
					} catch (const std::exception &ex) {
						native_progress_callback_failed = true;
					} catch (...) {
						native_progress_callback_failed = true;
					}
				};
				if (has_native_progress_callback) {
					py::gil_scoped_acquire acquire;
					try {
						auto stats = BuildNativeTaskStatsDict(stable_pipeline_snapshots, scan_task_map,
						                                      exchange_source_task_map, fte_scan_source_queue_map,
						                                      fte_exchange_source_queue_map, terminal_exchange_sink);
						native_progress_callback(stats);
					} catch (const std::exception &ex) {
						native_progress_callback_failed = true;
					} catch (...) {
						native_progress_callback_failed = true;
					}
				}
				std::atomic<bool> native_progress_stop {false};
				std::mutex native_progress_mutex;
				std::condition_variable native_progress_cv;
				std::thread native_progress_thread;
				if (has_native_progress_callback) {
					native_progress_thread = std::thread([&]() {
						const auto interval = std::chrono::milliseconds(native_progress_ms);
						while (!native_progress_stop.load()) {
							std::unique_lock<std::mutex> lock(native_progress_mutex);
							if (native_progress_cv.wait_for(lock, interval,
							                                [&]() { return native_progress_stop.load(); })) {
								break;
							}
							lock.unlock();
							if (PythonIsFinalizing()) {
								break;
							}
							emit_native_progress(false);
						}
					});
				}
				struct NativeProgressThreadGuard {
					std::atomic<bool> &done;
					std::condition_variable &cv;
					std::thread &thread;
					void Stop() {
						done.store(true);
						cv.notify_all();
						if (thread.joinable()) {
							thread.join();
						}
					}
					~NativeProgressThreadGuard() {
						Stop();
					}
				} native_progress_guard {native_progress_stop, native_progress_cv, native_progress_thread};

				PendingExecutionResult execution_result = PendingExecutionResult::RESULT_NOT_READY;
				while (!PendingQueryResult::IsResultReady(execution_result)) {
					execution_result = pending->ExecuteTask();
					if (execution_result == PendingExecutionResult::BLOCKED ||
					    execution_result == PendingExecutionResult::NO_TASKS_AVAILABLE) {
						pending->WaitForTask();
					}
				}
				native_progress_guard.Stop();
				emit_native_progress(true);
				query_result = pending->Execute();
			} catch (const std::exception &ex) {
				throw py::value_error(string("Execution failed: ") + ex.what());
			}

			if (!query_result) {
				throw py::value_error("Execution failed: no query result");
			}
			if (query_result->HasError()) {
				// Build a detailed diagnostic string that includes expression details
				// so we can see the exact column mapping state even through Ray's output buffering
				std::ostringstream diag;
				diag << "Execution failed: " << query_result->GetError() << "\n";
				diag << "--- DIAGNOSTIC DUMP ---\n";
				std::function<void(const PhysicalOperator &, int)> dump_diag = [&](const PhysicalOperator &op,
				                                                                   int depth) -> void {
					string indent(depth * 2, ' ');
					diag << indent << "OP: " << duckdb::EnumUtil::ToString(op.type) << " types=[";
					for (idx_t i = 0; i < op.types.size(); i++) {
						if (i > 0)
							diag << ", ";
						diag << op.types[i].ToString();
					}
					diag << "]\n";
					if (op.type == duckdb::PhysicalOperatorType::PROJECTION) {
						auto &proj = op.Cast<duckdb::PhysicalProjection>();
						for (idx_t i = 0; i < proj.select_list.size(); i++) {
							if (!proj.select_list[i])
								continue;
							auto &expr = *proj.select_list[i];
							diag << indent << "  proj[" << i
							     << "]: class=" << duckdb::EnumUtil::ToString(expr.GetExpressionClass())
							     << " type=" << expr.return_type.ToString() << " alias='" << expr.GetAlias() << "'"
							     << " name='" << expr.GetName() << "'";
							if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
								auto &ref = expr.Cast<duckdb::BoundReferenceExpression>();
								diag << " INDEX=" << ref.index;
							}
							diag << "\n";
							// Dump children of non-trivial expressions
							duckdb::ExpressionIterator::EnumerateChildren(
							    const_cast<duckdb::Expression &>(expr), [&](duckdb::Expression &child) {
								    diag << indent << "    child: class="
								         << duckdb::EnumUtil::ToString(child.GetExpressionClass())
								         << " type=" << child.return_type.ToString() << " alias='" << child.GetAlias()
								         << "'";
								    if (child.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
									    auto &cref = child.Cast<duckdb::BoundReferenceExpression>();
									    diag << " INDEX=" << cref.index;
								    }
								    diag << "\n";
							    });
						}
					} else if (op.type == duckdb::PhysicalOperatorType::TABLE_SCAN) {
						auto &scan = op.Cast<duckdb::PhysicalTableScan>();
						diag << indent << "  column_ids=[";
						for (idx_t i = 0; i < scan.column_ids.size(); i++) {
							if (i > 0)
								diag << ", ";
							diag << scan.column_ids[i].GetPrimaryIndex();
						}
						diag << "] projection_ids=[";
						for (idx_t i = 0; i < scan.projection_ids.size(); i++) {
							if (i > 0)
								diag << ", ";
							diag << scan.projection_ids[i];
						}
						diag << "] names=[";
						for (idx_t i = 0; i < scan.names.size(); i++) {
							if (i > 0)
								diag << ", ";
							diag << scan.names[i];
						}
						diag << "] returned_types=[";
						for (idx_t i = 0; i < scan.returned_types.size(); i++) {
							if (i > 0)
								diag << ", ";
							diag << scan.returned_types[i].ToString();
						}
						diag << "]\n";
						// Dump projected output mapping
						diag << indent << "  projected_output: ";
						if (!scan.projection_ids.empty()) {
							for (idx_t i = 0; i < scan.projection_ids.size(); i++) {
								if (i > 0)
									diag << ", ";
								auto pid = scan.projection_ids[i];
								auto cid = scan.column_ids[pid].GetPrimaryIndex();
								diag << "out[" << i << "]=col_ids[" << pid << "]=" << cid;
								if (cid < scan.names.size())
									diag << "(" << scan.names[cid] << ")";
							}
						} else {
							for (idx_t i = 0; i < scan.column_ids.size(); i++) {
								if (i > 0)
									diag << ", ";
								auto cid = scan.column_ids[i].GetPrimaryIndex();
								diag << "out[" << i << "]=" << cid;
								if (cid < scan.names.size())
									diag << "(" << scan.names[cid] << ")";
							}
						}
						diag << "\n";
					} else if (op.type == duckdb::PhysicalOperatorType::HASH_GROUP_BY) {
						auto &agg = op.Cast<duckdb::PhysicalHashAggregate>();
						for (idx_t i = 0; i < agg.grouped_aggregate_data.groups.size(); i++) {
							if (!agg.grouped_aggregate_data.groups[i])
								continue;
							auto &expr = *agg.grouped_aggregate_data.groups[i];
							diag << indent << "  group[" << i
							     << "]: class=" << duckdb::EnumUtil::ToString(expr.GetExpressionClass());
							if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
								auto &ref = expr.Cast<duckdb::BoundReferenceExpression>();
								diag << " INDEX=" << ref.index;
							}
							diag << "\n";
						}
						for (idx_t i = 0; i < agg.grouped_aggregate_data.aggregates.size(); i++) {
							if (!agg.grouped_aggregate_data.aggregates[i])
								continue;
							auto &expr = *agg.grouped_aggregate_data.aggregates[i];
							diag << indent << "  agg[" << i << "]: " << expr.GetName() << "\n";
						}
					}
					for (auto &child : op.children) {
						dump_diag(child.get(), depth + 1);
					}
				};
				dump_diag(prepared_data->physical_plan->Root(), 0);
				diag << "--- END DIAGNOSTIC ---";
				throw py::value_error(diag.str());
			}
			if (query_result->type != QueryResultType::MATERIALIZED_RESULT) {
				throw py::value_error("Execution failed: unexpected query result type");
			}

			auto &materialized = query_result->Cast<MaterializedQueryResult>();
			auto &collection = materialized.Collection();
			if (terminal_exchange_sink) {
				auto task_stats =
				    BuildNativeTaskStatsDict(stable_pipeline_snapshots, scan_task_map, exchange_source_task_map,
				                             fte_scan_source_queue_map, fte_exchange_source_queue_map, true);
				MarkTaskStatsCompleted(task_stats);
				plan_guard.release();
				if (collection.Count() != 0) {
					throw py::value_error("Execution failed: terminal exchange sink produced local result rows");
				}
				return build_executed_result(std::move(task_stats));
			}
			if (root_op.type == PhysicalOperatorType::DISTRIBUTED_EXTENSION_WRITE &&
			    (collection.Types().size() != 1 || collection.Types()[0].id() != LogicalTypeId::BLOB ||
			     collection.Count() != 1)) {
				throw py::value_error(
				    "Execution failed: distributed extension write must produce exactly one BLOB task envelope");
			}

			auto table = CollectionToArrowTable(collection, context);
			const auto payload_bytes = static_cast<idx_t>(GetPyPayloadSizeBytes(table));
			auto materialized_stats = BuildMaterializedInputTaskStats(
			    root_op, scan_task_map, exchange_source_task_map, fte_scan_source_queue_map,
			    fte_exchange_source_queue_map, collection.Count(), payload_bytes, &materialized);
			auto task_stats =
			    BuildNativeTaskStatsDict(stable_pipeline_snapshots, scan_task_map, exchange_source_task_map,
			                             fte_scan_source_queue_map, fte_exchange_source_queue_map);
			OverlayMaterializedProgressStats(task_stats, materialized_stats);
			MarkTaskStatsCompleted(task_stats);
			plan_guard.release();
			return build_table_result(table, collection.Count(), query_result->names, query_result->types,
			                          std::move(task_stats));
		}

		DuckDBPyTransactionGuard tx_guard(context);

		// Standard execution path with Executor (for plans with sinks)
		Executor executor(context);

		executor.Initialize(root_op);
		// Executor::ExecuteTask clears its pipeline list when it transitions to
		// EXECUTION_FINISHED. Keep the last non-empty snapshot so short-lived
		// sink-only plans still publish their terminal counters instead of an
		// empty task_stats payload.
		vector<PipelineProgressSnapshot> stable_pipeline_snapshots = executor.GetPipelinesProgressSnapshots();

		auto types = executor.GetTypes();

		// Release GIL during execution
		{
			const bool has_native_progress_callback = !native_progress_callback.is_none();
			const auto native_progress_env_ms = DuckdbGetEnvIntMs("VANE_NATIVE_PROGRESS_INTERVAL_MS");
			const auto native_progress_ms =
			    has_native_progress_callback ? (native_progress_env_ms > 0 ? native_progress_env_ms : 500) : 0;
			auto last_native_progress = std::chrono::steady_clock::now();
			bool native_progress_callback_failed = false;
			auto emit_native_progress = [&](bool force) {
				auto now = std::chrono::steady_clock::now();
				auto elapsed_ms =
				    std::chrono::duration_cast<std::chrono::milliseconds>(now - last_native_progress).count();
				const bool callback_due = has_native_progress_callback && !native_progress_callback_failed &&
				                          (force || elapsed_ms >= native_progress_ms);
				if (!callback_due && !executor.ExecutionIsFinished()) {
					return;
				}
				auto snapshots = executor.GetPipelinesProgressSnapshots();
				if (!snapshots.empty()) {
					stable_pipeline_snapshots = std::move(snapshots);
				}
				if (!callback_due) {
					return;
				}
				py::gil_scoped_acquire acquire;
				try {
					auto stats =
					    BuildNativeTaskStatsDict(stable_pipeline_snapshots, scan_task_map, exchange_source_task_map,
					                             fte_scan_source_queue_map, fte_exchange_source_queue_map);
					native_progress_callback(stats);
					last_native_progress = now;
				} catch (const std::exception &ex) {
					native_progress_callback_failed = true;
				} catch (...) {
					native_progress_callback_failed = true;
				}
			};
			py::gil_scoped_release release;

			duckdb::PendingExecutionResult exec_result = duckdb::PendingExecutionResult::RESULT_NOT_READY;
			while (exec_result != duckdb::PendingExecutionResult::EXECUTION_FINISHED &&
			       exec_result != duckdb::PendingExecutionResult::EXECUTION_ERROR) {
				// A pipeline can finish on a scheduler thread while this caller is
				// waiting. Capture that terminal state before ExecuteTask clears it.
				emit_native_progress(false);
				exec_result = executor.ExecuteTask();
				emit_native_progress(false);
				if (exec_result == duckdb::PendingExecutionResult::BLOCKED ||
				    exec_result == duckdb::PendingExecutionResult::NO_TASKS_AVAILABLE) {
					executor.WaitForTask();
				}
			}
			emit_native_progress(true);
			if (exec_result == duckdb::PendingExecutionResult::EXECUTION_ERROR) {
				executor.ThrowException();
			}
		}

		// Re-acquire GIL to build Python results
		auto final_pipeline_snapshots = executor.GetPipelinesProgressSnapshots();
		if (final_pipeline_snapshots.empty()) {
			final_pipeline_snapshots = std::move(stable_pipeline_snapshots);
		}
		py::object final_task_stats =
		    BuildNativeTaskStatsDict(final_pipeline_snapshots, scan_task_map, exchange_source_task_map,
		                             fte_scan_source_queue_map, fte_exchange_source_queue_map);

		if (executor.HasResultCollector()) {

			auto result = executor.GetResult();

			if (result->HasError()) {
				throw py::value_error("Execution failed: " + result->GetError());
			}

			auto &materialized_result = result->Cast<MaterializedQueryResult>();
			auto &collection = materialized_result.Collection();

			auto table = CollectionToArrowTable(collection, context);
			tx_guard.Commit();
			return build_table_result(table, collection.Count(), result->names, result->types,
			                          std::move(final_task_stats));
		}
		tx_guard.Commit();
		return build_executed_result(std::move(final_task_stats));
	}

	// Execute physical plan using DuckDB's native Executor (DistributedPhysicalPlan version)
	py::object execute_native(py::object conn_obj, const PyPhysicalPlanWrapper &plan) {
		try {
			auto &mutable_plan = const_cast<PyPhysicalPlanWrapper &>(plan);
			py::object exec_conn = plan.resolve_execution_connection(conn_obj);
			mutable_plan.ensure_connection_snapshot(exec_conn);
			mutable_plan.apply_udf_actor_handles();

			// Validate that we have a physical plan
			if (!plan.plan_ || !plan.plan_->physical_plan()) {
				throw py::value_error("DistributedPhysicalPlan is missing underlying physical plan");
			}

			auto result =
			    execute_native_impl(exec_conn, plan.plan_->physical_plan(), plan.idx(), plan.resource_query_id_,
			                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, py::none(), py::none());
			return result;
		} catch (...) {
			throw;
		}
	}
};

static py::dict DescribeNativeProgress(py::object conn_obj, const PyPhysicalPlanWrapper &plan) {
	if (!plan.IsInitialized()) {
		throw py::value_error("DistributedPhysicalPlan is uninitialized");
	}

	py::object exec_conn = ResolveConnectionForSnapshot(conn_obj, plan.connection_snapshot_);
	PyPhysicalPlanWrapper topology_plan;
	topology_plan.query_id_ = plan.idx();
	topology_plan.resource_query_id_ = plan.resource_query_id_;
	topology_plan.udf_registrations_ = plan.udf_registrations_;
	topology_plan.udf_actor_handles_ = plan.udf_actor_handles_;
	topology_plan.connection_snapshot_ = plan.connection_snapshot_;
	topology_plan.scoped_secret_refs_ = plan.scoped_secret_refs_;
	topology_plan.serialized_root_ = plan.serialize_root_for_clone();
	topology_plan.worker_connection_ = exec_conn;
	topology_plan.materialize_deferred_root(exec_conn);
	topology_plan.apply_udf_actor_handles();
	if (!topology_plan.plan_ || !topology_plan.plan_->physical_plan() || !topology_plan.has_root()) {
		throw py::value_error("DistributedPhysicalPlan topology clone has no physical root");
	}

	auto &conn_wrapper = ExtractPyConnectionWrapper(exec_conn);
	auto &context = *conn_wrapper.con.GetConnection().context;
	py::dict topology;
	try {
		context.RunFunctionInTransaction([&]() {
			topology = BuildNativeProgressTopology(context, topology_plan.plan_->physical_plan());
			// Some extension scan states finish asynchronous metadata work from
			// their destructors. Release the topology clone before committing so
			// those destructors retain a valid catalog transaction.
			topology_plan.plan_.reset();
		});
	} catch (...) {
		if (topology_plan.plan_) {
			context.RunFunctionInTransaction([&]() { topology_plan.plan_.reset(); }, false);
		}
		throw;
	}
	return topology;
}
