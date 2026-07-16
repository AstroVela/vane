// Included by ray_module.cpp inside namespace duckdb.

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
	std::shared_ptr<duckdb::distributed::DistributedPhysicalPlan> plan_;
	py::object arrow_schema_;        // Arrow schema for type information
	py::object udf_registrations_;   // Connection-local Python UDF registrations captured from the source relation
	py::object udf_actor_handles_;   // Node-id keyed actor handles for UDF worker execution
	py::object connection_snapshot_; // Connection settings/extensions snapshot captured from the source relation
	string serialized_root_;         // Deferred serialized PhysicalOperator bytes (for pickle round-trips)

	bool has_root() const {
		return plan_ && plan_->physical_plan() && plan_->physical_plan()->HasRoot();
	}

	void ensure_connection_snapshot(py::object conn_obj) const {
		if (connection_snapshot_.is_none()) {
			return;
		}
		ApplyConnectionSnapshot(conn_obj, connection_snapshot_);
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
	}

	// Materialize deferred serialized_root_ into the plan's PhysicalPlan.
	// Requires a DuckDB connection for deserialization context.
	void materialize_deferred_root(py::object conn_obj) {
		if (serialized_root_.empty())
			return;
		if (has_root())
			return; // Already has a root

		ensure_plan_identity();
		ensure_connection_snapshot(conn_obj);

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
		}
		// Keep connection alive to ensure allocator validity
		worker_connection_ = conn_obj;
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
		duckdb::MemoryStream stream;
		duckdb::SerializationOptions options;
		options.serialization_compatibility = duckdb::SerializationCompatibility::Latest();
		options.serialize_default_values = true;
		duckdb::BinarySerializer serializer(stream, options);
		serializer.Begin();
		physical_plan->Serialize(serializer);
		serializer.End();
		auto data_ptr = stream.GetData();
		auto data_size = stream.GetPosition();
		return string(reinterpret_cast<const char *>(data_ptr), data_size);
	}

	PyPhysicalPlanWrapper clone(py::object conn_obj) const {
		PyPhysicalPlanWrapper result;
		result.query_id_ = idx();
		result.udf_registrations_ = udf_registrations_;
		result.udf_actor_handles_ = udf_actor_handles_;
		result.connection_snapshot_ = connection_snapshot_;
		result.serialized_root_ = serialize_root_for_clone();
		result.ensure_plan_identity();
		RememberQueryUDFRegistrations(result.query_id_, result.udf_registrations_);
		RememberQueryUDFActorHandles(result.query_id_, result.udf_actor_handles_);
		if (!conn_obj.is_none()) {
			result.materialize_deferred_root(conn_obj);
		}
		return result;
	}

	PyPhysicalPlanWrapper()
	    : init_magic_(INIT_MAGIC), worker_connection_(py::none()), client_context_(nullptr), query_id_(),
	      plan_(nullptr), arrow_schema_(py::none()), udf_registrations_(py::none()), udf_actor_handles_(py::none()),
	      connection_snapshot_(py::none()), serialized_root_() {
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
	      query_id_(p ? p->query_id() : string()), plan_(std::move(p)), arrow_schema_(py::none()),
	      udf_registrations_(py::none()), udf_actor_handles_(py::none()), connection_snapshot_(py::none()),
	      serialized_root_() {
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

	py::dict collect_execution_stages(py::object conn_obj) {
		if (!IsInitialized() || !plan_ || !has_root()) {
			throw duckdb::InternalException("DistributedPhysicalPlan must have a root before stage collection");
		}
		auto query_id = idx();
		if (query_id.empty()) {
			throw duckdb::InternalException("DistributedPhysicalPlan stage collection requires a query_id");
		}

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

		vector<std::pair<duckdb::distributed::NodeID, const UDFFunctionData *>> pipeline_udfs;
		for (auto &node : pipeline_nodes) {
			const UDFFunctionData *bind_data = nullptr;
			auto implementation = node->implementation();
			if (auto table_inout = std::dynamic_pointer_cast<duckdb::distributed::TableInOutNode>(implementation)) {
				bind_data = TryGetUDFBindData(table_inout->bind_data());
			} else if (auto streaming =
			               std::dynamic_pointer_cast<duckdb::distributed::StreamingUDFPassthroughNode>(implementation)) {
				bind_data = TryGetUDFBindData(streaming->bind_data());
			}
			if (bind_data) {
				pipeline_udfs.emplace_back(node->node_id(), bind_data);
			}
		}
		std::sort(pipeline_nodes.begin(), pipeline_nodes.end(),
		          [](const auto &left, const auto &right) { return left->node_id() < right->node_id(); });

		vector<UDFFunctionData *> physical_udfs;
		auto physical_plan = plan_->physical_plan();
		std::function<void(PhysicalOperator &)> collect_physical_udfs = [&](PhysicalOperator &op) -> void {
			CollectMutableUDFBindData(op, physical_udfs);
			for (auto &child : op.children) {
				collect_physical_udfs(child.get());
			}
		};
		collect_physical_udfs(physical_plan->Root());
		if (physical_udfs.size() != pipeline_udfs.size()) {
			throw duckdb::InternalException(
			    "physical/pipeline UDF stage count mismatch: physical=%llu pipeline=%llu",
			    static_cast<unsigned long long>(physical_udfs.size()),
			    static_cast<unsigned long long>(pipeline_udfs.size()));
		}

		std::unordered_map<duckdb::distributed::NodeID, UDFFunctionData *> udf_by_node;
		for (idx_t i = 0; i < physical_udfs.size(); i++) {
			auto node_id = pipeline_udfs[i].first;
			auto *bind_data = physical_udfs[i];
			auto pipeline_backend = UDFPayloadStringOrDefault(pipeline_udfs[i].second->payload, "execution_backend");
			auto physical_backend = UDFPayloadStringOrDefault(bind_data->payload, "execution_backend");
			if (pipeline_backend != physical_backend) {
				throw duckdb::InternalException(
				    "physical/pipeline UDF backend mismatch at node %llu: physical=%s pipeline=%s",
				    static_cast<unsigned long long>(node_id), physical_backend, pipeline_backend);
			}
			auto stage_id = "stage:" + query_id + ":node:" + std::to_string(node_id) + ":udf";
			auto existing_query_id = UDFPayloadStringOrDefault(bind_data->payload, "query_id");
			if (!existing_query_id.empty() && existing_query_id != query_id) {
				throw duckdb::InvalidInputException(
				    "distributed UDF payload query_id mismatch: payload=%s plan=%s", existing_query_id, query_id);
			}
			auto existing_stage_id = UDFPayloadStringOrDefault(bind_data->payload, "stage_id");
			if (!existing_stage_id.empty() && existing_stage_id != stage_id) {
				throw duckdb::InvalidInputException(
				    "distributed UDF payload stage_id mismatch: payload=%s plan=%s", existing_stage_id, stage_id);
			}
			bind_data->payload = UDFPayloadWithStringField(bind_data->payload, "query_id", query_id);
			bind_data->payload = UDFPayloadWithStringField(bind_data->payload, "stage_id", stage_id);
			udf_by_node.emplace(node_id, bind_data);
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
			metadata[py::str("is_sink")] = py::bool_(node->implementation()->is_sink());
			metadata[py::str("num_partitions")] = py::int_(std::max<size_t>(1, node->num_partitions()));
			auto udf_entry = udf_by_node.find(node->node_id());
			if (udf_entry == udf_by_node.end()) {
				metadata[py::str("udf_payload")] = py::none();
			} else {
				auto &payload = udf_entry->second->payload;
				metadata[py::str("udf_payload")] =
				    PythonObject::FromValue(payload, payload.type(), client_properties);
			}
			nodes.append(std::move(metadata));
		}

		py::dict result;
		result[py::str("query_id")] = py::str(query_id);
		result[py::str("nodes")] = std::move(nodes);
		py::list terminals;
		terminals.append(py::str(std::to_string(pipeline_root->node_id())));
		result[py::str("terminal_node_ids")] = std::move(terminals);
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
			idx_t next_id = max_id + 1;
			std::unordered_map<idx_t, idx_t> base_for_group;
			std::unordered_map<idx_t, idx_t> dup_to_base;
			std::function<void(PhysicalOperator &)> normalize = [&](PhysicalOperator &op) -> void {
				if (op.type == PhysicalOperatorType::TABLE_SCAN) {
					auto &scan = op.Cast<PhysicalTableScan>();
					if (!scan.extra_info.scan_group_id.IsValid()) {
						if (scan.extra_info.scan_node_id.IsValid()) {
							scan.extra_info.scan_group_id = scan.extra_info.scan_node_id;
						} else {
							scan.extra_info.scan_group_id = optional_idx(next_id++);
						}
					}
					if (!scan.extra_info.scan_node_id.IsValid()) {
						scan.extra_info.scan_node_id = optional_idx(next_id++);
					}
					const auto group_id = scan.extra_info.scan_group_id.GetIndex();
					auto node_id = scan.extra_info.scan_node_id.GetIndex();
					auto it = base_for_group.find(group_id);
					if (it == base_for_group.end()) {
						base_for_group[group_id] = node_id;
					} else {
						auto base_id = it->second;
						if (node_id == base_id) {
							node_id = next_id++;
							scan.extra_info.scan_node_id = optional_idx(node_id);
						}
						dup_to_base[node_id] = base_id;
					}
				}
				for (auto &child : op.children) {
					normalize(child.get());
				}
			};
			normalize(plan_->physical_plan()->Root());

			idx_t copied = 0;
			for (const auto &kv : dup_to_base) {
				auto dup_key = py::str(std::to_string(kv.first));
				if (out.contains(dup_key)) {
					continue;
				}
				auto base_key = py::str(std::to_string(kv.second));
				if (out.contains(base_key)) {
					out[dup_key] = out[base_key];
					copied++;
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

		// Need query_id for pool name construction.
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

				// Build pool name using the same sanitization as VLLMProjectNode.
				auto safe_query = duckdb::distributed::SanitizePoolComponent(query_id);
				auto pool_name = "duckdb_vllm_" + safe_query + "_" + std::to_string(node_id);

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
			RememberQueryUDFActorHandles(query_id_, udf_actor_handles_);
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

PyPhysicalPlanWrapper PyLogicalPlan::to_physical_plan(py::object conn_obj) const {
	if (conn_obj.is_none()) {
		throw duckdb::InternalException("Connection is required for to_physical_plan");
	}

	string logical_payload;
	if (!serialized_logical_plan_.empty()) {
		logical_payload = serialized_logical_plan_;
	} else if (relation_) {
		logical_payload = SerializeLogicalPlanFromRelation(relation_);
	} else {
		throw duckdb::InternalException("Logical plan is empty and no relation is available");
	}

	py::object planning_conn = ResolveConnectionForSnapshot(conn_obj, connection_snapshot_);
	auto &conn_wrapper = ExtractPyConnectionWrapper(planning_conn);
	ApplyConnectionSnapshot(planning_conn, connection_snapshot_);
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
	plan_wrapper.worker_connection_ = planning_conn;
	plan_wrapper.client_context_ = conn_wrapper.con.GetConnection().context;
	plan_wrapper.udf_registrations_ = udf_registrations_;
	plan_wrapper.connection_snapshot_ = connection_snapshot_;
	RememberQueryUDFRegistrations(plan_wrapper.query_id_, plan_wrapper.udf_registrations_);
	return plan_wrapper;
}

class PyBackendWorkerManager : public duckdb::distributed::WorkerManager {
public:
	template <class T>
	using DuckDBResult = duckdb::distributed::DuckDBResult<T>;
	using DuckDBError = duckdb::distributed::DuckDBError;

	explicit PyBackendWorkerManager(py::object backend) : backend_(std::move(backend)) {
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
		duckdb::PythonGILWrapper gil;
		try {
			auto backend = backend_.get();
			if (py::hasattr(backend, "shutdown")) {
				backend.attr("shutdown")();
			}
			return DuckDBResult<void>::ok();
		} catch (const std::exception &e) {
			return DuckDBResult<void>::err(DuckDBError(string("Python backend shutdown failed: ") + e.what()));
		}
	}

	DuckDBResult<void> submit_fte_task_events(std::vector<duckdb::distributed::WorkerTask> tasks) override {
		try {
			auto query_id = QueryIdFromTaskEvents(tasks);
			if (!tasks.empty() && query_id.empty()) {
				return DuckDBResult<void>::err(DuckDBError::value_error("FTE task events require non-empty query_id"));
			}
			duckdb::PythonGILWrapper gil;
			py::list py_tasks;
			for (auto &task : tasks) {
				duckdb::distributed::python::ray::RayWorkerTask py_task_wrapper(std::move(task));
				py_tasks.append(py::cast(std::move(py_task_wrapper), py::return_value_policy::move));
			}
			auto backend = backend_.get();
			py::object raw_handles = backend.attr("submit_tasks")(py_tasks);
			StorePythonResultHandles(query_id, std::move(raw_handles));
			return DuckDBResult<void>::ok();
		} catch (const std::exception &e) {
			return DuckDBResult<void>::err(
			    DuckDBError(string("Python backend submit_fte_task_events failed: ") + e.what()));
		}
	}

	DuckDBResult<void> task_input_stream_exhausted_for_query(
	    const string &query_id, const std::unordered_set<duckdb::distributed::SourceNodeId> &source_node_ids) override {
		if (query_id.empty()) {
			return DuckDBResult<void>::err(
			    DuckDBError::value_error("FTE task input exhaustion requires non-empty query_id"));
		}
		try {
			duckdb::PythonGILWrapper gil;
			py::list py_source_node_ids;
			for (auto source_node_id : source_node_ids) {
				py_source_node_ids.append(std::to_string(source_node_id));
			}
			auto backend = backend_.get();
			py::object raw_handles =
			    backend.attr("task_input_stream_exhausted")(query_id, py_source_node_ids);
			StorePythonResultHandles(query_id, std::move(raw_handles));
			return DuckDBResult<void>::ok();
		} catch (const std::exception &e) {
			return DuckDBResult<void>::err(
			    DuckDBError(string("Python backend task_input_stream_exhausted failed: ") + e.what()));
		}
	}

	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> wait_fte_query(
	    const string &query_id, double timeout_s) override {
		return wait_fte_query(query_id, timeout_s, {});
	}

	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> wait_fte_query(
	    const string &query_id, double timeout_s, duckdb::distributed::MaterializedOutputCallback on_output) override {
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
			string status_message;
			try {
				duckdb::PythonGILWrapper gil;
				auto backend = backend_.get();
				py::object status_obj = backend.attr("fte_query_status")(query_id);
				status_message = PyStatusMessage(status_obj);
				failed = RequiredStatusBool(status_obj, "failed");
				finished = RequiredStatusBool(status_obj, "finished");
				canceled = OptionalStatusBool(status_obj, "canceled", false);
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
			if (finished) {
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
		auto drain_res =
		    DrainResultHandles(query_id, remaining_timeout_s, selected_attempt_task_ids,
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

	void drop_query_fragments(const string &query_id) {
		if (query_id.empty()) {
			return;
		}
		std::exception_ptr result_cleanup_error;
		try {
			ClearResultHandles(query_id);
		} catch (...) {
			result_cleanup_error = std::current_exception();
		}
		std::exception_ptr backend_drop_error;
		{
			duckdb::PythonGILWrapper gil;
			try {
				auto backend = backend_.get();
				if (py::hasattr(backend, "drop_query")) {
					backend.attr("drop_query")(query_id);
				}
			} catch (...) {
				backend_drop_error = std::current_exception();
			}
		}
		if (result_cleanup_error && backend_drop_error) {
			auto exception_message = [](const std::exception_ptr &error) {
				try {
					std::rethrow_exception(error);
				} catch (const std::exception &e) {
					return string(e.what());
				} catch (...) {
					return string("unknown exception");
				}
			};
			throw std::runtime_error(
			    "Python backend query teardown failed: result cleanup=" +
			    exception_message(result_cleanup_error) + "; backend drop=" +
			    exception_message(backend_drop_error));
		}
		if (result_cleanup_error) {
			std::rethrow_exception(result_cleanup_error);
		}
		if (backend_drop_error) {
			std::rethrow_exception(backend_drop_error);
		}
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
	mutable mutex mutex_;
	duckdb::distributed::python::ray::SafePyObject backend_;
	std::unordered_map<string, std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>>>
	    result_handles_by_query_;
	std::unordered_map<string, std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>>>
	    retained_result_handles_by_query_;

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

	void StorePythonResultHandles(const string &query_id, py::object raw_handles) {
		if (query_id.empty() || raw_handles.is_none()) {
			return;
		}
		std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> wrapped;
		for (auto item : raw_handles) {
			auto py_handle = py::reinterpret_borrow<py::object>(item);
			auto handle = duckdb::distributed::python::ray::MakePythonTaskResultHandle(std::move(py_handle));
			wrapped.push_back(
			    make_uniq<duckdb::distributed::python::ray::PythonTaskResultHandle>(std::move(handle)));
		}
		if (wrapped.empty()) {
			return;
		}
		lock_guard<mutex> guard(mutex_);
		auto &stored = result_handles_by_query_[query_id];
		stored.reserve(stored.size() + wrapped.size());
		for (auto &handle : wrapped) {
			stored.push_back(std::move(handle));
		}
	}

	void StorePythonTaskResultHandles(
	    const string &query_id,
	    std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> handles) {
		if (query_id.empty() || handles.empty()) {
			return;
		}
		lock_guard<mutex> guard(mutex_);
		auto &stored = result_handles_by_query_[query_id];
		stored.reserve(stored.size() + handles.size());
		for (auto &handle : handles) {
			stored.push_back(std::move(handle));
		}
	}

	void RetainPythonTaskResultHandles(
	    const string &query_id,
	    std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> handles) {
		if (query_id.empty() || handles.empty()) {
			return;
		}
		lock_guard<mutex> guard(mutex_);
		auto &retained = retained_result_handles_by_query_[query_id];
		retained.reserve(retained.size() + handles.size());
		for (auto &handle : handles) {
			retained.push_back(std::move(handle));
		}
	}

	void ClearResultHandles(const string &query_id) {
		std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> handles;
		std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> retained_handles;
		{
			lock_guard<mutex> guard(mutex_);
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
					errors.push_back(string(kind) + "[" + std::to_string(index) +
					                 "]: unknown release error");
					retry_handles.push_back(std::move(owned_handles[index]));
				}
			}
		};
		release_all(handles, "pending");
		release_all(retained_handles, "retained");
		StorePythonTaskResultHandles(query_id, std::move(retry_handles));
		if (!errors.empty()) {
			string message = "failed to release " + std::to_string(errors.size()) +
			                 " backend result handle(s)";
			for (const auto &error : errors) {
				message += "; " + error;
			}
			throw std::runtime_error(message);
		}
	}

	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> DrainResultHandles(
	    const string &query_id, double timeout_s, const std::unordered_set<string> &selected_attempt_task_ids,
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
						release_errors.push_back("unselected[" + std::to_string(index) +
						                         "]: unknown release error");
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
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError(message));
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
					drain_errors.push_back(
					    string("failed to finalize Python backend result handle: ") + ex.what());
				} catch (...) {
					still_pending.push_back(std::move(handle));
					drain_errors.push_back(
					    "failed to finalize Python backend result handle: unknown error");
				}
			}
			if (!drain_errors.empty()) {
				StorePythonTaskResultHandles(query_id, std::move(still_pending));
				RetainPythonTaskResultHandles(query_id, std::move(retained_handles));
				string message = "failed to drain Python backend result handle(s)";
				for (const auto &error : drain_errors) {
					message += "; " + error;
				}
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError(message));
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

	PyPhysicalPlanWrapperRunner()
	    : ray_worker_manager_(std::make_shared<duckdb::distributed::python::ray::RayWorkerManager>()) {
		worker_manager_ = ray_worker_manager_;
	}

	explicit PyPhysicalPlanWrapperRunner(py::object backend) {
		if (backend.is_none()) {
			ray_worker_manager_ = std::make_shared<duckdb::distributed::python::ray::RayWorkerManager>();
			worker_manager_ = ray_worker_manager_;
			return;
		}
		py_backend_worker_manager_ = std::make_shared<PyBackendWorkerManager>(std::move(backend));
		worker_manager_ = py_backend_worker_manager_;
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
		CleanupQueryPythonReplayState(query_id);
		if (teardown_error) {
			std::rethrow_exception(teardown_error);
		}
	}

	void warm_up() {
		worker_manager_->worker_snapshots();
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

	std::shared_ptr<ResultPartitionStream> run_plan(const PyPhysicalPlanWrapper &plan,
	                                                duckdb::shared_ptr<duckdb::ClientContext> client_context = nullptr,
	                                                duckdb::distributed::python::ray::SafePyObject py_conn_keepalive =
	                                                    duckdb::distributed::python::ray::SafePyObject()) {
		using namespace duckdb::distributed;
		RememberQueryConnectionSnapshot(plan.idx(), plan.connection_snapshot_);
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
		auto cleanup_failed_start = [&](const std::exception_ptr &execution_error) {
			std::exception_ptr cleanup_error;
			try {
				drop_query_fragments(plan.idx());
			} catch (...) {
				cleanup_error = std::current_exception();
			}
			if (cleanup_error) {
				throw std::runtime_error(
				    "distributed streaming plan startup and teardown both failed: startup=" +
				    exception_message(execution_error) + "; teardown=" + exception_message(cleanup_error));
			}
		};

		// Call PlanRunner::run_plan
		try {
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

			auto runner =
			    std::make_shared<duckdb::distributed::PlanRunner>(
			        worker_manager_, client_context);
			run_state->runner = runner;
			DuckDBResult<duckdb::distributed::PlanRunner::PlanResult> res;
			{
				py::gil_scoped_release release;
				res = runner->run_plan(plan.plan_);
			}

			if (!res.is_ok()) {
				throw py::value_error(res.error().what());
			}

			auto &plan_result = res.value();
			if (plan_result.tag != duckdb::distributed::PlanRunner::PlanResult::STREAMING) {
				throw py::value_error("run_plan returned copy result instead of stream for a streaming plan");
			}
			auto stream = std::make_shared<duckdb::distributed::PlanResultStream>(std::move(plan_result.stream));
			auto py_stream = std::make_shared<ResultPartitionStream>(stream);
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
	                             duckdb::distributed::python::ray::SafePyObject()) {
		using namespace duckdb::distributed;
		RememberQueryConnectionSnapshot(plan.idx(), plan.connection_snapshot_);
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
		std::exception_ptr cleanup_error;
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
			auto cleanup_started = std::chrono::steady_clock::now();
			try {
				drop_query_fragments(plan.idx());
				CleanupQueryPythonReplayState(plan.idx());
			} catch (...) {
				cleanup_error = std::current_exception();
			}
			cleanup_ms = elapsed_ms(cleanup_started);
		};

		try {
			if (!plan.plan_) {
				throw py::value_error("DistributedPhysicalPlan missing underlying physical plan");
			}
			bool has_root = plan.plan_->physical_plan() && plan.plan_->physical_plan()->HasRoot();
			if (!has_root) {
				throw py::value_error("DistributedPhysicalPlan has no root (empty placeholder)");
			}

			auto runner = std::make_shared<duckdb::distributed::PlanRunner>(worker_manager_, client_context);

			DuckDBResult<DistributedCopyResult> res;
			idx_t run_plan_ms = 0;
			{
				auto run_plan_started = std::chrono::steady_clock::now();
				py::gil_scoped_release release;
				DuckDBResult<duckdb::distributed::PlanRunner::PlanResult> plan_res;
				if (!client_context) {
					plan_res = runner->run_plan(plan.plan_);
				} else {
					client_context->RunFunctionInTransaction([&]() {
						plan_res = runner->run_plan(plan.plan_);
					});
				}
				if (!plan_res.is_ok()) {
					res = DuckDBResult<DistributedCopyResult>::err(plan_res.error());
				} else {
					auto &plan_result = plan_res.value();
					if (plan_result.tag == duckdb::distributed::PlanRunner::PlanResult::COPY) {
						res = DuckDBResult<DistributedCopyResult>::ok(std::move(plan_result.copy_result));
					} else {
						res = DuckDBResult<DistributedCopyResult>::err(
						    DuckDBError("run_plan returned stream instead of copy result for a COPY plan"));
					}
				}
				run_plan_ms = elapsed_ms(run_plan_started);
			}

			if (!res.is_ok()) {
				throw py::value_error(res.error().what());
			}

			auto result = std::move(res).value();
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
				entry["partition_keys"] = info.partition_keys.IsNull() ? py::object(py::none())
				                                                       : py::object(py::str(info.partition_keys.ToString()));
				files.append(entry);
			}
			py::dict out;
			out["rows_copied"] = result.rows_copied;
			out["files"] = files;
			AppendDistributedCopyResultMetadata(out, result);
			body_succeeded = true;
			cleanup();
			if (cleanup_error) {
				std::rethrow_exception(cleanup_error);
			}
			out["copy_total_ms"] = py::int_(elapsed_ms(copy_started));
			out["copy_run_plan_ms"] = py::int_(run_plan_ms);
			out["copy_runner_cleanup_ms"] = py::int_(cleanup_ms);
			out["copy_selected_file_count"] = py::int_(result.files.size());
			out["copy_duplicate_file_count"] = py::int_(0);
			return out;
		} catch (...) {
			auto execution_error = std::current_exception();
			if (body_succeeded) {
				std::rethrow_exception(execution_error);
			}
			cleanup();
			if (cleanup_error) {
				throw std::runtime_error(
				    "distributed COPY execution and teardown both failed: execution=" +
				    exception_message(execution_error) + "; teardown=" + exception_message(cleanup_error));
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
		std::vector<DistributedCopyFileInfo> files;
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
				client_context->RunFunctionInTransaction([&]() {
					res = runner->finalize_copy(ctx->copy_spec, ctx->staging_root, std::move(files));
				});
			}
			auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			    std::chrono::steady_clock::now() - finalize_started);
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
		(void)runtime_context_obj;

		// Validate that we have a physical plan with a root
		if (!physical_plan || !physical_plan->HasRoot()) {
			throw py::value_error("Physical plan is missing or has no root operator");
		}

		if (scan_task_map && !scan_task_map->empty()) {
			string error;
			if (!duckdb::distributed::ApplyScanTasksToPlan(*physical_plan, *scan_task_map, &error)) {
				throw py::value_error("Failed to apply scan tasks to plan: " + error);
			}
		}

		if (fte_scan_source_queue_map && !fte_scan_source_queue_map->empty()) {
			string error;
			if (!duckdb::distributed::ApplyFteScanSourceQueuesToPlan(*physical_plan, *fte_scan_source_queue_map,
			                                                         &error)) {
				throw py::value_error("Failed to apply FTE scan source queues to plan: " + error);
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

		auto &root_op = physical_plan->Root();
		py::object task_exchange_sink_instance_obj = py::none();
		if (exchange_sink_instance_task) {
			task_exchange_sink_instance_obj = py::bytes(exchange_sink_instance_task->SerializeToBytes());
		}

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
			                             std::move(task_stats), status,
			                             duckdb::distributed::FlightExchangeManager::GetLocalFlightServerPort(),
			                             task_exchange_sink_instance_obj);
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
			                             std::move(task_stats), "ok",
			                             duckdb::distributed::FlightExchangeManager::GetLocalFlightServerPort(),
			                             task_exchange_sink_instance_obj);
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
		const bool needs_result_collector = NativePlanNeedsResultCollector(root_op);
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
			std::unique_ptr<QueryResult> query_result;
			vector<PipelineProgressSnapshot> stable_pipeline_snapshots;
			try {
				const bool has_native_progress_callback = !native_progress_callback.is_none();
				const auto native_progress_env_ms = DuckdbGetEnvIntMs("VANE_NATIVE_PROGRESS_INTERVAL_MS");
				const auto native_progress_ms =
				    has_native_progress_callback ? (native_progress_env_ms > 0 ? native_progress_env_ms : 500) : 0;
				auto &client_config = duckdb::ClientConfig::GetConfig(context);
				// Force parallel result collector by providing a custom get_result_collector
				// callback that always creates PhysicalMaterializedCollector(parallel=true).
				// This enables multi-threaded pipeline execution for UDF-heavy workloads.
				client_config.get_result_collector =
				    [](duckdb::ClientContext &, duckdb::PreparedStatementData &data) -> duckdb::PhysicalOperator & {
						auto &physical_plan = *data.physical_plan;
						return physical_plan.Make<duckdb::PhysicalMaterializedCollector>(data, true);
					};

				DuckdbGilReleaseMarker gil_marker;
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
					if (!has_native_progress_callback || native_progress_callback_failed) {
						return;
					}
					auto now = std::chrono::steady_clock::now();
					auto elapsed_ms =
					    std::chrono::duration_cast<std::chrono::milliseconds>(now - last_native_progress).count();
					if (!force && elapsed_ms < native_progress_ms) {
						return;
					}
					auto snapshots = executor.GetPipelinesProgressSnapshots();
					if (!snapshots.empty()) {
						stable_pipeline_snapshots = snapshots;
					}
					py::gil_scoped_acquire acquire;
					try {
						auto stats = BuildNativeTaskStatsDict(stable_pipeline_snapshots, scan_task_map,
						                                      exchange_source_task_map,
						                                      fte_scan_source_queue_map,
						                                      fte_exchange_source_queue_map);
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
						auto stats = BuildNativeTaskStatsDict(
						    stable_pipeline_snapshots, scan_task_map, exchange_source_task_map,
						    fte_scan_source_queue_map, fte_exchange_source_queue_map);
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
							if (native_progress_cv.wait_for(
							        lock, interval, [&]() { return native_progress_stop.load(); })) {
								break;
							}
							lock.unlock();
							if (DuckdbPyIsFinalizing()) {
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
			auto table = CollectionToArrowTable(collection, context);
			const auto payload_bytes = static_cast<idx_t>(GetPyPayloadSizeBytes(table));
			auto materialized_stats =
			    BuildMaterializedInputTaskStats(root_op, scan_task_map, exchange_source_task_map,
			                                    fte_scan_source_queue_map, fte_exchange_source_queue_map,
			                                    collection.Count(), payload_bytes, &materialized);
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
					auto stats = BuildNativeTaskStatsDict(stable_pipeline_snapshots, scan_task_map,
					                                      exchange_source_task_map, fte_scan_source_queue_map,
					                                      fte_exchange_source_queue_map);
					native_progress_callback(stats);
					last_native_progress = now;
				} catch (const std::exception &ex) {
					native_progress_callback_failed = true;
				} catch (...) {
					native_progress_callback_failed = true;
				}
			};
			DuckdbGilReleaseMarker gil_marker;
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
		py::object final_task_stats = BuildNativeTaskStatsDict(final_pipeline_snapshots, scan_task_map,
		                                                        exchange_source_task_map, fte_scan_source_queue_map,
		                                                        fte_exchange_source_queue_map);

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
		auto cleanup = [&]() {
			CleanupQueryPythonReplayState(plan.idx());
		};
		try {
			auto &mutable_plan = const_cast<PyPhysicalPlanWrapper &>(plan);
			py::object exec_conn = ResolveConnectionForSnapshot(conn_obj, plan.connection_snapshot_);
			mutable_plan.ensure_connection_snapshot(exec_conn);
			mutable_plan.apply_udf_actor_handles();

			// Validate that we have a physical plan
			if (!plan.plan_ || !plan.plan_->physical_plan()) {
				throw py::value_error("DistributedPhysicalPlan is missing underlying physical plan");
			}

			auto result = execute_native_impl(exec_conn, plan.plan_->physical_plan(), plan.idx(), nullptr, nullptr,
			                                  nullptr, nullptr, nullptr, nullptr, py::none(), py::none());
			cleanup();
			return result;
		} catch (...) {
			cleanup();
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
	topology_plan.udf_registrations_ = plan.udf_registrations_;
	topology_plan.udf_actor_handles_ = plan.udf_actor_handles_;
	topology_plan.connection_snapshot_ = plan.connection_snapshot_;
	topology_plan.serialized_root_ = plan.serialize_root_for_clone();
	topology_plan.worker_connection_ = exec_conn;
	topology_plan.materialize_deferred_root(exec_conn);
	topology_plan.apply_udf_actor_handles();
	if (!topology_plan.plan_ || !topology_plan.plan_->physical_plan() || !topology_plan.has_root()) {
		throw py::value_error("DistributedPhysicalPlan topology clone has no physical root");
	}

	auto &conn_wrapper = ExtractPyConnectionWrapper(exec_conn);
	auto &context = *conn_wrapper.con.GetConnection().context;
	return BuildNativeProgressTopology(context, topology_plan.plan_->physical_plan());
}
