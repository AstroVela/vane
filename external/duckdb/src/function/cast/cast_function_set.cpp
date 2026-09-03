#include "duckdb/function/cast/cast_function_set.hpp"

#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/main/settings.hpp"

#include "duckdb/common/insertion_order_preserving_map.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/types/type_map.hpp"
#include "duckdb/function/cast_rules.hpp"
#include "duckdb/planner/collation_binding.hpp"
#include "duckdb/main/config.hpp"

namespace duckdb {

BindCastInput::BindCastInput(CastFunctionSet &function_set, optional_ptr<BindCastInfo> info,
                             optional_ptr<ClientContext> context)
    : function_set(function_set), info(info), context(context) {
}

BoundCastInfo BindCastInput::GetCastFunction(const LogicalType &source, const LogicalType &target) {
	GetCastFunctionInput input(context);
	input.query_location = query_location;
	input.file_cast_mode = file_cast_mode;
	return function_set.GetCastFunction(source, target, input);
}

BindCastFunction::BindCastFunction(bind_cast_function_t function_p, unique_ptr<BindCastInfo> info_p)
    : function(function_p), info(std::move(info_p)) {
}

CastFunctionSet::CastFunctionSet() : map_info(nullptr) {
	bind_functions.emplace_back(DefaultCasts::GetDefaultCastFunction);
}

CastFunctionSet::CastFunctionSet(DBConfig &config_p) : CastFunctionSet() {
	this->config = &config_p;
}

CastFunctionSet &CastFunctionSet::Get(ClientContext &context) {
	return DBConfig::GetConfig(context).GetCastFunctions();
}

CollationBinding &CollationBinding::Get(ClientContext &context) {
	return DBConfig::GetConfig(context).GetCollationBinding();
}

CastFunctionSet &CastFunctionSet::Get(DatabaseInstance &db) {
	return DBConfig::GetConfig(db).GetCastFunctions();
}

CollationBinding &CollationBinding::Get(DatabaseInstance &db) {
	return DBConfig::GetConfig(db).GetCollationBinding();
}

using governed_child_compatibility_t = bool (*)(const LogicalType &, const LogicalType &);

static bool StructGovernedChildrenCompatible(const LogicalType &source, const LogicalType &target,
                                             governed_child_compatibility_t child_compatible) {
	if (source.id() != LogicalTypeId::STRUCT || !source.AuxInfo()) {
		return false;
	}
	auto &source_children = StructType::GetChildTypes(source);
	auto &target_children = StructType::GetChildTypes(target);
	auto is_unnamed = source_children.empty() || target_children.empty() || StructType::IsUnnamed(source) ||
	                  StructType::IsUnnamed(target);
	if (is_unnamed) {
		if (source_children.size() != target_children.size()) {
			return false;
		}
		for (idx_t index = 0; index < target_children.size(); index++) {
			if (!child_compatible(source_children[index].second, target_children[index].second)) {
				return false;
			}
		}
		return true;
	}

	InsertionOrderPreservingMap<idx_t> target_children_map;
	for (idx_t index = 0; index < target_children.size(); index++) {
		target_children_map[target_children[index].first] = index;
	}

	bool has_any_match = false;
	for (auto &source_child : source_children) {
		auto target_child = target_children_map.find(source_child.first);
		if (target_child == target_children_map.end()) {
			if (TypeVisitor::Contains(source_child.second, GovernedLogicalType::IsGoverned)) {
				return false;
			}
			continue;
		}
		has_any_match = true;
		if (!child_compatible(source_child.second, target_children[target_child->second].second)) {
			return false;
		}
		target_children_map.erase(target_child);
	}

	for (auto &target_child : target_children_map) {
		if (TypeVisitor::Contains(target_children[target_child.second].second, GovernedLogicalType::IsGoverned)) {
			return false;
		}
	}
	return has_any_match;
}

static bool UnionGovernedChildrenCompatible(const LogicalType &source, const LogicalType &target,
                                            governed_child_compatibility_t child_compatible) {
	if (source.id() != LogicalTypeId::UNION || !source.AuxInfo()) {
		return false;
	}
	vector<bool> matched_targets(UnionType::GetMemberCount(target), false);
	for (idx_t source_index = 0; source_index < UnionType::GetMemberCount(source); source_index++) {
		auto &source_name = UnionType::GetMemberName(source, source_index);
		bool found = false;
		for (idx_t target_index = 0; target_index < UnionType::GetMemberCount(target); target_index++) {
			if (!StringUtil::CIEquals(source_name, UnionType::GetMemberName(target, target_index))) {
				continue;
			}
			if (!child_compatible(UnionType::GetMemberType(source, source_index),
			                      UnionType::GetMemberType(target, target_index))) {
				return false;
			}
			matched_targets[target_index] = true;
			found = true;
			break;
		}
		if (!found) {
			return false;
		}
	}
	for (idx_t target_index = 0; target_index < matched_targets.size(); target_index++) {
		if (!matched_targets[target_index] &&
		    TypeVisitor::Contains(UnionType::GetMemberType(target, target_index), GovernedLogicalType::IsGoverned)) {
			return false;
		}
	}
	return true;
}

static bool GovernedAliasRestorationCompatible(const LogicalType &source, const LogicalType &target) {
	if (GovernedLogicalType::IsGoverned(target)) {
		auto physical_type = target.DeepCopy();
		physical_type.SetAlias(string());
		return source == physical_type;
	}
	if (!TypeVisitor::Contains(target, GovernedLogicalType::IsGoverned)) {
		// Ordinary siblings retain the normal DuckDB cast rules. This is
		// required for validated UDF outputs such as STRUCT(IMAGE, UUID).
		return true;
	}

	switch (target.id()) {
	case LogicalTypeId::STRUCT:
		return StructGovernedChildrenCompatible(source, target, GovernedAliasRestorationCompatible);
	case LogicalTypeId::UNION:
		return UnionGovernedChildrenCompatible(source, target, GovernedAliasRestorationCompatible);
	case LogicalTypeId::LIST:
		if (source.id() == LogicalTypeId::LIST && source.AuxInfo()) {
			return GovernedAliasRestorationCompatible(ListType::GetChildType(source), ListType::GetChildType(target));
		}
		if (source.id() == LogicalTypeId::ARRAY && source.AuxInfo()) {
			return GovernedAliasRestorationCompatible(ArrayType::GetChildType(source), ListType::GetChildType(target));
		}
		return false;
	case LogicalTypeId::ARRAY:
		if (source.id() == LogicalTypeId::ARRAY && source.AuxInfo()) {
			return GovernedAliasRestorationCompatible(ArrayType::GetChildType(source), ArrayType::GetChildType(target));
		}
		if (source.id() == LogicalTypeId::LIST && source.AuxInfo()) {
			return GovernedAliasRestorationCompatible(ListType::GetChildType(source), ArrayType::GetChildType(target));
		}
		return false;
	case LogicalTypeId::MAP:
		return source.id() == LogicalTypeId::MAP && source.AuxInfo() &&
		       GovernedAliasRestorationCompatible(MapType::KeyType(source), MapType::KeyType(target)) &&
		       GovernedAliasRestorationCompatible(MapType::ValueType(source), MapType::ValueType(target));
	default:
		return false;
	}
}

static bool GovernedLeavesPreservedCompatible(const LogicalType &source, const LogicalType &target) {
	if (source.id() == LogicalTypeId::SQLNULL) {
		// Untyped NULL leaves carry no value that could bypass logical-value
		// validation. This also permits empty and NULL-only literals such as
		// []::IMAGE[] and [NULL]::IMAGE[].
		return true;
	}
	auto source_contains_governed = TypeVisitor::Contains(source, GovernedLogicalType::IsGoverned);
	auto target_contains_governed = TypeVisitor::Contains(target, GovernedLogicalType::IsGoverned);
	if (!source_contains_governed && !target_contains_governed) {
		// Ordinary siblings retain the normal DuckDB cast rules.
		return true;
	}
	if (target.id() == LogicalTypeId::UNION && source.id() != LogicalTypeId::UNION && target.AuxInfo()) {
		// DuckDB promotes a non-UNION value to the best matching UNION member.
		// Permit that native path only when at least one member preserves every
		// governed leaf; the selected member is checked again by its child cast.
		for (idx_t target_index = 0; target_index < UnionType::GetMemberCount(target); target_index++) {
			if (GovernedLeavesPreservedCompatible(source, UnionType::GetMemberType(target, target_index))) {
				return true;
			}
		}
		return false;
	}
	if (GovernedLogicalType::IsGoverned(source) || GovernedLogicalType::IsGoverned(target)) {
		return GovernedLogicalType::IsGoverned(source) && GovernedLogicalType::IsGoverned(target) && source == target;
	}
	if (!target_contains_governed) {
		return false;
	}

	switch (target.id()) {
	case LogicalTypeId::STRUCT:
		return StructGovernedChildrenCompatible(source, target, GovernedLeavesPreservedCompatible);
	case LogicalTypeId::UNION:
		return UnionGovernedChildrenCompatible(source, target, GovernedLeavesPreservedCompatible);
	case LogicalTypeId::LIST:
		return source.id() == LogicalTypeId::LIST && source.AuxInfo() &&
		       GovernedLeavesPreservedCompatible(ListType::GetChildType(source), ListType::GetChildType(target));
	case LogicalTypeId::ARRAY:
		return source.id() == LogicalTypeId::ARRAY && source.AuxInfo() &&
		       ArrayType::GetSize(source) == ArrayType::GetSize(target) &&
		       GovernedLeavesPreservedCompatible(ArrayType::GetChildType(source), ArrayType::GetChildType(target));
	case LogicalTypeId::MAP:
		return source.id() == LogicalTypeId::MAP && source.AuxInfo() &&
		       GovernedLeavesPreservedCompatible(MapType::KeyType(source), MapType::KeyType(target)) &&
		       GovernedLeavesPreservedCompatible(MapType::ValueType(source), MapType::ValueType(target));
	default:
		return false;
	}
}

static bool GovernedImplicitCastCompatible(const LogicalType &source, const LogicalType &target) {
	if (source.id() == LogicalTypeId::UNKNOWN || source.id() == LogicalTypeId::SQLNULL ||
	    target.id() == LogicalTypeId::ANY || target.id() == LogicalTypeId::TEMPLATE ||
	    target.id() == LogicalTypeId::UNKNOWN) {
		return true;
	}
	auto source_contains_governed = TypeVisitor::Contains(source, GovernedLogicalType::IsGoverned);
	auto target_contains_governed = TypeVisitor::Contains(target, GovernedLogicalType::IsGoverned);
	if (!source_contains_governed && !target_contains_governed) {
		return true;
	}
	if (!target.IsComplete()) {
		// Generic function signatures use incomplete composite types such as
		// UNION without child metadata. Their bind callbacks replace the
		// placeholder with the concrete input type before any cast is created.
		return true;
	}
	return GovernedLeavesPreservedCompatible(source, target);
}

BoundCastInfo CastFunctionSet::GetCastFunction(const LogicalType &source, const LogicalType &target,
                                               GetCastFunctionInput &get_input) {
	if (source == target) {
		return DefaultCasts::NopCast;
	}
	// Governed aliases are semantic types, not presentation aliases. Letting the
	// ordinary STRUCT cast path handle them would silently retag values and
	// bypass their constructors and field validation.
	auto source_contains_governed = TypeVisitor::Contains(source, GovernedLogicalType::IsGoverned);
	auto target_contains_governed = TypeVisitor::Contains(target, GovernedLogicalType::IsGoverned);
	auto internal_governed_cast_allowed = [&]() {
		switch (get_input.file_cast_mode) {
		case FileCastMode::INTERNAL_FORMATTING:
			return source_contains_governed && target == LogicalType::VARCHAR;
		case FileCastMode::INTERNAL_ALIAS_RESTORATION:
			return target_contains_governed && !source_contains_governed &&
			       GovernedAliasRestorationCompatible(source, target);
		case FileCastMode::STRICT:
			return target_contains_governed && GovernedLeavesPreservedCompatible(source, target);
		default:
			return false;
		}
	}();
	if ((source_contains_governed || target_contains_governed) && source.id() != LogicalTypeId::SQLNULL &&
	    target.id() != LogicalTypeId::SQLNULL && !internal_governed_cast_allowed) {
		throw BinderException(get_input.query_location,
		                      "Cannot cast from %s to %s: governed values require an exact logical type match",
		                      source.ToString(), target.ToString());
	}
	// the first function is the default
	// we iterate the set of bind functions backwards
	for (idx_t i = bind_functions.size(); i > 0; i--) {
		auto &bind_function = bind_functions[i - 1];
		BindCastInput input(*this, bind_function.info.get(), get_input.context);
		input.query_location = get_input.query_location;
		input.file_cast_mode = get_input.file_cast_mode;
		auto result = bind_function.function(input, source, target);
		if (result.function) {
			// found a cast function! return it
			return result;
		}
	}
	// no cast found: return the default null cast
	return DefaultCasts::TryVectorNullCast;
}

struct MapCastNode {
	MapCastNode(BoundCastInfo info, int64_t implicit_cast_cost)
	    : cast_info(std::move(info)), bind_function(nullptr), implicit_cast_cost(implicit_cast_cost) {
	}
	MapCastNode(bind_cast_function_t func, int64_t implicit_cast_cost)
	    : cast_info(nullptr), bind_function(func), implicit_cast_cost(implicit_cast_cost) {
	}

	BoundCastInfo cast_info;
	bind_cast_function_t bind_function;
	int64_t implicit_cast_cost;
};

template <class MAP_VALUE_TYPE>
static auto RelaxedTypeMatch(type_map_t<MAP_VALUE_TYPE> &map, const LogicalType &type) -> decltype(map.find(type)) {
	D_ASSERT(map.find(type) == map.end()); // we shouldn't be here
	switch (type.id()) {
	case LogicalTypeId::LIST:
		return map.find(LogicalType::LIST(LogicalType::ANY));
	case LogicalTypeId::STRUCT:
		return map.find(LogicalType::STRUCT({{"any", LogicalType::ANY}}));
	case LogicalTypeId::MAP:
		for (auto it = map.begin(); it != map.end(); it++) {
			const auto &entry_type = it->first;
			if (entry_type.id() != LogicalTypeId::MAP) {
				continue;
			}
			auto &entry_key_type = MapType::KeyType(entry_type);
			auto &entry_val_type = MapType::ValueType(entry_type);
			if ((entry_key_type == LogicalType::ANY || entry_key_type == MapType::KeyType(type)) &&
			    (entry_val_type == LogicalType::ANY || entry_val_type == MapType::ValueType(type))) {
				return it;
			}
		}
		return map.end();
	case LogicalTypeId::UNION:
		return map.find(LogicalType::UNION({{"any", LogicalType::ANY}}));
	case LogicalTypeId::ARRAY:
		return map.find(LogicalType::ARRAY(LogicalType::ANY, optional_idx()));
	case LogicalTypeId::DECIMAL:
		return map.find(LogicalTypeId::DECIMAL);
	case LogicalTypeId::ENUM:
		return map.find(LogicalTypeId::ENUM);
	default:
		return map.find(LogicalType::ANY);
	}
}

struct MapCastInfo : public BindCastInfo {
public:
	const optional_ptr<MapCastNode> GetEntry(const LogicalType &source, const LogicalType &target) {
		auto source_type_id_entry = casts.find(source.id());
		if (source_type_id_entry == casts.end()) {
			source_type_id_entry = casts.find(LogicalTypeId::ANY);
			if (source_type_id_entry == casts.end()) {
				return nullptr;
			}
		}

		auto &source_type_entries = source_type_id_entry->second;
		auto source_type_entry = source_type_entries.find(source);
		if (source_type_entry == source_type_entries.end()) {
			source_type_entry = RelaxedTypeMatch(source_type_entries, source);
			if (source_type_entry == source_type_entries.end()) {
				return nullptr;
			}
		}

		auto &target_type_id_entries = source_type_entry->second;
		auto target_type_id_entry = target_type_id_entries.find(target.id());
		if (target_type_id_entry == target_type_id_entries.end()) {
			target_type_id_entry = target_type_id_entries.find(LogicalTypeId::ANY);
			if (target_type_id_entry == target_type_id_entries.end()) {
				return nullptr;
			}
		}

		auto &target_type_entries = target_type_id_entry->second;
		auto target_type_entry = target_type_entries.find(target);
		if (target_type_entry == target_type_entries.end()) {
			target_type_entry = RelaxedTypeMatch(target_type_entries, target);
			if (target_type_entry == target_type_entries.end()) {
				return nullptr;
			}
		}

		return &target_type_entry->second;
	}

	void AddEntry(const LogicalType &source, const LogicalType &target, MapCastNode node) {
		casts[source.id()][source][target.id()].insert(make_pair(target, std::move(node)));
	}

private:
	type_id_map_t<type_map_t<type_id_map_t<type_map_t<MapCastNode>>>> casts;
};

int64_t CastFunctionSet::ImplicitCastCost(optional_ptr<ClientContext> context, const LogicalType &source,
                                          const LogicalType &target) {
	if (!GovernedImplicitCastCompatible(source, target)) {
		return -1;
	}
	if (target.id() == LogicalTypeId::UNION && source.id() != LogicalTypeId::UNION &&
	    source.id() != LogicalTypeId::UNKNOWN && source.id() != LogicalTypeId::SQLNULL &&
	    TypeVisitor::Contains(target, GovernedLogicalType::IsGoverned)) {
		// CastRules cannot account for governed aliases while recursively scoring
		// UNION members. Score each member through this guarded function so
		// invalid logical retagging is excluded before DuckDB selects a candidate.
		int64_t best_cost = -1;
		for (idx_t target_index = 0; target_index < UnionType::GetMemberCount(target); target_index++) {
			auto member_cost = ImplicitCastCost(context, source, UnionType::GetMemberType(target, target_index));
			if (member_cost >= 0 && (best_cost < 0 || member_cost < best_cost)) {
				best_cost = member_cost;
			}
		}
		return best_cost;
	}
	// check if a cast has been registered
	if (map_info) {
		auto entry = map_info->GetEntry(source, target);
		if (entry) {
			return entry->implicit_cast_cost;
		}
	}
	// if not, fallback to the default implicit cast rules
	auto score = CastRules::ImplicitCast(source, target);
	if (score < 0 && source.id() != LogicalTypeId::BLOB && target.id() == LogicalTypeId::VARCHAR) {
		bool old_implicit_casting = false;
		if (context) {
			old_implicit_casting = Settings::Get<OldImplicitCastingSetting>(*context);
		} else if (config) {
			old_implicit_casting = Settings::Get<OldImplicitCastingSetting>(*config);
		}
		if (old_implicit_casting) {
			// very high cost to avoid choosing this cast if any other option is available
			// (it should be more costly than casting to TEMPLATE if that is available)
			score = 10000000000;
		}
	}
	return score;
}

int64_t CastFunctionSet::ImplicitCastCost(ClientContext &context, const LogicalType &source,
                                          const LogicalType &target) {
	return CastFunctionSet::Get(context).ImplicitCastCost(&context, source, target);
}

int64_t CastFunctionSet::ImplicitCastCost(DatabaseInstance &db, const LogicalType &source, const LogicalType &target) {
	return CastFunctionSet::Get(db).ImplicitCastCost(nullptr, source, target);
}

static BoundCastInfo MapCastFunction(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	D_ASSERT(input.info);
	auto &map_info = input.info->Cast<MapCastInfo>();
	auto entry = map_info.GetEntry(source, target);
	if (entry) {
		if (entry->bind_function) {
			return entry->bind_function(input, source, target);
		}
		return entry->cast_info.Copy();
	}
	return nullptr;
}

void CastFunctionSet::RegisterCastFunction(const LogicalType &source, const LogicalType &target, BoundCastInfo function,
                                           int64_t implicit_cast_cost) {
	RegisterCastFunction(source, target, MapCastNode(std::move(function), implicit_cast_cost));
}

void CastFunctionSet::RegisterCastFunction(const LogicalType &source, const LogicalType &target,
                                           bind_cast_function_t bind_function, int64_t implicit_cast_cost) {
	RegisterCastFunction(source, target, MapCastNode(bind_function, implicit_cast_cost));
}

void CastFunctionSet::RegisterCastFunction(const LogicalType &source, const LogicalType &target, MapCastNode node) {
	if (!map_info) {
		// create the cast map and the cast map function
		auto info = make_uniq<MapCastInfo>();
		map_info = info.get();
		bind_functions.emplace_back(MapCastFunction, std::move(info));
	}
	map_info->AddEntry(source, target, std::move(node));
}

} // namespace duckdb
