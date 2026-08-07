// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// PhysicalOperator serialization examples
//===----------------------------------------------------------------------===//

#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"

namespace duckdb {

// Example: serialize and deserialize PhysicalFilter
void ExampleSerializePhysicalFilter() {
    // This is pseudocode; real usage requires a complete DuckDB context.
    
    /* 
    // 1. Create a PhysicalPlan.
    PhysicalPlan physical_plan;
    
    // 2. Create a filter expression, such as column > 10.
    vector<unique_ptr<Expression>> filter_expressions;
    auto left = make_uniq<BoundReferenceExpression>("column", LogicalType::INTEGER, 0);
    auto right = make_uniq<BoundConstantExpression>(Value::INTEGER(10));
    auto comparison = make_uniq<BoundComparisonExpression>(
        ExpressionType::COMPARE_GREATERTHAN,
        std::move(left),
        std::move(right)
    );
    filter_expressions.push_back(std::move(comparison));
    
    // 3. Create the PhysicalFilter operator.
    vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};
    auto filter_op = make_uniq<PhysicalFilter>(
        physical_plan,
        types,
        std::move(filter_expressions),
        1000  // estimated_cardinality
    );
    
    // 4. Serialize it.
    BinarySerializer serializer;
    filter_op->Serialize(serializer);
    auto serialized_data = serializer.GetData();
    
    // 5. Deserialize it.
    BinaryDeserializer deserializer(serialized_data);
    auto deserialized_filter = PhysicalFilter::Deserialize(deserializer, physical_plan);
    
    // 6. Use the deserialized operator like the original one.
    */
}

// Example: serialize and deserialize PhysicalProjection
void ExampleSerializePhysicalProjection() {
    /* 
    // 1. Create a PhysicalPlan.
    PhysicalPlan physical_plan;
    
    // 2. Create the projection expression list.
    vector<unique_ptr<Expression>> select_list;
    
    // Project the first column.
    select_list.push_back(
        make_uniq<BoundReferenceExpression>("col1", LogicalType::INTEGER, 0)
    );
    
    // Project the second column.
    select_list.push_back(
        make_uniq<BoundReferenceExpression>("col2", LogicalType::VARCHAR, 1)
    );
    
    // 3. Create the PhysicalProjection operator.
    vector<LogicalType> output_types = {LogicalType::INTEGER, LogicalType::VARCHAR};
    auto projection_op = make_uniq<PhysicalProjection>(
        physical_plan,
        output_types,
        std::move(select_list),
        1000  // estimated_cardinality
    );
    
    // 4. Serialize it.
    BinarySerializer serializer;
    projection_op->Serialize(serializer);
    auto serialized_data = serializer.GetData();
    
    // 5. Deserialize it.
    BinaryDeserializer deserializer(serialized_data);
    auto deserialized_projection = PhysicalProjection::Deserialize(deserializer, physical_plan);
    
    // 6. Verify the result.
    assert(deserialized_projection->select_list.size() == 2);
    */
}

// Example: try to serialize an unsupported operator
void ExampleUnimplementedOperator() {
    /* 
    // Operators without serialization support throw NotImplementedException.
    
    try {
        // Assume an operator does not implement serialization.
        PhysicalPlan physical_plan;
        // ... create an operator without serialization support ...
        
        BinarySerializer serializer;
        // operator->Serialize(serializer);  // Throws an exception.
        
    } catch (const NotImplementedException &e) {
        // The error resembles: "Serialization not implemented for operator type: XXX".
        std::cout << "Expected error: " << e.what() << std::endl;
    }
    */
}

// Example: serialize an operator tree with child operators
void ExampleSerializeOperatorTree() {
    /* 
    // Trees with child operators require recursive serialization.
    
    PhysicalPlan physical_plan;
    
    // 1. Create a leaf operator such as TableScan.
    // ... create table_scan ...
    
    // 2. Create a Projection operator with TableScan as its child.
    vector<unique_ptr<Expression>> proj_list;
    // ... add projection expressions ...
    auto projection = make_uniq<PhysicalProjection>(
        physical_plan,
        types,
        std::move(proj_list),
        1000
    );
    // projection->children.push_back(table_scan);  // Add the child operator.
    
    // 3. Create a Filter operator with Projection as its child.
    vector<unique_ptr<Expression>> filter_exprs;
    // ... add filter expressions ...
    auto filter = make_uniq<PhysicalFilter>(
        physical_plan,
        types,
        std::move(filter_exprs),
        500
    );
    // filter->children.push_back(projection);  // Add the child operator.
    
    // 4. Serialize the complete operator tree.
    // The current implementation requires child operators to be handled manually.
    // Common child-operator serialization could be added to the base class later.
    */
}

} // namespace duckdb

// Usage summary
/* 

## Basic usage

### Serialization
```cpp
// 1. Create an operator.
auto op = make_uniq<PhysicalFilter>(...);

// 2. Create a serializer.
BinarySerializer serializer;

// 3. Serialize the operator.
op->Serialize(serializer);

// 4. Retrieve the serialized data.
auto data = serializer.GetData();
```

### Deserialization
```cpp
// 1. Create a deserializer.
BinaryDeserializer deserializer(data);

// 2. Deserialize the operator.
auto op = PhysicalFilter::Deserialize(deserializer, physical_plan);

// 3. Use the operator normally.
```

## Supported operators

1. **PhysicalFilter** - fully implemented
   - Serializes filter expressions
   - Supports complex Boolean expressions

2. **PhysicalProjection** - fully implemented
   - Serializes projection expression lists
   - Supports multi-column projections

3. **PhysicalTableScan** - partially implemented
   - Declares the interface
   - Throws NotImplementedException
   - Requires catalog context and TableFunction serialization

## Error handling

Operators without serialization support throw a clear error:
```cpp
throw NotImplementedException(
    "Serialization not implemented for operator type: %s",
    PhysicalOperatorToString(type)
);
```

## Extension guide

To add serialization support to a new operator:

1. Declare the methods in the header:
```cpp
void Serialize(Serializer &serializer) const override;
static unique_ptr<PhysicalOperator> Deserialize(Deserializer &deserializer, PhysicalPlan &physical_plan);
```

2. Define them in the implementation file:
```cpp
void MyOperator::Serialize(Serializer &serializer) const {
    serializer.WriteProperty(100, "type", type);
    serializer.WriteProperty(101, "types", types);
    serializer.WriteProperty(102, "estimated_cardinality", estimated_cardinality);
    // ... serialize operator-specific fields ...
}

unique_ptr<PhysicalOperator> MyOperator::Deserialize(Deserializer &deserializer, PhysicalPlan &physical_plan) {
    auto types = deserializer.ReadProperty<vector<LogicalType>>(101, "types");
    auto estimated_cardinality = deserializer.ReadProperty<idx_t>(102, "estimated_cardinality");
    // ... deserialize operator-specific fields ...
    return make_uniq<MyOperator>(physical_plan, ...);
}
```

*/
