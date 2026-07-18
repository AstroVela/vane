// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// PhysicalOperator 序列化使用示例
//===----------------------------------------------------------------------===//

#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"

namespace duckdb {

// 示例：序列化和反序列化 PhysicalFilter
void ExampleSerializePhysicalFilter() {
    // 注意：这是伪代码示例，实际使用需要完整的 DuckDB 上下文
    
    /* 
    // 1. 创建 PhysicalPlan
    PhysicalPlan physical_plan;
    
    // 2. 创建过滤表达式 (例如: column > 10)
    vector<unique_ptr<Expression>> filter_expressions;
    auto left = make_uniq<BoundReferenceExpression>("column", LogicalType::INTEGER, 0);
    auto right = make_uniq<BoundConstantExpression>(Value::INTEGER(10));
    auto comparison = make_uniq<BoundComparisonExpression>(
        ExpressionType::COMPARE_GREATERTHAN,
        std::move(left),
        std::move(right)
    );
    filter_expressions.push_back(std::move(comparison));
    
    // 3. 创建 PhysicalFilter 算子
    vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};
    auto filter_op = make_uniq<PhysicalFilter>(
        physical_plan,
        types,
        std::move(filter_expressions),
        1000  // estimated_cardinality
    );
    
    // 4. 序列化
    BinarySerializer serializer;
    filter_op->Serialize(serializer);
    auto serialized_data = serializer.GetData();
    
    // 5. 反序列化
    BinaryDeserializer deserializer(serialized_data);
    auto deserialized_filter = PhysicalFilter::Deserialize(deserializer, physical_plan);
    
    // 6. 使用反序列化的算子
    // deserialized_filter 现在可以像原始算子一样使用
    */
}

// 示例：序列化和反序列化 PhysicalProjection
void ExampleSerializePhysicalProjection() {
    /* 
    // 1. 创建 PhysicalPlan
    PhysicalPlan physical_plan;
    
    // 2. 创建投影表达式列表
    vector<unique_ptr<Expression>> select_list;
    
    // 投影第一列
    select_list.push_back(
        make_uniq<BoundReferenceExpression>("col1", LogicalType::INTEGER, 0)
    );
    
    // 投影第二列
    select_list.push_back(
        make_uniq<BoundReferenceExpression>("col2", LogicalType::VARCHAR, 1)
    );
    
    // 3. 创建 PhysicalProjection 算子
    vector<LogicalType> output_types = {LogicalType::INTEGER, LogicalType::VARCHAR};
    auto projection_op = make_uniq<PhysicalProjection>(
        physical_plan,
        output_types,
        std::move(select_list),
        1000  // estimated_cardinality
    );
    
    // 4. 序列化
    BinarySerializer serializer;
    projection_op->Serialize(serializer);
    auto serialized_data = serializer.GetData();
    
    // 5. 反序列化
    BinaryDeserializer deserializer(serialized_data);
    auto deserialized_projection = PhysicalProjection::Deserialize(deserializer, physical_plan);
    
    // 6. 验证
    assert(deserialized_projection->select_list.size() == 2);
    */
}

// 示例：尝试序列化未实现的算子
void ExampleUnimplementedOperator() {
    /* 
    // 对于未实现序列化的算子，会抛出 NotImplementedException
    
    try {
        // 假设某个算子没有实现序列化
        PhysicalPlan physical_plan;
        // ... 创建某个未实现序列化的算子 ...
        
        BinarySerializer serializer;
        // operator->Serialize(serializer);  // 会抛出异常
        
    } catch (const NotImplementedException &e) {
        // 错误信息类似: "Serialization not implemented for operator type: XXX"
        std::cout << "Expected error: " << e.what() << std::endl;
    }
    */
}

// 示例：序列化包含子算子的算子树
void ExampleSerializeOperatorTree() {
    /* 
    // 对于包含子算子的情况，需要递归序列化
    
    PhysicalPlan physical_plan;
    
    // 1. 创建叶子算子（如 TableScan）
    // ... 创建 table_scan ...
    
    // 2. 创建 Projection 算子，以 TableScan 为子算子
    vector<unique_ptr<Expression>> proj_list;
    // ... 添加投影表达式 ...
    auto projection = make_uniq<PhysicalProjection>(
        physical_plan,
        types,
        std::move(proj_list),
        1000
    );
    // projection->children.push_back(table_scan);  // 添加子算子
    
    // 3. 创建 Filter 算子，以 Projection 为子算子
    vector<unique_ptr<Expression>> filter_exprs;
    // ... 添加过滤表达式 ...
    auto filter = make_uniq<PhysicalFilter>(
        physical_plan,
        types,
        std::move(filter_exprs),
        500
    );
    // filter->children.push_back(projection);  // 添加子算子
    
    // 4. 序列化整个算子树
    // 注意：当前实现需要手动处理子算子的序列化
    // 未来可以在基类中实现通用的子算子序列化逻辑
    */
}

} // namespace duckdb

// 主要用法总结
/* 

## 基本用法

### 序列化
```cpp
// 1. 创建算子
auto op = make_uniq<PhysicalFilter>(...);

// 2. 创建序列化器
BinarySerializer serializer;

// 3. 序列化
op->Serialize(serializer);

// 4. 获取序列化数据
auto data = serializer.GetData();
```

### 反序列化
```cpp
// 1. 创建反序列化器
BinaryDeserializer deserializer(data);

// 2. 反序列化
auto op = PhysicalFilter::Deserialize(deserializer, physical_plan);

// 3. 使用算子
// op 现在可以正常使用了
```

## 已实现的算子

1. **PhysicalFilter** - 完整实现
   - 序列化过滤表达式
   - 支持复杂的布尔表达式

2. **PhysicalProjection** - 完整实现
   - 序列化投影表达式列表
   - 支持多列投影

3. **PhysicalTableScan** - 部分实现
   - 声明了接口
   - 实现会抛出 NotImplementedException
   - 原因：需要 catalog context 和 TableFunction 序列化

## 错误处理

未实现序列化的算子会抛出清晰的错误：
```cpp
throw NotImplementedException(
    "Serialization not implemented for operator type: %s",
    PhysicalOperatorToString(type)
);
```

## 扩展指南

为新算子添加序列化支持：

1. 在头文件中声明：
```cpp
void Serialize(Serializer &serializer) const override;
static unique_ptr<PhysicalOperator> Deserialize(Deserializer &deserializer, PhysicalPlan &physical_plan);
```

2. 在实现文件中：
```cpp
void MyOperator::Serialize(Serializer &serializer) const {
    serializer.WriteProperty(100, "type", type);
    serializer.WriteProperty(101, "types", types);
    serializer.WriteProperty(102, "estimated_cardinality", estimated_cardinality);
    // ... 序列化算子特定字段 ...
}

unique_ptr<PhysicalOperator> MyOperator::Deserialize(Deserializer &deserializer, PhysicalPlan &physical_plan) {
    auto types = deserializer.ReadProperty<vector<LogicalType>>(101, "types");
    auto estimated_cardinality = deserializer.ReadProperty<idx_t>(102, "estimated_cardinality");
    // ... 反序列化算子特定字段 ...
    return make_uniq<MyOperator>(physical_plan, ...);
}
```

*/
