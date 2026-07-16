# PhysicalOperator 序列化/反序列化实现总结

## 概述
为 DuckDB 的 PhysicalOperator 基类及其子类添加了序列化/反序列化功能。

## 实现内容

### 1. PhysicalOperator 基类（抽象接口）

**文件**: `src/include/duckdb/execution/physical_operator.hpp`

添加的方法：
```cpp
// 序列化接口
virtual void Serialize(Serializer &serializer) const;
static unique_ptr<PhysicalOperator> Deserialize(Deserializer &deserializer, PhysicalPlan &physical_plan);
```

**文件**: `src/execution/physical_operator.cpp`

默认实现：
- `Serialize()`: 抛出 `NotImplementedException`，提示具体算子类型未实现序列化
- `Deserialize()`: 抛出 `NotImplementedException`

这确保了未实现序列化的算子会明确报错，而不是静默失败。

### 2. PhysicalFilter 算子

**文件**: 
- `src/include/duckdb/execution/operator/filter/physical_filter.hpp`
- `src/execution/operator/filter/physical_filter.cpp`

**序列化内容**:
- 算子类型 (type)
- 返回类型列表 (types)
- 估计基数 (estimated_cardinality)
- 过滤表达式 (expression)
- 子算子数量 (children_count)

**实现特点**:
- 完整实现了 `Serialize()` 方法
- 完整实现了静态 `Deserialize()` 方法
- 使用 DuckDB 的 Serializer/Deserializer 接口
- 依赖 Expression 的序列化功能（已在 DuckDB 中实现）

### 3. PhysicalProjection 算子

**文件**:
- `src/include/duckdb/execution/operator/projection/physical_projection.hpp`
- `src/execution/operator/projection/physical_projection.cpp`

**序列化内容**:
- 算子类型 (type)
- 返回类型列表 (types)
- 估计基数 (estimated_cardinality)
- 投影表达式列表 (select_list)

**实现特点**:
- 完整实现了 `Serialize()` 方法
- 完整实现了静态 `Deserialize()` 方法
- 正确处理表达式向量的序列化

### 4. PhysicalTableScan 算子

**文件**:
- `src/include/duckdb/execution/operator/scan/physical_table_scan.hpp`
- `src/execution/operator/scan/physical_table_scan.cpp`

**实现特点**:
- 添加了序列化方法声明
- **部分实现**: 由于 TableScan 的复杂性，序列化方法抛出 `NotImplementedException`
- 原因说明：
  - 需要 TableFunction 的序列化支持（依赖目录）
  - 需要 FunctionData 的序列化（特定于函数）
  - 需要 TableFilterSet 的完整序列化
  - 需要 catalog context

这是一个合理的设计决策，因为表扫描的序列化确实更复杂，需要额外的基础设施支持。

## 设计模式

### 1. 统一接口
所有算子都通过 PhysicalOperator 基类的虚函数接口实现序列化：
```cpp
virtual void Serialize(Serializer &serializer) const;
```

### 2. 工厂模式
使用静态方法进行反序列化：
```cpp
static unique_ptr<PhysicalOperator> Deserialize(Deserializer &deserializer, PhysicalPlan &physical_plan);
```

### 3. 渐进式实现
- 简单算子（Filter, Projection）：完整实现
- 复杂算子（TableScan）：报错说明未实现
- 其他算子：继承基类的默认行为（报错）

### 4. 属性ID设计
使用明确的字段ID进行序列化：
- 100: type
- 101: types
- 102: estimated_cardinality
- 103+: 算子特定属性

## 错误处理

所有未实现的算子在序列化时会抛出清晰的错误信息：
```cpp
throw NotImplementedException("Serialization not implemented for operator type: %s", 
                              PhysicalOperatorToString(type));
```

这使得：
1. 开发者能清楚知道哪些算子尚未支持
2. 用户能得到有意义的错误信息
3. 便于后续逐步添加更多算子的支持

## 依赖关系

序列化功能依赖于：
1. `duckdb/common/serializer/serializer.hpp` - 序列化框架
2. `duckdb/common/serializer/deserializer.hpp` - 反序列化框架
3. `duckdb/planner/expression.hpp` - 表达式序列化（已实现）
4. PhysicalPlan 引用 - 用于创建算子

## 使用示例

```cpp
// 序列化
BinarySerializer serializer;
physical_operator->Serialize(serializer);

// 反序列化
BinaryDeserializer deserializer(data);
auto deserialized_op = PhysicalFilter::Deserialize(deserializer, physical_plan);
```

## 未来扩展

可以继续为以下算子添加序列化支持：
- PhysicalHashAggregate
- PhysicalHashJoin
- PhysicalOrder
- PhysicalLimit
- 等等...

遵循相同的模式：
1. 在头文件中声明 Serialize() 和 Deserialize()
2. 在实现文件中添加序列化逻辑
3. 对于复杂算子，先实现基本功能，标记复杂部分为未实现

## 测试

创建了测试文件 `test_physical_operator_serialization.cpp` 用于验证功能。

## 总结

此实现为 DuckDB 的物理算子提供了一个可扩展的序列化框架：
- ✅ 基类抽象接口完成
- ✅ PhysicalFilter 完整实现
- ✅ PhysicalProjection 完整实现
- ⚠️ PhysicalTableScan 部分实现（标记为需要更多支持）
- ✅ 其他算子默认报错提示

这种设计允许系统逐步添加更多算子的序列化支持，同时保持清晰的错误报告机制。
