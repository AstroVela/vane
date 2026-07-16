# PhysicalOperator 序列化/反序列化功能

## 概述

为 DuckDB 的 `PhysicalOperator` 类及其子类添加了完整的序列化和反序列化功能。此功能允许将物理执行计划保存到磁盘或通过网络传输，用于分布式查询执行、查询计划缓存等场景。

## 实现的文件

### 基类修改

1. **src/include/duckdb/execution/physical_operator.hpp**
   - 添加了序列化相关的头文件引用
   - 添加了虚函数 `Serialize()`
   - 添加了静态方法 `Deserialize()`

2. **src/execution/physical_operator.cpp**
   - 实现了默认的 `Serialize()` 方法（抛出 NotImplementedException）
   - 实现了默认的 `Deserialize()` 方法（抛出 NotImplementedException）

### PhysicalFilter 算子

3. **src/include/duckdb/execution/operator/filter/physical_filter.hpp**
   - 声明了 `Serialize()` 和 `Deserialize()` 方法

4. **src/execution/operator/filter/physical_filter.cpp**
   - ✅ 完整实现了序列化逻辑
   - ✅ 完整实现了反序列化逻辑
   - 序列化内容：type, types, estimated_cardinality, expression

### PhysicalProjection 算子

5. **src/include/duckdb/execution/operator/projection/physical_projection.hpp**
   - 声明了 `Serialize()` 和 `Deserialize()` 方法

6. **src/execution/operator/projection/physical_projection.cpp**
   - ✅ 完整实现了序列化逻辑
   - ✅ 完整实现了反序列化逻辑
   - 序列化内容：type, types, estimated_cardinality, select_list

### PhysicalTableScan 算子

7. **src/include/duckdb/execution/operator/scan/physical_table_scan.hpp**
   - 声明了 `Serialize()` 和 `Deserialize()` 方法

8. **src/execution/operator/scan/physical_table_scan.cpp**
   - ⚠️ 部分实现（标记为未完成）
   - 原因：需要 TableFunction 和 FunctionData 的序列化支持
   - 抛出 NotImplementedException 并说明原因

### 文档和示例

9. **PHYSICAL_OPERATOR_SERIALIZATION.md**
   - 详细的实现文档
   - 设计模式说明
   - 扩展指南

10. **physical_operator_serialization_examples.cpp**
    - 使用示例代码
    - 最佳实践
    - 错误处理示例

11. **test_physical_operator_serialization.cpp**
    - 基础测试框架

## 功能特性

### ✅ 已完成

- [x] PhysicalOperator 基类序列化接口
- [x] PhysicalFilter 完整序列化/反序列化
- [x] PhysicalProjection 完整序列化/反序列化
- [x] 未实现算子的错误处理机制
- [x] 完整文档和示例

### ⚠️ 部分完成

- [~] PhysicalTableScan（接口已声明，实现标记为需要更多支持）

### 📋 待扩展

其他物理算子可以按需添加：
- PhysicalHashAggregate
- PhysicalHashJoin
- PhysicalOrder
- PhysicalLimit
- PhysicalUnion
- 等等...

## 使用方法

### 序列化

```cpp
#include "duckdb/common/serializer/binary_serializer.hpp"

// 创建算子
auto filter_op = make_uniq<PhysicalFilter>(...);

// 序列化
BinarySerializer serializer;
filter_op->Serialize(serializer);
auto data = serializer.GetData();
```

### 反序列化

```cpp
#include "duckdb/common/serializer/binary_deserializer.hpp"

// 反序列化
BinaryDeserializer deserializer(data);
auto filter_op = PhysicalFilter::Deserialize(deserializer, physical_plan);
```

## 设计原则

### 1. 统一接口
所有算子通过基类的虚函数接口实现序列化：
```cpp
virtual void Serialize(Serializer &serializer) const;
```

### 2. 类型安全的反序列化
使用静态工厂方法：
```cpp
static unique_ptr<PhysicalOperator> Deserialize(Deserializer &deserializer, PhysicalPlan &physical_plan);
```

### 3. 明确的错误报告
未实现的算子会抛出清晰的异常信息，而不是静默失败。

### 4. 渐进式实现
- 简单算子优先实现
- 复杂算子可以先标记为未实现
- 保持代码库的一致性

## 序列化字段约定

使用固定的字段 ID：
- **100**: operator type
- **101**: return types
- **102**: estimated_cardinality
- **103+**: 算子特定字段

## 测试

运行测试（示例）：
```bash
# 编译测试文件
g++ -o test_serialization test_physical_operator_serialization.cpp

# 运行测试
./test_serialization
```

## 扩展新算子

### 步骤 1: 头文件声明

在算子的头文件中添加：
```cpp
void Serialize(Serializer &serializer) const override;
static unique_ptr<PhysicalOperator> Deserialize(Deserializer &deserializer, PhysicalPlan &physical_plan);
```

### 步骤 2: 实现序列化

```cpp
void MyOperator::Serialize(Serializer &serializer) const {
    // 序列化基本信息
    serializer.WriteProperty(100, "type", type);
    serializer.WriteProperty(101, "types", types);
    serializer.WriteProperty(102, "estimated_cardinality", estimated_cardinality);
    
    // 序列化算子特定字段
    serializer.WriteProperty(103, "my_field", my_field);
}
```

### 步骤 3: 实现反序列化

```cpp
unique_ptr<PhysicalOperator> MyOperator::Deserialize(Deserializer &deserializer, PhysicalPlan &physical_plan) {
    // 反序列化基本信息
    auto types = deserializer.ReadProperty<vector<LogicalType>>(101, "types");
    auto estimated_cardinality = deserializer.ReadProperty<idx_t>(102, "estimated_cardinality");
    
    // 反序列化算子特定字段
    auto my_field = deserializer.ReadProperty<MyFieldType>(103, "my_field");
    
    // 创建并返回算子
    return make_uniq<MyOperator>(physical_plan, std::move(types), my_field, estimated_cardinality);
}
```

## 注意事项

### 1. 依赖项
确保序列化的对象（如 Expression）也支持序列化。

### 2. PhysicalPlan 引用
反序列化时需要 PhysicalPlan 引用来正确创建算子。

### 3. 子算子处理
当前实现中，子算子的序列化需要手动处理。未来可以在基类中实现通用逻辑。

### 4. 复杂状态
某些算子的状态（如 GlobalSinkState）可能不适合序列化，需要在执行时重新创建。

## 应用场景

1. **分布式查询执行**
   - 将物理计划发送到远程节点
   - 支持跨进程的查询执行

2. **查询计划缓存**
   - 保存编译好的查询计划
   - 加速重复查询的执行

3. **计划分析和调试**
   - 导出查询计划用于分析
   - 在不同环境中重现查询执行

4. **Ray 集成**
   - 支持在 Ray 集群中分发 DuckDB 查询
   - 实现真正的分布式查询处理

## 已知限制

1. **TableScan 复杂性**
   - TableFunction 和 FunctionData 的序列化需要额外支持
   - 需要 catalog context

2. **状态管理**
   - 运行时状态（如 GlobalSinkState）不被序列化
   - 反序列化后需要重新初始化

3. **子算子**
   - 当前需要手动处理子算子的序列化
   - 可以在未来版本中添加通用支持

## 贡献指南

欢迎为更多算子添加序列化支持！请遵循：
1. 使用统一的字段 ID 约定
2. 提供清晰的错误信息
3. 添加单元测试
4. 更新文档

## 许可证

与 DuckDB 项目保持一致。

## 参考资料

- [DuckDB 序列化框架](https://github.com/duckdb/duckdb/tree/main/src/include/duckdb/common/serializer)
- [Expression 序列化实现](https://github.com/duckdb/duckdb/blob/main/src/planner/expression.cpp)
