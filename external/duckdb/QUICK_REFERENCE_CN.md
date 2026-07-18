# PhysicalOperator 序列化 - 快速参考（中文版）

## 🎯 快速开始

### 序列化一个算子
```cpp
#include "duckdb/common/serializer/binary_serializer.hpp"

BinarySerializer serializer;
my_operator->Serialize(serializer);
auto data = serializer.GetData();
```

### 反序列化一个算子
```cpp
#include "duckdb/common/serializer/binary_deserializer.hpp"

BinaryDeserializer deserializer(data);
auto op = PhysicalFilter::Deserialize(deserializer, physical_plan);
```

---

## 📋 已实现的算子

| 算子 | 序列化状态 | 反序列化状态 | 备注 |
|------|-----------|-------------|------|
| PhysicalFilter | ✅ 完整实现 | ✅ 完整实现 | 支持表达式序列化 |
| PhysicalProjection | ✅ 完整实现 | ✅ 完整实现 | 支持表达式列表 |
| PhysicalTableScan | ⚠️ 占位实现 | ⚠️ 占位实现 | 需要目录上下文支持 |
| 其他算子 | ❌ 未实现 | ❌ 未实现 | 会抛出异常说明 |

---

## 🔧 为新算子添加序列化支持

### 步骤 1: 在头文件中声明（my_operator.hpp）
```cpp
class MyOperator : public PhysicalOperator {
public:
    // 序列化方法
    void Serialize(Serializer &serializer) const override;
    
    // 反序列化静态方法
    static unique_ptr<PhysicalOperator> Deserialize(
        Deserializer &deserializer, 
        PhysicalPlan &physical_plan
    );
};
```

### 步骤 2: 在实现文件中实现（my_operator.cpp）
```cpp
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"

void MyOperator::Serialize(Serializer &serializer) const {
    // 序列化基本字段
    serializer.WriteProperty(100, "type", type);
    serializer.WriteProperty(101, "types", types);
    serializer.WriteProperty(102, "estimated_cardinality", estimated_cardinality);
    
    // 序列化算子特定字段
    serializer.WriteProperty(103, "my_field", my_field);
}

unique_ptr<PhysicalOperator> MyOperator::Deserialize(
    Deserializer &deserializer, 
    PhysicalPlan &physical_plan
) {
    // 反序列化基本字段
    auto types = deserializer.ReadProperty<vector<LogicalType>>(101, "types");
    auto cardinality = deserializer.ReadProperty<idx_t>(102, "estimated_cardinality");
    
    // 反序列化算子特定字段
    auto my_field = deserializer.ReadProperty<MyType>(103, "my_field");
    
    // 创建并返回算子实例
    return make_uniq<MyOperator>(physical_plan, types, my_field, cardinality);
}
```

---

## 📊 字段 ID 使用约定

| ID 范围 | 用途 | 类型 | 说明 |
|---------|------|------|------|
| 100 | type | PhysicalOperatorType | 算子类型标识 |
| 101 | types | vector<LogicalType> | 算子返回类型列表 |
| 102 | estimated_cardinality | idx_t | 估计的结果集大小 |
| 103+ | 自定义字段 | 变化 | 各算子的特定字段 |

**注意**: 所有算子都应该使用相同的字段 ID (100-102) 来序列化基本信息。

---

## 🚨 错误处理机制

### 未实现序列化的算子
```cpp
try {
    unknown_operator->Serialize(serializer);
} catch (const NotImplementedException &e) {
    // 会收到错误信息: "Serialization not implemented for operator type: XXX"
    std::cerr << "序列化失败: " << e.what() << std::endl;
}
```

### PhysicalTableScan 特殊情况
```cpp
try {
    table_scan->Serialize(serializer);
} catch (const NotImplementedException &e) {
    // 会收到错误信息: "PhysicalTableScan serialization requires catalog context..."
    std::cerr << "表扫描算子需要目录上下文支持" << std::endl;
}
```

---

## 📁 关键文件位置

### 核心实现文件
```
# 基类
src/include/duckdb/execution/physical_operator.hpp
src/execution/physical_operator.cpp

# PhysicalFilter 算子
src/include/duckdb/execution/operator/filter/physical_filter.hpp
src/execution/operator/filter/physical_filter.cpp

# PhysicalProjection 算子
src/include/duckdb/execution/operator/projection/physical_projection.hpp
src/execution/operator/projection/physical_projection.cpp

# PhysicalTableScan 算子
src/include/duckdb/execution/operator/scan/physical_table_scan.hpp
src/execution/operator/scan/physical_table_scan.cpp
```

### 文档和示例文件
```
PHYSICAL_OPERATOR_SERIALIZATION.md           # 详细的实现文档
PHYSICAL_OPERATOR_SERIALIZATION_README.md    # 完整使用指南
IMPLEMENTATION_SUMMARY.md                     # 文件修改清单
QUICK_REFERENCE.md                           # 英文快速参考
QUICK_REFERENCE_CN.md                        # 中文快速参考（本文件）
physical_operator_serialization_examples.cpp # 代码使用示例
test_physical_operator_serialization.cpp     # 测试框架
```

---

## 🎓 完整使用示例

### PhysicalFilter 示例
```cpp
// 第一步: 创建过滤表达式（例如: column > 10）
vector<unique_ptr<Expression>> filter_expressions;
auto comparison = make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN,
    make_uniq<BoundReferenceExpression>("column", LogicalType::INTEGER, 0),
    make_uniq<BoundConstantExpression>(Value::INTEGER(10))
);
filter_expressions.push_back(std::move(comparison));

// 第二步: 创建 PhysicalFilter 算子
vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};
auto filter = make_uniq<PhysicalFilter>(
    physical_plan,
    types,
    std::move(filter_expressions),
    1000  // 估计基数
);

// 第三步: 序列化
BinarySerializer serializer;
filter->Serialize(serializer);
auto serialized_data = serializer.GetData();

// 第四步: 反序列化
BinaryDeserializer deserializer(serialized_data);
auto restored_filter = PhysicalFilter::Deserialize(deserializer, physical_plan);

// 第五步: 验证
assert(restored_filter->type == PhysicalOperatorType::FILTER);
assert(restored_filter->types.size() == 2);
```

### PhysicalProjection 示例
```cpp
// 第一步: 创建投影表达式列表
vector<unique_ptr<Expression>> select_list;

// 投影第一列
select_list.push_back(
    make_uniq<BoundReferenceExpression>("col1", LogicalType::INTEGER, 0)
);

// 投影第二列
select_list.push_back(
    make_uniq<BoundReferenceExpression>("col2", LogicalType::VARCHAR, 1)
);

// 第二步: 创建 PhysicalProjection 算子
vector<LogicalType> output_types = {LogicalType::INTEGER, LogicalType::VARCHAR};
auto projection = make_uniq<PhysicalProjection>(
    physical_plan,
    output_types,
    std::move(select_list),
    1000  // 估计基数
);

// 第三步: 序列化
BinarySerializer serializer;
projection->Serialize(serializer);
auto serialized_data = serializer.GetData();

// 第四步: 反序列化
BinaryDeserializer deserializer(serialized_data);
auto restored_projection = PhysicalProjection::Deserialize(deserializer, physical_plan);

// 第五步: 验证
assert(restored_projection->select_list.size() == 2);
```

---

## 💡 最佳实践

### ✅ 应该做的事情
- ✓ 使用统一的字段 ID 约定（100-102 为基本字段）
- ✓ 序列化所有必要的成员变量
- ✓ 为未实现的功能提供清晰的错误信息
- ✓ 添加单元测试验证序列化的正确性
- ✓ 及时更新相关文档

### ❌ 不应该做的事情
- ✗ 不要序列化运行时状态（如 GlobalSinkState、OperatorState）
- ✗ 不要在反序列化时依赖全局状态或外部上下文
- ✗ 不要假设序列化字段的顺序
- ✗ 不要忘记处理子算子的序列化
- ✗ 不要使用魔术数字，使用命名常量

---

## 🔍 调试和测试技巧

### 打印序列化数据大小
```cpp
BinarySerializer serializer;
op->Serialize(serializer);
auto data = serializer.GetData();
std::cout << "序列化数据大小: " << data.size() << " 字节" << std::endl;
```

### 验证反序列化结果
```cpp
// 序列化
BinarySerializer serializer;
original_op->Serialize(serializer);

// 反序列化
BinaryDeserializer deserializer(serializer.GetData());
auto restored_op = MyOperator::Deserialize(deserializer, physical_plan);

// 验证基本属性
assert(restored_op->type == original_op->type);
assert(restored_op->types == original_op->types);
assert(restored_op->estimated_cardinality == original_op->estimated_cardinality);
```

### 测试错误处理
```cpp
// 测试未实现的算子
try {
    unimplemented_op->Serialize(serializer);
    assert(false);  // 不应该执行到这里
} catch (const NotImplementedException &e) {
    std::cout << "正确捕获异常: " << e.what() << std::endl;
}
```

---

## 🏗️ 架构设计说明

### 设计原则

1. **统一接口**: 通过基类虚函数提供统一的序列化接口
2. **工厂模式**: 使用静态方法进行类型安全的反序列化
3. **渐进实现**: 优先实现简单算子，复杂算子可以延后
4. **明确错误**: 未实现的功能会抛出清晰的异常而非静默失败
5. **可扩展性**: 易于为新算子添加序列化支持

### 类图结构
```
PhysicalOperator (基类)
    ├── Serialize() [虚函数 - 默认抛出异常]
    ├── Deserialize() [静态方法 - 默认抛出异常]
    │
    ├── PhysicalFilter (子类)
    │   ├── Serialize() [重写 - 完整实现]
    │   └── Deserialize() [静态 - 完整实现]
    │
    ├── PhysicalProjection (子类)
    │   ├── Serialize() [重写 - 完整实现]
    │   └── Deserialize() [静态 - 完整实现]
    │
    └── PhysicalTableScan (子类)
        ├── Serialize() [重写 - 抛出 NotImplementedException]
        └── Deserialize() [静态 - 抛出 NotImplementedException]
```

---

## 📞 获取帮助

### 查看详细文档
- `PHYSICAL_OPERATOR_SERIALIZATION.md` - 实现总结文档
- `PHYSICAL_OPERATOR_SERIALIZATION_README.md` - 完整使用指南
- `IMPLEMENTATION_SUMMARY.md` - 文件修改清单

### 查看示例代码
- `physical_operator_serialization_examples.cpp` - 详细的代码示例
- `test_physical_operator_serialization.cpp` - 测试代码框架

---

## 🚀 下一步计划

### 短期目标
1. 为更多常用算子添加序列化支持（HashJoin、HashAggregate 等）
2. 实现子算子的自动序列化机制
3. 添加完整的单元测试覆盖

### 中期目标
1. 完善 PhysicalTableScan 的序列化实现
2. 支持 TableFunction 和 FunctionData 的序列化
3. 优化序列化性能和数据大小

### 长期目标
1. 集成到分布式查询执行系统
2. 支持查询计划的持久化缓存
3. 实现跨版本的序列化兼容性

---

## 📈 性能考虑

### 序列化性能
- 当前实现使用 DuckDB 的 BinarySerializer，性能较好
- 对于大型查询计划，建议进行性能测试
- 可以考虑使用压缩来减小数据大小

### 内存使用
- 序列化过程会创建额外的内存缓冲区
- 对于非常大的表达式树，需要注意内存峰值
- 建议分批处理大型算子树

---

## ⚡ 应用场景

### 1. 分布式查询执行
```cpp
// 在主节点序列化查询计划
BinarySerializer serializer;
query_plan->Serialize(serializer);
auto plan_data = serializer.GetData();

// 发送到工作节点
send_to_worker(plan_data);

// 在工作节点反序列化
BinaryDeserializer deserializer(received_data);
auto local_plan = PhysicalOperator::Deserialize(deserializer, physical_plan);
```

### 2. 查询计划缓存
```cpp
// 缓存编译好的查询计划
string query_hash = compute_hash(sql_query);
if (plan_cache.contains(query_hash)) {
    BinaryDeserializer deserializer(plan_cache[query_hash]);
    return PhysicalOperator::Deserialize(deserializer, physical_plan);
}
```

### 3. Ray 集成示例
```cpp
// 在 Ray 任务中执行 DuckDB 查询
@ray.remote
def execute_duckdb_query(serialized_plan):
    deserializer = BinaryDeserializer(serialized_plan)
    plan = PhysicalOperator::Deserialize(deserializer, physical_plan)
    return execute(plan)
```

---

## 🎯 总结

本实现为 DuckDB 的物理算子提供了一个完整、可扩展的序列化框架：

- ✅ **基础框架完成**: PhysicalOperator 基类提供统一接口
- ✅ **核心算子实现**: Filter 和 Projection 完整实现
- ⚠️ **复杂算子标记**: TableScan 等复杂算子标记为需要进一步支持
- ✅ **清晰的错误处理**: 未实现的功能会明确报错
- ✅ **完善的文档**: 提供多层次的文档和示例

该实现为后续的分布式查询、计划缓存等高级功能奠定了基础！
