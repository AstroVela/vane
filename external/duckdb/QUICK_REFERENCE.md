# PhysicalOperator 序列化 - 快速参考

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
| PhysicalFilter | ✅ 完整 | ✅ 完整 | 包含表达式序列化 |
| PhysicalProjection | ✅ 完整 | ✅ 完整 | 包含表达式列表 |
| PhysicalTableScan | ⚠️ 占位 | ⚠️ 占位 | 需要 catalog 支持 |
| 其他算子 | ❌ 未实现 | ❌ 未实现 | 会抛出异常 |

---

## 🔧 添加新算子支持

### 1️⃣ 头文件（my_operator.hpp）
```cpp
class MyOperator : public PhysicalOperator {
public:
    void Serialize(Serializer &serializer) const override;
    static unique_ptr<PhysicalOperator> Deserialize(
        Deserializer &deserializer, 
        PhysicalPlan &physical_plan
    );
};
```

### 2️⃣ 实现文件（my_operator.cpp）
```cpp
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"

void MyOperator::Serialize(Serializer &serializer) const {
    serializer.WriteProperty(100, "type", type);
    serializer.WriteProperty(101, "types", types);
    serializer.WriteProperty(102, "estimated_cardinality", estimated_cardinality);
    serializer.WriteProperty(103, "my_field", my_field);
}

unique_ptr<PhysicalOperator> MyOperator::Deserialize(
    Deserializer &deserializer, 
    PhysicalPlan &physical_plan
) {
    auto types = deserializer.ReadProperty<vector<LogicalType>>(101, "types");
    auto cardinality = deserializer.ReadProperty<idx_t>(102, "estimated_cardinality");
    auto my_field = deserializer.ReadProperty<MyType>(103, "my_field");
    
    return make_uniq<MyOperator>(physical_plan, types, my_field, cardinality);
}
```

---

## 📊 字段 ID 约定

| ID | 字段名 | 类型 | 说明 |
|----|--------|------|------|
| 100 | type | PhysicalOperatorType | 算子类型 |
| 101 | types | vector<LogicalType> | 返回类型 |
| 102 | estimated_cardinality | idx_t | 估计基数 |
| 103+ | - | varies | 算子特定字段 |

---

## 🚨 错误处理

### 未实现的算子
```cpp
try {
    unknown_operator->Serialize(serializer);
} catch (const NotImplementedException &e) {
    // 错误: "Serialization not implemented for operator type: XXX"
}
```

### PhysicalTableScan
```cpp
try {
    table_scan->Serialize(serializer);
} catch (const NotImplementedException &e) {
    // 错误: "PhysicalTableScan serialization requires catalog context..."
}
```

---

## 📁 关键文件位置

### 核心实现
```
src/include/duckdb/execution/physical_operator.hpp
src/execution/physical_operator.cpp

src/include/duckdb/execution/operator/filter/physical_filter.hpp
src/execution/operator/filter/physical_filter.cpp

src/include/duckdb/execution/operator/projection/physical_projection.hpp
src/execution/operator/projection/physical_projection.cpp

src/include/duckdb/execution/operator/scan/physical_table_scan.hpp
src/execution/operator/scan/physical_table_scan.cpp
```

### 文档
```
PHYSICAL_OPERATOR_SERIALIZATION.md           # 实现总结
PHYSICAL_OPERATOR_SERIALIZATION_README.md    # 完整指南
IMPLEMENTATION_SUMMARY.md                     # 修改清单
physical_operator_serialization_examples.cpp # 代码示例
```

---

## 🎓 示例代码

### PhysicalFilter
```cpp
// 创建过滤表达式
vector<unique_ptr<Expression>> exprs;
exprs.push_back(make_uniq<BoundComparisonExpression>(...));

// 创建算子
auto filter = make_uniq<PhysicalFilter>(
    physical_plan,
    types,
    std::move(exprs),
    1000
);

// 序列化
BinarySerializer serializer;
filter->Serialize(serializer);

// 反序列化
BinaryDeserializer deserializer(serializer.GetData());
auto restored = PhysicalFilter::Deserialize(deserializer, physical_plan);
```

### PhysicalProjection
```cpp
// 创建投影表达式
vector<unique_ptr<Expression>> select_list;
select_list.push_back(make_uniq<BoundReferenceExpression>(...));

// 创建算子
auto projection = make_uniq<PhysicalProjection>(
    physical_plan,
    types,
    std::move(select_list),
    1000
);

// 序列化
BinarySerializer serializer;
projection->Serialize(serializer);

// 反序列化
BinaryDeserializer deserializer(serializer.GetData());
auto restored = PhysicalProjection::Deserialize(deserializer, physical_plan);
```

---

## 💡 最佳实践

### ✅ DO
- 使用统一的字段 ID (100-104+)
- 序列化所有必要的成员变量
- 提供清晰的错误信息
- 添加单元测试
- 更新文档

### ❌ DON'T
- 不要序列化运行时状态（如 GlobalSinkState）
- 不要在反序列化时依赖全局状态
- 不要假设序列化顺序
- 不要忘记处理子算子

---

## 🔍 调试技巧

### 打印序列化数据
```cpp
BinarySerializer serializer;
op->Serialize(serializer);
auto data = serializer.GetData();
std::cout << "Serialized size: " << data.size() << " bytes" << std::endl;
```

### 验证反序列化
```cpp
auto restored = MyOperator::Deserialize(deserializer, physical_plan);
assert(restored->type == original->type);
assert(restored->types == original->types);
```

---

## 📞 需要帮助？

查看完整文档：
- `PHYSICAL_OPERATOR_SERIALIZATION_README.md` - 详细使用指南
- `IMPLEMENTATION_SUMMARY.md` - 实现细节
- `physical_operator_serialization_examples.cpp` - 示例代码

---

## 🚀 下一步

1. 为常用算子添加序列化支持
2. 实现 TableScan 的完整序列化
3. 添加性能测试
4. 集成到分布式查询系统
