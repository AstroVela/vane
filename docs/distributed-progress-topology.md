# 分布式 Fragment/Pipeline 进度拓扑长期方案

## 状态与范围

本文档定义 Vane 分布式执行进度系统的最终生产契约。实现不保留旧快照形状、兼容字段、缺失拓扑 fallback 或 remote worker 二次拓扑发现路径。

范围包括：

- Fragment/Pipeline 拓扑的产生、注册、生命周期和渲染；
- FTE descriptor 尚未提交时的 `PENDING` 语义；
- native PipelineTask 的 `QUEUED/RUN/DONE` 语义；
- pipeline boundary 算子的 `(source)/(sink)` 标识；
- Ray actor 初始化与 native plan 启动的显式屏障。

不改变查询结果、算子算法、数据输出语义或 FTE admission 策略。

## 问题

旧实现存在四类结构性问题：

1. Fragment 行混合逻辑 descriptor partition 和 native PipelineTask，导致父子 Q/R/D 无法相加。
2. 拓扑依赖第一个 task 开始执行后产生的动态统计，actor/model 初始化期间 UI 只有时间行。
3. coordinator 和 remote worker 都会构建同一 Fragment 拓扑，并通过异步 callback 再次发布；错误无法可靠传播，且会重复 clone、连接和 native planning。
4. 拓扑与动态计数以渲染后的名称关联。UDF 名称或 `(source)/(sink)` 展示变化会静默丢失计数。

## 核心不变量

### 单一拓扑所有者

- 每个 `(query_id, fragment_id)` 只允许 coordinator 构建一次拓扑。
- coordinator 在提交任何该 Fragment 的 remote task 之前完成拓扑注册。
- remote worker 只注册执行模板并返回 ACK，不构建、不返回、不校验进度拓扑。
- 同一 Fragment 的并发注册通过 registry 中的 build ownership 串行化；只有 owner 执行 clone 和 native planning。

### 拓扑与计数分离

拓扑是不可变结构，不包含任何 rows、bytes 或 task counters：

```text
PipelineTopology {
  schema: "pipeline_topology"
  pipelines: [
    {
      pipeline_id: positive integer
      operators: non-empty list[string]
      operator_details: list[map], length == operators.length
      stage_ids: list[integer]
    }
  ]
}
```

动态 task stats 使用相同的 `pipeline_id`，并携带 operators 作为结构断言。合并规则为：

1. `pipeline_id` 必须存在于已注册拓扑；
2. operators 必须逐项相等；
3. 任何未知、重复或结构不一致都立即报错；
4. 不按展示名称匹配，不过滤未知 pipeline，不降级到无拓扑渲染。

### 两层状态不混合

Fragment 行只显示：

```text
Fragment N [PENDING x]
```

其中 `PENDING` 是满足以下条件的逻辑 descriptor partition 数：

- 已 ready 或因 execution-width admission 被 deferred；
- 尚无 running attempt；
- 未完成、未失败。

Pipeline 行只显示 DuckDB native PipelineTask 的 `QUEUED/RUN/DONE`。两套计数不相加，也不在 Fragment 行重复聚合。

`PENDING` 的判定只存在于 `FteFragmentExecution`，registry 和 renderer 不复制状态条件。

### Pipeline boundary 角色

同时实现 `IsSink()` 和 `IsSource()` 的物理算子是 pipeline boundary。DuckDB native topology 根据算子地址与当前 pipeline 的 source/sink 指针自动产生：

```text
pipeline_role = "source" | "sink"
```

不维护算子类型白名单。典型展示为：

```text
ResNetModel(source)->Projection->CopySink(sink)
CopySink(source)->RESULT_COLLECTOR(sink)
RESULT_COLLECTOR(source)
```

同一个 `RESULT_COLLECTOR` 出现在两条 pipeline 中表示其 sink/source 两侧，不代表创建或执行了两个 collector。

terminal remote `EXCHANGE_SINK` 是例外：其 source 接口只用于 DuckDB completion，不产生用户数据。该计划不附加 `RESULT_COLLECTOR`，并过滤 singleton completion-only `EXCHANGE_SINK` pipeline。

## 组件设计

### Native topology API

`duckdb.ray_cxx.describe_native_progress(conn, plan)` 是无 WorkerManager 的 module-level planning API：

- 对 deferred clone 物化 root；
- 使用 `Executor::InitializeProgressTopology` 构建真实 MetaPipeline；
- 不 schedule event，不执行 operator，不创建 Ray worker manager；
- 返回纯 `PipelineTopology`；
- 不清理 query-owned Python replay state，生命周期清理由 query teardown 负责。

`DistributedPhysicalPlanRunner` 不再暴露 topology method，也不为 topology planning 被实例化。

### Coordinator registry

registry 保存：

- immutable topology map；
- topology build ownership set；
- condition variable，用于并发 builder、COPY startup barrier 和 query teardown。

发布顺序：

```text
claim topology build
  -> clone fragment plan
  -> native topology planning
  -> validate exact schema
  -> atomically publish topology
  -> notify waiters
  -> submit remote fragment registration
```

构建失败会释放 ownership 并唤醒 waiter；不会写入部分拓扑。query closing 会唤醒所有 waiter，后续发布失败。

### Native/Python backend contract

Ray backend 必须在 Fragment 创建前拥有非空 native topology。

Python native backend也显式提供 `progress_topology`。如果 backend 尚未收到包含 pipeline stats 的 callback，使用合法的空 topology；第一次收到结构后将其冻结。之后结构变化立即报错。这是显式 backend contract，不是 renderer fallback。

### Renderer

renderer 输入必须包含 `progress_topology`。它以 topology 顺序建立零计数 pipeline，再按 `pipeline_id` 叠加当前 attempt/selected attempt 的动态统计。

renderer 只负责：

- operator label 和角色格式化；
- 跨 partition 聚合同一 pipeline 的 counters；
- rows/bytes/rate 格式化。

renderer 不推断 topology，不合成未知 pipeline，不接受旧 `pipeline_index` 字段。

## Actor/native 启动屏障

actor handle 的不可变计划字段称为 `actor_dispatch_indices`，表示在 QRM admission 打开后允许调度的 actor，不表示注入计划时 actor 已初始化。

Ray actor stage 初始为 `actor_ready=false`。COPY 启动使用两个独立信号：

1. 至少一个 Fragment topology 已由 native plan path 发布；
2. 所有 actor `init_payload` refs 成功完成。

只有两个信号都满足后，coordinator 才将 QRM actor stage 设置为 ready。plan execution、topology barrier、actor initialization 任一提前失败都会立即触发统一 teardown；不使用 `asyncio.sleep(0)` 作为启动握手。

streaming `run_plan` 的 native runner 返回即表示 fragment registration 已进入稳定执行生命周期，随后等待 actor refs 并打开 QRM stage。

## 错误策略

以下情况都是查询错误，不做 fallback：

- topology schema/field/identity 非法；
- 同一 Fragment topology 发生变化；
- live stats 出现未知 pipeline；
- live operators 与 topology 不一致；
- query closing 时仍尝试发布 topology；
- COPY native plan 在 topology barrier 前异常结束；
- topology 初始化超时；
- actor init 失败或超时。

进度 callback 自身的 Python 渲染异常可以停止 UI 刷新，但结构契约错误必须在 registry/build 路径同步传播，不能藏在 future callback 中。

## 删除项

实现完成后删除：

- remote worker `_describe_native_fragment_topology`；
- fragment registration 返回值中的 `fragment_topologies`；
- registration future 上的 topology callback；
- per-worker-handle topology runner、lock 和 key cache；
- `DistributedPhysicalPlanRunner.describe_native_progress`；
- topology task-stats 归零逻辑（C++ 和 Python）；
- `pipeline_index` 字段和基于 display name 的 identity；
- 缺失 topology 时直接使用 live stats 的 renderer fallback；
- `actor_ready_indices` 旧字段；
- COPY 启动中的 `asyncio.sleep(0)` 握手。

## 验证

提交前必须通过：

- DuckDB topology-only executor 单测；
- ResultCollector/CopySink/UDF/Repartition/ExchangeSink role 与 topology 契约测试；
- coordinator topology exactly-once 并发测试；
- remote registration 不构建 topology 的测试；
- unknown/mismatched pipeline fail-fast 测试；
- PENDING 与 execution-width/deferred/running/finished 状态测试；
- COPY plan failure、actor-init failure、topology-timeout 的 barrier/teardown 测试；
- C++ 增量编译及 Ray/FTE 相关回归测试。
