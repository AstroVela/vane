# Vane 统一执行架构设计

本文定义同时支持 pipelined 与 FTE 的新执行架构，面向 Vane 执行引擎维护者。设计从两种执行方式的语义出发，允许直接替换已有 API、计划协议、调度器和结果接口，不承担已有代码的兼容责任。

local 直接使用 DuckDB 原生执行，不选择 pipelined 或 FTE。ray 的两种分布式策略共享一套 FragmentGraph 和 TaskRuntime，分别采用直接 exchange 或物化 exchange。两条执行路径共享查询身份、资源、取消和结果所有权契约。新的 Ray FTE 也在新架构中实现，不通过包装旧 PlanRunner 获得。

| 项目 | 基线 |
| --- | --- |
| 状态 | 目标设计；实现与验收进度见实施 roadmap |
| 日期 | 2026 年 10 月 7 日（P5.1 执行入口切换） |
| 开发分支 | feat/analytical-execution |
| 基础分支 | integration/pipelined-execution |
| Vane 参考提交 | a69e60ca43d9（PR #963 合入） |
| Trino 参考提交 | [6ead7e6c2f04c0bcfe5caf8e938dc1f5d3344f31][trino-revision]，调研时的 master，提交时间为 2026 年 10 月 2 日 02:30:56 UTC |
| 兼容策略 | 不保留旧 API、旧协议、旧默认行为或旧执行入口 |
| 实施记录 | [PIPELINED_EXECUTION_ROADMAP.md](PIPELINED_EXECUTION_ROADMAP.md) |

未注明已实现的接口、目录和实施阶段描述目标状态。既有源码提供算法、实现经验和问题证据；它的类名、继承关系、序列化格式及模块位置均不约束新设计。构建和测试流程遵循 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 设计决策

1. 查询选择 local 或 ray 后端。local 直接原生执行；只有 ray 提供 pipelined 或 FTE 策略选择。
2. Ray 的两种模式消费同一套可执行 FragmentGraph，运行同一种 TaskRuntime。local 不经过分布式计划图、调度器或 TaskService；两条路径返回同一种 QueryResult。
3. PipelinedScheduler 管理同时推进的任务和直接通道；RecoveryScheduler 管理物化边、attempt 提交与重试。二者独立实现调度状态机。
4. ExchangeReader 和 ExchangeWriter 统一数据操作；DirectExchange 与 MaterializedExchange 分别实现输出可见性和生命周期。
5. Ray 负责 worker 放置和控制 RPC。worker 间数据与分布式根结果使用 C++ 数据通道，结果交付不依赖 Ray ObjectRef。
6. ray 默认 pipelined，FTE 显式选择。local 不带分布式策略字段，显式传入任意分布式策略均报错。模式隐含恢复策略，不暴露任意组合的 retry_policy。
7. 删除被替代的执行路径。不开设 legacy 模式，不保留旧入口别名，不实现新旧协议转换，也不静默回退到旧引擎。

### 明确放弃的兼容责任

| 范围 | 新设计的决定 |
| --- | --- |
| local、local-fast、ray 的历史 runner 分工 | 统一为 local 和 ray 两种执行后端，两者使用共同查询语义 |
| PlanRunner、task stream 和旧 FTE manager 的调用约定 | 用 FragmentGraph 和 TaskRuntime 替换，新调度器不调用这些旧入口 |
| ManagedResult、MaterializedOutput 等结果包装的形状 | 直接定义 QueryResult 与 BatchLease，调用方一次性改用新契约 |
| ResourceVector 字段及已有预算模块的边界 | 按真实所有者重新定义 ResourceDemand、Reservation 和 MemoryLease |
| 旧 handle、ticket、计划缓存和 RPC 格式 | 同时替换生产端与消费端；旧缓存失效，同集群使用同一协议版本 |
| 未指定配置时的历史行为 | 使用新默认值；不根据旧环境变量或旧 runner 推断模式 |
| 旧内部测试、mock 和模块 import 路径 | 根据新契约重写或删除；SQL、恢复和资源安全场景重新落到新实现 |

可以继续使用经过验证的 DuckDB 算子、扫描器、Arrow 编解码和实用代码，但需要让它们服从新契约。代码复用不要求保留原来的外围结构，也不要求所有文件重写。

### 必须保留的语义约束

取消、背压、所有权、attempt fencing 和 EOF 判断仍是分布式执行的必要条件。它们分别防止任务泄漏、内存失控、悬空引用、重复输出和不完整结果。这些约束不会因取消兼容责任而消失。

新的 Ray 执行层第一版范围是只读 DAG 查询，固定分区路由。以下分布式能力暂不实现：查询自动重试、coordinator 故障恢复、执行中改变并行度、同一查询内混用直接边和物化边，以及写入提交。未支持的分布式能力直接拒绝；不会借旧实现继续提供。本地 SQL 能力由原生执行器决定。

FTE 描述恢复方式，OLAP 描述工作负载，因此公开执行模式使用 pipelined 和 fte，不新增 olap runner。

## 总体架构

~~~mermaid
flowchart TD
    A["Query API 与不可变查询配置"] --> B["共享 SQL 绑定与优化"]
    B --> L{"backend"}
    L -->|local| N["DuckDB 原生查询与增量结果"]
    L -->|ray| R["FragmentGraph"]
    R --> C{"execution"}
    C -->|pipelined| D["PipelinedScheduler"]
    C -->|fte| E["RecoveryScheduler"]
    D --> F["TaskService 与 TaskRuntime"]
    E --> F
    F --> G["ExchangeReader 与 ExchangeWriter"]
    G --> H["DirectExchange"]
    G --> I["MaterializedExchange"]
    H --> J["ResultService"]
    I --> J
    J --> K["QueryResult 与 BatchLease"]
    N --> K
~~~

图中表达组件关系。Ray scheduler 传递计划、split、路由、预算和状态；数据面由 native runtime 负责。Ray 后端负责 worker 放置、控制通信、进程生命周期和失效发现，故“后端”不只是运行地址。

local 直接推进 DuckDB 原生查询，内部使用 DuckDB 自己的 pipeline。它不需要新增跨 fragment 直接传输，不为结构一致而序列化本地计划、启动 actor 或建立分布式通道。只有 Ray 路径需要分布式 scheduler、TaskService 和 exchange。

Ray FragmentGraph 的基本结构与分布式策略无关。调度器为图绑定直接通道或物化存储，决定任务何时运行。以后允许模式专属的优化规则，但无需先维护两套分布式规划器。

### 最小抽象集合

| 抽象 | 必要职责 | 不承担的职责 |
| --- | --- | --- |
| 查询配置与 QuerySpec | SQL 或绑定计划、执行目标、预算、超时、连接快照 | 给 local 添加分布式策略，或根据历史 runner 猜测行为 |
| FragmentGraph | Ray 的可执行 fragment、端口、分区、依赖及资源需求 | 强迫 local 分片，或在构图期间运行任务 |
| Scheduler | 查询状态、任务准入、失败处理和取消 | 转发每个数据批次 |
| TaskService 与 TaskRuntime | 创建并推进一次 fragment attempt | 决定是否重试或哪个 attempt 可见 |
| ExchangeReader 与 ExchangeWriter | 有界异步读写、schema、结束和错误 | 隐藏任务恢复或查询重跑 |
| QueryResult 与 BatchLease | 增量消费、结果状态、关闭及借用所有权 | 继承模型请求或保存整个执行引擎 |

ResourceManager 为本地查询和 Ray scheduler 提供 Reservation，为 native 分配提供 MemoryLease；它服务于这些组件，不额外建立一套可执行资源图。

第一版使用具体实现和小接口。Ray 内有两个调度器和两种 exchange，local 有直接原生执行入口。不建设插件框架，也不为预想的第三种模式增加基类层次。分布式策略判断集中在 Ray 入口和 exchange 绑定处；普通算子不读取全局执行模式。

## 查询配置与公开接口

### 模式与执行后端

| 公开配置 | 执行路径 |
| --- | --- |
| backend=local，不传 execution | DuckDB 原生查询，直接消费 native 结果 |
| backend=ray，execution=pipelined | Ray worker 上的直接传输与并发推进 |
| backend=ray，execution=fte | Ray worker 上的物化输出提交与任务恢复 |
| backend=local，显式传任意 execution | 配置错误，拒绝执行 |

local 不公开 local+pipelined 或 local+FTE 两种组合，也不需要再区分 local 与 local-fast。其原生 pipeline 是 DuckDB 内部实现，不是新的分布式执行模式。

ray FTE 要求已提交输出的存储故障域独立于计算 worker。内存通道、进程内 TaskService 和本地临时文件可以用于分布式引擎的契约测试，不构成新的公开 local 执行路径。

~~~text
QuerySpec
  query_id
  target: LocalExecution | RayExecution
  statement_or_bound_plan
  connection_snapshot
  resources
  admission_timeout
  execution_timeout
  delivery_timeout

LocalExecution
  backend: LOCAL

RayExecution
  backend: RAY
  mode: PIPELINED | FTE
  fte_options: FteOptions | None

FteOptions
  exchange_store
  max_attempts
  retry_backoff_seconds
~~~

PIPELINED 只执行一个 attempt。FTE 按显式失败分类和重试上限创建后续 attempt。FteOptions 仅进入 FTE 查询快照，向 pipelined 查询传入重试参数时直接报配置错误。Runtime 可以预先登记 exchange_store，供后续 FTE 查询使用；pipelined 查询不使用该存储。

[query_options.py](vane/execution/query_options.py) 实现 LocalExecution、RayExecution、FteOptions 和 QueryExecutionOptions；[submission.py](vane/execution/submission.py) 实现内部 RayQuerySpec。P3 已接通 RayResources.exchange_stores、文件冻结、RecoveryScheduler 与公开 FTE QueryResult。exchange_store 解析为 Runtime 注册的 ExchangeStore，所有 worker 检查相同 root/store_id；独立故障域仍由部署方保证，不能用路径或 marker 自动证明。local 不使用 RayQuerySpec。

纯 `prepare_ray_query` 只编译和验证，不复制文件；普通 Parquet 不能经此入口声明为可重放。公开 FTE 查询在获得存储预留后调用明确具有文件写入效果的 `stage_ray_query`，先冻结输入，再绑定及优化。失败产生的部分副本由查询存储 lease 清理，不改变数据库状态。

### 公开 API 与后续目标

P1.1 接通以下 local 入口。QueryResources 是会话共享的容量，独立 cursor 共用准入和结果预算；QueryExecutionOptions 是本次查询的不可变期限快照。

`connect(":default:")` 只获取已有连接，不接受 `backend` 或 `resources` 等配置选项。需要 local 默认连接时，先通过 `connect(backend="local", resources=...)` 创建，再调用 `set_default_connection`；后续获取默认连接继续共用该会话的 runtime 和资源计费。

~~~python
import vane

limits = vane.QueryResources(
    max_active_queries=4,
    max_queued_queries=64,
    max_results=4,
    result_buffer_bytes=64 * 1024 * 1024,
)
options = vane.QueryExecutionOptions(
    target=vane.LocalExecution(),
    admission_timeout=30,
    execution_timeout=300,
    delivery_timeout=300,
)
with vane.connect(backend="local", resources=limits) as local:
    with local.query("SELECT ?::BIGINT AS x", [42], options=options) as result:
        table = result.collect()
        assert table.to_pylist() == [{"x": 42}]
        assert result.execution_state == "SUCCEEDED"
~~~

local.query 只接受自动提交下的单条只读 SELECT；命令使用 execute。支持位置/具名参数、原生算子与批次输出，模型 UDF 尚未接入。query 不读取 VANE_RUNNER，不构建 FragmentGraph，也不创建 LocalModelRequest。local 的 connect/query 拒绝任何 execution override，包括显式 None。未指定 backend 时使用 local；第一次 query 惰性建立会话共享的 QueryRuntime，并在会话锁内保证只发布一次。显式 resources 也默认选择 local。惰性 Relation 使用 sql 或 from_query。Ray 支持下述 pipelined 入口。

QueryResult 暴露 schema（Arrow schema）、query_id、context、read_batch/迭代、collect、cancel、close、execution_state 和交付 state。read_batch 返回 RecordBatch，正常 EOF 抛出 StopIteration，部分交付后的 native 错误继续抛出。collect 只收集剩余行，逐批复制到调用方内存并释放传输 lease。关闭结果或连接后，已经导出的 Arrow 切片、NumPy 零拷贝视图仍可读取并持续占用预算，直到最后一个视图释放。

当前默认 rows_per_batch=2048，资源与期限默认值如上例；这些是初始功能配置，性能验收后再调整。result_buffer_bytes 只限制结果交付持有的 IPC 缓冲，不包含 DuckDB 算子、native 预取缓冲或 collect 的完整副本。超过窗口的单批立即报容量错误；能够放入窗口的下一批等待旧 lease 释放，可由取消或期限唤醒。需要控制 native 预取时使用连接的 streaming_buffer_size 设置。查询与交付期限分别从准入及结果句柄就绪开始计算，慢消费期间二者都可能到期。

Ray 查询通过应用级 Runtime 创建会话。RayResources 定义整个服务的 worker 池、准入、结果和存储容量；不同 Session 及其 cursors 共用这些资源。Session 可传入更小的 QueryResources，服务与会话额度在同一个准入队列和结果账本中同时检查。

~~~python
import ray
import vane

ray.init()  # 也可以连接已经运行的 Ray 集群。
resources = vane.RayResources(worker_count=2, partitions=2)
with vane.Runtime(resources) as runtime, runtime.connect(execution="pipelined") as connection:
    with connection.query("SELECT range AS value FROM range(10000) WHERE range > 10") as result:
        for batch in result:
            print(batch)
            del batch
~~~

当前入口支持 P0 已验证的只读 SQL 子集：常量、range、普通本地 Parquet、filter、projection 与 GATHER；HASH 图可通过 FragmentCompileOptions 验收。文件必须使用所有 worker 可访问的绝对路径，prepare 和 start 均校验快照。整数、浮点、布尔、字符串及 NULL 在 native Flight 中传输，空结果保留 schema。单次 query 可显式选择 execution="pipelined" 或 execution="fte"；SQL 参数、聚合/join、模型 UDF 等尚未接线的能力明确报错。P3 的 FTE 入口如下。

Runtime 注册共享存储，Session 或单条查询选择 FTE。部署前将下例路径替换为所有参与者挂载的同一目录；输入文件只需要在提交端可读，worker 读取冻结副本。

~~~python
store = vane.ExchangeStore(
    "shared", "/mnt/vane-exchange",
    capacity_bytes=16 << 30, query_bytes=1 << 30,
    object_bytes=64 << 20, source_bytes=256 << 20,
    lease_seconds=60,
)
resources = vane.RayResources(exchange_stores=(store,))
options = vane.QueryExecutionOptions(
    vane.RayExecution("fte", vane.FteOptions("shared", 3, 0.1)),
    admission_timeout=30, execution_timeout=300, delivery_timeout=300,
)
with vane.Runtime(resources) as runtime, runtime.connect() as connection:
    with connection.query("SELECT range AS value FROM range(10000)", options=options) as result:
        for batch in result:
            print(batch)
            del batch
~~~

`runtime.connect(execution="fte")` 可设置会话默认策略。未提供 options 时，FTE 要求恰好一个注册 store，默认最多 3 次 attempt、退避 0.1 秒；多个 store 时必须显式选择。options 与 execution override 冲突会报错。两个策略共用 worker 池、查询准入和 native 资源账本。

客户端必须能访问 ResultService actor 公布的节点地址及 TCP 端口，worker 之间也须互通。消费者完成 Flight schema 握手后才启动生产，因此地址、ticket 或 schema 错误会在启动任务前失败。当前使用 Ray 节点地址和动态端口；网关、TLS 和固定端口部署属于后续部署能力。

连接提供默认值，query 提交时冻结快照。同一 Ray 连接的并发查询可以选择不同分布式策略，不修改进程环境；local 连接拒绝分布式策略 override。已经提交的查询不能原地切换目标。Relation 等上层表达入口如继续提供，也必须提交到同一个 QuerySpec 入口。

所有查询返回 QueryResult，默认逐批消费。collect 是显式完整收集操作，不受流式传输的固定内存保证保护。不存在按模式返回两种不同结果包装的分支。

预算、帧上限、并行度和重试上限集中定义在适用的配置中，经过容量验证后生效。local 不承担网络 exchange 或重试参数。数值默认值通过基准确定；Ray 默认 pipelined 不依赖保持历史行为。

### 协议与能力校验

所有参与者使用相同的 protocol revision 和 engine identity；不支持滚动混用新旧计划格式。缓存键包含 engine identity、执行配置、schema 和分区规则；重构时旧计划缓存直接失效。

提交后先检查完整计划、类型、scan 可执行性、存储保证及 worker 能力，再启动任务。QUERY 级版本和能力判断是防止误执行的校验，不实现旧格式转换。未知协议、无法回放的 FTE 输入或不支持的算子返回明确错误。

绑定期间可能调用扩展或 Python 代码；“任务启动前拒绝”只承诺没有启动执行任务，不能被理解为绑定期间完全没有用户代码运行。

## FragmentGraph 与执行身份

本节仅适用于 Ray 分布式执行。local 可以输出诊断计划，但不构建分布式 FragmentGraph；不能把诊断图当作提交分布式任务的依据。

### 一份可执行计划

~~~text
FragmentGraph
  query_id
  fragments: FragmentSpec[]
  edges: ExchangeSpec[]
  root_result: ResultSpec

FragmentSpec
  fragment_id
  native_plan
  input_ports
  output_ports
  partition_count
  sources
  source_dependencies
  semantic_dependencies
  resource_demand

ExchangeSpec
  exchange_id
  producer_fragment_id
  consumer_fragment_id
  consumer_input_port
  distribution: GATHER | HASH | BROADCAST
  partitioning_spec
  schema
  ordering_requirement
~~~

ExchangeSpec 描述数据关系，不自带历史 handle 或物化 barrier 标记。策略绑定阶段生成 DirectBinding 或 MaterializedBinding。第一版一条查询的所有跨 fragment 边使用同一种绑定，不提供混合恢复。

当前 [plan.py](vane/execution/plan.py) 实现端口、native payload、交换边、扫描 source/split、数据源依赖和单根结果的数据契约及校验。`sources` 描述优化后的可执行扫描，`source_dependencies` 保留优化前绑定的 Parquet 文件集合；两者都进入 native envelope 和严格的 Python 传输格式。依赖不需要 task 的 split 分配，不产生额外扫描或分区。schema、native 计划和 HASH 表达式以不可变 bytes 承载；解码时先校验协议和目标 engine identity。[compiler.py](vane/execution/compiler.py) 连接 native 编译器和加载验证。查询级资源声明和当前 SQL 子集的提交快照由 RayQuerySpec 承载；逐 fragment 资源分配、算子语义依赖随 scheduler 和分析算子接入。仅通过 Python 数据契约校验不代表 native payload 已可执行。

构图是无执行副作用的过程。远程 exchange 边界切分 fragment，扫描器产生可执行 source 描述。资源需求附着在同一图上，诊断图从它派生，不另行维护可能失真的可执行 ResourceGraph。

同一查询内使用一致的哈希、类型、NULL、排序和 collation 规则；分区函数由 native 实现提供。不能在 Python 与 C++ 分别计算近似分区规则。P4 将全局排序表达式保存在单分区 fragment 的 native 载荷中，排序后的根结果只有一个 producer。普通 exchange 的异步到达顺序不构成 SQL 顺序保证；没有 ORDER BY 的 LIMIT 只保证全局行数与 OFFSET。

### Native 编译与加载

内部入口 compile_fragment_graph 接收连接、SQL、query_id 和不可变 FragmentCompileOptions。选项只有分区数及可选的 HASH 输出列位置，不包含执行策略、worker 地址或 exchange store。连接锁覆盖读取绑定与优化设置的过程；编译器直接调用 Parser、Planner/Binder、Optimizer 和 PhysicalPlanGenerator，不创建 executor 或提交 task。需要准备提交时使用 prepare_ray_query，它在同一连接锁内完成快照与构图。

[native 编译器与加载器](src/vane_py/execution/fragment_plan.cpp) 支持单条无参数只读 SELECT：常量、整数 range/generate_series、显式 Parquet scan、filter、projection，以及 P4 的聚合、等值 hash join、ORDER BY/TopN 和全局 LIMIT。标量与聚合函数先按已验收的内置集合检查，用户函数在常量折叠前拒绝；相关子查询、窗口及媒体扩展类型仍拒绝。普通本地查询继续由原生查询入口处理，不调用该编译器。

解析树检查之后，编译器通过当前 Planner 的 catalog lookup callback 校验实际解析到的函数和宏必须为内置项。`current_user` 等 SQL value function 在解析时可能是列引用，绑定时才转为函数；这一检查发生在宏展开、表函数参数求值之前，并由子 Binder 继承，覆盖限定名、嵌套表达式和表子查询。内置宏间接解析到的用户函数同样被拒绝，不能依赖最终的只读属性检查或事务回滚来撤销 `nextval()` 等副作用。普通列和别名按 Binder 的实际解析结果处理，允许与被覆盖的函数同名；检查只属于这次规划，不改变后续原生查询行为。

绑定后的表达式在优化前按 native `IsConsistent()` 检查：同时拒绝 volatile 和 `CONSISTENT_WITHIN_QUERY` 函数，包括 `CURRENT_TIMESTAMP`、`CURRENT_DATE`、`CURRENT_TIME`、`LOCALTIMESTAMP` 和 `LOCALTIME`。当前提交描述尚未冻结查询时间，不能把单次查询内稳定误当成跨 task/attempt 稳定，也不能依赖可关闭的常量折叠。该限制统一应用于 pipelined 和 FTE；未来支持这类表达式时，需要在提交时冻结查询上下文，并让所有 task 与重试复用。

优化器入口遵循原生查询路径：仅当连接的 `enable_optimizer` 为真且逻辑计划要求优化时调用 `Optimizer::Optimize()`；启用时继续遵循 `disabled_optimizers` 的逐项设置。`PRAGMA disable_optimizer` 因而保留可直接执行的 `IN` 表达式。优化前的表达式、扫描列和数据源依赖校验始终执行。提交快照冻结这一连接选项，worker 准备和每次重放都恢复相同设置。

扫描、过滤和投影先合并到一个 source fragment。并行输出通过 GATHER 进入单分区根结果；显式指定 hash_columns 时，native 按结果列类型绑定 BoundReferenceExpression，生成 HASH 边，再按需要 GATHER。hash_columns 仍是内部结果分区请求；分析算子的分区由以下 native planner 自动生成。HASH 求值和 NULL、多列键合并均使用 DuckDB 原生表达式执行与 DataChunk.Hash。

P4 的 `AnalyticalPlanner` 保留已有 native 算子，在分布边界创建唯一端口与 fragment，不生成 SQL 字符串或 Python 行计算。具体规则如下：

| 算子 | 分布与执行规则 |
| --- | --- |
| COUNT、非 DECIMAL/HUGEINT 输入的 SUM、单参数 MIN/MAX | 每个 source 分区先执行 native partial aggregate；GROUP BY 键使用 HASH，无分组使用 GATHER；final aggregate 合并，投影恢复原始返回类型 |
| FLOAT/DOUBLE AVG | partial SUM 与 COUNT，final 合并后计算商；FILTER 在两项中一致传播；NULL 与空组保持 native 语义 |
| DISTINCT、带排序的聚合、DECIMAL/HUGEINT SUM、整数/decimal/时间 AVG、双参数 MIN/MAX(x,n)、Binder 内部聚合改写 | 原始行先按 group key 分区，再执行完整 native aggregate；全局聚合汇入一个分区，保留原生去重、精度/舍入和前 n 个极值语义；单参数 MIN/MAX 的 LIST 输入仍按普通标量合并 |
| 等值 hash join | INNER/LEFT/RIGHT/FULL/SEMI/ANTI（含优化器翻转后的 RIGHT_SEMI/RIGHT_ANTI）；等值与 IS NOT DISTINCT FROM 键由 native 计算，残余条件保留在 join 中 |
| BROADCAST | native 估计右侧不超过 32 行且为 INNER/LEFT/SEMI/ANTI 时广播 build 输入；其他 join 两侧按等值键 HASH；外连接不能广播产生重复未匹配 build 行 |
| ORDER BY | 在排序之前 GATHER，单分区执行完整 native sort；collation、NULL 顺序及 spill 仍由 DuckDB 决定 |
| TopN | 各分区先保留 LIMIT+OFFSET 个候选，再 GATHER 并执行最终 native TopN；检查 LIMIT+OFFSET 溢出，清除引用原物理图的动态过滤器 |
| LIMIT/OFFSET | GATHER 后使用串行 native streaming limit 维护全局计数；提前结束关闭对应输入消费者，既有通道错误继续具有优先级 |

整数和 decimal AVG 的 native finalizer 使用精确累加值及 long double 中间运算，不能用过早转为 DOUBLE 的 SUM/COUNT 替代。例如 `[9007199254740993, 2, 2]` 的均值必须保持 `3002399751580332.5`，否则 HAVING/WHERE 会改变结果。包含这类 AVG 的聚合节点按完整 group 执行；跨节点的类型与最终精度仍由 native 决定。

DECIMAL SUM 的原生累加器可以保存 39 位的 signed 128-bit 中间值，而其返回类型声明为 `DECIMAL(38,s)`。例如三个 `4×10³⁷` 和三个 `−3×10³⁷` 的总和合法，但前一个分区的部分和 `1.2×10³⁸` 不能作为 38 位 Arrow decimal 传输。包含 DECIMAL SUM 的整个聚合节点因此采用完整分组策略，保留 FILTER、NULL、scale 及同节点其他聚合的语义；代价是交换原始行，无法提前缩减为部分和。公开 DECIMAL 的 Arrow schema 继续使用声明的精度与 scale。

仅取消 partial 仍不足以避免溢出：异步输入可能让三个 `6×10³⁷` 先于三个 `−4×10³⁷` 到达。因此，对使用 128 位存储的 HUGEINT/DECIMAL 输入，fragment 的 SUM 与 AVG 共用 signed 192-bit 私有累加状态，以三个 unsigned 64-bit limb 实现精确加法与 combine，并维护非 NULL 行数。状态覆盖最多 64-bit 行数的 signed 128-bit 输入。SUM 在组完成后检查返回值的 128 位范围，保留 HUGEINT 或 DECIMAL 的 SQL 类型与 scale；AVG 允许总和超出 128 位，在完整组上除以行数及 DECIMAL scale。总和落在原生范围内时，AVG 继续使用原生 HUGEINT 到 long double 的转换，避免改变既有舍入；超出时从宽整数直接转换，保留正负抵消后的有效数位。

NULL、空输入、FILTER、DISTINCT 和排序仍由原生聚合执行器处理。普通本地 SQL 继续使用既有累加器，BIGINT 输入的 SUM 仍可拆分，Arrow profile 不变。函数序列化显式记录宽累加器选择，AVG 从输入类型恢复 DECIMAL scale，使 worker 与重放使用相同实现。

DuckDB 物理规划会把聚合内部的 ORDER BY 包装到函数中并清空 `order_bys`。fragment 在拆分判断前调用原生 `UnbindSortedAggregate`，恢复原始参数和排序键；带排序的 SUM/AVG 等聚合随后按完整组执行，避免把排序键当作普通函数参数或改变浮点运算顺序。Binder 为 INTERVAL 分组键生成的 `first`、为带 collation 的 MIN/MAX 生成的 `arg_min`/`arg_max` 也走完整组。解析阶段的公开函数集合不变，不因此开放直接调用这些内部聚合。

物理计划反序列化时，聚合参数已指向计算好的输入列。重新绑定函数数据使用这些表达式的副本，保留原参数，避免 collation 绑定器重复包装输入列。该规则由物理反序列化作用域标记控制；逻辑表达式反序列化继续保留绑定器的参数改写。

聚合的 FILTER 在 native hash/perfect-hash 算子内部会从输入列改为 payload 列。拆分之前通过原算子的映射恢复输入索引，构造 partial/final 后再由 native 绑定。Join 的 build/probe 依赖交给 DuckDB 的 pipeline events；在 probe source 首次被调度时记录 BuildReady，空 build 直接完成的情形由 finalizer 记录。状态读取只访问原子标记，不读取并发修改的哈希表，也不等待 `pump()` 的执行锁。没有引入兼容旧分布式聚合拆分器的接口。

fragment 使用自己的原生 envelope。普通节点保存不含子节点的 native 算子载荷；子节点和 input port 在 envelope 中显式表达。PhysicalOperator.SerializeNode 提供单节点序列化，无需拆改原计划树。加载器逐节点重建真实 PhysicalPlan，绑定每个输入端口，并验证 schema、算子子集、source capability、split codec 和分片身份。缺少绑定直接失败；不能用空扫描替代尚未接入的 exchange reader。

engine identity 同时包含 DuckDB SourceID 与 Vane 编译器/加载器源码摘要。schema、fragment 和 HASH 表达式分别带版本及身份，加载器在解释 native 算子之前检查身份与完整载荷边界。Python 图里的端口、source 和 source dependency 是原生描述的视图；validate_native_graph 对照 native 解码结果校验，拒绝两者不一致。

range 的扫描 split 由 table function 的 native 回调规划；Parquet 从绑定后的 MultiFileList 枚举文件，使用独立的文件 split codec，不依赖旧 FTE 的 split 管理器。载荷带稳定 split_id、能力和 codec 身份。worker bind 独立持有可移植状态，加载时必须显式提供所分配的 split；空列表代表空任务，未知或重复 split_id 被拒绝。无扫描的常量查询只执行一次，不因请求多个分区而复制结果。Parquet 的 requires_snapshot 标记为真：固定文件列表只封闭了枚举，提交层还必须检查访问条件与回放保证。

编译器在优化前的 logical validation 中捕获每个 Parquet bind 的完整文件集合，作为最终根 fragment 的 `source_dependencies`。所有 worker 的提交准备都会校验这些查询级依赖。统计信息可把扫描优化成空结果，hive/file pruning 也可移除部分文件，但这些优化仍依赖原始文件。依赖使用同一 native 文件 codec，随扫描节点的移除继续保留；实际 split 分配与并行度仍由优化后的 `sources` 决定。纯空结果的 source fragment 只执行一次，HASH/GATHER 下游不携带源文件分配。

当前 Parquet split 未携带原始文件序号，因此明确拒绝虚拟列 `file_index`。编译器在优化前检查 native 虚拟列 ID，覆盖投影和仅用于过滤的引用，物理计划导出与加载也检查同一限制；普通数据列使用相同名称仍可执行。未来开放该虚拟列时，split 和 scan bind 必须保留绑定时的原始文件索引，不能使用 task 内重新编号的文件列表代替。

Parquet 的 late materialization 会为 TopN/LIMIT 引入依赖 `file_index` 的双扫描回查。fragment 编译在优化前关闭每个绑定 Parquet scan 的 `late_materialization` 能力，继续使用普通扫描和 native TopN，再按上述规则分布执行。此能力只属于本次逻辑计划中的 TableFunction 副本，不修改共享 catalog、`disabled_optimizers` 或 local 查询；编译成功和失败均不影响后续原生优化。用户显式引用虚拟 `file_index` 时仍在优化前拒绝。

[native 编译测试](tests/fast/test_native_fragment_compiler.py) 使用有限数据的物化测试设施执行反序列化后的 fragment，并对照原生 SQL；它不接入公开 local 查询，也不作为 Ray 流水调度器。P1.2 的 TaskRuntime 通过独立的原生直接通道推进 fragment；跨进程 exchange 与根结果服务继续按 P2 实现。

### Ray 提交描述与 worker 准备

[RayQuerySpec](vane/execution/submission.py) 是可传输的不可变计划蓝图，包含 FragmentGraph、QueryExecutionOptions、ResourceDemand、连接快照、逐 fragment 的 source 快照以及结果列名；结果 schema 来自根输出端口。source-free fragment 也带快照 envelope。严格解码拒绝额外字段、未知协议、错误 engine identity 和不完整的快照集合。它不包含 worker 地址、task attempt 或已经兑现的 reservation。

prepare_ray_query 只接受 RayExecution。在连接锁内先读取语义设置，再完成 native 绑定和构图，捕获固定 source 状态，最后确认设置没有变化。该入口不启动执行器、旧 PlanRunner 或旧 FTE manager。local 原生查询既不序列化这份描述，也不读取分布式执行选项。

需要连接的 native fragment 入口统一使用 `DuckDBPyConnection::LockConnection()`，在等待连接锁或 context 锁前执行已有的 `CheckCallbackEntry()`。Python 输入回调内调用编译、提交准备、能力查询、计划加载或 HASH 校验时立即抛出 `InvalidInputException`；这一限制也适用于空闲 sibling cursor，避免嵌套工作再次依赖当前回调。回调处理该异常后，外层查询和后续正常规划仍可继续。

首版连接 profile 为 vane.builtin-session:1，限定于当前受支持的内置 SQL 子集：

- 捕获整数除法、IEEE 浮点语义、隐式转换、默认 collation、默认排序与 NULL 顺序、标识符大小写、表达式深度、optimizer 配置、TimeZone 和 Calendar。
- 使用 native 规则规范化等价的设置值，例如默认 ASCENDING 与显式 ASC；快照和缓存身份不因这类别名不同而变化。
- worker 使用查询独占的连接恢复 session 设置。只有全局 setter 的配置（当前包括 disabled_optimizers）必须与提交值一致；准备阶段不修改共享数据库配置。失败的准备连接由调用方关闭，不复用于其他查询。
- 自定义 collation 暂不支持。该 profile 不导出 attached database、Python 注册、远程文件系统或 secret；这些能力须在对应 SQL/scan 扩展时定义自己的可移植契约，不能借旧连接导出器隐式恢复。

数据源的保证按能力区分：

| 数据源 | pipelined 准备 | FTE 准备 |
| --- | --- | --- |
| 无扫描常量、整数 range/generate_series | 固定 native 计划和 split | 可按同一输入回放 |
| 普通 Parquet 文件 | 固定文件集合；要求绝对路径、worker 可见的本地普通文件；准备时核对路径、大小和包含原生亚秒精度的修改时间 | 纯准备入口拒绝；公开 FTE 查询先冻结为下述 snapshot codec |
| P3 冻结的 Parquet 副本 | codec 可验证同一内容 | 查询存储 lease 保护的副本；按 SHA-256、长度和 stat 验证，可重放 |
| 远程文件、其他 scan 或自定义文件系统 | 当前提交 profile 不支持 | 当前提交 profile 不支持 |

source 快照同时覆盖实际扫描和 `source_dependencies`，同一路径的文件元数据只捕获一次。扫描完全被优化掉时仍校验原始文件；部分文件被裁剪时也保留被裁剪文件的校验。普通 Parquet 的绝对路径、访问条件和 FTE 限制应用于全部依赖，不能因优化后的结果为空而绕过。依赖的 capability/codec 纳入 worker 能力检查，文件版本和依赖身份纳入蓝图缓存键；跨进程传输后不能从 Python 图中删掉 native 计划携带的依赖。

本地文件类型、大小和 mtime 来自同一次已打开句柄的 stat。快照同时保存标准微秒时间戳和 `mtime_nsec` 原生小数部分：Linux/macOS 保留纳秒，Windows 保留 FILETIME 的 100 纳秒精度；精度受底层文件系统限制。缺少原生小数部分时拒绝提交。同一秒内的等大小改写以及低于一微秒的 mtime 变化也参与 worker 校验和缓存身份。Parquet 元数据缓存使用的本地文件版本标识同步保留完整时间精度，重新规划可发现更新后的文件统计信息。

Parquet 元数据检查是访问前置条件，不是内容快照：相同大小和修改时间的替换可能无法检测，检查后再修改也不被阻止。调用方须在执行期间保持文件稳定；每次准备 task/attempt 时重新检查。P3 的公开 FTE 先复制到受查询存储 lease 保护的 staging，再绑定和优化；worker 重放该副本。未来其他 source 也须提供不可变版本或同等保证。文件内容哈希本身不能提供重试时的旧版本可读性。

[ResourceDemand](vane/execution/resource_demand.py) 声明查询的 CPU share、task context 数、I/O 并发和 operator/result/exchange/staging 四类内存。当前为 CPU 算子子集，不预先声明尚无执行能力的 GPU/UDF 资源。Ray 必须显式提供 exchange 和 staging 预算。首版 pipelined 按整张图同时活动，context 声明至少覆盖分区数之和；严格阶段 FTE 至少覆盖最大阶段分区数。这只验证声明能描述计划；实际 worker 容量、最小可推进窗口和原子预留由 P2/P3 的准入实现。

prepare_worker_plan 先检查 worker 的 engine、协议、类型/连接 profile、exchange distribution 和 scan/dependency capability/codec，再恢复连接、检查 source 与原始数据源依赖、加载原生 fragment，并对照 Python 图中的端口、source、source dependency、结果列名和 HASH 规则。能力清单由 native catalog 与当前编译器产生，表示加载能力；它不证明 TaskRuntime、网络端点或 exchange store 已就绪。未来 scheduler 必须在每个 task/attempt 启动前使用这些检查，并完成自身的准入与存储检查。

RayQuerySpec.cache_key 对规范序列化计算 SHA-256，包含 engine、完整图、schema、分区规则、执行/超时选项、资源声明、连接与 source 快照，只排除 query_id。它标识可复用的计划蓝图；不缓存结果、连接、attempt、端点或 reservation，命中后仍必须重新检查 source 与 worker。

[提交验收测试](tests/fast/test_execution_submission.py) 覆盖独立进程加载、规划连接关闭、设置冻结与 session 隔离、文件变化、能力不匹配、快照损坏、native 元数据不一致和缓存身份。测试通过有限物化设施核对 SQL 结果，尚不代表 Ray 流水查询已可公开执行。

### 固定路由与 split

启动前确定分区数、逻辑 task 集合及分区到 task 的映射。pipelined 扫描 split 可以增量发现，但已经分配的 split 有稳定身份。FTE 第一版在相应扫描阶段启动前封闭每个逻辑 task 的输入快照和 split 清单；失败重试重放同一输入，不重新发现一份可能变化的数据。

空表、空分区和无 source 的常量查询同样有 schema 和显式完成信息，不能靠“第一批数据到达”初始化协议。未完成 split 枚举的 task 不能被提交为成功。

~~~text
QueryId = 本次提交的唯一身份
TaskId = QueryId + FragmentId + PartitionId
AttemptId = TaskId + AttemptNumber
WorkerEpoch = 本次 worker 进程实例
~~~

数据描述和控制事件同时携带 QueryId、AttemptId、WorkerEpoch、路由版本及 schema 身份。逻辑 task 与物理 attempt 分离，用于拒绝迟到提交和跨查询串流；不为兼容旧 ID 格式增加字段映射。

### 三类依赖

| 依赖 | 含义 |
| --- | --- |
| 路由就绪 | 输入输出端点、身份和预算已经安装 |
| 并发消费 | 上下游可以在对方尚未完成时处理数据 |
| 算子语义依赖 | join build-ready、sort 输入完成等条件 |

pipelined 可以让 sort 在上游运行时接收数据，但不能绕过完整排序的语义。join task 的 build 消费必须能够先运行，不能因为 probe 尚未就绪就阻塞整个 task。

FTE 第一版另外采用完整上游阶段提交屏障，属于 RecoveryScheduler 的保守调度选择，不写死在物理算子或 FragmentGraph 中。

## 共同任务运行时

TaskRuntime 用 C++ 执行 DuckDB fragment，管理输入队列、输出 writer、取消、native 等待和进度。相同的 fragment 在两种模式下只更换输入输出 binding。

TaskService 为 Ray 的两种调度器提供同一组操作，测试可使用进程内调用：

~~~text
prepare(TaskSpec, Reservation) -> PreparedTask
start(AttemptId, StartToken)
add_splits(AttemptId, UpdateSequence, Splits)
seal_splits(AttemptId, UpdateSequence)
install_inputs(AttemptId, RoutingVersion, Inputs)
seal_inputs(AttemptId, RoutingVersion)
watch(AttemptId, MinStatusVersion) -> TaskStatus
cancel(AttemptId, Reason)
release(AttemptId) -> CleanupStatus
~~~

prepare 创建上下文、绑定端点和预算，不开始扫描。start 对相同 token 幂等。更新携带单调序号，相同序号与相同内容可重复；相同序号与不同内容属于协议错误。

任务状态为 CREATED → PREPARED → RUNNING → OUTPUT_PENDING → FINISHED，另有 FAILED 和 CANCELED。RUNNING 的输入、输出或预算等待通过 blocked_reason 表示，不重新创建 task。

计算完成与输出完成分开。direct 输出尚有未确认 ownership，或物化输出尚未完成存储封存及提交接受时，任务停留在 OUTPUT_PENDING。TaskRuntime 报告 OutputSealed，scheduler 完成相应的排空或提交检查后记录 FINISHED，并通过幂等释放操作确认输出所有权转移。

ray 后端通过 actor RPC 调用 native 服务；进程内测试覆盖同样的协议。公开 local 入口直接调用原生查询，不经过 TaskService。Ray 的 actor 自动重启或方法重试不负责重放 task。worker epoch 变化后，由 scheduler 决定查询失败或创建新的 FTE attempt。

QueryCoordinator 串行处理一个查询的状态转换，I/O 和等待在状态入口外执行。终止状态不可被迟到事件改写。订阅队列有界，可合并重复的进度快照，但不能丢弃终止事件。

两个 scheduler 具有 start、on_event、cancel 和 snapshot 四个共同操作。查询状态为 PLANNING → ADMISSION_WAIT → RUNNING → FINALIZING → SUCCEEDED，失败与取消分别进入 FAILED 和 CANCELED。FINALIZING 检查必需的输出终结条件；清理进度和客户端交付状态单独记录，不用它们覆盖执行结局。

## PipelinedScheduler

### 活动组准入

通过直接通道相互等待的任务形成活动组。启动生产前，整个组必须获得实际 worker 的上下文容量、输入窗口和最小输出推进预算。第一版采用保守的组划分，必要时整张连通图作为一组。

任务上下文数量与 CPU 执行线程数分开。阻塞的 native pipeline 让出线程，但其算子状态、连接和缓冲仍然占用预算。逻辑 CPU 配额不能替代真实 worker reservation。

组预留采用全部成功或撤销的协议，具有有限 deadline 和幂等释放。申请按确定顺序进行；部分失败时释放本轮临时占用，再重新排队，禁止带着部分资源无限等待。若无法满足最小推进容量，在生产前降低并行度并重新构图，或明确拒绝。

第一版不抢占运行中的 pipelined task。多个查询公平排队，取消能中断准入和准备。不能通过临时突破内存硬上限来解除死锁。

### 启动与推进

~~~mermaid
sequenceDiagram
    participant S as PipelinedScheduler
    participant R as ResourceManager
    participant P as Producer
    participant C as Consumer
    participant O as ResultService
    S->>R: Reserve active group
    R-->>S: Reservations
    S->>P: Prepare outputs
    S->>C: Prepare inputs and outputs
    S->>O: Prepare root consumer
    P-->>S: Prepared
    C-->>S: Prepared
    O-->>S: Prepared
    S->>C: Start consumer
    S->>P: Start producer and splits
    P-->>C: Bounded batches
    C-->>O: Incremental result
    O-->>C: Release credit
    C-->>P: Release credit
~~~

消费者及结果出口先准备好，再启动生产者。join 的 build 输入先推进，probe 输入根据 BuildReady 事件解除语义等待。单个 producer 暂时没有数据不会阻止读取其他就绪输入。

PIPELINED 不重试已启动的 task。worker 丢失、不可恢复的数据连接中断或算子失败使整个查询失败，唤醒读端并关闭相关通道。已经返回的部分结果不能以正常 EOF 收尾。

## RecoveryScheduler 与新的 FTE 路径

### 第一版恢复边界

FTE 使用同一 FragmentGraph 和 TaskRuntime，将跨 fragment 边绑定到 MaterializedExchange。每个 fragment 的所有逻辑 task 输出提交后，封闭该阶段的 manifest；下游从固定的已提交输入开始运行。

第一版不做 FTE 阶段重叠、推测执行或动态分区。一个逻辑 task 同时只认可一个活动 attempt。严格的阶段屏障减少重放与可见性状态，后续优化需保持同样的提交语义。

任务可恢复的前提是输入可重放、计算符合声明的重放能力、成功输出独立于失败的 worker。只读 SQL 也可能包含 volatile 表达式、外部请求或变化的数据源，必须逐项校验。第一版不允许未声明重放保证的 UDF、扩展和扫描器进入 FTE。

### 输出封存与提交

~~~text
attempt 写入独立命名空间
  → 写完不可变分区对象
  → 校验完整性并封存 AttemptManifest
  → coordinator 对 TaskId 选定唯一成功 AttemptId
  → 所有逻辑 task 已提交后封闭 StageManifest
  → 下游读取 StageManifest 指定的对象
~~~

AttemptManifest 包含分区对象、长度、校验值、schema、输入身份与 attempt 身份。完成上传不等于成功 attempt；只有 coordinator 接受的提交才能进入 StageManifest。对象存储读取使用明确 object key，不依赖目录 listing 推断完整性。

任务提交按 TaskId 和当前 attempt fencing token 原子选择。相同提交重复到达返回相同决定，过期 attempt 的迟到成功被拒绝。下游不能同时消费两个 attempt 的输出。

如果提交已接受但确认丢失，重复 RPC 查询并返回既有决定，不创建新 attempt。coordinator 的提交表在本次查询内有效；第一版不提供 coordinator 故障恢复，因此无需为此增加跨 coordinator 选主协议。

### P3.1 物化 I/O 与提交基础

实现位于 [native MaterializedIO](src/vane_py/execution/materialized_exchange.cpp)、[不可变 manifest](vane/execution/materialized_exchange.py) 与 [共享目录和提交账本](vane/execution/materialized_store.py)。native 读写与 Flight 共用 [Arrow frame codec](src/vane_py/execution/arrow_frame.cpp)，类型范围由 P4 的 analytical-types profile 统一扩展。没有新增执行器或调用旧 FTE manager。

MaterializedIO 在独立 C++ 线程中读写对象，通过有界 native channel 与现有 TaskRuntime 的 source/sink 连接。该 channel 是任务内部缓冲，不发布 Flight ticket，不把存储对象伪装成远程直接通道。Python 只处理身份、路径、长度、哈希和 manifest，数据批次不经过 Python 或 Ray ObjectRef。

对象采用版本为 `vane.materialized-arrow:1` 的封存格式：

| 部分 | 内容 |
| --- | --- |
| 文件头 | `VANEMAT1`、native schema 长度及字节 |
| 数据帧 | Arrow IPC 长度、行数、该帧 SHA-256、含一个 batch 的独立 Arrow IPC stream |
| 文件尾 | 零长度结束标记、总帧数、总行数 |
| manifest 元数据 | 整个文件的长度和 SHA-256、行数、帧数、schema、分区与 attempt 身份 |

整数使用 64 位小端编码。空分区也必须写入带 schema 和文件尾的对象。写端排空已封闭生产者后写入文件尾，sync 并关闭文件，才报告封存完成；关闭 writer 不删除文件。读端先以固定 64 KiB 缓冲核对整个文件的大小、SHA-256、schema 和文件尾，再逐帧校验、解码和交付；缺失、截断、变更或元数据不匹配均以错误结束。EOF 与 reader 完成状态在同一取消锁内发布，读到 EOF 后立即 close 不会把成功交付改为取消。

每个对象最多 1 TiB，native frame 最多 256 MiB；schema 最多 256 列、64 KiB。单帧编码长度上界为 `4 * frame_bytes + 64 KiB`，调用方须声明至少 `16 * frame_bytes + 256 KiB` 的 staging，另外持有 channel 的窗口预算。native I/O 同时只处理一个帧，等待容量时保留所有权。每个 attempt 启动前在同一个 worker 账本中预留 context、operator share、channel window、native staging 和 I/O 槽；两个策略同时运行也不能突破各项容量。这些约束不等价于完整的进程 RSS 硬限。

SharedDirectoryStore 要求所有参与者访问同一个绝对路径和 `store_id`，支持独占创建、原子 hard link 发布和文件 sync。目录必须由部署方放在独立于计算 worker 的共享存储上；路径名称与 marker 只能校验身份和可见性，无法证明挂载的物理故障域。当前测试验证独立生产进程退出后数据保留，不宣称本地临时目录可承受整台存储节点失效。

CommitCoordinator 按阶段接收固定的逻辑 task、输入指纹和全部输出分区声明；下游指纹包含实际已提交的上游 manifest，不能在上游选定之前伪造。每次 begin 产生递增 attempt、随机 fence 和指定 worker epoch；创建后续 attempt 会立即废弃旧提交资格，但不会提前归还旧对象配额。每个查询最多 4096 个逻辑 task/输出分区，活动 attempt 的总分区数也有相同上限。存储配额按对象最大预留长度计费，成功清理后才释放。

commit 先核对 engine、查询、task、输入、epoch、fence、分区、schema 和对象预算，在决策锁外读取对象并发布不可变 `attempt.json`，最后再次验证 fence 并选择唯一成功输出。取消或新 attempt 可以在慢存储验证期间抢先生效。相同提交重复到达返回既有决定，冲突提交报错。`attempt.json` 是封存记录；即使文件存在，也不代表已被接受。只有本次 coordinator 的提交表选中的 attempt 可以进入 StageManifest。

stage 的全部预声明 task 提交后才能发布 StageManifest；读取方必须持有该 stage 的 ReadLease，不能用目录 listing 补全对象或从失败 attempt 取数据。成功对象归 query 所有，仍有 ReadLease 时 close 会保留对象及配额。调用方必须先停止 native I/O 或确认 worker 已退出，才能 discard 未提交 attempt；删除失败保持原有资源责任，可再次清理。已记录的输入错误优先于后续取消原因。

### P3.2 不可变文件输入

[file_snapshot.cpp](src/vane_py/execution/file_snapshot.cpp) 在 Binder/Optimizer 之前复制本次查询的文件集合。当前支持绝对本地 Parquet 路径、glob 和字面量路径列表，最多 4096 个不同文件，每个 scan 的模式和展开后的文件引用也各限 4096；以 64 KiB 缓冲复制、核对源文件两次 SHA-256 与元数据，sync 后绑定冻结路径。复制受查询中断和 source_bytes 容量约束；不接受动态路径表达式、未声明能力的远程文件系统或其他 scan。

创建任何快照前，先展开整个查询所有 scan 的路径模式，固定每个 scan 的完整文件引用列表并检查模式与引用数量。列表保留顺序和重复引用，之后才按快照身份去重复制和计费。exchange store 位于递归 glob 目录内时，本次复制产生的文件不会成为后续模式的新输入；展开时已存在且匹配的文件仍是合法输入。

冻结路径编码从原始文件引用解析出的 Hive 分区键和值；解析发生在任何路径规范化之前。FTE 在优化前按绑定列 ID 拒绝对生成 filename 列的引用，包括虚拟列以及 `filename=true`、`filename='origin'` 创建的普通索引列；仅在过滤中引用或之后被剪枝也拒绝。虚拟 file_index 同样拒绝，真实物理同名列和未引用生成 filename 的扫描仍可使用。local 与 Ray pipelined 保留生成 filename 的原始路径语义。冻结发生在统计信息剪枝之前，被优化成 EMPTY_RESULT 的 scan 仍保留 source dependency。`vane.parquet-snapshot:1` split 携带冻结文件 SHA-256/长度，worker prepare/start 校验内容与精确文件状态。原文件覆盖、删除或 glob 新增成员不改变重试输入；冻结副本缺失/损坏则失败。

展开后的文件通过 scan 的原生 dependency 作为确定文件列表交给 Binder，校验与路径参数逐项一致后构造 SimpleMultiFileList；冻结路径中的 `*`、`?`、`[]` 不再作为 glob 解释。普通查询仍按原规则展开路径。快照目标由解析后的物理源路径哈希、原始引用的 Hive 分区键值和固定文件名组成。Hive 键值直接使用原生 HivePartitioning::Parse 的首次键优先及编码规则，例如 `part=42/../` 仍表示分区 42。同一物理文件、相同分区键值的别名复用副本并保留重复扫描；不同分区值保留独立副本并分别计费。symlink/`..` 解析到不同物理文件时也不会误合并。

[fte_plan.py](vane/execution/fte_plan.py) 将不可变 QuerySpec、固定 split assignment、partition 与实际上游 StageManifest 指纹组成输入身份。每次 attempt 使用相同身份；不重新编译 SQL，不重新展开 glob，也不替换已提交对象。

### P3.3 恢复调度、配额与租约

[RecoveryScheduler](vane/execution/recovery_runtime.py) 使用 P2 的 worker 池与 native TaskService，每个阶段按 worker 数量分批运行，每个查询在一个 worker 上至多运行一个 attempt。所有 stage task 提交后才声明下一阶段的输入。worker actor 禁用自动重启和 RPC 自动重放；未提交 attempt 遇到 RayActorError 时，scheduler 创建新 epoch 的 worker，按固定输入生成新 fence。Ray 状态探测读取独立 native production/error 快照，不等待 pump 执行锁；准备和源文件校验受查询执行期限约束。

[StorePool](vane/execution/fte_store.py) 通过原子元数据与跨进程锁，在所有注册同一 root 的会话之间预留 query_bytes。容量配置必须一致；源文件按实际字节计入查询容量，exchange 按各对象最大长度预留。失败 attempt 未清理完成时继续占用配额。metadata 有独立的 4096 项和序列化长度限制；数据配额不包含文件系统 metadata 开销。

每个查询具有随机 namespace、lease_id 和有限到期时间。coordinator 独立线程续租；worker 与 ResultService 独立线程检查 lease/fence，过期即取消 native 执行并清理。所有 native 参与者持有查询共享锁，attempt 另有 I/O 锁。查询锁位于数据目录外的 `allocations/<namespace>.lock`，不会被递归删除带走。删除必须取得独占锁；Unix 使用 open-description flock，Windows 使用 LockFileEx，不能用会被同进程其他 close 释放的进程级锁替代。

query 配额记录位于对象目录外，部分删除失败不会丢失计费记录。actor 死亡通知可能先于进程释放锁，清理对此提供有限宽限；仍不能取得锁时保留 CleanupPending，调用方可以重试 close。文件提供者不能中断内核阻塞的文件 I/O，因此不承诺任意存储故障下的物理清理时限。RPC/线程等待有期限，不能因此先释放仍在使用的对象。

全局 `allocation.lock` 仅保护配额和租约元数据：扫描候选后，在全局锁内再次检查有效期并取得查询独占锁，释放全局锁后删除数据，最后重新进入全局锁核对 lease_id 并移除查询锁及 allocation。整个删除期间继续计费，其他回收者即使看到数据目录已消失，也必须先取得查询独占锁。普通 close 的清理回调和递归删除同样在全局锁外；close 先使租约过期，阻止迟到的 native 参与者，清理失败保留所有权与配额。

每次准入和活跃查询心跳请求后台回收；每个 StorePool 同时至多运行一个后台清理线程，慢删除不阻塞续租或有剩余容量的新查询。尚未清理的 allocation 继续占用容量，容量不足时准入明确拒绝，可在回收完成后重试。worker orphan 清理以及显式 `StorePool.collect_expired()` 可以同步回收，但仍不占用全局锁执行删除。租约使用墙钟，部署要求节点时钟同步；存储要求共享可见性、跨进程 advisory lock、原子创建/替换和 sync。coordinator 丢失会终止查询并回收 orphan，不恢复查询或接管旧提交账本。

单个过期目录的加锁或删除 I/O 失败按目录隔离，继续回收其他 orphan；未删除的外部 allocation 保留配额，后续心跳或准入重试，有剩余容量时仍允许新查询。当前查询续租失败、存储身份变化及元数据损坏仍按原契约报错。

### 重试与存储故障

| 情况 | 新 FTE 的处理 |
| --- | --- |
| attempt 提交前计算 worker 丢失 | 废弃未提交输出，在可用 worker 重放相同逻辑输入 |
| 已提交输出的生产 worker 丢失 | 从独立存储继续读取，不重跑已经提交的 task |
| 未提交下游 attempt 的 worker 丢失 | 重试该下游，输入仍指向同一批已提交对象 |
| 幂等 manifest 发布遇到 EAGAIN/EINTR/ETIMEDOUT/ECONNRESET | 原操作最多尝试 3 次，重复确认沿用既有提交；耗尽后失败 |
| 其他控制超时或 native 存储 I/O 错误 | 明确失败，不把所有异常当作可恢复 worker 丢失 |
| SQL、类型、权限错误或内存硬限不足 | 查询失败；不重复执行确定失败的任务 |
| 已提交对象永久丢失或损坏 | 查询失败；第一版不重建已被下游消费的恢复区域 |
| coordinator 丢失 | 查询失败，存活资源依据 lease 清理 |

每次任务重试都消耗重试次数和同一个执行 deadline。退避不能重置总时限。attempt 的失败、取消和提交接受必须经同一串行状态入口排序。

未提交对象属于 attempt，失败后可清理；已提交对象属于 query 或结果 lease，生产者退出不能删除。临时对象与失联查询具有有限 lease 和回收机制。终止清理不能先删除仍被消费者使用的对象。

### 根结果提交

根 fragment 同样写入物化对象。查询执行成功后发布不可变 ResultManifest，ResultService 才开始向客户端交付 FTE 结果。因此 FTE 不会把待重试 attempt 的行暴露给客户端。

根 manifest 记录 schema、分区及明确的顺序要求；不能把对象列举顺序当作 ORDER BY。成功结果的保留期限由 result lease 管理，不受计算 worker 的生命周期影响。

## Exchange 数据接口

### 共同读写与分开的协调语义

~~~text
ExchangeInput
  DirectInput { channels, membership_version }
  MaterializedInput { committed_manifest }

ExchangeOutput
  DirectOutput { channels, limits }
  MaterializedOutput { attempt_namespace, store, limits }

Writer.try_write(partition, owned_batch) -> ACCEPTED | BLOCKED | ERROR
Writer.seal() -> PENDING | SEALED(OutputSeal) | ERROR
Writer.subscribe_writable(wakeup)
Writer.abort(reason)

Reader.poll() -> DATA(BatchLease) | BLOCKED | EOF | ERROR
Reader.subscribe_readable(wakeup)
Reader.close(reason)
~~~

OutputSeal 是带类型的结果：DirectFinish 或 AttemptManifest。direct seal 表示不再生产新数据，剩余缓冲交给 channel 管理；materialized seal 表示本 attempt 的存储对象已经完成，不表示 coordinator 已接受提交。

DirectExchange 管理活跃通道与消费者需求。MaterializedExchange 管理对象、manifest 与存储 lease。两种实现直接满足新接口，不相互模拟 endpoint、文件路径、成功 attempt 或 committed manifest。

schema 在数据之前可得。Reader.poll 的暂时无数据必须返回 BLOCKED；EOF、错误、消费者主动停止具有不同的状态。

### 直接通道协议

每条通道具有 QueryId、AttemptId、双方 WorkerEpoch、exchange 和分区身份、路由版本、schema 身份与访问 capability。可猜测的 channel ID 不构成读取或取消授权，日志不输出 capability。

同进程直接交换使用有界内存通道。跨进程使用新的 native Flight 服务：长连接 DoGet 传输 Arrow 帧，DoAction 承载累计 ACK 与关闭。可以复用 Flight 库和实现经验，但不必保留旧 server 的 ticket 分支或处理器。

帧有单调 sequence、长度与明确的 schema/dictionary 信息。重复数据帧、跳号和 schema 不匹配是协议错误。控制 ACK 可以重复，确认尚未发送的帧是协议错误。第一版不提供中断连接的续传或数据重放。

I/O 使用独立、有界的执行资源，native 计算线程不等待 socket。控制动作能够在数据窗口耗尽时推进。Flight 的并发 DoGet、DoAction、取消和关闭能力必须通过原型验证；如库接口不满足，需要修改传输实现。

### 背压与所有权

每个通道在接收端预留字节窗口 W，发送端保持未确认字节数不超过 W。ACK 表示对应接收所有权已释放，或已经转移到另一个明确计费的内存所有者；网络收到数据本身不能退还消费窗口。

若 downstream 借用零拷贝缓冲，lease 持续存在；若复制进算子状态，原传输 lease 可以释放，新分配交给算子所有者。发送端保留未确认 payload，用于明确缓冲生命周期，不据此承诺任务恢复。

所有计划通道的最小窗口和每个任务的输出推进额度，在活动组准入时一并预留。不能为任意多的连接隐式增加内存。广播共享不可变 payload，实际分配计一次，每个消费者具有独立窗口与游标。

ACK 可以按字节或时间合并，但必须有上限，不能让等待该 ACK 的生产者无限停顿。慢消费者只背压相关路径；一个无数据的输入不能阻止读取其他就绪输入。

### 成员封闭与 EOF

输入成员可以逐个安装，之后明确发送 NoMoreProducers。输入正常 EOF 同时要求：

1. 成员集合已经封闭。
2. 每个有效 producer 已发出 FINISH(last_sequence)，且数据已经消费到该位置。
3. 没有未上报的 task、channel 或查询错误。

空通道也发送 FINISH。NoMoreSplits、NoMoreProducers 和 FINISH 分别描述扫描输入、远程成员集合和单个生产者，不能互相代替。

广播消费者关闭时只释放自己的需求和游标。所有消费者均不再需要某个输出时，才允许终止对应上游工作。

### P1.2 已实现的进程内通道与任务服务

[direct_exchange.cpp](src/vane_py/execution/direct_exchange.cpp) 实现固定 schema 的有界 native channel，消费窗口包含队列中的帧和已借出的帧。DirectLimits 定义每个消费者的 window_bytes、每帧 frame_bytes/frame_rows 与未释放帧数 frame_slots。窗口至少容纳一行的固定布局，否则准备时拒绝；运行时遇到超出单帧容量的变长行则明确报错，不创建超额缓冲。

每帧分配一个独立缓冲，包含对齐后的有效性位图、定长值、长字符串/二进制数据，以及 P4 嵌套值的全部子向量。列表按选中元素紧凑复制；NULL 列表的未定义 offset 不参与寻址。写入先测量并检查额度，再复制和发布；暂时无法写入时不分配 payload。原生输入由执行器持有，通道不引用瞬时 Sink 参数。读取构造借用该帧的 Vector，每层 Vector auxiliary 均持有 lease，嵌套字段、切片及字符串引用继续保留整个帧的所有权。最后一个引用释放后归还窗口。广播共享一次物理分配，每个消费者独立计费；关闭一个消费者释放其队列，已经借出的视图继续有效并保持计费。

嵌套 ARRAY 的长度按其所在层维护，不能沿用最外层批次行数。Arrow 解码在递归写入每个 ARRAY 前设置实际数组数量；原生 LIST/MAP/STRUCT 扩容时同步更新普通 ARRAY 的子元素长度。固定 Image/Tensor 的延迟存储仍由写入器单独预留，不随父容器扩容而分配像素或张量元素。

Poll 和 TryWrite 在同一个 channel mutex 内检查条件并注册等待者。发布数据、FINISH、封闭成员、归还额度、关闭及错误都在锁外执行唤醒回调。回调复制 DuckDB 的 InterruptState，使用 weak task 引用及 interrupt epoch；不保留裸 pipeline 指针，不调用 Python。每个生产者/消费者最多保存一个等待者，元数据数量由固定成员和 frame_slots 限定。FINISH(last_sequence) 必须匹配最后一个已接受序号；拒绝重放、跳号和 FINISH 后的数据。错误保持可见，不能转成正常 EOF。

[direct_task.cpp](src/vane_py/execution/direct_task.cpp) 的 DirectSource 支持一个输入端口连接多个通道，轮询就绪通道，不等待空闲输入；无数据时返回 BLOCKED。已读到 EOF 或已关闭的输入仍参加后续轮询，Poll 始终先检查持久错误，不缓存永久结束状态。DirectCollector 在 native 中求 HASH 分区，按输出、目标分区和行位置保存提交进度。BLOCKED 后只恢复未发送部分。每次计算目标帧大小前先检查通道错误和消费者；已无消费者的目标直接丢弃剩余行，不因该分区的超大行取消其他分区。仍有消费者的目标继续执行帧容量限制，通道错误始终传播。source/sink 强制使用 DuckDB 的 ExecutionBatch 路径，在获取下一批前释放已消费的中间引用，避免一帧窗口被执行器的旧视图占住。Finalize 在发送 FINISH 前检查全部输入通道的持久错误，避免 sink 提前停止绕过下一次 source 轮询；控制结果为空，不收集 fragment 数据，也不等待消费者。

DirectTaskService 为每个 attempt 创建独立 native Connection。prepare 恢复连接快照、校验 source、加载输入 binding 和输出路由，尚不创建 PendingQuery。所有 task 准备完后才能 start；同一 task 的相同 start token 幂等，不同 token 报错。start 再次验证数据源，然后创建带 DirectCollector 的原生执行器。pump 轮转调用 PendingQuery.ExecuteTask，一个执行线程也可推进多个相互等待的 fragment。native 生产完成后，pump 收取控制结果并释放查询上下文，状态进入 OUTPUT_PENDING；所有输出没有错误且 lease 均释放后，才进入 FINISHED。

取消先通过独立控制入口中断 context、将 channel 置为持久错误并唤醒等待者，后续 release 负责确认清理。它不等待 pump 持有的操作锁。接受取消或执行超时前，先检查 native 执行器已记录的错误及所有输入、输出通道的持久错误；已有失败优先，发生错误的任务保持 FAILED，其余任务因该错误停止，结果端收到原始原因。执行器的 TaskErrorManager 以共享所有权保留，在开始调度前交给 TaskService；定时器读取它无需获取 context 锁，执行器释放后仍可读取已记录的错误。清理前移除该句柄，避免清理产生的中断覆盖既定结局。

执行期限同时读取各输出生产者在 channel 锁下发布的 FINISH，不使用由 pump 更新的完成计数。没有已有失败且全部输出已完成生产时，即使后台线程完成后尚未再次 pump，迟到的执行定时器也不能取消借用中的结果。尚未完成或尚未启动的生产者仍受执行期限约束。失败和完成检查均独立于 pump；后续 pump、status、release 及再次取消保留首次接受的停止原因。

控制层在 pump、status 和 release 中刷新输出状态。任务的所有输出均已失去消费者时，先检查所有输入通道的持久错误及已有执行错误，再停止并清理原生执行器、关闭上游消费端，最后封闭输出生产。输入 abort 只唤醒执行器，CheckPulse 不一定已经观察到错误；因此提前收尾必须直接检查输入通道。已有输入错误使任务进入 FAILED，保留并传播原始原因，不发送成功的 FINISH。该路径不依赖 Sink 再次被调用，等待空输入的任务也能退出。无错误时，广播或多个输出只关闭一部分消费者仍继续执行；已借出的输出仍保持计费及 OUTPUT_PENDING，直到最后引用释放。

刷新时检查所有输入、输出的持久错误，不能因前一个输出尚未排空就跳过后面的错误。输入错误检查由 native finalize、状态刷新及取消/期限判断共同复用；即使输入已读到 EOF、执行器已清理，只要输出交付尚未完成，已有输入错误仍使任务进入 FAILED。abort 丢弃队列只代表归还容量，不代表交付成功；错误传播取消以停止其余任务，原错误在后续清理和取消中保留。Python 输入回调重入在获取服务锁或修改控制状态前拒绝。状态快照区分 native context 清理和输出所有权，取消、失败不会因为清理完成而变为成功。

[InProcessTaskService](vane/execution/direct_exchange.py) 是内部契约设施，接收 pipelined RayQuerySpec，按图预先检查任务上下文数量、exchange 窗口总额和 result 窗口，再准备所有任务、固定 split 分配和封闭通道成员。Python 只处理元数据、控制与测试结果查看；fragment 间的数据不经过 Python，也不调用编译器的物化执行测试入口或旧 runner。根结果采用相同的 native channel，测试用 DirectBatch 支持显式关闭和保留切片。

P1.2 的预算保证限定为通道拥有的实际值缓冲；operator 输入/状态属于独立内存域。进程内传递无需 Arrow 编解码 staging。此设施不实现跨查询的分布式资源池、Flight、动态 split/routing 更新、worker epoch 或公开 Ray QueryResult；这些分别在 P2/P3 接线。原生 operator 的完整预算与能力扩展继续按后续阶段实现。local 公开入口仍直接执行原生查询。

### P2 已实现的跨进程数据面与 Ray 调度

[direct_flight.cpp](src/vane_py/execution/direct_flight.cpp) 实现独立的 native Flight 服务，与旧 shuffle server 没有 ticket 或执行路径适配。传输对象先获得有限的 link 数量和 staging 预留，publish/subscribe 消耗这些额度。每条路由绑定完整的 query、attempt、双方 worker epoch、exchange、分区、schema 指纹、routing version 和随机 capability；服务按已注册 ticket 的完整字节匹配，日志与错误不回显凭据。

DoGet 首先发送固定 Arrow schema，然后传输带 `D:sequence` 元数据的 RecordBatch，最后发送零行的 `F:last_sequence` 并关闭数据流。没有 FINISH 的 EOF、重复/跳号、类型或帧上限不匹配均失败。每条流第一版只允许一个未确认帧；服务端保留 DirectBatch lease，直到独立 DoAction 收到累计 ACK。ACK 可以重复，超过已发送位置则失败；流不支持重连重放。消费者把收到的批次复制到已计费的 native input channel 后释放接收 staging，并 ACK 上游，所有权由发送窗口转入接收窗口。

每个 link 的 staging 上界预留为 `16 * frame_bytes + 256 KiB`，覆盖 Arrow/native 编解码、IPC payload 与传输帧；gRPC 接收消息限制为 `4 * frame_bytes + 64 KiB`。native 帧仍由 DirectLimits 精确计费，元数据受最多 256 列、ticket 长度和 link 数限制。库的连接管理、线程栈以及 DuckDB 算子属于各自资源域，不把该预留解释为进程总 RSS。I/O 与 native 计算独立；每个订阅有有限的读取和控制线程，控制动作不等待数据额度，服务关闭有强制终止活动 RPC 的截止时间。

数据流 FINISH 后控制检查继续存在，持续传播上游通道的持久错误。DirectTaskService.production_status 直接读取 native 错误记录及全部输入/输出通道，无需等待 pump 的操作锁。协调器的周期监控、执行期限探测和最终 EOF 核查均使用该入口，检查全部 worker 和结果服务。详细 task status 会等待执行锁，仅用于诊断；合法的长时间 native 执行不能因此被状态 RPC 期限误判为失效。查询失败会唤醒正在等待结果容量的客户端。已有失败、取消和执行/交付超时保持各自的结局。

[pipelined_worker.py](vane/execution/pipelined_worker.py) 的 Ray actor 只接收计划、固定 split/routing 和控制信息。服务共享的 worker 池按显式 CPU/memory 资源放置，不启用 actor restart 或方法重试；每个 worker 有不可复用的 epoch。每个查询获得独立 native database、TaskService 和 Flight 服务，operator memory 按 max_active_queries 分配固定份额，上下文、exchange/staging 字节及 link/I/O 数由 worker 账本跨查询计费。第一版在 prepare 一次分配并封闭所有 split 和通道成员；后续动态扫描与路由版本扩展保持显式协议。

[pipelined_runtime.py](vane/execution/pipelined_runtime.py) 的 PipelinedScheduler 将整张图作为活动组，按任务固定分配到 worker。先准备全部任务及结果上下文，再绑定输入、验证客户端及 worker 的 Flight 握手，最后逆拓扑启动消费者和生产者。任一准备失败都会向原 worker epoch 发起新的 release，确认原生清理后撤销整组预留；release 失败保留重试所有者。独立 native pump 推进已经启动的任务，Ray RPC 不搬运 RecordBatch。

结果服务是一个常驻 Ray actor 内的 native relay，数据路径为 root worker → ResultService → 客户端 native channel → QueryResult。两跳各自拥有窗口和 staging。每条查询拥有独立 ResultContext；服务 max_results 和会话 max_results 同时限制尚未释放的结果数量，已生产完成但仍持有的结果继续占位。交付 IPC 缓冲受服务与会话 result_buffer_bytes 双重约束，导出的 Arrow/NumPy 视图通过 BatchLease 持续计费，关闭会话后也不会提前归还这些字节。

应用使用 `with vane.Runtime(RayResources(...)) as runtime` 明确持有服务。构造 Runtime 不启动 Ray；第一次 connect 建立进程内 QueryService 核心并注册 Session，第一次分布式查询按需启动 worker 及唯一 ResultService actor。后续 Session 和查询复用这些进程。Session.close 取消、清理本会话查询，保留服务和其他 Session；Runtime.close 禁止新会话与查询，清理所有 Session 后停止共享进程。清理失败保留所有者与额度，调用方可重试 close。没有隐式全局 Runtime、每会话 actor 池或自动重建结果服务的路径。

Session 还持有对应的原生连接，并在 Session 层独立索引所有 native cursors；索引使用弱引用，允许不再被使用的中间 cursor 正常回收。关闭 Session 时禁止继续创建 cursor，持有存活连接的强引用并逐个关闭，因此中间节点回收不会遗漏嵌套 cursor。最后一个原生连接释放数据库和查询资源后才注销 Session；Runtime 确认注册表为空后才停止共享进程并标记关闭。worker 或结果服务暂时不可达时，保留清理所有者和配额；只有 release 确认成功或 Ray 明确报告 ActorDiedError 才完成该资源的清理。

创建/准备 RPC 的失败 ObjectRef 不会随 actor 恢复而变为成功，清理不等待旧回执，且每次重试都发起新的 release RPC。结果服务创建、pipelined prepare 和 FTE prepare 共用以下规则：调用方在发送前记录每个 actor epoch 内连续递增的序号；release 在注册表锁内先封闭该序号，再等待已注册 owner 的原生资源关闭。迟到的创建/准备调用在注册原生状态前检查该序号，因此 release 先到也不会在确认后重新留下 context。已释放序号按连续区间合并，区间间隙只对应尚未完成清理的调用，不逐条保留历史查询。新的 release 仍不可达或超时时，Session、查询配额和结果槽位继续由原调用方持有。

`Runtime.close(timeout=...)` 从入口建立单一单调时钟截止时间，取消、RPC 等待、线程退出、存储清理和原生连接关闭共用剩余预算。每个服务最多执行一个关闭任务；即使原生操作或 RPC 提交暂时阻塞，调用方也会在等待期限内收到 TimeoutError。超时不注销尚未关闭的 Session，也不丢弃其资源和关闭任务；后续 close 等待正在进行的任务，或在它结束后用新的剩余预算重试，不并发释放同一资源。

[runtime.py](vane/execution/runtime.py) 中的 QueryService 持有 Session 注册表、worker 池、全局准入、结果预算和存储注册；RayQueryRuntime 负责一个 Session 的 SQL 查询上下文及配置。准入只使用一个服务队列，原子检查全局和会话的 active/queued 上限，保持会话内 FIFO；一个会话已满时允许其他会话使用空闲容量。会话快照报告自身占用，执行/排队历史计数位于 Runtime 服务快照。ResultServiceClient 只持有同一个结果 actor 及尚未释放的创建 RPC，不缓存空闲 actor。所有结果控制调用使用唯一 query_id；迟到 cancel/release 幂等，其他已关闭上下文调用报错。已进入的 RPC 持有其原 ResultContext，无法访问其他查询。

Flight、channel、物化读取器和存储 lease 按查询创建和关闭。FTE watchdog 退出后才释放上下文额度；失败的上下文保留在注册表，其他上下文继续服务。Ray 为常驻结果 actor 预留 max_results 份容量，每份包含两个结果窗口、两条 Flight 链路 staging，配置 FTE 存储时还包含物化读取 staging。这是传输资源预留，不代表进程总 RSS。该进程丢失会使其中所有活动结果失败，随后查询也报告服务不可用；调用方关闭 Runtime 后创建新的 Runtime，不自动重放结果或替换进程。worker 的 FTE attempt 重试仍按原协议执行。

P5.2.3 交付应用内的服务核心：规划、协调器、FTE 续租在应用进程，结果服务和执行 worker 在 Ray 进程。P5.2.4 已按[独立服务设计](SERVER_DESIGN.md)实现 Flight 独立进程、会话鉴权与租约、远程查询控制及原生结果交付，并完成单机故障验收。会话核心不依赖线协议，后续 DuckDB 2.0 升级时再接 Quack；当前不实现兼容或 fallback。local 继续直接执行原生查询。

## Native 算子的异步推进

### Source 的等待与唤醒

source 无数据可取时返回 SourceResultType::BLOCKED。检查状态、注册等待者和重新检查形成同一同步协议，防止数据恰好到达时丢失唤醒。

唤醒回调只重新调度任务，不持有 channel 锁执行 DuckDB 或 Python 代码。数据、取消、错误、消费者关闭和 deadline 都能够唤醒等待者。回调引用有生命周期保护，task 释放后不能再访问裸指针。

### Sink 的部分提交

sink 返回 SinkResultType::BLOCKED 后会再次尝试处理尚未完成的输入。因此它需要保存当前 batch、分区选择及已提交游标。

例如分区 0 已入队而分区 1 没有容量，恢复时只发送剩余部分。重新发送整个 chunk 会产生重复行，不能用接收端去重掩盖。

异步队列持有稳定的 owned batch，不能持有一次 Sink 调用的临时 DataChunk、selection vector 或 Arrow 导出的裸引用。所有权转移或复制均先获得预算。

编码前申请有界 staging 额度，编码结束后结算实际占用。大 batch 按行切分；单行仍超过硬帧上限时明确失败。schema、dictionary 和解码临时分配也需要自己的有界策略。

### 计算结束与输出结束

native finalize 宣布计算结束，并异步完成必要的本地封存，不等待远程消费者释放全部视图。TaskRuntime 可以让出计算线程，由独立输出状态推进 task 的 OUTPUT_PENDING。

同样，物化上传和 manifest 封存不能长期阻塞计算线程。两个 writer 都遵守异步等待契约，模式差异由输出协调器处理。

## 资源与生命周期

### 统一所有者模型

~~~text
ResourceDemand
  cpu_share
  gpu_slots
  task_contexts
  memory_by_domain
  io_concurrency

Reservation
  owner_id
  worker_epoch
  granted_capacity
  deadline

MemoryLease
  allocation_id
  owner_id
  domain
  bytes
~~~

这组类型可以直接替换原有资源字段和预算接口。内存域至少区分 exchange、staging、operator、udf 和 result。Ray object store 不承担新引擎的 exchange 或结果载体。

Reservation 表示获准使用的容量，MemoryLease 表示实际所有权；两者是同一资源的不同视角，不能相加当作实际内存使用量。跨组件转移通过 allocation_id 转移 lease；真实复制会产生新的分配并分别计费。

| 所有者 | 负责的资源 | 释放条件 |
| --- | --- | --- |
| QueryContext | 准入、任务集合、取消、查询级预算 | 执行退出且相关清理已确认 |
| TaskRuntime | 上下文、算子状态、运行中的异步引用 | 计算与异步引用安全结束 |
| DirectExchange | 排队帧、未确认帧和输入窗口 | 确认消费或完成关闭 |
| MaterializedExchange | 上传缓冲、未提交对象、已提交对象 | 对应 attempt、query 或 result lease 结束 |
| ResultService 与 QueryResult | 结果窗口、客户端批次和导出视图 | 最后借用者释放 |
| 模型服务 | 模型实例及设备资源 | 模型服务自己的生命周期结束 |

QueryContext 不继承 LocalModelRequest。模型服务和查询服务是独立所有者；未来接入 UDF 时显式建立子请求和取消关系，无需让整个查询服从模型请求的状态机。

### 硬限与推进预算

worker 的 exchange 容量至少覆盖：

~~~text
producer_owned_bytes
  + receiver_reserved_windows
  + staging_reserved_bytes
  + schema_and_dictionary_bytes
  <= exchange_hard_limit
~~~

同一分配转移角色时不重复计费。每条通道的最大帧不超过接收窗口；每个活动任务另有足以推进至少一个输出步骤的预算，避免输入占满后无法产生输出。

申请失败返回 BLOCKED 或明确的容量错误，不先分配再补记账。第一版不运行中缩减已授予的硬 reservation，不靠超额放行恢复活性。

算子内存沿用或改造 DuckDB 的分配与 spill 能力，模型内存依赖其实际 backend。没有纳入分配器的对象不能被声明为已受硬限保护。该账本不代表整个进程 RSS 或物理 VRAM 的强制隔离。

网络层额外限制连接数、在途消息、预取数和消息大小，并报告应用账本之外的 gRPC 与 socket 内存。流控不能替代应用层预算。

### 混跑

两种调度器共用会话内的 [WorkerResourceManager](vane/execution/worker_resources.py)。它以 contexts、operator bytes、exchange bytes、staging bytes 和 I/O links 为维度，一次预留一个图的全部 worker，或一个 FTE attempt 的目标 worker。actor 独立核对相同需求及容量，作为实际资源所有者的第二层校验。

排队采用 FIFO。单个需求超过 worker 硬限时立即拒绝；只是被其他查询占用时进入可取消队列，不保留部分预留。Pipelined 在准入期限内等待，成功后的预留持续到所有 worker 确认清理。FTE 每个 attempt 清理完后归还容量，再以新的队列位置申请下一批，不能不断抢在已经等待的 pipelined 图前面。重试保持已排队 task 的身份和次序；取消只移除等待项，尚存活的 owner 必须完成清理才能退还容量。第一版不抢占已运行任务；FIFO 可能牺牲一部分利用率来保证先来的图获得完整容量。

Pipelined 的会话名额与整图 worker 资源分两步取得。取得会话名额后仍保持 `ADMISSION_WAIT`，编译、池准备及 worker 排队共用从请求创建时开始的准入期限，不在第二个队列重置期限。整图 worker 预留成功后才进入 `RUNNING` 并启动执行期限，随后 task/Flight 准备属于执行时间。排队超时返回 `RequestQueueTimeout`；执行超时返回 `RequestExecutionTimeout`。计时统计把 worker 等待计入 queue_wait，尚未执行就退出的请求不生成 execution sample；复制出来的旧准入定时器回调不能取消已经开始的执行。FTE 按查询名额启动冻结及阶段执行，已运行查询的 attempt 排队和重试继续计入其执行期限。

FTE 的取消检查、入队和 attempt 发布与取消共享同一调度锁；取消移除等待项后，调度线程不得再次为该查询入队。`close()` 等待调度线程退出后再完成资源回收，后续查询不受已取消查询的队首等待项阻塞。

FTE 重试同样消耗准入和预算，不能成为不受限的额外任务。高优先级也不能突破硬内存上限。CPU、公平性、native 内存和慢结果消费均纳入混跑验证。

## 统一结果交付

### ResultService 与 QueryResult

QueryResult 提供 schema、批次迭代、collect、close 和状态查询。P1.1 通过 execution_state 与 state 分别观察执行和交付；P2 根据全图生产状态停止执行期限，并在最终 EOF 前核实分布式结局。P3 的 FTE 在 ResultManifest 提交后停止执行期限，交付期限继续有效。提交请求返回结果句柄，不等待所有任务完成；FTE 的首批读取等待 ResultManifest 发布，pipelined 可以读取运行中任务的结果。

分布式结果使用原生 ResultService，部署在客户端可访问的查询服务端点：

~~~text
pipelined 根通道或 FTE ResultManifest
  → native ResultService
  → 有界 Flight 结果流
  → QueryResult 的 native reader
  → 带 BatchLease 的 Arrow batch
~~~

ResultService 是有界的终端消费者，负责查询结果的稳定端点、流状态和所有权。它与 QueryCoordinator 可以在同一进程，但数据面在 C++。不通过 Python batch 中转、ray.put、RayMaterializedResult 或旧结果包装。

ResultService 需要能够访问 worker 的数据端点以拉取根输出，客户端只需要能访问结果服务的公开端点，不要求客户端直连每个内网 worker。这个额外 hop 是网络部署的选择；第一版统一走该路径，不再增加另一套客户端直连模式。local 使用同一结果契约的进程内实现，无需启动网络服务。

ResultService 的输入与输出都受预算约束，必须在拉取下一批前取得容量。交付窗口耗尽时停止读取上游。ResultService 本身进入 pipelined 活动组的资源计算，不能在任务启动后才发现根结果没有消费者。

### BatchLease

查询身份与结果借用身份分开。执行结束后，Arrow 或 NumPy 导出视图可由独立 BatchLease 维持有效性，无需保留已经结束的 task。

零拷贝借用持续计费，不能在 iterator 前进或 QueryResult.close 时提前退还仍被视图占用的内存。复制和序列化重叠时，实际同时存在的缓冲分别计费。

collect 需要把批次复制或明确转移到用户收集容器，及时释放传输窗口；不能一边保留所有受限 lease，一边等待新的传输额度。用户主动收集的完整容器有独立的内存责任，不承诺固定占用。

### 执行状态与交付状态

execution_completion 记录计算与输出提交的结局；结果交付有自己的 EOF、错误和关闭状态。FTE 可以在客户端尚未读完已提交结果时执行成功，结果对象由 result lease 继续持有。pipelined 的输出排空依赖实际 ownership 转移，不能在输出丢失后宣称执行成功。

pipelined 调用方需要持续消费或明确关闭结果，不能将等待 execution_completion 当作开始读取的前提，否则有限窗口可能一直等待客户端。collect 必须主动驱动消费；观察完成状态本身不隐式收集结果。

QueryResult 只有在结果源已正常结束且执行状态成功时才返回正常 EOF。此前交付过批次、随后查询失败，下一次读取报告失败。执行已成功后出现客户端网络错误仍要报告交付失败，不将其伪装为完整结果，也不重跑整个查询。

LIMIT 达到表示特定消费者不再需要输入。scheduler 依据算子状态记录预期取消，只停止已无消费者的上游；它不伪造 source EOF，也不把任意 task 取消当成查询成功。影响结果正确性的真实失败不能被迟到的 LIMIT 通知覆盖。

### 关闭与超时

用户 close、迭代器提前关闭、session 退出和 owner lease 到期停止不再需要的工作。取消传播到 ResultService、exchange、native pipeline 和已接入的 UDF 子请求。

准入超时从申请容量开始；执行超时从获得准入开始，覆盖准备、计算、背压和 FTE 重试；交付超时从结果句柄就绪开始，覆盖等待首批和整个消费期，不按 batch 重置。FTE 的交付等待包含其物化阶段，配置应明确这一点。

进程内使用绝对 monotonic deadline，跨机器传递剩余时长，不比较不同机器的 monotonic 时间戳。关闭或 deadline 到期禁止继续发布尚未提交给调用者的新 batch。

## 失败与清理

| 情况 | pipelined | FTE |
| --- | --- | --- |
| 计划、类型、协议或存储能力不支持 | 启动前拒绝 | 启动前拒绝 |
| Prepare 部分成功后失败 | 撤销本组预留，有限重新准入或终止 | 撤销相应任务预留，有限重新准入或终止 |
| worker 丢失 | 查询失败 | 按提交状态与存储保证恢复未提交 attempt |
| 数据连接不可恢复地中断 | 查询失败 | 读取已提交对象的操作可有限重试，不能消费部分未提交输出 |
| 错误帧、schema 错误、确定性算子错误 | 查询失败 | 查询失败 |
| 内存硬限不足 | 查询失败 | 查询失败，不反复重试相同容量的 attempt |
| coordinator 丢失 | 查询失败 | 查询失败 |
| ResultService 丢失 | 结果交付失败，终止仍在执行的查询 | 相同处理；第一版不提供跨结果服务续传 |
| 用户取消或 deadline 到期 | 唤醒等待并清理 | 停止重试，唤醒等待并清理 |

清理按照禁止新工作、关闭消费需求、停止生产、唤醒等待、确认计算与 I/O 退出、释放真实所有权的顺序推进。导出视图和已交付结果继续由独立 lease 管理。

关闭 RPC、native interruption 和 Python 对象析构不在 coordinator 或预算锁内执行。控制队列和数据队列分开，数据满载不能阻止取消。

清理动作幂等；超时则记录 CleanupPending、保留资源 owner 并允许重试，不能直接把缓冲计数清零。根因和清理诊断分别记录，清理错误不能覆盖最初的执行错误。

模式不会在失败后自动改变。特别是 pipelined 不能因为客户端尚未看到结果就切成 FTE 重跑；内部可能已经执行扩展或外部调用。

## SQL 与类型范围

下表定义新 Ray 执行器的能力矩阵，两种分布式策略分别验收。local 根据原生执行器及查询接口自己的能力验收，不因为 Ray 暂未实现某个算子而人为限制本地 SQL。旧分布式实现曾经支持某个算子不构成新实现已经支持的证据，未完成的分布式能力返回明确错误。

| 能力 | 最小闭环 | 后续第一版目标 |
| --- | --- | --- |
| 常量、空输入、range、文件扫描 | 可在目标 worker 执行的子集 | 扩展扫描与快照能力；FTE 额外要求重放 |
| filter、projection、GATHER、HASH | 两种模式共同支持 | 扩展类型与分区规则 |
| UNION ALL | 基础图完成后加入 | 多输入公平消费和独立结束 |
| 分组聚合、hash join | P4 支持，具体函数/连接类型见上表 | 扩展 grouping sets、其他聚合及非等值连接 |
| BROADCAST | P4 自动选择小 build 广播；消费者独立关闭 | 基于实测调整代价模型 |
| LIMIT | P4 全局 LIMIT/OFFSET，支持提前关闭 | 分布式 LIMIT pushdown 优化 |
| ORDER BY、TopN | P4 单根全局排序及 partial/final TopN | 排序性能和 spill 基准 |
| window、ASOF、MARK、delim 等复杂算子 | 明确拒绝 | 按语义和分布式计划逐项扩展 |
| 递归 CTE 与迭代反馈 | 明确拒绝 | 需要独立的反馈执行设计 |
| Python、AI 与 GPU UDF | 明确拒绝 | 按 backend 验证资源、取消、类型及 FTE 重放能力 |
| COPY、DataSink、DML 与扩展写入 | 明确拒绝 | 独立定义写入提交和副作用语义 |

P4 使用 `vane.analytical-types:4`：BOOLEAN、有/无符号整数（有符号包括 HUGEINT）、FLOAT/DOUBLE、DECIMAL、VARCHAR/BLOB、DATE、TIME、各精度 TIMESTAMP、TIMESTAMPTZ、INTERVAL，以及递归 LIST/STRUCT/MAP/ARRAY。嵌套深度不超过 32；native alias/扩展类型、UHUGEINT、TIME_TZ、UNION、ENUM、UUID、tensor、FILE 与媒体类型仍明确拒绝。NULL、空批次和 schema-only 结果保持支持。

DirectFlight、MaterializedIO 与 Ray 结果导出共用递归 Arrow codec，由用途决定 schema。使用 128 位存储的 DECIMAL 在内部交换中使用 `decimal256(39,scale)`，保留原生系数的完整 signed 128-bit 范围；例如内层 SUM 的 39 位结果可供外层 SUM 抵消，不能因声明精度只有 38 位而在中途拒绝。使用更小原生存储的 DECIMAL 仍按 `decimal128(width,scale)` 交换。公开结果中的所有 DECIMAL 保留 SQL 声明的 `decimal128(width,scale)`，包括嵌套字段和空结果。HUGEINT 在交换及公开结果中均使用 `decimal256(39,0)`。宽化通过符号扩展完成，解码先检查 signed 128-bit 边界，保留 scale、NULL 及 LIST/STRUCT/MAP/ARRAY 内的子值；额外缓冲计入既有 staging 预留。

TIME 使用 `int64` 微秒，允许闭区间 `[0, 86400000000]`，因此保留合法的 `24:00:00`。INTERVAL 使用 `struct<months:int32, days:int32, micros:int64>`，在 native Arrow exporter 之前转换，避免微秒乘 1000 溢出；非 NULL interval 的三个分量均不得为 NULL。两种存储表示同样用于公开 Ray Arrow 结果和嵌套子值，调用方可在 SQL 中转为 VARCHAR 获得时间文本。timestamp 保持单位，TIMESTAMPTZ 采用 UTC instant。

旧 profile 不再接受，Arrow codec 源码纳入 fragment build identity。输入 Arrow schema 与 native profile 一致后才解码。列表、map、struct 和定长 array 的子值、有效性位图及非内联字符串均属于 native 帧预算；时间类型转换在私有紧凑向量上进行，有界副本属于已有 staging 预留，计算过程的输入向量及测量临时空间属于 operator/staging 域。

算子检查依据实际物理函数、scan 和 owned subplan，不能只看 SELECT 关键字。阻塞聚合、sort 和 join 可以合法等待完整输入；pipelined 只承诺允许的执行重叠，不承诺每条 SQL 都早产出。

## 模块组织与旧路径删除

### 目标模块

以下目录为建议的目标职责，可在实现时调整文件粒度。第一版直接围绕这些职责组织代码，不新增 v2 命名空间来长期并行维护旧引擎。

~~~text
vane/execution/
  query_options.py          执行目标与不可变查询配置
  query_runtime.py          local QueryContext、会话容量与取消生命周期（已实现）
  api.py                    QuerySpec 与查询入口
  coordinator.py            查询状态与服务生命周期
  plan.py                   FragmentGraph 的 Python 视图
  compiler.py               Ray native 图编译与加载验证
  submission.py             RayQuerySpec、快照及 worker 计划准备
  resource_demand.py         不可变资源声明
  resources.py              资源准入与 reservation
  result_delivery.py        QueryResult 的 Python API 与有界交付（已实现）
  batch_lease.py            Arrow 批次及导出视图的计费所有权（已实现）
  native_cancellation.py    原生中断与 cursor 复用的生命周期隔离（已实现）
  schedulers/
    pipelined.py            活动组与直接执行
    recovery.py             阶段提交与任务恢复
  backends/
    local.py                DuckDB 原生查询与结果流
    ray.py                  Ray worker 放置与控制

src/vane_py/execution/
  local_query.cpp           原生 local query 入口与增量 reader（已实现）
  fragment_plan.cpp         native fragment、连接/source 快照及能力
  fragment_plan_bindings.cpp 编译与准备的 Python 绑定
  task_service.cpp          native 任务服务绑定
  query_result.cpp          批次与 lease 绑定

external/duckdb/src/execution/distributed/
  plan/                     FragmentGraph 与 fragment builder
  runtime/                  TaskRuntime 与执行事件
  exchange/                 DirectExchange 与 MaterializedExchange
  result/                   native ResultService
~~~

Python 管理查询和放置，C++ 负责 fragment、exchange 和数据所有权。每批 worker 数据不经过 Python；API 迭代把客户端 batch 暴露为 Python 对象属于用户消费边界。

### 删除或替换的职责

| 当前实现位置 | 新职责接管方式 |
| --- | --- |
| [PlanRunner](external/duckdb/src/include/duckdb/execution/distributed/plan/runner.hpp) 与产生 task stream 的编排 | FragmentGraph builder 只构图，scheduler 单独执行；删除构图中执行查询的路径 |
| `vane/runners/fte/backend.py`（已删除） 与旧 FTE 控制结构 | TaskService 和 RecoveryScheduler 直接接管，不留下调用旧 manager 的包装器 |
| `vane/runners/local/runner.py`（已删除） 与 [LocalQueryRuntime](vane/execution/local_query.py) 的查询分流 | 统一 local backend 与 QueryContext；模型服务另行拥有模型生命周期 |
| `vane/runners/ray/driver.py`（已删除） 与 `vane/runners/ray/worker.py`（已删除） 中耦合执行模式的部分 | Coordinator、scheduler 和 Ray backend 分担职责；worker 数据执行进入共同 native runtime |
| `vane/runners/ray/partition_metadata.py`（已删除） 与 [Python 结果源](src/vane_py/pyresult_source.cpp) | QueryResult 和 native ResultService 接管公开结果链路 |
| [result_delivery](vane/execution/result_delivery.py) 与 [local_result_delivery](vane/execution/local_result_delivery.py) 的查询专用包装 | 将有用的 ownership 实现迁入 BatchLease；删除旧查询入口，不继承 LocalModelRequest |
| [ResourceGraph](vane/execution/resource_graph.py) 与 [ResourceVector](vane/execution/resources.py) 的执行关联 | 资源需求进入 FragmentGraph，预算重新定义；保留与查询无关的实用代码需有独立职责 |
| [Flight server](external/duckdb/src/execution/distributed/exchange/flight_server.cpp) 的旧 ticket 分支 | 使用新协议服务 direct 通道；物化读取由新的存储实现负责 |

“替换”针对职责和调用链，不要求无条件删除整个文件中仍有独立用途的代码。被其他受支持模块使用的公共工具可以移动或重新实现；所有调用方同步更新，不能留下只为旧查询接口服务的转换层。

旧内部格式、环境开关、入口别名、文档示例和专有 mock 与对应旧路径一起移除。最终架构的验收包括依赖检查：新引擎不经旧 PlanRunner、旧 FTE manager 或旧结果包装执行查询。

## 实施阶段

### P0 统一契约与纯计划图

定义执行目标与查询配置、Ray FragmentGraph、RayQuerySpec、资源声明和结果 schema。实现无副作用的 Ray fragment builder；明确 local 原生入口与 Ray 策略入口的边界，确定新 API 与序列化格式。TaskSpec、reservation 与结果 lease 在 P1/P2 接入真实生命周期。

退出条件：同一受支持查询能够生成两种模式可用的图；plan round-trip、分区语义、配置隔离和能力拒绝通过。构图不提交任务，不通过旧执行器补全缺失信息。

### P1 原生结果入口与直接通道契约

local 直接连接原生查询与 QueryResult，验证结果、资源与取消。另行实现分布式 TaskRuntime 的进程内测试设施、内存 DirectExchange 和异步 source/sink，用两个及以上 fragment 验证传输契约，不将其接入公开 local 查询。

退出条件：local 不进入分布式规划或调度，能够交付原生增量结果；分布式通道测试中下游在上游完成前消费，极小窗口、部分 sink 提交、空输入、取消和导出视图均通过。

### P2 Ray pipelined 与 native 结果服务

实现 Flight 通道、Ray backend、活动组准入和 ResultService。部署客户端可访问的结果端点，验证 worker 间及结果链路的数据面均不通过 Python 中转。

退出条件：两个 worker 的 scan/filter/HASH/GATHER 查询可运行；慢客户端产生有界背压；worker、数据连接或结果服务失败能够明确终止；ray pipelined 入口使用完整新调用链。

### P3 在共同核心上实现 FTE

实现 MaterializedExchange、存储提供者、AttemptManifest、StageManifest、RecoveryScheduler 和 ResultManifest。Ray FTE 与 Ray pipelined 使用同一 TaskRuntime 和 QueryResult；不实现 local FTE。

按可独立审查的增量交付：P3.1 原生对象 I/O 与提交基础，P3.2 不可变文件输入与重放准备，P3.3 恢复调度、结果提交和失联资源回收。P3.1—P3.3 均已接通并按 roadmap 验收；后续 SQL/type 扩展属于 P4。

退出条件：提交前后 worker 丢失、下游失败、重复提交、迟到 attempt、对象缺失和重试耗尽均有可控验证；恢复不会重复暴露行；生产 worker 退出后可从独立存储读到已提交输出。

P3 不能通过委托旧 FTE 引擎完成。新的任务服务、计划格式、结果协议与 pipelined 共用，是本阶段的核心验收条件。

### P4 分析能力与混跑

实现按四部分落地：native partial/final 聚合；等值 hash join 与广播及 BuildReady；全局排序、TopN 和 LIMIT；递归类型传输、共享 FIFO worker 准入与诊断。两种 Ray 策略执行同一份图，验收覆盖与 native SQL 对照、空输入/NULL/倾斜、小窗口、重复执行以及 FTE worker 故障重放。

`QueryResult.diagnostics()` 返回 query_id、执行与交付状态、清理阶段、session 容量及执行模式。Pipelined 报告各 worker 的 task、BuildReady、输入/输出等待原因及每条 channel 的队列/借用计费；FTE 报告 attempt/fence、提交历史、存储配额、task/channel 状态。独立的 native `TaskService.diagnostics()` 不取得执行锁；诊断 RPC 失败仅记录 unavailable，不能触发查询取消。结果关闭后保留最终诊断，活动查询与结果 lease 的共享预算仍可从 session snapshot 查看。

退出条件：SQL 对照、空输入、NULL、倾斜、低容量活性、多消费者及两种模式混跑通过。AI/GPU UDF 按独立 backend 矩阵扩展，不作为纯 SQL 闭环的隐含前提。

### P5 删除旧入口并完成发布验收

清除被替代的 runner 分流、任务协议、结果包装、配置别名和文档，更新所有受支持调用方。重写依赖旧内部实现的测试，保留并扩展其有效语义场景。

退出条件：所有公开查询入口只进入新引擎，旧调度器和结果路径不在运行依赖中；对应 release gate 通过；提供新 API、支持矩阵、可复现基准和数值默认值依据。

各阶段按新能力组织可审阅的变更。开发期间尚未删除的旧源码只能作为参考，不能成为新实现的执行依赖；阶段性原型不代表完整双模式已经交付。正式发布以新架构和声明的能力范围验收，不以旧接口继续可用为条件。

分支仍基于 feature/local-runtime，使用其中可取的实现经验和代码。基础分支合入 main 后再调整分支基线；继承提交历史不产生 API 或模块结构的兼容承诺。

### P5.1 公开入口与删除边界

| 入口 | local（默认） | ray |
| --- | --- | --- |
| `connect()` / `connect(resources=...)` | 本地原生连接；query 运行时按需建立 | `Runtime(...).connect()` |
| `query(SELECT, ...)` | QueryContext/QueryResult，原生流式结果 | 新 fragment compiler 和 pipelined/FTE runtime |
| `execute` / `executemany` / `sql` / `from_query` | 原生 SQL、命令及 Relation | 在执行或绑定前明确拒绝，使用 query |
| Relation 终端、DataSink | 原生本地执行，保留参数、事务和模型生命周期 | 当前分布式支持范围不包含此入口 |
| `configure_local_runtime` / `register_model` | 独立的本地模型服务；与普通 query runtime 互斥 | 拒绝 |

删除全局 runner 单例、Python `vane.runners`、`ray_cxx` 计划执行对象和旧结果分区桥接。UDF 共用的环境隔离、远程等待和资源协议位于 `vane.execution`，不导入旧 scheduler。模型原生图观察不改变 UDF payload 或 scan identity。

`VANE_RUNNER` 不再决定连接或查询行为，`configure(runner=...)` 等旧执行配置不再接受。`connect(":default:")` 只能获取已有连接；任何显式 backend/resources/execution 配置均拒绝，避免重置会话计费。Ray 的 SQL 参数、模型 UDF、媒体类型及写入仍在当前支持范围之外；失败不会转交其他后端。

## 验证计划

### 正确性与活性

| 场景 | 验收条件 |
| --- | --- |
| 上游暂不完成 | 用 latch 控制 producer，pipelined 下游已经收到正确数据 |
| 首批结果 | 允许早产出的查询在最后一个 producer 完成前返回批次 |
| 极小窗口与慢消费者 | 字节不越界，产生等待并在释放后恢复 |
| 部分分区发送 | BLOCKED 后恢复无重复、无丢行 |
| 空输入、空分区、常量查询 | schema 可得，正常结束，结果正确 |
| 暂时无数据与多输入 | 无假 EOF、无忙轮询，其他就绪输入继续推进 |
| broadcast 消费者退出 | 其余消费者仍获得完整数据 |
| join 小配额 | build 能推进；不满足活动组容量时有界拒绝 |
| 全局 LIMIT | 行数正确，只取消不再需要的工作 |
| prepare 与取消竞争 | 无晚启动、无取消后新增交付 |
| 准入容量持续不足 | admission deadline 到期退出，无残留部分预留 |
| deadline 与最后一批竞争 | 不因最后一批到达绕过已过期状态 |
| 导出视图与 close | 视图有效、持续计费，最后释放后账本归零 |
| 清理失败后重试 | owner 与计数真实保留，成功后才释放 |
| 错误帧与错误身份 | sequence、schema、attempt、worker epoch 校验拒绝串流 |
| pipelined worker 丢失 | 查询失败，后续读取报告错误，不启动替代 attempt |
| FTE 提交前 worker 丢失 | 同一逻辑输入重试，失败输出不可见 |
| FTE 提交后 worker 丢失 | 成功对象仍可读，不重跑已提交 task |
| 重复提交与迟到 attempt | 只接受一个成功 attempt，下游不重复消费 |
| FTE 下游失败 | 重放固定 manifest，SQL 结果完整且不重复 |
| 已提交对象丢失 | 明确失败，不将缺失分区当空结果 |
| 结果服务失败 | 交付失败被观察，存活资源有界清理 |
| 混跑与并发配置 | 模式独立，预算共享正确，重试不越过准入 |
| local 边界 | 拒绝显式分布式策略；原生查询不进入 FragmentGraph、TaskService、Ray 或网络 exchange |
| 新调用链边界 | 两种模式共用计划和运行时，不调用旧执行器或旧结果包装 |

测试使用事件、latch、状态版本和有界 watchdog 控制顺序。避免用固定 sleep 或机器相关的延迟阈值充当正确性条件。故障快照在清理前保留。

### SQL 与接口验收

支持的 SQL 与单机 DuckDB 语义对照：无序结果按多重集合比较，显式 ORDER BY 检查顺序，浮点聚合采用明确容差。分别执行 pipelined、FTE 和 FTE 故障注入，不能只对无故障数据路径验收。

旧测试分成两类处理：验证结果、恢复、取消、资源等产品语义的场景迁入新测试；仅验证旧类名、旧参数或旧序列化格式的测试删除或重写。测试目标是新契约，不以重新增加兼容代码使旧 mock 通过。

涉及模型服务但未改变其语义的独立测试继续运行。历史 [local runtime 启动超时](LOCAL_SERVING_ACCEPTANCE.md#historical-model-entry-timeout-investigation-status) 仍需保留故障诊断；新执行器的一次成功不能证明历史根因已经解决。

执行代码修改后按 [Python 测试工作流](DEVELOPMENT.md#python-tests) 先验证受影响测试，再运行 base release gate。完整 fast suite 采用仓库 launcher；新测试直接调用 ray.init 时标记 real_ray 和 ray_cluster_owner，依赖 CUDA 时额外标记 gpu。

native 部分按 [C++ 测试流程](DEVELOPMENT.md#native-c-tests) 验证，并采用非 editable 的增量安装。文档阶段仅检查链接、结构和差异，不将尚未运行的执行测试写成已通过。

### 性能实验

分别报告冷启动、预热 worker 和输入已缓存条件，测量：

1. scan/filter/project 与 LIMIT 的首批延迟、总耗时和提前停止效果。
2. 聚合、hash join、ORDER BY/TopN 的计算与传输重叠，以及必要的算子阻塞。
3. 大批 FTE 与短 pipelined 查询混跑的公平性、吞吐和尾延迟。
4. 慢客户端与长期保留 Arrow 视图时的内存和取消回收。
5. FTE task 重试的额外 I/O、重算量和完成耗时。
6. native ResultService 的额外 hop、编码副本和 CPU 开销。

记录输入规模、计划、分区数、预算、线程数、提交版本、冷热条件与失败数。报告首批 p50/p95/p99、总耗时、吞吐、CPU、峰值内存、网络字节、物化字节、算子 spill 和清理耗时。

pipelined 的跨 fragment 物化字节应为零，native 算子仍可按其算法 spill。降低物化 I/O 不能单独证明延迟改善。性能收益与数值默认值由实验决定。

## 观测

QueryStatus 显示实际 execution、backend、图身份、资源准入和终止原因。TaskStatus 显示 attempt、worker epoch、计算状态、输出状态及 blocked_reason。

至少提供 planning、admission、prepare、first output、first client batch 和 total 时间；活动上下文和 runnable 数；各内存域的 reservation 与 ownership；打开、结束、放弃和失败的 channel 数；FTE 重试数、提交数和存储字节；结果借用与 CleanupPending 所有者。

超时快照包含依赖图、活动组、任务与通道状态、等待原因、预算持有者和最近事件版本。指标采用 native 聚合与有界周期上报，不为每行或每个微小 chunk 发 Python RPC。

## Trino 参考与当前代码依据

### 从 Trino 借鉴的边界

| 固定版本源码 | 对本设计的依据 |
| --- | --- |
| [SqlQueryExecution][trino-query] 与 [QueryScheduler][trino-scheduler] | NONE/QUERY 使用 PipelinedQueryScheduler，TASK 使用 EventDrivenFaultTolerantQueryScheduler；共同生命周期不要求合并内部调度状态机 |
| [LazyExchangeDataSource][trino-source] 与 [LazyOutputBuffer][trino-buffer] | 统一入口可以绑定 direct 和 spooling 两类实现，二者有不同的协调生命周期 |
| [PhasedExecutionSchedule][trino-phases] | 流水执行仍需要 build/probe 等依赖与阻塞推进 |
| [ExchangeSink][trino-sink] 与 [HttpPageBufferClient][trino-http] | 异步等待和 token/确认可用于实现有界数据所有权 |
| [FTE scheduler][trino-fte] | 满足条件时也可提前调度下游；是否阶段重叠不定义恢复边界 |

上述源码支持的是职责划分和协议原则。本设计的新 API、local 原生入口、Ray 双策略、严格阶段 FTE、Flight 传输及 ResultService 是 Vane 的设计选择，不是对 Trino 具体实现的复制。

Trino 的[配置文档][trino-docs]仍指出同集群模式切换未测试，并建议分离大批任务和短查询。因此 Vane 的按查询混跑需要自己的正确性、隔离和性能验收。

### 当前实现提供的证据

当前 [FlightExchangeSink](external/duckdb/src/execution/distributed/exchange/flight_exchange_manager.cpp) 在 Finish 封存并发布 attempt 输出；[RepartitionNode](external/duckdb/src/execution/distributed/pipeline_node/shuffles/repartition_node.cpp) 可以按成功 task 增量发布 handle，但不能据此读取尚未提交的中间 chunk。

[RemoteExchangeSink](external/duckdb/src/execution/operator/exchange/physical_remote_exchange_sink.cpp) 和 [RemoteExchangeSource](external/duckdb/src/execution/operator/exchange/physical_remote_exchange_source.cpp) 存在同步 WaitUnblocked；[FteSplitQueue](external/duckdb/src/include/duckdb/execution/distributed/plan/fte_split_queue.hpp) 与 [pipeline executor](external/duckdb/src/parallel/pipeline_executor.cpp) 可用于研究正确的 native 阻塞和唤醒边界。

local-runtime 的 [资源图](LOCAL_MODEL_RUNTIME.md#native-resource-graph) 是 structural_only，不能直接执行；[managed native streams](LOCAL_MODEL_RUNTIME.md#managed-native-result-streams) 提供了结果借用、预算和清理经验。这些证据说明需要哪些能力，不要求新引擎继承原有类、接口或分流方式。

## 第一版之后

后续优化按实际证据推进：更细的 pipelined 活动组、FTE 阶段重叠、更多类型与 UDF、结果服务复制减少、自动选模式，以及写入提交。

QUERY 重试、混合直接边与物化边、动态并行度和 coordinator 恢复分别需要新的恢复语义。不能仅增加一个枚举值就宣称已支持；尤其必须回答哪些消费者已看过数据、哪些算子状态需要丢弃及哪些副作用可重放。

[trino-revision]: https://github.com/trinodb/trino/commit/6ead7e6c2f04c0bcfe5caf8e938dc1f5d3344f31
[trino-query]: https://github.com/trinodb/trino/blob/6ead7e6c2f04c0bcfe5caf8e938dc1f5d3344f31/core/trino-main/src/main/java/io/trino/execution/SqlQueryExecution.java#L536
[trino-scheduler]: https://github.com/trinodb/trino/blob/6ead7e6c2f04c0bcfe5caf8e938dc1f5d3344f31/core/trino-main/src/main/java/io/trino/execution/scheduler/QueryScheduler.java
[trino-source]: https://github.com/trinodb/trino/blob/6ead7e6c2f04c0bcfe5caf8e938dc1f5d3344f31/core/trino-main/src/main/java/io/trino/exchange/LazyExchangeDataSource.java#L123
[trino-buffer]: https://github.com/trinodb/trino/blob/6ead7e6c2f04c0bcfe5caf8e938dc1f5d3344f31/core/trino-main/src/main/java/io/trino/execution/buffer/LazyOutputBuffer.java#L174
[trino-phases]: https://github.com/trinodb/trino/blob/6ead7e6c2f04c0bcfe5caf8e938dc1f5d3344f31/core/trino-main/src/main/java/io/trino/execution/scheduler/policy/PhasedExecutionSchedule.java
[trino-fte]: https://github.com/trinodb/trino/blob/6ead7e6c2f04c0bcfe5caf8e938dc1f5d3344f31/core/trino-main/src/main/java/io/trino/execution/scheduler/faulttolerant/EventDrivenFaultTolerantQueryScheduler.java#L1309
[trino-sink]: https://github.com/trinodb/trino/blob/6ead7e6c2f04c0bcfe5caf8e938dc1f5d3344f31/core/trino-spi/src/main/java/io/trino/spi/exchange/ExchangeSink.java
[trino-http]: https://github.com/trinodb/trino/blob/6ead7e6c2f04c0bcfe5caf8e938dc1f5d3344f31/core/trino-main/src/main/java/io/trino/operator/HttpPageBufferClient.java
[trino-docs]: https://github.com/trinodb/trino/blob/6ead7e6c2f04c0bcfe5caf8e938dc1f5d3344f31/docs/src/main/sphinx/admin/fault-tolerant-execution.md
