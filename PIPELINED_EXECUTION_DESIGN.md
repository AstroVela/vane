# Vane 统一执行架构设计

本文定义同时支持 pipelined 与 FTE 的新执行架构，面向 Vane 执行引擎维护者。设计从两种执行方式的语义出发，允许直接替换已有 API、计划协议、调度器和结果接口，不承担已有代码的兼容责任。

local 直接使用 DuckDB 原生执行，不选择 pipelined 或 FTE。ray 的两种分布式策略共享一套 FragmentGraph 和 TaskRuntime，分别采用直接 exchange 或物化 exchange。两条执行路径共享查询身份、资源、取消和结果所有权契约。新的 Ray FTE 也在新架构中实现，不通过包装旧 PlanRunner 获得。

| 项目 | 基线 |
| --- | --- |
| 状态 | 目标设计；实现与验收进度见实施 roadmap |
| 日期 | 2026 年 10 月 4 日（P2 更新） |
| 开发分支 | feat/native-flight-exchange |
| 基础分支 | integration/pipelined-execution |
| Vane 参考提交 | 31191cae217d（PR #944 合入） |
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

PIPELINED 只执行一个 attempt。FTE 按显式失败分类和重试上限创建后续 attempt。FteOptions 仅进入 FTE 查询快照，向 pipelined 查询传入重试参数时直接报配置错误。连接可以预先登记 exchange_store，供后续 FTE 查询使用；pipelined 查询不使用该存储。

当前 [query_options.py](vane/execution/query_options.py) 实现 LocalExecution、RayExecution、FteOptions 和 QueryExecutionOptions；[submission.py](vane/execution/submission.py) 实现内部 RayQuerySpec。exchange_store 目前只是注册存储的名字，P3 必须在实际准入前解析并验证其可用性与故障域。RayQuerySpec 覆盖计划准备所需的配置、快照和资源声明，尚不代表已预留资源或提交任务。公开连接入口继续按 roadmap 接线；local 不使用 RayQuerySpec。

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

local.query 只接受自动提交下的单条只读 SELECT；命令使用 execute。支持位置/具名参数、原生算子与批次输出，模型 UDF 尚未接入。query 不读取 VANE_RUNNER，不构建 FragmentGraph，也不创建 LocalModelRequest。local 的 connect/query 拒绝任何 execution override，包括显式 None。未指定 backend 的既有连接尚未切换；它们调用 query 会报错，惰性 Relation 使用 sql 或 from_query，不保留旧 query 别名。P2 接通下述 Ray pipelined 入口。

QueryResult 暴露 schema（Arrow schema）、query_id、context、read_batch/迭代、collect、cancel、close、execution_state 和交付 state。read_batch 返回 RecordBatch，正常 EOF 抛出 StopIteration，部分交付后的 native 错误继续抛出。collect 只收集剩余行，逐批复制到调用方内存并释放传输 lease。关闭结果或连接后，已经导出的 Arrow 切片、NumPy 零拷贝视图仍可读取并持续占用预算，直到最后一个视图释放。

当前默认 rows_per_batch=2048，资源与期限默认值如上例；这些是初始功能配置，性能验收后再调整。result_buffer_bytes 只限制结果交付持有的 IPC 缓冲，不包含 DuckDB 算子、native 预取缓冲或 collect 的完整副本。超过窗口的单批立即报容量错误；能够放入窗口的下一批等待旧 lease 释放，可由取消或期限唤醒。需要控制 native 预取时使用连接的 streaming_buffer_size 设置。查询与交付期限分别从准入及结果句柄就绪开始计算，慢消费期间二者都可能到期。

P2 接通以下 Ray pipelined 入口。先连接 Ray 集群，RayResources 定义会话共享的 worker 池及结果容量；相同连接的 cursors 共用这个池和准入账本。

~~~python
import ray
import vane

ray.init()  # 也可以连接已经运行的 Ray 集群。
resources = vane.RayResources(worker_count=2, partitions=2)
with vane.connect(backend="ray", execution="pipelined", resources=resources) as connection:
    with connection.query("SELECT range AS value FROM range(10000) WHERE range > 10") as result:
        for batch in result:
            print(batch)
            del batch
~~~

当前入口支持 P0 已验证的只读 SQL 子集：常量、range、普通本地 Parquet、filter、projection 与 GATHER；HASH 图可通过 FragmentCompileOptions 验收。文件必须使用所有 worker 可访问的绝对路径，prepare 和 start 均校验快照。整数、浮点、布尔、字符串及 NULL 在 native Flight 中传输，空结果保留 schema。单次 query 可显式传入 execution="pipelined"；FTE、SQL 参数、聚合/join、模型 UDF 等尚未接线的能力明确报错。FTE 的公开执行由 P3 接入。

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

同一查询内使用一致的哈希、类型、NULL、排序和 collation 规则；分区函数由 native 实现提供。不能在 Python 与 C++ 分别计算近似分区规则。排序要求需要进入端口和结果描述，异步到达顺序不构成 SQL 顺序保证。

### Native 编译与加载的第一步

内部入口 compile_fragment_graph 接收连接、SQL、query_id 和不可变 FragmentCompileOptions。选项只有分区数及可选的 HASH 输出列位置，不包含执行策略、worker 地址或 exchange store。连接锁覆盖读取绑定与优化设置的过程；编译器直接调用 Parser、Planner/Binder、Optimizer 和 PhysicalPlanGenerator，不创建 executor 或提交 task。需要准备提交时使用 prepare_ray_query，它在同一连接锁内完成快照与构图。

[native 编译器与加载器](src/vane_py/execution/fragment_plan.cpp) 的首个子集为单条无参数只读 SELECT：常量、整数 range/generate_series、显式 Parquet scan、filter 和 projection。标量函数先按已验收的内置函数集合检查，用户函数在常量折叠前拒绝；聚合、连接、排序、LIMIT、相关子查询及扩展类型按后续阶段实现。普通本地查询继续由原生查询入口处理，不调用该编译器。

解析树检查之后，编译器通过当前 Planner 的 catalog lookup callback 校验实际解析到的函数和宏必须为内置项。`current_user` 等 SQL value function 在解析时可能是列引用，绑定时才转为函数；这一检查发生在宏展开、表函数参数求值之前，并由子 Binder 继承，覆盖限定名、嵌套表达式和表子查询。内置宏间接解析到的用户函数同样被拒绝，不能依赖最终的只读属性检查或事务回滚来撤销 `nextval()` 等副作用。普通列和别名按 Binder 的实际解析结果处理，允许与被覆盖的函数同名；检查只属于这次规划，不改变后续原生查询行为。

绑定后的表达式在优化前按 native `IsConsistent()` 检查：同时拒绝 volatile 和 `CONSISTENT_WITHIN_QUERY` 函数，包括 `CURRENT_TIMESTAMP`、`CURRENT_DATE`、`CURRENT_TIME`、`LOCALTIMESTAMP` 和 `LOCALTIME`。当前提交描述尚未冻结查询时间，不能把单次查询内稳定误当成跨 task/attempt 稳定，也不能依赖可关闭的常量折叠。该限制统一应用于 pipelined 和 FTE；未来支持这类表达式时，需要在提交时冻结查询上下文，并让所有 task 与重试复用。

优化器入口遵循原生查询路径：仅当连接的 `enable_optimizer` 为真且逻辑计划要求优化时调用 `Optimizer::Optimize()`；启用时继续遵循 `disabled_optimizers` 的逐项设置。`PRAGMA disable_optimizer` 因而保留可直接执行的 `IN` 表达式。优化前的表达式、扫描列和数据源依赖校验始终执行。提交快照冻结这一连接选项，worker 准备和每次重放都恢复相同设置。

扫描、过滤和投影先合并到一个 source fragment。并行输出通过 GATHER 进入单分区根结果；显式指定 hash_columns 时，native 按结果列类型绑定 BoundReferenceExpression，生成 HASH 边，再按需要 GATHER。hash_columns 是内部物理分区请求，尚不表示已实现 aggregate/join 的自动分布式规划。HASH 求值和 NULL、多列键合并均使用 DuckDB 原生表达式执行与 DataChunk.Hash。

fragment 使用自己的原生 envelope。普通节点保存不含子节点的 native 算子载荷；子节点和 input port 在 envelope 中显式表达。PhysicalOperator.SerializeNode 提供单节点序列化，无需拆改原计划树。加载器逐节点重建真实 PhysicalPlan，绑定每个输入端口，并验证 schema、算子子集、source capability、split codec 和分片身份。缺少绑定直接失败；不能用空扫描替代尚未接入的 exchange reader。

engine identity 同时包含 DuckDB SourceID 与 Vane 编译器/加载器源码摘要。schema、fragment 和 HASH 表达式分别带版本及身份，加载器在解释 native 算子之前检查身份与完整载荷边界。Python 图里的端口、source 和 source dependency 是原生描述的视图；validate_native_graph 对照 native 解码结果校验，拒绝两者不一致。

range 的扫描 split 由 table function 的 native 回调规划；Parquet 从绑定后的 MultiFileList 枚举文件，使用独立的文件 split codec，不依赖旧 FTE 的 split 管理器。载荷带稳定 split_id、能力和 codec 身份。worker bind 独立持有可移植状态，加载时必须显式提供所分配的 split；空列表代表空任务，未知或重复 split_id 被拒绝。无扫描的常量查询只执行一次，不因请求多个分区而复制结果。Parquet 的 requires_snapshot 标记为真：固定文件列表只封闭了枚举，提交层还必须检查访问条件与回放保证。

编译器在优化前的 logical validation 中捕获每个 Parquet bind 的完整文件集合，作为 source fragment 的 `source_dependencies`。统计信息可把扫描优化成空结果，hive/file pruning 也可移除部分文件，但这些优化仍依赖原始文件。依赖使用同一 native 文件 codec，随扫描节点的移除继续保留；实际 split 分配与并行度仍由优化后的 `sources` 决定。纯空结果的 source fragment 只执行一次，HASH/GATHER 下游不携带源文件分配。

当前 Parquet split 未携带原始文件序号，因此明确拒绝虚拟列 `file_index`。编译器在优化前检查 native 虚拟列 ID，覆盖投影和仅用于过滤的引用，物理计划导出与加载也检查同一限制；普通数据列使用相同名称仍可执行。未来开放该虚拟列时，split 和 scan bind 必须保留绑定时的原始文件索引，不能使用 task 内重新编号的文件列表代替。

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
| 普通 Parquet 文件 | 固定文件集合；要求绝对路径、worker 可见的本地普通文件；准备时核对路径、大小和包含原生亚秒精度的修改时间 | 拒绝，尚无不可变版本保证 |
| 远程文件、其他 scan 或自定义文件系统 | 当前提交 profile 不支持 | 当前提交 profile 不支持 |

source 快照同时覆盖实际扫描和 `source_dependencies`，同一路径的文件元数据只捕获一次。扫描完全被优化掉时仍校验原始文件；部分文件被裁剪时也保留被裁剪文件的校验。普通 Parquet 的绝对路径、访问条件和 FTE 限制应用于全部依赖，不能因优化后的结果为空而绕过。依赖的 capability/codec 纳入 worker 能力检查，文件版本和依赖身份纳入蓝图缓存键；跨进程传输后不能从 Python 图中删掉 native 计划携带的依赖。

本地文件类型、大小和 mtime 来自同一次已打开句柄的 stat。快照同时保存标准微秒时间戳和 `mtime_nsec` 原生小数部分：Linux/macOS 保留纳秒，Windows 保留 FILETIME 的 100 纳秒精度；精度受底层文件系统限制。缺少原生小数部分时拒绝提交。同一秒内的等大小改写以及低于一微秒的 mtime 变化也参与 worker 校验和缓存身份。Parquet 元数据缓存使用的本地文件版本标识同步保留完整时间精度，重新规划可发现更新后的文件统计信息。

Parquet 元数据检查是访问前置条件，不是内容快照：相同大小和修改时间的替换可能无法检测，检查后再修改也不被阻止。调用方须在执行期间保持文件稳定；每次准备 task/attempt 时重新检查。不可变对象版本、snapshot isolation 或受生命周期保护的 staging 输入应随对应 source 接入，再开放其 FTE 支持。文件内容哈希本身也不能提供重试时的旧版本可读性。

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

### 重试与存储故障

| 情况 | 新 FTE 的处理 |
| --- | --- |
| attempt 提交前计算 worker 丢失 | 废弃未提交输出，在可用 worker 重放相同逻辑输入 |
| 已提交输出的生产 worker 丢失 | 从独立存储继续读取，不重跑已经提交的 task |
| 下游 attempt 失败 | 重试该下游，输入仍指向同一批已提交对象 |
| 短暂存储或控制传输错误 | 对幂等操作有限重试；超出上限按错误分类终止或重试 attempt |
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

每帧分配一个独立缓冲，包含对齐后的有效性位图、定长值及长字符串数据，覆盖 P0 的 basic-types profile。写入先测量并检查额度，再复制和发布；暂时无法写入时不分配 payload。原生输入由执行器持有，通道不引用瞬时 Sink 参数。读取构造借用该帧的 Vector，Vector auxiliary 持有 lease，切片及字符串引用继续保留所有权。最后一个引用释放后归还窗口。广播共享一次物理分配，每个消费者独立计费；关闭一个消费者释放其队列，已经借出的视图继续有效并保持计费。

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

每个 link 的 staging 上界预留为 `16 * frame_bytes + 256 KiB`，覆盖 basic-types 的 Arrow/native 编解码、IPC payload 与传输帧；gRPC 接收消息限制为 `4 * frame_bytes + 64 KiB`。native 帧仍由 DirectLimits 精确计费，元数据受最多 256 列、ticket 长度和 link 数限制。库的连接管理、线程栈以及 DuckDB 算子属于各自资源域，不把该预留解释为进程总 RSS。I/O 与 native 计算独立；每个订阅有有限的读取和控制线程，控制动作不等待数据额度，服务关闭有强制终止活动 RPC 的截止时间。

数据流 FINISH 后控制检查继续存在，持续传播上游通道的持久错误。DirectTaskService.production_status 直接读取 native 错误记录及全部输入/输出通道，无需等待 pump 的操作锁。协调器在执行期限到期时使用该入口重新判断生产是否完成，在向用户返回最终 EOF 前再次检查全部 worker 和结果服务；查询失败会唤醒正在等待结果容量的客户端。已有失败、取消和执行/交付超时保持各自的结局。

[pipelined_worker.py](vane/execution/pipelined_worker.py) 的 Ray actor 只接收计划、固定 split/routing 和控制信息。会话 worker 池按显式 CPU/memory 资源放置，不启用 actor restart 或方法重试；每个 worker 有不可复用的 epoch。每个查询获得独立 native database、TaskService 和 Flight 服务，operator memory 按 max_active_queries 分配固定份额，上下文、exchange/staging 字节及 link/I/O 数由 worker 账本跨查询计费。第一版在 prepare 一次分配并封闭所有 split 和通道成员；后续动态扫描与路由版本扩展保持显式协议。

[pipelined_runtime.py](vane/execution/pipelined_runtime.py) 的 PipelinedScheduler 将整张图作为活动组，按任务固定分配到 worker。先准备全部任务和独立 ResultService，再绑定输入、验证客户端及 worker 的 Flight 握手，最后逆拓扑启动消费者和生产者。任一准备失败都会等待已发出的 prepare 结算并撤销整组预留；release 失败保留重试所有者。独立 native pump 推进已经启动的任务，Ray RPC 不搬运 RecordBatch。

结果服务是另一个 Ray actor 内的 native relay，数据路径为 root worker → ResultService → 客户端 native channel → QueryResult。两跳各自拥有窗口和 staging，结果服务在启动生产前占有资源。会话 max_results 同时限制这些结果服务及客户端通道的数量；交付 IPC 缓冲另外受 result_buffer_bytes 约束，导出的 Arrow/NumPy 视图继续由 BatchLease 计费。用户看到 QueryResult，与 local 相同；local 入口继续直接执行原生查询。

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

两种调度器使用同一个 ResourceManager。FTE 的阶段任务可以逐批准入，pipelined 的活动组需要整组推进容量。公平排队控制两类工作；第一版不引入运行中任务抢占。

FTE 重试同样消耗准入和预算，不能成为不受限的额外任务。高优先级也不能突破硬内存上限。CPU、公平性、native 内存和慢结果消费均纳入混跑验证。

## 统一结果交付

### ResultService 与 QueryResult

QueryResult 提供 schema、批次迭代、collect、close 和状态查询。P1.1 通过 execution_state 与 state 分别观察执行和交付；P2 根据全图生产状态停止执行期限，并在最终 EOF 前核实分布式结局。FTE 的提交完成通知随 P3 接入。提交请求返回结果句柄，不等待所有任务完成；FTE 的首批读取等待 ResultManifest 发布，pipelined 可以读取运行中任务的结果。

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
| 分组聚合、hash join | 基础闭环之后实现 | NULL、倾斜、build/probe readiness 和重试对照 |
| BROADCAST | 基础闭环之后实现 | 多消费者、共享所有权和独立关闭 |
| LIMIT | 无序单根输出的提前结束 | 全局计数、多个消费者及取消竞争 |
| ORDER BY、TopN | 分析算子阶段实现 | 全局顺序、内存预算与 native spill |
| window、ASOF、MARK、delim 等复杂算子 | 明确拒绝 | 按语义和分布式计划逐项扩展 |
| 递归 CTE 与迭代反馈 | 明确拒绝 | 需要独立的反馈执行设计 |
| Python、AI 与 GPU UDF | 明确拒绝 | 按 backend 验证资源、取消、类型及 FTE 重放能力 |
| COPY、DataSink、DML 与扩展写入 | 明确拒绝 | 独立定义写入提交和副作用语义 |

最小类型集合从布尔、整数、浮点和字符串开始，同时支持 NULL、空批次和 schema-only 结果。decimal、时间、嵌套类型、tensor、FILE 和媒体扩展逐项验收。能够序列化不等于类型和算子语义已经一致。

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
| [FTE backend](vane/runners/fte/backend.py) 与旧 FTE 控制结构 | TaskService 和 RecoveryScheduler 直接接管，不留下调用旧 manager 的包装器 |
| [LocalRunner](vane/runners/local/runner.py) 与 [LocalQueryRuntime](vane/execution/local_query.py) 的查询分流 | 统一 local backend 与 QueryContext；模型服务另行拥有模型生命周期 |
| [Ray driver](vane/runners/ray/driver.py) 与 [worker](vane/runners/ray/worker.py) 中耦合执行模式的部分 | Coordinator、scheduler 和 Ray backend 分担职责；worker 数据执行进入共同 native runtime |
| [Ray 结果包装](vane/runners/ray/partition_metadata.py) 与 [Python 结果源](src/vane_py/pyresult_source.cpp) | QueryResult 和 native ResultService 接管公开结果链路 |
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

退出条件：提交前后 worker 丢失、下游失败、重复提交、迟到 attempt、对象缺失和重试耗尽均有可控验证；恢复不会重复暴露行；生产 worker 退出后可从独立存储读到已提交输出。

P3 不能通过委托旧 FTE 引擎完成。新的任务服务、计划格式、结果协议与 pipelined 共用，是本阶段的核心验收条件。

### P4 分析能力与混跑

增加 aggregate、join、broadcast、全局 LIMIT、排序及相应类型，分别验证两种执行模式。完善 BuildReady、资源公平性和状态诊断。

退出条件：SQL 对照、空输入、NULL、倾斜、低容量活性、多消费者及两种模式混跑通过。AI/GPU UDF 按独立 backend 矩阵扩展，不作为纯 SQL 闭环的隐含前提。

### P5 删除旧入口并完成发布验收

清除被替代的 runner 分流、任务协议、结果包装、配置别名和文档，更新所有受支持调用方。重写依赖旧内部实现的测试，保留并扩展其有效语义场景。

退出条件：所有公开查询入口只进入新引擎，旧调度器和结果路径不在运行依赖中；对应 release gate 通过；提供新 API、支持矩阵、可复现基准和数值默认值依据。

各阶段按新能力组织可审阅的变更。开发期间尚未删除的旧源码只能作为参考，不能成为新实现的执行依赖；阶段性原型不代表完整双模式已经交付。正式发布以新架构和声明的能力范围验收，不以旧接口继续可用为条件。

分支仍基于 feature/local-runtime，使用其中可取的实现经验和代码。基础分支合入 main 后再调整分支基线；继承提交历史不产生 API 或模块结构的兼容承诺。

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

local-runtime 的 [资源图](LOCAL_MODEL_RUNTIME.md#shared-resource-graph-and-local-execution-identity) 是 structural_only，不能直接执行；[managed native streams](LOCAL_MODEL_RUNTIME.md#managed-native-result-streams) 提供了结果借用、预算和清理经验。这些证据说明需要哪些能力，不要求新引擎继承原有类、接口或分流方式。

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
