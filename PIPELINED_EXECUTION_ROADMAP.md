# Vane 执行架构实施 Roadmap

本 roadmap 将[详细设计](PIPELINED_EXECUTION_DESIGN.md)拆成可验证的实现增量。local 直接原生执行；只有 Ray 选择 pipelined 或 FTE。目标是替换旧分布式执行层，不维护旧接口适配器。

P0 原开发分支为 feat/pipelined-execution，基于 feature/local-runtime 的 36bdc721fa6f060bcd59d2e1df61e45359a9f292，已通过 PR #935 合入 integration/pipelined-execution。P1.1 已通过 PR #943 合入同一集成分支，提交为 9f956003ae。P1.2 已通过 PR #944 合入同一集成分支，提交为 31191cae217d。P2 通过 PR #962 合入同一集成分支，提交为 68b5407a4ca4。P3 从此提交切出 feat/materialized-exchange，完成物化 I/O、不可变文件输入、恢复调度及公开 FTE 结果，已通过 PR #963 合入 a69e60ca43d9。P4 已通过 PR #970 合入 integration/pipelined-execution，提交为 5d41a675e6。P5.1 从该提交切出 refactor/execution-cutover。每一步以代码、相关测试和验收记录更新进度，尚未实现的接口不写成已可执行。

## 实施规则

- 本地查询不经过 FragmentGraph、分布式 scheduler、TaskService 或网络 exchange。
- Ray 两种策略共用计划图与 native TaskRuntime，新的 FTE 不调用旧 FTE manager。
- 先实现有界的正确执行与取消，再测量性能并优化。
- 新模块只有完成实际入口接线后才成为公开能力；数据结构测试通过不代表查询已经可运行。
- 旧源码在接管对应职责后删除；调用方与测试按新契约修改，不增加 legacy 别名或 fallback。
- 优先运行受影响测试；本轮按要求只验收相关测试，不运行完整 release gate。测试环境必须使用非 editable 安装，并记录运行的代码和 native 基线。

## 里程碑概览

| 阶段 | 可交付能力 | 依赖 | 状态 |
| --- | --- | --- | --- |
| P0 | 执行目标契约与无执行副作用的 Ray 计划图 | 无 | P0.1–P0.4 已完成 |
| P1 | local 原生结果入口；分布式直接通道的进程内契约验证 | P0 | P1.1、P1.2 均已合入 |
| P2 | Ray pipelined 完整查询和 native 结果服务 | P1 | P2.1–P2.3 已合入 |
| P3 | 新 Ray FTE 的物化、提交与重试 | P0、P1；复用 P2 的服务与结果设施 | P3.1—P3.3 已实现，完整相关验收见下文 |
| P4 | 分析算子、类型扩展和两种策略混跑 | P2、P3 | P4.1—P4.4 已完成，相关验收通过 |
| P5 | 旧路径删除、支持矩阵、发布与性能验收 | P4 | P5.1、P5.2.1–P5.2.3 已完成相关验收；P5.2.4 独立 Server、历史 Flight 超时定位及 P5.3 待完成 |

P2 是首个新的分布式流水执行交付点；P3 完成之后才具备新架构的双策略执行。P1 的进程内通道测试不新增 local+pipelined 公开模式。

## P0 契约与纯计划图

### P0.1 执行目标与不可变配置

- [x] 定义 LocalExecution 和 RayExecution；local 不带分布式策略字段。
- [x] 只接受 local、ray/pipelined 和 ray/fte 三种有效目标。
- [x] 定义 FTE 配置和查询超时，拒绝未知字段、无效数值及错误组合。
- [x] 配置形成独立不可变快照，不读取或写入旧 runner 环境变量。
- [x] 测试模式隔离、快照、序列化及无效组合；不对 vane.connect 暴露未接线的执行参数。

建议落点：vane/execution/query_options.py 与对应 fast tests。

### P0.2 Ray FragmentGraph 数据契约

- [x] 定义 native fragment payload、输入输出端口、交换边和根结果。
- [x] 验证 DAG、唯一身份、端口完整性、schema 匹配、分区约束及所有 fragment 可到达根结果。
- [x] 图中不包含 FTE 专用 handle 或 direct endpoint；策略在执行阶段绑定。
- [x] 固定协议版本与 engine identity，提供确定的序列化和图指纹。
- [x] 通过多输入、并行边、空输入、非法依赖及损坏载荷测试。

建议落点：vane/execution/plan.py 与对应 fast tests。此步只验证计划载体，不能将测试用 opaque payload 当作 native 可执行计划。

### P0.3 Native fragment builder

- [x] 从真实绑定后的 native 计划生成 fragment 和 exchange 边，不提交 task，不调用 materialize。
- [x] 导出可执行 fragment 字节、端口 schema、source 描述和所需能力。
- [x] HASH 分区表达式来自 native 规则，不能在 Python 重写哈希算法。
- [x] 支持常量、range、基础文件 scan、filter、projection、GATHER 和 HASH 的最小子集。
- [x] native round-trip 与 SQL 规划测试证明载荷可执行；local 原生查询不调用 builder。

实现见 [compiler.py](vane/execution/compiler.py) 与 [fragment_plan.cpp](src/vane_py/execution/fragment_plan.cpp)。入口直接进入 Binder、Optimizer、PhysicalPlanGenerator，以显式输出分布生成 GATHER/HASH 边；aggregate/join 的自动分布式规划留待 P4。连接锁覆盖绑定与优化设置的读取，当前子集的连接和数据源快照由 P0.4 接入。builder 只导出端口，direct 或 materialized 数据描述在执行时绑定。

### P0.4 提交边界与能力检查

- [x] 将查询身份、执行目标、native 计划、连接快照和资源声明组成提交描述。
- [x] 明确扫描回放能力、类型能力、结果 schema 和协议身份检查。
- [x] 新图拒绝不支持能力，不借旧 PlanRunner 完成计划。
- [x] 缓存键同时涵盖 engine、图与执行选项；本地提交不承担网络协议开销。

实现见 [submission.py](vane/execution/submission.py) 与 [resource_demand.py](vane/execution/resource_demand.py)。内部 RayQuerySpec 冻结当前内置 SQL 子集的 session、source、资源和结果描述；worker 先检查能力，再恢复查询独占连接并对照 native 载荷。不可变计划蓝图不代表已经准入或拥有可恢复的 exchange store。普通 Parquet 的纯准备入口只提供固定成员与大小/修改时间检查，允许 pipelined，拒绝直接声明为 FTE；P3 已通过文件 staging 和独立 snapshot codec 接通公开 FTE。[详细保证](PIPELINED_EXECUTION_DESIGN.md#ray-提交描述与-worker-准备)见设计文档。

P0 退出条件：真实 SQL 可形成两种 Ray 策略共用的可执行计划描述，完整往返与能力校验通过，构图无执行副作用。仅完成 P0.1/P0.2 不标记整个 P0 完成。

## P1 本地结果与直接通道基础

### P1.1 本地原生查询入口

- [x] QueryContext 管理查询身份、准入、取消和期限，不继承 LocalModelRequest。
- [x] local 直接推进原生查询，QueryResult 提供逐批读取、collect 和 close。
- [x] BatchLease 覆盖 Arrow 与 NumPy 导出视图，close 后仍借用的内存持续计费。
- [x] 接通 local 公开入口，拒绝任何 execution override。

验收：真实 SQL、参数、空结果、增量结果、取消、保留视图及清理失败重试；通过调用边界测试确认不创建分布式计划或任务。

实现位于 [query_runtime.py](vane/execution/query_runtime.py)、[result_delivery.py](vane/execution/result_delivery.py)、[batch_lease.py](vane/execution/batch_lease.py) 与 [local_query.cpp](src/vane_py/execution/local_query.cpp)。独立 QueryContext 复用准入和结果容量原语；查询入口直接调用 native PendingQuery 并交付 RecordBatch。ManagedResult 已重命名为 QueryResult，所有调用方使用新名字，无别名；原来的惰性 connection.query 调用方迁移到 sql。新入口暂限自动提交下的只读 SELECT，模型 UDF 后续接入。

### P1.2 DirectExchange 的进程内契约

- [x] 有界 channel、消费窗口、成员封闭、FINISH 和消费者关闭。
- [x] native source 无数据返回 BLOCKED，唤醒无丢失。
- [x] sink 部分发送后恢复不重发已接受的行。
- [x] 生产完成与输出排空分开，finalize 不等待远程消费。
- [x] 进程内 TaskService 测试两个 fragment 的并发推进。

验收：受控上游尚未完成时下游已消费；极小预算、慢消费者、空分区、多输入及取消不会死锁，资源最终回到真实基线。

实现位于 [native channel](src/vane_py/execution/direct_exchange.cpp)、[native TaskService 与算子](src/vane_py/execution/direct_task.cpp) 和 [进程内控制设施](vane/execution/direct_exchange.py)。123 项[契约测试](tests/fast/test_direct_exchange.py)验证真实 GATHER/HASH fragment 在单线程与多线程下执行，64 字节、单帧窗口中的暂停和恢复，广播共享缓冲及保留切片，动态封闭成员、多输入、部分输出后的错误、消费者提前关闭、期限、并发取消和回调重入，并覆盖后台完成后不再 pump、等待输入时失去全部消费者、输出完成后的通道错误、已关闭 HASH 分区的超大行丢弃、已有 native/通道失败优先于后续超时、最后一个消费者关闭时保留输入错误，以及输入已到 EOF、native 提前 finalize 或输出等待交付时仍检查输入的持久错误。数据在 C++ 通道内传递，不调用旧 runner 或物化测试入口。

P1 退出条件已满足。预算保证覆盖通道实际拥有的值缓冲；跨查询资源池、Flight、Ray 调度、动态 split/routing 更新和分布式根结果交付仍属于 P2/P3。公开 local 查询路径不进入此设施。

## P2 Ray pipelined

### P2.1 Native Flight 数据面

- [x] Direct ticket、QueryId/AttemptId/WorkerEpoch 校验与访问 capability。
- [x] DoGet 数据帧、累计 ACK、关闭与有界 I/O。
- [x] 独立控制通道保证窗口耗尽时仍可取消。
- [x] 两进程故障注入验证断连不伪造 EOF。

### P2.2 调度与 worker 控制

- [x] Ray backend 放置固定 epoch 的 worker；prepare 绑定并封闭全部 split，提供 start、状态、生产结局和取消控制。
- [x] 活动组获得可兑现的上下文与最小推进容量；预留全部成功或撤销。
- [x] PipelinedScheduler 先准备消费者再启动生产者。
- [x] worker epoch 变化使查询失败，不通过 Ray 方法重试重放任务。

### P2.3 根结果与公开入口

- [x] 原生 ResultService 从 root worker 拉取数据，为客户端提供可达的 Flight 端点。
- [x] 上游和客户端两个方向的窗口都有界，批次不经 Python/Ray ObjectRef 中转。
- [x] 接通 ray/pipelined 查询及 QueryResult。
- [x] 区分执行结局和交付结局，部分结果之后的失败可观察。

实现位于 [native Flight](src/vane_py/execution/direct_flight.cpp)、[固定放置与容量](vane/execution/pipelined_plan.py)、[worker/ResultService](vane/execution/pipelined_worker.py) 及 [scheduler/QueryContext](vane/execution/pipelined_runtime.py)。公开 `Runtime(RayResources(...)).connect()` 使用服务共享资源；Ray 须先初始化，当前 SQL/type 范围延续 P0，SQL 参数、分析算子和 FTE 明确拒绝。

P2 退出条件：两个 worker 的真实查询提前交付首批；慢客户端产生背压；取消和 worker/结果服务失效有明确结局；数据面没有 shuffle 物化文件。

## P3 Ray FTE

### P3.1 物化 I/O 与提交基础

- [x] 独立 C++ 线程在有界任务缓冲与 Arrow IPC 对象之间读写，复用共同的 native TaskRuntime。
- [x] SharedDirectoryStore 使用独立的 query/attempt 对象命名空间、存储身份和原子元数据发布。
- [x] 不可变 AttemptManifest，固定输入身份、worker epoch、唯一成功 attempt 的 fencing 与幂等提交。
- [x] 所有 task 和输出分区（包括空分区）提交后才能封闭 StageManifest。
- [x] 校验对象长度、SHA-256、schema 和封存记录；缺失或损坏明确失败。
- [x] 显式 attempt 清理、query 对象所有权和 ReadLease；失败清理保留存储配额。

实现见 [native I/O](src/vane_py/execution/materialized_exchange.cpp)、[manifest](vane/execution/materialized_exchange.py) 与 [store/commit](vane/execution/materialized_store.py)。共享目录必须由部署方提供独立于计算 worker 的故障域；身份 marker 无法自动验证底层挂载的物理可靠性。P3.2/P3.3 已将这些设施接入公开 FTE、重试和共享资源准入。

验收：原生 fragment 写入后杀死生产进程，封存输出仍可提交和读取；封存前退出只留下不可提交的私有输出，丢弃后可用同一输入身份重放。单线程和四线程的共同 TaskRuntime 能消费物化输入。并发重复提交、迟到 attempt、校验期间取消/重试、空分区、对象损坏及清理失败均有定向测试。

P3.1 本地相关验证为 **283 passed**：物化 I/O/提交 56、DirectExchange 123、DirectFlight 24、QueryResult 67、真实 Ray pipelined 13。非 editable 安装中的 277 个 Python/类型文件与 checkout 一致，native 与增量 Release 构建产物一致；格式、Ruff、全仓库 mypy 和源码版权清单通过。未运行完整 release/fast 套件。这是 P3.1 阶段记录；P3 完整验收记录见下文。

### P3.2 不可变数据源与重放准备

- [x] 为文件 scan 接入不可变数据源版本或受查询生命周期保护的 staging；重试仍可读到同一份输入。
- [x] 冻结优化阶段依赖的数据源，在 worker 准备与重试时恢复同一连接/source 快照。
- [x] 将逻辑 task 的固定 split 与已提交上游 StageManifest 组成可验证的输入身份。

### P3.3 恢复调度与公开结果

- [x] 注册并验证 exchange store，接通 worker、native 物化 binding、存储和 staging/I/O 准入。
- [x] StageManifest 封闭后才调度下游；重试时复用固定输入与 split。
- [x] RecoveryScheduler 的失败分类、重试上限、退避与共享 deadline。
- [x] ResultManifest 发布后才交付 FTE 结果。
- [x] 取消和清理期限、失联查询的有限 lease 与 orphan 回收。
- [x] 接通公开 ray/fte QueryResult，在真实 Ray worker 故障下验收。

实现见 [文件冻结](src/vane_py/execution/file_snapshot.cpp)、[固定输入绑定](vane/execution/fte_plan.py)、[共享存储准入](vane/execution/fte_store.py)、[native attempt owner](vane/execution/fte_worker.py) 和 [RecoveryScheduler](vane/execution/recovery_runtime.py)。公开 `vane.ExchangeStore` 注册到 `RayResources.exchange_stores`；`FteOptions` 指定 store、attempt 上限与退避。同一会话可并发运行 pipelined/FTE，共用 worker 和资源计费。

文件冻结发生在优化前，保留被剪枝的数据源；重试只读取 SHA-256 验证的副本，Hive 路径语义保留。动态路径、虚拟 filename/file_index、未支持的 scan/算子/type 明确拒绝。共享存储须由部署方提供独立故障域、原子文件操作、锁、sync 与同步时钟；marker 只校验身份和可见性。

查询租约与 native I/O 锁共同保护对象，过期 lease 不会使仍有 native 读写者的目录被删除。全局存储计费记录独立于对象目录，失败清理保留配额；actor 死亡到实际进程退出之间有有限清理宽限。orphan 在活跃服务心跳、worker 清理或下次准入回收，也可显式运行 StorePool.collect_expired。coordinator 恢复和内核阻塞文件 I/O 的强制中止不在本阶段承诺内。

验收：提交前 worker 丢失可重试；提交后 worker 丢失仍可读；迟到或重复提交不会重复输出；下游重试读取同一 manifest；对象永久丢失明确失败；全路径不调用旧 FTE 引擎。不实现 local FTE。

### P3 完整验收

P3 初版（`6097810`）分组运行受影响测试，共 **685 passed**：物化交换/提交 56、文件冻结 17、存储租约/配额 8、配置与放置 13、native 编译/提交描述 295、查询配置 45、DirectExchange 123、DirectFlight 24、QueryResult 67、真实 Ray FTE 24、真实 Ray pipelined 13。仅运行相关测试，未运行完整 release/fast 套件。

真实 Ray 验收覆盖上游/下游 worker 丢失、提交后计算节点退出、重试耗尽、退避与原执行期限、原文件变更、HASH 阶段、空结果、取消和交付期限、已提交输入损坏/丢失、ResultService 丢失、两种策略并发与共享准入、独立于 native pump 的状态探测、三种 manifest 发布的确认丢失，以及真实 native worker 的租约过期和 orphan 清理。测试同时核对无重复行、固定输入身份、新 epoch/fence 及结束后的资源账本。

C++ 已增量 Release 构建并非 editable 安装；281 个 Python/类型文件与 checkout 一致，native 与构建产物 SHA-256 一致。engine identity 为 `346ef5b69e:fragment:39b1018536dc95747a225e38632714a553a2a550a6cfddb6a026a911a708b244`。root 格式、适用 pre-commit（含全仓库 mypy）、源码版权清单、文档本地链接及源码包检查通过。新增物化交换、文件冻结、存储和真实 Ray 恢复测试已加入 release launcher 与 sdist 清单；原有旧 FTE 测试保留到 P5 迁移阶段。

P3.1—P3.3 退出条件已满足。后续进入 P4 的分析算子、类型、调度公平性和诊断扩展。本地验证平台为 Linux；macOS/Windows 构建与运行仍由 CI 验证。

### PR #963 审查修复

- 冻结文件通过原生 scan dependency 作为确定列表绑定，保留顺序及重复引用；路径中的通配符不再展开，子查询复制和 union_by_name 也使用同一列表。
- 按实际快照目标去重，`./` 与普通父目录别名只复制、计费一次，扫描仍返回重复引用的行。目标同时包含解析后的源路径哈希，避免 symlink/`..` 将不同文件合并。
- orphan 加锁或删除 I/O 失败按目录隔离，保留外部 allocation 和配额，继续清理其他目录并允许有剩余容量的查询准入。后续回收重试；当前查询续租和存储校验错误仍正常传播。

新增 18 项长期回归，覆盖文件名/目录/存储前缀中的通配符、相邻快照目录、嵌套子查询、路径别名的字节计费、symlink 父目录遍历、native 锁失败、部分删除、allocation 删除失败，以及真实 Ray 心跳和新查询准入。

本轮仅运行相关测试，共 **435 passed**：文件冻结 31、存储租约 11、native 编译/提交描述 295、物化交换 56、普通 Parquet 14、真实 Ray 恢复 25，以及审查者提供的 3 条复现（修复前均失败，修复后均通过）。未运行完整 release/fast 套件。

已增量 Release 构建、非 editable 安装；281 个 Python/类型文件及 native 构建产物一致性检查通过。当前 engine identity 为 `b4a4a5b19d:fragment:5b6fc20c60dd8a909ed06793a52f5e36d5721f4de09731f92eb5a89903db1760`。root/DuckDB 格式检查、适用 pre-commit（含 mypy）、源码版权清单与 diff 检查通过；本轮验证平台为 Linux。

### PR #963 慢清理与 Hive 路径修复

- 全局 allocation.lock 只保护配额和租约元数据。回收时重新核对候选的有效期并取得查询独占锁，在全局锁外递归删除，完成后才移除 allocation、释放配额。查询锁移至 allocations 目录，避免部分删除或目录已经消失时被第二个回收者绕过。
- 普通 close 的清理回调与递归删除也在全局锁外执行；查询关闭后阻止新的 native 参与者。并发回收已持锁时 close 返回可重试的 CleanupPending，回收完成后仍执行 coordinator 的内存清理回调。
- 心跳和准入请求每个 StorePool 的单个后台回收任务。慢删除不会阻塞续租或有剩余容量的新查询；仍未删除的对象继续占用配额。
- 在路径规范化前使用原生 HivePartitioning::Parse 提取分区信息，以物理源路径哈希和原始 Hive 键值构造冻结目标。保留 `part=42/../`、同一文件的不同分区引用、首次键优先、编码值及 NULL 语义；相同物理文件且分区信息相同的别名才复用副本。

新增 15 项回归并加强原有清理测试。最终非 editable 安装上共 **154 passed**：文件冻结 42、存储租约 14、物化交换 56、普通 Parquet 14、真实 Ray 恢复 26，以及审查者提供的两条复现（修复前均失败）。慢删除测试覆盖另一会话续租、准入、竞争回收、配额持有、close 回调及重试；真实 Ray 测试将删除暂停超过两倍租约时长，验证持续续租、结果读取及后续查询。

已增量 Release 构建；281 个 Python/类型文件与最终 checkout 一致，native 与构建产物 SHA-256 一致。root 格式、适用 pre-commit（含 mypy）、源码版权清单和 diff 检查通过。仅运行上述相关测试，未运行完整 release/fast 套件；本轮验证平台为 Linux。

### PR #963 生成文件名列修复

- FTE 在优化前同时检查虚拟 filename ID 和 MultiFileBindData.reader_bind.filename_idx，拒绝 `filename=true` 与自定义名称的生成列引用。投影、星号展开、仅用于过滤和之后被统计信息剪枝的引用都明确报错；未引用生成列的扫描继续支持。
- 原生文件剪枝使用 filename 选项指定的实际列名，避免启用自定义生成列时误把真实物理 filename 列替换成文件路径。真实 filename/origin 列继续支持投影和过滤，local 与 Ray pipelined 保留生成列语义。
- 新增 26 项长期回归，覆盖优化器启用/禁用、两种 Parquet 入口、物理列与自定义生成列并存，以及真实 Ray 中编译失败后的配额和快照清理、后续查询及 pipelined 对照。

最终非 editable 安装上共 **385 passed**：文件冻结 64、native 编译/提交描述 295、普通 Parquet 16、CSV 文件名列对照 1、真实 Ray 定向用例 6，以及审查者提供的 3 条复现。审查复现修复前均失败，新增的两条物理列剪枝回归也验证了修复前失败。仅运行上述相关测试，未运行完整 release/fast 套件。

已完成增量 Release 构建；281 个 Python/类型文件与 checkout 一致，native 与构建产物 SHA-256 一致。engine identity 为 `c205cae1b6:fragment:2a5322a87d904c4d62808269f81d5e8f91ac5ae94e23a25682cc0e437db7b806`。root/DuckDB 格式检查、适用 pre-commit、源码版权清单和 diff 检查通过；本轮验证平台为 Linux。

### PR #963 glob 展开与快照隔离修复

- 在创建任何快照前，先收集整个查询所有 scan 的完整文件引用列表，并完成每个 scan 的模式和引用数量校验。然后按既有快照身份去重复制、计费和绑定，保留原始引用顺序与重复扫描。
- exchange store 可以位于递归扫描目录内；本次创建的副本不会混入后续 glob。展开时已存在且匹配的存储目录文件仍正常参与扫描，不按目录前缀排除合法输入。
- 新增 7 项长期回归，覆盖精确路径加递归 glob、重复递归 glob、已有存储目录文件、跨 scan 展开、精确 source_bytes 预算、删除原文件后的重放，以及真实 Ray 单/多分区的连续查询和资源清理。加强原有引用上限测试，确认超限在复制前被拒绝。

最终非 editable 安装上共 **375 passed**：文件冻结 69、native 编译/提交描述 295、真实 Ray 定向用例 8，以及审查者提供的 3 条复现（修复前均失败）。仅运行上述相关测试，未运行完整 release/fast 套件。

已完成增量 Release 构建；281 个 Python/类型文件与 checkout 一致，native 与构建产物 SHA-256 一致。engine identity 为 `c205cae1b6:fragment:4d3ce533e0efce81679fd69ace1590c6fd29da17cc57a1c196291ec15e47f4e9`。root 格式检查、适用 pre-commit、源码版权清单和 diff 检查通过；本轮验证平台为 Linux。

## P4 分析与混跑

- [x] P4.1：COUNT、非 DECIMAL/HUGEINT 输入的 SUM、MIN/MAX 与浮点 AVG partial/final 聚合；DECIMAL/HUGEINT SUM、DISTINCT、排序聚合、Binder 内部改写、整数/decimal/时间 AVG 按完整 group 执行；FILTER 与空输入保持原生语义。
- [x] P4.2：等值 hash join、自动小 build 广播、native BuildReady、多个消费者独立关闭。
- [x] P4.3：全局 LIMIT/OFFSET、单根 ORDER BY、partial/final TopN 与确定排序语义。
- [x] P4.4：decimal、时间、二进制、LIST/STRUCT/MAP/ARRAY 的 native 帧及 Flight/物化传输。
- [x] 两种 Ray 策略共用 FIFO WorkerResourceManager；图原子准入、FTE attempt 归还与重新排队、取消和清理所有权。
- [x] QueryResult 诊断贯通 query、task、BuildReady、channel、预算和清理；native 状态探测不等待执行锁。
- [x] 完成全部相关验收并记录构建身份、测试数量及边界。

验收：与原生 DuckDB 比较 SQL 结果，覆盖 NULL、空输入、倾斜、重复执行及 FTE 故障注入；资源公平性同时覆盖只有三个上下文的一 worker 混跑。AI/GPU UDF 独立排期，按 backend 验证生命周期及重放保证。不运行完整 release/fast 套件。

## P5 删除与发布

### P5.1 入口切换与旧执行路径删除

- [x] 默认连接为 local；`query()` 惰性创建会话共有的 QueryRuntime，独立 cursor 共享一次准入计费。
- [x] Ray 只通过 `query()` 执行声明支持的 SELECT；native SQL/Relation 执行拒绝 Ray 连接。
- [x] 删除 Python runners、旧 FTE manager、原生 Ray plan/task/worker 绑定、全局 runner 设置及旧 Arrow 分区结果适配。
- [x] 删除 `VANE_RUNNER` 和 `configure(runner=...)` 的选择作用；连接参数决定 backend，查询参数决定 Ray 策略。
- [x] 保留独立模型服务；模型注册直接读取原生 UDF 元数据，资源图直接观察原生算子，不借用旧分布式计划或执行器。
- [x] 迁移受支持调用方、类型声明、文档、release gate 和 sdist 清单；历史 GPU 基准标明需要改用注册模型。
- [x] 完成相关测试、非 editable native 构建及安装内容校验。

本步不引入兼容适配或自动 fallback。旧计划对象和 runner 内部接口的测试退出；有效的 SQL、schema、取消、资源计费和结果所有权场景通过公开 query、SQL/Relation 和模型接口验收。

### P5.2 差分与性能验收

- [x] P5.2.1：系统化比较 native local、Ray pipelined、Ray FTE，加入可重放的种子/分区/线程矩阵、确定到达顺序和重复故障/清理验收。实现与运行方法见[执行验收](EXECUTION_ACCEPTANCE.md)。
- [ ] 完成 P5.1 偶发 Flight 超时的根因定位与验证。P5.2.1 已加入数据/控制操作归因、重复长查询及失败现场保存；本轮未复现，根因仍未确认。
- [x] P5.2.2：测量冷启动、预热、首批、吞吐、混跑、慢客户端与故障恢复；每个指标记录配置及重复次数。工具、计时边界与结果见[执行基准](EXECUTION_BENCHMARKS.md)。
- [x] 根据实测评估容量默认值。两组容量、两个数据规模的对照支持保留当前默认值；较小窗口的缓冲预留更低，但扫描延迟更高，数值依据见基准记录。
- [x] P5.2.3：应用级 Runtime、服务共享 worker/结果服务、多 Session 配额与独立查询上下文；完成连续两轮代码审查、一次增量构建及相关测试。
- [ ] P5.2.4：独立 Server 部署及远程会话/查询协议，迁移规划、协调器与续租至 Server 进程；客户端断连租约及服务故障边界验收。
  - [x] P5.2.4a：Flight 会话控制、独立启动、鉴权、租约与可重试关闭；已完成相关验证。详见 [Server 设计](SERVER_DESIGN.md)。
  - [x] P5.2.4b：远程查询控制、客户端及原生结果交付；规划与所有权由 Server 管理，已完成两种模式和 TLS 的相关验证。
  - [ ] P5.2.4c：客户端断连、服务故障、混跑与部署验收。

Server 先使用 Flight 对外接入；DuckDB 整体升级到 2.0 时再集成 Quack。SessionService / QueryService 独立于线协议，内部 Flight exchange 保留，不实现双协议兼容或 fallback。

### P5.3 发布验收

- [ ] 固定支持矩阵和失败边界，完成跨平台 CI 与 release gate。
- [ ] 更新发布文档与版本，按验收结果发布。

本地只运行受影响测试，不运行完整 release/fast 套件。P5.2 的历史 Flight 超时定位及 P5.3 发布验收未完成前不宣称 P5 整体完成。

## 增量验收记录

### P0.1/P0.2

已实现 P0.1 和 P0.2：

- [query_options.py](vane/execution/query_options.py)：LocalExecution 无分布式策略字段；RayExecution 验证策略与 FTE 参数；查询配置快照、超时和严格序列化。
- [plan.py](vane/execution/plan.py)：不可变 native payload、端口、交换边与根结果；依赖、schema、分区、协议和 engine identity 校验；确定的拓扑顺序及指纹。
- [配置测试](tests/fast/test_query_execution_options.py)和[计划图测试](tests/fast/test_execution_fragment_graph.py)：合计 76 项通过，并已加入 [release launcher](scripts/run_release_tests.sh)。
- 格式检查、ruff 和新模块的严格 mypy 检查通过；源码版权清单检查通过。
- `scripts/run_release_tests.sh` 全部通过：非 Ray 分片 3471 passed、8 skipped，共享 Ray 分片 74 passed，自建 Ray 集群分片 2 passed；合计 3547 passed、8 skipped。跳过项因未安装可选依赖 qdrant_client（7 项）和 adbc_driver_manager（1 项）。

该增量验证使用本 worktree 的非 editable wheel。Python 包来自当时源码；native 复用与基线 src/vane_py、external/duckdb 和 CMakeLists.txt 无差异的已编译产物，DuckDB source ID 为 a1b4927e0ad741903521aacc7fcf82a74620a269。新增模块的安装字节已与 checkout 比较一致，该增量没有 C++ 修改。

上述记录对应 P0.1/P0.2 增量，P0.3 的构建与验收记录如下。当前仍不能执行新的 Ray pipelined 查询。

### P0.3 增量（2026 年 10 月 3 日）

- [compiler.py](vane/execution/compiler.py) 提供内部编译和 native 图验证入口，不接入 vane.connect 的执行参数。
- [fragment_plan.cpp](src/vane_py/execution/fragment_plan.cpp) 与 [bindings](src/vane_py/execution/fragment_plan_bindings.cpp) 负责真实 SQL 规划、逐节点 native 编码、显式输入绑定、扫描分片和 HASH 表达式；不依赖旧 PlanRunner。
- [PhysicalOperator.SerializeNode](external/duckdb/src/execution/physical_operator.cpp) 为 fragment codec 提供单节点序列化。其余 SQL 算子继续复用 DuckDB。
- 首步文件扫描限定为显式 Parquet scan；SQL 参数、自动 aggregate/join 分布式规划及完整连接/数据源快照尚未实现。HASH 通过内部输出分区请求覆盖 native 表达式与路由。
- [验收测试](tests/fast/test_native_fragment_compiler.py) 共 53 项通过，连同 P0.1/P0.2 共 129 项通过，已加入 release gate。覆盖真实 SQL 与 native 结果对照、独立进程加载、输入绑定、空任务、Parquet 文件枚举与 union schema、HASH 的 NULL/多列键、损坏载荷及不支持能力的拒绝。
- 已从当前 C++ 源码在 build/python-release 完成 Release 构建并非 editable 安装。安装后的 Python 字节与 checkout 一致；engine identity 为 `b4ed3b0c03:fragment:6586fb3c6ca1e1854f1d4b010b4efae305987af49b1f930368bf8b64b970d5d7`，其后半部分与编译器/加载器源码摘要一致。
- 格式、ruff、全仓库 mypy、源码版权清单和文档链接检查通过。
- 完整 `scripts/run_release_tests.sh` 通过：非 Ray 分片 3524 passed、8 skipped，共享 Ray 分片 74 passed，自建 Ray 集群分片 2 passed；合计 3600 passed、8 skipped。跳过项因未安装可选依赖 qdrant_client（7 项）和 adbc_driver_manager（1 项）。这次完整验收使用上述从当前 C++ 源码重建的 native。

### P0.4 增量（2026 年 10 月 3 日）

- [RayQuerySpec](vane/execution/submission.py) 和 [ResourceDemand](vane/execution/resource_demand.py) 组成不可变、严格序列化的内部提交描述；两种 Ray 策略共用原生图，local 在入口直接拒绝。
- Native 在同一连接锁内捕获受支持的语义设置并完成构图。worker 只写自己的 session；全局配置必须一致。排序设置别名由 native 规则规范化。
- worker 校验 engine、协议、类型/连接 profile、scan capability、split codec、交换分布、文件元数据、结果 schema/列名与 HASH；不调用旧 runner，不启动任务。
- 计划蓝图的缓存身份涵盖完整快照和资源/执行配置，只排除 query_id；缓存命中仍需重新准备 worker 和检查 source。
- [提交验收测试](tests/fast/test_execution_submission.py) 新增 72 项通过；连同前三步共 201 项通过。覆盖规划连接关闭后的跨进程加载、session 冻结与隔离、文件成员封闭和变化拒绝、FTE 回放限制、资源声明、损坏快照、元数据不一致及缓存身份。
- 当前 C++ 已在 build/python-release 增量 Release 构建并非 editable 安装；安装后的 Python 字节与 checkout 一致。engine identity 为 `b4ed3b0c03:fragment:069f2ec4040b47cae54268a31e205efd68758dc611997680b0266c87aa1119c0`，与当前编译器/加载器源码摘要一致。
- root/DuckDB 格式、ruff、全仓库 mypy、源码版权清单和仓库文档链接检查通过。
- 完整 `scripts/run_release_tests.sh` 通过：非 Ray 分片 3596 passed、8 skipped，共享 Ray 分片 74 passed，自建 Ray 集群分片 2 passed；合计 3672 passed、8 skipped。跳过项因未安装可选依赖 qdrant_client（7 项）和 adbc_driver_manager（1 项）。这次完整验收使用上述从当前 C++ 源码重建的 native。

P0.1–P0.4 的实现与完整验收已完成，P0 收口。P1.1 接通 local 原生 QueryContext/QueryResult、增量结果与 BatchLease 生命周期；P1.2 完成原生 TaskRuntime 和进程内直接通道契约，不新增 local+pipelined 模式。Ray 调度器及网络数据面继续按 P2 接线。

### P0 审查修复（PR #935）

- Parquet 的虚拟 `file_index` 在分配文件后会从零重新编号；当前编译器按原生列 ID 显式拒绝，覆盖投影、过滤及物理计划加载，同名普通数据列仍可执行。完整支持需在 split/scan 中保留原始文件索引。
- 表达式在优化前按 `IsConsistent()` 校验，拒绝 volatile 与仅单次查询内稳定的函数。ICU 的 `LOCALTIME`/`LOCALTIMESTAMP` 补齐 `CONSISTENT_WITHIN_QUERY` 声明；两种 Ray 模式都拒绝尚未冻结的查询时间，未来支持需把查询上下文纳入提交快照。
- 新增 37 项回归验收，覆盖多个文件/分区、Parquet 两个入口、同名普通列、五类时间表达式、过滤与 CASE，以及优化器开启/关闭。旧实现中 36 项拒绝用例失败，修复后连同 P0 原有用例共 238 项通过。
- 源码包清单补齐 P0 的四个 release 测试文件与设计/roadmap 文档，修复 CI 的缺失文件校验失败。
- 本次 native 已重新构建并非 editable 安装；engine identity 为 `fbbdb1efe0:fragment:ec00dd1647150041b52e0ce2d2dd7dcfcca038403c10a766e7352137fcdf46c7`，与 DuckDB 源码及编译器/加载器摘要一致。
- 完整 `scripts/run_release_tests.sh` 通过：非 Ray 分片 3633 passed、8 skipped，共享 Ray 分片 74 passed，自建 Ray 集群分片 2 passed；合计 3709 passed、8 skipped。跳过项仍为可选依赖 qdrant_client（7 项）和 adbc_driver_manager（1 项）。root/DuckDB 格式、ruff、pre-commit、源码版权清单与实际源码包发布校验通过。

### P0 优化依赖审查修复（PR #935）

- 在优化前保存绑定的 Parquet 文件集合到 `FragmentSpec.source_dependencies`，覆盖扫描整体移除及 hive/file pruning。优化后的 `sources` 只负责实际扫描与 split 分配，依赖不创建额外扫描任务；空结果 source fragment 保持单分区。
- native 文件快照、普通 Parquet 的 FTE 拒绝、绝对路径条件、worker capability/codec、图一致性和缓存身份同时检查执行 source 与原始数据源依赖；重复文件路径只捕获一次元数据。依赖随 native envelope 和严格 JSON 描述传输，不支持旧格式转换。
- 新增 19 项回归，覆盖统计信息生成空结果、常量 false、文件重写/mtime/删除、部分文件裁剪、FTE、能力、缓存、描述损坏以及 HASH/GATHER。连同 P0 原有用例共 257 项通过；审查方的 4 项临时复现用例从全部失败变为全部通过。
- 当前 C++ 已在 build/python-release 增量 Release 构建并非 editable 安装；engine identity 为 `fbbdb1efe0:fragment:e87a3f9db06b0ca9b67034b39ac8b47cf673af5043cb9fc477dd9bb2073afac4`，与编译器/加载器源码摘要一致。安装后 260 个 Python 源码文件已与 checkout 比较一致。
- 完整 `scripts/run_release_tests.sh` 通过：非 Ray 分片 3652 passed、8 skipped，共享 Ray 分片 74 passed，自建 Ray 集群分片 2 passed；合计 3728 passed、8 skipped。跳过项仍为可选依赖 qdrant_client（7 项）和 adbc_driver_manager（1 项）。
- root 格式、ruff、全仓库 mypy、pre-commit、源码版权清单、文档链接和实际源码包发布校验通过。

### P0 文件时间精度与优化器设置修复（PR #935）

- 本地文件元数据保存微秒时间戳与原生纳秒小数部分；提交快照从同一次句柄 stat 获取文件类型、大小和完整 mtime，覆盖 Linux/macOS 的纳秒与 Windows 的 100 纳秒精度。
- 本地文件版本标识保留完整 mtime，避免 Parquet 元数据缓存把同一秒内的等大小改写识别为未变化；同时覆盖显式路径和 glob 枚举后的重新规划。
- fragment 编译遵循 `enable_optimizer` 和逻辑计划的 `RequireOptimizer()`，逐项禁用设置继续由 native 优化器处理。关闭优化器时，受支持的 `IN` 表达式可直接生成物理计划；提交传输和 worker 重放恢复相同开关。
- 新增 24 项回归，覆盖 100 毫秒、100 纳秒和 1 纳秒的文件变化、优化为空的扫描、缓存身份、元数据缓存重新规划、两种 Ray 策略和分区执行。旧实现中 20 项失败、4 项对照通过；修复后 P0 四个模块共 281 项通过，审查方的 3 项临时用例也全部通过。
- 当前 C++ 已在 build/python-release 增量 Release 构建并非 editable 安装；engine identity 为 `263045b861:fragment:54c4fb92b4c4f973c224d5db2a262242d4dc742b6edf5846d35912696e69bb60`，与 DuckDB 源码及编译器/加载器摘要一致。安装后 260 个 Python 源码文件已与 checkout 比较一致。
- 完整 `scripts/run_release_tests.sh` 通过：非 Ray 分片 3676 passed、8 skipped，共享 Ray 分片 74 passed，自建 Ray 集群分片 2 passed；合计 3752 passed、8 skipped。跳过项仍为可选依赖 qdrant_client（7 项）和 adbc_driver_manager（1 项）。
- root 与 DuckDB 格式、ruff、适用的 pre-commit 检查、源码版权清单、文档链接和实际源码包发布校验通过。本地构建与测试运行于 Linux；macOS/Windows 由对应 CI 验证。

### P0 回调重入审查修复（PR #935）

- 删除自行获取锁的 `LockPlanningConnection`，八个需要连接的 native fragment 入口统一调用 `DuckDBPyConnection::LockConnection()`，复用已有的 Python 输入回调检查与 GIL 释放规则；回调内重入在等待连接或 context 锁前被拒绝。
- 新增 18 项独立进程回归，覆盖编译、提交准备、能力查询、worker 准备、直接计划加载、HASH 校验及有限测试执行，对同一连接和空闲 sibling cursor 均验证立即报错；捕获异常后外层 Parquet 查询和后续规划、执行继续成功。
- 旧版本的首批 16 项回归全部失败（同一连接死锁或允许回调进入）；修复后新增 18 项全部通过，P0 四个模块合计 299 passed。审查方的 3 项临时用例由 2 failed、1 passed 变为 3 passed。
- 当前 C++ 已在 build/python-release 增量 Release 构建并非 editable 安装；engine identity 为 `263045b861:fragment:f2b6f432831e448fb5fa5de40bec1c01c463cb11361a60aed2acbf8dac6422d1`，与源码摘要一致。安装后 260 个 Python 源码文件已与 checkout 比较一致。
- 完整 `scripts/run_release_tests.sh` 通过：非 Ray 分片 3694 passed、8 skipped，共享 Ray 分片 74 passed，自建 Ray 集群分片 2 passed；合计 3770 passed、8 skipped。跳过项仍为可选依赖 qdrant_client（7 项）和 adbc_driver_manager（1 项）。
- root 格式、ruff、适用的 pre-commit 检查、源码版权清单和文档链接检查通过。

### P0 隐式函数绑定副作用修复（PR #935）

- 显式函数预检查与 Binder 的 catalog lookup callback 复用内置函数来源校验；在宏展开或表函数参数求值前拒绝实际解析到的用户函数和宏，覆盖 SQL value function、内置宏间接调用及子 Binder。检查限定在本次 Planner，不拦截正常列/别名绑定，也不影响后续原生执行。
- 新增 72 项回归，覆盖两个编译入口、pipelined/FTE 提交、11 种 SQL value function 引用、限定名、CASE、表子查询、range/generate_series 与 LIMIT。内置宏、同名列/别名和显式 nextval 的拒绝作为对照；旧版本 42 failed、30 passed。
- 修复后新增 72 项全部通过，P0 四个模块合计 371 passed；审查方的 4 项临时用例由 2 failed、2 passed 变为 4 passed。
- 当前 C++ 已在 build/python-release 增量 Release 构建并非 editable 安装；engine identity 为 `263045b861:fragment:18497cb82d5466056628bb30c840bc924f5137d61255a71861e002fa76da51cf`，与源码摘要一致。安装后 260 个 Python 源码文件已与 checkout 比较一致。
- 本次按要求只验收相关测试，完整 release gate 已在完成前停止；本次修复未宣称通过完整套件。root 格式、ruff、适用的 pre-commit 检查、源码版权清单和文档链接检查通过。

### P1.1 原生查询结果入口（2026 年 10 月 4 日）

- `vane.connect(backend="local", resources=QueryResources(...))` 接通独立 QueryRuntime；`connection.query()` 与 `vane.query(..., connection=connection)` 共用 native 入口，返回 QueryResult。配置不依赖 VANE_RUNNER，拒绝 local 的任何 execution override。
- QueryContext 管理不可变选项、查询身份、共享准入、执行期限及中断隔离；准备失败和取消都保留清理责任，确认回收后才归还准入与结果名额。新路径不构建 FragmentGraph，不创建 LocalModelRequest。
- 结果逐批交付 RecordBatch；执行与交付各自记录状态。BatchLease 在导出 Arrow/NumPy 视图期间持续计费，关闭查询不会使视图失效；collect 逐批复制，避免完整收集占满有限窗口。结果预算范围和当前只读 SELECT 支持边界见[公开 API](PIPELINED_EXECUTION_DESIGN.md#公开-api-与后续目标)。
- [新入口验收](tests/fast/test_query_result_runtime.py) 最终 **46 passed**：真实 SQL、位置/具名参数、嵌套类型、空结果、上游未完成时首批可读、部分结果后的错误、容量拒绝、保留视图、取消/期限、排队关闭、清理失败重试、回调重入以及环境变量隔离。
- 共享结果、local 查询、取消/期限及 model-serving 相关回归 **840 passed**（含当时 44 项新入口用例）；惰性 API 调用方迁移回归 **189 passed、4 skipped**。模块级 query 入口补齐后重跑新入口模块，得到上述 46 passed；按用例去重合计 **1031 passed、4 skipped**。跳过项为已有 Arrow BIT/UUID 两项及已禁用的 DuckDB create_function 两项。按要求未运行完整 release gate。
- 当前 C++ 已使用 build/python-release 增量 Release 构建并非 editable 安装。270 个 Python 源码/类型声明文件与 checkout 字节一致；安装的 native 与构建产物 SHA-256 相同。DuckDB SourceID 仍为 `263045b861`，本增量未修改 DuckDB 子树。
- root 格式、ruff、全包 mypy、适用的 pre-commit 检查、源码版权清单、文档相对链接和实际源码包校验通过。新模块已加入 release launcher 与 sdist 清单，供后续 CI 执行。

### P1.1 审查修复（PR #943）

- `LockForQuery` 在持有连接锁后再次检查活动查询，覆盖两个调用同时通过入口检查的竞争窗口；后来的调用立即报错，已有结果流可以完整收集。
- `query()` 在选项、SQL 和参数转换之前保存中断 generation，并在进入准入前检查中断；转换期间接受的 `interrupt()` 抛出原生 `InterruptException`，后续查询仍可正常执行。
- 原生物化结果单独保存 Arrow schema，空结果保持零数据分区；`LocalModelRequest.execute_result().collect()` 因而可以返回保留类型的空表，包含嵌套类型、Decimal 和执行连接的时区。
- 新增 11 项回归，覆盖查询竞争、转换期间中断、空/非空物化结果及结果名额和字节预算释放。
- 6 个相关测试模块 **346 passed**，原生物化结果契约定向检查 **2 passed**；审查者提供的独立复现 **6 passed**。使用增量 Release 非 editable 安装，270 个 Python 源码/类型文件与 checkout 一致，native 与构建产物一致。仅运行相关测试，未运行完整 release/fast 套件。

### P1.1 原生批次读取异常修复（PR #943）

- `QueryContext.read` 在原生读取失败时重新检查取消原因和执行期限，将 Arrow 包装的中断错误恢复为 `RequestCancelled` 或 `RequestExecutionTimeout`，并保留原始异常上下文；没有取消原因时继续传播原始读取错误。
- 新增 4 项回归，在已交付 8192 行后的原生扫描中触发中断或超时，分别覆盖 `read_batch()` 和 `collect()`；同时检查重复读取的异常类型、查询状态、资源释放和连接复用。
- 查询结果、结果交付、请求准入及期限四个相关模块 **178 passed**，审查者提供的独立复现 **2 passed**。本次仅修改 Python、测试及文档；非 editable 安装的 270 个 Python 源码/类型文件与 checkout 一致，native 与既有构建产物一致。格式、ruff、全包 mypy、pre-commit 和源码版权检查通过；未运行完整 release/fast 套件。

### P1.1 默认连接会话资源修复（PR #943）

- 默认连接获取统一拒绝显式 runner/backend，覆盖路径转换后的大小写字符串及 `Path`；`connect(":default:", backend="local", ...)` 抛出 `InvalidInputException`，不会替换已有 runtime 或其资源计费。
- 不带配置的 `connect(":default:")` 继续返回已有连接并共用 runtime。通过创建 local 连接后调用 `set_default_connection` 设置默认连接；普通默认连接也不能通过额外 backend 选项原地升级。
- 新增 13 项回归，覆盖省略 resources、相同及不同容量、字符串及路径写法，并检查活动 cursor 的准入限制、结果名额、完整结果和释放后的复用。
- 查询结果与连接/cursor/默认连接相关验证 **89 passed**，审查者的独立复现 **1 passed**。C++ 已增量 Release 构建并非 editable 安装；270 个 Python 源码/类型文件与 checkout 一致，native 与构建产物一致。格式、ruff、适用的 pre-commit 与源码版权检查通过；未运行完整 release/fast 套件。

### P1.2 原生 DirectExchange 与进程内 TaskService（2026 年 10 月 4 日）

- 每帧拥有独立 native 缓冲，消费窗口包含排队、借出及切片引用；广播物理分配计一次，消费者各自计费。暂时没有额度时返回 BLOCKED，超大单行明确拒绝。成员封闭、严格序号、FINISH、持久错误及消费者关闭分别处理。
- DirectSource/DirectCollector 使用原生 ExecutionBatch 和带 epoch 的弱 task 唤醒；检查与订阅共用通道锁，回调在锁外执行。HASH 继续使用原生分区表达式。sink 保存输出/分区/行游标，重试不重发已经接受的部分；finalize 只封闭生产，不等待输出排空。
- TaskService.prepare 恢复快照并加载显式 binding；start 再次校验 source，对相同 token 幂等；pump 轮转推进真实 native fragment。取消无需等待 pump 操作锁，release 幂等清理上下文。OUTPUT_PENDING 与 FINISHED 依据实际输出 lease 区分，清理保留失败/取消状态。
- 执行期限以 native FINISH 判断生产完成，已完成生产后的执行定时器失效。服务析构和关闭取消定时器、回收原生执行状态；晚释放的 native 切片仍有效。Python 输入回调重入在服务锁及状态修改前被拒绝。
- 新增 **59 项**测试全部通过；连同 P1.1 的 46 项和 P0 的 371 项，最终相关回归共 **476 passed**。涵盖真实 Parquet、字符串/NULL/HASH、单帧窗口、迟到生产者、多输入、广播、部分发送、错误、超大行、取消、期限、原生与定时器清理、回调重入、单/多执行线程及源文件准备后变化。按要求未运行完整 release/fast 套件。
- 初次实现的 C++ 已从 build/python-release 增量 Release 构建并非 editable 安装，当时未修改 DuckDB 子树；后续执行器错误记录修复见下文。新测试已加入 release launcher 和 sdist 清单，供 CI 使用。
- 该次验收的 272 个 Python 源码/类型声明文件与 checkout 字节一致，native 与构建产物 SHA-256 一致；DuckDB SourceID 和 fragment engine identity 保持 P0/P1.1 基线。root 格式、ruff、全包 mypy、pre-commit、源码版权清单、67 个仓库文档链接及实际 sdist 发布校验全部通过。

### P1.2 合入基线与专项审查（PR #944）

- P1.2 已迁到 PR #943 的合入提交 `9f956003ae`；迁移时差异限于原来的 14 个 P1.2 文件，没有重复引入 P1.1 提交。后续的错误保留修复另修改两个 DuckDB 执行器文件，见下文。
- 首轮检查覆盖 channel 条件判断与唤醒注册、部分发送恢复、广播及借用视图计费、并发取消、期限与生产完成、上下文清理；后续审查补充复现了三个异步收尾遗漏，修复见下文。
- 最新基线上的 DirectExchange **59 项**与 QueryResult runtime **67 项**测试全部通过，合计 **126 passed**。仅运行这两个相关模块，未运行完整 release/fast 套件。
- 本次未修改 C++ 或 Python 实现；非 editable 安装的 272 个 Python 源码/类型文件与 checkout 一致，native 与增量构建产物一致。
- root 格式、Ruff、全包 mypy、适用的 pre-commit 检查、源码版权清单与 diff 检查通过。
- PR #944 完成审查并合入后，从最新 integration/pipelined-execution 切独立的 P2.1 分支；跨平台构建与测试由该 PR 的 CI 验证。

### P1.2 异步收尾修复（PR #944）

- 执行期限直接检查每个任务输出生产者的 native FINISH，删除依赖 pump 更新的 `unfinished_production`。后台已完成生产后，不再 pump 或继续持有最后一批数据，都不会导致迟到的执行超时取消结果。
- 控制层在推进和观察任务时检查全部输出的消费者。最后一个消费者关闭后，即使任务仍阻塞于空输入，也会检查执行错误、停止并回收执行器、关闭上游消费端和封闭输出；广播与多输出仍有消费者时继续执行，借用视图的计费持续到实际释放。
- 通道在同一锁下提供生产完成、排空、消费者和错误状态。任务先检查所有输出错误，再判断交付完成；abort 丢弃队列不能把 OUTPUT_PENDING 变成 FINISHED。失败保留原始原因并停止其余任务，直接调用 producer_drained 也会显式报告通道错误。
- 新增 **18 项**回归在旧版本全部失败，修复后全部通过。覆盖空/非空结果、真实定时器及直接过期调用、单/多线程、广播/多输出、保留批次，以及 status/pump/release 三种观察入口。审查者提供的 **3 项**独立复现也全部通过。
- 两个相关模块 **144 passed**（DirectExchange 77、QueryResult runtime 67），加上独立复现共 **147 passed**。仅运行相关测试，未运行完整 release/fast 套件。
- C++ 已增量 Release 构建并非 editable 安装；272 个 Python 源码/类型文件与 checkout 一致，native 与构建产物一致。root 格式、Ruff、全包 mypy、适用的 pre-commit、源码版权及 diff 检查通过。

### P1.2 已关闭 HASH 分区修复（PR #944）

- Sink 在每次计算目标帧大小前调用 HasConsumers；已无消费者的通道直接跳过剩余行。检查仍先报告持久通道错误，其他活跃分区继续交付并遵守帧容量限制。
- 新增 **12 项**测试覆盖分区 0/1、单线程/四线程及关闭、仍开放、报错三种目标状态。旧实现中关闭分区的 **4 项失败**，其余 **8 项对照通过**；修复后全部通过，关闭分区未分配 payload，活跃分区无丢行或重复。
- 两个相关模块 **156 passed**（DirectExchange 89、QueryResult runtime 67）；审查者的复现及对照 **6 passed**，合计 **162 passed**。仅运行相关测试。
- C++ 已增量 Release 构建并非 editable 安装；安装源码及 native 与当前 checkout/构建产物一致。格式、Ruff、全包 mypy、适用的 pre-commit、源码版权与 diff 检查通过。

### P1.2 基础安装与失败优先级修复（PR #944）

- Python 文件系统回调测试在启动子进程前执行 `pytest.importorskip("fsspec")`；可选依赖缺失时只跳过对应的 5 项用例，不影响基础安装的 release gate。
- native 执行器以共享所有权保存 TaskErrorManager，TaskService 在后台任务调度前取得其句柄。接受超时或取消前检查该错误记录和全部输入/输出通道；已经发生的失败优先于后来的停止原因，无需等待 pump 或 context 锁。已有失败即使发生在 FINISH 后也须传播。
- 失败记录独立于上下文清理；发生错误的任务保持 FAILED，关联任务收到原始错误并停止。后续 pump、release、再次取消及超时不会覆盖原始原因；已借出的 batch 在清理后仍有效，最终关闭时归还容量。
- 新增 13 项回归及对照覆盖后台转换失败、真实定时器、直接 expiry、pump/release 两种清理入口、只有消费者任务的输入错误、FINISH 后的输出错误，以及超时中断正在执行的 pump。旧二进制上的定向验证为 **8 failed、4 passed**，并发中断对照随修复后的相关模块一起验证。
- 修复后的相关验证共 **187 passed**：DirectExchange 102、QueryResult runtime 67、审查者独立复现及数据对照 12、原生转换异常和中断 6。缺少 fsspec 的基础依赖环境中，回调测试从 **5 failed** 变为 **5 skipped**。只运行相关测试，未运行完整 release/fast 套件。
- C++ 已增量 Release 构建并非 editable 安装；272 个 Python 源码/类型文件与 checkout 一致，native 与新构建产物一致。DuckDB SourceID 更新为 `346ef5b69e610bdb273d66138e6b4bc995857d2b`；未提交生成的身份清单。root/DuckDB 格式、Ruff、适用的 pre-commit、源码版权及 diff 检查通过。

### P1.2 最后消费者关闭时的输入错误修复（PR #944）

- 任务因失去全部输出消费者而提前收尾时，直接检查所有输入通道的持久错误。输入 abort 可能只唤醒原生任务，尚未被 CheckPulse 观察；已有错误必须先进入失败路径，不能清理后发送成功的 FINISH。
- 新增 12 项回归覆盖单线程/四线程、status/pump/release 三个入口，以及持有或未持有输出批次。两个输入中第一个保持健康、第二个报错，确保检查不会漏掉后面的输入。失败后上下文和上游消费端均释放，原始原因在后续取消、超时和 release 中保留；借出的批次继续有效，关闭后归还容量。
- 新回归在旧二进制上 **12 failed**；审查者的复现及对照为 **3 failed、5 passed**。修复后相关验证共 **189 passed**：DirectExchange 114、QueryResult runtime 67、审查者用例 8。只运行相关测试，未运行完整 release/fast 套件。
- C++ 已增量 Release 构建并非 editable 安装；272 个 Python 源码/类型文件和 native 均与当前 checkout/构建产物一致。本次未改动 DuckDB 子树，engine identity 保持上一轮验证值。root 格式、Ruff、适用的 pre-commit、源码版权及 diff 检查通过。

### P1.2 EOF 后输入错误的统一检查（PR #944）

- DirectSource 不再缓存并永久跳过已结束输入；每轮仍通过 Poll 检查通道，持久错误优先于 EOF/CLOSED。多个输入中 A 已到 EOF 后发生 abort，不能在 B 正常结束时被当作整个任务成功。
- native DirectCollector 在发送 FINISH 前检查全部输入，覆盖 sink 提前停止而没有再次轮询 source 的路径。TaskService 对所有未完成交付的任务检查输入错误，包括 native 执行器已经清理的 OUTPUT_PENDING；错误使状态变为 FAILED，结果通道保留原始原因。
- finalize、任务刷新、取消/期限判断复用同一输入错误检查，删除原来仅在最后消费者关闭分支中的局部检查。错误传播仍使用已有失败及清理路径，借出的批次不会因失败而失效。
- 新增 9 项回归在旧版本为 **8 failed、1 passed**；审查者的复现和对照为 **2 failed、6 passed**。修复后相关验证共 **198 passed**：DirectExchange 123、QueryResult runtime 67、审查者用例 8。覆盖单线程/四线程、后台 native 完成前的通道观察、sink 提前结束、持有批次、status/pump/release 及无错误对照。只运行相关测试，未运行完整 release/fast 套件。
- C++ 已增量 Release 构建并非 editable 安装；272 个 Python 源码/类型文件和 native 均与当前 checkout/构建产物一致，DuckDB 子树及 engine identity 未变。root 格式、Ruff、适用的 pre-commit、源码版权及 diff 检查通过。

### P2 Native Flight、Ray 调度与结果服务（2026 年 10 月 4 日）

- [DirectFlight](src/vane_py/execution/direct_flight.cpp) 接通 native 通道与 Arrow Flight。完整 ticket 固定查询、attempt、双方 epoch、schema、路由和访问 capability；每条流最多一个未确认帧，累计 ACK 归还发送端所有权。独立控制检查覆盖窗口耗尽、提前关闭、断连及 FINISH 后的持久错误。
- [PipelinedWorker/ResultService](vane/execution/pipelined_worker.py) 与 [PipelinedScheduler](vane/execution/pipelined_runtime.py) 实现会话共享 worker 池、整图准备和资源预留、固定 split、消费者握手及逆拓扑启动。准备或清理失败保留资源所有者，成功释放后才退还额度；worker 不自动重启或重放任务。
- 公开 `Runtime(...).connect().query()` 返回相同的 QueryResult。数据通过 root worker → native ResultService → 客户端传输，Ray 只负责控制；慢客户端背压一直传回扫描端。最终 EOF 重新检查全图结局，已完成生产不再被执行期限取消，后台失败会唤醒结果容量等待者。
- 最终相关测试共 **283 passed**：DirectFlight 24、配置/放置 13、DirectExchange 123、QueryResult runtime 67、执行选项 45、真实 Ray 11。覆盖两个独立 native 进程、两个真实 Ray worker、Parquet/HASH、空 schema、小窗口、部分交付后杀 worker/结果服务、预留回滚、清理失败重试、共享会话准入、取消与执行/交付期限。按要求未运行完整 release/fast 套件。
- 当前 C++ 已在 build/python-release 增量 Release 构建并非 editable 安装。275 个 Python 源码/类型文件与 checkout 一致，native 与构建产物 SHA-256 一致；engine identity 为 `346ef5b69e:fragment:18497cb82d5466056628bb30c840bc924f5137d61255a71861e002fa76da51cf`。DuckDB 子树与 fragment codec 没有修改。
- 新增测试已加入 release launcher 和源码包清单。当前 SQL/type 范围延续 P0，原生传输支持 basic types；SQL 参数、聚合/join、模型 UDF 和 FTE 尚未接线。Parquet 使用 worker 共同可访问的绝对路径；客户端须能访问 ResultService 公布的节点地址和动态端口。
- root 格式、Ruff、全仓库 mypy、适用的 pre-commit、源码版权清单、文档本地链接和 diff 检查通过。本地构建与测试平台为 Linux；其余平台交由 CI 验证。

P2 退出条件已满足；后续 P3 已在同一 FragmentGraph、worker 身份与 QueryResult 契约上实现物化 exchange、原子提交和失败重试。

### P2 状态监控审查修复（PR #962）

- 周期监控复用独立的 native production/error 探测，与执行期限和最终 EOF 检查保持一致。详细 task status 会等待 pump 的执行锁，退出存活监控路径；保留监控的 5 秒 RPC 期限及实际执行期限，不通过延长超时掩盖长时间 native 执行。
- 新增真实 Ray 回归：60 秒执行期限下的长过滤查询正常完成，2 秒期限仍抛出 RequestExecutionTimeout，结束后资源账本清空。新增用例在旧版本为 **1 failed、1 passed**；审查者的监控/无监控对照在旧版本也为 **1 failed、1 passed**。修复后监控开启和关闭均约 14.3 秒成功完成。
- 本次相关验证共 **107 passed**：Ray pipelined 13、DirectFlight 24、QueryResult runtime 67、审查者补充用例 3。覆盖故障传播、已完成生产、取消、执行/交付期限、背压与多查询共享 worker。按要求未运行完整 release/fast 套件。
- 修复仅修改 Python 调度和测试，已重新进行非 editable 安装；275 个 Python 源码/类型文件与 checkout 一致，native 二进制及 engine identity 没有变化。root 格式、Ruff、全仓库 mypy、适用的 pre-commit、源码版权清单、文档本地链接及 diff 检查通过。


### P4 分析执行与混跑（2026 年 10 月 6 日）

- 从 PR #963 的合入提交 `a69e60ca43d9` 切出 `feat/analytical-execution`；继续以 `integration/pipelined-execution` 为目标分支。local 仍走原生查询；两种 Ray 策略共用新的分析 FragmentGraph，没有接入旧 runner 或 fallback。
- [native planner](src/vane_py/execution/fragment_plan.cpp) 自动生成 partial/final 聚合、等值 hash join 与小 build 广播、全局 LIMIT/OFFSET、ORDER BY 和 partial/final TopN。拆分前恢复聚合 FILTER 的输入索引，保留优化前的数据源依赖。整数/decimal/时间 AVG 在完整分组内使用原生 finalizer，覆盖大整数 AVG 精度影响过滤结果的回归。
- [原生帧布局](src/vane_py/execution/frame_vector.hpp) 按选中行递归复制全部嵌套子向量，精确计费值缓冲；每层视图保留同一个帧 lease。NULL 列表不读取未定义 offset，超大嵌套行在分配帧前拒绝；测量临时选择向量先检查上限再预留。DirectFlight 与物化 I/O 共用扩展后的 [Arrow codec](src/vane_py/execution/arrow_frame.cpp)。
- [WorkerResourceManager](vane/execution/worker_resources.py) 统一整图和 attempt 的原子预留与 FIFO 排队。小容量混跑测试证明后续 FTE task 不会越过已经等待的 pipelined 图；故障测试同时覆盖资源排队与 FTE 重试，保留队首 task 身份而不遗留等待项。
- native `TaskService.diagnostics()` 通过原子状态观察 BuildReady、输入/输出等待及清理，避免等待执行锁。公开 `QueryResult.diagnostics()` 同时提供查询、task、channel、资源和清理状态，诊断失败只标记 unavailable。
- **相关用例去重合计 808 项通过：756 个非 Ray、52 个真实 Ray。** 完整相关批次为 746 个非 Ray 和 52 个 Ray；最终 AVG 精度修正及新增 10 项用例之后，定向重跑分析编译器/通道 **142 passed**、公开 Ray 分析与故障用例 **9 passed**。覆盖与 native SQL 对照、NULL/空输入、倾斜、过滤聚合、DISTINCT、六类 join、排序/TopN、嵌套类型、重复执行、Parquet、慢消费者、取消/期限和 FTE worker 故障。按用户要求未运行完整 release/fast 套件。
- 当前 C++ 已用 `build/python-release` 增量 Release 编译并非 editable 安装；282 个 Python 源码/类型文件与 checkout 一致，native 与构建产物一致。engine identity 为 `7fc33f6daf:fragment:07178bf8aa7e6049795561d19131eb719d6099fca67155b0fc3fd842b0d2c15e`。
- 新测试已加入 release launcher 和源码包清单；root 格式、Ruff、mypy、适用的 pre-commit、源码版权清单、文档本地链接及 diff 检查通过。没有修改 DuckDB 子树或引入新依赖。验证平台为 Linux，其他平台由 CI 验证。

P4 退出条件已满足。能力范围与聚合精度策略见[详细设计](PIPELINED_EXECUTION_DESIGN.md#native-编译与加载)。FTE 仍按每个对象的声明上限预留存储，多阶段分析计划需要为保留对象及重试留足 query_bytes；不会通过少计配额获得成功。下一阶段为 P5：旧路径删除、支持矩阵及发布/性能验收。

### P4 PR #970 审查修正（2026 年 10 月 6 日）

- 双参数 `MIN/MAX(x,n)` 改为完整分组聚合，保留极值列表、FILTER、NULL、空输入和同节点其他聚合的原生语义；单参数 LIST 输入仍支持 partial/final 合并。
- FTE 取消检查、入队与 attempt 发布统一使用调度锁，取消清除等待项后不会再留下孤立队首。确定性回归在旧实现上分别复现 `close()` 和 `interrupt()` 的遗留等待项；修复后验证共享池归还及后续 `SELECT 7` 成功。
- HUGEINT 在 Flight、物化 I/O 和 Ray 结果导出中统一使用 `decimal256(39,0)`，递归处理 LIST/STRUCT/MAP/ARRAY，覆盖正负边界和中间 SUM；解码拒绝超出 signed 128-bit 范围的值。类型 profile 升为 2，Arrow codec 纳入 fragment build identity。
- 修正 CI 纳秒时间戳测试对可选 pandas 的隐式依赖，改为直接比较 Arrow 数据。在禁止 pandas 导入的进程中，两个定向用例均通过。
- 本轮新增 42 项回归；非 editable 增量 Release 安装上 **748 个非 Ray + 58 个真实 Ray，共 806 项相关测试通过**，未运行完整 release/fast 套件。格式、Ruff、mypy、pre-commit、源码版权清单、文档链接及 diff 检查通过；282 个 Python/类型文件与 checkout 一致，native 与构建产物一致。engine identity 为 `7fc33f6daf:fragment:7b2c9dea48426d648a0a4d09ef8f2b9643c2c95909095e51ea162513128e70ff`。

### P4 时间类型与准入期限修正（基于 `ed28974`）

- INTERVAL 在进入 native Arrow exporter 前转换为月份、天数和原生微秒三个分量，避免微秒乘 1000 溢出；TIME 使用微秒整数，保留 `24:00:00`。两种编码统一覆盖 Flight、物化 I/O、公开 Ray 结果及嵌套子值，类型 profile 升为 3。解码拒绝越界 TIME 和有效 INTERVAL 内的 NULL 分量。
- Pipelined 取得会话名额后保持 `ADMISSION_WAIT`，从请求创建起共用一个准入期限；整图 worker 预留成功才启动执行期限。排队超时保留 `RequestQueueTimeout` 类型，worker 等待计入 queue_wait，未执行的请求不生成 execution sample。旧准入定时器回调与执行启动在状态锁内裁决，取消回调在锁外派发。
- 新增回归覆盖微秒正负极限、月/日极限、午夜结束边界、NULL、空 schema、LIST/STRUCT/MAP/ARRAY、恶意输入、准入超时和排队中断，以及过期回调与执行启动的交接。审查者提供的 5 个真实 Ray 复现用例在旧版本全部失败，修复后全部通过。
- 本轮新增 51 项长期回归。相关验证分批去重合计 **911 passed：843 个非 Ray、63 个仓库 Ray、5 个审查脚本用例**。两个既有取消竞态用例继续断言 FIFO 清空，后续查询的准入期限调整为覆盖新结果 actor 启动，并定向重跑通过。未运行完整 release/fast 套件。
- 已完成非 editable 增量 Release 安装；282 个 Python/类型文件与 checkout 一致，native 与构建产物一致。engine identity 为 `7fc33f6daf:fragment:d8cd56a3a0d20563320ef07e6ce7c4a9fae17656972842ac0078240bd98ca769`。root 格式、Ruff、mypy、pre-commit、源码版权清单、文档链接和 diff 检查通过。

### P4 Parquet TopN 与 DECIMAL 部分和修正（基于 `db99aeb4`）

- fragment 在优化前关闭绑定 Parquet scan 的 late materialization 能力，避免 TopN/LIMIT 引入依赖虚拟 `file_index` 的回查 join。修改仅作用于本次计划的 TableFunction 副本；成功和失败之后，原连接与 sibling cursor 的优化器设置、原生 EXPLAIN 结果均保持一致。显式虚拟列限制继续执行。
- DECIMAL SUM 改为完整分组聚合，避免合法最终结果因 39 位分区部分和无法进入 `decimal128(38,s)` 而失败。原始行按 group key 分区，保留 FILTER、NULL、scale 与同节点其他聚合；代价是交换行数增加，公开 Arrow DECIMAL schema 不变。
- 新增 47 项长期回归：43 个 native 用例覆盖多文件、空分区、TopN/OFFSET、连接设置隔离、正负部分和、带 scale 的数值、分组与空输入；4 个公开 Ray 用例分别验证两种模式。4 个 Ray 用例在旧版本全部失败，修复后全部通过。
- 本轮相关验证去重合计 **587 passed：573 个非 Ray、14 个真实 Ray**。包括 native/分析编译器、提交与源快照校验、分析类型交换、FTE 生成文件列限制，以及公开 Ray 分析 SQL/类型回归。未运行完整 release/fast 套件。
- C++ 已用 `build/python-release` 增量 Release 编译并非 editable 安装；282 个 Python/类型文件与 checkout 一致，native 与构建产物一致。engine identity 为 `7fc33f6daf:fragment:4301d09c42a8f8968382355c4ef454c6a5bf4efb6a97c45a7c93a6806b213959`。root 格式、Ruff、适用的 pre-commit、源码版权清单、文档本地链接和 diff 检查通过；未修改 DuckDB 子树或 Arrow 传输 profile。

### P4 排序聚合、内部改写与 HUGEINT 累加修正（基于 `8ab28102`）

- fragment 在决定拆分前使用 `UnbindSortedAggregate` 恢复被 native 包装的原始参数和 ORDER BY，排序聚合按完整组执行。覆盖敏感浮点顺序、不同类型排序键的 AVG、FILTER、DISTINCT、NULL、空输入及关闭优化器的情况。
- 允许 Binder 为 INTERVAL 分组键生成的 `first`、为 collated MIN/MAX 生成的 `arg_min`/`arg_max`，统一按完整组执行；公开 SQL 的函数集合不变。物理计划反序列化时对参数副本重新绑定函数数据，避免再次包装已计算的 collation 输入列；逻辑反序列化保持原行为。
- HUGEINT 输入的 SUM 改为完整组并使用私有 signed 192-bit 原生累加器，保留最终 HUGEINT 类型，完成后检查范围。确定性验证分别让正数、负数分区先到达，避免只取消 partial 后仍因输入顺序发生 128-bit 中间溢出。普通本地 SUM 与 BIGINT partial/final 策略保持原有选择，Arrow profile 仍为 3。
- 本轮新增 135 项长期回归，相关验证合计 **412 passed：382 个非 Ray、20 个仓库真实 Ray、10 个审查脚本用例**。审查脚本在修复前 10 项全部失败，修复后全部通过。未运行完整 release/fast 套件。
- 已完成非 editable 增量 Release 安装；282 个 Python/类型文件与 checkout 一致，native 与构建产物一致。engine identity 为 `14394740e9:fragment:483580233978e529e45be8a5f6728929a5f020bc7c3ae0d7f89919cdded42b70`。root 与 DuckDB 格式、Ruff、适用的 pre-commit、源码版权清单、文档本地链接和 diff 检查通过。

### P4 嵌套 ARRAY 与宽数值聚合修正（基于 `5d1428cf`）

- Arrow 递归解码按每一层实际数组数量更新 ARRAY 长度，原生容器扩容同步更新普通 ARRAY 的长度元数据。覆盖 LIST、MAP、STRUCT、多维 ARRAY、NULL、空列表及字符串叶子；Image/Tensor 的延迟分配规则保持独立。
- 将 192-bit 累加逻辑抽为 SUM/AVG 共用的原生状态，对 HUGEINT 及使用 128 位存储的 DECIMAL 输入统一启用。DECIMAL SUM 保留 scale 与最终范围检查；AVG 支持总和超过 128 位但平均值合法的情况，并沿用原生 long double 与 scale 处理。宽状态选择和 DECIMAL 输入类型随物理计划序列化。
- 到达顺序回归显式先启动正数或负数 producer，在单线程、四线程下验证 SUM/AVG；另外覆盖 scale 0/5/38、恒定/变化输入、跨批次累加、排序、DISTINCT、FILTER、分组、空输入、精确抵消和真正的最终溢出。
- 新增 182 项回归。本轮相关验证去重合计 **608 passed、2 skipped：584 个非 Ray、24 个真实 Ray 通过**；跳过项为原生 Arrow 测试中已有的 CI FIXME。公开嵌套 ARRAY 用例在修复前两种 Ray 模式均失败，修复后通过。未运行完整 release/fast 套件。
- 已完成非 editable 增量 Release 安装及最终产物的 124 项定向复测（不重复计入总数）；282 个 Python/类型文件与 checkout 一致，native 与构建产物一致。engine identity 为 `a971e0b82e:fragment:51c8a77b708bcdd01f1a16ae73bc0a4e236b4d03b69a61ed38b7e9a356ac4968`。root/DuckDB 格式、Ruff、适用的 pre-commit、源码版权清单、文档本地链接和 diff 检查通过。

### P4 多层 DECIMAL 聚合交换修正（基于 `80e775470c`）

- Arrow codec 区分 native 导入、内部交换与公开结果 schema。128 位 DECIMAL 在 Flight/物化交换中使用 `decimal256(39,scale)`，无损保留 signed 128-bit 系数；公开结果继续按声明的 `decimal128(width,scale)` 导出。更小存储的 DECIMAL 保持原有交换表示，新增宽化副本沿用 staging 预算。
- 宽化与还原递归覆盖 LIST、STRUCT、MAP 键/值和 ARRAY，保留 scale、NULL 和空 schema。解码拒绝超出 native signed 128-bit 范围的系数。Python 提交和 native worker 的类型 profile 同步升为 4，旧交换 profile 无法混用。
- 增加多层 SUM/MAX 回归：内层产生 `1.2×10³⁸` 和 `−9×10³⁷`，经过两层或三层聚合后返回合法结果。两种 Ray 模式覆盖正负系数与 scale 0/5/38，并精确比较最终 Arrow schema。原生/Flight/物化回归覆盖 39 位容器子值、完整 native 系数边界、越界拒绝和空结果精度。
- 本轮新增 66 项长期回归；相关验证分批去重合计 **943 passed：910 个仓库非 Ray、30 个仓库真实 Ray、3 个审查脚本用例**。审查脚本在修复前 1 项通过、两种 Ray 模式失败，修复后全部通过。未运行完整 release/fast 套件。
- 已完成非 editable 增量 Release 安装；282 个 Python/类型文件与 checkout 一致，native 与构建产物一致。engine identity 为 `a971e0b82e:fragment:847f6b376f191fa98542d9fa34d21ba9704a831f8e4f2f97470effa8f169696c`。root 格式、Ruff、mypy、适用的 pre-commit、源码版权清单、文档本地链接和 diff 检查通过。

### P5.1 入口切换与旧路径删除（2026 年 10 月 7 日）

- 从 P4 合入提交 `5d41a675e6` 切出 `refactor/execution-cutover`，目标为 `integration/pipelined-execution`。默认连接使用 native local；首次 query 在会话锁内只发布一个 QueryRuntime，独立 cursor 共享准入计费。Ray 连接通过 query 使用 pipelined/FTE，native SQL/Relation 终端明确拒绝 Ray。
- 删除 Python runners、旧 FTE 调度器、native Ray 计划/任务/worker 绑定、全局配置接口和旧分区结果适配。独立 UDF 服务需要的环境、等待和资源协议移入 execution；模型注册和资源图直接读取原生计划。
- 保留参数、事务、模型复用、取消和结果所有权语义的相关回归，迁移原生 UDF byte-wait 的 146 项算子用例。旧协议专用测试随实现删除。媒体 provider 的跨节点身份测试继续使用独立原生连接验证快照，不再构造旧分布式计划，也不宣称 Ray query 支持媒体 SQL。
- 迁移示例、分析基准、类型声明、包清单及 CI 选择器；移除空的 release cluster-owner 分组，独立制品测试仍由对应 CI/fast 分组执行。Ray 模型 UDF、媒体 SQL、分布式写入和未迁移的历史 GPU 基准不在本轮支持范围内。删除依赖旧默认 Ray runner 的 Cosmos GPU 端到端用例，保留 provider/SQL 绑定回归；其 GPU 验收仍需迁移到注册模型接口。
- 最终非 Ray 定向集合为 **1,795 passed、28 skipped**：初次运行 1,793 项通过，两个 Cosmos 旧执行预期修正后，该模块 20 项全部通过。覆盖编译与提交、分析交换、直接通道/Flight、QueryResult、公开入口、类型与包契约、模型/AI 调用方、GPU 模型准入的 CPU 用例及本地服务验收。另已运行调用方迁移回归和原生 UDF 所有权完整矩阵（146 项通过）。跳过项需要可选 ADBC 依赖或 native_media 制品；未运行 CUDA 硬件验收。
- 真实 Ray 三个相关模块首轮为 **82 passed、1 failed**；失败为慢查询的 Flight 超时，独立复跑成功与超时两个对照用例均通过（**2 passed**）。最终 wheel 的公开入口、分析 SQL 和慢查询/超时复核 **6 passed**。该次超时暂未稳定复现，不据此宣称已完成性能或 release 验收。
- 增量 Release 构建、非 editable 安装、202 个 Python/类型文件及 native 构建产物一致性校验通过。当前 engine identity 为 `cf29aa2922:fragment:847f6b376f191fa98542d9fa34d21ba9704a831f8e4f2f97470effa8f169696c`；DuckDB 子树仅更新两条过时 GPU 错误提示。
- root/DuckDB 格式、Ruff、全包 mypy、安装后的类型用例、适用 pre-commit、源码版权清单、文档本地链接、源码包校验和安装后 Quickstart 通过。fast 全树仅做收集及夹具依赖检查，未执行完整 release/fast 套件。本地平台为 Linux/Python 3.12；跨平台及硬件/制品 CI 仍需验收。

P5.1 已通过 PR #971 合入 `integration/pipelined-execution`，提交为 `b1c363e8d`。最终 Required CI、AI、六个 fast 分组、Linux Python 3.10–3.14、macOS、Windows 及媒体构建检查通过。P5.2.1 从此基线切出 `test/execution-acceptance`，先完成 SQL/type 差分与重复生命周期验收；性能基准及默认容量调整作为下一 PR，之后进行 P5.3 发布资格验证。

### P5.2.1 差分与稳定性验收（2026 年 10 月 7 日）

- 加入 14 类 SQL/type 样本；两个确定 seed、三组分区/线程配置、两种 Ray 模式共完成 **168 次**公开 API 与 native local 对照。覆盖重复文件引用、NULL/空输入、排序/浮点聚合、join/TopN、嵌套 ARRAY、完整 HUGEINT/TIME/INTERVAL 边界及多层 DECIMAL 中间值。
- 12 项原生测试控制 HUGEINT/DECIMAL 生产分片的全部到达排列。真实 Ray 验收重复混跑、保留 Arrow 视图、关闭/取消、排队取消、不支持查询拒绝和两次 worker 丢失，逐步检查结果与资源所有权归零。
- 两种模式各执行两轮长查询成功与两轮执行超时取消，保留 native pump、状态监控和 Flight。本轮历史超时未复现；原生错误现在记录具体数据或控制操作，三个故意延迟服务端的用例验证 `status`、`ack`、数据读取超时归因，未调整超时或失败语义。
- 验收保存 seed、文件顺序与哈希、重放 SQL、版本/engine identity、完成记录及失败现场。新模块已加入 release 清单和 sdist，CI 沿用现有诊断制品上传。
- 最终非 editable Release wheel 上：差分辅助/到达顺序与 Direct Flight **59 passed**，真实 Ray **15 passed**，合计 **74 passed**。本轮仅执行相关测试，未运行完整 release/fast 套件。
- Linux/Python 3.12.14、Ray 2.59.0、PyArrow 25.0.1；202 个 Python/类型文件与 checkout 一致，native 与构建产物 SHA-256 一致。engine identity 为 `cf29aa2922:fragment:847f6b376f191fa98542d9fa34d21ba9704a831f8e4f2f97470effa8f169696c`。
- root 格式、适用 pre-commit、源码版权清单、diff 和源码包校验通过。性能测量、默认容量调整、历史 Flight 超时根因及 P5.3 发布资格仍待完成。

### P5.2.2 可复现性能基准（2026 年 10 月 7 日）

- `scripts/benchmark_execution.py` 通过安装后的公开 `query()` API 测量 native local、Ray pipelined、Ray FTE；记录冷会话、预热、首批、读完/清理、输出吞吐、按行速率消费、共享池混跑和成对故障恢复。完整正确性对照、资源快照与计时分开。
- 基准逐组释放 worker 池，在有限 CPU 的私有 Ray 集群上比较默认与 compact 容量。CLI 从临时工作目录启动，并将已安装包放到 worker 导入路径首位，避免仓库源码遮蔽非 editable wheel。
- 在 `0abbe6dfb6` 的干净源码上完成 10 万行全部场景和 100 万行 warm 场景，后者反转容量配置顺序。每个组合 3 次测量、1 次预热；共 **214 条样本**（含 40 条预热）、**40 次完整结果对照**和 **6 次 worker 故障恢复**，全部成功并归还资源。
- 100 万行扫描中，default/compact 的 pipelined 中位数为 **2.411/4.492 秒**，FTE 为 **5.536/10.972 秒**。compact 减少缓冲预留但牺牲吞吐，因此保留现有默认容量。warm Ray tiny 查询约 1 秒，主要发生在 `query()` 返回之前，需单独分解准备成本。
- 相关测试 **20 passed**：辅助/本地 CLI 18、真实 Ray 全流程 1、从源码目录启动并限定一个 Ray CPU 的独立 CLI 1。格式、适用 pre-commit、源码版权清单、文档链接、diff 和 sdist 校验通过。只运行相关测试，未运行完整 release/fast 套件。
- 本轮未修改生产执行器或 native；使用与 P5.2.1 相同的非 editable wheel。测量边界、环境、脚本/构建身份及详细结果见[执行基准](EXECUTION_BENCHMARKS.md#initial-measurements-2026-10-07)。历史 Flight 超时根因、提交准备开销与 P5.3 发布资格仍待完成。

### P5.2.3 应用级 Runtime 与常驻结果服务（2026 年 10 月 8 日）

- 引入 `Runtime(RayResources(...)).connect(resources=QueryResources(...))`。Runtime 按需建立服务核心，多个 Session 共用 worker、结果服务及存储注册。Session 关闭仅清理自身，Runtime 关闭才停止共享资源。
- ResultService 常驻并同时持有多个独立 ResultContext，控制 RPC 使用唯一 query_id。删除 actor 借还池和隐式每会话服务创建；结果服务死亡不自动替换或重放。失败清理保留上下文与额度，直到成功重试。
- 服务与会话的准入使用同一个队列；结果槽位、缓冲和导出视图同时计入两层预算。新增会话不能绕过服务总额度，Session 的关闭、取消和清理失败不会操作其他会话的查询。
- 规划、协调器及 FTE 续租在本阶段由应用内 QueryService 管理；独立 Server 的进程与远程协议在 P5.2.4 实现。现有 Flight 数据路径保持原生传输。
- 完成两轮无待修问题的代码审查后，一次增量 Release 构建并非 editable 安装。203 个 Python/类型文件与源码一致，native 与构建产物 SHA-256 一致。格式、mypy、版权清单及修改文档的本地链接检查通过。
- 相关验证去重合计 **382 passed：271 个非 Ray、110 个共享集群 Ray、1 个独立基准 CLI**。首批 Ray 测试中新增的保留视图用例把 512 字节裸数据误作完整 IPC 预算；改为可容纳单批但不能同时容纳两批的预算，并释放 Future 持有的 Arrow 引用，重新审查后定向重跑通过。生产代码在构建后未改动，未重新编译，未运行完整 release/fast 套件。
- 既有 `5374cf1c4f` 的性能测量仅代表已替换的 actor 池实现；方法和原始数据保留在[执行基准](EXECUTION_BENCHMARKS.md#result-actor-reuse-2026-10-07)，不能视为当前服务实现的测量结果。历史 Flight 超时及 P5.3 发布资格仍未完成。

### P5.2.4a Flight 会话服务（2026 年 10 月 8 日）

- 新增 `vane-server` / `python -m vane.server`，提供鉴权后的能力发现和会话 Open / Renew / Close。默认回环监听，其他地址要求 TLS；当前只声明 sessions 能力。
- SessionService 独立于传输协议，持有 Runtime、会话租约和创建/清理所有者。打开与关闭中的会话持续计入容量；租约过期不可恢复，失败清理自动重试，阻塞清理不占注册表锁。
- 创建原生连接前记录对应 RayQueryRuntime，覆盖连接创建和回滚同时失败的路径。Close 区分 CLOSING 与 CLOSED；Server 关闭从入口计算等待期限，超时保留后台任务，失败的 Flight shutdown 也可重试。
- 连续两轮无待修问题的审查后，完成一次非 editable 安装；207 个 Python/类型文件与源码一致，原生二进制 SHA-256 未变。格式、适用 pre-commit、mypy 和源码版权清单通过。
- 相关验证去重合计 **85 passed、1 skipped**：会话所有权 21、真实 Flight/鉴权/TLS/独立进程 30、包契约 30、真实 Ray 两模式关闭/过期 4。跳过项需要可选 ADBC 依赖。首次测试仅修正嵌套 cursor 的错误文案断言，并定向复跑通过；安装后未修改生产代码，未重建，未运行完整 release/fast 套件。
- [Server 设计](SERVER_DESIGN.md) 明确后续远程查询与原生结果交付、故障验收，以及 DuckDB 2.0 升级后替换 Quack 对外入口的边界。本步未提供远程 SQL，P5.2.4b / c、历史 Flight 超时及 P5.3 发布资格仍待完成。

### P5.2.4b 远程查询与原生结果（2026 年 10 月 8 日）

- Flight 增加 Execute / Status / Cancel / Finish / CloseQuery；先登记查询所有者再异步执行。会话内连续序号防止回执丢失后重复执行，终态句柄和清理失败记录受全局上限约束。Status 不等待规划锁或 native pump。
- 两种调度器显式接入 NativeResultConsumer。Server 发布固定、可配置 TLS 的原生结果端口，使用独立 capability；客户端不连接 Ray，Server 的 Python 控制层不读取或转发 Arrow 批次。撤销等待旧流退出后才回收容量，覆盖 Arrow 错误路径不调用 Close 的情况。
- `vane.client.Client` 提供 query / submit，会话续租、类型化取消与超时、持有 Arrow 视图的字节计费及可重试关闭。FINISH 后通过服务端确认生产、持久错误与清理；成功交付后的清理失败保留成功结局与资源所有者。
- 两轮无问题审查后进行一次非 editable native 增量 Release 构建。相关测试最终 **270 passed、1 skipped**：非 Ray 203、真实 Ray 66、CLI 1；可选 ADBC 缺失而跳过。首次测试修正不支持的 SQL fixture 与 CLI SIGTERM 注册顺序；再次两轮审查后重新打包 Python，native SHA-256 保持 `dd3706ddb13cee62fa5fc5f26faf3e3fd9b6ebd3f1e4ed88d07e6da3530f205b`，只复跑失败和未运行用例。
- root 格式、lint、mypy、CI copyleft 检查通过；未运行完整 release/fast 套件。P5.2.4c 的客户端失联/服务故障矩阵、混跑性能、历史 Flight 超时定位及 P5.3 仍待完成。
