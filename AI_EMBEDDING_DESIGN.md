# AI Embedding 增量设计

状态：文本 embedding 的 P0–P2 已实现；实际能力范围与用法见 [AI_EMBEDDING.md](AI_EMBEDDING.md)。图片 P3 保留为后续设计。调研日期：2026-09-18。

## 1. 结论与范围

沿用 `vane.ai.embed`、`Relation.embed` 和 SQL `ai_embed`，复用现有 AI expression、Descriptor、actor 和 Arrow 批处理路径。第一阶段完善文本 embedding，不新建一套推理执行器，也不把 embedding 转成 Prompt 请求。

优先解决四个问题：远程请求并发、输出维度与请求降维的区分、检索任务编码、长文本行为的显式控制。图片 embedding 作为后续独立入口；音视频、多向量和稀疏向量暂不纳入第一阶段。

## 2. 调研基线

Daft 参考的是调研时远端 HEAD：[`1feced9b5af78586da19dc10d32c5bc0c25855e0`](https://github.com/Eventual-Inc/Daft/commit/1feced9b5af78586da19dc10d32c5bc0c25855e0)，提交日期 2026-09-17。以下判断来自该提交源码，不以 stable 文档代替最新代码。

| 关注点 | Daft 基线 | Vane 当前实现与设计取舍 |
| --- | --- | --- |
| 公共入口 | `embed_text`、`embed_image`，构建列表达式 | 已有文本 `embed`，保留现有名称和默认 provider |
| 执行 | Descriptor + class UDF，区分同步和异步调用 | 已有同类结构，继续复用 actor / executor async runtime |
| 向量类型 | `EmbeddingDimensions` 描述定长类型 | 已有 `FLOAT[D]` 与 Arrow fixed-size list，继续使用 |
| 本地模型 | SentenceTransformers 批量编码 | 已实现，补齐编码选项和资源控制 |
| 远程服务 | OpenAI 异步批请求、token 预算、长文本合并、usage 指标 | 已有批请求及长文本处理，补齐受控并发和指标 |
| 维度发现 | 部分路径读取 HF 配置或请求服务探测维度 | Vane 继续要求可信元数据或显式维度，绑定阶段无网络与模型加载 |
| 请求参数 | 提供 `supports_overriding_dimensions` 和 `extra_body` | 借鉴维度区分；继续采用封闭、可校验的 options |

源码依据：[函数入口](https://github.com/Eventual-Inc/Daft/blob/1feced9b5af78586da19dc10d32c5bc0c25855e0/daft/functions/ai/__init__.py)、[OpenAI adapter](https://github.com/Eventual-Inc/Daft/blob/1feced9b5af78586da19dc10d32c5bc0c25855e0/daft/ai/openai/protocols/text_embedder.py)、[Transformers adapter](https://github.com/Eventual-Inc/Daft/blob/1feced9b5af78586da19dc10d32c5bc0c25855e0/daft/ai/transformers/protocols/text_embedder.py)、[options](https://github.com/Eventual-Inc/Daft/blob/1feced9b5af78586da19dc10d32c5bc0c25855e0/daft/ai/typing.py)。

Vane 当前实现重点：

- [functions.py](vane/ai/functions.py)：`_prepare_embed_call`、`_EmbedTextBatch`、三个 Python 调用形态。
- [protocols.py](vane/ai/protocols.py)：`TextEmbedder` 与可序列化的 `TextEmbedderDescriptor`。
- [options.py](vane/ai/options.py)：provider options 白名单及校验。
- [_sql.py](vane/ai/_sql.py) 与 [ai_sql_functions.cpp](src/vane_py/ai_sql_functions.cpp)：SQL 绑定、定长返回类型、expression UDF lowering。
- [openai.py](vane/ai/providers/openai.py)、[google.py](vane/ai/providers/google.py)、[transformers.py](vane/ai/providers/transformers.py)：现有三类文本 provider。

现有实现已经处理 NULL 跳过、行数保持、结果维度和有限数校验、OpenAI 响应 index 重排、失败行隔离、worker 客户端生命周期。上述能力是保留的基础，不重复列为新功能。

## 3. 公共接口

保留主签名、Relation 重载和 `provider="openai"` 默认值：

```python
embed(text, *, provider="openai", model=None, dimensions=None,
      on_error="raise", **options) -> Expression

embed(rel, text, *, output_column="embedding", ...) -> Relation
rel.embed(text, *, output_column="embedding", ...) -> Relation
```

以下是目标接口示例；`request_batch_size`、`max_concurrency_per_actor` 用于 Embed 的支持属于本设计新增：

```python
from vane import col
from vane.ai import embed

documents.select(
    col("id"),
    embed(
        col("text"),
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1024,
        batch_size=256,
        request_batch_size=64,
        actor_number=2,
        max_concurrency_per_actor=4,
        normalize=True,
    ).alias("embedding"),
)
```

SQL 保持现有六参数宏，新增控制项放入常量 STRUCT：

```sql
SELECT id, ai_embed(
    text,
    provider := 'openai',
    model := 'text-embedding-3-small',
    dimensions := 1024,
    options := {
        'batch_size': 256,
        'request_batch_size': 64,
        'max_concurrency_per_actor': 4,
        'normalize': true
    }
) AS embedding
FROM documents;
```

`provider/model/dimensions/on_error/options` 继续在绑定时确定，只有文本是逐行输入。Expression、Relation、SQL 共享语义和校验。`output_column`、`execution_backend` 以及旧字符分块配置仍保持现有 Relation 专用边界。

## 4. 新增 options 与类型契约

| 参数 | 建议语义 | 默认和兼容性 |
| --- | --- | --- |
| `request_batch_size: int` | 每次远程请求的最大输入数；同时受 provider 上限与 token 预算约束 | 未设置时沿用已有批处理大小并限制到服务上限 |
| `max_concurrency_per_actor: int` | 每个 actor 的远程在途请求上限，与 Prompt 使用同一名称 | 默认 1；本地模型仅接受 1，避免并发访问同一模型 |
| `supports_overriding_dimensions: bool` | OpenAI-compatible endpoint 是否发送 `dimensions` 请求字段 | 显式 false 时只声明结果 schema；省略保持当前行为 |
| `input_type: "query" \| "document"` | 请求 provider/model 对应的检索编码方式 | 默认不指定，保持原编码路径；不支持时明确报错 |
| `prompt_name: str` / `prompt: str` | Transformers 的已命名编码模板或显式编码前缀 | 两者互斥，也与 `input_type` 互斥 |
| `overlength: "error" \| "truncate" \| "chunk_mean"` | 超过单条输入预算时的处理方式 | 省略保留旧 provider 行为，新文档示例推荐显式指定 |

所有新增项进入 `EmbedOptions` 和对应 provider 白名单；不会将执行参数透传给 SDK。原有 `normalize=False`、`on_error="raise"`、默认 batch size 和 actor 数保持不变。

输出固定为可空 `FLOAT[D]`：每个非空向量恰有 D 个有限 float32 元素，不接受含 NULL 元素、NaN、Inf 或长度错误的向量。运行时不能以首行结果改变 schema。

`dimensions` 同时涉及两种概念，内部必须拆开：

- `output_dimensions`：规划和结果校验所需的 D，始终已知。
- `request_dimensions`：是否向服务请求指定维度，可以为空。

Google Descriptor 已有类似拆分；OpenAI adapter 对齐这一结构。固定输出维度的兼容服务可传 `dimensions=D, supports_overriding_dimensions=False`，既不做网络探测，也不发送服务不支持的字段。官方模型使用本地能力表校验，不能靠该开关绕过已知模型能力限制；不支持降维时只允许声明与原生维度相同的 D。

Transformers 继续保留已有显式 `truncate_dim` 行为，但文档需说明截断是否保持检索质量取决于模型。不能因为返回维数相同，就认为不同模型或不同 revision 的向量可比较。

## 5. 规划与运行时

```text
Python Expression / Relation / SQL ai_embed
                  |
          _prepare_embed_call
     校验参数、解析能力、确定 FLOAT[D]
                  |
       Descriptor + execution options
                  |
       现有 expression UDF / actor
                  |
  NULL 过滤 -> 输入处理 -> 请求分包 -> provider
                  |
       结果校验 -> 重排行序 -> 可选归一化
                  |
        Arrow fixed_size_list<float32, D>
```

Descriptor 只携带可序列化配置；绑定和 EXPLAIN 不创建客户端、不请求 endpoint、不下载或加载权重。SDK client 和本地模型在 worker 首次执行时初始化，actor 重用；异步客户端的创建、调用、关闭都在现有 executor 绑定的事件循环内完成。

保留现有 `TextEmbedder.embed_text(list[str])` 协议。为内置远程 adapter 提供共享请求执行辅助类，由 adapter 按服务限制分包并提交请求；辅助类负责有界并发、请求级重试、取消和指标。自定义 provider 继续走现有协议；未声明支持的新增能力直接拒绝，避免新增抽象方法破坏现有插件。

批处理分三层：`batch_size` 决定 UDF 输入行数，`request_batch_size` / token budget 决定 HTTP 分包，`max_concurrency_per_actor` 决定在途请求数。跨 actor 总并发近似为 actor 数乘以单 actor 上限；这是并发控制，不等于账号级 RPM/TPM 限流。

只在内置 adapter 内并发独立请求，不并发重入整个 `_EmbedTextBatch.__call__`。固定数量的 async worker 从共享迭代器领取请求，避免一次创建与全部 chunk 数量等长的 task 列表。保存原行号、chunk 编号以及响应 index；按映射还原结果，不使用完成顺序作为输出顺序。

本地模型继续批量 `encode`，一次 actor 调用串行执行。现有 `device="cuda"` 已声明 GPU 资源；第一阶段沿用一份模型对应一个 actor 的资源约束，显存和吞吐数据充分后再考虑小数 GPU、多模型共卡或专用 native embedding operator。

## 6. 检索编码与长文本

`input_type` 由 adapter 映射到所选模型支持的语义。Google 在支持的模型上映射到 `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT`，与显式 `task_type` 互斥；Transformers 使用模型支持的 query/document 编码方法或已声明模板。其他服务没有对应能力时拒绝，不能静默忽略或通用地拼接 `query:`。

Transformers 加载后才能确认的模板能力在 worker 初始化阶段校验。`prompt_name` / `prompt` 与模型 revision 一并构成 embedding 配置身份，便于文档和查询侧使用成对配置。

当前存在两层长文本行为：Relation 的字符分块加权平均，以及 OpenAI adapter 的 token 预算分块合并。新增策略必须避免两层重复执行：显式 `overlength` 与旧 `max_chunk_chars` 配置互斥；省略新参数时保留原行为。

新增策略定义如下：

- `error`：超长输入进入行级错误处理。
- `truncate`：使用对应模型 tokenizer 按输入预算截断，明确包含模板和特殊 token 的预算。
- `chunk_mean`：按 token 预算生成无重叠 chunk，按有效 token 数加权合并，再根据 `normalize` 决定是否 L2 归一化；任一 chunk 最终失败则整行失败。

精确截断和 token 分块需要匹配的 tokenizer；不可用时拒绝显式策略，不能用字符数估计伪装成精确 token 保证。Transformers Router 的顶层 tokenizer 和最大长度可能来自不同路由，因此必须沿实际编码任务和文本 modality 解析输入模块。预算解析遵循该模块的预处理优先级：从 `max_seq_length` 开始，应用实际收到的 query/document 任务长度，再应用 `processing_kwargs.text` 与 `processing_kwargs.common` 的长度覆盖；显式 `max_length=None` 恢复 tokenizer 默认长度。旧 Router 消费而不转发 task 时，不应用下游任务长度。

当前显式策略支持可直接计数的纯文本 tokenizer 路径。聊天模板、query expansion、独立 processor、额外大小写转换或未知 tokenization 参数均拒绝，避免预检查与实际编码使用不同输入。无法解析路由或有效预算属于配置错误，不被 `on_error="ignore"` 吞掉。旧路径的估计方法继续作为兼容行为。零向量归一化保持零向量，避免除零。

RAG 推荐显式先分块，保留 `document_id/chunk_id/text`，再逐块调用 `embed`。`chunk_mean` 返回文档级单向量，与逐块建索引的检索语义不同。旧分块路径会自行归一化，即使外层 `normalize=False`；这项兼容行为应记录并单独迁移，不能在补并发时悄悄更改。

## 7. 错误、重试与观测

NULL 输入直接输出 NULL，不初始化 provider；空字符串属于有效字符串输入，由 adapter 按模型能力处理，不能自动替换为空向量。`on_error="ignore"` 输出失败行 NULL，`raise` 终止执行。绑定参数、类型和执行器配置错误不被 ignore 吞掉。

Vane 保持唯一的重试责任方，SDK retries 关闭。内置远程 provider 新路径按实际请求重试，外层不得再次重试已管理请求；自定义 provider 的旧调用路径保持兼容。`max_retries` 表示每个请求首次尝试之外的重试次数。

429、可恢复网络错误和可重试服务端错误使用退避、抖动与 Retry-After；认证、模型能力、schema 和确定性输入错误不做同样重试。先检查 SDK 结构化错误字段中的认证、账号和计费错误，再判断重试和行级隔离；例如 Google 的 `API_KEY_INVALID` 即使使用 HTTP 400，也只失败一次，不触发二分拆分。错误消息或任意 metadata 中的输入文本不参与分类。批量输入错误可以二分定位失败行；认证失败或耗尽重试的限流不得触发逐行请求放大。取消时停止排队请求，取消并等待在途 task 完成清理。

保留成功子请求结果，只重试失败子请求。分布式故障恢复仍可能再次调用远端，因此不承诺外部请求 exactly-once，也不默认跨查询缓存 embedding。

共享请求执行器记录 worker 累积请求数、重试数、失败输入数、token usage、估计 token、请求耗时和排队时间。失败输入数按请求输入计算，可能包含同一文档的多个 chunk；实际 usage 只在服务返回时累计，与估计值分开。输入/NULL 行数和模型加载时间未纳入这组指标。日志和 EXPLAIN 继续遵守现有脱敏约束。

## 8. 后续图片入口

参考 Daft 的模态拆分，后续新增 `vane.ai.embed_image(image, ...)` 与 SQL `ai_embed_image`，复用执行基础设施，增加 `ImageEmbedderDescriptor`，不把 BLOB 或图片对象隐式转成文本。

图片输入类型需要先与仓库 native media 的类型及解码契约对齐；不能直接复用 Prompt 的任意多模态消息列表。模型必须显式声明支持图片编码。跨模态检索要求文本/图片分支来自匹配模型和预处理配置；维度相同不能证明向量空间相同。

## 9. 实施顺序与验收

| 阶段 | 交付 | 主要改动位置 |
| --- | --- | --- |
| P0 | 维度声明与请求降维拆分、options 契约、保持三个入口一致 | `options.py`、OpenAI Descriptor、`functions.py`、`_sql.py` |
| P1 | 请求分包、有界并发、请求级重试和指标 | 共享远程请求辅助模块、OpenAI/Google adapter、现有 async runtime 接入 |
| P2 | 检索编码、显式超长策略、RAG 示例 | Transformers/Google adapter、输入处理模块、文档与 typing |
| P3 | 图片 embedding | 图片协议、media 绑定、Python/SQL 入口 |

P0–P2 的目标是不改 SQL 宏签名，不新增 C++ 物理执行器；如实际实现需要改变 native lowering，按 [DEVELOPMENT.md](DEVELOPMENT.md) 做增量构建。不要为选项扩展直接进入 native embedding operator 开发。

验收覆盖：

- Python Expression、Relation、SQL 对同一配置具有相同类型和输出语义；未知模型显式声明维度，绑定与 EXPLAIN 无 I/O。
- 全 NULL、混合 NULL、空串、非法结果、响应乱序、缺失/重复 index、float32 溢出、零向量。
- 固定维度兼容 endpoint 不接受 `dimensions` 时可正常运行，结果维度仍严格验证。
- instrumented fake client 验证真实在途请求数、分包上限、行序恢复、取消清理和成功子请求不重复发送。
- 429 不逐行放大，认证错误不重试，行级错误隔离；SDK 与 wrapper 不叠加重试。
- query/document 模板映射、互斥参数、token 边界、chunk 合并与旧默认行为的兼容性。
- 本地和 Ray actor 的序列化、模型复用及客户端关闭；必要的真实 Ray 测试按仓库约定标记。

实现时先运行受影响的 AI / async runtime / typing 测试，再执行 `scripts/run_release_tests.sh`；完整 fast suite 使用 `scripts/run_fast_tests.sh`。真实 provider、模型下载和 GPU 测试作为有条件的独立验证。实现的验收结果以实际运行的测试为准。
