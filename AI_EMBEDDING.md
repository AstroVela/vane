# AI embedding

`vane.ai.embed`、`Relation.embed` 和 SQL `ai_embed` 为每行文本生成一个可空的 `FLOAT[D]` 向量。模型和参数在规划时确定，SDK 客户端和模型在 worker 上首次执行时创建；构建表达式、查看 schema 和 EXPLAIN 不调用远程模型。

## 远程批处理

```python
import vane
from vane.ai import embed

result = documents.select(
    vane.col("id"),
    embed(
        vane.col("text"),
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

- `batch_size` 是每批 UDF 输入行数。
- `request_batch_size` 是每个 HTTP 请求的最大输入数，同时受服务上限和 token 预算约束。OpenAI 和 Google 均支持。
- `max_concurrency_per_actor` 是每个 actor 的在途请求上限，默认 1。它控制并发数，不代表账号级 RPM/TPM 配额。
- `max_retries` 是每个失败请求首次尝试之外的重试次数。成功的子请求不会因为另一子请求失败而重发。分布式任务恢复仍可能重发请求，不提供远程 exactly-once 保证。
- `normalize=True` 对最终向量做 L2 归一化；默认 false，零向量保持不变。

NULL 输入不会发送给 provider。`on_error="ignore"` 将失败行置为 NULL；批量输入错误按需拆分定位，认证失败、重试耗尽的限流和服务错误不逐行放大。默认 `on_error="raise"`。

SQL 参数保持相同含义：

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

`options` 必须是常量 STRUCT。Expression、Relation、SQL 都支持上述参数；`output_column`、`execution_backend` 和旧字符分块选项仍为 Relation 专用。

## 固定维度的兼容服务

未知 endpoint 的维度必须显式声明。如果服务不接受请求中的 `dimensions` 字段，可以只声明结果维度：

```python
vector = embed(
    vane.col("text"),
    provider="openai",
    model="my-fixed-embedding-model",
    dimensions=768,
    base_url="http://localhost:8000/v1",
    supports_overriding_dimensions=False,
)
```

这时 schema 仍为 `FLOAT[768]`，每行结果仍校验为 768 个有限 float32 元素，但请求不发送 `dimensions`。省略该开关保留原行为。官方已知模型不能用该开关声明与原生维度不同的结果。

## 检索编码

Google 在支持的模型上将 `input_type="query"` / `"document"` 映射为相应 retrieval task type，与显式 `task_type` 互斥。模型不支持该参数时直接拒绝。

Transformers 支持 `input_type`、`prompt_name` 或 `prompt`，三者互斥：

```python
document_vector = embed(
    vane.col("text"),
    provider="transformers",
    model="your-reviewed-model",
    dimensions=768,
    prompt="passage: ",
    device="cuda",
)
```

`input_type` 要求加载的模型具有对应命名模板：query 使用 `query`；document 按 `document`、`passage`、`corpus` 顺序查找。存在对应 `encode_query` / `encode_document` 方法时同时使用该方法，否则使用带模板的 `encode`。没有已声明模板时拒绝该选项，可改用模型说明中的显式 `prompt`。模板缺失属于配置错误，`on_error="ignore"` 不会隐藏它。

本地模型每个 actor 串行编码，`max_concurrency_per_actor` 只接受 1。文档与查询侧需要使用匹配的模型、revision 和编码配置；相同维度不保证向量空间相同。

## 显式长文本策略

`overlength` 可取：

| 值 | 行为 |
| --- | --- |
| `error` | 超出单条输入预算时报告行错误 |
| `truncate` | 取 tokenizer 预算内的文本前缀 |
| `chunk_mean` | 无重叠分块，按有效文本 token 数加权平均，是否归一化由 `normalize` 控制 |

支持范围为官方 OpenAI 已知 embedding 模型，以及能提供 tokenizer 和 `max_seq_length` 的 Transformers 模型。Transformers 的预算包括编码前缀和特殊 token；Router 模型使用实际选中的文本路由的 tokenizer 和长度限制，包括 query/document、task/modality 映射及默认路由。无法解析该路由或其元数据时抛出配置错误，即使设置 `on_error="ignore"` 也不会吞掉。Google 和未知兼容模型暂不接受显式策略，避免把字符估计当作精确 token 保证。Unicode 字符保持完整，截断前缀不保证填满全部 token 预算。

省略 `overlength` 保持旧行为：OpenAI 自动分块合并时仍会归一化；Transformers 继续使用模型默认处理方式。该参数不能与旧 `max_chunk_chars` 同时使用。

RAG 推荐先显式分块并保留 `document_id/chunk_id/text`，再逐块调用 `embed`；`chunk_mean` 得到的是文档级单向量。任何一个远程 chunk 最终失败，整条文档输出 NULL（ignore）或报错（raise）。

## 调试与验证

`vane.ai._embedding_requests` 的 DEBUG 日志提供 worker 累积请求数、重试数、失败输入数、实际/估计 token 数以及请求和排队耗时。失败输入数按请求输入计算，可能包含同一文档的多个 chunk；实际 token 数只在服务返回 usage 时累计。日志不包含文本、向量或凭据。

实现设计见 [AI_EMBEDDING_DESIGN.md](AI_EMBEDDING_DESIGN.md)，开发与测试流程见 [DEVELOPMENT.md](DEVELOPMENT.md)。图片 embedding 仍是后续独立入口。
