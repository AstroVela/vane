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

Google 根据实际客户端选择请求上限：Gemini Developer API 最多每批 100 条，Vertex 的 Gemini embedding 模型每次一条，即使 `request_batch_size` 更大也在调用 SDK 前拆分；UDF 的 `batch_size` 不变。

NULL 输入不会发送给 provider。`on_error="ignore"` 将失败行置为 NULL；批量输入错误按需拆分定位，HTTP 413 请求体过大也会拆分恢复，只有拆到单行仍失败的输入输出 NULL。认证失败、账号或计费错误、重试耗尽的限流和服务错误不逐行放大。认证和账号错误依据 SDK 的结构化错误字段识别，包括 Google 以 HTTP 400 返回的 `API_KEY_INVALID`，不会因状态码为 400 而二分拆分。默认 `on_error="raise"`。

Google SDK 的无 HTTP 状态码输入/响应校验错误也支持拆分恢复。例如 SDK 在解析整批响应时因一个坏向量抛出 `ValidationError`，有效的相邻行仍能通过子请求恢复；这类校验失败不做原批重试。默认 `raise` 模式仍立即报错，错误信息不携带 SDK 的输入和响应内容。

返回向量条数不匹配，或 OpenAI 响应含重复、混合缺失、越界等非法索引时，无法可靠对应输入行，`ignore` 模式也会拆分恢复。能明确定位到某行的坏向量只置空该行，有效相邻行无需重新请求。

OpenAI 的 HTTP 429 / `insufficient_quota` 属于终止性配额错误，不重试。Adapter 在生成 Retry-After 重试信号之前检查原始结构化错误，保留这项分类；普通 429 限流仍遵循 `max_retries` 和 Retry-After。

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

Google 模型的裸名称、`models/...`、`google/...`、`publishers/google/models/...` 和 `projects/.../locations/.../publishers/google/models/...` 使用相同的维度、task 与请求上限校验。原始模型名原样发送给 SDK。其他 publisher 和自定义模型资源不会因末尾名称相同而继承内置模型元数据。

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

支持范围为官方 OpenAI 已知 embedding 模型，以及能提供 tokenizer 和有效预处理长度限制的 Transformers 模型。Transformers 的预算包括编码前缀和特殊 token；Router 模型使用实际选中的文本路由，包括 query/document、task/modality 映射及默认路由。长度限制按预处理优先级解析：`max_seq_length`、选中任务的 `query_length` / `document_length`，再应用 `processing_kwargs` 的 `text` 和 `common` 覆盖。

计数匹配选中输入模块的预处理：旧版 Transformer 先拼接提示前缀，再去掉首尾空白；新版纯文本路径保留空白。分块权重也使用预处理后的文本 token 数。无法确认预处理行为、解析路由或有效预算时抛出配置错误，即使设置 `on_error="ignore"` 也不会吞掉。自定义预处理覆盖、聊天模板、query expansion、独立 processor、额外大小写转换及无法对应 token 计数的预处理参数暂不支持显式策略。Google 和未知兼容模型也暂不接受显式策略。Unicode 字符保持完整，截断前缀不保证填满全部 token 预算。

省略 `overlength` 保持旧行为：OpenAI 自动分块合并时仍会归一化；Transformers 继续使用模型默认处理方式。该参数不能与旧 `max_chunk_chars` 同时使用。

RAG 推荐先显式分块并保留 `document_id/chunk_id/text`，再逐块调用 `embed`；`chunk_mean` 得到的是文档级单向量。任何一个远程 chunk 最终失败，整条文档输出 NULL（ignore）或报错（raise）。

## 调试与验证

`vane.ai._embedding_requests` 的 DEBUG 日志提供 worker 累积请求数、重试数、失败输入数、实际/估计 token 数以及请求和排队耗时。失败输入数按请求输入计算，可能包含同一文档的多个 chunk；实际 token 数只在服务返回 usage 时累计。日志不包含文本、向量或凭据。

实现设计见 [AI_EMBEDDING_DESIGN.md](AI_EMBEDDING_DESIGN.md)，开发与测试流程见 [DEVELOPMENT.md](DEVELOPMENT.md)。图片接口使用同一执行和结果契约。


## 图片 embedding 与以文搜图（P3）

`vane.ai.embed_image`、`Relation.embed_image` 和 SQL `ai_embed_image` 接收已解码的 `IMAGE`，输出可空的 `FLOAT[D]`。默认 provider 为 `transformers`，默认模型为 `sentence-transformers/clip-ViT-B-32`，原生维度 512。安装 `pip install 'vane-ai[transformers,image]'`；权重只在 worker 首次处理非空图片时加载。

```python
from vane.ai import embed_image

# images 的 image 列可以是 IMAGE、IMAGE('RGB') 或固定尺寸 IMAGE。
result = images.select(
    vane.col("path"),
    embed_image(vane.col("image"), normalize=True, batch_size=16).alias("embedding"),
)
# Relation 形式保留其他列，也可指定 execution_backend。
result = images.embed_image(vane.col("image"), output_column="image_vector")
```

```sql
SELECT path, ai_embed_image(
    decode_image_file(image_file(path), mode => 'RGB', on_error => 'null'),
    model => 'sentence-transformers/clip-ViT-B-32',
    options => {normalize: true, batch_size: 16}
) AS embedding
FROM image_paths;
```

`VARCHAR` 路径、编码后的 `BLOB`、`IMAGEFILE`、普通 STRUCT 和数组必须先显式解码/转换为 `IMAGE`。解码失败策略属于 `decode_image` / `decode_image_file`：`on_error='null'` 产生的 NULL 会直接跳过模型；embedding 的 `on_error='ignore'` 不拦截上游解码异常。

首版内置图片模型只声明上述 CLIP 模型及简写 `clip-ViT-B-32`；其他模型在规划时拒绝。自定义 provider 可实现 `get_image_embedder()`，返回 `ImageEmbedderDescriptor`，运行时接收保持行序的 HWC NumPy 数组。它可以同步返回或返回 awaitable，沿用 executor 的异步生命周期。

CLIP 接收 UInt8 的 L、LA、RGB、RGBA 图片，使用 Pillow 转为 RGB，再交给模型自带 processor 完成缩放、裁剪和像素归一化。UInt16/Float32 图片须先显式 `convert_image(image, 'RGB')`；不会猜测高位深像素的取值范围。无需预先 resize 到模型尺寸。生成的临时 Pillow 图片在成功或异常路径都会关闭；Arrow 输入读取使用像素缓冲区，不逐像素创建 Python 对象。

支持 `normalize`、`batch_size`、`actor_number`、`max_retries` 以及 Transformers 加载选项 `device`、`cache_folder`、`local_files_only`、`revision`、`trust_remote_code`。`execution_backend` 仅用于 Relation。图片不接受文本模板、token 分块或远程请求并发参数。默认 CPU、关闭 remote code；显式启用 remote code 仍要求完整 commit SHA。与文本侧相同，可显式截取较小的 `dimensions`，但 CLIP 不保证维度截取后的检索质量。

NULL 不加载模型，不发起推理。`on_error='ignore'` 隔离图片推理失败的行，维度错误及非有限向量遵守既有校验规则；合法邻行保持原行序。配置错误不能通过 `ignore` 绕过。

以文搜图时，文本侧必须显式选择同一 CLIP 模型，并保持 `revision`、`dimensions` 和归一化配置一致。文本侧默认 MiniLM/OpenAI 向量不能与 CLIP 图片向量混用。首版 CLIP 的文本侧沿用 SentenceTransformers 默认 token 截断；显式 `overlength` 会拒绝无法确立预算的 CLIP processor，避免假定它是普通文本 tokenizer。

完整示例见 [image_embedding_search.py](examples/image_embedding_search.py)：读取本地图片，显式解码，物化图片向量一次，再用文本向量和 `array_cosine_similarity` 返回 Top K。

```bash
python examples/image_embedding_search.py ./photos 'a dog in the snow' --top-k 5
# 两个 encoder 固定到同一权重版本：
python examples/image_embedding_search.py ./photos 'a red car' --revision <full-commit-sha>
```
