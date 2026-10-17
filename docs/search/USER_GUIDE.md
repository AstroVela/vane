# Vane 全文检索用户文档

Vane 的 `vane.search.FullTextIndex` 使用 Rust Tantivy 的 Python 绑定建立本地持久化倒排索引。它可以接收 Vane Relation 或 Arrow 数据，按 BM25 返回相关文档，再通过 Arrow 与 Vane SQL、过滤、关联分析组合使用。

## 安装

在包含本功能的 Vane 版本中安装：

```bash
pip install 'vane-ai[search]'
```

可选依赖为 `tantivy>=0.25.1,<0.26`。普通 `import vane` 不会加载 Tantivy；仅创建或打开检索索引时才加载。开发分支的构建与安装遵循仓库 [DEVELOPMENT.md](../../DEVELOPMENT.md)，不要使用 editable install。这里的例子需要已安装包含本功能的 Vane wheel。

## 从 SQL 数据建立索引并检索

以下例子使用临时目录，可以直接重复执行：

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import vane
from vane.search import FullTextIndex

with TemporaryDirectory() as directory, vane.connect() as con:
    documents = con.sql("""
        SELECT * FROM (VALUES
          ('doc-1', 'Rust search engine', 'Tantivy provides fast full text search'),
          ('doc-2', 'Python analytics', 'Vane combines SQL and multimodal data'),
          ('doc-3', 'Distributed search', 'Build search pipelines with Vane')
        ) AS docs(id, title, body)
    """)
    index = FullTextIndex.create(
        Path(directory) / 'articles',
        id_column='id',
        text_columns=['title', 'body'],
    )
    assert index.upsert(documents, batch_size=8192) == 3

    hits = index.search('search', limit=10)
    print(hits.to_pylist())
    # Arrow 列：id VARCHAR, _score FLOAT, title VARCHAR, body VARCHAR
    con.register('search_hits', hits)
    print(con.sql('''
        SELECT id, title, _score FROM search_hits
        WHERE _score > 0 ORDER BY _score DESC
    ''').fetchall())

    # 结果也可以直接进入 Relation API。
    print(con.from_arrow(hits).project('id, title').fetchall())
    reopened = FullTextIndex.open(Path(directory) / 'articles')
    print(reopened.search('title:Rust').to_pylist())
```

生产索引应放在稳定的本地目录，例如应用自己的数据目录。`create()` 只接受尚不存在的目标目录，连空目录也不会覆盖；`open()` 用于重新打开已有索引。路径是调用进程所在机器的文件系统路径，不接受 S3/HTTP 索引存储。

## 输入约束与更新

索引创建时确定一个主键列、一个或多个文本列及分词器，之后不能修改。列名必须是 ASCII 标识符，主键和文本列不能重名，`_score` 与 `_vane_id` 是保留名。

`upsert(source, batch_size=8192)` 支持：

- Vane Relation：通过 `to_arrow_reader()` 消费结果，结束后关闭本次创建的 reader。
- `pyarrow.Table`、`pyarrow.RecordBatch`、`pyarrow.RecordBatchReader`。调用方提供的 reader 仍由调用方管理，读取位置会前进。

所有配置列必须存在，Arrow 类型必须是 `string` 或 `large_string`。额外列被忽略，文本值允许 `None`，主键必须是非空字符串，最多 512 个 UTF-8 字节，不能包含 NUL。数字主键请先显式转换成字符串，例如 SQL `CAST(id AS VARCHAR)`；`'001'` 与 `'1'` 是不同主键。全 NULL 文本列也要显式指定字符串类型。

```python
import pyarrow as pa

# 假设 index 是上文创建并仍在使用的索引。
updates = pa.table({
    'id': ['doc-1', 'doc-4'],
    'title': ['Updated Rust guide', 'New search guide'],
    'body': ['Tantivy indexing and ranking', 'Full text retrieval'],
})
processed_rows = index.upsert(updates)
processed_ids = index.delete(['doc-2', 'missing-id'])
```

更新采用完整文档替换：同一主键的旧文本不再参与检索，传入 NULL 会清空对应文本。同一次调用中重复主键按输入顺序采用最后一行；需要确定性的结果时，应先在 SQL 中去重或排序，尤其是分布式来源。返回值分别是处理的输入行数、输入 ID 数，包含重复值；它们不是净新增数量或实际删除数量。删除不存在的 ID 成功且无副作用。

每次 `upsert` 或 `delete` 是一个独立 Tantivy 提交。参数校验或数据流在提交前失败时，整个调用回滚，之前成功提交的内容保留。`batch_size` 控制逐批转换，不会把一次调用拆成多个可见提交。空输入合法。

## 查询

```python
index.search('rust')                          # 默认在所有文本列中查询
index.search('rust python')                   # 默认 OR
index.search('rust python', conjunction=True) # 默认 AND
index.search('rust AND search')
index.search('rust OR python')
index.search('search NOT distributed')
index.search('"full text"')                   # 短语，保存词项位置
index.search('title:Rust')                    # 显式限定字段
index.search('search', fields=['body'])        # 设置未限定词项的默认字段
index.search('search', limit=20, offset=20)
```

`fields` 只决定未指定字段的词项在哪些文本列搜索，不是访问控制；显式字段查询仍能访问索引中的其他字段。语法由 Tantivy QueryParser 解析，错误语法或未知字段会抛出 `ValueError`，不会静默转成普通字符串匹配。空白查询也被拒绝。

返回一个 `pyarrow.Table`，依次包含主键列、`_score: float32` 和所有文本列。无匹配时返回相同 schema 的空表，NULL 文本保留为 NULL。BM25 分数降序排列；相同分数的顺序以及写入/合并后的分页顺序不保证稳定。SQL 对结果重新处理时请显式 `ORDER BY _score DESC`。

`limit` 必须为正整数，`offset` 必须为非负整数，且 `limit + offset <= 10000`。结果会在调用进程中物化，长文本和较大的结果窗口会增加内存开销。分页不是跨提交快照，也不提供全量导出功能。当前接口没有总命中数、highlight 或 snippet。

先选出 top-k 再在 SQL 中过滤可能得到少于 k 条结果，不能保证等价于“先对全集过滤再取 top-k”。业务必须按需要预过滤建索引，或选择足够大的候选集并接受该限制。

## 分词与中文

| `tokenizer` | 行为 | 示例 |
| --- | --- | --- |
| `default` | 按字母/数字词项切分，转小写，过滤超长词项 | `Hello` 可用 `hello` 查找 |
| `en_stem` | 默认处理加英文词干提取 | `running` 可用 `run` 查找 |
| `whitespace` | 以空白切分，保留大小写 | 适合预分词中文 |
| `raw` | 整段作为一个词项，区分大小写 | 适合精确文本 |

默认分词器不提供中文词典分词，也不是任意子串搜索。中文可在上游统一分词，用空格连接词项，并在创建索引时选择 `tokenizer='whitespace'`，例如索引 `'中文 全文 搜索'`，查询 `'全文 AND 搜索'`。查询端也应采用相同预处理。不要把整段无空格中文的默认索引当作已完成中文分词。

分词器在索引元数据中持久化，重新打开后保持一致。变更字段或分词规则需要在新目录重建索引。

## 一致性、并发与部署

每次检索先刷新已提交的索引，再用同一个 searcher 读取匹配项和存储字段。成功写入后的下一次查询可见该提交；其他句柄的后续查询也会刷新。一次查询过程中发生的提交不会改变本次快照。

一个目录同时只能有一个 Tantivy writer。单个 `FullTextIndex` 实例串行化写入和查询快照获取；不同实例或进程争抢写锁时直接报错，不自动重试。写者退出时释放锁；异常后的再次写入有回归测试。应用应将该目录的更新路由给一个写入进程，不要让 Ray worker 并发写入同一目录。多个进程可以读取同一目录。

Vane 的源表事务与 Tantivy 提交彼此独立。源数据变化不会自动同步到索引；应用应安排更新/删除，或重建索引。若源 Relation 在 Ray 上计算，结果会汇总到调用进程建索引，这不是分布式分片检索。没有自动 SQL 优化器下推或 `CREATE INDEX` 集成。

使用本地磁盘并保留整个索引目录。备份前暂停写入并复制完整目录；不要只复制 `vane-search.json`。磁盘满、提交 I/O 错误或进程中断时，提交结果可能需要重新打开索引核实，不能假定报错就代表没有提交。重复提交相同主键的完整文档和重复删除可用于恢复。

当前 Vane 格式版本为 1，支持 Tantivy Python 0.25.x。升级 Tantivy 大/次版本前应单独验证兼容性；本实现不提供自动格式迁移。索引保存了配置的原始文本，应使用与源数据匹配的文件权限和数据保留策略。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 提示安装 `vane-ai[search]` | 在执行建索引或检索的 Python 环境安装可选依赖 |
| `FileExistsError` | 用 `open()`，或选择一个新目录重建 |
| 缺少 `vane-search.json` | 确认目录；初始化失败留下的目录不能直接打开 |
| writer lock 错误 | 结束竞争写入，统一路由到一个写者；不要手动删除活跃写锁 |
| Arrow 类型错误 | 在上游显式转换配置列为字符串，NULL 列也指定类型 |
| 中文或子串查不到 | 检查分词规则；中文使用一致的预分词 |
| schema/manifest 不匹配 | 检查目录完整性和版本，恢复备份或重建，不要手改元数据 |

接口实现、目录格式和测试策略见 [设计文档](DESIGN.md)。
