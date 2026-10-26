# 合并前静态检查收尾

性能[路线图](../../PERFORMANCE_ROADMAP.md)已经关闭，本次只处理草稿 PR 的静态检查失败。
没有运行新的性能实验，没有改动原生代码、持久化策略、验收门槛或既有测量结果。

## 已定位问题和修改

检查起点为 `5f13479740e5d94c36370aa0ebf12a47d8f65e9a`，比较基线为
`b7c00df040418a1e252044960503349ebeb4f62c`。公开接口确认
[PR #967](https://github.com/AstroVela/vane/pull/967)仍为草稿，描述与收尾交付版本一致。
该提交的远端检查有 32 项成功、1 项跳过、1 项失败；失败发生在
[Ruff 检查](https://github.com/AstroVela/vane/actions/runs/39285302587/job/118169833134)。
本地按同一比较范围复现了全部 6 条错误。

| 脚本 | 唯一修改 |
| --- | --- |
| `lookup_reads/run.py` | 将 sqlite3 放回标准库导入顺序 |
| `performance_closeout/shipping.py` | 删除未使用的 sys 导入 |
| `staged_payload/begin_gate_matched/analyze.py` | 修正导入区空行 |
| `staged_payload/gate_sync_waits/sample.py` | 删除未使用的 struct 导入 |
| `staged_payload/paced_request_waits/run.py` | 修正导入区空行 |
| `staged_payload/sqlite_step_waits/attribute.py` | 将赋值的 lambda 改为同参数、同返回表达式的局部函数 |

## 历史身份的读取方式

旧的性能报告和审计索引保持原样，代表它们记录的那次运行。
[snapshots.json](snapshots.json)记录 6 个修改前脚本的 Git 提交、原始路径、SHA-256 和无损 gzip 快照；
[verification.json](verification.json)记录修改后的脚本身份和核对结果。
查验旧报告的 `path + SHA-256` 时，先核对当前文件；若身份已变，使用此处相同原始路径和哈希的快照。
不能把修改后的脚本当作旧哈希的来源。需要回放旧工具时，在独立副本中恢复快照。

6 个快照均与起点提交的 Git 内容一致。6 处修改经限定转换后的语法树核对；
所有函数体保持一致，仅上述 lambda 改为同等的函数定义。
449 项冻结身份中 448 项仍直接匹配当前文件，shipping.py 的 1 项由原始快照精确匹配；
原发布索引的 307 项同理为 306 项当前匹配、1 项快照匹配。
296 个原归档流仍可逐字节恢复。11 项默认身份全部直接匹配，59 个清理路径均不存在。
验证命令本身保存在 `evidence/verify.py.gz.part`，解压后可检查其范围。

## 验证范围与后续

本次验证只覆盖上述静态问题和证据完整性，不代替整个 PR 的人工代码审查。
只重跑本次相关的格式、pre-commit 和源码检查；原生实现未变，不重复构建或性能测量。
具体命令和退出码记录在 `checks.json`，日志保留在同目录。
修复后的远端结果以 PR 最新提交的 CI 为准；起点的绿色检查不代表后续提交已经通过。
PR 保持草稿，性能候选预算仍为 1 个使用、2 个关闭未用。
