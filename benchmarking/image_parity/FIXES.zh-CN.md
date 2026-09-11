# Image 后端对齐修复（2026-09-11）

已实现基线审计中推荐的七项变更：BMP 灰度解码、CMYK JPEG 默认模式、
JPEG 解码、GIF 调色板、JPEG 编码质量、显式抗锯齿选项，以及 WebP 解码和
metadata 支持。原始审计结果保留在 [REPORT.zh-CN.md](REPORT.zh-CN.md)。

- 工作分支：`fix/image-backend-parity-20260911`。
- upstream main 基线：`364212c49b8f0e48c238574be548763857cb7e2d`。
- 已测试的 DuckDB SourceID：`bf37a41c12b46b5b7b36e78f4415cf7af06bfd9f`。
- Daft 对照：main `b14432104c`；实测其 image 实现一致的发布 wheel 0.7.24。
- Vane 使用非 editable Release 安装，native image 扩展来自相同构建。

## 修复结果

| 项目 | 修复前的固定输入结果 | 修复后的结果 |
| --- | --- | --- |
| 灰度 BMP 解码 | 391 个像素中，55 个偏差 1 | 与 Python 完全一致；新增测试覆盖全部 256 个灰度值 |
| CMYK JPEG 默认解码 | Python RGB、native RGBA | 两端均为 RGB，metadata 仍为 CMYK |
| RGB JPEG 解码 | 最大单通道差 80 | 原样本最大差 0 |
| RGB GIF 编码 | 纯灰 `[127,127,127]` 变为 `[109,109,85]` | 保留 `[127,127,127]`；低于或等于 256 种颜色时保留原色 |
| RGB JPEG 编码 | 同一参考解码器读出的最大差 24 | 原样本最大差 0；两端统一质量 95 和 4:4:4 |
| Resize 抗锯齿 | `[0,0,0,240]` 缩为 1 像素得到 0 | `antialias=True` 得到 50；默认仍为 0 |
| WebP | 表达式解码两端均不支持，native metadata 不支持 | 两端支持有损、无损、透明和动画首帧；metadata 一致 |

八位 JPEG 使用 libjpeg 的准确整数 IDCT 和 fancy chroma upsampling；
原有宽位深 native JPEG 路径继续使用 FFmpeg。GIF 两端采用确定性加权
median-cut：超过 256 种颜色时使用有界的 5-bit 直方图，调色板颜色取
原始 8-bit 样本的加权均值，不做抖动。

抗锯齿是显式选项，支持逐行布尔值和 NULL。两端均保留预乘 alpha，
两个滤波步骤之间保留 double 精度，最后才做整数舍入。中间缓冲区上限为
256 MiB；纯放大和相同尺寸使用原有路径。

```python
vane.resize(image, 224, 224, antialias=True)
image_expr.resize(224, 224, antialias=True)
```

```sql
SELECT resize(image, 224, 224, true);
```

具体接口、模式矩阵和资源限制见 [IMAGE.md](../../IMAGE.md)。WebP 本次增加
解码和 metadata；编码格式矩阵仍为 PNG/JPEG/TIFF/GIF/BMP。

## 固定输入复测

复用原审计的 43 个原始像素数组和 33 个编码文件；已验证新旧输入
manifest 完全一致。三个引擎均重新执行，并校验输入、脚本和输出摘要。
结果保存在 `build/image-parity-fix/corpus/`，摘要在
`build/image-parity-fix/metrics.json`。

| 操作 | Python/native 成功样本的结果 |
| --- | --- |
| 字节解码 | 126 项，shape/dtype/像素完全一致 |
| 文件解码 | 126 项，shape/dtype/像素完全一致 |
| 编码 | 108 项，经同一参考解码器读取、统一颜色布局后全部一致 |
| Resize 默认路径 | 172 项完全一致 |
| 模式转换 | 430 项完全一致 |
| Crop | 129 项完全一致 |
| Hash | 353 项成功结果的字节完全一致 |
| Metadata | 31 项共同支持的表达式结果完全一致 |

编码输出中，10 项文件字节相同、98 项不同。10 项灰度 BMP/GIF 的参考
解码布局仍为 L 与 RGB 之别，展开灰度通道后颜色全部相同。编码容器的
字节或存储布局差异没有被当作像素误差。

这批样本中仍有一项格式支持差异：ICO metadata 在 Python 成功、native
不支持。Vane 与 Daft 的灰度权重、越界裁剪、metadata 原始模式、哈希
版本和 NULL/I/O 错误语义继续采用原审计中建议保留的 Vane 契约。

## 首次实现验证

- 两轮代码自查后完成增量 Release 构建和 native image 扩展构建。
- 新增回归测试：106 passed，覆盖 JPEG 色度采样与渐进式编码、CMYK、
  BMP 全灰度值、GIF 低色数与量化质量、WebP 各种输入及限制、全部十种
  像素模式的抗锯齿和 native 路径独立性。
- 原有八个 image 测试文件：1265 passed；image 合计 1371 项通过。
- 基础测试：非 Ray 1225 passed；独立真实 Ray 进程 30 passed。
  本轮合计 **2626 项通过，零失败、零跳过**；可选 deselected 用例未计入。
- 格式、Ruff、`git diff --check` 通过。libwebp 使用固定 vcpkg baseline，
  独立扩展依赖 notices 已生成并核验；base license bundle 保持其依赖范围。

测试和构建日志保存在 `build/image-parity-fix/`。本次为 Linux x86-64 的
固定输入和回归测试结果，不推导所有图片、编解码器版本或平台逐位相同。

## 最终本地 review 与验证

本地 review 补齐了 Python WebP 分配解码画布前的像素、工作内存和输出容量
检查，以及只需 25/30 字节预算的 WebP metadata 文件头检查。两端均保留
metadata 抛错、decode 可使用 `on_error='null'` 的接口规则。另补齐
`Expression.resize` 的 `antialias` 类型声明和严格类型检查用例。

新增测试曾因 SQL 预算参数缺少 `UBIGINT` 转换、误给 metadata 传入
`on_error` 而失败；修正测试后重新完成连续两轮本地代码自查，均无待修项。
最终生产代码经非 editable 安装验证，与已编译 native 扩展的 SourceID 一致。

- 对齐回归：128 passed，包含新增的 22 个 WebP 边界测试组合。
- 既有 metadata/解码回归：561 passed（image_file 44、image_compute 517）。
- 基础测试：非 Ray 1225 passed，独立真实 Ray 进程 30 passed。
- 最终共覆盖 1944 个通过的不同测试用例；无未解决的失败或跳过。
- installed mypy 严格检查：2 个文件通过，类型目标为 Python 3.10。

最终评审快照、详细范围和日志保存在 `build/image-parity-fix/local-review/`，
不包含在源码提交中。提交准备另补齐 GIF 量化数组的 NumPy 类型及 `np.bool_` 参数声明，
并更新验证说明、移除本机路径；编解码和缩放的运行逻辑未改动。
