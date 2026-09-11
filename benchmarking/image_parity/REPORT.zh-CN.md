# Image Python / native / Daft 对比（2026-09-11）

本文保留修改前的基线审计。后续修复及复测见 [FIXES.zh-CN.md](FIXES.zh-CN.md)。

当前 Vane 的纯像素算子在本次样例中内部一致，但编解码仍存在实际差异；
Vane 与 Daft 也不是相同的图像处理契约。优先应修 Vane 内部的无损 BMP
解码和 CMYK 默认模式，再统一 JPEG 路径及改进 GIF 量化。不能直接把
Daft 的所有行为作为标准复制。

## 基线和验证方式

- 审计在独立 worktree 中执行，原始工作目录未改动。
- 审计分支：`audit/image-daft-parity-20260911`。
- 最新 Vane upstream main：[`364212c49b`](https://github.com/AstroVela/vane/commit/364212c49b8f0e48c238574be548763857cb7e2d)。
- 最新 Daft main：[`b14432104c`](https://github.com/Eventual-Inc/Daft/commit/b14432104c67f0ef80eab0d40b1b433ec301da56)。
- 实测 Daft `0.7.24`；已逐路径确认它与上述 main 的 image crates、相关
  Python wrappers、core array/schema/file 源码及完整 `Cargo.lock` 相同。
  这是发布 wheel 的实测，不冒充重新编译了 Daft main。
- Vane 在该 worktree 做了非 editable Release 构建及安装，运行时为
  `v1.5.5-vane.364212c49b`，完整 SourceID 为
  `97ca844a5dec3f51ee78bf511c2be858db259b6d`。native 扩展来自相同构建。
- Linux x86-64、Python 3.12.13；两套环境均使用 NumPy 2.2.6、Pillow
  12.3.0、tifffile 2026.9.9、imagecodecs 2026.8.16。
- 43 个原始 HWC 像素数组、33 个编码文件、每套引擎 14 项额外接口探针。
  文件和数组输入均校验 SHA-256；每套引擎使用独立文件副本，结果保留
  manifest、版本、脚本摘要及输出数组。错误后重建 Vane 连接，避免共享
  事务中止污染后续独立样例。

原始结果、比较、构建和测试日志在 `build/image-audit/`；复现步骤见
[README](README.md)。本次只增加审计材料，没有修改产品实现。

## 1. Vane Python 与 native

### 本次完全一致的部分

| 操作 | 可比较成功样例 | 结果 |
| --- | ---: | --- |
| crop | 129 | shape、dtype、每个像素完全一致 |
| resize | 172 | shape、dtype、每个像素完全一致 |
| convert_image，十种模式互转 | 430 | 完全一致 |
| image_to_tensor，已知 mode 的动态 Image | 43 | 完全一致；该转换本身在 base C++ 执行 |
| image_hash，八种方法及额外参数 | 353 | hash 字节完全一致；另有 4 项双方均报错 |
| 五类共同支持文件的 metadata | 30 | width、height、format、mode 全部一致 |

这里的“一致”仅指当前构建及样例。resize 文档仍允许浮点运算在舍入边界
产生差异，不能从 172 项通过推导所有输入和平台逐位相同。

### 确认未对齐的部分

| 问题 | 同输入实测结果 | 判断 |
| --- | --- | --- |
| 灰度 BMP 解码 | `L.bmp` 有 55/391 个像素不同，最大差值 1；例如 Python `209`、native `208` | 无损格式不应引入这类差异，建议修复 |
| RGB JPEG 解码 | `RGB.jpeg` 在明确 RGB 输出下最大单通道差值 80，平均绝对差 10.438；灰度 JPEG 最大差值 1 | 两条解码/色度上采样路径不同，不能称为像素对齐 |
| CMYK JPEG 默认解码 | Python 为 `(16,16,3)` RGB，native 为 `(16,16,4)` RGBA；指定 RGB 后仍最大差值 2 | 默认输出模式应统一，优先修复 |
| RGB GIF 编码 | 输入纯灰 `[127,127,127]`，Python 编码后解码仍是原值，native 变成 `[109,109,85]` | native 固定 RGB332 调色板明显损失颜色；建议改进 |
| RGB JPEG 编码 | 对随机图分别编码，再由相同参考解码器读取，最大差值 24、平均差 4.676 | 编码器及质量设置没有对齐 |
| metadata 支持范围 | WebP、ICO 的 Python metadata 成功，native 报不支持 | 后端切换有能力差异；应明确契约或补齐 native |

RGB GIF 随机图编码后的最大单通道差值是 84。此处 JPEG/GIF 比较使用了
**相同的独立参考解码器**，因此不会把 native 解码差异混入编码比较。

108 项双方成功的编码中，只有 5 项输出字节完全一样。解码后的 91 项
像素和布局相同，另有 10 项灰度 BMP/GIF 的参考解码布局为 L 与 RGB
之别，展开到 RGB 后颜色完全相同；剩余 7 项 JPEG/GIF 存在颜色数值差异。
PNG/TIFF 的压缩字节、头信息或 strip 布局不同，不应仅据此判为像素错误。

源码解释：Python 编解码使用 Pillow、tifffile 和 imagecodecs；native
主要使用 FFmpeg/libswscale 和 libtiff，灰度 JPEG 编码单独使用 libjpeg。
Python JPEG 编码是 quality=95、4:4:4，native RGB JPEG 是 FFmpeg quantizer=2；
Python GIF 是 median-cut，native RGB GIF 是固定 RGB332。
见 [Python codec](https://github.com/AstroVela/vane/blob/364212c49b8f0e48c238574be548763857cb7e2d/vane/_image_compute.py)、
[native codec](https://github.com/AstroVela/vane/blob/364212c49b8f0e48c238574be548763857cb7e2d/external/duckdb/extension/image/image_codec.cpp)、
[像素转换](https://github.com/AstroVela/vane/blob/364212c49b8f0e48c238574be548763857cb7e2d/external/duckdb/extension/media_common/image_convert.cpp)。

## 2. Vane 与 Daft

### Resize：下采样和透明像素处理都不同

- 单行灰度 `[0,0,0,240]` 缩成 `1×1`：Vane Python/native 都是 **0**，Daft 是 **50**。
- Vane 是 half-pixel 双线性采样，不做下采样抗锯齿预滤波。
  Daft 调用 Rust image 0.25.10 的 Triangle，缩小时扩大滤波支持范围。
- 对不透明红 `[255,0,0,255]` 与全透明蓝 `[0,0,255,0]` 的中间像素，
  Vane 得到 **`[255,0,0,128]`**，Daft 得到 **`[128,0,128,128]`**。
  Vane 先预乘 alpha 再插值，Daft 当前路径直接逐通道插值。
- 76 项双方支持的 resize 中，47 项完全一样，29 项像素不同。

建议：下采样抗锯齿值得引入，但应保留 Vane 的预乘 alpha 处理。
若修改默认 resize，应明确兼容性影响或增加滤波选项，不能直接照搬
Daft 的透明像素路径。
来源：[Daft resize](https://github.com/Eventual-Inc/Daft/blob/b14432104c67f0ef80eab0d40b1b433ec301da56/src/common/image/src/cow_image.rs#L147)、
[Rust image 采样实现](https://github.com/image-rs/image/blob/v0.25.10/src/imageops/sample.rs#L964)、
[Vane transform](https://github.com/AstroVela/vane/blob/364212c49b8f0e48c238574be548763857cb7e2d/external/duckdb/extension/image/include/image_transform.hpp)。

### 灰度转换：BT.601 与 BT.709 风格权重

| 输入 | Vane Python/native | Daft |
| --- | ---: | ---: |
| 纯红 | 76 | 54 |
| 纯绿 | 150 | 182 |
| 纯蓝 | 29 | 18 |

Vane 使用 299/587/114 权重并四舍五入；Daft 的 image crate 使用
2126/7152/722 权重和整数除法。两者不是谁必然错误，而是约定不同。
当前 Vane 的 Pillow 风格兼容性合理，不建议为了“对齐”悄悄改默认值。
来源：[image crate 灰度实现](https://github.com/image-rs/image/blob/v0.25.10/src/color.rs#L606)。

### Crop：坐标顺序相同，边界行为不同

二者都接受 `(x,y,width,height)`。对一张 `3×2` 图片裁剪 `(2,1,4,3)`：
Vane 保留 `4×3` 输出并补零，Daft 只返回剩余的 `1×1`。
完全越界时，Vane 返回请求尺寸的全零图，Daft 实测返回 None。
Vane 支持负起点补零，Daft 此路径的负坐标转换/截边行为不同。
零宽裁剪 Vane 给出明确参数错误，Daft 返回 None；零宽 resize 在本次
Daft 运行中触发除零 panic。

建议保留 Vane 已文档化的补零语义及参数校验；若需要 Daft 式截边，
应单独暴露边界策略。

### Metadata：字段名已一致，mode 表示的层次不同

两边表达式都返回 `width/height/format/mode`，format 使用 PNG/JPEG 等
大写名称。主要差异是 Vane 尽量描述文件原始模式，Daft 表达式返回
Rust decoder 展开后的模式：

| 输入 | Vane Python/native metadata mode | Daft 表达式 mode |
| --- | --- | --- |
| 调色板 PNG，无透明度 | P | RGB |
| 调色板 PNG，有透明度 | P | RGBA |
| GIF | P | RGBA |
| 1-bit PNG | 1 | L |
| CMYK JPEG | CMYK | RGB |
| 灰度调色板 BMP | L | RGB |

此外，Vane 对 NULL ImageFile 返回整个 NULL；Daft 返回非 NULL 的 struct，
里面四个字段为 NULL。这会影响 `IS NULL` 判断。

Daft 的 `ImageFile.metadata()` 值方法另用 Pillow，与其表达式路径自身
也可能不同；Vane 的 Python 值方法与 Python SQL 在本次文件中一致。
建议保留原始 metadata 的含义；如果需要解码输出模式，增加独立字段或
查询 decoded Image 的 mode，避免把两种含义混在同一个字段中。
来源：[Daft metadata 表达式](https://github.com/Eventual-Inc/Daft/blob/b14432104c67f0ef80eab0d40b1b433ec301da56/src/daft-image/src/functions/image_file_metadata.rs)、
[Daft 值方法](https://github.com/Eventual-Inc/Daft/blob/b14432104c67f0ef80eab0d40b1b433ec301da56/daft/file/image.py)。

### 位深、解码能力和 Tensor 返回

- Vane 十种模式的 UInt8/UInt16/Float32 像素运算在本次全部跑通。
  Daft 的 24 个 UInt16/Float32 原始数组构造均遇到 UInt8 downcast panic；
  16-bit PNG 保留原模式解码也会 panic。定义了枚举不代表执行路径完整。
- Daft 有 WebP/ICO/HDR codec 支持，Vane 表达式解码的明确范围是
  PNG/JPEG/TIFF/GIF/BMP。本次 WebP 在 Daft 可解码，Vane 两后端拒绝；
  测试 ICO 被 Daft 解码器拒绝，因为内嵌 PNG 是 RGB 而非其要求的 RGBA。
  不能把源码支持格式等同于支持该格式的所有布局。
- 已知 UInt8 mode 的动态 Image 转 Tensor，双方 19 项像素、shape、dtype
  完全一致。通用 Image 则不同：Vane 得到 Float32，Daft 得到 UInt8。
  Vane 用 Float32 通用存储以容纳不同位深；这是类型设计差异。
- 固定尺寸 Image 转 Tensor 后，Vane `fetchone()` 物化为平铺 tuple，
  HWC 形状保留在逻辑/Arrow 类型；Daft 直接返回 HWC ndarray。
  Vane 的 Arrow `to_numpy_ndarray()` 可取回带形状的数组。

建议保留 Vane 已实现的宽位深能力，不要为追齐 Daft 降级。
固定 Tensor 的标量返回若要统一，应作为独立 Tensor API 变更评估。
来源：[Daft UInt8 执行边界](https://github.com/Eventual-Inc/Daft/blob/b14432104c67f0ef80eab0d40b1b433ec301da56/src/daft-core/src/array/ops/image.rs)、
[Vane Tensor 契约](../../IMAGE.md#image-to-tensor)。

### Hash：同名方法不能保证跨引擎相等

双方共同成功的 160 项 hash 中，120 项相同，40 项不同。
例如随机 RGB 图的 dhash 相差 4 bit、whash 相差 5 bit、默认
crop_resistant 相差 52 bit。Vane 内部的 353 项成功 hash 均一致。

影响因素包括灰度权重、Triangle 中间缓冲精度及舍入、DCT 系数量化、
colorhash 的普通二进制与 Daft 的阈值位编码等。
Vane 的 `crop_resistant(hash_size=3)` 返回 11 字节；Daft 因逐片补齐
产生 18 字节而触发长度不匹配 panic。

建议继续使用已定义的 Vane hash version 1。只有需要复用 Daft 的既存
hash 数据时，才增加明确版本的兼容选项；不建议静默改变默认 hash。
来源：[Daft hash 实现](https://github.com/Eventual-Inc/Daft/blob/b14432104c67f0ef80eab0d40b1b433ec301da56/src/daft-image/src/ops.rs#L420)、
[Vane hash 契约](../../IMAGE.md#perceptual-hashes)。

### 错误处理

缺失文件配合 `on_error='null'`：Vane Python/native 都继续抛 I/O 错误，
Daft 返回 NULL。Vane 仅吞掉已分类的坏媒体内容错误，资源限制、I/O、
依赖和取消继续传播。建议保留 Vane 这一边界，否则数据访问故障容易
被误认为空媒体值。
来源：[Daft 文件解码错误分支](https://github.com/Eventual-Inc/Daft/blob/b14432104c67f0ef80eab0d40b1b433ec301da56/src/daft-image/src/functions/decode_image_file.rs)。

## 3. 建议的后续顺序

1. 修 Vane 内部灰度 BMP 无损像素误差、CMYK 默认输出通道差异。
2. 统一 native/Python JPEG 解码和编码的库配置，单独核对色度上采样、
   IDCT、CMYK 和质量参数；现有 libjpeg-turbo 依赖可作为实现候选。
3. 改善 native GIF 量化，至少保留单色/低色数图像，不再强制 RGB332。
4. 明确 metadata 与 decode 的支持范围；按需求补齐 WebP 等格式。
5. 独立评估 resize 抗锯齿选项。保持 alpha、crop、metadata、宽位深、
   错误传播等现有合理契约；跨引擎 hash 兼容应显式版本化。

## 4. 测试与限制

- 最新基线的非 editable Release 构建、安装与 native image 扩展构建成功。
- 8 个 image 测试文件：**1265 passed**，零失败、零跳过。
- 仓库标准基础测试：非 Ray **1225 passed**，独立真实 Ray 进程
  **30 passed**；合计本次 **2520 项测试通过**。使用仓库默认 marker
  筛选，未把 deselected 的可选测试计为通过。
- 审计脚本 Ruff 格式/检查、输入与输出摘要验证通过。
- 产品源文件未修改；结论及反例针对上述基线，不代表相关差异已修复。

本次实测按独立操作记录结果，1828 条 Vane 记录不是 1828 个自动通过的
测试，也不应把“双方都报错”算作成功。Daft 不支持某些输入构造时，
其依赖操作标为未执行，而非假设与 Vane 相同。

未进行性能测试、跨平台验证或完整格式模糊测试。现有回归测试通过
也不等于跨后端逐像素一致，本报告的反例说明两者需要分开验证。
