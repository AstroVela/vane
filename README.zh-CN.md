<h1 align="center">
  <img src="assets/vane-logo.svg" alt="VANE" width="336" height="96">
</h1>

<p align="center">
  <strong>面向 AI 工作负载的高性能多模态原生引擎</strong>
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/vane-ai/">
    <img src="https://img.shields.io/pypi/v/vane-ai?logo=pypi" alt="PyPI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="Apache License 2.0">
  </a>
  <a href="https://deepwiki.com/AstroVela/vane">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
  </a>
</p>

<p align="center">
  <a href="https://discord.gg/BuKhPQcqs">
    <img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&amp;logoColor=white&amp;style=for-the-badge" alt="加入 Discord">
  </a>
  <a href="https://x.com/AstroVelaAI">
    <img src="https://img.shields.io/badge/X-Follow_%40AstroVelaAI-black?logo=x&amp;style=for-the-badge" alt="在 X 上关注 AstroVelaAI">
  </a>
</p>

## Vane Data

Vane Data 是面向 AI 工作负载的高性能、多模态原生数据引擎。它基于 [DuckDB](https://duckdb.org) 的分支构建，在核心执行引擎中加入原生多模态处理能力，并提供统一的单机与分布式执行框架。

![Vane Data 多模态原生数据引擎架构](https://vane.astrovela.ai/img/blog/ai-workloads-need-a-new-data-engine/vane-data-architecture-zh-cn.png)

### 核心特性

- **多模态原生处理** — 通过统一的类型系统处理图片、视频、音频、文本、文档、事件、传感器数据和表格。动态批处理与背压机制适应数据大小和计算成本的差异。
- **Python 与 SQL 接口** — 使用 DuckDB SQL 或 Python Relation API 构建数据与 AI 流水线。
- **内置 AI 操作** — 通过 OpenAI、Anthropic API 或原生 vLLM 集成调用大语言模型、生成向量表示并执行批量推理。前缀感知分桶提高 vLLM 前缀缓存命中率和推理吞吐量。
- **异构执行** — 通过异步调度，让 CPU、GPU、I/O 和模型推理阶段重叠执行。
- **从本地扩展到云端** — 同一条流水线可在本地运行，也可分布式运行于 Ray 集群，并为未来端云协同奠定基础。
- **面向生产环境中的 AI 工作负载** — 构建多模态训练数据预处理流水线和企业级批量推理流程。

---

## 开始使用

### 安装

Vane 支持 Python 3.10–3.14，推荐使用 Python 3.12，这也是主要开发版本。

通过 PyPI 安装 `vane-ai`：

```bash
pip install vane-ai
```

更多信息请参阅[安装指南](https://vane.astrovela.ai/zh-CN/docs/data/quickstart/installation)。

### 快速入门

按照[快速入门指南](https://vane.astrovela.ai/zh-CN/docs/data/quickstart/quickstart)构建并运行你的第一条 Vane 流水线。

对于需要保持音视频同步和源时间戳的视频片段编码，请参阅[视频裁剪示例](examples/video_clip.py)。示例涵盖 `VideoFile.clip`、Python/SQL 中的 `video_clip` 表达式、输出编解码器及资源限制。裁剪结果会在 MP4 元数据中保留恒定的采样宽高比，包括超出输出视频编解码器表示精度的比例。

[多模态 Provider 示例](examples/multimodal_providers.py)介绍了原生 vLLM/SGLang Python 引擎，以及可配置的模型与多模态向量生成。

[Qwen3-VL SQR 示例](examples/qwen3_vl_sqr.py)介绍了可选的 `qwen3_vl` Provider、其固定的文本/媒体编码约定，以及如何通过 Vane Context 的模型配置选择该 Provider。

### 更多资源

- [示例](https://vane.astrovela.ai/zh-CN/docs/data/examples)
- [生产环境部署](https://vane.astrovela.ai/zh-CN/docs/data/deploy/deployment)

---

## 多模态推理基准测试

硬件配置：1 个节点、36 个 CPU 核心、64 GB 内存，以及 1 张 NVIDIA GeForce RTX 2080 Ti（22 GB 显存）。

我们使用 [Ray Data 基准测试套件](https://www.anyscale.com/blog/ray-data-daft-benchmarking-multimodal-ai-workloads)对比 Vane、Ray Data 和 Daft。[基准测试源码](multimodal_inference_benchmarks)已包含在本仓库中。

![Vane Data、Ray Data 与 Daft 的多模态推理基准测试对比](assets/benchmark.png)

Ray Runner 面向分布式工作负载。当前结果仅来自单节点测试，在 Ray Data 基准测试所使用的多节点环境中的验证尚待完成。

详细结果请参阅[基准测试页面](https://vane.astrovela.ai/zh-CN/benchmarks)。

---

## 参与贡献

欢迎参与贡献与合作。随着项目进一步开放，我们会陆续完善贡献指南和社区交流渠道。

---

## 许可证

Vane 采用 Apache License 2.0 许可证。具体条款与第三方署名信息请参阅 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。

---

## 致谢

Vane Data 基于 DuckDB 构建，并从 Ray Data、Daft、Trino 等基础设施项目中获得启发。

- **[DuckDB](https://github.com/duckdb/duckdb)**：核心模块化架构与设计灵感的来源。DuckDB 是一个高性能分析型数据库系统，致力于提供快速、可靠、可移植且易用的体验。
- **[DuckDB-Python](https://github.com/duckdb/duckdb-python)**：核心模块化架构与设计灵感的来源，也是 DuckDB 的 Python 包。
- **[Ray Data](https://github.com/ray-project/ray)**：基于 Ray 构建、面向 AI 工作负载的可扩展数据处理库。
- **[Daft](https://github.com/eventual-inc/daft)**：面向 AI 与多模态工作负载的高性能数据引擎。
- **[Trino](https://github.com/trinodb/trino)**：面向大数据分析的高性能分布式 SQL 查询引擎。

**特别感谢这些项目。**

---

<div align="center">

**如果 Vane 对你有帮助，欢迎点亮一颗 ⭐️！**

</div>
