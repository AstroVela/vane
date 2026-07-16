<h1 align="center">Vane</h1>

<p align="center">
  <strong>A distributed, multimodal data engine for AI workloads</strong>
</p>

Vane Data extends DuckDB with Ray-based distributed execution and Python-native AI operators. It is designed for pipelines that combine SQL, tabular data, documents, images, audio, video, model inference, and user-defined Python functions.

> [!WARNING]
> Vane is an alpha-quality developer preview. APIs, distributed protocols, packaging, and operational behavior can change without notice. It has not yet been qualified for production use.

## Scope

This repository contains **Vane Data**. The names **Vane RL** and **Vane Agent** describe possible future work only; they are not part of the current release. See [ROADMAP.md](ROADMAP.md).

Vane Data currently focuses on:

- DuckDB SQL and Relation APIs with Vane-specific Python expressions and UDFs.
- Local and Ray-backed execution of the same logical pipeline.
- Multimodal image, document, audio, and video processing.
- Dynamic batching, backpressure, heterogeneous CPU/GPU work, and progress reporting.
- Optional OpenAI, Anthropic, Google, Transformers, and vLLM providers.

The project is tested primarily on Linux x86-64. Other platforms and Python versions listed in package metadata are release targets, not yet a production support promise.

## How it fits together

```text
Python Relation API / SQL
          |
          v
  DuckDB planning and execution
          |
          +---- local runner
          |
          +---- Ray runner ---- CPU / GPU / I/O / model workers
```

Vane keeps DuckDB's embedded execution model while adding distributed plan fragments and exchange operators. Ray supplies the cluster runtime; it is not a replacement SQL engine.

## Install from source

The first public package release is being prepared. Until a release is published, install from a checkout with its submodule initialized:

```bash
git clone --recurse-submodules https://github.com/AstroVela/vane.git
cd vane

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

bash scripts/bootstrap_vcpkg.sh

python -m pip install .
```

The native build is substantial. See [DEVELOPMENT.md](DEVELOPMENT.md) for prerequisites, incremental builds, tests, and troubleshooting. Do not use an editable install: Ray workers can otherwise trigger unexpected rebuilds while importing the package.

## Quick start

```python
import vane

con = vane.connect()
print(con.sql("SELECT 42 AS answer").fetchall())
con.close()
```

Ray is the default runner. With no Ray address configured, Vane starts a local
Ray runtime; set `RAY_ADDRESS` when the application should connect to an
existing Ray cluster. You can also select Ray explicitly in application code:

```python
import vane

vane.configure(runner="ray")
con = vane.connect()
con.sql("SELECT range AS item FROM range(10)").show()
con.close()
```

Run the checked-in smoke test after installation:

```bash
env -u VANE_RUNNER python scripts/verify_base_install.py
```

## Optional AI providers

Provider clients are opt-in:

```bash
python -m pip install 'vane-ai[openai]'
python -m pip install 'vane-ai[anthropic]'
python -m pip install 'vane-ai[google]'
python -m pip install 'vane-ai[transformers]'
python -m pip install 'vane-ai[vllm]'
```

User-defined Python functions, serialized callable payloads, remote model code, and model artifacts are executable code. Only use functions, models, Ray clusters, and artifacts you trust. See [SECURITY.md](SECURITY.md).

## Benchmarks

The [multimodal benchmark suite](multimodal_inference_benchmarks) compares Vane, Ray Data, and Daft implementations of representative image, document, audio, and video pipelines. Benchmark results are meaningful only with the exact dataset, hardware, dependency versions, warm-up policy, and batch configuration recorded alongside them.

## Project documents

- [Contributing](CONTRIBUTING.md)
- [Development](DEVELOPMENT.md)
- [Security](SECURITY.md)
- [Governance](GOVERNANCE.md)
- [Roadmap](ROADMAP.md)
- [Release process](RELEASE.md)
- [Source provenance](SOURCE_PROVENANCE.md)
- [Third-party software](THIRD_PARTY.md)

## License and independence

New Vane work is licensed under the [Apache License 2.0](LICENSE). DuckDB-derived portions remain under DuckDB's MIT license, and other bundled components retain their own terms. See [NOTICE](NOTICE), [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md), and [THIRD_PARTY.md](THIRD_PARTY.md).

Vane is an independent project. It is not an Apache Software Foundation project and must not be described as “Apache Vane.” It is also not affiliated with, endorsed by, or maintained by the DuckDB Foundation.

## Acknowledgements

Vane builds on [DuckDB](https://duckdb.org/) and [Ray](https://www.ray.io/), and draws inspiration from data systems including Ray Data, Daft, and Trino. We are grateful to their contributors.
