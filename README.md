<h1 align="center">
  <img src="assets/vane-logo.svg" alt="VANE" width="336" height="96">
</h1>

<p align="center">
  <strong>A high-performance, multimodal-native engine for AI workloads</strong>
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
    <img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&amp;logoColor=white&amp;style=for-the-badge" alt="Join Discord">
  </a>
  <a href="https://x.com/AstroVelaAI">
    <img src="https://img.shields.io/badge/X-Follow_%40AstroVelaAI-black?logo=x&amp;style=for-the-badge" alt="Follow AstroVelaAI on X">
  </a>
</p>

Vane unifies multimodal data, intelligence, and continuous learning with Python and SQL interfaces, seamlessly scaling from local environments to Ray clusters.

![Vane platform overview](assets/vane-platform.png)

> [!NOTE]
> **Project status**
>
> - **Vane Data** — Supports most of the capabilities described below and is under active development. Its interfaces and internals may continue to evolve as the codebase is reviewed and hardened.
> - **Vane RL** and **Vane Agent** — In the early stages of design and implementation. Their source code will be released in future updates.
  - **Vibe Coding and Agentic Engineering** — Some parts of our system were initially built through Vibe Coding. We are now continuously analyzing, understanding, and improving the codebase, applying an Agentic Engineering approach to drive iterative optimization and enhance the quality, maintainability, and efficiency of the system.

---

## Vane Data

Vane Data is a high-performance, multimodal-native data engine for AI workloads. Built on a fork of [DuckDB](https://duckdb.org), it extends the core execution engine with native multimodal processing and a unified framework for local and distributed execution.

![Vane Data architecture](assets/vane-data.png)

### Key Features

- **Multimodal-native processing** — Process images, video, audio, text, documents, events, sensor data, and tables through a unified type system. Dynamic batching and backpressure control handle variations in data size and computational cost.
- **Python and SQL interfaces** — Build data and AI pipelines with DuckDB SQL or the Python Relation API.
- **Built-in AI operations** — Invoke LLMs, generate embeddings, and run batch inference through OpenAI and Anthropic APIs or native vLLM integration. Prefix-aware bucketing improves vLLM prefix-cache hit rates and inference throughput.
- **Heterogeneous execution** — Overlap CPU, GPU, I/O, and model inference workloads through asynchronous scheduling.
- **Local-to-cloud execution** — Run the same pipeline locally or across distributed Ray clusters, with a foundation for future edge-cloud coordination.
- **Designed for production AI workloads** — Build multimodal training-data preprocessing pipelines and enterprise-scale batch inference workflows.

---

## Getting Started

### Installation

Vane supports Python 3.10 through 3.14. Python 3.12 is recommended and is the primary development version.

Install the `vane-ai` package from PyPI:

```bash
pip install vane-ai
```

Optional DuckDB extensions are separate platform packages. Install a provider
with pip from the package index configured for your deployment, then load it by
name on the connection that will plan the query:

```bash
python -m pip install vane-extension-iceberg
```

```python
import vane

connection = vane.connect()
vane.load_installed_extension("iceberg", connection=connection)
vane.vane_extensions(connection=connection).show()
```

`vane.extension_catalog()` reads the live, independent
[`vane-extensions`](https://github.com/AstroVela/vane-extensions) registry, so
new provider packages do not require a Vane release. See the
[distributed extension architecture](DISTRIBUTED_EXTENSIONS.md) for discovery,
installation, verification, and Ray worker requirements.

For more details, see the [Installation Guide](https://vane.astrovela.ai/docs/data/quickstart/installation).

### Apache Doris Arrow Stream Load

Install the HTTP transport and write Arrow batches directly to a Doris FE or
BE Stream Load endpoint:

```bash
pip install 'vane-ai[doris]'
```

```python
import pyarrow as pa
import vane

relation = vane.sql(
    """
    SELECT
        i AS id,
        (CASE WHEN i = 1 THEN [0.1, 0.2, 0.3] ELSE [0.4, 0.5, 0.6] END)::FLOAT[] AS embedding,
        CASE WHEN i = 1 THEN 'one' ELSE 'two' END AS title
    FROM range(1, 3) AS t(i)
    """
)
summary = relation.write_datasink(
    vane.DorisStreamLoadSink(
        "analytics",
        "items",
        endpoint="http://doris-fe.example:8030",
        destination_schema=pa.schema(
            [
                pa.field("id", pa.int32(), nullable=False),
                pa.field("embedding", pa.list_(pa.float32()), nullable=False),
                pa.field("title", pa.string(), nullable=False),
            ]
        ),
        vector_dimensions={"embedding": 3},
        worker_count=4,
    )
)
```

The `ray` runner also accepts in-memory `from_df()` relations and `from_arrow()`
relations built from a PyArrow `Table` or `RecordBatch`, for reads and sink writes.
It snapshots the referenced columns into the Ray object store during planning;
subsequent changes to the source do not change that query's snapshot. Materialize
Arrow datasets, scanners, readers, and C Stream capsules into an eager table
before using them with Ray. The `local` runner requires a distributable SQL or
file relation for distributed sink writes.
`write_datasink()` is synchronous. In an async caller, offload the complete
connection/relation/write operation with `asyncio.to_thread()`; the Ray runner
explicitly rejects blocking execution on the caller's event-loop thread.

The required `destination_schema` lists the selected Doris columns in upload
order and declares their exact Arrow physical types: for example, Doris `INT`
is `pa.int32()`, `FLOAT` is `pa.float32()`, and `ARRAY<FLOAT>` is
`pa.list_(pa.float32())`. The sink safely casts every input column to this
schema before opening an HTTP request, so inferred Python integers (`int64` in
Arrow) cannot be misread as Doris `INT`; overflow, incompatible nested values,
and nulls for non-nullable fields fail locally. Floating-point narrowing rejects
finite values that become infinity while allowing normal rounding. The currently
supported destination types are booleans, signed integers, float32/float64, UTF-8
strings, and recursive regular lists of those types. Execution workers may use
64-bit Arrow offsets (`large_string`, `large_list`); the sink accepts equivalent
input representations and safely normalizes them to the destination schema,
including nested lists. Only visible list children are converted; hidden
payloads beneath null list slots cannot cause false overflow errors. Supported
inputs are null, boolean, integer, floating-point, string, binary, and standard
list arrays recursively composed from them. Dictionary, run-end encoded, view,
and other unsupported representations must be converted and rebatched before
writing. String destinations require string or binary input; explicitly
stringify other types in the input relation. Temporal Arrow types are
rejected, including nested values, because Doris 4.1.3 does not preserve their
timezone semantics; explicitly convert them to a supported non-temporal type
before writing.

The sink uses Arrow IPC throughout and does not materialize Python rows.
`max_batch_bytes` limits input Arrow batches to 128 MiB by default, while
`max_request_bytes` independently caps encoded HTTP bodies at 160 MiB.
Destination buffer sizes are conservatively checked against the request budget
before casting. The full IPC stream, including schema and chunk metadata, is
sized before allocating one fixed-size request buffer. Peak
worker memory includes at least the input and encoded buffers plus any safe-cast
buffers and vector offsets; the HTTP transport drains each request through
bounded 256 KiB views instead of enqueueing the complete body again. Each worker performs one
synchronous request at a time; increase `worker_count` for concurrent Stream
Loads and tune `send_batch_parallelism` for Doris-side fan-out. When an FE
endpoint redirects to a different BE host, list that host in
`trusted_redirect_hosts` before Vane will send the configured Basic Auth
credentials. Entries contain only a hostname or IP address, without a port;
IPv6 literals can be bare (`2001:db8::42`) or bracketed (`[2001:db8::42]`).
Passwords must be supplied through `EnvironmentSecret`. Vane does
not retry an Arrow Stream Load request: if a connection fails after upload, the
batch outcome is unknown and its reported Doris label must be inspected before
submitting new data. `timeout` sets the Doris import deadline; the HTTP
transport adds 30 seconds to receive the terminal response without racing that
server-side deadline. For a large initial vector load, create and build the
Doris ANN index after ingestion so index construction does not slow every
incoming batch.

### Quick Start

Follow the [Quickstart guide](https://vane.astrovela.ai/docs/data/quickstart/quickstart) to build and run your first Vane pipeline.

`Connection.execute()` routes `SELECT` queries through the configured runner,
including queries with positional or named parameters:

```python
import vane

vane.set_runner_ray()
with vane.connect() as conn:
    rows = conn.execute(
        "SELECT i FROM range(?) AS t(i) WHERE i >= ? ORDER BY i",
        [10, 7],
    ).fetchall()
```

Set `VANE_RUNNER=local-fast` before connecting to use native DuckDB execution.
Ray is the default when that variable is unset or empty. Each connection fixes
its runner at creation; cursors and derived relations inherit that policy.
Later environment changes and runner-selection calls affect new connections
only. Module-level helpers such as `vane.sql()` share the default connection
and its fixed policy. Set the variable before importing Vane to choose the
default connection's policy, or create an explicit connection to choose a new one.
Ray and local FTE runner instances are initialized separately and retain their
explicit configuration. `get_runner()` and `get_or_create_runner()` select by
the current environment; `teardown_runner()` closes both initialized runners.
Ray initializes when a query or write first needs it. Ray queries require auto-commit mode;
planning and execution errors propagate without local fallback. `execute()`
returns the connection and shares one cursor across row, DataFrame, and Arrow
consumers. Multiple statements execute in order and retain only the last result.
SQL `COPY TO` also uses the connection runner and shares the Relation write
APIs' planning, commit, and failure-cleanup protocol. `execute()` returns its
`Count` row; `sql()` completes the write and returns `None`. Ray and local FTE
use the Relation writer's dataset layout: a new target such as `output.parquet`
is a directory containing worker output files. Both runners reject
`COPY FROM`, `RETURN_FILES`, `RETURN_STATS`, and explicit transactions before
writing. Other unsupported write capabilities fail explicitly. A committed
write whose result cannot be delivered raises `CopyResultUnavailableError`
with `safe_to_retry=False`; an uncertain outcome remains
`CopyOutcomeUnknownError`. `executemany()` uses the same query/COPY routing for
every parameter set and retains the final result. The `local` FTE runner
supports writes; its SELECT result consumption continues to use native DuckDB.

Session configuration, `ATTACH`, transaction control, DDL, and other SQL DML
continue executing on the client coordinator connection. SQL is bound there;
Ray receives serialized bound logical plans for both SQL and Relation queries
and writes. Moving catalog and session operations to the driver is outside
this routing change.
Ray uses the same source support as the Relation runner: scans of ordinary
in-memory tables and temporary tables are rejected. Select `local-fast` for
those queries when creating the connection, or use a distributed source such as Parquet.

`conn.sql()` (also `query()` and `from_query()`) returns a lazy relation for
`SELECT`, including when `params` supplies positional or named values. Values
are captured when the relation is created; modifying the original parameter
container does not change the query. Filtering, joining, exporting SQL, or
creating a view preserves those values. Reading the result uses the configured
runner and does not first materialize the SELECT on the coordinator:

```python
with vane.connect() as conn:
    relation = conn.sql(
        "SELECT i + $offset AS value FROM range($rows) AS t(i)",
        params={"offset": 10, "rows": 5},
    )
    rows = relation.filter("value >= 12").order("value").fetchall()
```

The final SELECT remains lazy; preceding statements in the same call execute
in order, with SELECTs using the runner. Parameters belong only to the final
statement. Ray result consumption rejects an explicit transaction, including
one started after the relation was created.

Call `conn.interrupt()` from another thread to cancel an active Ray result wait.
Row consumers raise `InterruptException` after query cleanup; exported Arrow
readers report interruption through the Arrow stream error. The connection can
execute another query afterward, and other connections keep their own queries.

### More Resources

- [Examples](https://vane.astrovela.ai/docs/data/examples)
- [Production deployment](https://vane.astrovela.ai/docs/data/deploy/deployment)

---

## Multimodal Inference Benchmarks

Hardware configuration: 1 node, 36 CPU cores, 64 GB memory, and 1× NVIDIA GeForce RTX 2080 Ti (22 GB VRAM).

We use the [Ray Data benchmark suite](https://www.anyscale.com/blog/ray-data-daft-benchmarking-multimodal-ai-workloads) to compare Vane with Ray Data and Daft. The [benchmark source code](multimodal_inference_benchmarks) is included in this repository.

![Multimodal inference benchmark comparing Vane Data, Ray Data, and Daft](assets/benchmark.png)

The Ray runner targets distributed workloads. The current results are single-node only; validation on the multi-node environments used in the Ray Data benchmarks is still pending.

See the [benchmarking page](https://vane.astrovela.ai/benchmarks) for detailed results.

---

## Contributing

Contributions and collaborations are welcome. Contribution guidelines and community channels will be published as the project opens further.

---

## License

Vane is distributed under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details and third-party attributions.

---

## Acknowledgements

Vane Data is built on top of DuckDB and inspired by infrastructure systems such as Ray Data, Daft, and Trino.
*   **[DuckDB](https://github.com/duckdb/duckdb)**: The core modular architecture and inspiration. A high-performance analytical database system. It is designed to be fast, reliable, portable, and easy to use.
*   **[DuckDB-Python](https://github.com/duckdb/duckdb-python)**: The core modular architecture and inspiration. The DuckDB Python package.
*   **[Ray Data](https://github.com/ray-project/ray)**: A scalable data processing library for AI workloads built on Ray
*   **[Daft](https://github.com/eventual-inc/daft)**: High-Performance Data Engine for AI and Multimodal Workloads
*   **[Trino](https://github.com/trinodb/trino)**: A fast distributed SQL query engine for big data analytics.

**Special thanks to these projects.**

---

<div align="center">

**Give Vane a ⭐️ if it helps you!**

</div>
