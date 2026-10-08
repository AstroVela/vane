# Execution support and release qualification

This matrix describes the execution candidate.
Published versions are identified by their approved GitHub Release notes.
The [release process](RELEASE.md) defines versioning and artifact promotion.

## Platforms and entry points

| Target | Release artifact / CI scope |
| --- | --- |
| Linux x86_64, glibc 2.28 or newer, CPython 3.10–3.14 | Five manylinux wheels; each wheel runs the installed execution smoke before promotion |
| macOS arm64 | Native C++ build and unit tests; Python Runtime/Flight release qualification remains pending |
| Windows x64 | Native MSVC build and unit tests; Python Runtime/Flight release qualification remains pending |
| Other architectures / operating systems | Require separate qualification; outside this candidate's binary release matrix |

| Entry point | Scope |
| --- | --- |
| `vane.connect()` | Local native execution; managed `query()` plus the local SQL/Relation/model APIs |
| `vane.Runtime(...).connect()` | Shared Ray workers and result service; pipelined and FTE selected per query |
| `vane.client.Client(...)` | Remote analytical SELECTs through a server-owned Runtime; the client does not initialize Ray |
| `vane-server` / `python -m vane.server` | Standalone Flight control and native result listeners; authentication, session leases and bounded shutdown |

The distributed SELECT and type boundaries are defined in the
[execution design](PIPELINED_EXECUTION_DESIGN.md#sql-与类型范围) and exercised by
the [differential acceptance matrix](EXECUTION_ACCEPTANCE.md#correctness-matrix).
Unsupported plans fail explicitly. SQL parameters, distributed DML/DataSink,
model/AI/GPU UDFs, media types, windows and recursive queries require further
implementation and qualification. Local SQL capabilities remain independent.

## Deployment and failure boundaries

- Both strategies use the same plan/type protocol and resource accounting.
  Pipelined worker loss fails the query. FTE retries eligible attempts within
  its declared limit using fixed inputs and committed manifests.
- FTE requires a registered shared filesystem with stable identity, locking,
  atomic publication and durable sync. Every worker must see that mount; its
  failure domain must be independent of compute. A temporary local store in a
  single-host smoke test only validates the protocol on that host.
- A result-service failure fails its resident results. A server restart creates
  a new identity; previous sessions/queries cannot be resumed. Recovery does not
  transparently switch execution strategies or replay client delivery.
- The public control and result ports both use TLS and validate certificates
  and hostnames. Non-loopback listeners require TLS. Clients and internal
  exchange connect directly to their advertised endpoints, independently of
  environment HTTP proxies. Current internal Ray transport assumes a trusted
  cluster network. Details are in the [server design](SERVER_DESIGN.md).
- The Flight client and server require the same native engine identity. This is
  Vane's control/result protocol. A Quack entry point is planned with the DuckDB
  2.0 upgrade; the current release does not provide Flight SQL interoperability.

## Installed artifact gate

`scripts/verify_execution_install.py` needs only the base wheel, its declared
dependencies and the OpenSSL CLI. Extract that file from the matching sdist and
run it outside the checkout:

```bash
python -I /path/to/verify_execution_install.py --report /path/to/execution-smoke.json
```

It owns one local two-CPU Ray cluster and a temporary filesystem. Three queries
(grouped SUM including NULL, Parquet TopN, and typed empty output) run through
five paths: local, embedded pipelined/FTE, and remote pipelined/FTE. The remote
client runs in a separate isolated interpreter, verifies both TLS endpoints,
rejects invalid credentials, and stays disconnected from Ray. All results are
checked for exact rows, order, column names and types. Session/result cleanup
must complete. The JSON report records completed checks, the failing stage,
package/Python/platform identity and the native library SHA-256. A failed step
returns a nonzero exit status; missing TLS tooling is an error, not a skip.

The gate runs in three places:

1. `cibuildwheel` runs it against each built manylinux wheel in its test
   environment, alongside the local Quickstart.
2. The base release launcher runs its isolated regression and standalone server
   CLI after the non-Ray and shared-Ray processes have exited. The regression
   copies only this script and poisons source import resolution, ensuring the
   driver, workers and remote client use the installed package.
3. TestPyPI and PyPI verification first compare downloaded wheels with the
   approved artifacts, then extract the smoke scripts from the approved sdist
   and run them without a checkout. Their JSON evidence is retained in Actions.

The gate is an artifact smoke test. Detailed SQL, lifecycle, recovery, resource
and performance qualification continues to use the existing acceptance suites.

## Candidate promotion checklist

- Record the exact candidate commit, package version, native engine identity,
  artifact checksums and matching CI/build-only release run URLs.
- Require the supported Python matrix, native platform jobs, full CI release
  gate, artifact inspection and signing/provenance checks from [RELEASE.md](RELEASE.md).
- Retain the [historical Flight timeout investigation](EXECUTION_ACCEPTANCE.md#flight-proxy-isolation)
  as open: the proxy-induced failure has a reproducible fix, but the original
  incident lacked operation/proxy evidence. Passing smoke tests do not establish
  its cause or complete P5.
- Record multi-host deployment and real network-partition qualification
  separately from the completed single-host acceptance.
- Promote only the exact approved artifacts through the existing protected
  release workflow. This work adds no version tag, publication or index upload.

The [roadmap](PIPELINED_EXECUTION_ROADMAP.md#p53-发布验收) records which candidate
checks have actually run. Local development runs only affected tests; full CI
and release qualification remain separate gates.
