# Changelog

All notable user-visible changes are documented here. Vane is currently in alpha, so incompatible changes may occur between prereleases.

## Unreleased

### Added

- Public governance, contribution, security, release, provenance, and third-party documentation.
- Release artifact validation and a reproducible native dependency license bundle.
- Added one closed basic Prompt contract across the Python Expression,
  functional Relation, Relation method, and typed SQL entry points. Ordered
  `VARCHAR`, `BLOB`, and `BLOB[]` message parts support OpenAI, Anthropic, and
  Google; native vLLM remains text-only. NULL image parts are omitted, while a
  zero-length image follows the selected row-level `on_error` policy.
- Added first-class Prompt `return_format` and `return_raw_response` parameters.
  A portable finite JSON Schema subset maps to native DuckDB `STRUCT` values
  and constrains OpenAI, Anthropic, Google, and vLLM requests. Raw mode returns
  the provider SDK response body as JSON `VARCHAR`; vLLM rejects raw mode at
  planning time.

### Changed

- Positioned the current project as the Vane Data developer preview.
- Prompt uses ordered `messages`, first-class call parameters, and the closed
  `PromptOptions` keyword surface. OpenAI Responses is the default endpoint.
- Defined `DuckDBPyRelation.map` exclusively as a row-wise scalar UDF with a
  required `return_type`; batch transforms use `map_batches` with an explicit
  output `schema`. The inherited pandas DataFrame-style DuckDB `map` contract
  is no longer supported.
- Restricted source distributions to the DuckDB components required by Vane.
- Imported the official DuckDB baseline as a squashed Git subtree and retained
  Vane engine customizations as monorepo commits, so normal clones no longer
  require submodule initialization or carry DuckDB's complete commit history.

### Fixed

- Released per-database runner cache entries after relation write failures
  without resetting the process-wide runner used by other queries.
- Kept provider capability failures serializable and credential-safe across
  local and Ray execution while preserving bounded upstream diagnostics.
- Stopped Google Embed metadata dimensions and SDK retries from overriding the
  public request contract, rejected Anthropic zero-token structured requests,
  and restricted Pydantic structured formats to actual `BaseModel` subclasses.

### Security

- Defined the Ray driver, workers, submitted code, and east-west network as one
  trusted boundary. Cross-worker local-disk shuffle uses a worker-owned plaintext
  Flight service, while same-process and object-storage reads remain network-free;
  worker identity is now separate from the advertised Flight endpoint.
- Documented the trust boundaries around Python UDFs, Ray workers, credentials, native parsers, and remote model code.
- Redacted AI provider credentials from descriptor and provider-option `repr`,
  logs, exception formatting, and assertion diffs; plaintext is revealed only at
  provider execution, and SQL continues to reject inline credentials. Option
  mappings held by AI descriptors now store sensitive values wrapped in an
  internal secret type, so code that compared those mappings against plain
  dictionaries must compare revealed values instead.

## 0.1.0a1

First planned public alpha release. Not yet published.
