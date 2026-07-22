# Changelog

All notable user-visible changes are documented here. Vane is currently in alpha, so incompatible changes may occur between prereleases.

## Unreleased

### Added

- Public governance, contribution, security, release, provenance, and third-party documentation.
- Release artifact validation and a reproducible native dependency license bundle.

### Changed

- Positioned the current project as the Vane Data developer preview.
- Defined `DuckDBPyRelation.map` exclusively as a row-wise scalar UDF with a
  required `return_type`; batch transforms use `map_batches` with an explicit
  output `schema`. The inherited pandas DataFrame-style DuckDB `map` contract
  is no longer supported.
- Restricted source distributions to the DuckDB components required by Vane.
- Imported the official DuckDB baseline as a squashed Git subtree and retained
  Vane engine customizations as monorepo commits, so normal clones no longer
  require submodule initialization or carry DuckDB's complete commit history.

### Fixed

- OpenAI embedder: empty or whitespace-only rows are no longer sent to the
  API (the API rejects empty strings, failing the whole batch); they receive
  deterministic zero-vector placeholders, or nulls when the embedding
  dimension is unknowable.
- OpenAI embedder token estimation counts each non-ASCII character as a full
  token, so CJK text is chunked conservatively instead of being sent
  oversized.
- Chunks of oversized embedding inputs are sent through the same
  token-limited request batching instead of one unbounded request, and
  chunk-averaged embeddings are float32 like normal rows.
- Non-positive `batch_token_limit`/`input_text_token_limit` values are
  rejected early, and a legitimate token usage of 0 is recorded as 0 instead
  of null.

### Security

- Documented the trust boundaries around Python UDFs, Ray workers, credentials, native parsers, and remote model code.
- Redacted AI provider credentials from descriptor and provider-option `repr`,
  logs, exception formatting, and assertion diffs; plaintext is revealed only at
  provider execution, and SQL continues to reject inline credentials. Option
  mappings held by AI descriptors now store sensitive values wrapped in an
  internal secret type, so code that compared those mappings against plain
  dictionaries must compare revealed values instead.

## 0.1.0a1

First planned public alpha release. Not yet published.
