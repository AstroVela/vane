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

- AI execution options now take effect on every surface instead of being
  silently dropped: explicit `batch_size` and the bare `concurrency` kwarg
  reach the API prompter/embedder descriptors (non-positive values are
  rejected, matching SQL), API providers no longer require `num_gpus` for
  actor fan-out on the relation path, and provider-level OpenAI defaults
  (`batch_size`, `max_api_concurrency`, `on_error`) are routed to the
  request options.
- The vLLM `on_error` vocabulary (`ignore`/`null`) is translated at the
  engine boundary, and `return_format` smuggled inside option dicts or
  `image_columns` on a text-only batch prompter now raise clear errors.
- Chunked embedding honours `max_retries`/`on_error`, and `on_error="log"`
  now emits a warning log for substituted failures.
- A failed batch call falls back to one attempt per row so only
  genuinely-bad rows are substituted, and `RetryAfterError` survives
  pickling across worker boundaries.

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
