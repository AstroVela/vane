# Iceberg knowledge synchronization

`python -m vane.knowledge` runs a local Python service that watches one Iceberg
branch and synchronizes document changes to OpenWiki, GBrain, or both. It uses
the catalog's committed snapshots and never watches object-store file creation.
The service requires a POSIX host, a private local state directory, and an
already configured destination CLI. No native engine changes are required.

## Install and configure

Install the optional dependencies into the same environment as Vane:

```bash
pip install 'vane-ai[knowledge]'
```

The supported reader is PyIceberg 0.12.x; the OpenWiki transport uses the Python
MCP SDK 2.x (at least 2.2). Configure the named catalog through
[PyIceberg configuration](https://py.iceberg.apache.org/configuration/), including
its object-store credentials. Keep credentials out of the sync configuration.
The service does not store catalog credentials in its checkpoint or pass them
to an LLM.

Create `sync.json`:

```json
{
  "name": "company-docs",
  "catalog": "production",
  "table": ["knowledge", "documents"],
  "branch": "main",
  "id_column": "doc_id",
  "content_column": "body",
  "title_column": "title",
  "uri_column": "source_url",
  "state_dir": "./company-docs-state",
  "poll_seconds": 30,
  "timeout_seconds": 600,
  "targets": [
    {
      "name": "brain",
      "kind": "gbrain",
      "command": ["gbrain"],
      "cwd": "./brain-workspace",
      "destination": "default"
    },
    {
      "name": "wiki",
      "kind": "openwiki",
      "command": ["openwiki"],
      "cwd": ".",
      "destination": "custom-mcp-company-docs"
    }
  ]
}
```

Remove either target if only one destination is needed. Paths are relative to
the configuration file; target working directories must already exist. Catalog
and CLI authentication use the service's environment. `destination` is a GBrain
source ID or a specific OpenWiki source instance ID, respectively.

Each row must have a unique, stable string or integer ID and nonempty text
content. The optional title and URI columns must also contain nonempty text;
omitting them derives a title from the ID and an `iceberg://` provenance URI.
The default maximum serialized document size is 64 KiB, adjustable downward
with `max_document_bytes`. Split larger documents upstream. IDs are distinct
across types: integer `1` and string `"1"` identify different documents.

For a table of business facts, first use Vane SQL/AI operations to materialize
an Iceberg document table containing entity summaries. For PDFs or media,
materialize extracted text. This connector reads the configured text columns;
it does not fetch document URLs, parse binary files, or infer a schema.

## GBrain

Use a GBrain installation supporting the committed page-write receipt contract
in [page operations](https://github.com/garrytan/gbrain/blob/d13aa742fd68b71bfd6c98be3dda5813791f1d6c/src/core/ops/pages.ts)
and [write receipts](https://github.com/garrytan/gbrain/blob/d13aa742fd68b71bfd6c98be3dda5813791f1d6c/src/core/persistence/service.ts).
Initialize its source and authentication before running this service.

The adapter invokes `gbrain put` with Markdown on stdin, or `gbrain delete` for
a removed row, using `--source-id`, `--force`, `--request-id`, and `--json`.
It requires a matching **committed** receipt and outcome before acknowledging
each event. The same event keeps the same request UUID across crashes and
timeouts. A successful exit without that receipt is a failed delivery.

Pages under `iceberg/<name>/` are exclusively connector-owned: the adapter
replaces them and soft-deletes them as the table changes. Do not manually edit
those pages. Give each independent source/branch a unique `name`. A later
upsert can recreate a deleted document at the same stable slug. The receipt
confirms the canonical page and text projection; embedding and graph enrichment
have GBrain's separate completion semantics. This adapter does not use
`IngestionSource.emit()`, which has no durable acknowledgement.

## OpenWiki

Initialize a personal wiki and configure its model/authentication first. Then
generate the custom MCP source entry:

```bash
python -m vane.knowledge openwiki-config --config sync.json --target wiki
```

Add the printed object to `sourceInstances` in OpenWiki's `onboarding.json`.
Preserve existing entries and the rest of the onboarding configuration. Save
the same object's `connectorConfig` as `connectors/custom-mcp/config.json`
under that OpenWiki home as well. Live OpenWiki MCP tool calls read this shared
connector file, not the source-instance override. Use a dedicated OpenWiki home
if its existing `custom-mcp` configuration belongs to another source.

The entry specifies the installed Python interpreter and absolute state directory,
so OpenWiki can launch the MCP server without access to the source checkout.
Regenerate this entry if you move the Python environment or state directory.
Do not schedule additional ingestion runs for this source: this service owns
its ingestion lifecycle.

The service runs `openwiki ingest <instance> --print` for bounded windows of the
durable batch (at most 100 events and approximately 64 KiB per run). The tools
`list_changes` and `read_changes` expose only the active window, containing
explicit upsert/delete events and provenance. They cannot query arbitrary SQL,
read arbitrary files, or mutate the lake. After applying the changes, OpenWiki
calls `acknowledge_changes` with the receipt returned alongside their content.
This third tool updates only local delivery state and is explicitly included
in the generated connector's `allowedTools`; it is not marked read-only.

A window advances only after OpenWiki exits successfully **and returns that
attempt's receipt**. A lost read response, skipped source, stale receipt, or
failed synthesis leaves the window pending. Completed windows are preserved
across retries. This confirms source consumption and successful synthesis
execution; it cannot prove that an LLM revised every dependent sentence correctly.

The integration targets OpenWiki's
[custom MCP connector](https://github.com/langchain-ai/openwiki/blob/715109a8ab1cda6d47680fcc8170e203c751bf61/src/connectors/sources/mcp.ts)
and [ingest CLI](https://github.com/langchain-ai/openwiki/blob/715109a8ab1cda6d47680fcc8170e203c751bf61/src/cli/runners.ts).
Raw document content remains untrusted evidence during synthesis. The private
state database contains document text, including unacknowledged deletions;
protect and retain it accordingly.

## Run and recovery

```bash
python -m vane.knowledge sync --config sync.json --once
python -m vane.knowledge sync --config sync.json
```

The first command makes one attempt and returns nonzero on failure. The second
polls continuously and retries destination failures after `poll_seconds`.
Run it under your usual service supervisor for restart after a catalog or
process failure. SIGINT/SIGTERM stops the service and its current destination
process group; restart with the same configuration and state directory.

The state directory is created with mode 0700, and its database with mode 0600.
Existing public directories, symlink state files, and unknown state schemas are
rejected. Use local storage with working SQLite transactions and POSIX file
locks. One writer owns the directory; MCP readers can run alongside delivery.

The first scan pins one snapshot and imports its document projection. Each
subsequent change validates table UUID and branch ancestry, pins the new head,
and compares its **complete selected-column projection** against the durable
document index. This is incremental *delivery*, not a row-level CDC reader:
reading cost is proportional to that projection on every changed snapshot.
It coalesces intermediate commits into net document changes. A history segment
consisting solely of `replace` snapshots advances the checkpoint without a
data scan. An append followed by compaction is still scanned.

Changes and the new document index enter SQLite in one transaction. Destinations
consume the immutable batch independently; successful deliveries are not
repeated just because another destination failed. The applied snapshot advances
only when every destination has confirmed the batch. An interrupted scan leaves
the old index intact. An interrupted delivery resumes its durable batch before
contacting the catalog. Delivery is at least once; GBrain receipts provide
idempotent replay, while OpenWiki may synthesize the same batch again.

There are no compatibility shims, state migrations, or automatic fallback paths.
A changed table UUID, branch rollback/divergence, expired required history,
duplicate/null IDs, unsupported reader features (including equality deletes),
or invalid document data fails explicitly. Configure Iceberg snapshot retention
to exceed maximum service lag and scan duration. Do not delete checkpoint files
to hide these errors: initializing a new state does not remove stale pages from
an old destination. Changing the table, mapping, or destination binding requires
a separately provisioned state directory and destination namespace.

## Development and dependencies

Follow [DEVELOPMENT.md](DEVELOPMENT.md) for package installation and the base
release gate. Run the affected tests with the installed-package launcher:

```bash
uv pip install --group knowledge-test
scripts/run_installed_pytest.sh tests/fast/test_knowledge_sync.py tests/fast/test_knowledge_integration.py
scripts/run_release_tests.sh
```

Unit tests require only the base installation. The integration tests use an
actual local SQLite Iceberg catalog, Parquet files, and the MCP stdio server.
Destination contract fixtures stand in for GBrain storage and paid OpenWiki
synthesis; no production service or model credentials are used.

PyIceberg is Apache-2.0; the Python MCP SDK is MIT. They are separately installed
optional dependencies, not copied into Vane. SQLAlchemy (MIT) is a test-only
dependency for the local catalog. No upstream implementation code is vendored.
