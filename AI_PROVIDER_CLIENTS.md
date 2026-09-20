# AI provider client configuration

OpenAI, Google, and Anthropic providers capture client settings in the
application process. Their embedding and Prompt descriptors carry that snapshot
to the executing worker, including when Vane connects to an existing Ray
cluster. No runner settings or worker environment changes are required for
credential transport. Workers still need the selected provider's optional SDK
and network access to the configured endpoint.

## Explicit configuration

Create a provider once and pass it to the existing AI functions:

```python
import os
import vane
from vane.ai import embed, prompt
from vane.ai.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.openai.com/v1",
    # Optional: organization="org-...", project="proj_..."
)

with vane.connect() as connection:
    documents = connection.sql("SELECT 'A document' AS text")
    vectors = documents.select(
        embed(vane.col("text"), provider=provider, model="text-embedding-3-small")
    ).fetchall()
    answers = documents.select(
        prompt(vane.col("text"), provider=provider, model="gpt-4.1")
    ).fetchall()
```

`vane.ai.load_provider("openai", api_key=..., base_url=...)` accepts the same
constructor settings. Client credentials remain separate from inference
`options`; passing credentials in SQL or in embedding/Prompt inference options
is rejected. Existing per-call `base_url` and `timeout` options override the
provider endpoint and request timeout where supported.

Google uses the same client snapshot for embedding and Prompt:

```python
from vane.ai.providers.google import GoogleProvider

google = GoogleProvider(
    api_key=os.environ["GOOGLE_API_KEY"],
    vertexai=False,  # Explicit Gemini Developer API mode
)
```

For Vertex/Enterprise mode, pass `vertexai=True`. Supported authentication
sources are an API key or an explicitly supplied, serializable
`google.auth.credentials.Credentials` object. OAuth credentials require a
project; location defaults to `global` if absent from the application settings.
Obtain ADC in the application if needed and pass the resulting credentials and
project explicitly. Vane does not run ADC discovery on the worker, transport
credential-file paths, or implement a credential refresh broker. Refreshable
credential objects still require their own refresh dependencies on the worker.

```python
import google.auth

credentials, project = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
vertex = GoogleProvider(
    vertexai=True,
    credentials=credentials,
    project=project,
    location="us-central1",
)
```

Anthropic supports `api_key` or `auth_token` (exactly one), plus `base_url`:

```python
from vane.ai.providers.anthropic import AnthropicProvider

anthropic = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
# Pass model=... and max_tokens=... to Prompt as usual.
```

## Environment settings and snapshot lifetime

Omitted or `None` constructor arguments read the application environment when
the provider is created. Using a provider name directly in an AI function or
SQL resolves and captures its settings while binding that expression. Creating
a descriptor directly captures settings at descriptor construction. Binding,
schema inspection, and EXPLAIN do not contact a model or discover cloud
credentials.

| Provider | Captured application variables |
| --- | --- |
| OpenAI | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` |
| Google | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`), `GOOGLE_GENAI_USE_ENTERPRISE` (or `GOOGLE_GENAI_USE_VERTEXAI`), `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and the selected `GOOGLE_GEMINI_BASE_URL` / `GOOGLE_VERTEX_BASE_URL` |
| Anthropic | `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL` |

Explicit arguments take precedence. An explicit Anthropic credential selects
that authentication method without reading the other credential from the
environment. An explicit Google credentials object does not read an API key.
Google cloud project/location environment settings only apply in Vertex mode;
Developer API mode is the default when neither backend-selection variable is
set. Enterprise takes precedence when both mode variables are set, matching
the SDK's selection rule.

The snapshot includes absent optional identity fields. Workers cannot fill in
an absent OpenAI organization/project or choose a different authentication
method from their environment. SDK-specific ambient custom headers are not
used. Vane does not mutate shared process environments during construction.
Changing application environment variables does not change an existing
provider or descriptor: construct a new provider to rotate credentials.

Missing credentials fail when inference initializes the client, with an error
that identifies the missing application configuration. Metadata-only calls
remain usable without credentials. There is no worker-identity fallback.

## Serialization boundary

This implementation sends actual credential values in serialized descriptors
and execution payloads. `repr`, inference options, and sanitized execution
errors keep credentials redacted; redaction does not encrypt serialized bytes.
Anyone allowed to read these payloads may recover credentials. Opaque secret
references and credential brokers are separate follow-up work tracked by
[#243](https://github.com/AstroVela/vane/issues/243).

## Regression checks

`tests/fast/test_ai_client_config.py` checks serialized descriptors with real SDK
constructors, conflicting worker settings, missing credentials, Google Vertex
authentication, and concurrent client isolation. Its owned-cluster Ray cases
start the cluster before application configuration and exercise embedding and
Prompt through all four Python execution backends with real SDK requests to a
local HTTP fixture. They use fake credentials and never call live model APIs.
Install the OpenAI, Google, and Anthropic extras to run those SDK cases.

The release launcher runs non-Ray, shared-cluster Ray, and owned-cluster Ray
checks in separate processes. See [DEVELOPMENT.md](DEVELOPMENT.md) for the
installed-package workflow.
