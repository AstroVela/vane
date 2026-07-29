# Security policy

Vane is an alpha developer preview. It has not undergone a complete independent security review and should not be exposed to untrusted tenants or untrusted code without additional isolation.

## Supported versions

| Version | Security fixes |
| --- | --- |
| Latest default branch | Yes |
| Latest `0.1.x` prerelease | Best effort |
| Older commits and prereleases | No |

## Report a vulnerability

Use GitHub's **Report a vulnerability** private advisory flow for this repository. Do not open a public issue for an unpatched vulnerability and do not include live credentials or confidential datasets. If private vulnerability reporting is unavailable, contact a maintainer privately through the profile listed in [MAINTAINERS.md](MAINTAINERS.md) and ask for a secure channel before sending details.

Include the affected commit or version, impact, prerequisites, a minimal reproducer, and any suggested mitigation. You should receive an acknowledgement within five business days. Timelines for validation, fixes, and disclosure depend on severity and maintainer availability.

## Trust model

Several Vane features intentionally execute code. Treat these boundaries explicitly:

- Python UDFs and Cloudpickle payloads can execute arbitrary Python in the driver or Ray workers. Never deserialize or run a callable from an untrusted source.
- A Ray cluster is one trusted computing boundary: the driver, every worker, submitted Python/UDF/native code, the east-west network, and administrators can affect one another. Follow [Ray's security guidance](https://docs.ray.io/en/latest/ray-security/index.html), including network isolation, least-privilege identities, and compatible package versions on every node.
- Cross-worker local-disk shuffle uses plaintext Arrow Flight. It provides neither transport encryption nor client authentication.
- Model repositories can contain executable custom code. Keep remote-code loading disabled unless a trusted model specifically requires it, and pin reviewed model revisions.
- API keys and cloud credentials may be propagated to workers. Prefer short-lived, scoped credentials and secret managers; never place secrets in SQL text, logs, source files, or benchmark output.
- Image, video, audio, document, Parquet, Arrow, and compressed inputs reach native parsers. Process hostile inputs in isolated workers with resource limits.
- SQL can consume unbounded CPU, memory, storage, network, or model tokens. Multi-tenant deployments need admission control, quotas, and cancellation outside Vane's current defaults.

## Plaintext Flight transport

Vane supports plaintext Arrow Flight inside a controlled, isolated Ray cluster. Same-process local-disk reads use the process-local registry, object-storage reads use committed manifests, and only cross-worker local-disk reads use Flight. Object-storage failures do not fall back to a producer's local Flight endpoint.

Each worker process owns at most one lazily started Flight service. A Flight ticket identifies a live published exchange attempt and partition; it is not an authentication or authorization credential. The service validates the ticket format, service epoch, producer identity, selected attempt, committed manifest, partition, query lifecycle, and reader lease.

Distributed exchange plans and task results are a same-version contract and do not support mixed-version endpoint metadata. Drain in-flight distributed queries and recreate persistent Ray worker actors when upgrading Vane.

Vane does not provide confidentiality between users or jobs in one Ray cluster, protection from a malicious worker, or per-query authorization for this transport. Deploy mutually untrusted workloads in separate Ray clusters and separate networks.

The worker's Ray private address is used for binding and advertisement by default. Operators may set `VANE_FLIGHT_BIND_HOST=0.0.0.0` when container networking requires a wildcard listener, but `VANE_FLIGHT_ADVERTISE_HOST` must be a routable non-wildcard address. The advertised-host override belongs to the worker node's environment and is not copied from the driver to every worker. Vane rejects this override in a Ray Job or actor runtime environment because Ray inherits those values across nodes. Firewall, Security Group, or NetworkPolicy rules must restrict the configured or dynamically allocated `DUCKDB_FLIGHT_PORT` to the same Ray cluster. Do not expose it through a public Service, Ingress, LoadBalancer, or NodePort.

## Secure deployment baseline

- Run the driver and workers as unprivileged users in isolated networks.
- Keep Ray and Vane Flight ports reachable only from the same trusted cluster.
- Restrict worker egress and filesystem access to what a pipeline needs.
- Pin Vane, Ray, model, container, and native dependency versions.
- Keep credentials outside source and rotate any credential exposed in output.
- Disable optional providers and extension auto-installation when they are not required.
- Review the release checksums, signatures, SBOM, provenance, and third-party notices before deployment.

Resource exhaustion from an intentionally expensive trusted query is usually an operational issue rather than a vulnerability. A sandbox escape, cross-tenant data exposure, unsafe default credential handling, signature bypass, or code execution across a stated trust boundary should be reported privately.
