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
- A Ray cluster is part of the trusted computing base. Use Ray authentication, network isolation, least-privilege identities, and compatible package versions on every node.
- Model repositories can contain executable custom code. Keep remote-code loading disabled unless a trusted model specifically requires it, and pin reviewed model revisions.
- API keys and cloud credentials may be propagated to workers. Prefer short-lived, scoped credentials and secret managers; never place secrets in SQL text, logs, source files, or benchmark output.
- Image, video, audio, document, Parquet, Arrow, and compressed inputs reach native parsers. Process hostile inputs in isolated workers with resource limits.
- SQL can consume unbounded CPU, memory, storage, network, or model tokens. Multi-tenant deployments need admission control, quotas, and cancellation outside Vane's current defaults.

## Secure deployment baseline

- Run the driver and workers as unprivileged users in isolated networks.
- Restrict worker egress and filesystem access to what a pipeline needs.
- Pin Vane, Ray, model, container, and native dependency versions.
- Keep credentials outside source and rotate any credential exposed in output.
- Disable optional providers and extension auto-installation when they are not required.
- Review the release checksums, signatures, SBOM, provenance, and third-party notices before deployment.

Resource exhaustion from an intentionally expensive trusted query is usually an operational issue rather than a vulnerability. A sandbox escape, cross-tenant data exposure, unsafe default credential handling, signature bypass, or code execution across a stated trust boundary should be reported privately.
