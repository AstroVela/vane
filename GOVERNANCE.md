# Governance

Vane is an independent, maintainer-led open-source project. It borrows values such as public decision-making, merit, and consensus from the Apache Way, but it is not governed by or affiliated with the Apache Software Foundation.

## Roles

- **Contributor**: anyone who reports, discusses, documents, tests, or submits a change.
- **Committer**: a trusted contributor who may merge changes within an agreed area after review.
- **Maintainer**: a committer responsible for project direction, releases, security response, governance, and repository administration.

Sustained, constructive, technically sound participation is the basis for additional responsibility. Maintainers nominate new committers or maintainers in a public issue. Existing non-recused maintainers seek consensus; when there are at least three maintainers, a promotion requires at least two approvals and no unresolved veto based on a concrete project risk.

## Decisions

Routine changes use pull-request review. Larger changes begin with a public issue or design document and normally remain open for at least 72 hours so affected contributors can respond.

The preferred outcome is consensus. A technical veto must identify a specific correctness, compatibility, security, legal, operational, or maintenance risk and explain what would resolve it. If consensus is not possible, non-recused maintainers decide by simple majority and record the reasoning. With only one active maintainer, that maintainer makes the final decision after a reasonable public comment period.

Security incidents, credential exposure, legal takedowns, and active abuse may require private or immediate action. The project will publish a non-sensitive record afterward when possible.

## Releases

A maintainer proposes a release using [RELEASE.md](RELEASE.md). Until multiple release managers are available, one maintainer may approve a prerelease after all automated gates pass and the artifact contents are manually reviewed. Stable releases should have a second maintainer review.

## Inactivity and removal

A role may be marked emeritus after twelve months without project activity, with no loss of credit. Access may be suspended immediately for a security incident or serious Code of Conduct concern. Permanent involuntary removal requires a documented decision by non-recused maintainers.

Governance changes use the same public design process as other major changes.
