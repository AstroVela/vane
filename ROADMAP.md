# Roadmap

The roadmap communicates direction, not a commitment to dates or compatibility.

## Current: Vane Data alpha

- Make local and Ray execution behavior observable and reproducible.
- Stabilize expression UDFs, batching, backpressure, cancellation, retries, and progress reporting.
- Define supported Python, Ray, operating-system, and GPU matrices from automated evidence.
- Publish auditable sdists and wheels with licenses, checksums, signatures, provenance, and SBOMs.
- Turn multimodal benchmarks into versioned, repeatable performance gates.
- Improve isolation and documentation for executable UDF and model workloads.

## Next: public API and operational hardening

- Version the distributed plan and worker protocols.
- Document compatibility and deprecation policies.
- Add multi-node correctness, chaos, upgrade, and recovery testing.
- Add resource governance, admission control, and clearer failure diagnostics.
- Expand platform support only after release CI validates it.

## Exploratory

Vane RL and Vane Agent are research directions. They will be proposed and reviewed independently if concrete implementations become ready. Their names in project material do not imply that those products or APIs currently exist.
