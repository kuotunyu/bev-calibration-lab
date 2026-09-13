# Formal publication and interchange verification

On 2026-09-13, validator/exporter commit
`b4f6272ae0c2f7723dd4f30c42f58fa3e9447e97` passed the real-input validation
entry point with the frozen protocol, independently recorded source identity,
three checkpoint digests and all 150 scene files. It verified 90,126 scalar
bindings. This validates the declared inputs and source; it does not rerun the
analysis or establish calibration efficacy.

Two fresh publication outputs contained identical claims, HTML, five evidence
copies and two SVG files. The five original evidence files were unchanged.
The claims SHA-256 is
`fd6d25a6bc9c0023cbf862b45c67b1022498602f99691009d236ccb442c048d0`.
The complete [registry](../claims.yaml) is intentionally large: it binds every available scalar in the predeclared publication mapping.
Other descriptors remain available in the complete metrics evidence.

- [Recovery by fault level](../figures/recovery-by-fault-level.svg)
- [BEV error by ground-truth range](../figures/bev-error-by-range.svg)

The figures retain all methods, seeds, predefined conditions, support and missing
values. Pointwise intervals are not simultaneous bands. Timing remains a separate
identity-only stress test. Browser inspection remains pending.

## Optional cross-project artifact

The [distribution](../evidence/calibration-distribution-v1.json) was exported twice
from the real identity run, at the predefined yaw +1 degree condition, using the
same UTC timestamp. Both files have SHA-256
`13c1ce28532080ec11636e0633381314ce9e3093f930bb03332246f8fc839228`.
It contains 30 scene means from 1,207 pose-valid frames, with no exclusions.
The consumer's pinned container accepted it and refused separate corruptions of
payload digest, artifact type and protocol hash for their expected reasons.

This is the uncorrected identity condition: its yaw error is the injected value,
not evidence of learned correction. Interchange acceptance does not demonstrate
an AEB benefit or alter the consumer's formal experiment configurations.
The `v1.0.0` label records the intended release, not an existing public tag.

## Remaining acceptance

The generated assets and their registry audit passed the eight-stage gate before
publication-source commit `4d75808131884536e6b1705518ee84fc4d9f651c` was created.
Both test rounds passed 1,362 tests, with four explicit mini-data skips; first-party
statement and branch coverage were 100%. This is the publication-source gate,
not verification of later release-engineering changes.
Human figure/explorer inspection, clean final package builds, Linux clean-clone
checks and public release checks remain separate. No raw sensor files or model
weights are included.
