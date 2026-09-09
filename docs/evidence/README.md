# Portable calibration evidence

`nuscenes_calibration_v1/` contains the five validated locked-evaluation summaries:
metrics, paired scene intervals, recovery, identity-only timing, and exclusions.
They cover the frozen cohort and retain source hashes and explicit support counts.
They contain no raw sensor data, sample/box tokens, manifests or model weights.

See [analysis reproduction](../verification/analysis-reproduction.md) for source
revisions, numerical pairing policy, validation, and byte-comparison hashes.
Method-performance claims and publication readiness require the separate claims
audit and experiment/model-card steps; these files alone do not imply a release.
