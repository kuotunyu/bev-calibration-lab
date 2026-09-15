# bev-calibration-lab

[正體中文](README.md) · [Documentation](docs/README.md) · [Experiment card](docs/experiment-card.md) · [Model card](docs/model-card.md)

How much does a small error in LiDAR-camera calibration metadata cost you, and how much of it
can you get back?

This repository measures that on nuScenes. Controlled rotation and translation faults
are evaluated with identity, a classical 6DoF optimizer, and three training seeds of a
ConvNeXtV2-Tiny corrector. Original sensor contents stay unchanged. Timing separately
selects a LiDAR sweep relative to a fixed camera exposure and remains identity-only
stress evidence, not a seventh recoverable pose axis.

**Status: formal training, inference, analysis and presentation acceptance are complete;
v1.0.0 is published.** See the [release](https://github.com/kuotunyu/bev-calibration-lab/releases/tag/v1.0.0)
and [hosted presentation](https://kuotunyu.github.io/bev-calibration-lab/).
The [five formal documents](docs/evidence/README.md)
are preserved, with byte-identical results from two independent CPU analyses. The
[reproduction record](docs/verification/analysis-reproduction.md) identifies their sources
and limitations. Publication and engineering acceptance do not establish corrector efficacy.

Retain every seed, predefined condition, identity and zero-fault reference when reading
the results. A fixed-three-seed mean is not a prediction ensemble. Ground-contact
reconstruction is neither object-detector performance nor real-vehicle safety evidence.
The [analysis contract](docs/contracts/formal-analysis.md) defines units, paired support,
unavailable estimates and intervals.

## Reading the results

Start with zero fault: identity preserves the calibration reference, while classical
and each learned seed can degrade it. Then compare recovery and paired intervals
across fault conditions, followed by BEV error, support and missing estimates.
Weak recovery and zero-fault degradation remain results of the study; successful
training or outperforming one baseline is insufficient evidence of reliable correction.

- [Recovery](docs/figures/recovery-by-fault-level.svg) retains identity, classical,
  individual learned seeds and predefined conditions. The seed mean is not an ensemble.
- [BEV error](docs/figures/bev-error-by-range.svg) shows ground-contact reconstruction;
  the ground-plane assumption itself can leave a nonzero residual.
- [Bounded diagnostics](docs/verification/corrector-diagnostics.md) cover checkpoint
  identity, inverse transforms and units. They neither explain every weak outcome
  nor exclude every implementation defect.

The [calibration explorer](docs/demo/calibration-explorer.html) explains perturbations
using fixed synthetic geometry. It does not display formal learned-corrector performance;
use the formal evidence for method comparisons.

## The coordinate contract

Getting this wrong silently is the most expensive mistake available in this problem, so it is
fixed in one place and enforced by types and tests.

- Every transform is named `T_target_source` and maps points expressed in `source` into
  `target`.
- Public point arrays are `[N, 3]` row vectors. Any column-vector convention stays inside an
  adapter and never reaches a public signature.
- The formal chain is LiDAR sensor to LiDAR-time ego to global to camera-time ego to camera
  sensor. Two ego poses, not one, because the LiDAR sweep and the camera exposure do not
  happen at the same instant.
- 3D boxes are natively in global coordinates.
- LiDAR features keep `x, y, z, intensity, ring`.

The full rules, the row-vector boundary, the two ego timestamps and a worked numeric
example are in [docs/coordinate-contract.md](docs/coordinate-contract.md).

## Sensors and cohort

`CAM_FRONT` and `LIDAR_TOP`. The formal cohort is 150 nuScenes scenes: 100 official-train
development, 20 calibration scenes drawn from distinct logs, and 30 official-validation
scenes reserved for locked evaluation. Scene assignment is stratified by location and ordered
by token SHA-256, so it is reproducible and independent of anything measured. The nuScenes
mini split is for development and integration only and never appears in a reported result.

## Requirements

Python 3.12 and [uv](https://docs.astral.sh/uv/) 0.11.x. nuScenes data is licensed to the
account holder, is not distributed here, and is never committed.

```bash
# The train extra is not optional for development: the learned corrector is
# first-party code and the coverage gate covers it.
uv sync --frozen --all-groups --extra train --extra report
uv run bev-calib --help
```

Native commands, private aggregation and portable claim-bound rendering are documented
in [docs/commands-and-report.md](docs/commands-and-report.md).

## Quality gates

`uv run python -m bevcalib.dev verify` runs every gate in a fixed order and stops at the
first failure: the private-file guard, format check, lint, type check, the full test suite,
100% statement and branch coverage on first-party code, schema contracts, and documentation
links. Coverage exemptions are not used; there are no `pragma: no cover` comments and no
omitted first-party paths.

## Licence

MIT, see [LICENSE](LICENSE). nuScenes itself is distributed under its own terms by Motional
and is not redistributed here.

## Formal evidence views

[Recovery](docs/figures/recovery-by-fault-level.svg), [BEV error](docs/figures/bev-error-by-range.svg), and the [complete claims registry](docs/claims.yaml) preserve the declared conditions and support. See [publication and interchange verification](docs/verification/publication-and-interchange.md) for the completed presentation acceptance and release checks.
