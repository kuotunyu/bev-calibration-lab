# bev-calibration-lab

How much does a small error in LiDAR-camera calibration metadata cost you, and how much of it
can you get back?

This repository measures that on nuScenes. It injects controlled rotation, translation and
timing faults into the sensor calibration chain, holds the observations fixed, and reports
what the faults do to LiDAR-camera edge alignment and to bird's-eye-view ground-contact
reconstruction. Three correctors then try to recover the true extrinsics: an identity
baseline that does nothing, a classical 6DoF optimizer, and a learned ConvNeXtV2-Tiny
corrector.

**Status: under construction.** Packaging foundation only. No dataset has been read, no
experiment has been run, and this README carries no results. Every number published here
later will be labelled `observed`, `derived`, `synthetic` or `illustrative`, and every
`observed` number will be reproducible from a committed artifact.

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
uv sync --frozen --all-groups --extra train
uv run bev-calib --help
```

## Quality gates

`uv run python -m bevcalib.dev verify` runs every gate in a fixed order and stops at the
first failure: the private-file guard, format check, lint, type check, the full test suite,
100% statement and branch coverage on first-party code, schema contracts, and documentation
links. Coverage exemptions are not used; there are no `pragma: no cover` comments and no
omitted first-party paths.

## Licence

MIT, see [LICENSE](LICENSE). nuScenes itself is distributed under its own terms by Motional
and is not redistributed here.
