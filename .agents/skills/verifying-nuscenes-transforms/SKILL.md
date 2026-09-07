---
name: verifying-nuscenes-transforms
description: Use when implementing, reviewing, debugging, or reporting nuScenes LiDAR-to-CAM_FRONT transforms, point projection, frame direction, sensor timestamp handling, visibility masks, or official-devkit parity in this repository.
---

# Verifying nuScenes transforms

## Core rule

A plausible overlay is not evidence. Record the typed chain, both sensor times,
the public point shape, a behind-camera counterexample, and absolute-only devkit
parity before accepting or reporting a projection.

## Resolve the repository's native records

`NuScenesInstallation` stores the raw nuScenes tables but does not add the
official devkit's `sample["data"]` channel map. Resolve a `SceneRecord` through
`installation.scene_records()`, find the sample index, and use its paired
`camera_sample_data_tokens` and `lidar_sample_data_tokens`. Use
`sample["data"]` only when the object actually came from the official devkit.

## Trace this exact chain

| Applied transform | Target | Source | Time evidence |
| --- | --- | --- | --- |
| `T_lidar_ego_lidar_sensor` | `lidar_ego` | `lidar_sensor` | none |
| `T_global_lidar_ego` | `global` | `lidar_ego` | `lidar_timestamp_us` |
| `T_camera_ego_global` | `camera_ego` | `global` | `camera_timestamp_us` |
| `T_camera_sensor_camera_ego` | `camera_sensor` | `camera_ego` | none |

The camera-side transforms are inverses of the poses nuScenes stores. Keep the
LiDAR and camera timestamps as separate integer fields even when their values
happen to match.

Public points are float64 `[N, 3]` row vectors. Column-major `3×N` arrays may
exist only inside the official-devkit comparison adapter. Projection must retain
one row per input and keep `in_front`, `in_image`, and `valid` separate. Include
a negative-depth point that lands numerically in the image and prove it is
`in_front=false`, `in_image=true`, `valid=false`.

## Validate the evidence

Create a private JSON trace using the executable example in
`tests/contract/skills/test_transform_skill.py::clean_trace`. It must record the
native `scene_records` lookup, both timestamps, `[N, 3]`, the four directed
links, the behind-camera test, and four nonempty parity cases: LiDAR projection
and global box centres on each of the two pinned mini samples.

Use the same native intrinsic and this repository's depth/mask policy. Compare
camera-frame points, projected UV/depth, and box centres to official-devkit
primitives with `atol=1e-6, rtol=0`. Do not adopt visualization-only margins or
minimum-depth filters as scientific policy.

```bash
uv run --frozen python .agents/skills/verifying-nuscenes-transforms/scripts/validate_transform_chain.py --trace PATH
NUSCENES_ROOT=/path/to/nuscenes uv run --frozen pytest tests/integration/test_nuscenes_mini_parity.py -m slow -vv -s
```

Exit 0 means the trace is complete and every recorded maximum absolute error is
at most `1e-6`. Exit 1 means the trace is unsafe; correct the implementation or
evidence before reporting parity. Exit 2 means the trace could not be checked.

## What this skill is not for

It does not authorize data download, cohort freezing, training, calibration
fault studies, or changing geometry operators to make parity pass.
