# Experiment card: sensitivity to and recovery from extrinsic faults

[正體中文](experiment-card.md)

## Research question

On fixed nuScenes sensor observations, how does a LiDAR-camera extrinsic error affect
geometric error, and how much of it can identity, classical optimization and a learned
corrector recover? The study measures geometry and correction behaviour. It does not
treat ground-contact reconstruction as detector accuracy, and it includes no
closed-loop AEB or real-vehicle safety validation.

## Frozen design and data use

The study uses `CAM_FRONT` and `LIDAR_TOP`. 100 official-train development scenes are
used for training; 20 official-train calibration scenes, from distinct logs, fix the
calibration objective and select checkpoints; 30 official-val scenes are reserved for
the locked evaluation. Logs, scenes, samples and sensor identifiers are checked not to
overlap between roles. The mini split serves only development and integration tests.

Assignment is stratified by location and ordered by token SHA-256, and the cohort cannot
be changed according to results. The fixed settings are in the
[protocol](../configs/protocols/nuscenes_calibration_v1.yaml), the
[fault matrix](../configs/perturbations/formal_v1.yaml) and the
[cohort contract](cohort-contract.md). Rotations and translations are applied one axis
at a time. Timing is a separate identity-only stress test of sweep selection with no
learned timing recovery; in v1 it carries no information (see
[known issues](errata.md)).

### Fault axes

Faults are composed on the camera side of the CAM_FRONT extrinsic, so the formal axis
names refer to the camera optical frame (x right, y down, z forward), not to vehicle
axes. A formal `yaw` is a rotation about the optical axis, not a heading error. The
identity column shows how far projected LiDAR points move at 1° or 0.1 m, before any
correction.

| Formal label | Axis in the optical frame | Physical effect | Identity pixel P50 at 1° or 0.1 m |
| --- | --- | --- | ---: |
| `roll` | x (right) | tilt | 22.44 px <!-- bind: 22.44 = metrics#/runs/identity/roll:1/pixel_frame_p50_px/value --> |
| `pitch` | y (down) | pan | 23.95 px <!-- bind: 23.95 = metrics#/runs/identity/pitch:1/pixel_frame_p50_px/value --> |
| `yaw` | z (optical axis) | in-plane rotation | 7.74 px <!-- bind: 7.74 = metrics#/runs/identity/yaw:1/pixel_frame_p50_px/value --> |
| `x` | x (right) | lateral offset | 10.92 px <!-- bind: 10.92 = metrics#/runs/identity/x:0.1/pixel_frame_p50_px/value --> |
| `y` | y (down) | vertical offset | 10.98 px <!-- bind: 10.98 = metrics#/runs/identity/y:0.1/pixel_frame_p50_px/value --> |
| `z` | z (optical axis) | forward offset | 3.57 px <!-- bind: 3.57 = metrics#/runs/identity/z:0.1/pixel_frame_p50_px/value --> |

The learned models use seeds 17, 42 and 73, each keeping its own checkpoint and results.
Under one calibration corruption policy, the earliest checkpoint with the lowest
calibration loss is selected; the locked evaluation takes no part in model selection,
hyperparameters or the choice of figure conditions. The
[training configuration](../configs/correctors/convnextv2_tiny_v1.yaml) and the
[training contract](training-contract.md) record the complete rules.

## Comparison and statistical reading

- Every seed, identity, classical and every predeclared axis and level, including zero,
  is kept. Presenting only the conditions that improved is not allowed.
- Each operator first averages eligible frames within a scene with equal weight, then
  averages scenes with equal weight; global row validity cannot stand in for operator
  support.
- Paired intervals use the scene as the bootstrap unit. The mean of the three fixed seeds
  on common support is not a prediction ensemble and does not estimate the uncertainty
  of an arbitrary training seed.
- Recovery requires the geodesic rotation and the translation norm to be within their
  thresholds at the same time; improvements are in percentage points. Raw translation is
  in metres, the formal pose tables convert it to centimetres, and BEV error stays in
  metres.
- Pixel metrics are scene-balanced means of per-frame quantiles, not pooled quantiles of
  all pixels. GT distance bins, matched objects and the valid cutoff keep their original
  definitions, and missing values keep their reasons.

The complete estimands, thresholds, intervals and support definitions are those of the
[formal analysis contract](contracts/formal-analysis.md). The five results start from
the [evidence index](evidence/README.md); no other descriptive summary replaces the
formal statistics.

## Provenance and numerical repair

The original evaluator producer is `aeb3f28265c2ee3c0f7556417f9c8bf8a550acb7` and the
analysis repair producer is `33ff76485ed17194626c51333c941a33fe712902`. The provenance
of the three training seeds binds the original producer, the same pretrained source and
each seed's checkpoint; weight identities are in the [model card](model-card.en.md).

The first analysis was refused because the GT range of matched objects differed in the
last floating-point bits. Analysis policy V2 fixes a whole-group absolute span limit,
zero relative tolerance, the same bin and the same inclusive cutoff; it changed no raw
error, cohort, checkpoint or bootstrap setting, and no tolerance was chosen by
performance. After the repair two independent CPU analyses produced bit-identical
files. The [reproduction record](verification/analysis-reproduction.md) keeps the exact
versions, limits and SHAs.

## Failures and open questions

A corrector can change a calibration that was already correct, so zero-fault behaviour
is a required result. An improvement of the classical edge objective does not guarantee
a better pose or BEV error, and completing learned training does not guarantee an
effect. The results do not support a claim of a general, reliable correction model.

When the real box bottom is not on the ground plane, the flat-ground assumption leaves
a nonzero reconstruction error at zero fault. The full causes of weak recovery and
degradation are not established, and the absence of implementation problems is not
proven. Later diagnostics should keep failures, separate code fixes from new research
hypotheses, and never tune directly on the locked evaluation already seen or swap in a
better-looking seed.

Three reading notes found after the release, on identity recovery at ±0.25°, the
uninformative timing stress and BEV error beyond 10 m, are in
[v1.0.0 known issues](errata.md). A derived analysis of where correction helps is in the
[operating envelope](analysis/operating-envelope.md).

The formal report, the interactive demo and the v1.0.0 release passed acceptance; see
the [publication record](verification/publication-and-interchange.md). Engineering
acceptance does not change the research limits above. Any new analysis or experiment
needs its own source identity and cannot write back into these frozen results.
