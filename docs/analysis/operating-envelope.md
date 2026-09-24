# Operating envelope of the v1 correctors (derived)

This is a derived analysis of the released v1.0.0 evidence. It re-estimates nothing:
it counts, orders and copies values that already exist in
[metrics.json](../evidence/nuscenes_calibration_v1/metrics.json),
[intervals.json](../evidence/nuscenes_calibration_v1/intervals.json) and
[recovery.json](../evidence/nuscenes_calibration_v1/recovery.json). It has its own
source identity, separate from the formal evidence, as the
[experiment card](../experiment-card.en.md) requires for any new analysis.

- Data: [operating-envelope.json](operating_envelope_v1/operating-envelope.json),
  labelled `evidence_type: derived`, with the SHA-256 of every source document it read
  and a digest of its own content.
- Figure: [operating-envelope.svg](operating_envelope_v1/operating-envelope.svg).
- Producer: [`bevcalib.analysis.operating_envelope`](../../src/bevcalib/analysis/operating_envelope.py).
  It refuses source documents whose digest differs from the released ones, and a
  contract test rebuilds both files from the evidence and compares their bytes.

Every number on this page is bound to the derived document or to a verified claim of
the formal evidence, and the test suite checks each one.

## Definitions

A paired comparison `before->after` has an improvement whose sign follows the source
documents: for errors it is before minus after, for the edge score and recovery it is
after minus before. A positive improvement therefore always means that the `after`
method did better.

For each of the 60 extrinsic conditions, the paired 95% scene-bootstrap interval of that improvement is classified as above zero (`after` better), below zero (`before` better), containing zero (inconclusive) or unavailable. <!-- bind: 60 = envelope#/interval_counts/identity->classical/pixel_frame_p50_px/total -->
The intervals are pointwise; counting them describes the grid and is not a
multiple-comparison test.

**Break-even** on an axis is the smallest grid magnitude at which the `after` method
has a lower pixel P50 at both signs and at every larger magnitude on the grid. Recovery
against identity is not counted, because of the
[±0.25° boundary artifact](../errata.md#1-identity-recovery-at-025-is-a-floating-point-boundary-artifact).

## Learned mean versus classical

The fixed-three-seed learned mean is better than classical in 60 of 60 conditions on rotation error, translation error, pixel P50, pixel P90 and recovery. <!-- bind: 60 = envelope#/interval_counts/classical->learned-fixed-three-seed-mean/rotation_geodesic_deg/after_better ; 60 = envelope#/interval_counts/classical->learned-fixed-three-seed-mean/translation_norm_cm/after_better ; 60 = envelope#/interval_counts/classical->learned-fixed-three-seed-mean/pixel_frame_p50_px/after_better ; 60 = envelope#/interval_counts/classical->learned-fixed-three-seed-mean/pixel_frame_p90_px/after_better ; 60 = envelope#/interval_counts/classical->learned-fixed-three-seed-mean/recovery_rate_pct/after_better -->
Classical is better in 60 of 60 conditions on the edge-alignment score, the objective it maximizes. <!-- bind: 60 = envelope#/interval_counts/classical->learned-fixed-three-seed-mean/edge_score_px/before_better -->

A better edge score does not come with a better pose: this is the proxy mismatch
already described in the [corrector diagnostics](../verification/corrector-diagnostics.md).

## Correcting versus leaving the calibration alone

On pixel P50 the learned mean beats identity in 14 of 60 conditions, loses in 40 and is inconclusive in 6. <!-- bind: 14 = envelope#/interval_counts/identity->learned-fixed-three-seed-mean/pixel_frame_p50_px/after_better ; 60 = envelope#/interval_counts/identity->learned-fixed-three-seed-mean/pixel_frame_p50_px/total ; 40 = envelope#/interval_counts/identity->learned-fixed-three-seed-mean/pixel_frame_p50_px/before_better ; 6 = envelope#/interval_counts/identity->learned-fixed-three-seed-mean/pixel_frame_p50_px/inconclusive -->
Classical beats identity in 4 of 60 conditions and loses in 53. <!-- bind: 4 = envelope#/interval_counts/identity->classical/pixel_frame_p50_px/after_better ; 60 = envelope#/interval_counts/identity->classical/pixel_frame_p50_px/total ; 53 = envelope#/interval_counts/identity->classical/pixel_frame_p50_px/before_better -->

Break-even of the learned mean against identity, per axis:

| Formal axis | Physical effect | Break-even magnitude |
| --- | --- | --- |
| `roll` | tilt | 1° <!-- bind: 1 = envelope#/break_even/identity->learned-fixed-three-seed-mean/roll/magnitude --> |
| `pitch` | pan | 1° <!-- bind: 1 = envelope#/break_even/identity->learned-fixed-three-seed-mean/pitch/magnitude --> |
| `yaw` | in-plane rotation | 2° <!-- bind: 2 = envelope#/break_even/identity->learned-fixed-three-seed-mean/yaw/magnitude --> |
| `x` | lateral offset | 0.2 m <!-- bind: 0.2 = envelope#/break_even/identity->learned-fixed-three-seed-mean/x/magnitude --> |
| `y` | vertical offset | 0.2 m <!-- bind: 0.2 = envelope#/break_even/identity->learned-fixed-three-seed-mean/y/magnitude --> |
| `z` | forward offset | none on the grid |

Classical has a break-even only for tilt, at 2°; its other axes have none on the grid. <!-- bind: 2 = envelope#/break_even/identity->classical/roll/magnitude -->

## The residual floor

| Method | Residual rotation, min–max (°) | Residual translation, min–max (cm) |
| --- | --- | --- |
| `classical` | 0.95–2.32 <!-- bind: 0.95 = envelope#/residual/classical/rotation_geodesic_deg/min ; 2.32 = envelope#/residual/classical/rotation_geodesic_deg/max --> | 12.30–23.25 <!-- bind: 12.30 = envelope#/residual/classical/translation_norm_cm/min ; 23.25 = envelope#/residual/classical/translation_norm_cm/max --> |
| `learned-17` | 0.63–0.83 <!-- bind: 0.63 = envelope#/residual/learned-17/rotation_geodesic_deg/min ; 0.83 = envelope#/residual/learned-17/rotation_geodesic_deg/max --> | 2.60–3.26 <!-- bind: 2.60 = envelope#/residual/learned-17/translation_norm_cm/min ; 3.26 = envelope#/residual/learned-17/translation_norm_cm/max --> |
| `learned-42` | 0.44–2.12 <!-- bind: 0.44 = envelope#/residual/learned-42/rotation_geodesic_deg/min ; 2.12 = envelope#/residual/learned-42/rotation_geodesic_deg/max --> | 2.45–4.05 <!-- bind: 2.45 = envelope#/residual/learned-42/translation_norm_cm/min ; 4.05 = envelope#/residual/learned-42/translation_norm_cm/max --> |
| `learned-73` | 0.62–0.74 <!-- bind: 0.62 = envelope#/residual/learned-73/rotation_geodesic_deg/min ; 0.74 = envelope#/residual/learned-73/rotation_geodesic_deg/max --> | 2.40–3.08 <!-- bind: 2.40 = envelope#/residual/learned-73/translation_norm_cm/min ; 3.08 = envelope#/residual/learned-73/translation_norm_cm/max --> |

The ranges are over all 60 conditions, from zero fault to the largest rotation and translation faults. <!-- bind: 60 = envelope#/interval_counts/identity->classical/pixel_frame_p50_px/total -->
Seeds 17 and 73 leave nearly the same error whatever was injected. A fault smaller
than that floor ends with a larger error than it started with, which is consistent
with both the weak recovery and the zero-fault degradation in the formal results.
At zero fault the learned mean moves a correct calibration by 0.60° and 2.58 cm. <!-- bind: 0.60 = intervals#/comparisons/identity->learned-fixed-three-seed-mean/pitch:0/rotation_geodesic_deg/after ; 2.58 = intervals#/comparisons/identity->learned-fixed-three-seed-mean/pitch:0/translation_norm_cm/after -->

## Seed 42 and pan

Formal `pitch` is pan. The table gives each method's residual pan error
(`rotation_abs_pitch_deg`, in degrees) at every pan fault. Identity keeps the injected
fault, so its column equals the fault size.

<!-- bind-table: envelope#/pan_residual/{column}/{row} -->
| Pan fault | `identity` | `classical` | `learned-17` | `learned-42` | `learned-73` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pitch:-2` | 2.00 | 1.73 | 0.50 | 1.99 | 0.40 |
| `pitch:-1` | 1.00 | 0.63 | 0.27 | 0.98 | 0.33 |
| `pitch:-0.5` | 0.50 | 0.57 | 0.25 | 0.48 | 0.33 |
| `pitch:-0.25` | 0.25 | 0.35 | 0.25 | 0.23 | 0.32 |
| `pitch:-0.1` | 0.10 | 0.20 | 0.27 | 0.08 | 0.30 |
| `pitch:0` | 0.00 | 0.14 | 0.27 | 0.04 | 0.30 |
| `pitch:0.1` | 0.10 | 0.22 | 0.30 | 0.12 | 0.29 |
| `pitch:0.25` | 0.25 | 0.36 | 0.32 | 0.28 | 0.28 |
| `pitch:0.5` | 0.50 | 0.53 | 0.37 | 0.53 | 0.26 |
| `pitch:1` | 1.00 | 0.24 | 0.34 | 1.03 | 0.23 |
| `pitch:2` | 2.00 | 1.11 | 0.60 | 2.04 | 0.42 |

Seed 42 leaves an injected pan almost untouched, although its pixel error under tilt
faults is lower than that of the other seeds. Its higher zero-fault recovery is
consistent with a seed that rarely moves the pan angle:
12.07% for seed 42 against 2.89% for seed 17 and 2.39% for seed 73. <!-- bind: 12.07 = recovery#/runs/learned-42/pitch:0/value ; 2.89 = recovery#/runs/learned-17/pitch:0/value ; 2.39 = recovery#/runs/learned-73/pitch:0/value -->
Seeds are fixed by the protocol and are never selected on these results.

## What this does not show

These counts describe the 30 locked scenes and the three fixed seeds. <!-- bind: 30 = metrics#/runs/identity/roll:1/pixel_frame_p50_px/support/scenes -->
They are not a guarantee for other scenes, cameras or training runs, and pointwise
intervals do not control the family-wise error across the grid. Break-even levels are
grid levels, so the true crossover lies between the break-even level and the next
smaller one. The implication that online correction needs a miscalibration detector or
an abstain gate is an argument from these results, not something this study tested.
