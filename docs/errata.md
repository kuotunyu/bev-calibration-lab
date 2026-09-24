# v1.0.0 known issues and reading notes

[正體中文](errata.zh-TW.md)

These notes were written after the v1.0.0 release, by re-reading the frozen evidence.
They change no released number, figure, claim, configuration or tag. They say how to
read three parts of the evidence that the release documents did not explain. Each
number below is bound to the released evidence and checked by the test suite.

## 1. Identity recovery at ±0.25° is a floating-point boundary artifact

Recovery counts a frame as recovered when its geodesic rotation error is at most
0.25° and its translation error at most 5 cm, inclusive of the edge
([`recovered()`](../src/bevcalib/metrics/calibration.py)). The fault grid also
contains ±0.25° on every rotation axis ([fault matrix](../configs/perturbations/formal_v1.yaml)).
Identity leaves the injected fault in place, so its error should equal the threshold
exactly. Floating-point rounding in the rotation conversions makes the recorded error 0.2500000000006049°, just above the threshold, so identity scores 0% recovery at these conditions instead of 100%. <!-- bind: 0.2500000000006049 = metrics#/runs/identity/roll:0.25/rotation_geodesic_deg/value ; 0 = recovery#/runs/identity/roll:0.25/value -->
The same rounding at ±0.1° gives 0.10000000000072487°, which is well inside the threshold, and 100% recovery. <!-- bind: 0.10000000000072487 = metrics#/runs/identity/roll:0.1/rotation_geodesic_deg/value ; 100 = recovery#/runs/identity/roll:0.1/value -->

**Affected.** Six conditions: `roll:±0.25`, `pitch:±0.25` and `yaw:±0.25`. Two
fields: identity's recovery rate there (`/runs/identity/<condition>` in
[recovery.json](evidence/nuscenes_calibration_v1/recovery.json) and
`recovery_rate_pct` in metrics.json), and the identity→* recovery comparisons at the
same conditions (`/comparisons/identity->*/<condition>` in recovery.json).
In 24 of these 30 comparisons the 95% interval lies above zero, which reads as a significant improvement over identity. It is not one. <!-- bind: 24 = envelope#/recovery_boundary/identity_comparisons_above_zero ; 30 = envelope#/recovery_boundary/identity_comparisons -->
For example, identity → learned-42 at `roll:0.25` goes from 0% to 11.25% with an interval of 5.63 to 17.40 percentage points. <!-- bind: 0 = recovery#/comparisons/identity->learned-42/roll:0.25/before ; 11.25 = recovery#/comparisons/identity->learned-42/roll:0.25/after ; 5.63 = recovery#/comparisons/identity->learned-42/roll:0.25/interval/low ; 17.40 = recovery#/comparisons/identity->learned-42/roll:0.25/interval/high -->
The [recovery figure](https://kuotunyu.github.io/bev-calibration-lab/figures/recovery-by-fault-level.svg)
shows the same artifact as identity dropping from 100% at ±0.1° to 0% at ±0.25°.

**Not affected.** Pose, pixel, edge and BEV estimands; every comparison that does not
involve identity; identity at every other condition.

**Correct reading.** At ±0.25° identity is inside the declared tolerance, so its
recovery should be 100% and every identity→* recovery difference at those conditions
would be negative: a corrector can only move frames out of tolerance. Do not cite
identity→* recovery at ±0.25° as a result.

**Remedy for a future protocol.** Compare with an explicit numerical tolerance (for
example `error <= threshold + 1e-9`), or choose thresholds that are not grid levels.
The v1 evidence stays as released.

## 2. The v1 timing stress carries no information

The timing stress keeps the camera exposure fixed and selects the LIDAR_TOP packet
nearest to the camera time plus the requested offset, valid within 25 ms. The locked
cohort was staged from the trainval keyframe archives, which contain only keyframe
sweeps ([nuScenes preflight](verification/nuscenes-preflight.md)); other sweep
payloads were on disk for a single scene. The nearest packet is therefore almost always
the keyframe sweep, about 36 ms from the camera exposure, whatever offset is requested.

- The realised offset median is 35.81 to 36.09 ms at all seven requested offsets. <!-- bind: 35.81 = timing#/offsets/-200/realized_offset_ms/median ; 36.09 = timing#/offsets/100/realized_offset_ms/median -->
- At +50 ms the keyframe sweep is within tolerance (median absolute error 14.02 ms), so 1207 of 1207 frames are valid. <!-- bind: 14.02 = timing#/offsets/50/absolute_error_ms/median ; 1207 = timing#/offsets/50/valid ; 1207 = timing#/offsets/50/total -->
- At every other offset only 40 frames from 1 scene are valid (39 at +200 ms). <!-- bind: 40 = timing#/offsets/0/valid ; 1 = metrics#/runs/identity/time:0/edge_score_px/support/scenes ; 39 = timing#/offsets/200/valid -->
- At +50 ms the selected sweep is the one the extrinsic study uses, so every measured estimand equals the zero-fault identity row: the edge score is -4.565473134856439 px and the 0-10 m BEV error 1.182511468842712 m in both. <!-- bind: -4.565473134856439 = metrics#/runs/identity/time:50/edge_score_px/value ; -4.565473134856439 = metrics#/runs/identity/pitch:0/edge_score_px/value ; 1.182511468842712 = metrics#/runs/identity/time:50/bev_frame_mean_m~10-10/value ; 1.182511468842712 = metrics#/runs/identity/pitch:0/bev_frame_mean_m~10-10/value -->

Pixel and BEV errors compare the true and the assumed calibration on the same points
and boxes. A timing fault changes neither calibration, so these errors cannot respond
to the sweep choice (the pixel error is numerically zero at every offset). Only the
edge proxy, which scores LiDAR edges from the selected sweep against image edges,
depends on the sweep.

**Correct reading.** The v1 timing stress shows neither robustness nor sensitivity to
LiDAR-camera time offsets. A timing study needs non-keyframe sweeps staged for every
scene.

## 3. Far-range BEV means are dominated by grazing rays

The BEV estimand reconstructs each box's ground contact by intersecting the camera ray
through the box bottom with a flat ground plane at the ego origin's height
([coordinate contract](coordinate-contract.md#oracle-controlled-ipm-baseline)). When
the ray meets the plane at a shallow angle, a small angle or height difference moves
the intersection far along the ground: for a camera at height h and a contact at
distance d, an angular error δ moves it by roughly d²·δ/h. A few near-horizon rays
therefore dominate the mean of the far range bins.

- Identity at zero fault, where the plane model is the only error, already has mean errors of 1.18, 8.16, 54.50 and 781.05 m in the 0-10, 10-20, 20-40 and 40-80 m bins. <!-- bind: 1.18 = metrics#/runs/identity/pitch:0/bev_frame_mean_m~10-10/value ; 8.16 = metrics#/runs/identity/pitch:0/bev_frame_mean_m~110-20/value ; 54.50 = metrics#/runs/identity/pitch:0/bev_frame_mean_m~120-40/value ; 781.05 = metrics#/runs/identity/pitch:0/bev_frame_mean_m~140-80/value -->
- A -2° tilt fault (formal `roll`) gives 1.19, 3.68, 11.99 and 26.42 m in the same bins: the fault lowers the far-bin means. <!-- bind: 1.19 = metrics#/runs/identity/roll:-2/bev_frame_mean_m~10-10/value ; 3.68 = metrics#/runs/identity/roll:-2/bev_frame_mean_m~110-20/value ; 11.99 = metrics#/runs/identity/roll:-2/bev_frame_mean_m~120-40/value ; 26.42 = metrics#/runs/identity/roll:-2/bev_frame_mean_m~140-80/value -->
- The largest 40-80 m mean in the evidence is 6813.33 m (learned-73 at `yaw:-0.25`). <!-- bind: 6813.33 = metrics#/runs/learned-73/yaw:-0.25/bev_frame_mean_m~140-80/value -->

**Correct reading.** Interpret BEV error only in the 0-10 m and 10-20 m bins. The
far-bin means reflect how well the ray-plane intersection is conditioned, not
calibration sensitivity. The published
[BEV figure](https://kuotunyu.github.io/bev-calibration-lab/figures/bev-error-by-range.svg)
shares one vertical scale across bins, so its scale is set by the far bins.
