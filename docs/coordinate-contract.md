# Coordinate contract

Every silent failure in a LiDAR-camera study is a frame or an ordering mistake. They
are dangerous because the result is never an exception: points still land in the
image, errors are still a few pixels, and the numbers look like a calibration
problem instead of a bug. This document fixes the conventions, and
`src/bevcalib/geometry/` enforces them.

## Naming

Every transform is named `T_target_source` and maps a point expressed in `source`
into `target`.

```
p_target = T_target_source · p_source
```

Read the subscripts right to left: the frame you have is on the right, the frame
you want is on the left. In code the same rule is spelled out in the parameter
names, `compose(target_from_mid, mid_from_source)`.

## Algebra

| Rule | Meaning |
| --- | --- |
| `compose(T_a_b, T_b_c) = T_a_c` | The right-hand transform is applied first |
| `inverse(T_a_b) = T_b_a` | Rotation transposes; translation becomes `-Rᵀt` |
| `compose(T, inverse(T)) = I` | Within `1e-9`, which is the tolerance everything here uses |
| `compose(compose(A, B), C) = compose(A, compose(B, C))` | Association is free, so chains may be grouped for readability |

A transform is stored as a unit quaternion `(w, x, y, z)` plus a translation in
metres. The quaternion is normalised when the value is constructed, and its sign
is canonicalised so that the first nonzero component is positive. Both matter:
`q` and `-q` are the same rotation, so without a fixed sign two records of one
rotation compare unequal; and a quaternion that is not unit-norm would scale
every point it touched.

## Rows, columns, and where the boundary is

Public point arrays are `[N, 3]` **row vectors**, `float64`. A rotation is
therefore applied on the right:

```python
points_target = points_source @ rotation.T + translation
```

Column-vector conventions are common in the literature and in some libraries, and
they are allowed to exist only inside an adapter. No public signature in this
package accepts or returns `[3, N]`. The reason is mechanical: a `[3, N]` array
broadcasts against a 3×3 rotation without raising anything and returns a
plausible, wrong answer. `transform_points` checks the shape instead of inferring
it, and `N = 0` is a legitimate input, because a sweep can project entirely out
of frame.

## The chain, and why there are two ego frames

```
lidar_sensor → lidar_ego → global → camera_ego → camera_sensor
```

The two ego frames are not redundant. The LiDAR sweep and the camera exposure
happen at different instants, and the vehicle moves in between, so they are two
different poses of the same rigid body:

- `T_global_lidar_ego` is the ego pose at **t_lidar**, the sweep timestamp.
- `T_camera_ego_global` is the inverse of the ego pose at **t_camera**, the
  exposure timestamp.

Collapsing them into one ego pose is the single most common way to get a result
that looks almost right. At 50 km/h the vehicle covers about 0.7 m in 50 ms, which
is the same order as the translation faults this study injects deliberately, so
the mistake would be indistinguishable from the effect being measured.

Timing offsets are therefore requested in milliseconds and recorded with the
realised offset actually achieved, never assumed to be zero.

3D boxes are natively in `global`. LiDAR features keep `x, y, z, intensity, ring`.

## Worked example

A point 2 m ahead of the LiDAR, carried into `global`. The LiDAR is mounted 1 m
ahead of the ego origin and yawed 90°; the ego is at (100, 200, 0) with no
rotation.

```
p_lidar_sensor           = (2, 0, 0)
T_lidar_ego_lidar_sensor = yaw 90° about z, t = (1, 0, 0)
T_global_lidar_ego       = identity rotation,  t = (100, 200, 0)
```

Step by step, applying the inner transform first:

```
p_lidar_ego = R_z(90°)·(2, 0, 0) + (1, 0, 0)
            = (0, 2, 0)          + (1, 0, 0)   = (1, 2, 0)

p_global    = I·(1, 2, 0)        + (100, 200, 0)  = (101, 202, 0)
```

Composing first must give the same answer:

```
T_global_lidar_sensor = compose(T_global_lidar_ego, T_lidar_ego_lidar_sensor)
  rotation    = I · R_z(90°)                    = R_z(90°)
  translation = I · (1, 0, 0) + (100, 200, 0)   = (101, 200, 0)

p_global = R_z(90°)·(2, 0, 0) + (101, 200, 0)
         = (0, 2, 0)          + (101, 200, 0)   = (101, 202, 0)   ✓
```

Swapping the two arguments does not raise. It returns
`translation = R_z(90°)·(100, 200, 0) + (1, 0, 0) = (-199, 100, 0)` and sends the
same point to `(-199, 102, 0)`, three hundred metres from the truth, still as
three finite numbers. That is the failure this contract exists to make
impossible, and `tests/unit/geometry/test_se3.py` asserts both the correct result
and that the reversed order differs.

## Where a calibration fault goes

A fault is a transform, and there are two places to put it. Both produce a valid
rigid transform, so the choice cannot be left to whoever writes the next caller.
The convention is:

```
assumed = true ∘ fault
```

The fault is composed on the **source** side, in the sensor's own frame. That is
what a miscalibrated extrinsic physically is: the sensor is believed to sit
slightly rotated or shifted from where it really does, measured along its own
axes. So a 0.2 m fault on the sensor x axis moves the assumed sensor 0.2 m along
the direction the sensor points, not 0.2 m east.

Composing on the target side is equally valid arithmetic and answers a different
question, which is why `tests/unit/perturbations/test_apply.py` asserts both that
the source-side result is produced and that the target-side result differs.

The Euler convention inside a fault is likewise fixed:

```
R = Rz(yaw) · Ry(pitch) · Rx(roll)
```

Roll about x first, then pitch about y, then yaw about z, all about fixed axes.
With only one angle nonzero every convention agrees, so the test that pins this
uses all three at once.

Two things a fault never does. It never touches an observation: no point and no
pixel changes, because an error mixing a sensing change with a calibration change
could not be attributed to either, and a test hashes the raw bytes before and
after to keep it that way. And a timing fault never moves the geometry: it
changes which LiDAR sweep is paired with the fixed camera, and it is refused as a
learned 6DoF target because no pose expresses it.

## Enforcement

`FramedTransform` carries `target` and `source` frame names alongside the
transform, and `compose_framed` refuses to compose when the left transform does
not start in the frame the right one lands in. The frame set is closed: adding a
frame is a protocol change, and `FRAME_NAMES` is asserted by a test rather than
assumed.

## Fixed camera timing selection

The keyframe CAM_FRONT image, token and exposure timestamp stay fixed. Select
actual available LIDAR_TOP payloads in that scene and log nearest to
`camera_timestamp_us + requested_offset_ms * 1000`. Metadata-only sweeps are not
available. Use both selected sensors' actual timestamped ego poses; never relabel
a timestamp or substitute the nominal LiDAR sweep for missing timing data.
Record selected token/timestamp, realized camera-relative offset, absolute error
and reason. The 25 ms limit is inclusive. No candidate means null selection and
null timing measurements; an out-of-tolerance candidate retains its evidence but
is invalid. Neither case contributes a zero measurement to valid denominators.

## Five-channel pretrained initialization

Start from the ImageNet RGB stem and append two mean-RGB kernels for normalized
projected depth and its validity mask. For each output channel, multiply the
entire five-channel kernel by
`sqrt(sum(W_rgb**2) / sum(W_five**2))`; a zero kernel remains zero. Preserve source
bias, dtype and device. Under independent, equal unit-variance input channels this
preserves convolution output variance. Real RGB, depth and mask distributions
are not assumed to match. Training targets remain inverse calibration faults in
degrees and metres, and timing is a separate stress condition.


## Native image alignment and availability snapshots

The camera stays at native resolution for evaluation, with its native intrinsic matrix
and pixel units. `bev-image-edges/v1:pillow-RGB-to-L:float64-sobel-axes01-reflect:hypot:positive-quantile0.90-linear:positive-and-ge`
uses Pillow RGB-to-L, float64 SciPy Sobel along axes 0 and 1 with reflect boundaries,
and the hypotenuse magnitude. The threshold is NumPy's linear 90th percentile of
strictly positive magnitudes; retain positive magnitudes at least that threshold.
This adaptive per-image threshold is a fixed algorithm, never fitted on evaluation
results. A constant image has no measurable edges. Evaluation provenance records
the policy and each actual threshold. The distance field is computed once per fixed
image and reused; floor indexing, negative mean and 90% trimming remain unchanged.
A classical candidate with no in-frame LiDAR edges receives only the search penalty
`-hypot(width, height)`. The final result records missing projection as invalid/null,
never this penalty as a measured score. An empty image edge mask invalidates before search.

A resolved installation captures original JSON table byte hashes and on-disk payload
availability once. Timing candidates are indexed by scene/log/channel and timestamp;
token ordering breaks ties. Refresh by resolving a new installation, not by changing
availability mid-run. Preflight rejects missing paired CAM_FRONT/LIDAR_TOP keyframe
payloads; optional missing sweeps remain explicit coverage information. Its per-offset
selection rows preserve fixed camera and selected LiDAR tokens/timestamps, actual gap,
approximation error, validity and reason. Actual payload reads still fail if a file
vanishes after resolution; snapshot selection never substitutes a different frame.

## Oracle-controlled IPM baseline

Ground-contact XY is reconstructed on the horizontal plane through ego origin and
compared with the actual GT box bottom XY. A GT bottom off that plane creates an
absolute residual even when true and assumed calibration are identical. For a camera
at height 1 m, contact at height 0.2 m and horizontal distance 10 m, the ray intersects
the zero-height plane at 12.5 m: a 2.5 m baseline without any calibration fault.
Report this plane-model baseline separately from changes under calibration faults.
GT range is the existing operator's 3D distance from the true camera origin to the
box bottom. All five declared range bins remain visible, including 80+: exactly
80 m can be valid while values greater than 80 m are excluded by the fixed cutoff.
Empty valid denominators have null means and explicit invalid counts/reasons.

### Cross-run GT range pairing

Analysis policy V2 pairs exact scene, sample and box identities while allowing a
GT range span of at most `1e-9 m`, with zero relative tolerance. This fixed absolute
budget follows the geometry contract; it is not fitted to measured recovery or
evaluation performance. Float64 composition, quaternion conversion and inverse
round trips can produce different last bits on different arithmetic runtimes.
Bitwise equality is therefore not a portable geometric identity check.

The entire group's maximum minus minimum must satisfy the budget. Every original
range must also have the same half-open range bin and the same `range <= 80 m`
classification. In particular, `80` and the next representable number above it
are rejected even though both are in the `80+` bin. Values straddling any bin edge
are also rejected, regardless of how small their difference is. Material range
drift still fails the comparison instead of becoming an exclusion.

The analyzer never rounds, replaces or averages the recorded ranges. Measured
errors, raw artifact hashes, missing/invalid-object exclusions, frame/scene
weighting and the fixed bootstrap are unchanged. The revised policy identifier
and exact numerical rule are included in every formal analysis document; raw
evaluation producer identities retain their original source commits.


## Complete evaluation inventories

The authoritative inventory is method-specific: identity produces all 67 declared
single-axis conditions per sample, including seven timing conditions. Classical
and learned runs produce only the 60 rotation/translation conditions, retaining
each extrinsic axis's zero reference. Timing is measured only by the identity
stress run; it is never a learned target, correction-method completion row or
correction/recovery denominator. Complete-run validation rejects missing, duplicate
or unsupported rows, including a time-zero row inserted into a correction run.

The V2 evaluation identity binds native decoded RGB byte hashes, image dimensions,
the actual per-image edge thresholds and full policy, plus original metadata table
hashes. A threshold-only first pass establishes identity without retaining full
images/fields for the entire cohort. The measurement pass checks RGB identity,
computes one distance field per camera and caches LiDAR edge extraction per selected
packet across conditions. Missing operators remain null with explicit reasons.
Scene documents are written atomically under distinct method/run roots; completion
is written only after the method-specific inventory and byte hashes revalidate.


## Runtime protocol support and public summaries

Protocol resolution parses the exact referenced perturbation bytes before hashing
them and refuses values different from the compiled V1 schedule, timing tolerance,
training bounds or recovery thresholds. This check is shared by formal freeze,
training and evaluation. Comments or YAML key ordering can change the byte identity
while leaving the supported mathematical values unchanged.

Private evaluation roots carry one verified V2 manifest sidecar. Reusing a root for
a different cohort is refused. Descriptive aggregation verifies completed run hashes,
method inventories, seeds/checkpoints, evidence type and measurement input identity.
It preserves signed component statistics, per-operator measured denominators, complete
row validity/reasons and all GT range bins with null summaries for empty measurements.
The three learned seeds remain separate. Identity alone contributes timing stress;
its timing conditions have no pose-recovery summary.

Public summary export contains aggregate numbers and provenance hashes, excluding
scene, log, sample and sensor tokens. A report consumes safe aggregate JSON and its
claims registry without requiring the private manifest, row files, data or checkpoints.
Formal paired scene bootstrap and the final five-document study artifact set remain
Task J; these descriptive summaries do not substitute for those statistical results.

## Dataset readiness verification

The [trainval preflight record](verification/nuscenes-preflight.md) binds the installed
metadata and frozen cohort identities. The official-devkit mini checks compare
camera-frame points, optical depth, projection masks, valid pixel coordinates and
global-box transformations using the two sensor timestamps. They require nonempty
comparisons, absolute tolerance `1e-6` and zero relative tolerance.

The CLI failure rehearsal swaps the calibrated-sensor and ego-pose lookup namespaces
and requires a diagnostic naming the unresolved input. It verifies this concrete
broken-chain refusal; it does not claim that arbitrary numerically valid inverted
transforms can be recognized without independent reference evidence.

Timing availability is measured from the actual installation snapshot. Retained mini
sweeps can provide coverage for their own scenes, while trainval keyframe archives
do not supply the remaining scenes' sweep payloads. Each offset keeps its own actual
nearest-candidate approximation errors, valid/invalid denominator and reasons. A
successful keyframe preflight therefore does not imply complete timing coverage.
