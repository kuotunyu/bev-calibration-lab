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
