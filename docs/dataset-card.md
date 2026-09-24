# nuScenes calibration cohort

This project uses the original nuScenes trainval metadata and paired CAM_FRONT and
LIDAR_TOP keyframes. Sensor payloads, scene and sensor identifiers, raw manifests and
model weights stay outside Git. Obtain the source dataset through its original
distribution and follow its terms; this repository does not redistribute it.

The formal roles are development, calibration and evaluation. Their construction is
defined by the [cohort contract](cohort-contract.md) and the versioned
[protocol](../configs/protocols/nuscenes_calibration_v1.yaml). Whole logs are disjoint
between roles. Calibration uses distinct logs and official training scenes;
evaluation uses official validation scenes. Evaluation never selects thresholds,
hyperparameters or checkpoints. The role manifests retain separate sample, camera
and LiDAR timestamps and bind the original protocol identity.

Only front-camera images and top-LiDAR keyframes are required for this study.
Unselected camera/radar channels and absent sweeps are not silently synthesized.
Timing stress fixes the camera exposure and selects an available LiDAR packet under
the declared timestamp tolerance. Coverage depends on actual on-disk packets;
missing or out-of-tolerance selections remain invalid with explicit reasons. The
staged keyframe archives hold no intermediate sweeps, so the v1 timing stress
carries no information; see the [v1.0.0 known issues](errata.md).

The existing mini installation is preserved for independent official-devkit geometry
parity. It is not a substitute for the trainval installation or a source of formal
performance claims. Ground-contact reconstruction also has a flat-plane model
residual at zero extrinsic fault, as explained in the
[coordinate contract](coordinate-contract.md).

Dataset readiness evidence and reproducible commands are recorded in
[nuScenes preflight](verification/nuscenes-preflight.md). Readiness establishes input
integrity and cohort separation; it does not establish calibration recovery,
downstream accuracy or timing robustness.
