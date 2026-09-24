# Model card: ConvNeXtV2-Tiny extrinsic corrector

[正體中文](model-card.md)

## Purpose and limits

This is the learned baseline of the nuScenes `CAM_FRONT`-`LIDAR_TOP` extrinsic fault
study. It estimates the inverse correction of an injected rigid transform. It does not
detect objects, predict braking or estimate a timing correction, and it is not a
calibration system validated for deployment on a vehicle. Read it together with the
[experiment card](experiment-card.en.md) and with every seed, zero-fault and baseline
result.

## Inputs and outputs

| Item | Fixed contract |
| --- | --- |
| Architecture | ConvNeXtV2-Tiny with a five-channel stem and six outputs |
| Input | 448 × 800; RGB, then assumed-calibration projected depth, then a validity mask |
| RGB | Divided by 255 and normalized with the ImageNet mean and std; a Pillow bilinear half-pixel resize also updates the intrinsics |
| Depth | Min-depth rasterization; `log1p(d)/log1p(80)` clipped to `[0,1]`; the mask marks missing depth explicitly |
| Output | roll/pitch/yaw in degrees and x/y/z in metres, in the CAM_FRONT optical frame |
| Target | The full inverse of the injected source-side rigid transform; the inverse translation is rotated, not negated component by component |

The five-channel stem extends the RGB weights: the two new channels take the mean RGB
kernel and the kernel is rescaled by weight energy. This does not mean that image,
depth and mask inputs share a distribution. Inference outputs stay in physical units
and must not be passed through the training loss normalization again. The complete
formulas and boundaries are in the [training contract](training-contract.md) and the
[coordinate contract](coordinate-contract.md). The output axes are camera optical-frame
axes: roll is tilt, pitch is pan and yaw is in-plane rotation (see the
[axis table](experiment-card.en.md#fault-axes)).

## Weights and training identity

The training provenance of the three formal seeds records the same initialization:

- timm profile: `convnextv2_tiny.fcmae_ft_in1k`.
- Input weight SHA-256: `fdfb4edfa5abb3b4ea29ab17e6fae5c117dc6cc32ecfb1d0726a156f40a4c1a5`.
- Training producer: `aeb3f28265c2ee3c0f7556417f9c8bf8a550acb7`.

The profile name and SHA are a provenance identity, not a download requirement that can
change over time. The adapter accepts only an explicitly named local weight file whose
hash matches; it never downloads anything. The repository's MIT licence does not replace
the separate licences of nuScenes or of the pretrained weights; the timm model card for
this profile declares CC BY-NC 4.0 (see [NOTICE](../NOTICE)). No checkpoint is
distributed here.

The fixed settings are seeds 17, 42 and 73, AdamW, a warmup/cosine schedule and a
normalized Huber loss; exact parameters are in the
[configuration](../configs/correctors/convnextv2_tiny_v1.yaml). Development scenes
train the parameters, calibration scenes select the checkpoint and evaluation stays
isolated. The three seeds are not merged into an ensemble.

When a formal checkpoint is loaded, the code verifies a successful run record, the
weight and provenance byte hashes, the configuration, cohort, producer, preprocessing
and calibration policy, and then uses strict model-state loading, eval mode and no-grad
inference. These engineering checks do not guarantee a good correction.

## Evidence and limits

The [five formal documents](evidence/README.md) keep the locked-cohort pose, pixel,
edge, BEV, recovery, timing and exclusion results; the
[reproduction record](verification/analysis-reproduction.md) explains the analysis
identity and the numerical pairing repair. Exact performance numbers should be shown
through the same artifact and claim binding, never replaced by the fact that training
completed. The README's key findings and the derived
[operating envelope](analysis/operating-envelope.md) summarize them: the mean of the
three seeds beats the classical baseline on pose, pixel error and recovery in every
condition but keeps a residual floor, so it improves on leaving the calibration alone
only for larger faults.

Input resolution, sensors, data distribution, projection and the ground assumption are
all fixed. Generalization across cameras, datasets or real-vehicle conditions is not
shown. Every predeclared fault condition and the zero-fault degradation must be kept;
the full causes of weak recovery are still open. Seed 42 does not correct pan
(formal `pitch`). Synthetic tests with random weights only check adapter behaviour and
are not evidence of the formal model's performance or of pretrained-weight eligibility.
