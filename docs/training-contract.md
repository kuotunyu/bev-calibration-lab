# Learned training and checkpoint contract

The production adapter uses ConvNeXtV2-Tiny with five input channels and six
outputs in roll/pitch/yaw degrees then x/y/z metres. Targets are the full inverse
of the source-side injected rigid transform, including the rotated inverse
translation. Timing offsets are stress conditions, never learned training targets.

The declared input is 448 by 800 pixels. Pillow bilinear resize uses pixel-centre
geometry: focal lengths scale by sx/sy and principal points become
`(c + 0.5) * scale - 0.5`. RGB is divided by 255 then normalized with means
`[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`.
Assumed-calibration projected depth is min-depth rasterized with a validity mask;
depth is `log1p(d)/log1p(80)` clipped to [0,1]. Preprocessing identity is
`bev-input/v1:pil-bilinear-half-pixel:imagenet-rgb:log1p80-depth:valid-mask`.

The RGB stem is widened with two mean-RGB kernels and each output's five-channel
weights are scaled by `sqrt(E_RGB / E_five)`. A zero kernel stays zero. Source
bias, dtype and device are preserved. This conserves weight energy under independent
unit-variance inputs; it does not claim RGB, depth and mask have equal distributions.

Production initialization requires an explicit local weight file, its SHA-256 and
a named compatible timm pretrained profile. The adapter never downloads weights.
The profile must be ConvNeXtV2-Tiny with ImageNet RGB mean/std and 1000 source classes;
the classifier is replaced by six outputs. For example the locally installed profile
metadata for `convnextv2_tiny.fcmae_ft_in22k_in1k` meets those input requirements.
Its default crop/resize is not silently inherited: this study's geometry above is
bound explicitly. Model weight licenses remain separate from this repository's MIT
code license. Local random-weight architecture tests are synthetic adapter evidence,
not proof of pretrained eligibility or accuracy. J records the actual chosen weight
identity before any formal training.

Verified V2 development/calibration cohorts must be trainval, official-train,
100/20 scenes, with twenty distinct calibration logs and disjoint log/sample/sensor
identities across roles. Role/protocol/config/cohort/provenance checks precede data,
optimizer and output creation. Explicit synthetic service fixtures may use smaller
verified cohorts and an injected real Torch architecture; default production paths
retain the formal constraints. Synthetic checkpoints cannot pass formal loading.

Development corruption is keyed by sample, epoch and training seed. The calibration
objective uses unsigned big-endian first four bytes of
`SHA256(UTF-8("calibration|" + protocol_hash))` and epoch zero for every sample.
Policy `bev-calibration-objective/v1:sha256(calibration|protocol):epoch0` is therefore
fixed across epochs and the three training seeds. It is bound with preprocessing,
raw config, both verified manifests and producer identity in checkpoint metadata
and the training provenance artifact. Evaluation never selects the checkpoint.

Training uses the configured real AdamW, warmup/cosine schedule and normalized Huber
loss. The selected checkpoint carries its model state and metadata digest. Loading
checks the sibling successful run record, checkpoint/provenance byte hashes, all
identity bindings, current preprocessing/calibration policies, source profile and
strict model-state compatibility. Formal evaluation requires a verified complete
30-scene official-val cohort, matching protocol/dataset, with no training log leakage.
