# Native commands and portable reports

Commands use the locked Python environment. Install the existing `train` and `report`
extras when preparing a new development environment; runtime adapters import Torch,
timm and Jinja only at their service boundaries. No command downloads dataset files
or pretrained weights. Formal execution remains subject to the approved data/workflow
checkpoints; synthetic adapter tests are not observed accuracy evidence.

```bash
uv sync --frozen --all-groups --extra train --extra report
uv run bev-calib data preflight --dataroot "$NUSCENES_ROOT" --version v1.0-trainval --output "$PRIVATE_PREFLIGHT"
uv run bev-calib cohort freeze --dataroot "$NUSCENES_ROOT" --version v1.0-trainval --protocol configs/protocols/nuscenes_calibration_v1.yaml --output-dir "$PRIVATE_MANIFESTS"
uv run bev-calib evaluate --protocol configs/protocols/nuscenes_calibration_v1.yaml --manifest "$PRIVATE_MANIFESTS/evaluation.json" --method identity --output-dir "$PRIVATE_RUNS"
uv run bev-calib evaluate --protocol configs/protocols/nuscenes_calibration_v1.yaml --manifest "$PRIVATE_MANIFESTS/evaluation.json" --method classical --output-dir "$PRIVATE_RUNS"
uv run bev-calib train --config configs/correctors/convnextv2_tiny_v1.yaml --development-manifest "$PRIVATE_MANIFESTS/development.json" --calibration-manifest "$PRIVATE_MANIFESTS/calibration.json" --output-dir "$PRIVATE_TRAIN_RUN" --seed 17
uv run bev-calib evaluate --protocol configs/protocols/nuscenes_calibration_v1.yaml --manifest "$PRIVATE_MANIFESTS/evaluation.json" --method learned --checkpoint "$PRIVATE_TRAIN_RUN/selected_checkpoint.pt" --output-dir "$PRIVATE_RUNS"
```

Paths above are illustrative shell variables, not checked-in configuration. Train and
evaluate read `NUSCENES_ROOT` only from the invoking shell and require trainval. Training
also requires `BEVCALIB_PRETRAINED_WEIGHTS` (local file), `BEVCALIB_PRETRAINED_SOURCE`
(compatible named timm profile) and `BEVCALIB_PRETRAINED_SHA256` (actual file digest).
`BEVCALIB_DEVICE` defaults to `cpu`; an explicitly prepared formal runtime may choose
another Torch device. Producer identity is supplied through the existing
`BEVCALIB_RUN_PROVENANCE` JSON contract. There is no public CLI flag that relaxes the
formal cohort or checkpoint requirements. Internal `Runtime(synthetic_fixture=True)`
is an explicit programmatic test interface; synthetic checkpoints remain ineligible
for formal evaluation. See [training-contract.md](training-contract.md).

Preflight accepts mini for installation/debug checks and trainval for formal readiness.
It snapshots actual metadata-table hashes and payload availability, requires every
paired CAM_FRONT/LIDAR_TOP keyframe, and records optional missing payloads separately.
Per-offset selection evidence includes actual timestamps, approximation errors and
failure reasons. Runtime validation failures exit 1 with the affected root/table/pose
diagnostic; argument parsing failures exit 2. Missing required pairs fail before output. Freeze only accepts
trainval, validates the supported matrix values, and writes three verified V2
manifests into a new directory. Underfilled allocations retain explicit shortages;
they cannot pass formal training/evaluation. Preflight and manifests contain private
tokens and must stay outside Git. A cohort-only transfer is validated by the evaluator's
cohort checks, not by pretending it is a full installation.

Evaluation writes complete atomic scene documents under a method/run identity and a
verified private `evaluation_manifest.json` sidecar. Existing roots cannot be reused
for another cohort. A completion marker verifies every expected row and file hash.
Identity measures all declared extrinsic and timing conditions; classical and learned
measure only extrinsic conditions, including each axis's zero reference. Timing is
identity stress evidence and never a learned target or classical recovery comparison.
Signed pose components, nonpositive edge scores and null unavailable measurements
retain their actual meanings. All declared range bins remain present. The ground-plane
baseline caveat is explained in [coordinate-contract.md](coordinate-contract.md).

## Private aggregation, public rendering

`bevcalib.metrics.summary.summarize_result_runs(private_root)` verifies the private
manifest and complete compatible identity/classical/three-seed learned inventory.
It produces descriptive per-condition validity, pose, pixel, edge, recovery and GT-range
statistics with explicit denominators. `synthetic_fixture=True` additionally permits
the explicitly synthetic identity/classical pair. Use
`bevcalib.artifacts.summary.write_safe_summary(document, public_artifacts / "calibration_summary.json")`
to validate and export the token-free `bev-calibration-summary/v1` contract. It retains
protocol/cohort/run/checkpoint/measurement digests, not raw observation identifiers.
This descriptive export does not replace the separately approved formal statistical
artifact set or paired scene bootstrap.

```bash
uv run bev-calib report --claims docs/claims.yaml --artifacts-dir artifacts --output-dir site
uv run bev-calib audit-claims --claims docs/claims.yaml
```

The renderer needs only `calibration_summary.json` and a claims registry whose artifact
paths resolve from the repository root. It validates the document digest, supported
measurement policy and run/condition identities. Every displayed numerical value,
including zero, invalid rates, counts, seed and fault level, requires a unique verified
claim at its exact scalar JSON pointer. Claim IDs are unique across the registry.
Parent-object claims may supply nonnumeric context but never authorize descendants.
Numeric claim text must match its scalar, and synthetic evidence cannot become observed.
The generic audit retains its legacy object-pointer support; report checks are stricter.

The report always displays validity and reasons, separates timing stress, and shows
all GT-range bins for the explicitly labeled yaw-zero baseline. Additional verified
scalar claims render as measurement cards. Empty bins show unavailable estimates.
The HTML has embedded style and SVG, escaped text, no remote assets, private manifests
or checkpoint dependencies, and deterministic bytes on repeated builds. There are no
formal values or automatically verified claims bundled with this implementation.
