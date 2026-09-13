# Calibration error distribution interchange

This optional artifact describes calibration errors; it does not establish an AEB
benefit. The planned producer release is `v1.0.0`, **not yet published**. The
exporter and an actual exported evidence artifact require separate commits so
the envelope can identify the exporter commit without a self-reference.

## Fixed condition and source

The portfolio export is fixed to `method="identity"`, `seed=None`, `axis="yaw"`,
`level=1.0` (degrees). This choice precedes inspecting the exported distribution.
Do not select a learned seed or change the condition to improve the result.

`export_calibration_distribution` in
`src/bevcalib/artifacts/calibration_distribution.py` accepts an input run-set
directory, an unused output path, and the keyword arguments `method`, `seed`,
`axis`, `level`, `producer_release`, and `created_at_utc`.
The input directory contains `evaluation_manifest.json` and the complete V2
identity, classical, and three learned-seed runs. The existing source validator
requires the formal 30-scene evaluation cohort, consistent measurement and
protocol identities, distinct learned checkpoint identities, and exact scene
file inventories. Each selected-run scene is validated and hash-checked when
read. Other runs' scene contents are not read by this export.

The producer commit is measured from the source checkout. The exporter must be
present in that commit; uncommitted source, configuration, or lock changes cause
refusal. A caller cannot supply a producer commit. Export from the reviewed,
committed source only after its full repository verification passes.

## Samples and support

For each scene, select the fixed condition and use every frame with a pose
measurement. Pose eligibility is independent of overall row validity. Average
each signed roll, pitch, yaw, x, y, and z error component over these frames.
Retain rotation and translation from the same scene at the same array index.
This is an arithmetic component mean, not a rotation norm or an SO(3) mean.

Each supported scene contributes one paired six-component sample, regardless of
its frame count. `sample_pose_valid_frames` records that count in sample order.
Scenes without eligible frames contribute no sample but remain in the support
counts. The payload reports total, valid, and excluded scenes and frames.
Rotation units are degrees; translation units are metres. Raw scene rows are
released before reading the next scene.

The `portfolio-artifact-envelope/v1` contains artifact type
`calibration-error-distribution/v1`, the protocol and manifest hashes, exporter
commit, release label, UTC timestamp, and canonical payload SHA256. Payload
source metadata retains the selected run identity, completion-file digest, and
evaluator commit and lock digest. Scene and sensor tokens are not exported.

## Refusal and reproduction

Timing conditions, undeclared conditions, missing or incompatible sources,
malformed paired components, non-finite values, and entirely empty support are
rejected. Each scene-mean rotation component must be within ±30 degrees and each
translation component within ±2 metres, inclusive. Values are not clipped and
large-error scenes are not dropped to satisfy the bounds. Any scene validation
failure prevents publication of the output. Existing output files are preserved.

Save the first export invocation's valid UTC timestamp and reuse it for both
reproduction runs. With identical source, arguments, and exporter commit, the
two output files must be byte-identical. The release label must use `vN.N.N`;
an `-rc` suffix is not accepted by the existing envelope schema. A planned label
does not prove that a public tag exists.

## Consumer acceptance

In the consumer's own pinned environment, call
`import_calibration_distribution(envelope_path, expected_protocol_hash)` using
the actual producer protocol hash. Verify acceptance of the unchanged envelope
and separate refusal of damaged payload SHA, wrong artifact type, and wrong
protocol. Neither repository imports the other's package.

These samples describe scene means, not object-level perception errors. Passing
the importer establishes format and provenance compatibility only. It does not
show that learned calibration improves AEB, and does not replace or modify the
consumer's 26 formal attribution configurations. Actual export, consumer checks,
and public-tag verification remain separate acceptance steps.
