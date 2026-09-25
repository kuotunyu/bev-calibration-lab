# Cohort contract before the first formal freeze

This contract covers data-independent construction and integrity checks. No formal
cohort, training result or dataset-derived claim was created by it. The synthetic tests
use invented identifiers and timestamps. Actual preprocessing, the production Torch
backend and evaluation services are specified in the
[training contract](training-contract.md); configuration alone does not establish the
resolution of a real tensor.

## Protocol identity

`bevcalib.cohort.resolve_protocol(path)` validates the
[cohort protocol](../configs/protocols/nuscenes_calibration_v1.yaml), resolves its
`perturbations` relative to that file, and reads the referenced bytes. Its identity is
SHA-256 of canonical sorted, compact JSON with these three fields:

- `schema_version: bev-calibration-protocol-identity/v1`
- `protocol`: the validated protocol document
- `perturbations_sha256`: SHA-256 of the referenced perturbation file bytes

Machine-specific absolute paths are outside the identity. Parsed YAML layout is not
significant; referenced perturbation bytes are significant. Missing references fail.
The trainer and future freeze caller must use this public resolver's `protocol_hash`
and `dataset_version`, never the hash of the cohort YAML alone. The existing
[perturbation matrix](../configs/perturbations/formal_v1.yaml) is unchanged.

## Records and persistence

`SceneRecord` is a frozen validated dataclass. Every scene records its scene/log token,
location and official scene-level split, aligned nonempty sample, camera sample-data
and LiDAR sample-data token sequences, and three separate nonnegative integer timestamp
sequences. Each sequence is strictly increasing. Sample and sensor identifiers are
unique within and across scenes. Sensor timestamps are not replaced by sample time.
Repeated identical input scenes deduplicate; conflicting duplicate scenes or a log
with conflicting locations fail. Different official splits within one log are valid.

`CohortManifestV2`, defined only in `cohort/manifest.py`, persists all record fields,
`schema_version: bev-calibration-cohort/v2`, source `dataset_version`, `protocol_hash`,
role, allocation diagnostics, and `manifest_sha256`. The digest covers the entire
canonical JSON body except its own field, including diagnostics. Saving revalidates,
refuses overwrite, and writes sorted JSON with LF line endings. Loading recomputes the
digest and validates diagnostic consistency. Legacy V1 remains explicitly readable
and hash-checked, but never gains invented metadata or enters formal consumption.

The formal loader checks role, resolved protocol identity, dataset version, exact role
count, official split and twenty distinct calibration logs before returning. The trainer
additionally refuses cross-role scene/log/sample/camera/LiDAR identifier overlap before
calling any backend method. All future evaluation consumers must use this same loader
and check their complete role set for overlap. A hash establishes integrity and binding,
not that an untrusted producer really examined the source data.

## Deterministic allocation

Requested counts are development 100 from train, calibration 20 from twenty train logs,
and evaluation 30 from val. Per-role location quotas are proportional to original
deduplicated scene availability in that role's official split. Integer division supplies
the floors; remaining seats go to largest integer remainders, tied by lexical location.
If a split is empty, uniform location weights preserve the requested total in diagnostics.
Quotas remain fixed through log assignment: no other-location backfill occurs.

The scene key is `(sha256(protocol_hash + ':' + scene_token), scene_token)`. All candidate
ordering and final within-location scene selection use that key. Locations use lexical
order. Exact SHA ties use the token itself, independently of input order.

For split-homogeneous logs, evaluation is independent. Reserving a calibration log costs
every training scene in that log. To accept a hash-first calibration candidate, the
allocator checks its cost plus the smallest remaining log capacities needed to finish
the calibration quota. It accepts only if that completion leaves the development quota
attainable. This lookahead is exact because each calibration scene costs one distinct log.
Once reserved, the earliest scene in each calibration log is selected, and development
selects the first eligible scenes from unreserved logs.

For logs shared by train and val scenes, a per-location suffix dynamic program tracks
attainable `(development scenes, calibration logs, evaluation scenes)` capacities. One log
can be assigned to development, calibration, evaluation or unused; only observations of
the role's official split contribute. Capacities are capped at the fixed quotas, giving
at most `(D+1)*(C+1)*(E+1)` states per suffix, rather than enumerating log subsets. Logs
are visited by their earliest scene key. Ties between feasible assignments prefer
calibration, then development, then evaluation, then unused. Each choice must leave the
selected target reachable in the remaining suffix. Selected scenes are hash-first within
the assigned logs; calibration takes one train scene per assigned log.

When the full target is impossible, both paths maximize development first, then calibration,
then evaluation, bounded by the original location quotas. This is an explicit underfill
policy, not a substitute formal cohort. Every role persists all four location diagnostics:
requested, original available scenes/logs, actual selected count and shortage reason.
`insufficient_scenes` means original availability is below the quota; `joint_log_capacity`
means scene availability was sufficient but exclusive log assignments were not. Complete
strata have a null reason. Underfilled manifests are saveable diagnostic artifacts but
formal consumers refuse their incomplete role counts.

## Adversarial and synthetic evidence

- With train log capacities 3, 1, 1 and quotas development 3/calibration 2, the first scene
  hash may belong to the three-scene log. Reserving that log would destroy feasibility.
  Lookahead skips it, reserves both singleton logs and keeps all three development scenes.
- With a shared log containing three train scenes and one val scene, another train-only
  singleton and a val-only singleton, quotas 3/1/1 are feasible. The shared log serves
  development, the train singleton calibration, the val singleton evaluation. Removing the
  val-only log gives 3/1/0 with an explicit evaluation shortage; no split is relabelled.
- Tests cover 400 synthetic train scenes over 40 logs plus 120 val scenes over 12 logs,
  exact 100/20/30 counts, location proportions, shuffled byte identity, hash changes,
  mixed-log permutation invariance, and a tiny exhaustive assignment oracle independent
  of the production solver. These are synthetic correctness checks, not nuScenes results.

The corrector configuration pins 448 by 800 with positive multiples of 32. Preprocessing
must apply the same geometry to RGB, projection, validity and camera intrinsics and validate
the final five-channel tensor. The existing array-stacking helper remains compatible
with explicitly synthetic shapes.
