# Native commands and portable reports

## Offline synthetic calibration explorer

The [explorer](demo/calibration-explorer.html)
([hosted copy](https://kuotunyu.github.io/bev-calibration-lab/demo/calibration-explorer.html))
embeds Plotly.js and preserves its copyright and MIT notice, including the
BSD-3-Clause notice of the MapLibre GL JS code in the bundle. It needs no dataset, model, server, or
network connection. Browser interaction acceptance completed for the release;
see [release verification](verification/publication-and-interchange.md).

With the report extra installed, reproduce it in a new local output file:

```python
from pathlib import Path
from bevcalib.report.explorer import build_explorer

output = Path("calibration-explorer.html")
with output.open("xb") as handle:
    handle.write(build_explorer().encode("utf-8"))
```

Each axis button resets to zero and replaces the slider with that axis's declared
levels. Fixed observations, assumed projection, reconstructed ground points, and
BEV errors update together. Ground-plot ranges adapt to retain every point; use
the metre values when comparing conditions. The three points are first-party
synthetic geometry, and neither calibration recovery nor AEB performance is
measured by this demonstration.

## Native evaluation and report commands

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
Annotations are indexed once by sample token when resolving the installation, preserving
native order and empty samples. Observation loading reads only that sample's tuple;
changes on disk appear after a fresh resolution. This avoids a full annotation-table
scan per training/evaluation observation, without claiming a measured full-data speedup.
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
retain their actual meanings. An unavailable or out-of-tolerance timing row carries
selection evidence only: null estimate/pose/edge, empty pixels/contacts and zero projection
count. V2 validation rejects any populated operator field there. Other partially invalid
rows retain independently measurable operators and their appropriate denominators.
All declared range bins remain present. The ground-plane
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
Each displayed scalar claim additionally requires an optional-to-legacy `report_binding`:

```yaml
report_binding:
  expected_summary_sha256: <digest of the exact verified safe summary>
  expected_value: <exact finite numeric scalar at metric_path>
```

The renderer requires both fields and compares them to the loaded document and scalar.
The expected whole-document digest binds its run, checkpoint and measurement identities;
a rehashed replacement at the same path cannot reuse old verified claims, even when its
numbers are unchanged. Values must be actual finite numbers, not strings or booleans.
Numeric-free wording is allowed only with this explicit binding. Claims must be verified
again when either the source identity or scalar changes. Legacy ClaimV1 registries and
the generic audit still work without the optional binding; they do not authorize report
numbers without it. Parent-object claims may supply nonnumeric context but never authorize descendants.
Numeric claim text must match its scalar, and synthetic evidence cannot become observed.
The generic audit retains its legacy object-pointer support; report checks are stricter.

The report always displays validity and reasons, separates timing stress, and shows
all GT-range bins for the explicitly labeled yaw-zero baseline. Additional verified
scalar claims render as measurement cards. Empty bins show unavailable estimates.
The HTML has embedded style and SVG, escaped text, no remote assets, private manifests
or checkpoint dependencies, and deterministic bytes on repeated builds. There are no
automatically generated formal claims in this legacy interface; use the explicit
formal mode below for the five-document result set.

## Formal five-document publication

The formal mode reads the existing `metrics.json`, `intervals.json`, `recovery.json`,
`timing.json`, and `exclusions.json` together. It validates their schemas, document
digests, shared source identity, declared estimands, and cross-document consistency.
It does not read sensor data, train a model, run inference, or recompute statistics.

Run from the repository root in the locked environment. These commands create a new
candidate registry and report under the ignored artifact directory; they do not
overwrite the reviewed `docs/claims.yaml`:

```bash
uv run --frozen bev-calib generate-claims --artifacts-dir docs/evidence/nuscenes_calibration_v1 --output artifacts/formal-publication/claims.yaml
uv run --frozen bev-calib audit-claims --claims artifacts/formal-publication/claims.yaml
uv run --frozen bev-calib report --formal --figures --claims artifacts/formal-publication/claims.yaml --artifacts-dir docs/evidence/nuscenes_calibration_v1 --output-dir artifacts/formal-publication/site
```

Existing output registries and report directories are refused. For another build,
choose a new output directory, preserving the prior result for comparison. Promoting
a candidate to the committed registry is a separate reviewed source change.

`--figures` adds two offline SVGs and linked previews to the same report. Open the
standalone SVG links to inspect exact-value tooltips; an HTML image preview does
not expose the SVG's individual point tooltips. Omitting this option produces the
tables alone. The option requires `--formal`.

The recovery figure has six axes and three columns: absolute recovery for all five
methods, paired improvement over identity, and paired improvement over classical.
Absolute rates use percent; improvements use percentage points. The fixed-three-seed
mean appears only in the paired comparisons where the source defines it, never as
an invented absolute ensemble curve. The BEV figure retains all sixty extrinsic
conditions, five methods and five GT range bins on a shared vertical scale. Identity
timing stress stays in the separate tables. Unsupported values remain unavailable,
break the plotted line, and retain their reasons and support; they are not zeros.
Point tooltips carry source document digests, exact claims, values and support.
Overlapping markers expose the neighboring observations' provenance together.

The report includes every method and condition for geodesic rotation, translation
norm, both pixel quantile estimands, edge alignment, and all fixed BEV range bins.
Recovery uses the recovery document; paired comparisons retain their original
before/after/improvement values and intervals. Timing remains identity-only stress.
Global validity counts, operator support, unavailable values and their reasons stay
visible. Signed and per-axis pose descriptors remain in the complete linked metrics
document rather than being relabeled as the displayed aggregate pose metrics.

Every displayed numeric result has a verified scalar claim. In formal mode,
`ReportScalarBinding.expected_summary_sha256` binds the referenced formal document's
`document_sha256`; it does not refer to a legacy `calibration_summary.json`.
`expected_value` retains its exact numeric value. Missing display claims, duplicate
IDs, stale bindings, incompatible evidence labels, and escaped artifact paths are
refused before the report is written. The standalone HTML carries the original five
source files and a registry copy; registry artifact paths remain relative to the
source repository for audit commands.

Source consistency checks compare validated models, including schema defaults, so
omitting an optional null field does not count as changing the result. Actual changes
to a validated source model or to the registry during construction are refused.
Copied evidence retains the original source bytes, including omitted optional fields.

Omitting `--formal` keeps the legacy safe-summary report interface.

## Derived operating envelope

The derived analysis in [docs/analysis](analysis/operating-envelope.md) is rebuilt
from the released evidence with:

```bash
uv run --frozen python -m bevcalib.analysis.operating_envelope --artifacts-dir docs/evidence/nuscenes_calibration_v1 --output-dir artifacts/operating-envelope
```

It writes `operating-envelope.json` and `operating-envelope.svg` into a new directory
and refuses an existing one. It reads `metrics.json`, `intervals.json` and
`recovery.json` through the formal artifact-set validator, refuses documents whose
digest differs from the released v1 evidence, and labels its output `derived` with the
digest of every source. A contract test compares a rebuild with the committed files.

## Pages site

```bash
uv run --frozen python -m bevcalib.report.site --claims docs/claims.yaml --artifacts-dir docs/evidence/nuscenes_calibration_v1 --output-dir site
```

The site root `index.html` is a small landing page. It states the result with every
observed number bound to its verified claim, shows the operating-envelope figure and
links the explorer, the full report, the repository, the release and the known issues.
The complete formal report is `evidence/index.html`, next to the five source
documents it copies; `claims.yaml`, `figures/`, `analysis/` and `demo/` keep fixed
paths, and `site-inventory.json` lists every file with its SHA-256. The Pages workflow
runs this command to build the published site.

## Validate a complete set of study inputs

`bev-calib validate-study` combines the formal publication audit with independently
frozen study/validator expectations, all five raw runs and all three selected
checkpoint byte hashes. Every expected raw scene is read, so this command requires
CPU, RAM and disk-read capacity even though it does not allocate a GPU. Do not
launch it concurrently with another memory-intensive job without resource planning.

```bash
uv run --frozen bev-calib validate-study \
  --expectations artifacts/validation/expected-study.json \
  --artifacts-dir docs/evidence/nuscenes_calibration_v1 \
  --claims artifacts/formal-publication/claims.yaml \
  --raw-runs-dir /path/to/private/completed-runs \
  --checkpoint-17 /path/to/private/seed17/selected_checkpoint.pt \
  --checkpoint-42 /path/to/private/seed42/selected_checkpoint.pt \
  --checkpoint-73 /path/to/private/seed73/selected_checkpoint.pt \
  --repository-root .
```

The paths above are interface examples, not prepared experiment inputs. Preserve
the separately reviewed expectation file; never fill it from the result being
accepted. See [the expectation contract](contracts/fault-study-expectations.md)
for its exact schemas and source snapshot scope. The CLI emits a JSON
`validated-inputs` receipt and exits zero only after all checks pass. That receipt
is not a full experiment/efficacy, package-installation or release acceptance.
