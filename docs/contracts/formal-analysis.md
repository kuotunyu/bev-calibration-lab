# Formal calibration analysis contract

The estimands, weighting and bootstrap were fixed before locked evaluation.
Analysis policy V2 clarifies the cross-runtime GT range identity check after
an exact-equality failure; it does not alter measured errors or tune recovery
thresholds. The five artifacts are
`metrics.json`, `intervals.json`, `recovery.json`, `timing.json`, and
`exclusions.json`. Their Pydantic models and generated JSON Schemas bind the same
protocol, frozen cohort, measurement identity, producer revision/lock, and five
completed source runs. A missing scene, sample, condition, method, or learned seed
is an incomplete study and is refused. An unavailable operator is a measured
exclusion, with its denominator and reason retained.

The required runs are identity, classical, and learned seeds 17, 42, and 73.
Identity contains all 67 declared conditions per sample. Correction methods
contain the 60 declared rotation/translation conditions, including each axis's
zero reference. Timing is an identity-only availability/stress measurement;
there is no timing-recovery comparison for a six-degree-of-freedom corrector.

## Estimands and units

Each eligible frame contributes equally within its scene. Each eligible scene
then contributes equally to the study estimate. The eligibility set belongs to
the operator being measured; an overall-invalid row can retain a usable pose,
projection, edge, or ground-contact measurement.

- Pose: rotation geodesic and axis components use degrees. Translation components
  and norm are converted from the source rows' metres to centimetres. Signed
  components are separate bias descriptors; their absolute magnitudes are the
  per-axis error quantities used for improvement intervals.
- Pixels: `pixel_frame_p50_px` and `pixel_frame_p90_px` are the mean of per-frame
  quantiles within each scene, then the mean across scenes. They are **not pooled
  pixel quantiles**. Each method's frame quantile uses its own within-row valid
  projection errors. The source rows have no point identifiers; cross-method
  point pairing is neither inferred nor fabricated. Match frames by sample ID
  and require both frame quantiles to exist. Preserve each method's input-point
  and measured-pixel counts separately.
- Edges: average measured distance-field scores per frame and scene, in pixels.
  Scores are nonpositive; a larger score is better. An absent score stays null.
- BEV: match the exact sample and box identity. Ground-truth ranges must have a
  whole-group span at most `1e-9 m` (zero relative tolerance), identical range
  bins and identical inclusive 80 m cutoff classifications. Original ranges
  and measured errors are never rewritten. See the
  [numerical pairing contract](../coordinate-contract.md#cross-run-gt-range-pairing).
  Average errors of common valid boxes within a frame, then frames within
  a scene, then scenes. Retain the fixed 0–10, 10–20, 20–40, 40–80, and 80+ metre
  bins and disclose paired object, frame, and scene counts. Exact 80 m is within
  the operator cutoff; values above it remain excluded. No-data bins are null.
  The horizontal-plane reconstruction can retain a nonzero residual at zero
  calibration fault when a real box's bottom is off that plane.
- Recovery: the joint criterion is geodesic error at most 0.25 degrees and
  translation norm at most 5 cm. Compute the percentage of eligible paired
  frames within each scene and average scenes. Its improvement uses percentage
  points, not relative percentage change.

## Pairing and uncertainty

Comparisons are identity→classical, identity→each learned seed, and
classical→each learned seed. Additional comparisons use the arithmetic mean of
all three learned seeds on support common to the baseline and all three seeds.
Every seed remains separately reported. Those additional intervals are
conditional on these three fixed seeds; they do not estimate uncertainty over
training randomness or choose a winning seed.

For each metric, reduce matched observations to a before/after pair for each
scene. Reuse the deterministic paired scene bootstrap with 5,000 resamples,
seed 20260831, and a 95% interval. Scenes are sampled with replacement; frames
are never treated as independent bootstrap units. Cached resampling indices
are immutable and retain the original SHA-256 counter algorithm.

Error improvement is before minus after. Edge and recovery improvement is after
minus before. Positive means improvement in all comparison columns. Signed bias
descriptors are not silently converted into signed-error improvement columns.
No common eligible scene means null estimates and intervals with an explanation.
A one-scene interval can be degenerate; its scene count remains explicit.

The validator checks improvement against the reported before/after means in the
declared direction. Different arithmetic orders may differ by floating-point
rounding, so this consistency check allows 64 units in the last place at the
larger operand magnitude (with a minimum scale of one). This numerical check
does not change timing, recovery, geometric, or other physical thresholds.

## Integrity and bounded execution

`bevcalib.analysis.aggregate.aggregate_formal_results` validates the complete
source inventory, reads one verified run-scene at a time, and reduces raw pixel
arrays immediately. Compact frame/box data live for one five-method scene; only
scene estimands and counts persist. A corrupt late scene refuses the entire
publication. Existing evidence directories are never overwritten.

`bevcalib.artifacts.documents.load_formal_artifact_set` checks the five files,
their hashes, shared identity, complete metric inventories, support, and the
duplicated recovery/timing/exclusion views. It can validate the portable files
without private sensor payloads. Raw scene documents, sample/box identifiers,
cohort manifests, and checkpoints remain private.

The JSON document schema version remains V1 because its structure is unchanged;
the embedded analysis policy identifier is V2. The current validator refuses old
V1 policy documents and mixed policy descriptions instead of silently relabeling
them. Original raw evaluation documents retain their original producer identity.

## Learned inference device

The CLI reads `BEVCALIB_DEVICE` (`cpu` by default; `cuda` for Colab learned
inference). The checkpoint loader places the model on that device; the predictor
creates its input there and returns CPU measurements for serialization. An
unsupported or unavailable CUDA device is refused before output. Evaluation
provenance records the selected inference device. Host CPU and mocked placement
tests do not constitute evidence that a CUDA kernel executed; that check belongs
to the actual pinned Colab runtime before formal work.
