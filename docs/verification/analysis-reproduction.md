# Locked calibration analysis reproduction

The two independent CPU analysis processes completed successfully on 2026-09-10
(Asia/Taipei). All five portable JSON files are byte-for-byte identical. The
copied evidence was then loaded with the formal artifact-set validator on Windows.
This verifies artifact integrity and reproducibility; it is not a claim that a
learned corrector outperforms the classical baseline.

## Source and scope

- Raw evaluator producer: `aeb3f28265c2ee3c0f7556417f9c8bf8a550acb7`.
- Analyzer: `33ff76485ed17194626c51333c941a33fe712902`.
- Dependency lock SHA-256: `7ea241d52736074352f0435128768ab7460c18d5bebaee7c783e28a22b54ea82`.
- Python 3.12.13, NumPy 1.26.4, SciPy 1.17.1; Linux CPU execution, one BLAS/OMP
  thread, GPU hidden. Two fresh processes read the same immutable private inputs.
- Thirty locked scenes and 1,207 frames; identity, classical, and learned seeds
  17, 42 and 73. The 150 original scene files contain 370,549 method-condition rows.
- Identity has 67 conditions per frame; each correction run has 60. Timing is
  identity-only. No training, inference, cohort selection or sensor payload was changed.

## Numerical pairing repair

The first analysis attempt stopped because exact floating-point equality rejected
the ground-truth range of otherwise matched objects. A SHA-verified audit of all
30 scenes found 34,605 unique frame/box pairs: inventories, within-run ranges,
within-baseline agreement and within-learned agreement all matched. Cross-runtime
range differences affected 31,623 pairs, with a maximum span of
`3.744560217455728e-12 m`. Every range bin agreed. The nearest 10/20/40/80 m
boundary was `0.0003079877617082616 m` away, so neither bins nor the inclusive
80 m cutoff changed.

GT range comes from method-independent float64 true-camera/global-box geometry.
The historical Colab CPU/BLAS dispatch was not recorded; a particular instruction
or BLAS variant is not established as the cause. The demonstrated failure was
using bitwise equality as a cross-runtime geometric pairing condition.

Analysis policy V2 uses the fixed geometry precision of `1e-9 m`, with zero
relative tolerance, a bound on the entire range span, exact bin agreement and
exact inclusive-cutoff agreement. The budget was not selected from method
performance. Original ranges, errors, raw hashes, operator support, weighting,
bootstrap settings and producer identities remain unchanged. See the
[pairing contract](../coordinate-contract.md#cross-run-gt-range-pairing).

## Verification

Regression tests first failed under exact equality (3 failed, 6 passed), then the
44-test analysis suite passed with the repair. The complete eight-stage gate
passed: each test run had 1,169 passes and four expected unmounted-data skips;
3,521 statements and 998 branches had 100% coverage. Boundary tests reject bin
crossings, the inclusive 80 m cutoff crossing, material drift, relative-tolerance
scaling and non-transitive range chains. This gate preceded the analyzer commit.

| Pass | Actual exit | Wall seconds including monitoring |
| --- | --- | --- |
| A | 0 | 1163.055 |
| B | 0 | 1182.696 |

Only this job's result-file cache was released between reads; no global cache
drop or other project's processes were used. The complete private receipts retain
the analyzer tree, source hashes, run-marker hashes, exact exits and logs.

## Portable evidence

The [five evidence files](../evidence/README.md) contain aggregate statistics and
source digests, not sensor payloads, sample/box identifiers or checkpoints.

| File | SHA-256 |
| --- | --- |
| `exclusions.json` | `a1c31cd2d309c159e3e62053d3561ce53498bf4b090be7eac3aec1fe8cbf0604` |
| `intervals.json` | `b8bd607553198abb1085f119c0ee1efbc685ebcd50c05027e87ec8cfdd8e680e` |
| `metrics.json` | `7a7637f90106848afb8b8fe148cb112758a94be056c8279e084e0ae302e8f074` |
| `recovery.json` | `76a7096de6251dd494f1167540d160c10d13a027541e5136db432828a2d90a04` |
| `timing.json` | `59c3eabd861163fe8d869bf8a30147613324f9e3597df7328a1b2f65c676b7dd` |

For a licensed holder of the frozen private five-run input root, run
`bevcalib.analysis.aggregate.aggregate_formal_results(input_root, output_dir)`
twice in fresh processes from the pinned analyzer revision. Each output directory
must be new. Compare all five files byte-for-byte, then call
`bevcalib.artifacts.documents.load_formal_artifact_set(output_dir)`.
Portable validation alone needs no private dataset or GPU. The JSON structure
remains schema V1; the embedded analysis policy is V2 and older policy documents
are refused rather than relabeled.
