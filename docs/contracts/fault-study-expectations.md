# Frozen fault-study expectations

Internal agreement among the five formal documents does not establish that they
belong to the intended experiment. The mechanical comparison in
`bevcalib.artifacts.study_expectations` accepts a separate expected identity. Its
values must come from the reviewed protocol, locked manifest, run-producing
revision and selected training checkpoints. Copying them from the result under
review would make the comparison circular.

The strict `ExpectedStudy` input has these required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `bev-fault-study-expectations/v1` |
| `evidence_type` | `observed` or `synthetic`; fixture evidence cannot silently become observed |
| `protocol_hash` | Frozen evaluation protocol SHA256 |
| `dataset_manifest_hash` | Frozen evaluation manifest SHA256 |
| `raw_producer_commit` | Forty-character Git revision that produced the original raw runs |
| `raw_producer_lock_sha256` | SHA256 of that producer's dependency lock |
| `checkpoints` | Exactly `learned-17`, `learned-42`, and `learned-73`, with three distinct checkpoint SHA256 values |

Unknown fields, invalid digests, missing/extra seeds and repeated checkpoint hashes
are refused. The current analyzer revision may differ from the raw producer; it
must not be substituted for `raw_producer_commit`.

`validate_expected_identity(identity, expected)` compares an already validated
`FormalIdentity` against these expectations. `validate_checkpoint_files(expected,
paths)` additionally hashes the actual three checkpoint files supplied by path.
It reads bytes without model deserialization or device allocation. Matching names,
metadata or file sizes alone do not satisfy this check.

These functions are building blocks, not a full study acceptance command. The
caller must still validate the complete formal set, audit displayed claims, verify
its runtime/source, and separately validate raw shards where required. They do
not prove when the expectations were frozen, attest checkpoint semantics, or
establish model efficacy. Preserve the independently reviewed expectation source
and its digest alongside the eventual validation receipt.

## Validator process and source

`bevcalib.artifacts.validator_runtime` separates the current validator from the
historical raw-run producer. `ExpectedValidator` takes schema version
`bev-validator-runtime-expectations/v1`, the exact three-component Python version,
validator Git `commit`, `lock_sha256`, and `source_sha256`. These values must also
be frozen from a separately reviewed environment before accepting results. Do not
generate an expected object from the process being checked and call that acceptance.

`source_snapshot(root)` returns a relative-path-to-SHA256 inventory and the SHA256
of its UTF-8 JSON with sorted keys and compact separators. Its fixed scope is
`src/`, `configs/`, `schemas/`, `scripts/`, `.agents/skills/`, `pyproject.toml`, and
`uv.lock`. It hashes actual bytes, including untracked and Git-ignored files;
Git HEAD alone cannot hide local modifications. Python `__pycache__` directories
are excluded, source symlinks and Windows junctions (including scoped-root
ancestors such as `.agents`) are refused, and source plus both required root
files must exist. Newline changes affect this byte identity: use the exact approved
checkout when freezing expectations, rather than assuming two OS checkouts match.

`validate_validator_runtime(expected)` locates the checkout from its own imported
module, checks the actual Git top-level path and HEAD, reads the current process's
Python version and executable, then compares the lock and complete source snapshot.
Git's UTF-8 output is decoded explicitly, including checkouts with Chinese paths.
It cannot validate a caller-supplied clean checkout while executing from another.
An explicitly reviewed uncommitted candidate may have its own frozen snapshot;
matching it does not assert that its bytes were committed or release-approved.

This probe allocates no model or GPU and installs nothing. It does not inspect
third-party package versions or prove that installed dependencies match the lock.
It also does not attest to malicious process changes, in-memory monkeypatches,
historical training environments, or raw-shard validity. A complete study command
must retain those separate checks, the original producer identity, and the claim
audit; these building blocks alone do not produce a study completion receipt.

## Joined input validation

`bevcalib.artifacts.fault_study.validate_fault_study` joins these building blocks
with the existing formal publication and raw result validators. Its expectation
file uses `schema_version: bev-fault-study-validation-expectations/v1`, with two
required objects: `study` (`ExpectedStudy`) and `validator` (`ExpectedValidator`).
There is no automatic expectation-generation fallback.

All five formal documents, the publication claims registry, all five completed
raw runs and the three selected checkpoint files are required. The validator:

1. Checks the actual validator process/source before reading the study.
2. Loads and audits the entire formal five-document set and registry.
3. Compares formal identity with independent expectations and hashes all three
   checkpoint files as bytes.
4. Checks every scalar in the predeclared publication mapping against its exact
   claim, retaining unavailable values without treating them as verified numbers.
5. Requires the raw manifest, measurement identity and every run identity to match
   formal evidence, then reads every expected scene through the existing SHA,
   schema, condition and sensor-identity validators. Only one scene's rows are
   retained at a time; no estimates or bootstrap draws are recomputed.
6. Rechecks the raw manifest and all run markers after the scene reads, then
   rechecks formal documents, registry, expectations and validator source/runtime.

Success returns `bev-fault-study-input-validation/v1` with status
`validated-inputs`, input digests and validated scene/scalar counts. The receipt
does not contain local paths, raw scene tokens or the interpreter executable path.
The CLI prints it only after all checks succeed and writes no files itself.

This status establishes the documented input checks, not a new experiment result.
It does not rerun statistical aggregation, deserialize checkpoint semantics,
verify installed dependencies against the lock, assess calibration efficacy or
authorize publication. Runtime/source comparisons are against separately frozen
expectations, and the original raw producer remains distinct from the validator.
These are read-time consistency checks, not an atomic filesystem snapshot against
concurrent writers. Run validation on preserved, quiescent inputs.
