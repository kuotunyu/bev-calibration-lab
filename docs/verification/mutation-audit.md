# Mutation audit

Mutation testing asks a question coverage cannot: not "was this line executed"
but "would anything have noticed if it were wrong". This project runs it on the
pure core only — `geometry`, `perturbations`, `operators`, `correctors` and
`metrics` — because those are the modules whose arithmetic reaches a published
number. The nuScenes adapter, the CLI, the training glue and the artifact
writers are held to the same 100% branch gate but are not scored here: their
behaviour is defined by the devkit and by file formats rather than by
arithmetic this repository owns.

The release gate is a score of at least 90%, counted as mutants KILLED. A
surviving mutant may also be shown equivalent, and an equivalence has to be an
argument from the code and its environment rather than an assertion that it did
not matter — but the gate deliberately does not depend on any such argument,
because a wrong one inflates the score silently while a kill needs no argument
at all.

## How to reproduce

Mutmut rewrites source, so it cannot run on the Windows working tree; it runs
on a Linux clone.

```bash
wsl.exe -d Ubuntu-bench -- bash -lc "bash ~/drm-tools/p2_baseline.sh"
```

Launch it as a BACKGROUND command from the Windows side and leave it to finish.
The whole run takes about five minutes, because this suite executes in eleven
seconds. The script refuses to start if `NUSCENES_ROOT` is set, so the
portfolio's order gate cannot be crossed by accident.

Per-module work uses the same toolkit:

```bash
export MUT_CLONE=bcl-mut
bash ~/drm-tools/check_module.sh geometry.quaternions
```

That prints one verdict per surviving mutant in the module, against the working
tree's current tests. Run it before writing a test to record the RED baseline
and again afterwards for the GREEN result; every kill claimed below was
verified that way rather than argued.

The scripts live outside this repository, because they are operator tooling
rather than product code and they read a private working tree.

## The environment the score is measured in

**`uv sync --frozen --all-groups --extra train` is required, and leaving out
the extra silently corrupts the score.** `correctors/learned.py` is inside
`only_mutate`, and every test that exercises it calls
`pytest.importorskip("torch")`. Without the extra those tests skip, mutmut
finds no test covering that module, and its 142 mutants land in the denominator
marked "no tests" while never being run. The first baseline taken here was
measured that way and was discarded.

One test file is excluded from the sandbox, for a structural reason rather than
for convenience: `tests/contract/test_p1_envelope_conformance.py` SCANS `src/`
for any mention of P1's package, and mutmut rewrites `src/` to build mutants,
so inside the sandbox that test describes the sandbox rather than the project.
It exercises no mutable code path, so the exclusion costs no killing power.

## The score

### At `215d606`, run `2026-09-04T04:3xZ`

| | Count | Share |
| --- | ---: | ---: |
| Mutants generated | 1,703 | |
| **Killed by a test** | **1,567** | **92.01%** |
| Timed out | 22 | |
| Survived | 114 | 6.69% |

**The gate is cleared on kills alone, by 34 mutants.** 90% of 1,703 is 1,533
and the suite kills 1,567. Counting the timeouts as detections gives 93.31%;
that figure is reported for completeness and is not what the gate rests on. All
but two of the timeouts are in the learned corrector, where a mutated model
makes a torch test run long rather than fail.

The first honest measurement of this core was 1,476 kills, 86.67%. The 91
additional kills came from tests written for their own sake, each verified
against the specific mutant before and after.

## What the survivors taught

Two findings mattered more than the score.

### `np.asarray(x, dtype=np.float64)` is a promotion, not a restated default

The sibling project documents a family of "dtype arguments that restate what
numpy would infer anyway" and counts `np.asarray` among them. That is wrong for
`np.asarray`, and this repository is where it was caught: the mutant nulling
the dtype in `five_channel_stem_weight` was KILLED, because a float32 stem
weight is exactly what that line exists to promote. `np.asarray` infers its
dtype from the CALLER's array, so the argument is a real conversion whenever
that array is not already the target type.

The wider consequence was better than a corrected family. Those mutants
surviving at other call sites meant **no test ever passed a non-float64 array
to those entry points**, and callers get float32 from torch and from the
nuScenes devkit routinely. Without the promotion the whole computation runs in
single precision and drifts by about 2e-08: far too small to fail any tolerance
in this suite, and far too large for a value stored in an artifact and compared
by hash. Quaternion normalisation, camera projection, SE3 transforms, the edge
score and the reprojection percentiles now each assert that a float32 input
gives exactly what the promoted call gives.

### A rotation matrix cannot survive a round trip through float32

Rounding a rotation to single precision moves its singular values by a few
parts in 1e9, which is outside the 1e-9 orthonormality tolerance this project
declares. `matrix_to_quaternion` therefore refuses a float32 matrix, and that
is the right behaviour: casting has already destroyed the property being
validated. A caller holding a float32 rotation must recompose it from a
quaternion. The refusal is now pinned by a test, so loosening the tolerance to
accommodate float32 would be a visible decision rather than a quiet one.

## Survivors that gained a test instead

Mutation testing pointed at real gaps rather than only at noise.

- **The conversion branch was chosen but never checked.** `matrix_to_quaternion`
  uses Shepperd's method, whose four branches are algebraically equal wherever
  more than one is valid. A mis-selected branch therefore returns the right
  answer with worse conditioning, which is why every approximate test passed
  under a mutated branch condition. They are not equal to the last bit, and
  quaternions here are stored in artifacts compared by hash, so the branch
  selection is pinned by exact-value assertions: one matrix per branch, plus a
  trace of exactly zero and a tie between the second and third diagonal
  entries.
- **The orthonormality tolerance was never the thing being tested.** A uniform
  scaling moves the determinant three times as far as the singular values, so
  the right-handedness check fires first and the tolerance is never reached.
  Post-multiplying by `diag(1+d, 1, 1/(1+d))` leaves the determinant at exactly
  one while shifting two singular values by `d`, which is what finally
  exercised it — and showed that dropping either `np.allclose` argument admits
  a matrix ten times further from orthonormal than the declared bound.
- **The resampled scene indices were invisible.** `BootstrapInterval` records
  only the seed and the resample count, so a reader reproduces the interval by
  re-deriving the draw. Every part of that fill loop — where the counter
  starts, how it steps, whether the fill count is added to or assigned — changes
  which scenes are drawn while leaving a well-formed array of indices in range.
  The stream is now pinned directly.
- **The interval's tails were never compared to each other.** The bound is
  `(1 - confidence) / 2` at each end; multiplying instead sends the lower cut
  past the upper one, and dividing by three leaves an interval wider than the
  confidence claims. Neither shows in the recorded `confidence` field, which is
  copied through untouched.
- **Every image bound was checked by one test that could not separate them.**
  The four clauses are joined by `or`, so a point over any single edge is out;
  turning one into an `and` stops that edge being checked at all. Both the
  rasteriser and the edge scorer now test each edge alone, including the pixel
  exactly at the width and at the height.
- **Boundary values that a real cohort contains.** A pedestrian is about 0.6 m
  across, so a box-dimension guard written `> 1.0` would refuse most of the
  vulnerable road users the study exists to measure. A one-pixel canvas, a
  one-pixel focal length, a single resample, a single evaluation and a starting
  point exactly on its bound are all legitimate inputs that an off-by-one bound
  would refuse while naming the input as the fault.

- **A pitch of exactly a quarter turn.** Roll and yaw stop being separable
  there, and a naive decomposition returns zeros for both while looking
  entirely reasonable. The guard refuses it; the test covers both sides of the
  bound, because 89.9 degrees is a legitimate fault however extreme and an
  off-by-one there would refuse it.
- **Which shape check refused an input.** Two separate checks run in the
  five-channel builder — the colour tensor's own rank and channel count, then
  the depth and validity maps against the image it describes — and a bare
  `pytest.raises(ValueError)` could not tell them apart. A caller who cropped
  the depth map needs to be sent to the depth map, not to the channel count.

## Survivors still outstanding

114 survive, and they are recorded rather than hidden. Only three match an
equivalence family that has been argued and checked; the rest are unexplained
and would each need either a killing test or an argument that survives the
falsification check described below.

The largest single group is nine mutations of one line: the branch condition
`values[0, 0] > values[1, 1] and values[0, 0] > values[2, 2]` in
`matrix_to_quaternion`. Three half-turn axes were constructed that flip every
one of those nine conditions, and only the `and`-to-`or` mutation died. The
reason is the one given above: where both branches are well conditioned they
agree to the last bit, and no admissible rotation matrix was found that makes
the wrongly selected branch degenerate. They are recorded as outstanding rather
than claimed as equivalent, because "no such matrix was found" is not a proof
that none exists.

## What anchoring an error message did not buy

A `pytest.raises(..., match="finite")` cannot notice a reworded message, and
mutmut pads a plain string literal to `"XXfiniteXX"`, which that pattern still
matches. The argument is correct, and it predicts kills that do not exist here.

Every loose `match=` in this suite was anchored on the whole message at
`0ca959d`, and the score did not move: 114 survivors and 22 timeouts before and
after, on a clone taken from that commit. Two facts explain it. mutmut 3.7 does
not mutate f-strings, and almost every message in this core is an f-string that
names the offending value - `geometry/quaternions.py` raises seven and only two
are plain literals. The whole mutated core yields 14 padded-message mutants,
and every one of them was already killed before the anchoring began.

The anchoring was kept anyway, for something the score cannot see. In
`training/engine.py`, `match="calibration"` also matches three other refusals
that module raises - a calibration cohort of the wrong size, a NaN calibration
loss, and a development cohort sharing a scene with the calibration cohort - so
the test that meant to check a manifest playing the wrong part would have
passed had any of those fired instead. An assertion that cannot tell four
contracts apart is asserting less than the test claims, and no mutation score
reports that. Each row of that test now names the slot and the declared role
its own refusal reports.

## How an equivalence claim is kept honest

An equivalent mutant cannot be killed. So a family that claims a mutant which
mutmut actually killed is wrong, and `audit_families.py` checks exactly that:
it scans every mutant in the tree, not only the survivors, and reports any
contradiction. That check must be empty before any equivalence count is
trustworthy. It currently reports three claimed and zero contradictions here.

The check earns its place. Run against the sibling project it found 52
contradictions across five families on its first pass, from three causes worth
naming: a `dtype=` argument is not always redundant, a padded string literal is
not a nulled argument, and a dropped argument shifts the following line so that
mutmut's reported before-and-after can name two unrelated arguments.
