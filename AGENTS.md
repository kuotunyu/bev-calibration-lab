# Working in this repository

## Non-negotiables

1. Nothing licensed or private enters Git: no nuScenes sensor data, no tokens, no credentials,
   no model weights, no private handoff. `uv run python -m bevcalib.private_guard` decides,
   and it reads the Git index rather than the working tree because the index is what a commit
   captures.
2. The private handoff lives outside this repository, at
   `..\handoff\bev-calibration-lab.md`. Update it at the start of a task, at RED, at GREEN,
   before and after any long job, and before handing over. If it disagrees with real Git
   state, stop and reconcile before doing anything destructive.
3. Every change is test-driven: write one failing test, run it, confirm it fails because the
   behavior is missing rather than because of a typo, then write the smallest code that
   passes. Code written before its test is deleted, not adapted.
4. `uv run python -m bevcalib.dev verify` must exit 0 before any local commit. It runs eight
   stages in a fixed order and stops at the first failure.
5. First-party code holds 100% statement and branch coverage. `pragma: no cover`, coverage
   omits and deleting tests to reach the number are all forbidden. If a line is hard to
   cover, that is usually the design talking.
6. A local commit is a checkpoint. Creating a GitHub repository and pushing each need explicit
   human authorization at the time of the action.

## The coordinate contract

This is the one thing that breaks silently and stays broken.

- Every transform is named `T_target_source` and maps points expressed in `source` into
  `target`. `compose(T_a_b, T_b_c)` gives `T_a_c`.
- Public point arrays are `[N, 3]` row vectors. Column-vector conventions stay inside an
  adapter and never appear in a public signature.
- The formal chain is LiDAR sensor to LiDAR-time ego to global to camera-time ego to camera
  sensor. Two ego poses, because the sweep and the exposure happen at different instants;
  collapsing them into one is the most common way to get a plausible, wrong answer.
- 3D boxes are natively in global coordinates.
- LiDAR features keep `x, y, z, intensity, ring`.

## Order gate

P1 `driving-risk-metrics` released v1.0.1 on 2026-09-06; the P2 data order gate is
open. The sole active workspace plan is
`docs/superpowers/plans/2026-09-06-three-releases-master-plan.md`. Its Task F may
proceed independently under section 0.3 after P3 E3 local acceptance while E2 human
replay and E4 public release remain pending. Follow the current task boundaries:
F does not authorize a real cohort freeze, formal training, Colab or Task G.
P2 public Git operations remain gated on its own release-time human authorization.
Nothing here imports `drivemetrics` or depends on P1 at runtime; the shared artifact
envelope is a specification each repository implements independently.

## Evidence vocabulary

A claim is labelled `observed`, `derived`, `synthetic` or `illustrative`, and an `observed`
number must be reproducible from a committed artifact. Locked evaluation scenes are never
used to choose a hyperparameter, a threshold, a checkpoint or a figure.

## Commands

```bash
uv sync --frozen                              # locked environment
uv run bev-calib --help                       # command-line entry point
uv run pytest -q                              # tests
uv run python -m bevcalib.dev verify          # every gate, stops at the first failure
```
