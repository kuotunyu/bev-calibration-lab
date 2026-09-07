# RED baseline: `verifying-nuscenes-transforms`

The baseline was preserved before the skill, validator, or contract test existed.
Platform thread limits required reuse of an agent that had completed an unrelated P3
review, so this was a new bounded task rather than a fully fresh agent context.

The agent correctly delegated to the repository's typed geometry and documented the
four directed links, separate LiDAR/camera timestamps, public `[N, 3]` rows, and separate
front/image/valid masks. There was no transform omission or reversal, and this record does
not invent one.

The concrete failure was at the native adapter boundary. The draft read
`sample["data"]["CAM_FRONT"]`, an interface supplied by the official devkit. P2's
`NuScenesInstallation` retains the raw sample table and derives channel pairs through
`scene_records()` and indexed `sample_data` rows. Running the draft's function body
unchanged against the existing synthetic native installation reached that lookup and
failed with `KeyError: 'data'` at actual exit 1.

Two earlier check attempts are retained privately as harness failures: one placed a
future import after an extra module string, and one tried to import the namespace-style
test module directly. Neither is product or baseline evidence.

The validator contract then began with two REDs. The absent implementation produced 15
failures. A minimal permissive trace boundary exposed four intended DID-NOT-RAISE failures:
reversed direction, missing camera timestamp, a mismatched frame name, and a parity gap
just over `1e-6`. The required top-level behind-camera record was already refused when
absent. These failures define the strict validator implemented by the skill.
