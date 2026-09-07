# GREEN: `verifying-nuscenes-transforms`

The skill now requires an evidence trace rather than accepting a prose-only transform
claim. Its thin command loads that trace into a strict first-party model which refuses
unknown fields, reversed or mismatched edges, a missing sensor timestamp, the wrong public
point shape, an absent or ineffective behind-camera example, an incomplete or empty parity
set, and any tolerance other than absolute `1e-6` with zero relative tolerance.

The same reused agent applied the finished skill to the baseline projection task. This was a
distinct forward role, but the platform thread limit meant it was not a fully fresh agent
context. The forward draft retained the correct four-link chain and changed the native lookup
to `NuScenesInstallation.scene_records()`. It marked all runtime evidence as pending instead
of fabricating results.

The forward function body was then run unchanged against the same valid synthetic native
installation used for the meaningful baseline check. The baseline body had failed at actual
exit 1 with `KeyError: 'data'`; the forward body exited 0 and produced two projected rows,
optical depths 10 and 20, LiDAR/camera timestamps 1000000 and 1020000, and true front,
image, and valid masks for both rows. A separately executed camera point `[0, 0, -2]`
projected numerically inside the image while producing `in_front=false`, `in_image=true`,
and `valid=false`. One earlier forward harness attempt is retained privately: a hand-written
expected float used a stricter `1e-12` comparison and missed the fixture's float32 value by
about `2.98e-10`; correcting only that harness literal produced the successful run.

The validator contract passes all 16 tests. The actual mini parity run executes both operators
on both pinned samples with the official devkit primitive, the same native camera intrinsic,
the repository's declared mask policy, and `atol=1e-6, rtol=0`. All-point camera-frame and
optical-depth arrays agree, and the exact masks agree before UV is compared over valid pixels:

| case | total points or boxes | valid UV | point max (m) | UV max (px) | depth/center max (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LiDAR projection sample 0 | 34688 | 3067 | 6.3948846218409017e-13 | 1.6348167264368385e-10 | 3.2684965844964609e-13 |
| LiDAR projection sample 1 | 34720 | 2977 | 1.2221335055073723e-12 | 2.7814905934064882e-10 | 1.2221335055073723e-12 |
| box centers sample 0 | 48 | n/a | n/a | n/a | 4.9737991503207013e-13 |
| box centers sample 1 | 50 | n/a | n/a | n/a | 1.0231815394945443e-12 |

The UV claim is limited to the 3067 and 2977 pixels selected by the agreed valid masks. The
camera-frame point and optical-depth claims cover all 34688 and 34720 LiDAR points. All four
comparisons are nonempty and remain below the declared tolerance.
