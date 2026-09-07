# nuScenes dataset readiness

Machine-generated from [the safe readiness artifact](nuscenes-preflight.json).
This record establishes input integrity, geometry parity and cohort separation. It contains no calibration-recovery or downstream study result.

## Archive verification

Every listed archive completed gzip verification and independent installed-file checks before its exact removal. Existing matching mini files and maps were reused.

| Archive | Selected files | Selected bytes | Archive SHA-256 |
|---|---:|---:|---|
| v1.0-trainval_meta.tgz | 17 | 2607648154 | `db48746b10e3544d5ef619eaa3d687e3960626fe1b4422ed856711da5aa7325b` |
| v1.0-trainval01_keyframes.tgz | 6752 | 2852373183 | `37f61460e6b1f1d9a02d3147a5557a5fa654e33752beb1d1cc711e0b0fb1b6e1` |
| v1.0-trainval02_keyframes.tgz | 6734 | 2803849263 | `1dcc10c082220e78b9a694f7e9a69b0174f7b4bb5f58643e7f9737bdce8a0348` |
| v1.0-trainval03_keyframes.tgz | 6754 | 2815453883 | `caa143a4895a540ae54f0b2a4c1b7b5defa1f641962aa2e6ebc87f45aea40227` |
| v1.0-trainval04_keyframes.tgz | 6818 | 2881849247 | `f968f8fdcf72798d87eccc880c888985e2ef5fa29572e088c55b19c71083fd81` |
| v1.0-trainval05_keyframes.tgz | 6890 | 2804386506 | `fe1dcdf1acf15691692952f7afc8691f034efa52ce61c49abb235a5f6db5d941` |
| v1.0-trainval06_keyframes.tgz | 6888 | 2779755582 | `7af5ffc6ed7b4abdaaa153180af13218fda2bc031bf4bb4c7be68e2c993239dc` |
| v1.0-trainval07_keyframes.tgz | 6882 | 2835872137 | `5b93fb90fcb87c14648e9b2a100cc32e17c0c1a00d322b14ee812d54118a48d5` |
| v1.0-trainval08_keyframes.tgz | 6864 | 2837508385 | `df239077d653c6713172eca928c1cb0eeb70b03e43a979f592bff15781e2b427` |
| v1.0-trainval09_keyframes.tgz | 6876 | 2946150195 | `35fb8c5b04825ad382c6a10fbb9e011511ba7072442cf17b0caf0aa2454461ba` |
| v1.0-trainval10_keyframes.tgz | 6840 | 3245722365 | `0657ec5daf0b285070c9dd329dded2f6abe5bca45150e8dbb68655e24dc25403` |

## Production preflight

Original trainval metadata resolves 850 scenes and 34149 paired samples. Missing required paired keyframes: 0. Missing optional sensor payloads: 2532383.

Timing fixes the camera and uses actual available LiDAR packets. The keyframe packs do not provide general sweep coverage; preserved mini payloads can contribute only where actually present.

| Requested offset (ms) | Total | Valid | Invalid | Valid fraction | Nearest error min/max (ms) |
|---:|---:|---:|---:|---:|---|
| -200 | 34149 | 393 | 33756 | 0.011508389703944478 | 8.359 / 242.542 |
| -100 | 34149 | 394 | 33755 | 0.01153767313830566 | 10.301 / 142.542 |
| -50 | 34149 | 392 | 33757 | 0.011479106269583296 | 9.96 / 92.542 |
| 0 | 34149 | 394 | 33755 | 0.01153767313830566 | 9.459 / 42.542 |
| 50 | 34149 | 34149 | 0 | 1.0 | 7.458 / 17.436 |
| 100 | 34149 | 394 | 33755 | 0.01153767313830566 | 9.765 / 67.436 |
| 200 | 34149 | 393 | 33756 | 0.011508389703944478 | 10.036 / 167.436 |

Table byte identities:

| Table | SHA-256 |
|---|---|
| calibrated_sensor | `67781a5dd7b2504b046ef89d6dcb267b12d1cc91af21f1ec5758624588e99865` |
| ego_pose | `be12bd501f694b344628ba0680a37d6e1ec83e62b37f310c205c1dbe41cd09ff` |
| log | `774c8256554a5569f64660e5f450394ce7e3a82168edfab530edab81f9ae7edc` |
| sample | `6035ac58b6e971622be2bb1be15b917e7cb4e05ae984d6b339c27c1699c4ad9d` |
| sample_annotation | `1f4f3835cb86f4efe49ccad3ae6f5f6a2d068422696de564c5892b5b06bea2a9` |
| sample_data | `6dcad49f0b9bd7b1cef04e2a0ff2ac2879b46b938e8d48153f00693cebb4de21` |
| scene | `bdc3211bfbfda652d5e9dab73c45c7132925ca84f869bf45e8e14edeafb1d20b` |
| sensor | `4d5c96570e2d8b09b88ce4c605e40ee43c4909f97bea5741185f18f92eb491ae` |

## Official-devkit mini parity

All 4 pinned cases passed with absolute tolerance 1e-06 in the component units below and relative tolerance 0. Cases did not skip.

Camera-frame XYZ and optical depth compare every input point; projection checks require exact validity-mask equality. Pixel coordinates compare only the nonempty valid-UV subset. The point and UV denominators are therefore different.

| Pinned index | Total points | Valid UV count | Camera XYZ max (m) | Depth max (m) | UV max (pixels) |
|---:|---:|---:|---:|---:|---:|
| 0 | 34688 | 3067 | 6.394884621840902e-13 | 3.268496584496461e-13 | 1.6348167264368385e-10 |
| 1 | 34720 | 2977 | 1.2221335055073723e-12 | 1.2221335055073723e-12 | 2.781490593406488e-10 |

| Box-center pinned index | Compared boxes | Maximum center error (m) |
|---:|---:|---:|
| 0 | 48 | 4.973799150320701e-13 |
| 1 | 50 | 1.0231815394945443e-12 |

## Frozen cohort

Protocol identity: `ad119b37b47d17ddf087f0727a62eb8a50f049707f71620d939da4d1907ac877`. Each role was regenerated into a fresh directory with byte-identical results. Whole logs and all observation identifiers are disjoint between roles.

| Role | Scenes | Distinct logs | Samples | Official split | Manifest SHA-256 |
|---|---:|---:|---:|---|---|
| calibration | 20 | 20 | 804 | train | `d989bece59f20ba1b9da05725d9561a6a0129d8ed36354386906b28b20fa0d3d` |
| development | 100 | 26 | 4020 | train | `f52feff8d03ea19c27ff765716b90c708b869dd903f4d55ee412544496a101f7` |
| evaluation | 30 | 13 | 1207 | val | `dc0315639e6e107d152220db83b729819cfce6701d0f4e84963ac15c9fabd847` |

Location allocation:

| Role | Location | Requested | Selected | Available scenes | Available logs |
|---|---|---:|---:|---:|---:|
| calibration | boston-seaport | 11 | 11 | 390 | 22 |
| calibration | singapore-hollandvillage | 2 | 2 | 70 | 4 |
| calibration | singapore-onenorth | 4 | 4 | 148 | 17 |
| calibration | singapore-queenstown | 3 | 3 | 92 | 7 |
| development | boston-seaport | 56 | 56 | 390 | 22 |
| development | singapore-hollandvillage | 10 | 10 | 70 | 4 |
| development | singapore-onenorth | 21 | 21 | 148 | 17 |
| development | singapore-queenstown | 13 | 13 | 92 | 7 |
| evaluation | boston-seaport | 15 | 15 | 77 | 8 |
| evaluation | singapore-hollandvillage | 3 | 3 | 15 | 1 |
| evaluation | singapore-onenorth | 7 | 7 | 35 | 6 |
| evaluation | singapore-queenstown | 5 | 5 | 23 | 3 |

## Reproduction and evidence scope

Use the licensed dataset root in the process-local `NUSCENES_ROOT` environment variable. These PowerShell commands operate on local private inputs:

```powershell
uv run --frozen bev-calib data preflight --dataroot "$env:NUSCENES_ROOT" --version v1.0-trainval --output artifacts/preflight/trainval.json
uv run --frozen pytest tests/integration/test_nuscenes_mini_parity.py -m slow -s
uv run --frozen bev-calib cohort freeze --dataroot "$env:NUSCENES_ROOT" --version v1.0-trainval --protocol configs/protocols/nuscenes_calibration_v1.yaml --output-dir artifacts/manifests/nuscenes_calibration_v1
```

Repeat freeze with a fresh output directory and compare all role bytes. Missing-root and swapped calibrated-sensor/ego-pose lookup rehearsals refused their inputs before output creation. The swap rehearsal tests those unresolved fields; it does not detect every numerically valid inverse transform.

Private raw receipts, removal dispositions, logs, manifests and generator sources are retained under the ignored `artifacts/preflight/extraction/` evidence directory. The JSON companion binds their hashes and includes complete invalidity reasons and allocation diagnostics. Original sensor IDs, absolute data paths, media and weights are excluded from this public record.

See the [dataset card](../dataset-card.md) and [coordinate contract](../coordinate-contract.md) for the measurement scope and the flat-plane residual at zero extrinsic fault.
