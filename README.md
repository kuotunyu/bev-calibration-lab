# bev-calibration-lab

[English](README.en.md) · [網站](https://kuotunyu.github.io/bev-calibration-lab/) · [文件入口](docs/README.md) · [研究卡](docs/experiment-card.md) · [模型卡](docs/model-card.md) · [已知問題](docs/errata.zh-TW.md)

LiDAR 與相機之間的外參標定只要偏一點點，代價有多大？修正器又能救回多少？這是一個在 nuScenes 上進行的受控故障研究。English readers: see the [English README](README.en.md).

## 重點摘要

在 30 個鎖定的 nuScenes validation 場景上，CAM_FRONT–LIDAR_TOP 外參只要有 +1° 的俯仰（tilt）或偏擺（pan）誤差，投影的 LiDAR 點就會偏移 22.44–23.95 px（每個 frame 取中位數後的場景平均）。 <!-- bind: 30 = metrics#/runs/identity/roll:1/pixel_frame_p50_px/support/scenes ; 22.44 = metrics#/runs/identity/roll:1/pixel_frame_p50_px/value ; 23.95 = metrics#/runs/identity/pitch:1/pixel_frame_p50_px/value -->
ConvNeXtV2-Tiny 修正器（三個 seed 的平均）在故障矩陣的全部 60 個條件中（54 個單軸故障，加上在每個軸各列一次的零故障），旋轉、平移、像素誤差與 recovery 都勝過單張影像的邊緣對齊最佳化器；但 seed 17 與 73 不論故障大小都留下 0.62–0.83° 的旋轉殘差，因此和「不修正」相比，它只有在 ±1° 以上的俯仰或偏擺、±2° 的影像平面內旋轉、±0.2 m 的橫向或垂直偏移，才以配對 95% 區間確認降低像素誤差，還會把原本正確的標定改動 0.60°（+10.55 px）。 <!-- bind: 60 = envelope#/interval_counts/classical->learned-fixed-three-seed-mean/rotation_geodesic_deg/after_better ; 60 = envelope#/interval_counts/classical->learned-fixed-three-seed-mean/translation_norm_cm/after_better ; 60 = envelope#/interval_counts/classical->learned-fixed-three-seed-mean/pixel_frame_p50_px/after_better ; 60 = envelope#/interval_counts/classical->learned-fixed-three-seed-mean/recovery_rate_pct/after_better ; 0.62 = envelope#/residual/learned-73/rotation_geodesic_deg/min ; 0.83 = envelope#/residual/learned-17/rotation_geodesic_deg/max ; 1 = envelope#/break_even/identity->learned-fixed-three-seed-mean/roll/magnitude ; 1 = envelope#/break_even/identity->learned-fixed-three-seed-mean/pitch/magnitude ; 2 = envelope#/break_even/identity->learned-fixed-three-seed-mean/yaw/magnitude ; 0.2 = envelope#/break_even/identity->learned-fixed-three-seed-mean/x/magnitude ; 0.2 = envelope#/break_even/identity->learned-fixed-three-seed-mean/y/magnitude ; 0.60 = intervals#/comparisons/identity->learned-fixed-three-seed-mean/pitch:0/rotation_geodesic_deg/after ; 10.55 = intervals#/comparisons/identity->learned-fixed-three-seed-mean/pitch:0/pixel_frame_p50_px/after ; 60 = envelope#/grid/conditions ; 54 = envelope#/grid/injected_fault_conditions -->
推論（本研究未驗證）：線上重新標定需要在修正器前面加上標定異常偵測或放棄修正的閘門。

## 主要發現

下表是 pixel P50：以真實標定與各方法最後採用的標定（identity 即為有故障的標定）分別投影 LiDAR 點，每個 frame 取偏移量的中位數，再做場景平均，單位為像素，越低越好。故障名稱見[故障軸對照表](#故障與故障軸)；零故障列為所有軸共用。負向故障、其他指標與所有配對區間見[完整 evidence 報告](https://kuotunyu.github.io/bev-calibration-lab/evidence/)。

<!-- bind-table: metrics#/runs/{column}/{row}/pixel_frame_p50_px/value -->
| 故障 | `identity` | `classical` | `learned-17` | `learned-42` | `learned-73` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 無故障 `pitch:0` | 0.00 | 19.46 | 11.70 | 7.94 | 12.01 |
| 俯仰 0.5° `roll:0.5` | 11.23 | 17.68 | 11.89 | 7.78 | 11.91 |
| 俯仰 1° `roll:1` | 22.44 | 17.86 | 11.61 | 6.76 | 12.11 |
| 俯仰 2° `roll:2` | 44.87 | 25.30 | 12.24 | 6.69 | 12.03 |
| 偏擺 0.5° `pitch:0.5` | 11.99 | 22.75 | 13.43 | 15.89 | 11.31 |
| 偏擺 1° `pitch:1` | 23.95 | 22.12 | 12.57 | 27.62 | 10.67 |
| 偏擺 2° `pitch:2` | 47.78 | 37.02 | 17.38 | 52.82 | 13.80 |
| 平面內旋轉 2° `yaw:2` | 15.41 | 24.61 | 12.33 | 7.79 | 13.09 |
| 橫向 0.2 m `x:0.2` | 21.74 | 25.50 | 14.29 | 12.94 | 12.49 |
| 垂直 0.2 m `y:0.2` | 21.95 | 25.09 | 11.84 | 8.04 | 11.88 |
| 前向 0.2 m `z:0.2` | 6.99 | 21.67 | 11.93 | 8.61 | 12.75 |

- **Learned 與 classical。** 三個 seed 的 learned 平均在 60 個條件中的 60 個，旋轉誤差、平移誤差、pixel P50 與 P90 都較低、recovery 較高（配對 95% 場景 bootstrap 區間完全大於零）。Classical 只在它自己最佳化的目標，也就是邊緣對齊分數上 60 個條件全勝，可見這個代理目標並不反映真實 pose。 <!-- bind: 60 = envelope#/interval_counts/classical->learned-fixed-three-seed-mean/pixel_frame_p90_px/after_better ; 60 = envelope#/interval_counts/classical->learned-fixed-three-seed-mean/edge_score_px/before_better -->
- **Learned 與「不修正」。** 以 pixel P50 來看，learned 平均只在 60 個條件中的 14 個勝過 identity（±1° 與 ±2° 的俯仰和偏擺、±2° 平面內旋轉、±0.2 m 橫向與垂直偏移，前向偏移則從未勝過），在 40 個條件較差（含在每個軸各計一次的零故障）、6 個無法區分。Classical 只在 60 個條件中的 4 個勝過 identity。 <!-- bind: 14 = envelope#/interval_counts/identity->learned-fixed-three-seed-mean/pixel_frame_p50_px/after_better ; 60 = envelope#/interval_counts/identity->learned-fixed-three-seed-mean/pixel_frame_p50_px/total ; 1 = envelope#/break_even/identity->learned-fixed-three-seed-mean/roll/magnitude ; 1 = envelope#/break_even/identity->learned-fixed-three-seed-mean/pitch/magnitude ; 2 = envelope#/break_even/identity->learned-fixed-three-seed-mean/yaw/magnitude ; 0.2 = envelope#/break_even/identity->learned-fixed-three-seed-mean/x/magnitude ; 0.2 = envelope#/break_even/identity->learned-fixed-three-seed-mean/y/magnitude ; 40 = envelope#/interval_counts/identity->learned-fixed-three-seed-mean/pixel_frame_p50_px/before_better ; 6 = envelope#/interval_counts/identity->learned-fixed-three-seed-mean/pixel_frame_p50_px/inconclusive ; 4 = envelope#/interval_counts/identity->classical/pixel_frame_p50_px/after_better -->
- **殘差下限。** 在全部 60 個條件中，seed 17 的旋轉殘差維持在 0.63 到 0.83°，seed 73 維持在 0.62 到 0.74°；三個 seed 的平移殘差介於 2.40 與 4.05 cm。比這個下限小的故障，修正後反而比修正前更差；弱恢復與零故障退步都與這個現象一致。 <!-- bind: 60 = envelope#/grid/conditions ; 0.63 = envelope#/residual/learned-17/rotation_geodesic_deg/min ; 0.83 = envelope#/residual/learned-17/rotation_geodesic_deg/max ; 0.62 = envelope#/residual/learned-73/rotation_geodesic_deg/min ; 0.74 = envelope#/residual/learned-73/rotation_geodesic_deg/max ; 2.40 = envelope#/residual/learned-73/translation_norm_cm/min ; 4.05 = envelope#/residual/learned-42/translation_norm_cm/max -->
- **零故障。** 沒有注入故障時，learned 平均把標定改動 0.60° 與 2.58 cm，像素誤差增加 10.55 px（改善量，即 identity 減 learned 的配對 95% 區間：-11.72 到 -9.47 px）；classical 改動 1.08° 與 12.55 cm，增加 19.46 px。 <!-- bind: 0.60 = intervals#/comparisons/identity->learned-fixed-three-seed-mean/pitch:0/rotation_geodesic_deg/after ; 2.58 = intervals#/comparisons/identity->learned-fixed-three-seed-mean/pitch:0/translation_norm_cm/after ; 10.55 = intervals#/comparisons/identity->learned-fixed-three-seed-mean/pitch:0/pixel_frame_p50_px/after ; -11.72 = intervals#/comparisons/identity->learned-fixed-three-seed-mean/pitch:0/pixel_frame_p50_px/interval/low ; -9.47 = intervals#/comparisons/identity->learned-fixed-three-seed-mean/pitch:0/pixel_frame_p50_px/interval/high ; 1.08 = intervals#/comparisons/identity->classical/pitch:0/rotation_geodesic_deg/after ; 12.55 = intervals#/comparisons/identity->classical/pitch:0/translation_norm_cm/after ; 19.46 = intervals#/comparisons/identity->classical/pitch:0/pixel_frame_p50_px/after -->
- **Seed 42 不修正偏擺。** 在 -2° 與 +2° 的偏擺故障下，它的偏擺殘差是 1.99° 與 2.04°，零故障時則是 0.04°。細節見 [operating envelope 說明](docs/analysis/operating-envelope.md)。 <!-- bind: 1.99 = envelope#/pan_residual/learned-42/pitch:-2 ; 2.04 = envelope#/pan_residual/learned-42/pitch:2 ; 0.04 = envelope#/pan_residual/learned-42/pitch:0 -->

![各相機軸的 pixel P50 對注入故障大小，包含 identity、classical 與三個 learned seed，並標出與不修正相比的損益平衡點](docs/analysis/operating_envelope_v1/operating-envelope.svg)

上圖與上述計數是對已發布 evidence 的衍生分析（derived），記錄自己的內容摘要與所讀每份來源文件的 SHA-256，見 [operating envelope](docs/analysis/operating-envelope.md)。正式的 [Recovery 圖](https://kuotunyu.github.io/bev-calibration-lab/figures/recovery-by-fault-level.svg)與 [BEV 誤差圖](https://kuotunyu.github.io/bev-calibration-lab/figures/bev-error-by-range.svg)保留每個方法、seed 與條件，在圖上停留即可看到精確數值與來源。引用 ±0.25° 的 identity recovery、timing 壓力測試或 10 m 以外的 BEV 誤差之前，請先讀[已知問題](docs/errata.zh-TW.md)。

## 對自駕車的意義

相機與 LiDAR 融合依賴外參標定。支架會位移、感測器會更換、結構會老化，系統相信的標定可能逐漸偏離實際安裝，而且不會有任何錯誤訊息：LiDAR 深度只是落到了錯誤的像素上。用 SOTIF（ISO 21448）的語言，這種偏移可以當成融合錯誤的觸發條件（triggering condition）來分析。上面的結果支持監測標定、並對線上修正加上閘門，因為在這裡一直開著的修正器會把正確的標定改壞。這是研究性質的專案，不宣稱符合 ISO 21448、ISO 26262 或任何其他標準。

## 方法

### 感測器與 cohort

只用 `CAM_FRONT` 與 `LIDAR_TOP`。正式 cohort 是 150 個 nuScenes 場景：100 個官方 train 開發用、20 個來自不同 log、用於選 checkpoint 的校準場景（calibration split）、30 個官方 validation 保留給鎖定評估。場景分派依 location 分層並以 token SHA-256 排序，可完整重現，且與任何量測結果無關。nuScenes mini 只用於開發與整合測試，不會出現在任何回報結果中。

### 故障與故障軸

故障只改標定 metadata，不改任何像素或 LiDAR 點。每個故障都是單軸的剛體轉換，組合在 CAM_FRONT 外參的相機端（`assumed = true ∘ fault`）。因此正式的 roll、pitch、yaw 與 x、y、z 是相機光學座標系（x 向右、y 向下、z 向前）的軸，而不是車體座標的軸。

| 正式名稱 | CAM_FRONT 光學座標系中的軸 | 物理效果 | identity 在 1° 或 0.1 m 的 pixel P50 |
| --- | --- | --- | ---: |
| `roll` | x，向右 | 俯仰（tilt）：影像內容上下移動 | 22.44 px <!-- bind: 22.44 = metrics#/runs/identity/roll:1/pixel_frame_p50_px/value --> |
| `pitch` | y，向下 | 偏擺（pan）：影像內容左右移動 | 23.95 px <!-- bind: 23.95 = metrics#/runs/identity/pitch:1/pixel_frame_p50_px/value --> |
| `yaw` | z，光軸 | 影像平面內旋轉 | 7.74 px <!-- bind: 7.74 = metrics#/runs/identity/yaw:1/pixel_frame_p50_px/value --> |
| `x` | x，向右 | 橫向偏移 | 10.92 px <!-- bind: 10.92 = metrics#/runs/identity/x:0.1/pixel_frame_p50_px/value --> |
| `y` | y，向下 | 垂直偏移 | 10.98 px <!-- bind: 10.98 = metrics#/runs/identity/y:0.1/pixel_frame_p50_px/value --> |
| `z` | z，光軸 | 沿視線方向的前向偏移 | 3.57 px <!-- bind: 3.57 = metrics#/runs/identity/z:0.1/pixel_frame_p50_px/value --> |

旋轉故障的等級是 0、±0.1、±0.25、±0.5、±1、±2°；平移故障是 0、±2、±5、±10、±20 cm（[故障矩陣](configs/perturbations/formal_v1.yaml)）。正式的 `yaw` 是繞光軸的旋轉，不是航向誤差。[合成 explorer](https://kuotunyu.github.io/bev-calibration-lab/demo/calibration-explorer.html) 則用車體座標命名軸，頁面上有對照說明。另有一個只用 identity 的 timing 壓力測試，原本要把相機配上其他 LiDAR sweep；v1 幾乎在每個偏移都選到同一個 keyframe sweep，因此沒有提供資訊（見[已知問題](docs/errata.zh-TW.md#2-v1-的-timing-壓力測試沒有提供資訊)）。

### 方法與指標

- **Identity** 不改動標定，作為參考基準。
- **Classical** 是確定性、有界的由粗到細搜尋，在單張 frame 上調整六個參數，讓影像與 LiDAR 的邊緣對齊最大化。
- **Learned** 是 ConvNeXtV2-Tiny，讀入 RGB、投影深度與有效遮罩，回歸故障的逆轉換。Seeds 17、42、73 在開發場景訓練，各自在校準場景選擇 checkpoint，並分別回報。三個固定 seed 的平均是共同支持上的配對統計量，不是預測 ensemble。

指標包括 geodesic 旋轉誤差、平移誤差、pixel P50 與 P90、邊緣對齊分數、recovery（旋轉不超過 0.25° 且平移不超過 5 cm）以及依 GT 距離分 bin 的地面接觸點 BEV 誤差。frame 先在場景內平均，場景之間等權；配對 95% 區間來自 5,000 次場景 bootstrap，是逐點區間，不是同時區間。每個指標的定義見[分析契約](docs/contracts/formal-analysis.md)。

### 座標契約

這是這個問題裡最容易錯、而且錯了不會有人發現的地方，所以固定在一處，並由型別與測試強制。

- 每個轉換命名為 `T_target_source`，意義是把以 `source` 座標表示的點轉到 `target`。
- 公開的點陣列一律為 `[N, 3]`，每個點佔一個 row；內部的齊次 column-vector 運算只允許出現在 adapter 內，不得混入公開介面。
- 正式轉換鏈是 LiDAR sensor → LiDAR 時刻 ego → global → 相機時刻 ego → camera sensor。是兩個 ego pose 而不是一個，因為 LiDAR 掃描與相機曝光不在同一瞬間；把它們併成一個，會得到看起來合理但錯誤的答案。
- 3D 邊界框的原生座標系是 global。
- LiDAR 特徵保留 `x, y, z, intensity, ring`。

完整規則、列向量邊界、兩個 ego 時間戳與一個可手算驗證的數值範例，見 [docs/coordinate-contract.md](docs/coordinate-contract.md)。

## 驗證

### 核對已發布的數字（不需要 nuScenes，也不需要 GPU）

需要 Python 3.12 與 [uv](https://docs.astral.sh/uv/) 0.11.x，基本安裝就夠：

```bash
uv sync --frozen
uv run --frozen bev-calib audit-claims --claims docs/claims.yaml
```

正式報告顯示的每個數值都綁定在一份由機器檢查的 registry 中，共 90,126 筆 claim，每筆記錄 evidence 文件、JSON pointer、數值與文件摘要（見 [evidence 索引](docs/evidence/README.md)）。稽核會重新讀取五份 evidence 文件，檢查 schema、摘要與文件之間的一致性，並確認每筆 claim 仍等於 evidence。這份 README 的重點摘要、主要發現與故障軸表中的每個結果數字，也都帶有測試會檢查的隱藏綁定：正式數值對照 registry 中已驗證的 claim，衍生的計數與範圍對照 operating envelope 文件，並驗證該文件自身與來源文件的摘要。要從 evidence 重建 registry、含圖的報告與衍生的 operating envelope：

```bash
uv run --frozen bev-calib generate-claims --artifacts-dir docs/evidence/nuscenes_calibration_v1 --output artifacts/check/claims.yaml
uv run --frozen bev-calib report --formal --figures --claims docs/claims.yaml --artifacts-dir docs/evidence/nuscenes_calibration_v1 --output-dir artifacts/check/report
uv run --frozen python -m bevcalib.analysis.operating_envelope --artifacts-dir docs/evidence/nuscenes_calibration_v1 --output-dir artifacts/check/envelope
```

重建出的 `claims.yaml` 與 `docs/claims.yaml` 逐位元相同，envelope 檔案也與 `docs/analysis/operating_envelope_v1/` 中的相同。這些步驟不能證明的是：它們不會從原始 frame 重新計算任何統計量，那需要 nuScenes 與私有的逐列結果；[重現紀錄](docs/verification/analysis-reproduction.md)記載了兩次獨立分析得到逐位元相同的輸出。

### 開發與品質關卡

`train` extra（PyTorch、timm）只有開發時需要，因為學習式修正器是第一方程式碼，受覆蓋率關卡檢查。nuScenes 資料授權給帳號持有者，不在此散布，也永遠不會提交進版控。

```bash
uv sync --frozen --all-groups --extra train --extra report
uv run --frozen python -m bevcalib.dev verify
```

`bevcalib.dev verify` 以固定順序執行所有關卡，遇到第一個失敗就停止：私有檔案防護、格式檢查、lint、型別檢查、完整測試、第一方程式碼 100% 陳述與分支覆蓋、schema 契約、文件連結。沒有 `pragma: no cover`，也沒有排除的第一方路徑。原生資料、訓練與評估命令見[命令文件](docs/commands-and-report.md)。

### 核心邏輯在哪裡

- [`geometry/se3.py`](src/bevcalib/geometry/se3.py)：有型別的剛體轉換、`T_target_source` 組合與逆轉換。
- [`perturbations/apply.py`](src/bevcalib/perturbations/apply.py)：故障建構、source 端組合與精確逆轉換。
- [`correctors/classical.py`](src/bevcalib/correctors/classical.py)：有界的由粗到細邊緣對齊搜尋。
- [`metrics/bootstrap.py`](src/bevcalib/metrics/bootstrap.py)：以 SHA-256 counter 產生索引的配對場景 bootstrap。
- [`test_nuscenes_mini_parity.py`](tests/integration/test_nuscenes_mini_parity.py)：在 v1.0-mini 上與官方 nuScenes devkit 的一致性檢查（需要資料，CI 會略過；實際執行紀錄見 [preflight 紀錄](docs/verification/nuscenes-preflight.md#official-devkit-mini-parity)）。

## 限制

- 只有一個資料集、一台相機（`CAM_FRONT`）與 30 個鎖定評估場景；無法說明對其他相機、車輛或資料集的泛化。
- 故障只改 metadata 且只有單軸。真實的標定偏差可能同時涉及多軸，也可能伴隨其他感測器效應。
- Classical 基準是單張 frame 的邊緣對齊搜尋，只是傳統方法的下界；多 frame 或 targetless 方法可能表現更好。
- 只有三個固定的訓練 seed，分別回報；它們的平均不是 ensemble，區間也不涵蓋訓練隨機性。Seed 42 不修正偏擺。
- identity 在 ±0.25° 的 recovery 是浮點數邊界假象（[已知問題](docs/errata.zh-TW.md)）。
- v1 的 timing 壓力測試沒有提供資訊（[已知問題](docs/errata.zh-TW.md)）。
- BEV 誤差依賴平面地面假設，在零故障就留下殘差，而且 10 m 以外的 bin 由條件很差的射線主導；只解讀 0-10 m（[已知問題](docs/errata.zh-TW.md)）。
- 區間是各條件的逐點區間，不是同時區間。
- 弱恢復的原因尚未完全確定；[有界診斷](docs/verification/corrector-diagnostics.md)排除了部分實作問題，但不是全部。
- 這裡量測的是標定幾何，不是偵測器表現、閉迴路行為或實車安全。

## 發布與文件

v1.0.0 已[發布](https://github.com/kuotunyu/bev-calibration-lab/releases/tag/v1.0.0)，evidence 已凍結：[五份 evidence 文件](docs/evidence/README.md)完整保存，兩次獨立 CPU 分析得到逐位元相同的結果。之後的文件更新不改變已 tag 的原始碼與 evidence。[報告與互通重現紀錄](docs/verification/publication-and-interchange.md)說明展示驗收與發布核對，[文件入口](docs/README.md)列出其餘契約與紀錄。

## 資料、模型與第三方授權

- **原始碼：** MIT，見 [LICENSE](LICENSE)；其中轉述的 nuScenes 衍生數值除外，適用下一項。
- **Evidence 與圖表：** `docs/evidence/`、claims registry `docs/claims.yaml`、`docs/figures/`、`docs/analysis/`，它們在網站上的副本與表格，以及在本 repository 其他地方轉述的這些數值（例如本 README 的結果表、已知問題、研究卡、座標契約與測試資料），都是由 nuScenes v1.0-trainval 衍生的彙總量測。它們不含影像、點雲、sample token 或場景名稱，依 nuScenes 使用條款（CC BY-NC-SA 4.0）供非商業研究使用。nuScenes 本身由 Motional 散布，不在此轉散布。使用這些結果時請引用 nuScenes（Caesar et al., CVPR 2020），引用資訊見 [NOTICE](NOTICE)。
- **預訓練權重：** 修正器從 timm 的 `convnextv2_tiny.fcmae_ft_in1k` 初始化，其模型卡宣告 CC BY-NC 4.0 授權。不散布任何訓練後的 checkpoint。
- **Explorer 頁面：** 內嵌 Plotly.js（MIT），其中包含 MapLibre GL JS（BSD-3-Clause）；頁面保留兩者的授權標示，以及 MapLibre GL JS 授權全文的連結。

[NOTICE](NOTICE) 列出上述條款，[CITATION.cff](CITATION.cff) 提供本 repository 的引用資訊。
