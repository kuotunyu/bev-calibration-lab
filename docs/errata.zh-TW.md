# v1.0.0 已知問題與閱讀注意事項

[English](errata.md)

這份說明是 v1.0.0 發布後重新檢查凍結 evidence 時寫下的。它不修改任何已發布的數值、圖表、claim、設定或 tag，只補充發布文件沒有解釋的三個部分該怎麼讀。下文每個數字都綁定已發布的 evidence，並由測試檢查。

## 1. identity 在 ±0.25° 的 recovery 是浮點數邊界假象

Recovery 的定義是：geodesic 旋轉誤差不超過 0.25°、平移誤差不超過 5 cm，邊界值也算恢復（[`recovered()`](../src/bevcalib/metrics/calibration.py)）。故障矩陣的每個旋轉軸也都包含 ±0.25°（[故障矩陣](../configs/perturbations/formal_v1.yaml)）。identity 不修正注入的故障，所以它的誤差理應正好等於門檻。
旋轉轉換中的浮點數捨入，讓記錄下來的誤差變成 0.2500000000006049°，比門檻大一點點，因此 identity 在這些條件下的 recovery 是 0%，而不是 100%。 <!-- bind: 0.2500000000006049 = metrics#/runs/identity/roll:0.25/rotation_geodesic_deg/value ; 0 = recovery#/runs/identity/roll:0.25/value -->
同樣的捨入在 ±0.1° 得到 0.10000000000072487°，仍明顯在門檻內，recovery 是 100%。 <!-- bind: 0.10000000000072487 = metrics#/runs/identity/roll:0.1/rotation_geodesic_deg/value ; 100 = recovery#/runs/identity/roll:0.1/value -->

**受影響範圍。** 六個條件：`roll:±0.25`、`pitch:±0.25`、`yaw:±0.25`。兩類欄位：identity 在這些條件的 recovery（[recovery.json](evidence/nuscenes_calibration_v1/recovery.json) 的 `/runs/identity/<condition>`，以及 metrics.json 的 `recovery_rate_pct`），以及同樣條件下 identity→* 的 recovery 比較（recovery.json 的 `/comparisons/identity->*/<condition>`）。
這 30 個比較中有 24 個的 95% 區間完全大於零，看起來像是相對 identity 的顯著改善，實際上不是。 <!-- bind: 30 = envelope#/recovery_boundary/identity_comparisons ; 24 = envelope#/recovery_boundary/identity_comparisons_above_zero -->
例如 identity → learned-42 在 `roll:0.25` 從 0% 變成 11.25%，區間為 5.63 到 17.40 個百分點。 <!-- bind: 0 = recovery#/comparisons/identity->learned-42/roll:0.25/before ; 11.25 = recovery#/comparisons/identity->learned-42/roll:0.25/after ; 5.63 = recovery#/comparisons/identity->learned-42/roll:0.25/interval/low ; 17.40 = recovery#/comparisons/identity->learned-42/roll:0.25/interval/high -->
[Recovery 圖](https://kuotunyu.github.io/bev-calibration-lab/figures/recovery-by-fault-level.svg)中 identity 從 ±0.1° 的 100% 掉到 ±0.25° 的 0%，也是同一個假象。

**不受影響。** Pose、pixel、edge 與 BEV 指標；所有不涉及 identity 的比較；identity 在其他條件的結果。

**正確讀法。** identity 在 ±0.25° 其實落在宣告的容許範圍內，recovery 應為 100%；這些條件下每個 identity→* 的 recovery 差值都應為負，因為修正器只可能把原本合格的 frame 移出容許範圍。不要把 ±0.25° 的 identity→* recovery 當成結果引用。

**未來 protocol 的修正方式。** 比較時使用明確的數值容差（例如 `error <= threshold + 1e-9`），或讓門檻不要與故障等級重合。v1 evidence 維持發布時的樣子。

## 2. v1 的 timing 壓力測試沒有提供資訊

Timing 壓力測試固定相機曝光時間，選擇最接近「相機時間加上指定偏移」的 LIDAR_TOP 封包，誤差 25 ms 以內才有效。鎖定 cohort 是從 trainval keyframe 壓縮檔建立的，裡面只有 keyframe sweep（[nuScenes preflight](verification/nuscenes-preflight.md)）；只有一個場景的磁碟上有其他 sweep。因此不論指定哪個偏移，最近的封包幾乎都是距離相機曝光約 36 ms 的 keyframe sweep。

- 七個指定偏移的實際偏移中位數都在 35.81 到 36.09 ms 之間。 <!-- bind: 35.81 = timing#/offsets/-200/realized_offset_ms/median ; 36.09 = timing#/offsets/100/realized_offset_ms/median -->
- 在 +50 ms，keyframe sweep 落在容許範圍內（絕對誤差中位數 14.02 ms），所以 1207 個 frame 中 1207 個有效。 <!-- bind: 14.02 = timing#/offsets/50/absolute_error_ms/median ; 1207 = timing#/offsets/50/total ; 1207 = timing#/offsets/50/valid -->
- 其他偏移都只有來自 1 個場景的 40 個 frame 有效（+200 ms 為 39 個）。 <!-- bind: 1 = metrics#/runs/identity/time:0/edge_score_px/support/scenes ; 40 = timing#/offsets/0/valid ; 39 = timing#/offsets/200/valid -->
- 在 +50 ms 選到的 sweep 正是外參研究所用的那一個，所以每個量測指標都與零故障的 identity 相同：兩者的 edge score 都是 -4.565473134856439 px，0-10 m BEV 誤差都是 1.182511468842712 m。 <!-- bind: -4.565473134856439 = metrics#/runs/identity/time:50/edge_score_px/value ; -4.565473134856439 = metrics#/runs/identity/pitch:0/edge_score_px/value ; 1.182511468842712 = metrics#/runs/identity/time:50/bev_frame_mean_m~10-10/value ; 1.182511468842712 = metrics#/runs/identity/pitch:0/bev_frame_mean_m~10-10/value -->

Pixel 與 BEV 誤差比較的是同一組點與框在真實標定與假設標定下的差異。timing 故障不改變任何一方的標定，所以這兩個誤差不可能隨 sweep 選擇而改變（每個偏移的 pixel 誤差在數值上都是零）。只有 edge proxy 用選到的 sweep 的 LiDAR 邊緣去比對影像邊緣，才會受 sweep 影響。

**正確讀法。** v1 的 timing 壓力測試既不能說明對 LiDAR-相機時間偏移的穩健性，也不能說明敏感度。要做 timing 研究，必須為每個場景準備非 keyframe 的 sweep。

## 3. 遠距 BEV 平均值由接近水平的射線主導

BEV 指標把相機穿過框底部的射線與位於 ego 原點高度的平面地面相交，重建每個框的接地位置（[座標契約](coordinate-contract.md#oracle-controlled-ipm-baseline)）。射線與地面夾角很小時，一點點角度或高度差就會讓交點沿地面移動很遠：相機高度為 h、接地點距離為 d 時，角度誤差 δ 大約造成 d²·δ/h 的位移。因此遠距 bin 的平均值由少數接近水平的射線主導。

- 在零故障下只有平面模型本身的誤差，identity 在 0-10、10-20、20-40、40-80 m 四個 bin 的平均誤差已經是 1.18、8.16、54.50、781.05 m。 <!-- bind: 1.18 = metrics#/runs/identity/pitch:0/bev_frame_mean_m~10-10/value ; 8.16 = metrics#/runs/identity/pitch:0/bev_frame_mean_m~110-20/value ; 54.50 = metrics#/runs/identity/pitch:0/bev_frame_mean_m~120-40/value ; 781.05 = metrics#/runs/identity/pitch:0/bev_frame_mean_m~140-80/value -->
- 加入 -2° 的俯仰（tilt，正式 `roll`）故障後，同樣四個 bin 是 1.19、3.68、11.99、26.42 m：故障反而讓遠距平均值變小。 <!-- bind: 1.19 = metrics#/runs/identity/roll:-2/bev_frame_mean_m~10-10/value ; 3.68 = metrics#/runs/identity/roll:-2/bev_frame_mean_m~110-20/value ; 11.99 = metrics#/runs/identity/roll:-2/bev_frame_mean_m~120-40/value ; 26.42 = metrics#/runs/identity/roll:-2/bev_frame_mean_m~140-80/value -->
- evidence 中最大的 40-80 m 平均誤差是 6813.33 m（learned-73 在 `yaw:-0.25`）。 <!-- bind: 6813.33 = metrics#/runs/learned-73/yaw:-0.25/bev_frame_mean_m~140-80/value -->

**正確讀法。** BEV 誤差只解讀 0-10 m 與 10-20 m 兩個 bin。遠距 bin 的平均值反映的是射線與平面相交的條件好壞，而不是對標定的敏感度。已發布的 [BEV 圖](https://kuotunyu.github.io/bev-calibration-lab/figures/bev-error-by-range.svg)各 bin 共用同一個縱軸，因此縱軸刻度由遠距 bin 決定。
