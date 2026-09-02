# bev-calibration-lab

LiDAR 與相機之間的外參標定只要偏一點點，代價有多大？又能救回多少？

這個專案在 nuScenes 上量測這件事。它把受控的旋轉、平移與時間偏移注入感測器標定鏈，保持觀測資料不變，然後回報這些故障對 LiDAR-相機邊緣對齊與鳥瞰圖地面接觸點重建造成多少誤差。接著用三種修正器嘗試還原真實外參：什麼都不做的 identity 基準線、傳統 6DoF 最佳化器，以及學習式 ConvNeXtV2-Tiny 修正器。

**目前狀態：施工中。** 只有 packaging 基礎。尚未讀取任何資料集、尚未執行任何實驗，本文件不含任何實驗結果。日後公開的每個數字都會標示為 `observed`、`derived`、`synthetic` 或 `illustrative`，其中 `observed` 的數字必須能從已提交的 artifact 重現。

## 座標契約

這是這個問題裡最容易錯、而且錯了不會有人發現的地方，所以固定在一處，並由型別與測試強制。

- 每個轉換命名為 `T_target_source`，意義是把以 `source` 座標表示的點轉到 `target`。
- 公開的點陣列一律是 `[N, 3]` 列向量。任何行向量慣例只能存在於 adapter 內部，不得出現在公開簽章。
- 正式轉換鏈是 LiDAR sensor → LiDAR 時刻 ego → global → 相機時刻 ego → camera sensor。是兩個 ego pose 而不是一個，因為 LiDAR 掃描與相機曝光不在同一瞬間；把它們併成一個，會得到看起來合理但錯誤的答案。
- 3D 邊界框的原生座標系是 global。
- LiDAR 特徵保留 `x, y, z, intensity, ring`。

## 感測器與 cohort

只用 `CAM_FRONT` 與 `LIDAR_TOP`。正式 cohort 是 150 個 nuScenes 場景：100 個官方 train 開發用、20 個來自不同 log 的校準場景、30 個官方 validation 保留給鎖定評估。場景分派依 location 分層並以 token SHA-256 排序，可完整重現，且與任何量測結果無關。nuScenes mini 只用於開發與整合測試，不會出現在任何回報結果中。

## 環境需求

Python 3.12 與 [uv](https://docs.astral.sh/uv/) 0.11.x。nuScenes 資料授權給帳號持有者，不在此散布，也永遠不會提交進版控。

```bash
uv sync --frozen
uv run bev-calib --help
```

## 品質關卡

`uv run python -m bevcalib.dev verify` 以固定順序執行所有關卡，遇到第一個失敗就停止：私有檔案防護、格式檢查、lint、型別檢查、完整測試、第一方程式碼 100% 陳述與分支覆蓋、schema 契約、文件連結。不使用任何覆蓋率豁免，沒有 `pragma: no cover`，也沒有排除的第一方路徑。

## 授權

MIT，見 [LICENSE](LICENSE)。nuScenes 本身由 Motional 依其自身條款散布，不在此轉散布。
