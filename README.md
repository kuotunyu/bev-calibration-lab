# bev-calibration-lab

[English](README.en.md) · [文件入口](docs/README.md) · [研究卡](docs/experiment-card.md) · [模型卡](docs/model-card.md)

LiDAR 與相機之間的外參標定只要偏一點點，代價有多大？又能救回多少？

這個專案在 nuScenes 上量測這件事。它保持原始感測器內容不變，注入受控的旋轉、平移故障，比較 identity 基準、傳統 6DoF 最佳化器與三個訓練 seed 的 ConvNeXtV2-Tiny 修正器。時間偏移另外作為 identity-only 壓力測試：改選 LiDAR sweep，不把它當成外參修正器能恢復的第七軸。

**目前狀態：正式訓練、推論與統計已完成；展示與發布仍在驗收。**
[五份正式統計](docs/evidence/README.md)已保存，兩次獨立 CPU 分析取得相同 bytes，來源與限制見[重現紀錄](docs/verification/analysis-reproduction.md)。這不代表學習式修正器已被證明有效，也不代表 GitHub release 或互動展示已完成。

閱讀結果時，保留每個 seed、所有預定故障條件、identity 與零故障基準。
三個固定 seed 的平均不是預測 ensemble；地面接觸點重建也不是物件偵測或實車安全表現。
正式數值的單位、配對支持、缺值與信賴區間定義見[分析契約](docs/contracts/formal-analysis.md)。

## 座標契約

這是這個問題裡最容易錯、而且錯了不會有人發現的地方，所以固定在一處，並由型別與測試強制。

- 每個轉換命名為 `T_target_source`，意義是把以 `source` 座標表示的點轉到 `target`。
- 公開的點陣列一律為 `[N, 3]`，每個點佔一個 row；內部的齊次 column-vector 運算只允許出現在 adapter 內，不得混入公開介面。
- 正式轉換鏈是 LiDAR sensor → LiDAR 時刻 ego → global → 相機時刻 ego → camera sensor。是兩個 ego pose 而不是一個，因為 LiDAR 掃描與相機曝光不在同一瞬間；把它們併成一個，會得到看起來合理但錯誤的答案。
- 3D 邊界框的原生座標系是 global。
- LiDAR 特徵保留 `x, y, z, intensity, ring`。

完整規則、列向量邊界、兩個 ego 時間戳與一個可手算驗證的數值範例，見
[docs/coordinate-contract.md](docs/coordinate-contract.md)。

## 感測器與 cohort

只用 `CAM_FRONT` 與 `LIDAR_TOP`。正式 cohort 是 150 個 nuScenes 場景：100 個官方 train 開發用、20 個來自不同 log 的校準場景、30 個官方 validation 保留給鎖定評估。場景分派依 location 分層並以 token SHA-256 排序，可完整重現，且與任何量測結果無關。nuScenes mini 只用於開發與整合測試，不會出現在任何回報結果中。

## 環境需求

Python 3.12 與 [uv](https://docs.astral.sh/uv/) 0.11.x。nuScenes 資料授權給帳號持有者，不在此散布，也永遠不會提交進版控。

```bash
# 開發時 train extra 不是選配：學習式修正器是第一方程式碼，覆蓋率關卡涵蓋它。
uv sync --frozen --all-groups --extra train --extra report
uv run bev-calib --help
```

原生資料命令與報告介面見[命令文件](docs/commands-and-report.md)。原始感測器資料與模型權重不在 repo；閱讀已保存的統計不需要下載它們。

## 品質關卡

`uv run python -m bevcalib.dev verify` 以固定順序執行所有關卡，遇到第一個失敗就停止：私有檔案防護、格式檢查、lint、型別檢查、完整測試、第一方程式碼 100% 陳述與分支覆蓋、schema 契約、文件連結。不使用任何覆蓋率豁免，沒有 `pragma: no cover`，也沒有排除的第一方路徑。

## 授權

MIT，見 [LICENSE](LICENSE)。nuScenes 本身由 Motional 依其自身條款散布，不在此轉散布。
