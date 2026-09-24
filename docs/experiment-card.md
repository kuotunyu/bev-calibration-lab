# 研究卡：外參故障的敏感度與恢復

## 研究問題

在固定的 nuScenes 感測器觀測上，LiDAR–相機外參偏差如何影響幾何誤差？identity、傳統最佳化與學習式修正能恢復多少？本研究衡量幾何與校正行為，不把地面接觸點重建當成 detector 準確率，也沒有閉迴路 AEB 或實車安全驗證。

## 凍結設計與資料用途

使用 `CAM_FRONT` 與 `LIDAR_TOP`。官方 train 的 100 個 development scenes 用於訓練；官方 train 的 20 個 calibration scenes 來自不同 logs，用於固定校準目標與 checkpoint 選擇；官方 val 的 30 個 scenes 留作 locked evaluation。角色之間檢查 log、scene、sample 與 sensor identifiers 不重疊。mini 只供開發與整合測試。

分派依 location 分層與 token SHA-256 排序，不能依結果換 cohort。固定設定見[protocol](../configs/protocols/nuscenes_calibration_v1.yaml)、[故障矩陣](../configs/perturbations/formal_v1.yaml)與[cohort 契約](cohort-contract.md)。旋轉和平移逐軸施加；timing 是獨立的 identity-only sweep selection 壓力測試，沒有學習式 timing recovery，而且在 v1 沒有提供資訊（見[已知問題](errata.zh-TW.md)）。

學習式模型使用 seeds 17、42、73，各自保留 checkpoint 與結果。依相同 calibration corruption policy，選擇最早達到最低 calibration loss 的 checkpoint；locked evaluation 不參與選模、超參數或圖表條件選擇。[訓練設定](../configs/correctors/convnextv2_tiny_v1.yaml)與[訓練契約](training-contract.md)記錄完整規則。

## 比較與統計解讀

- 每個 seed、identity、classical、所有預定軸與 level（包含 zero）都保留。不能只呈現改善條件。
- 各 operator 先在 scene 內等權平均 eligible frames，再在 scenes 之間等權；global row validity 不能代替 operator support。
- 配對區間以 scene 為 bootstrap 單位。三個固定 seed 的共同支持平均不等於 prediction ensemble，也不估計任意訓練 seed 的不確定性。
- Recovery 是同時滿足 geodesic rotation 與 translation norm 門檻；改善用 percentage points。原始 translation 是 metres，正式 pose 表轉成 centimetres；BEV error 仍是 metres。
- Pixel 指標是 frame quantile 的 scene-balanced 平均，不是把所有 pixels 混在一起的 pooled quantile。GT 距離 bins、配對物件與有效 cutoff 保持原定義，缺值保留原因。

完整 estimands、門檻、interval 與支持定義以[正式分析契約](contracts/formal-analysis.md)為準。五份結果入口為[evidence index](evidence/README.md)，不能用另一份 descriptive summary 替代正式統計。

## 來源與數值修復

原始 evaluator producer 為 `aeb3f28265c2ee3c0f7556417f9c8bf8a550acb7`，分析修正 producer 為 `33ff76485ed17194626c51333c941a33fe712902`。三個 training seeds 的 provenance 綁定原始 producer、相同預訓練來源及各自 checkpoint；權重身份見[模型卡](model-card.md)。

首次分析因配對物件的 GT range 有浮點微差而拒絕。Analysis policy V2 明定 whole-group absolute span 上限、零相對容差、同 bin 與同 inclusive cutoff；沒有改原始誤差、cohort、checkpoint 或 bootstrap 設定，也沒有依表現挑容差。修復後兩個獨立 CPU 分析產出逐位元相同的五檔。[重現紀錄](verification/analysis-reproduction.md)保存精確版本、限制與 SHA。

## 失敗與尚未確定之處

校正器會改動原本正確的標定，因此零故障表現也是必要結果。Classical edge objective 改善不保證真實 pose 或 BEV 改善；learned training completion 也不是效果保證。現有結果不能據以宣稱已取得通用、可靠的校正模型。

地面平面假設在真實 box bottom 不落在該平面時，可在零故障留下非零重建誤差。弱恢復與退步的全部原因尚未確定，不能宣稱已證明不存在實作問題。後續診斷應保留失敗，區分程式修復與新的研究假說；不得用已看過的 locked evaluation 直接調參或替換較漂亮的 seed。

發布後找到三項閱讀注意事項：±0.25° 的 identity recovery 是浮點數邊界假象、timing 壓力測試沒有提供資訊，以及遠距 BEV 平均值由接近水平的射線主導，見 [v1.0.0 已知問題](errata.zh-TW.md)。

正式報告、互動展示與 v1.0.0 發布已完成驗收，詳見[發布核對](verification/publication-and-interchange.md)。工程驗收不改變上述研究限制。任何新的分析或實驗都需獨立來源身份，不能回寫這份凍結結果。
