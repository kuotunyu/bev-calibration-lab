# 模型卡：ConvNeXtV2-Tiny 外參修正器

[English](model-card.en.md)

## 用途與界線

這是 nuScenes `CAM_FRONT`–`LIDAR_TOP` 外參故障研究的學習式基線，用來估計已注入 rigid transform 的逆修正。它不做物件偵測、不預測煞車、不估計 timing correction，也不是已驗證可部署於車輛的標定系統。應同時閱讀[研究卡](experiment-card.md)及全部 seed、zero-fault 與 baseline 結果。

## 輸入與輸出

| 項目 | 固定契約 |
| --- | --- |
| 架構 | ConvNeXtV2-Tiny，五通道 stem、六個輸出 |
| 輸入 | 448 × 800，依序 RGB、assumed-calibration projected depth、validity mask |
| RGB | 除以 255，使用 ImageNet mean/std；Pillow bilinear half-pixel resize 同步修改 intrinsics |
| Depth | min-depth rasterization；`log1p(d)/log1p(80)`，clip 至 `[0,1]`；mask 明確區分缺深度 |
| 輸出 | roll/pitch/yaw（degrees），x/y/z（metres），皆為 CAM_FRONT 光學座標系的軸 |
| Target | 注入 source-side rigid transform 的完整逆；inverse translation 要旋轉，不是逐分量取負 |

五通道 stem 由 RGB weights 擴充，新增兩通道取 mean-RGB kernel，並依權重能量縮放；這不表示影像、深度與 mask 的輸入分布相同。推論輸出保持物理單位，不能再套一次 training loss normalization。完整公式與邊界見[訓練契約](training-contract.md)與[座標契約](coordinate-contract.md)。輸出的 roll 是俯仰（tilt）、pitch 是偏擺（pan）、yaw 是影像平面內旋轉，對照表見[研究卡](experiment-card.md#故障軸)。

## 權重與訓練身份

三個正式 seed 的 training provenance 記錄相同初始化來源：

- timm profile：`convnextv2_tiny.fcmae_ft_in1k`。
- 輸入權重 SHA-256：`fdfb4edfa5abb3b4ea29ab17e6fae5c117dc6cc32ecfb1d0726a156f40a4c1a5`。
- Training producer：`aeb3f28265c2ee3c0f7556417f9c8bf8a550acb7`。

這個 profile 名稱與 SHA 是 provenance 身份，不是隨執行時間變動的下載要求。Adapter 只接受明確指定且 hash 相符的本機權重，不自行下載。Repository 的 MIT 授權不取代 nuScenes 或模型權重各自的授權；timm 上這個 profile 的模型卡宣告 CC BY-NC 4.0（見 [NOTICE](../NOTICE)）。這裡不散布任何 checkpoint。

固定設定為 seeds 17、42、73，AdamW、warmup/cosine schedule、normalized Huber loss；精確參數見[設定檔](../configs/correctors/convnextv2_tiny_v1.yaml)。Development 用於參數學習，calibration 用於 checkpoint 選擇，evaluation 保持隔離。三個 seed 的模型不合併為 ensemble。

載入正式 checkpoint 時，程式驗證成功 run record、權重/provenance byte hashes、config、cohort、producer、前處理與 calibration policy，並使用 strict model-state loading、eval mode 與 no-grad 推論。這些工程檢查不保證校正效果良好。

## 已有證據與限制

[正式五檔](evidence/README.md)保存 locked-cohort 的 pose、pixel、edge、BEV、recovery、timing 與 exclusions；[重現紀錄](verification/analysis-reproduction.md)說明分析身份與數值配對修復。精確效能數字應從相同 artifact/claim binding 呈現，不能以模型完成訓練代替結果。README 的主要發現與衍生的 [operating envelope](analysis/operating-envelope.md) 整理了這些結果：三個 seed 的平均在每個條件的 pose、像素誤差與 recovery 都勝過 classical 基準，但留有殘差下限，因此只有在較大的故障下才比不修正更好。

模型的輸入解析度、感測器、資料分布、投影與地面假設均固定。尚未證明跨相機、跨資料集或實車場景的泛化能力。所有預定故障條件與零故障退步都必須保留；弱恢復的全部原因仍未確定。Seed 42 不修正偏擺（正式 `pitch`）。Random-weight 的 synthetic 測試只驗證 adapter 行為，不能當成正式模型的效能或預訓練 eligibility 證據。
