# 文件入口

[正體中文首頁](../README.md) · [English overview](../README.en.md) · [English documentation index](README.en.md)

先讀[研究卡](experiment-card.md)理解比較問題與結果界線，再讀[模型卡](model-card.md)理解輸入、輸出與權重來源。引用 ±0.25° 的 recovery、timing 壓力測試或遠距 BEV 誤差之前，請先讀 [v1.0.0 已知問題](errata.zh-TW.md)。

| 要確認的事情 | 文件 |
| --- | --- |
| 正式統計與來源 | [五份 evidence](evidence/README.md)、[兩次分析重現](verification/analysis-reproduction.md) |
| 修正在哪些範圍有幫助（衍生分析） | [operating envelope](analysis/operating-envelope.md) |
| 發布後找到的閱讀注意事項 | [v1.0.0 已知問題](errata.zh-TW.md)（[English](errata.md)） |
| 統計單位、配對、bootstrap 與缺值 | [正式分析契約](contracts/formal-analysis.md) |
| 資料用途與排除界線 | [資料卡](dataset-card.md)、[cohort 契約](cohort-contract.md) |
| 座標方向、故障軸與不同感測器時間 | [座標契約](coordinate-contract.md) |
| 前處理、目標、選模與 provenance | [訓練契約](training-contract.md) |
| 校正器效果偏弱的有界診斷 | [診斷證據與未解問題](verification/corrector-diagnostics.md) |
| 安裝後可執行的命令 | [命令與報告](commands-and-report.md) |
| 獨立預期身份、原始產物與 claims 核對 | [study 輸入驗證契約](contracts/fault-study-expectations.md)（真實輸入已核對；範圍見重現紀錄） |
| 可選的描述性標定分布交換 | [calibration distribution 契約](contracts/calibration-distribution.md)（實際匯出與互通已驗證） |
| 固定合成場景的標定敏感度 | [離線 calibration explorer](https://kuotunyu.github.io/bev-calibration-lab/demo/calibration-explorer.html)（互動驗收完成） |
| mutation 的歷史證據與更新範圍 | [mutation audit](verification/mutation-audit.md)（保留 base campaign 與具名修補驗證的界線） |
| 已發布範圍與驗證 | [v1.0.0 說明](release-notes/v1.0.0.md) |

正式統計檔不含原始影像、點雲、sample/box tokens 或權重。它們可供檢查來源、定義與統計結果；完整重跑原始實驗仍需自行取得具授權的資料及凍結輸入。

v1.0.0 已完成發布與展示驗收。公開數值必須綁定相同 evidence 與 claims；工程測試也不能取代模型效能或實車安全證據。

正式圖表、完整 claims、與 perception-error-to-aeb 的實際互通結果及發布後核對見[報告與互通重現紀錄](verification/publication-and-interchange.md)。描述性互通沒有驗證模型效果對 AEB 的因果影響。
