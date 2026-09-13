# 文件入口

[正體中文首頁](../README.md) · [English overview](../README.en.md)

先讀[研究卡](experiment-card.md)理解比較問題與結果界線，再讀[模型卡](model-card.md)理解輸入、輸出與權重來源。

| 要確認的事情 | 文件 |
| --- | --- |
| 正式統計與來源 | [五份 evidence](evidence/README.md)、[兩次分析重現](verification/analysis-reproduction.md) |
| 統計單位、配對、bootstrap 與缺值 | [正式分析契約](contracts/formal-analysis.md) |
| 資料用途與排除界線 | [資料卡](dataset-card.md)、[cohort 契約](cohort-contract.md) |
| 座標方向與不同感測器時間 | [座標契約](coordinate-contract.md) |
| 前處理、目標、選模與 provenance | [訓練契約](training-contract.md) |
| 校正器效果偏弱的有界診斷 | [診斷證據與未解問題](verification/corrector-diagnostics.md) |
| 安裝後可執行的命令 | [命令與報告](commands-and-report.md) |
| 獨立預期身份、原始產物與 claims 核對 | [study 輸入驗證契約](contracts/fault-study-expectations.md)（真實輸入已核對；範圍見重現紀錄） |
| 可選的描述性標定分布交換 | [calibration distribution 契約](contracts/calibration-distribution.md)（實際匯出與互通已驗證；公開發布仍待） |
| 固定合成場景的標定敏感度 | [離線 calibration explorer](demo/calibration-explorer.html)（候選；人工互動驗收待完成） |
| mutation 的歷史證據與更新範圍 | [mutation audit](verification/mutation-audit.md)（新分析範圍尚未驗收） |
| 發布範圍與未完成驗收 | [v1.0.0 候選說明](release-notes/v1.0.0.md)（尚未發布） |

正式統計檔不含原始影像、點雲、sample/box tokens 或權重。它們可供檢查來源、定義與統計結果；完整重跑原始實驗仍需自行取得具授權的資料及凍結輸入。

這個索引不表示尚未發布的套件、圖表或互動展示已驗收。公開數值必須綁定相同 evidence 與 claims；工程測試也不能取代模型效能或實車安全證據。

正式圖表、完整 claims 與實際 P2/P3 互通結果見[報告與互通重現紀錄](verification/publication-and-interchange.md)。人工視覺驗收與正式發布仍待完成。
