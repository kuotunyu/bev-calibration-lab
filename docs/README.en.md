# Documentation

[English overview](../README.en.md) · [正體中文文件入口](README.md)

Start with the [experiment card](experiment-card.en.md) for the question and the limits
of the results, then the [model card](model-card.en.md) for inputs, outputs and weight
provenance. Read the [v1.0.0 known issues](errata.md) before citing recovery at ±0.25°,
the timing stress test or far-range BEV error.

| To check | Document |
| --- | --- |
| Formal results and their sources | [Five evidence documents](evidence/README.md), [two-analysis reproduction](verification/analysis-reproduction.md) |
| Where correction helps (derived) | [Operating envelope](analysis/operating-envelope.md) |
| Reading notes found after the release | [v1.0.0 known issues](errata.md) |
| Statistical units, pairing, bootstrap and missing values | [Formal analysis contract](contracts/formal-analysis.md) |
| Data use and exclusions | [Dataset card](dataset-card.md), [cohort contract](cohort-contract.md) |
| Frame directions, fault axes and sensor timing | [Coordinate contract](coordinate-contract.md) |
| Preprocessing, targets, model selection and provenance | [Training contract](training-contract.md) |
| Bounded diagnostics of the weak correctors | [Diagnostics and open questions](verification/corrector-diagnostics.md) |
| Commands after installation | [Commands and reports](commands-and-report.md) |
| Independent study expectations, raw artifacts and claims | [Study input validation contract](contracts/fault-study-expectations.md) |
| Optional descriptive calibration distribution interchange | [Calibration distribution contract](contracts/calibration-distribution.md) |
| Calibration sensitivity on a fixed synthetic scene | [Offline calibration explorer](https://kuotunyu.github.io/bev-calibration-lab/demo/calibration-explorer.html) |
| Mutation evidence and its scope | [Mutation audit](verification/mutation-audit.md) |
| Published scope and verification | [v1.0.0 release notes](release-notes/v1.0.0.md) |

The formal evidence files contain no raw images, point clouds, sample or box tokens, or
weights. They allow checking sources, definitions and statistics; rerunning the original
experiment still needs licensed data and the frozen inputs.

v1.0.0 is published and its presentation accepted. Public numbers must be bound to the
same evidence and claims; engineering tests cannot stand in for model performance or
real-vehicle safety evidence.

The formal figures, the complete claims registry, the interchange results with
[perception-error-to-aeb](https://github.com/kuotunyu/perception-error-to-aeb) and the
post-release checks are in the
[publication and interchange record](verification/publication-and-interchange.md).
Descriptive interchange does not validate any causal effect of the corrector on AEB.
