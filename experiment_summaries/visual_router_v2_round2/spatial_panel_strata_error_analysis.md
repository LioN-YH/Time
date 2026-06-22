# Visual Router V2 Round2 Spatial Panel Strata and Error Analysis

生成时间：2026-06-22 10:04:52 CST

## 结论

本分析只读既有 Round1 `film_mean_patch_aux` 与 Round2f `spatial_panel_3view + film_mean_patch_aux` 的 summary CSV 和 frozen `pilot_test` 逐样本 prediction CSV；未训练、未跑 ViT、未生成 feature cache、未新建样本，也未用 test 做选择。

`spatial_panel_3view` 的收益是稳定小幅收益，不是决定性突破。frozen `pilot_test` raw-soft MAE 从 0.417824 降到 0.413558，delta=-0.004266；MSE 从 183.353985 降到 182.771166，delta=-0.582819；raw-soft regret delta=-0.004266。hard top1 也改善，但幅度略小，MAE delta=-0.003372。oracle-label accuracy 从 0.515560 到 0.521733，delta=+0.006173；weight entropy 从 1.122232 降到 1.105016，mean max weight 从 0.514508 升到 0.518161。

收益主要来自困难样本和高误差区间：`error_gap_quantile=q5` MAE delta=-0.014968，`low` forecastability delta=-0.006588，`strong` trend delta=-0.008598，`highly_variable` CV delta=-0.008850。oracle_model 分层中 CrossFormer/PatchTST 改善最大，分别为 -0.009317/-0.008063；ES 改善最小，仅 -0.000946。`LOW_LOW_HIGH` group 是明确弱点，MAE/MSE 均轻微退化。

## 产物

- `spatial_panel_strata_metrics.csv`：overall、oracle_model、dataset/group、forecastability/season/trend/CV/error quantile 分层表。
- `spatial_panel_error_tail_metrics.csv`：top 1%/5% raw-soft MAE 与 raw-soft regret tail，对 Round1-tail 和 Round2-tail 分别评估两套系统。
- `spatial_panel_strata_metadata.json`：数据来源、边界、聚合口径和输出文件。
- `visual_router_experiments/stage1_vali_test_router/analyze_visual_router_v2_round2_spatial_panel_strata.py`：只读生成脚本。

## Overall Comparison

主指标仍是 raw-soft，不把 hard top1 或 oracle-label accuracy 当作主线目标。

| sample_set | method | Round1 MAE/MSE/regret | Round2 MAE/MSE/regret | delta MAE/MSE/regret |
| --- | --- | --- | --- | --- |
| pilot_selection | raw-soft | 0.300393 / 1.289872 / 0.033662 | 0.300227 / 1.269413 / 0.033496 | -0.000166 / -0.020458 / -0.000166 |
| pilot_selection | hard top1 | 0.317828 / 1.378600 / 0.051097 | 0.318488 / 1.329941 / 0.051756 | +0.000659 / -0.048659 / +0.000659 |
| pilot_test | raw-soft | 0.417824 / 183.353985 / 0.077539 | 0.413558 / 182.771166 / 0.073273 | -0.004266 / -0.582819 / -0.004266 |
| pilot_test | hard top1 | 0.431851 / 183.477411 / 0.091566 | 0.428479 / 182.957533 / 0.088194 | -0.003372 / -0.519879 / -0.003372 |

seed stability 可接受但 Round2 frozen test MAE std 更大：Round1 raw-soft pilot_test MAE_std=0.000657，Round2=0.002209。selection 上二者接近：Round1=0.000542，Round2=0.000546。

raw-soft 与 hard top1 同向改善，说明 spatial panel 不只是 soft calibration；但 raw-soft 增益更大，且 selection hard top1 MAE 轻微变差，说明 soft fusion 仍是主要收益承载方式。

## Oracle Model Strata

| oracle_model | count | Round1 MAE | Round2 MAE | delta MAE | delta MSE | oracle acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CrossFormer | 7,417 | 0.635316 | 0.625999 | -0.009317 | -0.035146 | +0.062020 |
| PatchTST | 16,016 | 0.585689 | 0.577627 | -0.008063 | -0.642778 | -0.024205 |
| NaiveForecaster | 11,038 | 0.369585 | 0.363845 | -0.005740 | -1.988553 | -0.000966 |
| DLinear | 9,432 | 0.628059 | 0.624993 | -0.003066 | -1.076565 | -0.021947 |
| ES | 31,097 | 0.232849 | 0.231903 | -0.000946 | -0.033838 | +0.019562 |

CrossFormer 的改善最清晰，并伴随 oracle-label accuracy 提升；PatchTST/DLinear 的 raw-soft MAE 改善与 oracle-label accuracy 下降并存，说明这些 strata 的收益更像权重分配或融合数值改善，而不是 hard classifier recall 改善。

ES 仍是最大样本组且 MAE 本身最低，空间 panel 对它的边际改善很小。CrossFormer/PatchTST 仍是困难 strata：即使改善后 MAE 仍分别为 0.625999/0.577627，显著高于 ES。

## Dataset and Group Strata

可用字段包括 `dataset_name`、`group_name`、`forecastability_cat`、`season_strength_cat`、`trend_strength_cat`、`cv_cat`、`missing_ratio_cat`。当前 `missing_ratio_cat` 全部为 `complete`，不能用于缺失率分层解释。

dataset 层面两个组都改善，没有只在单一 dataset 上成立的明显 shortcut 证据，但 `TEST_DATA_MIN` 占 67,808/75,000，仍会主导 overall：

| dataset | count | delta MAE | delta MSE | oracle acc delta |
| --- | ---: | ---: | ---: | ---: |
| TEST_DATA_MIN | 67,808 | -0.004497 | -0.622192 | +0.007615 |
| TEST_DATA_HOUR | 7,192 | -0.002081 | -0.211607 | -0.007416 |

group 层面，改善最大的是 `HIGH_HIGH_LOW`，MAE delta=-0.015945、MSE delta=-4.802910；其次是 `LOW_HIGH_LOW` 和 `HIGH_LOW_LOW`。弱点是 `LOW_LOW_HIGH`：MAE delta=+0.001635、MSE delta=+0.030233，是本次明确退化 group。`LOW_HIGH_HIGH`、`HIGH_HIGH_HIGH` 基本持平。

结构属性上，收益集中在更难的样本：`q5` error gap delta=-0.014968，`low` forecastability delta=-0.006588，`moderate` season strength delta=-0.007239，`strong` trend delta=-0.008598，`highly_variable` CV delta=-0.008850。这支持 spatial panel 并非纯 overall 微平移，而是在困难 strata 上有更明显贡献。

## Error Tail

tail 使用三 seed 按 sample_key 聚合后的 raw-soft MAE 和 `soft_fusion_mae - oracle_value` regret。重点结果：

- Round1 top 1% soft MAE tail 上，Round2 mean soft MAE 从 7.505428 降到 7.107567，mean regret 从 2.191740 降到 1.793879。
- Round1 top 5% soft MAE tail 上，Round2 mean soft MAE 从 2.699470 降到 2.592484，mean regret 从 0.700450 降到 0.593464。
- Round1 top 1% regret tail 上，Round2 mean regret 从 2.619191 降到 2.072519，mean MSE 从 378.085960 降到 318.322122。
- Round1 top 5% regret tail 上，Round2 mean regret 从 0.887476 降到 0.726327，mean MSE 从 77.979016 降到 65.967985。

tail overlap 不是完全相同：top 5% soft MAE tail Jaccard=0.816860，top 5% regret tail Jaccard=0.561199。这说明 spatial panel 对一部分 Round1 high-regret 样本确实有缓解，但也形成了新的/剩余的 Round2 high-regret tail。

Round2 自身 top 5% regret tail 中，oracle_model 分布为 ES 32.3%、DLinear 18.2%、PatchTST 18.1%、NaiveForecaster 15.8%、CrossFormer 15.5%；selected_model mode 分布为 PatchTST 49.4%、ES 20.6%、DLinear 11.9%、CrossFormer 10.4%、NaiveForecaster 7.8%。这提示 tail 中仍存在 PatchTST 选择偏重但 oracle 分布更分散的问题。

## Selected Model and Soft Weight

pilot_test selected_model ratio 均值：

| expert | Round1 ratio | Round2 ratio | delta |
| --- | ---: | ---: | ---: |
| CrossFormer | 0.042587 | 0.059298 | +0.016711 |
| DLinear | 0.215769 | 0.110951 | -0.104818 |
| ES | 0.275111 | 0.291529 | +0.016418 |
| NaiveForecaster | 0.122187 | 0.212742 | +0.090555 |
| PatchTST | 0.344347 | 0.325480 | -0.018867 |

没有单专家极端塌缩；PatchTST 仍最高但下降，DLinear 显著下降，NaiveForecaster 和 CrossFormer 上升。Round2 weight entropy 在 pilot_test 下降、mean max weight 上升，说明权重略更尖锐；这不是“更平滑”带来的收益。结合 PatchTST/DLinear oracle accuracy 下降但 raw-soft 改善，收益更可能来自更合理的 soft weight 数值和 tail 缓解，而不是 hard top1 accuracy 全面提升。

## Failure Modes

仍然较弱的方向：

- CrossFormer/PatchTST oracle strata 改善最大但绝对 MAE 仍高，是 full-scale 必须继续盯住的困难专家组。
- ES stratum 样本最多但改善最小，说明 spatial panel 对低 MAE 大样本区间只是微调。
- `LOW_LOW_HIGH` group 明确退化；`q4` error gap 基本无改善，且 high-regret rate 略升。
- Round2 自身 high-regret tail 仍有明显 PatchTST selected mode 偏重，oracle_model 分布却更分散。
- 本分析不能证明 spatial panel 已解决 hard period fold discontinuity、period semantics、padding mask、panel-wise pooling 等问题；这些仍应作为独立诊断或后续支线，而不是从当前 frozen eval 中外推。

## Implications for Full-Scale Reporting

本窗口不做 full-scale planning，但给 reporting 留下以下口径：

- frozen eval report 应保留 `oracle_model`、`dataset_name`、`group_name`、`error_gap_quantile`、forecastability/season/trend/CV 分层。
- 必须同时记录 raw-soft、hard top1、oracle-label accuracy、entropy、mean max weight、selected_model ratio；主结论仍以 raw-soft MAE/MSE/regret 为准。
- high-error/high-regret top 1%/5% tail 应成为固定表，且要报告 tail overlap 和 tail 内 oracle/selected 分布。
- `current_rgb_3view` 仍应作为 layout baseline 留在 full-scale reporting matrix 中，用于区分 spatial panel 相对 current layout 的稳定收益。
- 重点监控 CrossFormer/PatchTST、`LOW_LOW_HIGH`、`q4/q5`、low forecastability、strong trend、highly variable CV，以及 Round2 high-regret tail 中 PatchTST 选择偏重的问题。
