# Visual Router V2 Round 1 P2probe 摘要

生成时间：2026-06-21 08:41:21 CST

## 输入与边界

- P0 sample set：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples`
- P2a feature cache：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features`
- 输出目录：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_feature_probe`
- 本 probe 只训练 sklearn 线性分类器，不训练 Visual Router routing head，不生成 ViT feature，不读取 prediction manifest。

## Expert Suitability 主结果（pilot_selection）

| model_name | accuracy | macro_f1 | balanced_accuracy | top2_recall | cross_entropy | sample_count |
| --- | --- | --- | --- | --- | --- | --- |
| cls_mean_concat | 0.635067 | 0.547568 | 0.556613 | 0.803033 | 8.792659 | 30000 |
| mean_patch_plus_aux | 0.628667 | 0.536455 | 0.544150 | 0.805300 | 8.111806 | 30000 |
| cls_plus_aux | 0.627567 | 0.534012 | 0.541303 | 0.806767 | 8.216728 | 30000 |
| mean_patch_embedding | 0.626833 | 0.535799 | 0.543686 | 0.804933 | 8.152242 | 30000 |
| cls_embedding | 0.623767 | 0.528439 | 0.535169 | 0.800900 | 8.237272 | 30000 |
| revin_aux | 0.566567 | 0.421606 | 0.443330 | 0.768467 | 1.144880 | 30000 |
| tsf_only | 0.414433 | 0.328737 | 0.357353 | 0.592500 | 1.442994 | 30000 |
| train_frequency_prior | 0.349967 | 0.103696 | 0.200000 | 0.659633 | 1.456734 | 30000 |
| dataset_name_only | 0.319867 | 0.149043 | 0.262785 | 0.438833 | 1.576019 | 30000 |

## Structure Semantics 主结果（pilot_selection，各 target 最优三项）

### error_gap_quantile

| model_name | accuracy | macro_f1 | balanced_accuracy | top2_recall | cross_entropy | sample_count |
| --- | --- | --- | --- | --- | --- | --- |
| cls_mean_concat | 0.554767 | 0.520046 | 0.524000 |  | 9.849225 | 30000 |
| cls_embedding | 0.549100 | 0.512785 | 0.518414 |  | 8.612031 | 30000 |
| mean_patch_embedding | 0.548233 | 0.511906 | 0.517061 |  | 8.469357 | 30000 |

### forecastability_cat

| model_name | accuracy | macro_f1 | balanced_accuracy | top2_recall | cross_entropy | sample_count |
| --- | --- | --- | --- | --- | --- | --- |
| mean_patch_embedding | 0.845333 | 0.674554 | 0.745277 |  | 4.058131 | 30000 |
| cls_mean_concat | 0.844367 | 0.674265 | 0.747361 |  | 4.187887 | 30000 |
| cls_embedding | 0.843700 | 0.675207 | 0.750398 |  | 4.039186 | 30000 |

### season_strength_cat

| model_name | accuracy | macro_f1 | balanced_accuracy | top2_recall | cross_entropy | sample_count |
| --- | --- | --- | --- | --- | --- | --- |
| cls_mean_concat | 0.753100 | 0.688590 | 0.668465 |  | 5.497009 | 30000 |
| mean_patch_embedding | 0.746900 | 0.679751 | 0.658271 |  | 4.972853 | 30000 |
| cls_embedding | 0.746500 | 0.679727 | 0.658834 |  | 4.137178 | 30000 |

### trend_strength_cat

| model_name | accuracy | macro_f1 | balanced_accuracy | top2_recall | cross_entropy | sample_count |
| --- | --- | --- | --- | --- | --- | --- |
| cls_mean_concat | 0.671900 | 0.628149 | 0.630300 |  | 8.031070 | 30000 |
| mean_patch_embedding | 0.665767 | 0.620723 | 0.623815 |  | 7.035896 | 30000 |
| cls_embedding | 0.661233 | 0.617760 | 0.620284 |  | 7.270950 | 30000 |

### cv_cat

| model_name | accuracy | macro_f1 | balanced_accuracy | top2_recall | cross_entropy | sample_count |
| --- | --- | --- | --- | --- | --- | --- |
| cls_mean_concat | 0.663067 | 0.564596 | 0.560528 |  | 8.488010 | 30000 |
| cls_embedding | 0.662367 | 0.567612 | 0.563897 |  | 6.438703 | 30000 |
| mean_patch_embedding | 0.660700 | 0.558403 | 0.552649 |  | 7.381303 | 30000 |

### missing_ratio_cat

| model_name | accuracy | macro_f1 | balanced_accuracy | top2_recall | cross_entropy | sample_count |
| --- | --- | --- | --- | --- | --- | --- |
| cls_embedding | 1.000000 | 1.000000 | 1.000000 |  | -0.000000 | 30000 |
| mean_patch_embedding | 1.000000 | 1.000000 | 1.000000 |  | -0.000000 | 30000 |
| cls_mean_concat | 1.000000 | 1.000000 | 1.000000 |  | -0.000000 | 30000 |

### cluster

| model_name | accuracy | macro_f1 | balanced_accuracy | top2_recall | cross_entropy | sample_count |
| --- | --- | --- | --- | --- | --- | --- |
| cls_mean_concat | 0.462133 | 0.447716 | 0.493184 |  | 12.914215 | 30000 |
| mean_patch_embedding | 0.461067 | 0.445032 | 0.488465 |  | 10.100976 | 30000 |
| cls_embedding | 0.457367 | 0.445477 | 0.489364 |  | 9.905380 | 30000 |

### group_name

| model_name | accuracy | macro_f1 | balanced_accuracy | top2_recall | cross_entropy | sample_count |
| --- | --- | --- | --- | --- | --- | --- |
| cls_mean_concat | 0.467100 | 0.455783 | 0.500654 |  | 13.017137 | 30000 |
| cls_embedding | 0.464600 | 0.449231 | 0.492650 |  | 10.802596 | 30000 |
| mean_patch_embedding | 0.461033 | 0.447460 | 0.490743 |  | 10.602471 | 30000 |

## 验收问题回答

1. visual embedding 是否能预测 oracle expert，是否优于 dataset/TSF shortcut baseline？best_visual accuracy=0.6351, macroF1=0.5476, top2=0.8030；best_shortcut accuracy=0.4144, macroF1=0.3287, top2=0.5925。结论以 selection 表为准。
2. mean_patch 是否比 CLS 含有更多 expert suitability 信息？CLS accuracy=0.6238, macroF1=0.5284, top2=0.8009；mean_patch accuracy=0.6268, macroF1=0.5358, top2=0.8049。
3. visual embedding 是否能恢复 TSF/结构语义标签？visual 结构 probe 平均 macroF1=0.6194；需要逐 target 查看上表，尤其 cluster/group_name 用于 shortcut 风险参考。
4. revin_aux 的信息强度与 visual embedding 相比如何？revin_aux accuracy=0.5666, macroF1=0.4216, top2=0.7685；结构 probe 中 revin_aux 平均 macroF1=0.4638。
5. 当前证据更支持“视觉表示有结构语义增量”，还是“主要是 dataset/TSF/expert shortcut”？当前自动判读：视觉表示有结构语义增量。如果 dataset/TSF-only 接近或超过 visual，需要在 P2d 前优先做 group split/held-out dataset 复验。
6. 对 P2d visual+aux concat 和 Round 2 view/imageization 消融的启发：若 cls_plus_aux/mean_patch_plus_aux 相比单独 visual 与 aux 有稳定增益，P2d concat 值得推进；若 mean_patch 优于 CLS，Round 2 应优先保留 patch-token 聚合视角；若结构标签恢复主要集中在 dataset/group_name，Round 2 需要设计更强 held-out dataset/cell 消融来排除 shortcut。

## 输出文件

- `feature_probe_expert_suitability_results.csv`
- `feature_probe_structure_results.csv`
- `feature_probe_shortcut_baselines.csv`
- `feature_probe_confusion_matrices.csv`
- `feature_probe_within_dataset_summary.csv`
- `feature_probe_metadata.json`
