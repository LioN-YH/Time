# Visual Router V2 Round 1 All Variant Summary

生成时间：2026-06-21 11:51:43 CST

## 最终选择

- Round 1 best variant：`cls_mean_concat_plus_aux`
- 所属阶段：`P2d`
- 选择依据：pilot_selection raw-soft MAE_mean lowest; tie-breakers diagnostic raw-soft MAE_mean, seed std, regret, entropy stability
- pilot_test_used_for_selection=false。

## Pilot Selection Raw-Soft Ranking

| stage | sample_set | variant | method | method_kind | seed_count | sample_count | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std | source_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2d | pilot_selection | cls_mean_concat_plus_aux | cls_mean_concat_plus_aux_raw_soft_fusion | raw_soft_fusion | 3 | 30000 | 0.300605 | 0.001287 | 1.205401 | 0.058046 | 0.033874 | 0.001287 | 0.467911 | 0.146558 | 1.118137 | 0.016793 | 0.694737 | 0.010434 | 0.513148 | 0.006964 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat/round1_concat_selection_comparison.csv |
| P2d | pilot_selection | mean_patch_plus_aux | mean_patch_plus_aux_raw_soft_fusion | raw_soft_fusion | 3 | 30000 | 0.300831 | 0.000548 | 1.239938 | 0.182619 | 0.034100 | 0.000548 | 0.468133 | 0.148860 | 1.123557 | 0.017389 | 0.698105 | 0.010804 | 0.510815 | 0.010831 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat/round1_concat_selection_comparison.csv |
| P2b | pilot_selection | visual_mean_patch_only | visual_mean_patch_only_raw_soft_fusion | raw_soft_fusion | 3 | 30000 | 0.300996 | 0.001000 | 1.234168 | 0.163449 | 0.034265 | 0.001000 | 0.378856 | 0.006665 | 1.132446 | 0.019601 | 0.703628 | 0.012179 | 0.504598 | 0.008922 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/visual_pooling_selection_comparison.csv |
| P2b | pilot_selection | visual_cls_only | visual_cls_only_raw_soft_fusion | raw_soft_fusion | 3 | 30000 | 0.302048 | 0.001219 | 1.220514 | 0.109631 | 0.035317 | 0.001219 | 0.376122 | 0.006574 | 1.127909 | 0.020945 | 0.700809 | 0.013014 | 0.508793 | 0.008400 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/visual_pooling_selection_comparison.csv |
| P2b | pilot_selection | visual_cls_mean_concat | visual_cls_mean_concat_raw_soft_fusion | raw_soft_fusion | 3 | 30000 | 0.302220 | 0.003929 | 1.217317 | 0.118708 | 0.035489 | 0.003929 | 0.549878 | 0.159907 | 1.112957 | 0.023586 | 0.691519 | 0.014655 | 0.523813 | 0.011781 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/visual_pooling_selection_comparison.csv |
| P2c | pilot_selection | revin_aux_only_fusion_huber_kl | revin_aux_only_fusion_huber_kl_raw_soft_fusion | raw_soft_fusion | 3 | 30000 | 0.332987 | 0.000310 | 1.510008 | 0.064787 | 0.066256 | 0.000310 | 0.323044 | 0.001840 | 1.277712 | 0.008272 | 0.793887 | 0.005140 | 0.424578 | 0.004765 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_aux_only/aux_only_selection_comparison.csv |
| P1 | pilot_selection | p1_round0_visual_baseline | visual_router_raw_soft_fusion | raw_soft_fusion | 1 | 30000 | 0.334069 | 0.000000 | 1.181831 | 0.000000 | 0.067337 | 0.000000 | 0.579200 | 0.000000 | 1.292436 | 0.000000 | 0.803035 | 0.000000 | 0.439655 | 0.000000 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/round0_selection_comparison.csv |

## Diagnostic Balanced Raw-Soft Ranking

| stage | sample_set | variant | method | method_kind | seed_count | sample_count | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std | source_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2d | diagnostic_balanced | cls_mean_concat_plus_aux | cls_mean_concat_plus_aux_raw_soft_fusion | raw_soft_fusion | 3 | 20000 | 0.346514 | 0.001458 | 1.370437 | 0.033517 | 0.041024 | 0.001458 | 0.411800 | 0.073895 | 1.035812 | 0.017886 | 0.643586 | 0.011113 | 0.554780 | 0.006442 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat/round1_concat_diagnostic_summary.csv |
| P2b | diagnostic_balanced | visual_mean_patch_only | visual_mean_patch_only_raw_soft_fusion | raw_soft_fusion | 3 | 20000 | 0.346525 | 0.003262 | 1.388554 | 0.016225 | 0.041035 | 0.003262 | 0.356317 | 0.023181 | 1.051701 | 0.020309 | 0.653459 | 0.012619 | 0.548390 | 0.008799 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/visual_pooling_diagnostic_summary.csv |
| P2d | diagnostic_balanced | mean_patch_plus_aux | mean_patch_plus_aux_raw_soft_fusion | raw_soft_fusion | 3 | 20000 | 0.346638 | 0.001640 | 1.378566 | 0.020956 | 0.041148 | 0.001640 | 0.404933 | 0.094110 | 1.042023 | 0.013289 | 0.647445 | 0.008257 | 0.552775 | 0.007238 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat/round1_concat_diagnostic_summary.csv |
| P2b | diagnostic_balanced | visual_cls_mean_concat | visual_cls_mean_concat_raw_soft_fusion | raw_soft_fusion | 3 | 20000 | 0.348457 | 0.008445 | 1.359039 | 0.023454 | 0.042966 | 0.008445 | 0.461067 | 0.105146 | 1.036698 | 0.024174 | 0.644137 | 0.015020 | 0.559072 | 0.009164 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/visual_pooling_diagnostic_summary.csv |
| P2b | diagnostic_balanced | visual_cls_only | visual_cls_only_raw_soft_fusion | raw_soft_fusion | 3 | 20000 | 0.349310 | 0.004724 | 1.369623 | 0.055323 | 0.043820 | 0.004724 | 0.350133 | 0.029228 | 1.043780 | 0.020659 | 0.648537 | 0.012836 | 0.553225 | 0.007214 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/visual_pooling_diagnostic_summary.csv |
| P1 | diagnostic_balanced | p1_round0_visual_baseline | visual_router_raw_soft_fusion | raw_soft_fusion | 1 | 20000 | 0.388991 | 0.000000 | 1.326902 | 0.000000 | 0.083501 | 0.000000 | 0.430350 | 0.000000 | 1.235514 | 0.000000 | 0.767668 | 0.000000 | 0.472951 | 0.000000 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/round0_diagnostic_balanced_summary.csv |
| P2c | diagnostic_balanced | revin_aux_only_fusion_huber_kl | revin_aux_only_fusion_huber_kl_raw_soft_fusion | raw_soft_fusion | 3 | 20000 | 0.389291 | 0.000497 | 1.476549 | 0.023055 | 0.083801 | 0.000497 | 0.254850 | 0.002685 | 1.231025 | 0.010868 | 0.764879 | 0.006752 | 0.456420 | 0.005065 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_aux_only/aux_only_diagnostic_summary.csv |

## 口径说明

- 主选择指标固定为 pilot_selection raw-soft MAE_mean 最低。
- diagnostic_balanced 只用于诊断和 tie-breaker，不参与主选择。
- oracle-label accuracy 只作解释指标，不作为主选择指标。
