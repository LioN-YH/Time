# Visual Router V2 Round2d period continuity diagnostic

生成时间：2026-06-22 01:25:54 CST

## 结论
- `current_rgb_3view` hard top1 fold 对轻微扰动是否敏感：是。
- `top3fold_period_layout` 是否比 current hard top1 更连续：是。
- 周期选择变化是否放大 image / embedding / router weight 变化：{"changed_image_cos_mean": 0.06194966271755788, "unchanged_image_cos_mean": 0.0005830303637663705, "changed_weight_js_mean": 0.09912350738602844, "unchanged_weight_js_mean": 0.0017688161540695324}
- 是否必须先实现 `period_soft_mixture` 再做 65k：不是必须，可作为后续改进。
- `top3fold_period_layout` 是否建议进入 65k expanded validation：建议进入。
- top3fold 的 diagnostic-balanced 优势是否可能来自更稳定的周期表达：可能；本诊断中 top3fold 的 image/embedding/router JS 平均变化低于 current。
- 后续建议：保留 period tokens / soft period mixture / panelized top3fold 为 Round2e 候选；若 65k 资源有限，优先 `spatial_panel_3view`、`current_rgb_3view`，并按本诊断决定是否纳入 `top3fold_period_layout`。
- latency / GPU memory：本诊断未保存大 tensor，单任务按小 batch 运行；若所有任务完成且无 OOM，未发现阻塞 65k 规划的显著显存问题。

## 指定 Strata 回答

- `oracle_model` 高风险候选：ES。
- `season_strength_cat` 高风险候选：strong, moderate。
- `forecastability_cat` 高风险候选：medium, high。
- `error_gap_quantile` 高风险候选：q1。
- `top1_period_bucket` 高风险候选：p_le_4。
- 指定关注 `oracle_model` @ sigma=0.01：current_rgb_3view/CrossFormer: flip=0.0297, JS=0.000262, image_cos=0.000172; current_rgb_3view/PatchTST: flip=0.0170, JS=0.000249, image_cos=0.000184; top3fold_period_layout/CrossFormer: flip=0.0134, JS=0.000461, image_cos=0.000055; top3fold_period_layout/PatchTST: flip=0.0058, JS=0.000334, image_cos=0.000124。
- 指定关注 `season_strength_cat` @ sigma=0.01：current_rgb_3view/strong: flip=0.0644, JS=0.027210, image_cos=0.017259; top3fold_period_layout/strong: flip=0.1016, JS=0.003150, image_cos=0.000333。
- 指定关注 `error_gap_quantile` @ sigma=0.01：current_rgb_3view/q5: flip=0.0291, JS=0.001720, image_cos=0.000812; top3fold_period_layout/q5: flip=0.0310, JS=0.006781, image_cos=0.000939。
- 本次按 selected model flip 与 router JS 排序的最高风险 error_gap bucket 是 `q1`；`q5` 指标已单独列出，详细数值见 `round2_period_stratified_summary.csv`。

## Layout Mean Metrics

| layout | top1_changed | top3_jaccard | image_cos | mean_patch_cos | weight_js | selected_flip | fused_abs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| current_rgb_3view | 0.095775 | 0.914157 | 0.012659 | 0.060437 | 0.020471 | 0.042969 | 0.003293 |
| top3fold_period_layout | 0.095775 | 0.914157 | 0.000262 | 0.005865 | 0.001715 | 0.037507 | 0.003129 |

## High-Risk Strata

| layout | sample_set | perturbation_sigma | stratum_value | row_count | top1_period_changed_mean | top1_period_changed_std | top1_period_changed_max | image_cosine_distance_mean | image_cosine_distance_std | image_cosine_distance_max | mean_patch_embedding_cosine_distance_mean | mean_patch_embedding_cosine_distance_std | mean_patch_embedding_cosine_distance_max | router_weight_js_divergence_mean | router_weight_js_divergence_std | router_weight_js_divergence_max | selected_model_flipped_mean | selected_model_flipped_std | selected_model_flipped_max | soft_fused_abs_change_mean | soft_fused_abs_change_std | soft_fused_abs_change_max | stratum_column |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top3fold_period_layout | round2_diagnostic_balanced_small | 0.010000 | q1 | 423 | 0.881797 | 0.323231 | True | 0.000001 | 0.000000 | 0.000002 | 0.003951 | 0.002148 | 0.013140 | 0.000292 | 0.000245 | 0.001797 | 0.666667 | 0.471963 | True | 0.000009 | 0.000008 | 0.000053 | error_gap_quantile |
| top3fold_period_layout | round2_selection_small | 0.010000 | q1 | 468 | 0.858974 | 0.348420 | True | 0.000001 | 0.000000 | 0.000002 | 0.003611 | 0.002184 | 0.016510 | 0.000263 | 0.000246 | 0.002190 | 0.596154 | 0.491192 | True | 0.000116 | 0.000675 | 0.008586 | error_gap_quantile |
| top3fold_period_layout | round2_selection_small | 0.010000 | p_le_4 | 783 | 0.546616 | 0.498140 | True | 0.000796 | 0.003050 | 0.020305 | 0.020532 | 0.082353 | 0.750051 | 0.005760 | 0.046112 | 0.640758 | 0.376756 | 0.484883 | True | 0.004778 | 0.029517 | 0.339654 | top1_period_bucket |
| top3fold_period_layout | round2_diagnostic_balanced_small | 0.010000 | p_le_4 | 837 | 0.477897 | 0.499810 | True | 0.000775 | 0.003053 | 0.020452 | 0.025772 | 0.092266 | 0.648938 | 0.009861 | 0.062241 | 0.605106 | 0.364397 | 0.481548 | True | 0.011222 | 0.060106 | 0.547883 | top1_period_bucket |
| current_rgb_3view | round2_diagnostic_balanced_small | 0.010000 | q1 | 423 | 0.881797 | 0.323231 | True | 0.136075 | 0.048687 | 0.252983 | 0.617985 | 0.136725 | 0.738607 | 0.209280 | 0.075202 | 0.330715 | 0.328605 | 0.470262 | True | 0.000567 | 0.000140 | 0.000672 | error_gap_quantile |
| current_rgb_3view | round2_diagnostic_balanced_small | 0.005000 | q1 | 423 | 0.898345 | 0.302552 | True | 0.130596 | 0.048295 | 0.234959 | 0.612244 | 0.137028 | 0.749190 | 0.209876 | 0.073886 | 0.335338 | 0.323877 | 0.468508 | True | 0.000568 | 0.000137 | 0.000671 | error_gap_quantile |
| current_rgb_3view | round2_selection_small | 0.001000 | q1 | 468 | 0.863248 | 0.343953 | True | 0.122588 | 0.053075 | 0.261013 | 0.588355 | 0.175873 | 0.754390 | 0.206188 | 0.080581 | 0.335093 | 0.322650 | 0.467990 | True | 0.000576 | 0.000154 | 0.001602 | error_gap_quantile |
| current_rgb_3view | round2_selection_small | 0.005000 | q1 | 468 | 0.863248 | 0.343953 | True | 0.127562 | 0.053732 | 0.259627 | 0.593841 | 0.176602 | 0.739818 | 0.202029 | 0.079666 | 0.336148 | 0.322650 | 0.467990 | True | 0.000671 | 0.000602 | 0.005742 | error_gap_quantile |
| current_rgb_3view | round2_selection_small | 0.010000 | q1 | 468 | 0.858974 | 0.348420 | True | 0.126265 | 0.052635 | 0.241325 | 0.596262 | 0.176442 | 0.729095 | 0.207708 | 0.081266 | 0.337750 | 0.318376 | 0.466345 | True | 0.000857 | 0.001704 | 0.016296 | error_gap_quantile |
| current_rgb_3view | round2_diagnostic_balanced_small | 0.001000 | q1 | 423 | 0.865248 | 0.341863 | True | 0.126892 | 0.051939 | 0.311411 | 0.598787 | 0.162184 | 0.754390 | 0.206870 | 0.079088 | 0.335093 | 0.316785 | 0.465774 | True | 0.000557 | 0.000161 | 0.000671 | error_gap_quantile |
| top3fold_period_layout | round2_diagnostic_balanced_small | 0.010000 | ES | 1080 | 0.362963 | 0.481077 | True | 0.000341 | 0.002690 | 0.034366 | 0.006132 | 0.027546 | 0.257752 | 0.001277 | 0.009835 | 0.168362 | 0.271296 | 0.444835 | True | 0.003896 | 0.020787 | 0.287242 | oracle_model |
| top3fold_period_layout | round2_selection_small | 0.010000 | ES | 1161 | 0.347976 | 0.476534 | True | 0.000169 | 0.001329 | 0.016057 | 0.006298 | 0.036333 | 0.434738 | 0.001311 | 0.011992 | 0.216984 | 0.263566 | 0.440756 | True | 0.002720 | 0.010226 | 0.143387 | oracle_model |

## Metadata

```json
{
  "status": "completed",
  "script_version": "visual_router_v2_round2d_period_continuity_v1",
  "generated_at": "2026-06-22 01:25:54 CST",
  "round2_stage": "period_continuity_diagnostic",
  "trained_new_model": false,
  "built_feature_cache": false,
  "ran_vit_for_embedding_diagnostic": true,
  "saved_pseudo_image_tensor": false,
  "used_test_small_for_selection": false,
  "loaded_116m_prediction_manifest_to_memory": false,
  "layouts_diagnosed": [
    "current_rgb_3view",
    "top3fold_period_layout"
  ],
  "seeds": [
    16,
    17,
    18
  ],
  "sample_sets": [
    "round2_selection_small",
    "round2_diagnostic_balanced_small"
  ],
  "sample_manifest": "/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_small_samples/round2_small_sample_manifest.csv",
  "round2c_dir": "/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_screening",
  "perturbation_sigma_list": [
    0.001,
    0.005,
    0.01
  ],
  "num_perturbations": 3,
  "perturbation_space": "pre_revin_history_x",
  "devices_requested": [
    "cuda:0",
    "cuda:1",
    "cuda:2",
    "cuda:3"
  ],
  "devices_used": [
    "cuda:0",
    "cuda:1",
    "cuda:2",
    "cuda:3"
  ],
  "parallel_backend": "process_per_layout_seed_sample_set",
  "single_task_output_isolated": true,
  "period_soft_mixture_implemented": false,
  "next_step_recommendation": "direct_65k_with_top3fold"
}
```

## 下一步推荐

- 若 `top3fold_period_layout` 的 flip/JS 指标不高于 current，可与 `spatial_panel_3view`、`current_rgb_3view` 一起进入 65k expanded validation。
- 若 hard top1 周期 flip 明显放大 router weight 或 selected model 跳变，先实现 `period_soft_mixture` smoke，再决定是否做完整 head。
- panelized top3fold 和 period tokens 更适合作为后续表达增强，不应阻塞当前 65k 除非本诊断显示 hard fold 极不稳定。
