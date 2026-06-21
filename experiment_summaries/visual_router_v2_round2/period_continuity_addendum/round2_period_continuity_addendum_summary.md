# Visual Router V2 Round2d addendum: spatial_panel_3view period continuity diagnostic

生成时间：2026-06-22 02:06:38 CST

## 结论
1. `spatial_panel_3view` 是否也受 hard top1 period fold 扰动影响：是。
2. 相比 `current_rgb_3view`，spatial panel 是否降低 image / embedding / router weight 的不连续传播：部分降低：image_cos、CLS/mean-patch embedding 与 selected flip 低于 current，但 router weight JS/L1 未降低，因此不是完整阻断 router 侧不连续传播。
3. 相比 `top3fold_period_layout`，spatial panel 的连续性：整体弱于 top3fold 的 image/embedding/router 连续性，但 selected flip 低于 top3fold。
4. spatial panel 的 fold panel 是否成为高变化来源：是；changed/unchanged 对照={"changed_image_cos_mean": 0.10822960144176379, "unchanged_image_cos_mean": 0.0009186129870891708, "changed_weight_js_mean": 0.20266235228368593, "unchanged_weight_js_mean": 0.0018404143738877591, "changed_selected_flip_rate": 0.04305135951661632, "unchanged_selected_flip_rate": 0.01264}。
5. `spatial_panel_3view` 作为 Round2c best layout 是否仍应进入 65k expanded validation：应进入。
6. 是否需要在 65k 前实现 `period_soft_mixture`：不作为前置硬门槛。
7. 65k expanded validation 推荐 layout 保持：`spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout`。

## Layout Comparison

| layout | top1_changed | top3_jaccard | image_cos | mean_patch_cos | cls_cos | weight_js | weight_l1 | selected_flip | fused_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_rgb_3view | 0.095775 | 0.914157 | 0.012659 | 0.060437 | 0.044292 | 0.020471 | 0.121888 | 0.042969 | 0.003293 |
| spatial_panel_3view | 0.095775 | 0.914157 | 0.011196 | 0.024991 | 0.031178 | 0.021074 | 0.126698 | 0.015553 | 0.003997 |
| top3fold_period_layout | 0.095775 | 0.914157 | 0.000262 | 0.005865 | 0.004715 | 0.001715 | 0.018025 | 0.037507 | 0.003129 |

## High-Risk Strata

| source_result | layout | sample_set | perturbation_sigma | stratum_value | row_count | top1_period_changed_mean | top1_period_changed_std | top1_period_changed_max | image_cosine_distance_mean | image_cosine_distance_std | image_cosine_distance_max | mean_patch_embedding_cosine_distance_mean | mean_patch_embedding_cosine_distance_std | mean_patch_embedding_cosine_distance_max | router_weight_js_divergence_mean | router_weight_js_divergence_std | router_weight_js_divergence_max | selected_model_flipped_mean | selected_model_flipped_std | selected_model_flipped_max | soft_fused_abs_change_mean | soft_fused_abs_change_std | soft_fused_abs_change_max | stratum_column |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| existing_round2d | top3fold_period_layout | round2_diagnostic_balanced_small | 0.010000 | q1 | 423 | 0.881797 | 0.323231 | True | 0.000001 | 0.000000 | 0.000002 | 0.003951 | 0.002148 | 0.013140 | 0.000292 | 0.000245 | 0.001797 | 0.666667 | 0.471963 | True | 0.000009 | 0.000008 | 0.000053 | error_gap_quantile |
| existing_round2d | top3fold_period_layout | round2_selection_small | 0.010000 | q1 | 468 | 0.858974 | 0.348420 | True | 0.000001 | 0.000000 | 0.000002 | 0.003611 | 0.002184 | 0.016510 | 0.000263 | 0.000246 | 0.002190 | 0.596154 | 0.491192 | True | 0.000116 | 0.000675 | 0.008586 | error_gap_quantile |
| existing_round2d | top3fold_period_layout | round2_selection_small | 0.010000 | p_le_4 | 783 | 0.546616 | 0.498140 | True | 0.000796 | 0.003050 | 0.020305 | 0.020532 | 0.082353 | 0.750051 | 0.005760 | 0.046112 | 0.640758 | 0.376756 | 0.484883 | True | 0.004778 | 0.029517 | 0.339654 | top1_period_bucket |
| existing_round2d | top3fold_period_layout | round2_diagnostic_balanced_small | 0.010000 | p_le_4 | 837 | 0.477897 | 0.499810 | True | 0.000775 | 0.003053 | 0.020452 | 0.025772 | 0.092266 | 0.648938 | 0.009861 | 0.062241 | 0.605106 | 0.364397 | 0.481548 | True | 0.011222 | 0.060106 | 0.547883 | top1_period_bucket |
| existing_round2d | current_rgb_3view | round2_diagnostic_balanced_small | 0.010000 | q1 | 423 | 0.881797 | 0.323231 | True | 0.136075 | 0.048687 | 0.252983 | 0.617985 | 0.136725 | 0.738607 | 0.209280 | 0.075202 | 0.330715 | 0.328605 | 0.470262 | True | 0.000567 | 0.000140 | 0.000672 | error_gap_quantile |
| existing_round2d | current_rgb_3view | round2_diagnostic_balanced_small | 0.005000 | q1 | 423 | 0.898345 | 0.302552 | True | 0.130596 | 0.048295 | 0.234959 | 0.612244 | 0.137028 | 0.749190 | 0.209876 | 0.073886 | 0.335338 | 0.323877 | 0.468508 | True | 0.000568 | 0.000137 | 0.000671 | error_gap_quantile |
| existing_round2d | current_rgb_3view | round2_selection_small | 0.001000 | q1 | 468 | 0.863248 | 0.343953 | True | 0.122588 | 0.053075 | 0.261013 | 0.588355 | 0.175873 | 0.754390 | 0.206188 | 0.080581 | 0.335093 | 0.322650 | 0.467990 | True | 0.000576 | 0.000154 | 0.001602 | error_gap_quantile |
| existing_round2d | current_rgb_3view | round2_selection_small | 0.005000 | q1 | 468 | 0.863248 | 0.343953 | True | 0.127562 | 0.053732 | 0.259627 | 0.593841 | 0.176602 | 0.739818 | 0.202029 | 0.079666 | 0.336148 | 0.322650 | 0.467990 | True | 0.000671 | 0.000602 | 0.005742 | error_gap_quantile |
| existing_round2d | current_rgb_3view | round2_selection_small | 0.010000 | q1 | 468 | 0.858974 | 0.348420 | True | 0.126265 | 0.052635 | 0.241325 | 0.596262 | 0.176442 | 0.729095 | 0.207708 | 0.081266 | 0.337750 | 0.318376 | 0.466345 | True | 0.000857 | 0.001704 | 0.016296 | error_gap_quantile |
| existing_round2d | current_rgb_3view | round2_diagnostic_balanced_small | 0.001000 | q1 | 423 | 0.865248 | 0.341863 | True | 0.126892 | 0.051939 | 0.311411 | 0.598787 | 0.162184 | 0.754390 | 0.206870 | 0.079088 | 0.335093 | 0.316785 | 0.465774 | True | 0.000557 | 0.000161 | 0.000671 | error_gap_quantile |
| existing_round2d | top3fold_period_layout | round2_diagnostic_balanced_small | 0.010000 | ES | 1080 | 0.362963 | 0.481077 | True | 0.000341 | 0.002690 | 0.034366 | 0.006132 | 0.027546 | 0.257752 | 0.001277 | 0.009835 | 0.168362 | 0.271296 | 0.444835 | True | 0.003896 | 0.020787 | 0.287242 | oracle_model |
| existing_round2d | top3fold_period_layout | round2_selection_small | 0.010000 | ES | 1161 | 0.347976 | 0.476534 | True | 0.000169 | 0.001329 | 0.016057 | 0.006298 | 0.036333 | 0.434738 | 0.001311 | 0.011992 | 0.216984 | 0.263566 | 0.440756 | True | 0.002720 | 0.010226 | 0.143387 | oracle_model |

## Metadata

```json
{
  "status": "completed",
  "script_version": "visual_router_v2_round2d_period_continuity_v1",
  "generated_at": "2026-06-22 02:06:38 CST",
  "round2_stage": "period_continuity_addendum",
  "trained_new_model": false,
  "built_feature_cache": false,
  "ran_vit_for_embedding_diagnostic": true,
  "saved_pseudo_image_tensor": false,
  "used_test_small_for_selection": false,
  "loaded_116m_prediction_manifest_to_memory": false,
  "layouts_diagnosed": [
    "spatial_panel_3view"
  ],
  "compared_against_existing": [
    "current_rgb_3view",
    "top3fold_period_layout"
  ],
  "compare_with_existing_raw_results": "/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_period_continuity/round2_period_continuity_raw_results.csv",
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
  "next_step_recommendation": "direct_65k_with_spatial_current_top3fold"
}
```

## 下一步推荐

- 直接进入 65k expanded validation，候选保持 `spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout`。
- `period_soft_mixture` 作为后续表达改进单独 smoke，不阻塞本轮 65k。
