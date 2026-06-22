# Visual Router V2 Round2 panel-wise pooling architecture probe

生成时间：2026-06-22 12:57:06 CST

## 目的

本探针只研究 `spatial_panel_3view` 下 ViT patch embedding 是否可以按 view-region 分别 pooling，避免 global `mean_patch` 在后端再次混合 line/fold/FFT 三个 view。

## 边界

- 本窗口不做 full-scale validation，不启动 1M/116M 长跑，不修改 full-scale streaming pipeline。
- 本步骤只保存 pooled feature 与 RevIN aux，不保存 pseudo image tensor。
- `test_small` 若后续训练使用，只能做 frozen screening，不能用于选择 variant/seed/epoch。

## Region Mapping

- image_size=224，patch_size=16，patch_grid=[14, 14]。
- spatial panel 宽度：[74, 74, 76]，对应 line/fold/FFT 三个水平区域。
- 严格内部 patch 数：168；忽略跨边界 patch 数：28，ignored_patch_cols=[4, 9]。
- 由于 panel 边界落在 ViT patch 内，默认忽略边界列，保留 global mean_patch 作为 fallback 与 baseline。

## 候选结构

- `global_mean_patch`：当前 `film_mean_patch_aux` baseline 的视觉输入。
- `panel_mean_concat`：line/fold/FFT 三个 panel mean 直接 concat，形成 `film_panel_mean_aux`。
- `global_plus_panel_mean`：global mean 与三个 panel mean concat，形成 `film_global_panel_mean_aux`。
- `panel_variance`：仅作为轻量 disagreement probe，默认不作为主线训练输入。

## Smoke 结果

- sample_sets=['round2_train_small', 'round2_selection_small', 'round2_diagnostic_balanced_small', 'round2_test_small']，max_samples_per_set=32。
- feature_shapes={'global_mean_patch': [32, 768], 'panel_mean_concat': [32, 2304], 'global_plus_panel_mean': [32, 3072], 'panel_variance': [32, 768]}。
- finite_check=True，dtype=float32。
- global mean patch 重构最大误差：0.0。
- panel cosine/L2 统计：{'global_mean_reconstruct_max_abs_error': 0.0, 'line_vs_fold_cosine_mean': 0.8071441650390625, 'line_vs_fold_cosine_std': 0.0699643399712062, 'line_vs_fold_l2_mean': 13.3173828125, 'line_vs_fold_l2_std': 2.1937860904344104, 'fold_vs_fft_cosine_mean': 0.76446533203125, 'fold_vs_fft_cosine_std': 0.06995808500250007, 'fold_vs_fft_l2_mean': 15.52001953125, 'fold_vs_fft_l2_std': 2.3991800954146325, 'line_vs_fft_cosine_mean': 0.8016357421875, 'line_vs_fft_cosine_std': 0.11785962950572673, 'line_vs_fft_l2_mean': 12.4813232421875, 'line_vs_fft_l2_std': 4.90601263109866}。

## 初步判断

当前步骤完成 architecture probe 和 very-small feature smoke；panel means 之间存在稳定数值差异，说明该方向值得进入 35k small screening。尚未运行 35k training，因此不能把 panel-wise pooling 写成主线结论，也不建议直接进入 65k expanded validation 或 full-scale。若后续 35k 中 `film_panel_mean_aux` 或 `film_global_panel_mean_aux` 在 selection raw-soft MAE、tail regret 和 CrossFormer/PatchTST strata 上稳定优于 `film_mean_patch_aux`，才建议进入 65k expanded validation。

