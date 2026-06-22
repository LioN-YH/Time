# Visual Router V2 Round2 panel-wise pooling architecture probe

生成时间：2026-06-22 14:42:26 CST

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

- sample_sets=['round2_train_small', 'round2_selection_small', 'round2_diagnostic_balanced_small', 'round2_test_small']，max_samples_per_set=None。
- feature_shapes_by_sample_set={'round2_train_small': {'global_mean_patch': [20000, 768], 'panel_mean_concat': [20000, 2304], 'global_plus_panel_mean': [20000, 3072], 'panel_variance': [20000, 768], 'revin_aux': [20000, 6]}, 'round2_selection_small': {'global_mean_patch': [5000, 768], 'panel_mean_concat': [5000, 2304], 'global_plus_panel_mean': [5000, 3072], 'panel_variance': [5000, 768], 'revin_aux': [5000, 6]}, 'round2_diagnostic_balanced_small': {'global_mean_patch': [5000, 768], 'panel_mean_concat': [5000, 2304], 'global_plus_panel_mean': [5000, 3072], 'panel_variance': [5000, 768], 'revin_aux': [5000, 6]}, 'round2_test_small': {'global_mean_patch': [5000, 768], 'panel_mean_concat': [5000, 2304], 'global_plus_panel_mean': [5000, 3072], 'panel_variance': [5000, 768], 'revin_aux': [5000, 6]}}。
- finite_check=True，dtype=float32。
- global mean patch 重构最大误差：0.0。
- panel cosine/L2 统计：{'global_mean_reconstruct_max_abs_error': 0.0, 'line_vs_fold_cosine_mean': 0.800440788269043, 'line_vs_fold_cosine_std': 0.06648741024637232, 'line_vs_fold_l2_mean': 13.587570190429688, 'line_vs_fold_l2_std': 2.112132400454123, 'fold_vs_fft_cosine_mean': 0.7568016052246094, 'fold_vs_fft_cosine_std': 0.0708500311618438, 'fold_vs_fft_l2_mean': 15.814743041992188, 'fold_vs_fft_l2_std': 2.4094789018732863, 'line_vs_fft_cosine_mean': 0.782008171081543, 'line_vs_fft_cosine_std': 0.11499737471619696, 'line_vs_fft_l2_mean': 13.271240234375, 'line_vs_fft_l2_std': 4.665087141652068}。

## 初步判断

当前步骤完成 panel pooling feature cache/probe；panel means 之间存在稳定数值差异，说明该 feature 构造可用于后续 router screening。性能结论必须以独立训练汇总为准，不能只凭 feature smoke 把 panel-wise pooling 写成主线结论，也不能直接进入 65k expanded validation 或 full-scale。本次 35k small screening 的训练结论见 `panel_pooling_35k_screening_summary.md`：selection raw-soft MAE 仍由 `film_mean_patch_aux` baseline 最优，因此 panel-wise pooling 暂保留为 side branch，不进入 65k。

