# Visual Router V2 Round2 Small Screening Summary

生成时间：2026-06-21 21:18:57 CST

## 本步产物

本步只冻结 Round2 view layout small screening 的样本集、layout candidate registry 和 top3fold 复用审计。未训练 router，未运行 ViT，未生成 feature cache，未保存 pseudo image tensor。

## 样本集

| sample_set | split | count | 用途 |
| --- | --- | ---: | --- |
| round2_train_small | vali | 20000 | 后续小样本 layout router 训练 |
| round2_selection_small | vali | 5000 | 后续 layout/seed/epoch/hparam 选择 |
| round2_diagnostic_balanced_small | vali | 5000 | oracle expert balanced 诊断，不参与选择 |
| round2_test_small | test | 5000 | frozen screening only，不参与训练或选择 |

验证状态：`passed`；跨集合 sample_key 重复数：`0`；train/selection 交集：`0`。

## Layout candidates

第一轮默认 layout set：current_rgb_3view, spatial_panel_3view, line_only, line_difference_band, fft_absolute_energy, top3fold_period_layout

第一轮暂缓：period_soft_mixture, independent_view_encoder

后端 router/head 固定为 Round1 最强路线：`film_mean_patch_aux`，即 mean_patch visual embedding + RevIN aux FiLM modulation；主指标仍为 raw-soft MAE / MSE / regret，oracle-label accuracy 只作解释指标。

## P0 overlap

P0 overlap 状态：`computed`。详细比例见 `round2_small_sample_metadata.json` 的 `p0_overlap` 字段。

## 后续耗时估计

- P2a 约 200k feature cache ≈ 5h；
- P2d/P2e final_test_only 约 75k feature cache/eval ≈ 2.5h；
- 35k samples 的单 layout small feature cache 预计约 1-1.5h；
- 5 个 layout 顺序跑约 5-8h；
- 若使用 3 张 GPU 进程级并行，wall time 预计约 2-3h；
- `independent_view_encoder` 若每个 view 单独过 ViT，可能接近 2-3 倍成本，不建议第一轮默认执行。
