# P2c RevIN aux-only ablation 汇总

- 生成时间：2026-06-21 09:51:24 CST
- 输出目录：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_aux_only`
- 脚本版本：`visual_router_v2_round1_aux_only_v1`
- seeds：[16, 17, 18]
- best seed：{'selection_rule': 'min pilot_selection raw_soft_fusion MAE', 'seed': 17, 'best_epoch': 3, 'selection_raw_soft_MAE': 0.33280747856729237, 'selection_raw_soft_MSE': 1.5455269063081547}
- 输入特征：仅 P2a `revin_aux`，字段顺序为 `['mean', 'log_std', 'min', 'max', 'range', 'clip_ratio']`。
- 训练选择：scaler 只在 `pilot_train` fit；best epoch/seed 只按 `pilot_selection` raw-soft MAE 选择；`diagnostic_balanced` 不参与选择。

## Selection mean/std

| sample_set | method | seed_count | sample_count | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_selection | revin_aux_only_fusion_huber_kl_hard_top1 | 3 | 30000 | 0.351776 | 0.000113 | 1.615362 | 0.085622 | 0.085045 | 0.000113 | 0.323044 | 0.001840 | 1.277712 | 0.008272 | 0.793887 | 0.005140 | 0.424578 | 0.004765 |
| pilot_selection | revin_aux_only_fusion_huber_kl_raw_soft_fusion | 3 | 30000 | 0.332987 | 0.000310 | 1.510008 | 0.064787 | 0.066256 | 0.000310 | 0.323044 | 0.001840 | 1.277712 | 0.008272 | 0.793887 | 0.005140 | 0.424578 | 0.004765 |

## Diagnostic mean/std

| sample_set | method | seed_count | sample_count | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diagnostic_balanced | revin_aux_only_fusion_huber_kl_hard_top1 | 3 | 20000 | 0.417724 | 0.001056 | 1.620599 | 0.059919 | 0.112234 | 0.001056 | 0.254850 | 0.002685 | 1.231025 | 0.010868 | 0.764879 | 0.006752 | 0.456420 | 0.005065 |
| diagnostic_balanced | revin_aux_only_fusion_huber_kl_raw_soft_fusion | 3 | 20000 | 0.389291 | 0.000497 | 1.476549 | 0.023055 | 0.083801 | 0.000497 | 0.254850 | 0.002685 | 1.231025 | 0.010868 | 0.764879 | 0.006752 | 0.456420 | 0.005065 |

## 验收问题回答

1. aux-only 是否明显优于 P1 Round 0 Visual baseline？Round 0 visual selection hard MAE=0.356267，raw-soft MAE=0.334069。 当前 P2c selection hard MAE=0.351776，raw-soft MAE=0.332987。aux-only raw-soft MAE 优于 P1 Round 0 visual baseline，说明 RevIN 删除的尺度信息具有较强路由价值。
2. aux-only 是否接近或超过 P2b visual-only 结果？未接近。P2b 最佳 visual-only 变体为 `visual_mean_patch_only`，selection raw-soft MAE mean=0.300996、hard top-1 MAE mean=0.319706；P2c aux-only selection raw-soft MAE mean=0.332987、hard top-1 MAE mean=0.351776，明显落后于 P2b visual-only。
3. RevIN 删除掉的尺度信息是否是当前 Visual Router 的主要瓶颈之一？若 aux-only 接近或优于 visual baseline，则它至少是重要瓶颈之一；若仅小幅改善，则应视为可补充信号而非唯一主因。以上判断以 selection mean/std 为主，diagnostic 只看方向一致性。
4. 是否建议 P2d 做 visual+aux concat？建议做。P2c 只用 6 维尺度统计已形成独立路由信号，P2d 可以检验该信号与视觉 embedding 是否互补；但 P2d 必须继续保持 selection 选择、diagnostic 诊断和不使用 pilot_test 选择的边界。

## 约束确认

- 未读取 visual embedding 作为训练输入。
- 未重新生成 P2a features。
- 未使用完整 17 维 TimeFuse feature。
- 未使用 pilot_test。
