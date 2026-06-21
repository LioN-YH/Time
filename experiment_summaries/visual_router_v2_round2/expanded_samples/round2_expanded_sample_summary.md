# Visual Router V2 Round2e-a Expanded Samples Summary

生成时间：2026-06-22 03:16:13 CST

## 为什么先构建 65k expanded samples

Round2a/Round2c 只冻结并验证了 35k small screening 边界。后续 Round2e-b 要比较 `spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout` 的 65k layout validation，必须先固定 train/selection/diagnostic/test expanded 边界，避免 feature cache、router 训练和 layout 选择过程中重新抽样造成口径漂移。

## Expanded sample sets

| sample_set | split | count | 用途 |
| --- | --- | ---: | --- |
| round2_train_expanded | vali | 30000 | 后续 65k fixed FiLM 风格 router 训练 |
| round2_selection_expanded | vali | 10000 | 后续 layout/seed/epoch/hparam 选择；不含 train |
| round2_diagnostic_balanced_expanded | vali | 10000 | oracle expert balanced 诊断，不用于选择 |
| round2_test_expanded | test | 15000 | frozen expanded validation only，不用于训练、调参或选择 |

验证状态：`passed`；跨集合 sample_key 重复数：`0`；train/selection 交集：`0`。

## Small subset

35k small samples 是否是 65k expanded samples 的子集：`True`。

| small set | expanded set | small_in_expanded | ratio | strict_subset |
| --- | --- | ---: | ---: | --- |
| round2_train_small | round2_train_expanded | 20000/20000 | 1.000000 | True |
| round2_selection_small | round2_selection_expanded | 5000/5000 | 1.000000 | True |
| round2_diagnostic_balanced_small | round2_diagnostic_balanced_expanded | 5000/5000 | 1.000000 | True |
| round2_test_small | round2_test_expanded | 5000/5000 | 1.000000 | True |

本次通过先保留 small 边界、再稳定哈希补齐的方式保证四个 small set 均为对应 expanded set 的严格子集。

## Diagnostic balance

`round2_diagnostic_balanced_expanded` 保持 oracle expert balanced：CrossFormer=2000, DLinear=2000, ES=2000, NaiveForecaster=2000, PatchTST=2000。该集合只用于诊断，不参与 layout/seed/epoch/hparam 选择。

## Test boundary

`round2_test_expanded` 全部来自 `test` split，且只用于 frozen expanded validation；metadata 中 `used_test_expanded_for_selection=false`。

## Round2e-b recommendation

后续 Round2e-b 应验证以下 layout：`spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout`。

Round2e-b 应继续固定 `film_mean_patch_aux` 风格后端，即 mean_patch visual embedding + RevIN aux FiLM modulation，避免把 layout 效果与 head/hparam 改动混在一起。

Round2e-b 应继续使用多 GPU 进程级并行：feature cache 可按 layout 分配 GPU，training/eval 可按 layout×seed 分配 GPU；本步没有启动 GPU、ViT、feature cache 或 router 训练。
