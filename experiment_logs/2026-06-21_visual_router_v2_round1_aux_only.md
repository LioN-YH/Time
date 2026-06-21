# Visual Router V2 Round 1 P2c RevIN aux-only 消融

日志日期：2026-06-21 10:03:05 CST

## 目的

基于 P2a 已完成的 sharded feature cache，训练和评估 `revin_aux_only` 变体，判断 RevIN 删除掉的尺度信息本身是否具有专家路由价值。本步骤只做 RevIN aux-only，不做 visual-only pooling、visual+aux concat、feature probe、ViT 训练或 P2a feature 重建。

## 背景

P0 固定样本集位于 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples/`，包含 `pilot_train=150000`、`pilot_selection=30000`、`diagnostic_balanced=20000`。P1 Round 0 evaluator 输出位于 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/`，其中 Visual baseline selection raw-soft MAE=`0.334069`、hard top-1 MAE=`0.356267`。P2a feature cache 位于 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/`，每个 shard 保存 `cls_embedding`、`mean_patch_embedding` 和 `revin_aux`。

本实验只读取 P2a `.npz` shard 中的 `revin_aux`，字段顺序为 `mean/log_std/min/max/range/clip_ratio`。训练目标沿用当前 Visual Router 的 `fusion_huber_kl` 思路：router 权重融合五专家预测，主损失为 SmoothL1，KL 辅助目标来自训练 batch 的五专家误差 soft oracle。`StandardScaler` 只在 `pilot_train` fit，best epoch 和 best seed 只按 `pilot_selection` raw-soft MAE 选择，`diagnostic_balanced` 只用于诊断展示。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round1_aux_only.py`：
   - 只读取 P2a shard 的 `revin_aux`，不读取 `cls_embedding` / `mean_patch_embedding` 作为训练输入；
   - 按 P0 `order_index` 和 `sample_key` 严格校验 `pilot_train`、`pilot_selection`、`diagnostic_balanced` 对齐；
   - 为 P2c 输出目录构建专用 prediction subset SQLite index，覆盖 200000 个 P0 sample_key × 五专家，共 1000000 条记录；
   - 按 batch 读取 packed prediction arrays，训练 `revin_aux_only_fusion_huber_kl` 小型 MLP router；
   - 输出 hard top-1 与 raw soft fusion 的 MAE/MSE、regret、oracle-label accuracy、weight entropy、normalized entropy、mean max weight、selected_model counts 和分层 summary。
2. 使用 `quito` 环境完成语法检查：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round1_aux_only.py
   ```

3. 先运行 16 样本/集合、1 seed、1 epoch smoke 到独立目录 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_aux_only_smoke/`，确认 oracle subset、prediction subset index、训练、selection/diagnostic 评估和所有必需文件写出链路可用。
4. 首次正式后台任务在 oracle subset 之后出现高 CPU 长时间无日志。排查发现 `load_oracle_subset` 的 missing 检查在 list comprehension 内反复构造 20 万 sample_key 的 set，形成不必要的 O(n²) 开销。已停止该 P2c 进程，只清理/覆盖本 P2c 输出目录中的运行日志和 PID 记录，未删除 P0/P1/P2a/P2b 产物。
5. 修复为预先构造 `present_keys` 后重启正式任务，命令记录于 P2c 输出目录 `command.sh`，PID/PGID 和停止脚本记录在同目录：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round1_aux_only.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_aux_only \
     --seeds 16 17 18 \
     --epochs 3 \
     --batch-size 512 \
     --eval-batch-size 512 \
     --device auto \
     --overwrite
   ```

6. 正式任务使用 CUDA 运行，三 seeds 均完成 3 epochs。训练日志显示：
   - seed 16：epoch 3 selection raw-soft MAE=`0.333345`；
   - seed 17：epoch 3 selection raw-soft MAE=`0.332807`；
   - seed 18：epoch 3 selection raw-soft MAE=`0.332809`。
7. 完成后执行验收审计，检查必需文件、seed 数、sample_set、method、sample_count、metadata 约束和 SQLite index 完整性。

## 结果

正式输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_aux_only/
```

必需产物均已写出：

- `aux_only_variant_seed_results.csv`
- `aux_only_selection_comparison.csv`
- `aux_only_diagnostic_summary.csv`
- `aux_only_selected_model_counts.csv`
- `aux_only_stratified_summary.csv`
- `aux_only_best_seed.json`
- `aux_only_metadata.json`
- `aux_only_summary.md`

额外保留：

- `aux_only_epoch_history.csv`
- `aux_only_best_router.pt`
- `prediction_index_aux_only_p0.sqlite`
- `command.sh`、`main.log`、`pid.txt`、`pgid.txt`、`stop.sh`

验收审计结果：

```text
required_exists=true
seeds=[16,17,18]
sample_sets=[diagnostic_balanced,pilot_selection]
methods=[revin_aux_only_fusion_huber_kl_hard_top1,revin_aux_only_fusion_huber_kl_raw_soft_fusion]
pilot_test_used=false
input_features=P2a revin_aux only
scaler_fit_sample_set=pilot_train
sample_counts={pilot_train:150000,pilot_selection:30000,diagnostic_balanced:20000}
sqlite_count=1000000
sqlite_distinct_samples=200000
sqlite_models=五专家各200000条
best_seed=17
best_epoch=3
```

`pilot_selection` mean/std 主结果：

| method | MAE_mean | MSE_mean | regret_to_oracle_mean | oracle_label_accuracy_mean | normalized_weight_entropy_mean | mean_max_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hard top-1 | 0.351776 | 1.615362 | 0.085045 | 0.323044 | 0.793887 | 0.424578 |
| raw soft fusion | 0.332987 | 1.510008 | 0.066256 | 0.323044 | 0.793887 | 0.424578 |

`diagnostic_balanced` mean/std 诊断结果：

| method | MAE_mean | MSE_mean | regret_to_oracle_mean | oracle_label_accuracy_mean | normalized_weight_entropy_mean | mean_max_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hard top-1 | 0.417724 | 1.620599 | 0.112234 | 0.254850 | 0.764879 | 0.456420 |
| raw soft fusion | 0.389291 | 1.476549 | 0.083801 | 0.254850 | 0.764879 | 0.456420 |

与 P1 Round 0 Visual baseline 对比：

- Round 0 Visual selection hard top-1 MAE=`0.356267`，P2c aux-only hard top-1 MAE mean=`0.351776`；
- Round 0 Visual selection raw-soft MAE=`0.334069`，P2c aux-only raw-soft MAE mean=`0.332987`。

与 P2b visual-only 最佳结果对比：

- P2b `visual_mean_patch_only` selection raw-soft MAE mean=`0.300996`，hard top-1 MAE mean=`0.319706`；
- P2c aux-only selection raw-soft MAE mean=`0.332987`，hard top-1 MAE mean=`0.351776`；
- aux-only 明显落后于 P2b visual-only，但略优于 P1 Round 0 Visual baseline。

## 结论

P2c RevIN aux-only 消融已完成。六维 RevIN aux 在不使用视觉 embedding、不使用 17 维 TimeFuse feature 的情况下，selection raw-soft MAE 与 hard top-1 MAE 均略优于 P1 Round 0 Visual baseline，说明 RevIN 删除掉的尺度信息本身具有可测的专家路由价值。

不过，aux-only 明显不接近 P2b 最佳 visual-only pooling 结果。当前证据更支持“尺度信息是 Visual Router 的一个瓶颈和有用补充信号”，而不是“尺度信息单独解释主要性能缺口”。因此 P2d 做 visual+aux concat 有价值，尤其应检验 P2b 最佳 `visual_mean_patch_only` 与六维 RevIN aux 是否互补。

## 下一步方案

1. P2d visual+aux concat 建议默认使用 P2b 最佳 `visual_mean_patch_only` 作为视觉 pooling，并拼接本次验证过的六维 `revin_aux`。
2. P2d 继续保持 `pilot_train` 训练、`pilot_selection` 选择、`diagnostic_balanced` 诊断，不使用 `pilot_test` 做架构选择。
3. 若后续需要 final test eval，必须显式标记 `final_test_eval`，并确保不影响模型/epoch/variant 选择。
4. 若复用本次 P2c SQLite index，应只在 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_aux_only/` 下引用，不覆盖 P2b 或 P2a 输出。
