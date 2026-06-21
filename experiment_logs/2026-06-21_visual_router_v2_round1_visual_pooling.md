# Visual Router V2 Round 1 P2b Visual-only Pooling 消融

日志日期：2026-06-21 09:31:45 CST

## 目的

基于 P2a 已完成的 sharded feature cache，训练和评估三个 visual-only pooling 变体，判断当前 Visual Router 效果差是否与 ViT pooling 口径有关。本步骤只比较 `visual_cls_only`、`visual_mean_patch_only` 和 `visual_cls_mean_concat`，不做 RevIN aux-only、visual+aux concat、feature probe、ViT finetune 或 P2a feature 重建。

## 背景

P0 固定样本集已生成于 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples/`，包含 `pilot_train=150000`、`pilot_selection=30000`、`diagnostic_balanced=20000` 等固定 ordered sample keys。P1 Round 0 evaluator 输出位于 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/`，提供旧 Visual Router baseline 与 Round 0 selection/diagnostic 对照。P2a feature cache 位于 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/`，按 P0 `order_index` 保存 `cls_embedding`、`mean_patch_embedding` 和 `revin_aux`。

本轮 P2b 只读取 P2a `.npz` shards 中的视觉 embedding：CLS、patch-token mean pooling，以及二者现场 concat。训练目标沿用 Visual Router 的 `fusion_huber_kl` 思路，scaler 只在 `pilot_train` fit，best variant 只用 `pilot_selection` 选择，`diagnostic_balanced` 只用于诊断展示。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/visual_router_v2_round1_training.py`，实现 P2b 共享训练与评估 helper：
   - 按 P0 `order_index` 和 `sample_key` 校验并读取 P2a feature shard；
   - 支持 `visual_cls_only`、`visual_mean_patch_only`、`visual_cls_mean_concat`；
   - concat 仅在内存现场构造，不写回 P2a cache；
   - 使用 SQLite prediction subset index 按 batch 读取五专家 prediction arrays，并用 grouped packed `.npy` 读取降低重复 I/O；
   - 输出 hard top-1 与 raw soft fusion 的 MAE/MSE、regret、oracle-label accuracy、weight entropy、normalized entropy、mean max weight、selected_model count/ratio 和分层汇总所需字段。
2. 新增 `visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round1_visual_pooling.py`，作为 P2b 正式 CLI 入口：
   - 固定默认输入为 P0/P1/P2a 路径；
   - seeds 使用 `16,17,18`；
   - 每个 seed 训练 `3` epochs；
   - 默认输出 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/`；
   - 默认拒绝覆盖既有 P2b 产物，只有显式 `--overwrite` 才清理本脚本产物。
3. 使用 `quito` 环境进行语法检查：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/visual_router_v2_round1_training.py \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round1_visual_pooling.py
   ```

4. 先做 helper 级验证：从 `pilot_selection` 读取 32 个样本的 `visual_cls_mean_concat` feature，复用 Round 0 vali SQLite index 计算 hard/raw-soft fusion 指标，确认 feature shape 为 `(32, 1536)` 且 MAE/MSE 非空。
5. 正式后台任务通过 `setsid` 启动，命令写入 P2b 输出目录 `command.sh`：

   ```bash
   CUDA_VISIBLE_DEVICES=1 /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     /home/shiyuhong/Time-visual-router-v2/visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round1_visual_pooling.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling \
     --seeds 16,17,18 \
     --epochs 3 \
     --batch-size 256 \
     --eval-batch-size 512 \
     --device cuda
   ```

6. 正式任务先在 P2b 输出目录构建覆盖 `pilot_train`、`pilot_selection`、`diagnostic_balanced` 共 200000 个 sample_key 的 prediction SQLite 子集索引，避免误用只覆盖 selection/diagnostic 的 Round 0 index。
7. 正式任务完成后执行验收脚本，检查必需文件、metadata 约束、三变体三 seeds、selection/diagnostic 覆盖和 summary 中文回答。

## 结果

正式输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/
```

关键产物：

- `visual_pooling_variant_seed_results.csv`
- `visual_pooling_selection_comparison.csv`
- `visual_pooling_diagnostic_summary.csv`
- `visual_pooling_selected_model_counts.csv`
- `visual_pooling_stratified_summary.csv`
- `visual_pooling_best_variant.json`
- `visual_pooling_metadata.json`
- `visual_pooling_summary.md`
- 三个变体 × 三个 seeds 的 checkpoint 与 selection/diagnostic prediction CSV
- `prediction_index_p2b_train_selection_diagnostic.sqlite`

正式任务状态：

```text
status=completed
pilot_train=150000
pilot_selection=30000
diagnostic_balanced=20000
variants=visual_cls_only, visual_mean_patch_only, visual_cls_mean_concat
seeds=16,17,18
epochs=3
pilot_test_used=false
```

验收脚本输出：

```text
P2b verification passed {'seed_rows': 36, 'best_variant': 'visual_mean_patch_only', 'selection_rows': 6, 'diagnostic_rows': 6}
```

`pilot_selection` mean/std 主结果如下：

| variant | method | MAE_mean | MSE_mean | regret_to_oracle_mean | oracle_label_accuracy_mean |
| --- | --- | ---: | ---: | ---: | ---: |
| visual_mean_patch_only | raw soft fusion | 0.300996 | 1.234168 | 0.034265 | 0.378856 |
| visual_cls_only | raw soft fusion | 0.302048 | 1.220514 | 0.035317 | 0.376122 |
| visual_cls_mean_concat | raw soft fusion | 0.302220 | 1.217317 | 0.035489 | 0.549878 |
| visual_mean_patch_only | hard top-1 | 0.319706 | 1.351444 | 0.052975 | 0.378856 |
| visual_cls_only | hard top-1 | 0.320451 | 1.364566 | 0.053719 | 0.376122 |
| visual_cls_mean_concat | hard top-1 | 0.320836 | 1.391806 | 0.054105 | 0.549878 |

`diagnostic_balanced` mean/std 诊断主结果如下：

| variant | method | MAE_mean | MSE_mean | regret_to_oracle_mean | oracle_label_accuracy_mean |
| --- | --- | ---: | ---: | ---: | ---: |
| visual_mean_patch_only | raw soft fusion | 0.346525 | 1.388554 | 0.041035 | 0.356317 |
| visual_cls_mean_concat | raw soft fusion | 0.348457 | 1.359039 | 0.042966 | 0.461067 |
| visual_cls_only | raw soft fusion | 0.349310 | 1.369623 | 0.043820 | 0.350133 |
| visual_mean_patch_only | hard top-1 | 0.372762 | 1.477363 | 0.067272 | 0.356317 |
| visual_cls_mean_concat | hard top-1 | 0.373441 | 1.456282 | 0.067951 | 0.461067 |
| visual_cls_only | hard top-1 | 0.374926 | 1.449180 | 0.069435 | 0.350133 |

`visual_pooling_best_variant.json` 记录的 best variant 为 `visual_mean_patch_only`，选择依据为 `pilot_selection raw_soft_fusion MAE_mean` 最低；tie-breakers 为 hard top-1 MAE、regret_to_oracle 和 oracle-label accuracy。`diagnostic_balanced` 未参与选择，`pilot_test` 未使用。

`visual_pooling_summary.md` 已用中文回答四个验收问题：

1. mean_patch 优于 CLS；
2. CLS+mean concat 不优于单一 pooling；
3. 最佳 visual-only pooling 变体相对 P1 Round 0 Visual baseline 有改善；
4. 后续 visual+aux concat 建议使用 `visual_mean_patch_only` 作为视觉 pooling。

## 结论

P2b visual-only pooling 消融已完成。`visual_mean_patch_only` 在 `pilot_selection` raw-soft MAE 上优于 CLS 和 CLS+mean concat，并在 `diagnostic_balanced` 上也保持 raw-soft MAE 最优。与 P1 Round 0 Visual baseline 相比，最佳 P2b visual-only pooling 的 `pilot_selection` raw-soft MAE 从 `0.334069` 降至 `0.300996`，hard top-1 MAE 从 `0.356267` 降至 `0.319706`。

当前证据支持：ViT pooling 口径确实影响 Visual Router 效果，mean patch pooling 比默认 CLS 更适合作为下一步 visual+aux concat 的视觉输入。CLS+mean concat 虽然 oracle-label accuracy 在部分 seed/诊断上更高，但主选择指标 raw-soft MAE 不优于 mean_patch-only，因此本轮不建议将 concat 作为后续默认 pooling。

## 下一步方案

1. 若启动后续 P2d visual+aux concat，应使用 `visual_mean_patch_only` 作为默认 visual pooling，并继续保持 `pilot_train` 训练、`pilot_selection` 选择、`diagnostic_balanced` 诊断的口径。
2. 后续实验仍不得使用 `pilot_test` 做架构选择；如做 final test eval，必须显式标记为 `final_test_eval`。
3. 可结合已完成的 P2probe 结果进一步检查 mean_patch 的改进是否来自真实视觉结构信息，还是 dataset/TSF shortcut。
4. 若需要重跑 P2b，应只清理 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/` 下本轮 P2b 产物，不得覆盖 P0/P1/P2a 输出。
