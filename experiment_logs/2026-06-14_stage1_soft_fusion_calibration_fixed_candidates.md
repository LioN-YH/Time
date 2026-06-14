# Stage 1 Soft Fusion 校准与 Fixed Candidates Embedding 对照

日志日期：2026-06-14 03:27:47 CST

## 目的

继续推进 Stage 1 Visual Router 后续实验，围绕当前 `96_48_S`、120 个 `metric=mae` sample_key 完成两件事：

1. 在不改变 router 输入约束的前提下，对已有 `fusion_huber_kl` router test 权重做 soft fusion calibration；
2. 使用 `fixed_candidates` 周期桶重新生成 ViT embedding smoke，并对比 latency 与下游 router 指标。

## 背景

上一轮代表性 `fusion_huber_kl` Visual Router 输出为：

```text
experiment_logs/run_outputs/2026-06-14_025727_562553_visual_router_stage1_visual_router_smoke/
```

关键结果：

- hard top-1 MAE：`0.982425`
- raw soft fusion MAE：`1.085451`
- `global_best_single` MAE：`1.055190`
- TimeFuse 结构特征 router MAE：`1.079743`
- `oracle_top1` MAE：`0.805392`

主要问题是 raw soft fusion 明显弱于 hard top-1 和 `global_best_single`，权重归一化熵为 `0.757180`、平均最大权重为 `0.483784`，说明权重仍偏平滑。与此同时，伪图像化路径已新增 `fixed_candidates` 周期桶，但旧代表 embedding 未记录并未使用新 metadata 口径，需要重建一轮 embedding 才能比较。

## 操作

1. 阅读并遵守 `AGENTS.md`，确认本次实验 Python 使用 conda `quito` 环境：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python
   ```

2. 新增正式校准脚本：

   ```text
   visual_router_experiments/stage1_vali_test_router/evaluate_soft_fusion_calibration.py
   ```

   脚本功能：

   - 读取 `visual_router_predictions.csv` 中已输出的 test 权重；
   - 读取 prediction cache manifest 下同一 `sample_key` 的五专家 `y_pred` 和共享 `y_true`；
   - 支持 softmax temperature；
   - 支持 top-k 权重截断后重归一化；
   - 支持 raw soft、top1 hard、top2/top3 fusion 和 temperature sweep；
   - 输出 entropy、normalized entropy、max-weight、active weight count、selected-model 分布；
   - 不读取 test oracle error 或专家误差来调整每个样本权重。

3. 使用 quito 环境做语法检查：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/evaluate_soft_fusion_calibration.py
   ```

   结果：通过。

4. 基于旧代表 fusion router 输出运行 calibration smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/evaluate_soft_fusion_calibration.py \
     --router-predictions-path experiment_logs/run_outputs/2026-06-14_025727_562553_visual_router_stage1_visual_router_smoke/visual_router_predictions.csv \
     --temperatures 0.25,0.5,0.75,1.0,1.5,2.0 \
     --top-k-values all,1,2,3 \
     --print-rows 30
   ```

   输出目录：

   ```text
   experiment_logs/run_outputs/2026-06-14_032303_451482_visual_router_stage1_soft_fusion_calibration_smoke/
   ```

5. 使用 `fixed_candidates` 周期口径重新生成一轮 120 sample_key ViT embedding smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/build_vit_embeddings.py \
     --local-files-only \
     --batch-size 16 \
     --period-selection fixed_candidates \
     --print-rows 5
   ```

   输出目录：

   ```text
   experiment_logs/run_outputs/2026-06-14_032340_154510_visual_router_stage1_vit_embedding_smoke/
   ```

6. 对比旧 embedding run 与新 fixed_candidates embedding run 的 latency，生成：

   ```text
   experiment_logs/run_outputs/2026-06-14_032340_154510_visual_router_stage1_vit_embedding_smoke/embedding_latency_comparison_vs_old.csv
   experiment_logs/run_outputs/2026-06-14_032340_154510_visual_router_stage1_vit_embedding_smoke/embedding_latency_speed_ratio_vs_old.csv
   ```

   对比时同时记录全量 batch 与去掉首批 warm-up 后的统计。

7. 使用新 fixed_candidates embedding 重跑同参数 `fusion_huber_kl` router smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/train_visual_router.py \
     --embedding-manifest-path experiment_logs/run_outputs/2026-06-14_032340_154510_visual_router_stage1_vit_embedding_smoke/embedding_manifest.csv \
     --router-mode fusion_huber_kl \
     --epochs 300 \
     --batch-size 32 \
     --hidden-dim 64 \
     --dropout 0.0 \
     --huber-beta 0.1 \
     --kl-tau 0.1 \
     --lambda-kl 0.01 \
     --print-rows 5
   ```

   输出目录：

   ```text
   experiment_logs/run_outputs/2026-06-14_032518_167365_visual_router_stage1_visual_router_smoke/
   ```

8. 对新 fixed_candidates router 输出再跑同一套 calibration smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/evaluate_soft_fusion_calibration.py \
     --router-predictions-path experiment_logs/run_outputs/2026-06-14_032518_167365_visual_router_stage1_visual_router_smoke/visual_router_predictions.csv \
     --temperatures 0.25,0.5,0.75,1.0,1.5,2.0 \
     --top-k-values all,1,2,3 \
     --print-rows 20
   ```

   输出目录：

   ```text
   experiment_logs/run_outputs/2026-06-14_032647_499280_visual_router_stage1_soft_fusion_calibration_smoke/
   ```

9. 生成关键结果统一对比表：

   ```text
   experiment_logs/run_outputs/2026-06-14_032647_499280_visual_router_stage1_soft_fusion_calibration_smoke/stage1_fixed_candidates_router_calibration_key_comparison.csv
   ```

10. 读取旧/新 embedding 和 router predictions，完成差异诊断：

    - 120 个 embedding 中有 22 个样本最大绝对差异超过 `1e-6`；
    - 旧/新 hard top-1 在 60 个 test window 中有 13 个 sample_key 改变专家选择；
    - 变化主要集中在 `TEST_DATA_MIN` 的若干窗口。

11. 更新文档：

    - `visual_router_experiments/stage1_vali_test_router/README.md`
    - `visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md`
    - `WORKSPACE_STRUCTURE.md`
    - `experiment_logs/README.md`

## 结果

### 旧代表 Router Calibration

输出目录：

```text
experiment_logs/run_outputs/2026-06-14_032303_451482_visual_router_stage1_soft_fusion_calibration_smoke/
```

| 策略 | test MAE | oracle MAE | regret | normalized entropy | mean max weight | 相对 global_best_single |
| --- | --- | --- | --- | --- | --- | --- |
| `top1_hard` | 0.982425 | 0.805392 | 0.177033 | 0.000000 | 1.000000 | +6.895970% |
| `top2_fusion_T0p25` | 0.999014 | 0.805392 | 0.193622 | 0.232572 | 0.822011 | +5.323768% |
| `top3_fusion_T0p25` | 0.999879 | 0.805392 | 0.194487 | 0.279847 | 0.802877 | +5.241825% |
| `soft_T0p25` | 1.000585 | 0.805392 | 0.195193 | 0.295159 | 0.799357 | +5.174969% |
| `raw_soft` | 1.085451 | 0.805392 | 0.280059 | 0.757180 | 0.483784 | -2.867833% |
| `global_best_single` | 1.055190 | 0.805392 | 0.249798 | NA | NA | 0.000000% |

结论：temperature sharpen 和 top-k 截断能把 soft fusion 从 `1.085451` 拉回到 `0.999014`，已经超过 `global_best_single=1.055190`，但仍未超过 hard top-1 `0.982425`。

### Fixed Candidates Embedding Latency

新 embedding metadata 记录：

- `period_selection=fixed_candidates`
- `period_candidates=[2,3,4,5,6,8,10,12,16,24,32,48,64,96]`
- `sample_count=120`
- `embedding_dim=768`
- `device=cuda`
- `forward_dtype=float16`

latency 对比：

| latency slice | 旧 imageization ms/window | 新 imageization ms/window | 新/旧 | 旧 total ms/window | 新 total ms/window | 新/旧 |
| --- | --- | --- | --- | --- | --- | --- |
| all_batches | 7.766617 | 12.217573 | 1.573088 | 25.558282 | 15.042344 | 0.588551 |
| exclude_first_batch | 0.469106 | 0.222156 | 0.473573 | 1.692935 | 1.405437 | 0.830178 |

说明：首批 batch 包含 GPU/模型 warm-up，单独看会扭曲 imageization 和 encoder forward 的均值；去掉首批后，fixed_candidates 图像化阶段约降到旧 run 的 `47.36%`。

### Fixed Candidates Router 与 Calibration

新 embedding router 输出目录：

```text
experiment_logs/run_outputs/2026-06-14_032518_167365_visual_router_stage1_visual_router_smoke/
```

新 router calibration 输出目录：

```text
experiment_logs/run_outputs/2026-06-14_032647_499280_visual_router_stage1_soft_fusion_calibration_smoke/
```

关键指标：

| 方法 | test MAE | oracle MAE | regret | label accuracy | normalized entropy | mean max weight |
| --- | --- | --- | --- | --- | --- | --- |
| `oracle_top1` | 0.805392 | 0.805392 | 0.000000 | 1.000000 | NA | NA |
| `visual_router_mlp_v2_fusion_huber_kl` hard top-1 | 1.011773 | 0.805392 | 0.206381 | 0.433333 | 0.783611 | 0.453640 |
| `calibration_soft_T0p25` | 1.021081 | 0.805392 | 0.215689 | 0.433333 | 0.365525 | 0.756458 |
| `calibration_top2_fusion_T0p25` | 1.023443 | 0.805392 | 0.218051 | 0.433333 | 0.265783 | 0.794654 |
| `global_best_single` | 1.055190 | 0.805392 | 0.249798 | 0.050000 | NA | NA |
| `timefuse_single_variable_logistic_regression` | 1.079743 | 0.805392 | 0.274351 | 0.466667 | NA | NA |
| `visual_router_mlp_v2_fusion_huber_kl_soft_fusion` raw soft | 1.088799 | 0.805392 | 0.283407 | NA | 0.783611 | 0.453640 |

结果说明：

- 新 fixed_candidates embedding 下 hard top-1 仍超过 `global_best_single`，但从旧代表 run 的 `0.982425` 退化到 `1.011773`；
- raw soft fusion 仍失败，MAE 为 `1.088799`；
- 最佳校准策略为 `soft_T0p25`，MAE 为 `1.021081`，超过 `global_best_single`，但仍弱于 hard top-1；
- 新口径的平均最大权重更低、归一化熵更高，说明权重比旧代表 run 更平滑。

### 新旧差异诊断

旧代表 embedding：

```text
experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/embedding_manifest.csv
```

新 fixed_candidates embedding：

```text
experiment_logs/run_outputs/2026-06-14_032340_154510_visual_router_stage1_vit_embedding_smoke/embedding_manifest.csv
```

诊断结果：

- `merged_embedding_rows=120`
- `embedding_changed_count_gt_1e-6=22`
- `embedding_max_abs_diff_overall=1.872803`
- `embedding_mean_abs_diff_mean=0.035958`
- `selected_changed=13/60`
- 旧 hard top-1 mean MAE：`0.982425`
- 新 hard top-1 mean MAE：`1.011773`

## 结论

1. 本次新增的 `evaluate_soft_fusion_calibration.py` 可以在不改变 router 输入约束的前提下，对已有 test 权重做 temperature/top-k calibration，并输出统一 comparison 表和权重诊断。
2. 对旧代表 `fusion_huber_kl` router，校准能让 soft fusion 超过 `global_best_single`：最佳策略为 `top2_fusion_T0p25`，MAE 为 `0.999014`，相对 `global_best_single=1.055190` 提升约 `5.32%`。
3. 校准后的 soft fusion 仍未超过 hard top-1，说明当前概率排序比概率幅度可靠；继续直接全专家加权会混入较差专家。
4. `fixed_candidates` 周期桶显著降低去 warm-up 后的图像化 latency，但在当前 120 sample_key 上改变了 22 个 embedding 和 13 个 test hard 选择，使 hard top-1 从 `0.982425` 退化到 `1.011773`。
5. 新 fixed_candidates embedding 下 calibration 仍能超过 `global_best_single`，最佳 `soft_T0p25` MAE 为 `1.021081`，但也弱于 hard top-1。
6. 当前结果仍是 `96_48_S`、120 sample_key smoke，不应作为三 config 正式结论；尤其不能仅凭本轮小样本断定 fixed_candidates 对最终指标有利或不利。

## 下一步方案

1. 在更大 `96_48_S` 样本上复验：同时跑旧动态周期口径与 fixed_candidates 口径，比较 embedding latency、hard top-1、calibrated soft fusion 和 selected-model 稳定性。
2. 若继续优化 soft fusion，优先尝试训练阶段的稀疏/置信约束，例如 entropy penalty、top-k training loss、confidence-aware gating，而不是只在 test 后处理。
3. 对当前 calibration 输出做分 dataset / TSF cell 诊断，定位 raw soft 失败主要来自哪些窗口或专家组合。
4. 暂不扩大到三 config；先把 `96_48_S` 的 calibration 和 embedding 口径稳定后，再扩展到 `576_288_S` 与 `1024_512_S`。
5. 保留本次 run_outputs 作为 smoke 证据，但不要把 embedding npy、prediction cache、checkpoint 或模型权重纳入 Git。
