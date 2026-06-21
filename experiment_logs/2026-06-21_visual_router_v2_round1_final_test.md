# Visual Router V2 Round 1 P2d Frozen Final Test

日志日期：2026-06-21 15:30:19 CST

## 目的

对已经在 P2d 冻结选择的 Round 1 best variant `cls_mean_concat_plus_aux` 做一次 `pilot_test` final evaluation，验证 `pilot_selection` 上的改善是否能泛化到未参与选择的 P0 `pilot_test`，并与 Round0 TimeFuse、Round0 原始 Visual Router、`global_best_single` 和 `oracle_top1` 在同一批 75,000 个 sample_key 上比较。

## 背景

P2d 输出目录为 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat/`，`round1_concat_best_variant.json` 已冻结 best variant 为 `cls_mean_concat_plus_aux`，选择依据为 `pilot_selection` raw-soft MAE mean，且 `pilot_test_used_for_selection=false`。

本次只做 final eval，不训练新模型、不调参、不改变 variant/seed/epoch，不做 FiLM/gating/attention，不修改 P2d 选择结果，也不把 `pilot_test` 用于模型选择。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/evaluate_visual_router_v2_round1_final_test.py`。
   - 脚本固定校验 `p2d_best_variant=cls_mean_concat_plus_aux`。
   - 加载 P2d 已保存的 seed 16/17/18 checkpoint 与 scaler。
   - 只按三 seed mean/std 汇总，不根据 `pilot_test` 选择 best seed。
   - 输出 comparison、per-seed result、selected_model counts、stratified summary、metadata 和 summary。
2. 因 P2a 正式 feature cache 默认不含 `pilot_test`，使用 `build_visual_router_v2_round1_features.py --sample-sets pilot_test --include-pilot-test-final-test-only` 在线生成并保留独立 final-test-only feature cache：
   `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features_final_test_only/`。
3. 运行命令使用 `quito` 环境：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/evaluate_visual_router_v2_round1_final_test.py --device cuda --vit-data-parallel --overwrite
   ```

4. 运行过程中生成 `prediction_index_round1_final_test_pilot_test.sqlite`，只覆盖 P0 `pilot_test` 75,000 个 sample_key × 五专家记录，用于数组级 hard/raw-soft MAE/MSE 复算，未将 116M 行 prediction manifest 读入内存。
5. 修正 `round1_final_test_selected_model_counts.csv` 的 P2d counts 口径：P2d 三个 seed 保留 per-seed counts，每个 seed/method 分母均为 75,000；baseline 仍为确定性单次 counts。
6. 使用 `py_compile` 检查新增 evaluator 语法，并读取输出 CSV/JSON 进行验收检查。

## 结果

### final_test_only feature cache

输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features_final_test_only/
```

关键结果：

- `status=completed`
- `sample_sets=["pilot_test"]`
- `final_test_only_sets=["pilot_test"]`
- `sample_counts={"pilot_test": 75000}`
- `completed_shards=38`
- `total_cache_size_mb=198.192`
- manifest 中 `final_test_only` 全部为 `true`
- feature 只由历史窗口 `x` 构造，metadata 记录 `pseudo_image_tensor_saved=false`、`read_prediction_manifest=false`、`train_router_or_encoder=false`

该 cache 按用户确认保留，作为后续只读 final_test_only feature cache，不覆盖 P2a 原始 feature cache，不参与训练或选择。

### final eval 输出

输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_final_test/
```

必需产物已生成：

- `round1_final_test_comparison.csv`
- `round1_final_test_variant_seed_results.csv`
- `round1_final_test_selected_model_counts.csv`
- `round1_final_test_stratified_summary.csv`
- `round1_final_test_metadata.json`
- `round1_final_test_summary.md`

metadata 关键约束检查：

- `p2d_best_variant=cls_mean_concat_plus_aux`
- `pilot_test_used_for_selection=false`
- `pilot_test_not_used_for_selection=true`
- `pilot_test_feature_final_test_only=true`
- `trained_new_model=false`
- `changed_variant=false`
- `changed_seed_by_test=false`
- `changed_epoch_by_test=false`
- `changed_hyperparams_by_test=false`

### 主要指标

`pilot_test` 同一批 75,000 个 sample_key 上，P2d `cls_mean_concat_plus_aux` 三 seed mean：

- hard top-1 MAE：0.467320
- hard top-1 MSE：245.625781
- hard top-1 regret_to_oracle：0.127035
- hard top-1 oracle-label accuracy：0.432360
- raw-soft MAE：0.452942
- raw-soft MSE：245.459475
- raw-soft regret_to_oracle：0.112657
- raw-soft oracle-label accuracy：0.432360
- weight entropy：1.112345
- normalized weight entropy：0.691139
- mean max weight：0.521259

Round0 对照：

- Round0 TimeFuse raw-soft MAE：0.535220，MSE：568.502401，regret：0.194935，oracle-label accuracy：0.587240
- Round0 Visual raw-soft MAE：0.603009，MSE：510.975952，regret：0.262724，oracle-label accuracy：0.457960
- global_best_single hard MAE：0.599744
- oracle_top1 MAE：0.340285

per-seed raw-soft MAE：

- seed16：0.461857
- seed17：0.409803
- seed18：0.487166
- mean/std：0.452942 / 0.039445

## 结论

1. P2d best 在 `pilot_test` raw-soft MAE 上超过 Round0 TimeFuse：0.452942 vs 0.535220，改善 0.082279。
2. P2d best 在 `pilot_test` regret_to_oracle 上超过 Round0 TimeFuse：0.112657 vs 0.194935，改善 0.082279。
3. P2d best 的 raw-soft MSE 明显优于 Round0 原始 Visual Router：245.459475 vs 510.975952，也优于 Round0 TimeFuse。
4. P2d best 的 oracle-label accuracy 仍低于 TimeFuse：0.432360 vs 0.587240，但 MAE/regret/MSE 均更好，说明标签准确率不是本轮最终效果的充分指标。
5. raw-soft 明显优于 hard top-1：MAE 0.452942 vs 0.467320，regret 0.112657 vs 0.127035。
6. CrossFormer oracle stratum 上 P2d raw-soft MAE 约 0.625–0.636，优于 Round0 TimeFuse 0.658768，但 oracle-label accuracy 仍不高，是需要继续诊断的区域。
7. PatchTST oracle stratum 上 P2d raw-soft MAE 约 0.567–0.582，未超过 Round0 TimeFuse 0.542706，是 Round 2 或 P2e 需要关注的短板。
8. 未观察到相对 TimeFuse 的 selection 提升、test 退化风险；当前结果支持把 `cls_mean_concat_plus_aux` 作为后续 full-scale-safe pilot rerun 候选。

## 下一步方案

1. 若复跑 final eval，可复用已保留的 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features_final_test_only/`，避免再次在线生成 pilot_test feature。
2. 后续可把 seed 16/17/18 的 frozen eval 做成并行子任务，或重构为一次读取 prediction arrays、三 seed 同批计算，以减少重复 I/O。
3. 建议进入 Round 2 pseudo image / view layout 消融，同时继续把 `pilot_test` 作为冻结 final eval，不参与 variant/seed/epoch/hyperparameter 选择。
4. P2e FiLM/conditional modulation 可以作为后续方向，但必须另开实验并继续遵守 `pilot_train`/`pilot_selection` 选择、`pilot_test` final-only 的边界。
