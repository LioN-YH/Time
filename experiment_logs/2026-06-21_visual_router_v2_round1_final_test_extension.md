# Visual Router V2 Round 1 frozen final test extension

日志日期：2026-06-21 16:50:07 CST

## 目的

在 P2d best `cls_mean_concat_plus_aux` frozen pilot_test final eval 已完成且 Round 1 best 结论不变的前提下，补测三个关键候选变体在 P0 `pilot_test` 上的 frozen final performance：

- `mean_patch_plus_aux`
- `visual_mean_patch_only`
- `visual_cls_mean_concat`

本次只做 frozen eval extension，不训练新模型、不调参、不重新选择 Round 1 best、不使用 `pilot_test` 做 variant/seed/epoch/hyperparameter 选择。

## 背景

已有 P0 `pilot_test` 样本：

- `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples/pilot_test_sample_keys.csv`

已有 final_test_only feature cache：

- `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features_final_test_only/`

已有 P2d best final eval 输出：

- `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_final_test/`

已有 checkpoint 来源：

- P2d concat：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat/`
- P2b visual pooling：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/`

## 操作

1. 新增脚本：

   - `visual_router_experiments/stage1_vali_test_router/evaluate_visual_router_v2_round1_final_test_extension.py`

2. 脚本实现要点：

   - 复用 `evaluate_visual_router_v2_round1_final_test.py` 中的 final_test_only feature cache 校验、Round0 baseline 抽取、global/oracle baseline、metric summary、selected_model counts 和 stratified summary 逻辑。
   - 直接读取已有 P2d best 三个 seed 的 final eval prediction CSV，并纳入 extension comparison，不重新评估或改变 P2d best。
   - 只对三个补测变体加载已保存 checkpoint 和 scaler：
     - `mean_patch_plus_aux` 从 P2d concat checkpoint 加载。
     - `visual_mean_patch_only` 和 `visual_cls_mean_concat` 从 P2b visual pooling checkpoint 加载。
   - 从同一份 final_test_only feature cache 现场构造输入：
     - `mean_patch_plus_aux = concat(mean_patch_embedding, revin_aux)`
     - `visual_mean_patch_only = mean_patch_embedding`
     - `visual_cls_mean_concat = concat(cls_embedding, mean_patch_embedding)`
   - 输出 comparison、per-seed、selected counts、stratified summary、delta summary、metadata 和中文 summary。

3. 使用 `quito` 环境做语法检查：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/stage1_vali_test_router/evaluate_visual_router_v2_round1_final_test_extension.py
   ```

4. 运行 frozen eval extension：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/evaluate_visual_router_v2_round1_final_test_extension.py --skip-feature-build --device auto --overwrite
   ```

   本次尝试用 `tee` 写 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_final_test_extension/main.log`，但由于 `tee` 在脚本创建输出目录前打开文件失败，管道最终返回码为 1。脚本主体仍完整运行，并打印 `Round 1 final test extension outputs written ...`；最终以 `status.json`、metadata 和输出文件完整性为准。

5. 验证输出：

   - `status.json` 显示 `status=completed`、`sample_count=75000`。
   - metadata 显示 `status=completed`，`extension_variants` 为三个补测变体。
   - SQLite prediction index 记录数为 `375000 = 75000 sample_key * 5 experts`。
   - 九个新增 prediction CSV 均为 75,000 行。
   - `round1_final_test_extension_variant_seed_results.csv` 为 24 行，覆盖四个 Round1 variant × 三 seeds × hard/raw-soft。
   - `round1_final_test_extension_comparison.csv` 为 14 行，包含 P2d best、三个补测变体、Round0 TimeFuse、Round0 original Visual、global_best_single 和 oracle_top1。
   - `round1_final_test_extension_delta_summary.csv` 为 90 行，包含指定 six pairwise deltas。

6. 复制轻量结果到仓库：

   - `experiment_summaries/visual_router_v2_round1/p2d_final_test_extension/round1_final_test_extension_summary.md`
   - `experiment_summaries/visual_router_v2_round1/p2d_final_test_extension/round1_final_test_extension_comparison.csv`
   - `experiment_summaries/visual_router_v2_round1/p2d_final_test_extension/round1_final_test_extension_delta_summary.csv`
   - `experiment_summaries/visual_router_v2_round1/p2d_final_test_extension/round1_final_test_extension_metadata.json`

## 结果

正式输出目录：

- `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_final_test_extension/`

关键 raw-soft pilot_test 指标如下：

| 方法 | MAE | MSE | regret_to_oracle | oracle-label accuracy | MAE_std |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cls_mean_concat_plus_aux` | 0.452942 | 245.459475 | 0.112657 | 0.432360 | 0.039445 |
| `mean_patch_plus_aux` | 0.516108 | 486.102519 | 0.175823 | 0.429196 | 0.048081 |
| `visual_mean_patch_only` | 0.452976 | 303.486492 | 0.112691 | 0.337782 | 0.044625 |
| `visual_cls_mean_concat` | 0.443062 | 244.238487 | 0.102777 | 0.517329 | 0.021419 |
| `round0_timefuse_raw_soft_fusion` | 0.535220 | 568.502401 | 0.194935 | 0.587240 | 0.000000 |

关键 delta：

- `mean_patch_plus_aux - cls_mean_concat_plus_aux` raw-soft MAE delta = `+0.063166`。
- `mean_patch_plus_aux - visual_mean_patch_only` raw-soft MAE delta = `+0.063132`，说明 aux 在 mean_patch 路线上没有带来 test 改善，反而明显变差。
- `cls_mean_concat_plus_aux - visual_cls_mean_concat` raw-soft MAE delta = `+0.009880`，说明 aux 在 cls+mean 路线上也没有带来 test MAE 改善。
- `visual_mean_patch_only - Round0 TimeFuse` raw-soft MAE delta = `-0.082245`。
- `visual_cls_mean_concat - Round0 TimeFuse` raw-soft MAE delta = `-0.092158`。
- `cls_mean_concat_plus_aux - Round0 TimeFuse` raw-soft MAE delta = `-0.082279`。

## 结论

1. `mean_patch_plus_aux` 在 pilot_test 上不够强，raw-soft MAE 比 P2d best 高 `+0.063166`，也明显弱于 `visual_mean_patch_only`。
2. aux 对 `mean_patch` 的 test 边际贡献为负向：`mean_patch_plus_aux` 比 `visual_mean_patch_only` raw-soft MAE 高 `+0.063132`，MSE 也更差。
3. aux 对 `cls+mean` 的 test 边际贡献同样不是正向：`cls_mean_concat_plus_aux` 比 `visual_cls_mean_concat` raw-soft MAE 高 `+0.009880`。
4. 两个 visual-only 补测变体均超过 Round0 TimeFuse；其中 `visual_cls_mean_concat` 是本次 extension 中 pilot_test raw-soft MAE 最好的变体。
5. `cls_mean_concat_plus_aux` 的 seed17 最好，raw-soft MAE_std 为 `0.039445`；`mean_patch_plus_aux` MAE_std 为 `0.048081`，不更稳定。
6. 本次结果不改变 Round 1 best 的历史选择结论，因为 Round 1 best 仍是基于 `pilot_selection` 冻结选择；但从 pilot_test 解释角度看，P2e FiLM 若继续推进，应优先以 `cls_mean_concat`/visual-only cls+mean 为主线，`mean_patch` 只保留为简洁强基线，不建议把 `mean_patch_plus_aux` 作为主线。

## 下一步方案

1. 若继续 P2e FiLM，建议把 `visual_cls_mean_concat` 和 `cls_mean_concat_plus_aux` 的差异作为重点诊断对象，先解释 aux 在 test 上为什么没有改善。
2. 继续保留 `pilot_test` 只作冻结最终验证，不用于 seed/variant/epoch/hyperparameter 选择。
3. 若启动 Round 2 image/view layout 消融，应以本次 extension 的 visual-only 强结果作为对照基线，并重点检查 CrossFormer / PatchTST 分层。
4. 后续归档时继续只复制轻量 summary 到 `experiment_summaries/`，不把 checkpoint、SQLite、逐样本 prediction CSV 或 feature shard 入仓库。
