# Visual Router V2 Round2e-b 65k expanded layout validation 启动日志

日志日期：2026-06-22 03:54:37 CST

## 目的

在 Round2e-a 已冻结的 65k expanded samples 上启动 Round2e-b layout validation，只比较 `spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout` 三个 layout，并固定后端为 `film_mean_patch_aux` 风格。

## 背景

Round2c 35k screening 中 `spatial_panel_3view` 是 selection/test_small best，`current_rgb_3view` 是当前 RGB channel mixing baseline，`top3fold_period_layout` 在 diagnostic balanced 和 period continuity 上有价值。Round2d/addendum 显示 `period_soft_mixture` 不需要作为 65k 前置条件。

本轮必须使用：

- 样本 manifest：`/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_expanded_sample_manifest.csv`
- 输出目录：`/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/`
- 轻量 summary 目录：`experiment_summaries/visual_router_v2_round2/expanded_layout_validation/`
- 默认 GPU：`cuda:0,cuda:1,cuda:2,cuda:3`

## 操作

1. 读取用户目标文件，确认本轮验收标准、禁用项和精确输出文件名。
2. 核对当前分支为 `exp/visual-router-v2-pilot`，确认 expanded sample manifest 已存在且正式 output dir 初始不存在。
3. 修改 `build_visual_router_v2_round2_layout_features.py`：
   - 增加 `--artifact-prefix`，支持生成 `round2_expanded_layout_feature_manifest.csv`、latency、metadata 和 summary。
   - 修复 `--layout` worker 模式：单 layout worker 只写 layout 子目录，不再自动 aggregate，避免多个 feature 进程竞争写统一 CSV/JSON。
4. 修改 `train_visual_router_v2_round2_layout_film.py`：
   - 增加 `--artifact-prefix`、`--train-sample-set`、`--selection-sample-set`、`--diagnostic-sample-set`、`--test-sample-set`、`--experiment-label` 和 `--summary-title`。
   - 支持 expanded validation 的动态 sample set 与动态输出文件名。
   - 聚合输出改为可生成 `round2_expanded_layout_variant_seed_results.csv`、`round2_expanded_layout_selection_comparison.csv`、`round2_expanded_layout_diagnostic_summary.csv`、`round2_expanded_layout_test_summary.csv`、`round2_expanded_layout_selected_model_counts.csv`、`round2_expanded_layout_stratified_summary.csv`、`round2_expanded_layout_delta_summary.csv`、`round2_expanded_layout_best_layout.json`、`round2_expanded_layout_validation_metadata.json` 和 `round2_expanded_layout_validation_summary.md`。
   - summary 模板改为回答 65k selection/test best、一致性、spatial vs current、top3fold expanded 性能转化、35k 稳定性、seed stability/MSE tail/CrossFormer/PatchTST strata、是否升级 spatial 为 Round2 主线和下一步建议。
5. 新增 `launch_visual_router_v2_round2_expanded_validation_parallel.py`：
   - 固定 expanded manifest、四个 expanded sample set、`round2_expanded_layout` 前缀和轻量 summary 目录。
   - 支持 `--devices`、`--layouts`、`--feature-only`、`--train-only`、`--aggregate-only`、`--overwrite`、`--max-samples-per-set` 和 `--local-files-only`。
   - feature 阶段按 layout 进程级并行；training/eval 阶段按 layout × seed 进程级并行；统一产物只由 aggregate step 写出。
6. 在 `quito` conda 环境执行 py_compile：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_layout_features.py \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round2_layout_film.py \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_layout_screening_parallel.py \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_expanded_validation_parallel.py
   ```

7. 执行 feature smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_expanded_validation_parallel.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation_smoke \
     --layouts spatial_panel_3view \
     --devices cuda:0 \
     --feature-only \
     --max-samples-per-set 2 \
     --overwrite \
     --poll-seconds 2
   ```

8. 尝试执行极小 train/eval smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_expanded_validation_parallel.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation_smoke \
     --layouts spatial_panel_3view \
     --seeds 16 \
     --devices cuda:0 \
     --train-only \
     --max-samples-per-set 2 \
     --overwrite \
     --poll-seconds 2
   ```

   该 smoke 在 prediction manifest 子集 SQLite 构建阶段进行线性扫描，运行超过 2 分钟后手动中止；中止后检查没有残留相关进程。该中止不影响正式实验，因为正式任务同样只构建一次 SQLite 后供 9 个任务复用。

9. 检查 GPU，四张 RTX 3090 均基本空闲。
10. 使用 `setsid` 启动正式后台任务：

    ```text
    setsid bash -c 'exec /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_expanded_validation_parallel.py --devices cuda:0,cuda:1,cuda:2,cuda:3 --layouts spatial_panel_3view,current_rgb_3view,top3fold_period_layout --overwrite > /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/launcher.nohup.log 2>&1' &
    ```

    PID 写入：

    ```text
    /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/launcher.pid
    ```

## 结果

1. py_compile 通过，无语法错误。
2. feature smoke 成功，生成：
   - `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation_smoke/round2_expanded_layout_feature_manifest.csv`
   - `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation_smoke/round2_expanded_layout_feature_latency.csv`
   - `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation_smoke/round2_expanded_layout_feature_metadata.json`
3. feature smoke manifest 覆盖四个 expanded sample set，每 set 2 条，`visual_feature_dim=768`、`aux_feature_dim=6`、`order_index=0..1`，文件名前缀符合 `round2_expanded_layout_*` 要求。
4. 正式后台 launcher 已启动，PID 为 `3286985`。
5. 2026-06-22 03:54:22 CST 健康检查结果：
   - 主 launcher 进程存在，PPID=1，SID=3286985。
   - 三个 feature worker 已启动：
     - `spatial_panel_3view`：cuda:0，PID 3286987
     - `current_rgb_3view`：cuda:1，PID 3286988
     - `top3fold_period_layout`：cuda:2，PID 3286989
   - `nvidia-smi` 显示 cuda:0/1/2 各约 693 MiB，cuda:3 空闲。

## 结论

Round2e-b expanded validation 的代码入口、并行边界和正式后台启动已完成。当前状态为正式 feature cache 阶段运行中；尚未完成 65k feature cache、9 个 training/eval task 和最终 aggregation，因此本轮实验结果尚不能引用为完成结论。

## 下一步方案

1. 继续监控正式后台任务：

   ```text
   tail -f /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/launcher.nohup.log
   ```

2. 查看 feature worker 状态：

   ```text
   find /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/features -name layout_status.json -print -exec cat {} \;
   ```

3. 查看训练任务状态：

   ```text
   find /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/tasks -name status.json -print -exec cat {} \;
   ```

4. 如需停止正式任务，应先停止 launcher，再停止其子进程：

   ```text
   kill $(cat /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/launcher.pid)
   pkill -f '2026-06-22_visual_router_v2_round2_expanded_layout_validation'
   ```

5. 完成后验收必须检查目标中列出的全部输出文件、feature shard shape/finite/order_index、3 layouts × 3 seeds task 完成状态、metadata 中 test 不参与选择的约束，以及轻量 summary 是否已复制到 `experiment_summaries/visual_router_v2_round2/expanded_layout_validation/`。
