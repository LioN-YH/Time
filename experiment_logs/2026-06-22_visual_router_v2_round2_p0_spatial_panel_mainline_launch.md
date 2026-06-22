# Visual Router V2 Round2f P0 spatial panel mainline 启动

日志日期：2026-06-22 07:53:16 CST

## 目的

在原 P0 pilot 协议规模上启动 `spatial_panel_3view + film_mean_patch_aux` 主线验证，覆盖 `pilot_train=150000`、`pilot_selection=30000`、`diagnostic_balanced=20000` 和 `pilot_test=75000`，并保证 feature cache 与训练评估均使用多 GPU 进程级并行。

## 背景

Round2e-b 65k expanded layout validation 已完成，`spatial_panel_3view` 是 selection 与 test_expanded 双 best，且优于 `current_rgb_3view`。本轮目标是把 `spatial_panel_3view + film_mean_patch_aux` 升级到 P0 pilot 主线验证规模，只用 `pilot_selection` raw-soft MAE mean 做选择，`diagnostic_balanced` 只诊断，`pilot_test` 只做 frozen final eval。

## 操作

1. 读取用户目标文件 `/home/shiyuhong/.codex-tianyu/attachments/24f5de11-1ff2-427e-80d8-e35ce93a7cac/pasted-text-1.txt`，确认本轮固定 layout、backend、seed、输入输出和验收文件名。
2. 审计并复用既有脚本：
   - `build_visual_router_v2_round2_layout_features.py`
   - `train_visual_router_v2_round2_layout_film.py`
   - `launch_visual_router_v2_round2_expanded_validation_parallel.py`
3. 修改 `train_visual_router_v2_round2_layout_film.py`：
   - Round1 reference comparison 补入 `film_cls_mean_concat_aux`；
   - Round0 reference normalization 补入 `Round0 original Visual` 的 hard/raw-soft 行。
4. 修改 `build_visual_router_v2_round2_layout_features.py`：
   - 新增 `--sample-set-worker` 模式；
   - 同一 layout 下可按单个 sample_set 并行写 `worker_manifests/`、`worker_latencies/`、`worker_metadata/` 和 `worker_status/`；
   - `--aggregate-only` 可合并 sample-set worker manifest，避免多个进程同时写同一个 layout manifest。
5. 新增 `launch_visual_router_v2_round2_p0_spatial_panel_parallel.py`：
   - 自动把 P0 四个 `*_sample_keys.csv` 合并为输出目录内 `inputs/p0_sample_manifest.csv`；
   - feature 阶段按 `pilot_train`、`pilot_selection`、`diagnostic_balanced`、`pilot_test` 四个 worker 分配到 `cuda:0,cuda:1,cuda:2,cuda:3`；
   - training/eval 阶段按 seeds 16/17/18 多 GPU 并行；
   - aggregation 后生成 `round2_p0_spatial_*` 固定验收文件名和 P0 中文 summary。
6. 运行语法检查：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_layout_features.py \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round2_layout_film.py \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_p0_spatial_panel_parallel.py
   ```

7. 运行 1 sample/set feature smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_p0_spatial_panel_parallel.py \
     --feature-only \
     --max-samples-per-set 1 \
     --output-dir /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline_feature_smoke \
     --devices cuda:0,cuda:1,cuda:2,cuda:3 \
     --local-files-only \
     --overwrite \
     --poll-seconds 2
   ```

8. 启动正式后台任务：

   ```text
   setsid /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     /home/shiyuhong/Time-visual-router-v2/visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_p0_spatial_panel_parallel.py \
     --devices cuda:0,cuda:1,cuda:2,cuda:3 \
     --poll-seconds 30 \
     --output-dir /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline \
     > /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/launcher_stdout.log 2>&1 < /dev/null &
   ```

   正式 launcher PID 记录在：

   ```text
   /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/launcher.pid
   ```

## 结果

1. `py_compile` 通过。
2. P0 manifest 合并检查通过，`inputs/p0_sample_manifest.csv` 包含 275000 行：
   - `pilot_train`: 150000
   - `pilot_selection`: 30000
   - `diagnostic_balanced`: 20000
   - `pilot_test`: 75000
3. 1 sample/set feature smoke 通过：
   - 四个 sample_set worker 分别在 cuda:0/1/2/3 上启动并完成；
   - aggregation 写出 `round2_p0_spatial_feature_manifest.csv`；
   - 抽检 `mean_patch_embedding` shape 为 `(1, 768)`，`revin_aux` shape 为 `(1, 6)`，finite 为 True，`order_index=[0]`。
4. 正式后台任务已启动：
   - launcher PID：`3347085`
   - 主日志：`/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/launcher_stdout.log`
   - feature worker logs：`feature_logs/spatial_panel_3view_<sample_set>.log`
5. 截至本日志记录时的健康检查：
   - `diagnostic_balanced` 已完成 10 shards / 20000 samples；
   - `pilot_selection` 已完成 15 shards / 30000 samples；
   - `pilot_train` 正在运行，已完成 18 shards / 36000 samples；
   - `pilot_test` 正在运行，已完成 17 shards / 34000 samples；
   - GPU0/GPU3 正在运行剩余 train/test worker，GPU1/GPU2 已释放。

## 结论

P0 spatial panel mainline 的代码入口、sample-set 并行 feature worker、固定 FiLM 训练聚合路径和轻量验收文件名后处理已经准备完成，并通过语法检查与 feature smoke。正式 P0-scale feature cache 已在后台正常推进，目前尚未完成训练、final eval 和最终 aggregation，因此不能把本轮目标判定为完成。

## 下一步方案

1. 继续监控 feature 阶段直到 `pilot_train` 和 `pilot_test` 完成，并确认 `round2_p0_spatial_feature_manifest.csv`、`round2_p0_spatial_feature_latency.csv` 生成。
2. feature aggregation 完成后 launcher 会自动构建 prediction subset SQLite，并并行运行 seeds 16/17/18 的 fixed FiLM training/eval。
3. training aggregation 完成后检查以下轻量产物：
   - `round2_p0_spatial_variant_seed_results.csv`
   - `round2_p0_spatial_selection_comparison.csv`
   - `round2_p0_spatial_diagnostic_summary.csv`
   - `round2_p0_spatial_final_test_summary.csv`
   - `round2_p0_spatial_selected_model_counts.csv`
   - `round2_p0_spatial_stratified_summary.csv`
   - `round2_p0_spatial_delta_summary.csv`
   - `round2_p0_spatial_metadata.json`
   - `round2_p0_spatial_summary.md`
   - `status.json`
4. 最终验收时必须确认 `pilot_test` 没有用于训练、调参、选择 seed、选择 epoch 或选择 variant，并在 summary 中明确判断是否超过 Round1 `film_mean_patch_aux`。

## 监控命令

```text
ps -p $(cat /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/launcher.pid) -o pid,ppid,stat,etime,cmd

tail -f /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/launcher_stdout.log

find /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/features/spatial_panel_3view/worker_status -type f -maxdepth 1 -print -exec sed -n '1,80p' {} \;

nvidia-smi
```
