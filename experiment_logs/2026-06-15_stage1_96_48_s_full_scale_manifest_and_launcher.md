# Stage 1 `96_48_S` 正式 full-scale 全候选窗口 manifest 与 launcher

日志日期：2026-06-15 02:39:22 CST

## 目的

推进 Stage 1 的 `96_48_S` 正式 full-scale 全候选窗口实验，先把正式 sample manifest 和 prediction cache launcher 落到 `/data2`，再启动五专家 prediction cache worker。

## 背景

前序窗口已经完成 1k、dry-run 和 TimeFuse-style fusor baseline 的验证，本轮不能再重复 1k 或 dry-run。正式 full-scale 需要：

- 全候选窗口 sample manifest；
- packed prediction cache；
- streaming online router；
- soft fusion calibration；
- 全流程状态留痕和可恢复执行。

由于 `/home` 空间紧张，正式输出必须使用 `/data2/syh/Time/run_outputs/`。

## 操作

1. 检查项目状态：
   - `git status --short --branch`
   - `git rev-parse HEAD`
   - `nvidia-smi`
   - `df -h /home /data2`
   - `pgrep -af 'stage1|visual_router|prediction_cache|train_visual_router|run_full_scale|quito-cli|finetune|evaluate'`
2. 读取并确认关键文档：
   - `AGENTS.md`
   - `HANDOFF.md`
   - `visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md`
   - `WORKSPACE_STRUCTURE.md`
   - `experiment_logs/README.md`
3. 核对正式 full-scale 根目录：
   - `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/`
   - 其中 `sample_manifest_full_scale/` 已完成，`launcher_compat_check/` 仅是 launcher 兼容性检查。
4. 读取并确认正式 manifest：
   - `sample_manifest_shard_index.csv` 共 `64` 个分片；
   - `sample_count=23,275,170`；
   - 候选窗口来自 `vali/test` 的 `TEST_DATA_MIN` 与 `TEST_DATA_HOUR` 全候选枚举；
   - `selection_strategy=all_candidate_windows`。
5. 使用正式 manifest index 生成 prediction cache launcher：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/launch_full_scale_prediction_cache.py \
  --sample-manifest-shard-index-path /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/sample_manifest_full_scale/sample_manifest_shard_index.csv \
  --output-dir /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher \
  --gpus 0,1,2 \
  --array-storage packed_npy_v1
```

6. 启动长任务：
   - 第一次用一次性 `tmux new-session -d -s stage1_fullscale_launcher "bash launcher.sh"` 启动后，worker 很快退出，未生成有效 shard 状态；
   - 随后改为持久 `tmux` shell `stage1_fullscale_launcher_shell`，再执行 `bash launcher.sh`，五个 worker 成功进入运行态。
7. 现场核对：
   - `ps` 显示五个 worker 对应的 `bash launcher.sh` 与五个 Python 进程；
   - `status.json` 显示各专家首个 shard 为 `running`；
   - `worker.log` 已写出各专家开始第一批 shard 的日志；
   - `nvidia-smi` 显示 DLinear / PatchTST / CrossFormer 分别占用 GPU 0/1/2。

## 结果

1. 正式 full-scale sample manifest 已完成，输出目录为：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/sample_manifest_full_scale/
```

2. 正式 prediction cache launcher 已生成，输出目录为：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/
```

3. launcher 已在持久 `tmux` 会话中启动，当前五个 worker 的 PID 为：
   - DLinear worker `427566`，Python `427570`
   - PatchTST worker `427571`，Python `427575`
   - CrossFormer worker `427576`，Python `427580`
   - ES worker `427581`，Python `427585`
   - NaiveForecaster worker `427586`，Python `427590`
4. 当前 `status.json` 显示首个 shard 正在运行，未见 completed shard。
5. 第一次短暂启动方式未能维持 worker 运行，已确认不是正式状态；当前持久 `tmux` 启动方式有效。

### 2026-06-15 02:46:16 CST 追加进度

1. prediction cache 已产生首批 completed shard：
   - `DLinear/sample_shard_0000_of_0064`
   - `PatchTST/sample_shard_0000_of_0064`
   - `CrossFormer/sample_shard_0000_of_0064`
   - `NaiveForecaster/sample_shard_0000_of_0064`
2. 当前正在运行：
   - `DLinear/sample_shard_0001_of_0064`
   - `PatchTST/sample_shard_0001_of_0064`
   - `CrossFormer/sample_shard_0001_of_0064`
   - `NaiveForecaster/sample_shard_0001_of_0064`
   - `ES/sample_shard_0000_of_0064`
3. 已完成 shard 抽检结果：
   - 每个 completed shard 的 `manifest.csv` 均为 `363675` 行；
   - `sample_key + model_name` 无重复；
   - `array_storage` 均为 `packed_npy_v1`；
   - 同一 `sample_key` 下 `y_true_path + y_true_row_index` 唯一；
   - packed `y_true/y_pred` 数组 shape 与 manifest 按 split/dataset 的行数一致。
4. 当前没有发现 failed shard。

## 结论

正式 full-scale 链路已经从 manifest 阶段推进到 prediction cache 实跑阶段，且当前实跑是基于正式 `sample_manifest_shard_index.csv`，不是 1k、不是 dry-run、也不是 launcher 兼容性检查目录。

## 下一步方案

1. 继续监控五专家 64 个 sample shard 的运行状态。
2. 所有 shard 完成后执行 merge，并在 merged cache 上生成 oracle labels、TSF enrichment 和 baseline。
3. 之后用 `train_visual_router_online_streaming.py` 跑正式 streaming visual router，再做 soft fusion calibration。
4. 若中途出现失败，仅精确重跑失败 shard，不删除已完成 shard。
