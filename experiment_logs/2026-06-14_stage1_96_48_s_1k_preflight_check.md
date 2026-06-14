# Stage 1 `96_48_S` 1k 执行前状态检查

日志日期：2026-06-14 17:27:04 CST

## 目的

在启动 Stage 1 `96_48_S` 1k 中等规模五专家 prediction cache 之前，按可恢复实验要求确认 git 状态、GPU 状态、`quito` 环境、样本清单、launcher 和 shard 目录现状，避免重复运行已有有效产物。

## 背景

当前路线是在已有 1k sample manifest 和五专家 prediction cache launcher 基础上继续推进；不启动 1k ViT embedding launcher，不主动联网下载模型、依赖或数据。后续 online visual router 阶段将使用本地 ViT/HF cache 和 `--local-files-only`，且不长期保存伪图像 tensor 或 ViT embedding `.npy`。

## 操作

1. 检查 git 状态：

   ```text
   git status --short --branch
   ```

2. 检查 GPU：

   ```text
   nvidia-smi
   ```

3. 检查 `quito` conda 环境 Python：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -V
   ```

4. 检查当前是否已有 Stage 1 prediction/router 相关进程：

   ```text
   ps -eo pid,ppid,stat,etime,cmd | rg 'build_prediction_cache_from_manifest|train_visual_router_online|evaluate_soft_fusion_calibration|launch_96_48_s_1k|visual_router_stage1'
   ```

5. 使用 `quito` 环境读取 1k sample manifest、sample status、prediction launcher status，并逐一检查五专家 shard 目录中的 `status.json`、`main.log` 和 `manifest.csv` 是否存在。

## 结果

### Git 状态

检查时仓库位于 `main...origin/main`，存在以下未提交改动：

```text
 M experiment_logs/2026-06-14_stage1_online_visual_router_smoke.md
 M experiment_logs/README.md
?? experiment_logs/2026-06-14_stage1_completed_work_review.md
```

这些改动为已有实验日志和 README 总览更新，后续仅追加新日志和 README 行，不覆盖这些内容。

### GPU 状态

2026-06-14 17:26:47 CST 的 `nvidia-smi` 显示 4 张 RTX 3090 基本空闲，仅 Xorg 占用少量显存：

- GPU 0：714 MiB / 24576 MiB，util 0%；
- GPU 1：363 MiB / 24576 MiB，util 0%；
- GPU 2：10 MiB / 24576 MiB，util 0%；
- GPU 3：355 MiB / 24576 MiB，util 0%。

### 环境和进程

`quito` 环境 Python 版本为：

```text
Python 3.11.15
```

进程检查未发现正在运行的 `build_prediction_cache_from_manifest.py`、`train_visual_router_online.py`、`evaluate_soft_fusion_calibration.py` 或 `launch_96_48_s_1k` 相关长任务；输出中只有本次 `ps | rg` 检查命令自身。

### Sample Manifest

样本清单目录：

```text
experiment_logs/run_outputs/2026-06-14_095911_486696_visual_router_stage1_sample_manifest_96_48_s_1k/
```

检查结果：

- `status.json.status = completed`；
- `sample_manifest.csv` 行数为 1000；
- `sample_key` 唯一数为 1000；
- 重复 `sample_key` 数为 0；
- split 分布为 `vali=500`、`test=500`；
- 每个 split 下 `TEST_DATA_HOUR=250`、`TEST_DATA_MIN=250`。

### Prediction Cache Launcher 与 Shard

launcher 目录：

```text
experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/
```

检查结果：

- launcher `status.json.status = launcher_created`；
- DLinear、PatchTST、CrossFormer、ES、NaiveForecaster 五个 shard 目录均尚不存在；
- 因此不存在可跳过的已完成 shard，下一步应启动五专家 prediction cache。

## 结论

执行前状态满足启动条件：GPU 空闲，`quito` 环境可用，1k sample manifest 完整，launcher 已生成且尚未启动，五专家 shard 不存在。当前没有发现需要跳过的已完成 prediction cache shard。

## 下一步方案

1. 使用现有 launcher 目录启动五专家 prediction cache，不另建并行 launcher。
2. 启动后按 shard 监控 `status.json`、`main.log` 和 `manifest.csv`。
3. 每个 shard 完成或失败后分别写中文实验日志并更新 README 总览。
4. 若发生 429、503、timeout、connection 或远端服务临时失败，记录命令、时间、日志路径和错误，再按指数退避重试；不删除已有有效输出。
