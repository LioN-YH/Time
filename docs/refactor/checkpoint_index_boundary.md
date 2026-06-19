# Stage 1 P4e Checkpoint Index 边界复核与接入规划

创建日期：2026-06-19

## 1. 目标

本文件复核 Stage 1 当前 Visual Router / TimeFuse-style fusor 中 checkpoint、best/latest model、resume、launcher、monitor、`status.json` 和 `metadata.json` 的既有约定，并给出 checkpoint index 后续接入规划。

本次只做架构边界审查和 integration plan；不实现 checkpoint index helper，不修改任何正式训练入口，不替换 launcher / monitor / resume 行为，不改变既有输出文件 schema。

## 2. 当前证据范围

本次审查的主要代码入口：

- `visual_router_experiments/stage1_vali_test_router/train_visual_router.py`
- `visual_router_experiments/stage1_vali_test_router/train_visual_router_online.py`
- `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
- `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`
- `visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py`
- `visual_router_experiments/stage1_vali_test_router/launch_full_scale_prediction_cache.py`
- `visual_router_experiments/stage1_vali_test_router/build_prediction_cache_from_manifest.py`

审查结论以当前代码约定为准；本文件不回写历史 `/data2` 输出目录，也不把历史目录强行升级为新 schema。

## 3. Visual Router 当前约定

### 3.1 非 streaming / 早期 online 入口

`train_visual_router.py` 和 `train_visual_router_online.py` 当前主要写出 prediction/summary/comparison 与 metadata：

- `visual_router_predictions.csv`
- `visual_router_summary.csv`
- `visual_router_soft_fusion_predictions.csv`
- `visual_router_soft_fusion_summary.csv`
- `visual_router_selected_model_counts.csv`
- `visual_router_comparison.csv`
- `visual_router_metadata.json` 或 `visual_router_online_metadata.json`

边界：

- 这些入口没有通用 checkpoint/resume 语义。
- 不应为了统一 P4e 而给历史非 streaming 入口补 checkpoint index。
- 它们的 metadata 是评估/运行摘要，不等价于 checkpoint index。

### 3.2 Streaming online Visual Router 入口

`train_visual_router_online_streaming.py` 已有 checkpoint/resume 约定：

- CLI 支持 `--resume-checkpoint`。
- `checkpoints/router_{config_name}_epoch_{000N}.pt` 保存 epoch checkpoint。
- `checkpoints/latest_{config_name}.pt` 保存 config 级 latest checkpoint。
- `checkpoints/latest_checkpoint_index.json` 保存当前 latest 指针。
- `status.json` 在 `init`、`checkpoint_loaded`、`scaler_fit_completed`、`training`、`checkpoint_saved`、`train_only_done` 和 `done` 等阶段更新。
- `visual_router_metadata.json` 与 `visual_router_online_metadata.json` 记录 `resume_checkpoint`、`latest_checkpoint_path`、`checkpoint_dir`、`epochs_semantics` 和 config 级训练摘要。

当前 checkpoint payload 关键字段：

- `checkpoint_version`
- `config_name`
- `model_columns`
- `router_mode`
- `metric`
- `hidden_dim`
- `dropout`
- `lr`
- `weight_decay`
- `huber_beta`
- `kl_tau`
- `lambda_kl`
- `embedding_metadata`
- `stream_shard_index`
- `stream_shard_count`
- `labels_path`
- `prediction_manifest_path`
- `config_path`
- `router_state_dict`
- `optimizer_state_dict`
- `scaler_state`
- `completed_epochs`
- `epoch_summaries`
- `scaler_batches`
- `scaler_samples`
- `saved_at`

当前 `latest_checkpoint_index.json` 字段：

- `config_name`
- `completed_epochs`
- `checkpoint_path`
- `latest_checkpoint_path`
- `updated_at`

边界：

- 该 index 是 streaming Visual Router 入口私有的 latest 指针，不是跨入口公共 schema。
- `completed_epochs` 是复数命名；TimeFuse 当前使用 `completed_epoch`，二者不能直接合并。
- resume 前有严格签名校验，不能只凭 latest path 恢复。
- `status.json` 的 `latest_checkpoint_path`、`completed_epochs` 和 `phase` 被人工监控和 handoff 依赖，不能在未审计前改名。

## 4. TimeFuse-style Fusor 当前约定

### 4.1 Streaming train/eval 入口

`train_timefuse_fusor_streaming.py` 已有 checkpoint/resume 约定：

- CLI 支持 `--resume-checkpoint`、`--eval-only` 和 `--train-only`。
- `checkpoints/timefuse_fusor_epoch_{000N}.pt` 保存 epoch checkpoint。
- `checkpoints/latest_timefuse_fusor.pt` 保存 latest checkpoint。
- `checkpoints/latest_checkpoint_index.json` 保存 latest 指针。
- `metadata.json` 记录 `checkpoint_path`、`args`、`feature_shards`、`epoch_summaries`、`train_samples`、`test_samples` 和最终资源快照。
- `status.json` 记录 `phase`、`metadata_path`、`summary_path`、`checkpoint_path`、`train_samples` 和 `test_samples`。

当前 checkpoint payload 关键字段：

- `checkpoint_version`
- `saved_at`
- `fusor_state_dict`
- `scaler_state`
- `completed_epoch`
- `completed_shards`
- `model_columns`
- `feature_columns`
- `train_args`
- `epoch_summaries`

当前 `latest_checkpoint_index.json` 字段：

- `checkpoint_path`
- `latest_checkpoint_path`
- `completed_epoch`
- `updated_at`

边界：

- 该 index 是 TimeFuse fusor 入口私有的 latest 指针。
- `completed_epoch` 与 Visual Router 的 `completed_epochs` 不同，表示方式不能无审计地统一。
- checkpoint payload 中包含 `train_args` 和 shard 信息，resume 语义强依赖训练入口，而不是纯 IO 层。
- `eval-only` 要求显式 `--resume-checkpoint`，不能由 path helper 自动推断。

### 4.2 Full-scale launcher

`launch_timefuse_fusor_full_scale.py` 维护 launcher / monitor / resume 约定：

- `command.sh`：首次运行命令。
- `command_resume.sh`：若 `checkpoints/latest_timefuse_fusor.pt` 存在则追加 `--resume-checkpoint`，否则从头运行。
- `launcher.sh`：后台启动并写 `pid.txt`、`pgid.txt`。
- `stop.sh`：按 `pgid.txt` 或 `pid.txt` 停止。
- `resume.sh`：后台执行 `command_resume.sh` 并刷新 `pid.txt`、`pgid.txt`。
- `metadata.json` 记录 `scripts`、`monitor_commands`、`stop_command`、`resume_command`、`resume_policy`、`pid`、`pgid` 和 preflight 信息。
- `status.json` 记录 `status`、`phase`、`pid`、`pgid`、`metadata_path`、`main_log`、`launcher_log`、`stop_command`、`resume_command` 和 preflight 状态。

边界：

- launcher resume 依赖的是 shell 脚本、`pid.txt`、`pgid.txt`、`main.log` 和 `latest_timefuse_fusor.pt` 文件存在性。
- checkpoint index helper 不能替代 launcher script 生成、进程管理、preflight 或 monitor command。
- `output_dir_has_only_launcher_files(...)` 会把 `checkpoints/`、`indexes/`、prediction 等训练产物视为冲突保护边界；任何公共 helper 都不能绕过该保护。

## 5. Prediction cache launcher / builder 约定

`launch_full_scale_prediction_cache.py` 和 `build_prediction_cache_from_manifest.py` 使用的是 prediction cache 构建 resume，不是模型 checkpoint resume。

当前约定：

- launcher root 写 `launcher.sh`、`launch_plan.md`、`status.json`、`metadata.json` 和 `pids/{model}.pid`。
- 每个专家/sample shard 写独立 `main.log`、`status.json`、`metadata.json`、`manifest.csv` 和数组文件。
- launcher 通过 shard `status.json` 中的 `"status": "completed"` 跳过已完成 shard。
- builder 的 `--resume` 基于已有 `manifest.csv` 中可读取数组的完整记录，跳过已完成 item/model 组。
- builder metadata 记录 `resume`、`completed_group_count`、`resume_mode`、`checkpoint_selection`、`array_storage`、`record_count` 等字段。

边界：

- 这里的 `checkpoint_selection` 是专家模型选择口径，例如 `validation_mae_best_or_config_defined`，不是 router/fusor checkpoint index。
- 这里的 `--resume` 是 cache append/resume，不是 torch checkpoint resume。
- P4e checkpoint index helper 不应替代 prediction cache manifest 完整性校验、packed row index、数组存在性检查或 shard `status.json` 语义。

## 6. Launcher / monitor / resume 依赖清单

后续任何 checkpoint index 设计都必须保护以下现有依赖。

文件名和目录：

- `checkpoints/`
- `router_{config_name}_epoch_{000N}.pt`
- `latest_{config_name}.pt`
- `timefuse_fusor_epoch_{000N}.pt`
- `latest_timefuse_fusor.pt`
- `latest_checkpoint_index.json`
- `status.json`
- `metadata.json`
- `visual_router_metadata.json`
- `visual_router_online_metadata.json`
- `summary.md`
- `main.log`
- `launcher.log`
- `command.sh`
- `command_resume.sh`
- `launcher.sh`
- `stop.sh`
- `resume.sh`
- `pid.txt`
- `pgid.txt`
- `pids/*.pid`

关键字段：

- `status`
- `phase`
- `updated_at`
- `output_dir`
- `pid`
- `pgid`
- `metadata_path`
- `main_log`
- `launcher_log`
- `stop_command`
- `resume_command`
- `monitor_commands`
- `resume_policy`
- `resume_checkpoint`
- `checkpoint_path`
- `latest_checkpoint_path`
- `checkpoint_dir`
- `completed_epoch`
- `completed_epochs`
- `current_epoch`
- `epoch_summaries`
- `train_samples`
- `test_samples`
- `router_predictions`
- `resources`
- `resources_final`
- `error`

这些字段和文件名同时服务人工接手、后台进程控制、eval-only、train-only、失败复盘和结果引用。P4 helper 接入时只能向后兼容扩展，不能默认替换。

## 7. Checkpoint index 是否应独立成 helper

应当独立成 helper，但不应在本次实现。

建议原因：

- 现有 Visual Router 和 TimeFuse fusor 都已有 `latest_checkpoint_index.json`，但字段命名、payload、resume 校验和 latest 文件名不同。
- checkpoint index 会直接影响恢复训练、eval-only、latest 指针和人工接手，风险高于 P4a/P4b/P4c 的纯 JSON/path/metadata helper。
- checkpoint index 需要理解训练 epoch、config、model/fusor 类型、checkpoint payload 版本和 resume signature，不能被最小 metadata builder 直接替代。

建议未来最小 helper 只覆盖：

- 原子写入 latest index。
- 规范记录 epoch checkpoint path、latest checkpoint path、completed epoch(s)、updated_at 和可选 entrypoint/type。
- 不负责 `torch.save` payload 内容。
- 不负责选择 best checkpoint。
- 不负责 resume 校验。
- 不负责 launcher script、pid/pgid 或 monitor command。

## 8. Helper 所属层级

checkpoint index helper 不应直接放入当前 `time_router/io` 的低风险 helper 集合。

推荐方案：

1. 若只做“写 JSON 指针文件”的薄封装，可放在未来 `time_router/runtime` 或 `time_router/training` 下，例如 `time_router/training/checkpoints.py`。
2. 若后续建立训练运行时层，可把 checkpoint save/load/index/resume signature 统一放入 `time_router/training`，IO 层只复用 `atomic_write_json(...)`。
3. `time_router/io` 继续保留无训练语义的基础能力：原子 JSON、路径解析、metadata-like payload builder 和 prediction cache reader。

不建议把 checkpoint index 放入 `run_metadata.py`，因为 metadata 是运行说明，checkpoint index 是恢复状态机的一部分。

## 9. P4a/P4b/P4c 不能直接替代的内容

P4a JSON helper 不能替代：

- `torch.save` checkpoint 写入。
- latest checkpoint 指针语义。
- checkpoint payload version。
- resume signature 校验。
- best/latest checkpoint 选择。

P4b path helper 不能替代：

- run_dir 命名和冲突保护。
- full-scale 输出根选择。
- checkpoint 文件命名。
- launcher shell 脚本和 pid/pgid 文件。
- completed shard / completed epoch 判断。

P4c metadata helper 不能替代：

- 既有 `visual_router_metadata.json`、`visual_router_online_metadata.json` 或 TimeFuse `metadata.json` schema。
- checkpoint payload 内的 `router_state_dict` / `fusor_state_dict` / optimizer / scaler 状态。
- `resume_checkpoint` 的签名校验。
- launcher 的 `resume_policy`、`monitor_commands`、`stop_command` 和 `resume_command`。

## 10. 如何保证旧输出目录含义不变

后续如果实现并接入 checkpoint index helper，应遵守：

1. 不回写历史正式输出目录。
2. 不删除或重命名既有 `latest_checkpoint_index.json` 字段。
3. Visual Router 保留 `completed_epochs`，TimeFuse 保留 `completed_epoch`；若新增统一字段，只能并行增加。
4. 保留现有 latest 文件名：Visual `latest_{config}.pt`，TimeFuse `latest_timefuse_fusor.pt`。
5. 保留 launcher 对 `checkpoints/latest_timefuse_fusor.pt` 文件存在性的判断，不能只依赖 index JSON。
6. 保留 `status.json` 和 metadata 中的 `checkpoint_path` / `latest_checkpoint_path` 字段。
7. 先在 smoke/dry-run 新目录接入，再考虑 full-scale 新 run 目录。
8. 接入前后做字段级 diff，并运行 golden smoke、oracle/TSF smoke、P4 helper smoke 和 compileall。

## 11. 后续小步建议

建议后续顺序：

1. P4f：existing status/metadata schema audit for Visual Router and TimeFuse fusor outputs。
2. P4g：logging/path/config integration plan before P6。
3. P4h 或 P6 前小步：minimal checkpoint index helper design，先只定义 schema，不接正式入口。
4. P6：入口迁移时再决定是否把 Visual Router / TimeFuse fusor 的 checkpoint save/load/index 合并到 training/runtime 层。

如果短期进入 P5 FeatureProvider interface design，也可以把 checkpoint helper 推迟到 P6 前，避免在 FeatureProvider 边界尚未稳定时提前耦合训练状态机。

## 12. 本次明确不做

- 不实现 checkpoint index。
- 不修改任何正式训练脚本。
- 不迁移 Visual Router / TimeFuse fusor 入口。
- 不替换 launcher / monitor / resume 行为。
- 不改变 `status.json` / `metadata.json` / checkpoint 文件 schema。
- 不接入 `/data2` 或 full-scale 输出目录。
- 不创建输出目录。
- 不自动调用 git。
- 不实现 config system。
- 不实现 logging framework。
- 不改 `time_router/io` helper 行为。
- 不改 `PredictionBatchReader` / `OracleTsfReader` / evaluation helper。
- 不改模型结构、loss 或正式输出目录。

## 13. 验收

完整 P4e 门禁包括：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

本次验收重点是确认文档化 review 未改变任何 reader、evaluation、IO helper 或训练入口行为。
