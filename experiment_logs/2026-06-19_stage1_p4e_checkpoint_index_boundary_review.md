# Stage 1 P4e checkpoint index 边界复核与接入规划

日志日期：2026-06-19 20:22:51 CST

## 目的

对 Stage 1 当前 Visual Router / TimeFuse-style fusor 正式或历史脚本中的 checkpoint、best/latest model、resume、launcher、monitor、`status.json` 和 `metadata.json` 约定做一次文档化边界复核，并规划后续 checkpoint index helper 是否应独立抽取、应属于哪一层，以及如何避免改变旧输出目录含义。

## 背景

P4d 已确认 checkpoint index 不应并入最小 run metadata helper，而应作为更高风险的训练运行时状态机单独审查。当前仓库中 streaming Visual Router 和 TimeFuse-style fusor 已各自写出 `latest_checkpoint_index.json`，但字段、文件名、resume 校验和 launcher 依赖并不完全一致。直接用 P4a/P4b/P4c 的 JSON/path/metadata helper 替代这些语义，可能破坏 eval-only、train-only、后台监控和 handoff 接手。

本次 P4e 只做文档化 review 和 integration plan，不实现 checkpoint index，不修改任何正式训练入口，不接入 `/data2` 或 full-scale 输出目录。

## 操作

1. 阅读用户目标文件 `/home/shiyuhong/.codex-tianyu/attachments/30bc70ed-efa3-4e6b-8de7-e903012b635e/pasted-text-1.txt`，确认本次只做 checkpoint / resume / launcher / monitor 边界审查和规划。
2. 使用 `rg` 定位 `visual_router_experiments/stage1_vali_test_router/` 与 `docs/refactor/` 中的 `checkpoint`、`resume`、`status.json`、`metadata.json`、`latest`、`launcher`、`monitor` 相关代码和文档。
3. 重点阅读 `train_visual_router_online_streaming.py` 中 `--resume-checkpoint`、`load_checkpoint`、`assert_checkpoint_matches`、`save_checkpoint`、`latest_checkpoint_index.json`、`status.json` 和 metadata 写入逻辑。
4. 重点阅读 `train_timefuse_fusor_streaming.py` 中 `--resume-checkpoint`、`--eval-only`、`--train-only`、`checkpoint_payload`、`save_checkpoint`、`load_checkpoint`、`metadata.json` 和 `status.json` 写入逻辑。
5. 重点阅读 `launch_timefuse_fusor_full_scale.py` 中 `command.sh`、`command_resume.sh`、`launcher.sh`、`stop.sh`、`resume.sh`、`pid.txt`、`pgid.txt`、`monitor_commands`、`resume_policy`、launcher metadata/status 写入逻辑。
6. 补充阅读 `launch_full_scale_prediction_cache.py` 和 `build_prediction_cache_from_manifest.py`，区分 prediction cache 构建 resume、专家 `checkpoint_selection` 与 router/fusor checkpoint index。
7. 新增 `docs/refactor/checkpoint_index_boundary.md`，记录 P4e checkpoint index boundary review、当前约定、依赖清单、helper 所属层级建议和后续小步规划。
8. 更新 `docs/refactor/stage1_refactor_roadmap.md`，追加 P4e 当前状态、完成范围和明确不做事项。
9. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 P4e 文档。
10. 更新 `experiment_logs/README.md`，登记本篇实验日志。
11. 使用 Quito conda 环境运行 P4e 指定验收命令：
    - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py`
    - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py`
    - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py`
    - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py`
    - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py`
    - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke`

## 结果

- 新增 `docs/refactor/checkpoint_index_boundary.md`：
  - 明确非 streaming Visual Router 入口当前没有通用 checkpoint/resume 语义。
  - 明确 streaming Visual Router 使用 `router_{config}_epoch_000N.pt`、`latest_{config}.pt` 和 `latest_checkpoint_index.json`，index 字段包含 `completed_epochs`、`checkpoint_path` 和 `latest_checkpoint_path`。
  - 明确 TimeFuse-style fusor 使用 `timefuse_fusor_epoch_000N.pt`、`latest_timefuse_fusor.pt` 和 `latest_checkpoint_index.json`，index 字段包含 `completed_epoch`、`checkpoint_path` 和 `latest_checkpoint_path`。
  - 明确 TimeFuse full-scale launcher 依赖 shell 脚本、`pid.txt`、`pgid.txt`、`main.log`、`status.json`、`metadata.json`、`resume_policy` 和 monitor commands，checkpoint index helper 不能替代这些职责。
  - 明确 prediction cache builder/launcher 的 `--resume` 是 manifest/array 完整性层面的 cache resume，`checkpoint_selection` 是专家模型 checkpoint 选择口径，不是 router/fusor checkpoint index。
  - 建议未来 checkpoint index helper 更适合 `time_router/training` 或 runtime 层；`time_router/io` 继续只提供低风险 JSON/path/metadata 基础能力。
- `docs/refactor/stage1_refactor_roadmap.md` 已追加 P4e 章节。
- `WORKSPACE_STRUCTURE.md` 已登记 `docs/refactor/checkpoint_index_boundary.md`。
- 验收命令全部通过：
  - golden smoke 通过，五专家顺序、sample_key 顺序、`y_pred/y_true` shape、hard top-1、raw soft fusion、diagnostics、summary 和 per-sample rows 均保持既有 golden 口径。
  - oracle/TSF smoke 通过，覆盖禁止默认全扫描、显式 sample_key 保序、缺失策略和重复冲突报错。
  - P4a JSON utils、P4b path resolver、P4c run metadata smoke 均通过。
  - `compileall time_router tests/smoke` 通过。
- 本次未实现 checkpoint index，未修改任何正式训练脚本，未迁移 Visual Router / TimeFuse fusor 入口，未替换 launcher / monitor / resume 行为，未改变 `status.json` / `metadata.json` / checkpoint 文件 schema。

## 结论

checkpoint index 应独立审查和设计，但不应在当前 P4e 实现。当前 Visual Router 与 TimeFuse fusor 的 `latest_checkpoint_index.json` 都是入口私有 latest 指针，字段与语义不同；未来如果抽 helper，应放在 training/runtime 层，只复用 IO 层的原子 JSON 写入能力，并保留现有 latest 文件名、status/metadata 字段、launcher shell 脚本和 pid/pgid 接手方式。

## 下一步方案

1. 检查 `git diff`，确认没有正式训练入口修改、checkpoint index 实现、full-scale 输出目录接入或 helper 行为变更。
2. 小步提交并推送到远程 `refactor/stage1-route-audit` 分支。
3. 后续优先考虑 P4f 现有 status/metadata schema audit，或进入 P5 FeatureProvider interface design；checkpoint index helper 可推迟到 P6 前单独设计。
