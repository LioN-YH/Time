# Stage 1 P4d run artifacts 边界复核与接入规划

日志日期：2026-06-19 20:07:39 CST

## 目的

在已完成 P4a/P4b/P4c 的基础上，对 `time_router/io` 当前 run artifacts 相关工具做一次架构边界复核，明确 public API、private helper、低风险 IO helper 与正式训练入口 / launcher / resume 层之间的分工，并给出后续接入规划。

## 背景

P4a 已新增 `time_router/io/json_utils.py`，提供原子 JSON 写入和最小 status payload；P4b 已新增 `path_resolver.py`，提供 repo root、root 内安全路径和 status/metadata path helper；P4c 已新增 `run_metadata.py`，提供最小 metadata-like payload builder。当前这些 helper 都只在 smoke/tempfile 层验证，尚未接入正式 Visual Router / TimeFuse fusor 训练入口，也未接入 `/data2` 或 full-scale 输出目录。

本次 P4d 目标是文档化 review 和 integration plan，不做正式入口迁移，不改变既有 `status.json` / `metadata.json` schema，不实现 checkpoint index、config system 或 logging framework。

## 操作

1. 阅读用户目标文件 `/home/shiyuhong/.codex-tianyu/attachments/7f5b8632-2800-4278-8318-f0b959f120f4/pasted-text-1.txt`，确认本次只允许文档化 review、API 边界说明和 integration plan。
2. 复核 `time_router/io/json_utils.py`、`time_router/io/path_resolver.py`、`time_router/io/run_metadata.py`、`time_router/io/prediction_cache_reader.py` 和 `time_router/io/__init__.py` 的当前职责、导出列表和中文 docstring。
3. 阅读 `docs/refactor/stage1_refactor_roadmap.md` 中 P4a/P4b/P4c 的已有记录，以及 `docs/refactor/json_utils.md`、`docs/refactor/path_resolver.md`、`docs/refactor/run_metadata.md` 的边界说明。
4. 新增 `docs/refactor/run_artifacts_boundary.md`，记录 P4d run artifacts boundary review、public/private API、低风险 IO helper 与正式训练入口/launcher/resume 层分工、后续 integration plan 和 P4e/P4f/P4g 小步候选。
5. 更新 `docs/refactor/stage1_refactor_roadmap.md`，追加 P4d 当前状态、完成范围和明确不做事项。
6. 在 `time_router/io/__init__.py` 补充包级边界注释，说明该入口只聚合稳定 public API，不执行配置读取、路径探测、输出目录创建或训练相关副作用；未修改导出列表、函数签名或 helper 行为。
7. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 P4d 文档和 `time_router/io/__init__.py` 的 public API 聚合边界。
8. 更新 `experiment_logs/README.md`，登记本篇实验日志。
9. 使用 Quito conda 环境运行 P4d 指定验收命令：
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke`

## 结果

- 新增 `docs/refactor/run_artifacts_boundary.md`：
  - 明确 `prediction_cache_reader.py` 属于 Stage 1 prediction cache 数据读取层，不属于 run artifacts writer。
  - 明确 `json_utils.py` 只负责原子 JSON 写入和最小 status payload，不负责路径选择、训练状态采集或 logging framework。
  - 明确 `path_resolver.py` 只负责 repo root 查找、root 内安全拼接和 status/metadata path 计算，不创建正式输出目录、不访问 `/data2`。
  - 明确 `run_metadata.py` 只负责 metadata-like payload 构造和显式 path 写入，不自动调用 git、不读取命令行或训练配置、不替换既有 metadata schema。
  - 明确 `time_router/io/__init__.py` 是 public API 聚合入口，后续正式入口迁移应优先从 `time_router.io` 导入。
  - 列出当前 public API 和 private helper，并给出接入正式入口前需要比较的 status/metadata 字段、launcher/monitor/resume 依赖字段，以及 checkpoint index、config system 的后续处理建议。
- `docs/refactor/stage1_refactor_roadmap.md` 已追加 P4d 章节。
- `WORKSPACE_STRUCTURE.md` 已同步新增文档索引和 IO 包入口说明。
- 验收命令全部通过：
  - golden smoke 通过，五专家顺序、sample_key 顺序、`y_pred/y_true` shape、hard top-1、raw soft fusion、diagnostics、summary 和 per-sample rows 均保持既有 golden 口径。
  - oracle/TSF smoke 通过，覆盖禁止默认全扫描、显式 sample_key 保序、缺失策略和重复冲突报错。
  - P4a JSON utils、P4b path resolver、P4c run metadata smoke 均通过。
  - `compileall time_router tests/smoke` 通过。
- 本次未迁移 Visual Router / TimeFuse fusor 正式训练入口，未修改任何正式训练脚本，未接入 `/data2` 或 full-scale 输出目录，未改变既有 `status.json` / `metadata.json` schema，未实现 checkpoint index、config system 或 logging framework。

## 结论

P4a/P4b/P4c 当前适合作为低风险 IO helper 基础设施继续保留，但还不能直接接入正式 full-scale Visual Router / TimeFuse fusor 入口。正式接入前应先做现有真实输出目录的 status/metadata schema audit，并确认 launcher、monitor、resume 和 handoff 实际依赖字段；checkpoint index 应拆成 P4e 单独审查或实现，config system 应推迟到 P5/P6 前再根据 FeatureProvider 和入口迁移需求设计。

## 下一步方案

1. 检查 `git diff`，确认没有正式训练入口迁移、full-scale 输出目录接入、checkpoint index/config/logging 实现或 helper 行为改动。
2. 小步提交并推送到远程 `refactor/stage1-route-audit` 分支。
3. 后续优先考虑 P4f：existing status/metadata schema audit for Visual Router and TimeFuse fusor outputs；再考虑 P4e checkpoint index boundary review 或 P4g logging/path/config integration plan。
