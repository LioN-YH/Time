# Stage 1 P4c minimal run metadata payload builder extraction

日志日期：2026-06-19 19:51:57 CST

## 目的

在 P4a atomic JSON/status writer 和 P4b path resolver 已完成的基础上，继续抽取 P4 的最小 run metadata payload builder，用于构造 metadata-like JSON payload，记录实验阶段、输入输出路径、命令、git/refactor 信息和关键说明。

## 背景

Stage 1 后续需要逐步收束 metadata、logging、path、config 和 launcher/status 兼容能力，但本次只做最小 payload builder，不替换现有正式 `metadata.json` schema，不迁移 Visual Router / TimeFuse fusor 正式训练入口，不接入 `/data2` 或 full-scale 输出目录。

## 操作

1. 新增 `time_router/io/run_metadata.py`：
   - `build_run_metadata(...)`：构造至少包含 `stage`、`created_at_utc`、`inputs`、`outputs` 的 metadata-like payload。
   - `write_run_metadata(...)`：内部调用 `build_run_metadata(...)` 和 P4a `atomic_write_json(...)`，只写调用方显式传入的 path。
2. 字段边界：
   - `stage` 必须是非空字符串。
   - `inputs`、`outputs`、`extra` 必须是 dict 或 None。
   - `created_at_utc` 使用 timezone-aware UTC ISO 字符串。
   - `Path` / `os.PathLike` 在内存中转换为字符串。
   - `extra` 保留在 payload 的 `extra` 字段下，不展开覆盖基础字段。
3. 更新 `time_router/io/__init__.py`，导出 `build_run_metadata` 和 `write_run_metadata`。
4. 新增 `tests/smoke/stage1_run_metadata_smoke.py`，只在 `tempfile.TemporaryDirectory` 下验证 payload 构造、字段校验、UTC 时间、Path 转字符串和 writer JSON 可读。
5. 新增 `docs/refactor/run_metadata.md`，记录 P4c helper 边界和不做事项。
6. 更新 `docs/refactor/stage1_refactor_roadmap.md`，追加 P4c 当前状态、完成范围和明确不做事项。
7. 更新 `WORKSPACE_STRUCTURE.md` 和 `experiment_logs/README.md`，登记新增长期文件与 smoke。

## 结果

已在 Quito conda 环境运行并通过以下命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

关键验证结果：

- Stage 1 golden smoke 通过，确认 prediction/fusion/summary/rows 契约未受影响。
- Stage 1 oracle/TSF smoke 通过，确认 `OracleTsfReader` 契约未受影响。
- P4a JSON utils smoke 通过，确认 atomic JSON/status writer 仍可用。
- P4b path resolver smoke 通过，确认 path helper 仍可用。
- P4c run metadata smoke 通过，确认 metadata payload 基础字段、UTC 时间、Path 转字符串、类型校验和 tempfile writer 行为正确。
- `compileall time_router tests/smoke` 通过。

## 结论

本次只完成 P4c minimal run metadata payload builder extraction。新增 helper 不自动调用 git、不自动读取命令行或训练配置、不自动解析 full-scale 输出目录、不改变既有正式 `metadata.json` schema、不替换 launcher / monitor / resume 行为，也没有实现 config system、logging framework 或 checkpoint index。

## 下一步方案

后续如继续 P4，应另起小步评估 metadata schema 兼容检查、launcher/status 兼容、config 或 logging 边界；任何正式入口迁移仍需保留到后续阶段，并在迁移前后运行 golden smoke、oracle/TSF smoke、P4a/P4b/P4c smoke 和必要的小规模等价验证。
