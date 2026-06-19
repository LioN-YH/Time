# Stage 1 P4b minimal path resolver helper extraction

日志日期：2026-06-19 19:40:54 CST

## 目的

在 P4a atomic JSON / status writer 已完成的基础上，继续抽取 P4 的最小路径解析基础能力，只新增 repo root 查找、root 内安全拼接和 status/metadata path 计算 helper。

## 背景

Stage 1 后续需要逐步收束 logging、path、config、launcher/status 兼容能力，但本次不能直接替换现有正式训练入口或后台任务管理逻辑。P4b 只提供低风险路径解析基础，不读取训练配置，不访问 `/data2` 或 full-scale 输出目录，也不创建任何正式输出目录。

## 操作

1. 新增 `time_router/io/path_resolver.py`：
   - `find_repo_root(...)`：从调用方起点或 helper 文件位置向上查找 `.git`、`WORKSPACE_STRUCTURE.md` 或 pyproject-like marker，找不到时明确报错。
   - `resolve_under_root(...)`：返回 root 内部 resolved path，拒绝 `..` 或绝对路径逃逸 root；`must_exist=True` 时路径不存在会明确报错。
   - `resolve_status_path(...)`：只返回 `run_dir / "status.json"`。
   - `resolve_metadata_path(...)`：只返回 `run_dir / "metadata.json"`。
2. 更新 `time_router/io/__init__.py`，导出 `find_repo_root`、`resolve_under_root`、`resolve_status_path` 和 `resolve_metadata_path`。
3. 新增 `tests/smoke/stage1_path_resolver_smoke.py`，验证：
   - 从 `tests/smoke` 下可找到仓库根；
   - root 下可定位 `WORKSPACE_STRUCTURE.md`；
   - tempfile root 下正常路径可解析；
   - `..` 逃逸 root 会报错；
   - `must_exist=True` 对不存在路径报错；
   - status/metadata helper 只返回路径，不创建目录或文件。
4. 新增 `docs/refactor/path_resolver.md`，记录 P4b helper 边界和不做事项。
5. 更新 `docs/refactor/stage1_refactor_roadmap.md`，追加 P4b 当前状态、完成范围和明确不做事项。
6. 更新 `WORKSPACE_STRUCTURE.md` 和 `experiment_logs/README.md`，登记新增长期文件与 smoke。

## 结果

已在 Quito conda 环境运行并通过以下命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

关键验证结果：

- Stage 1 golden smoke 通过，确认 prediction/fusion/summary/rows 契约未受影响。
- Stage 1 oracle/TSF smoke 通过，确认 `OracleTsfReader` 契约未受影响。
- P4a JSON utils smoke 通过，确认 status writer 仍可用。
- P4b path resolver smoke 通过，确认 repo root 查找、root 逃逸防护、`must_exist=True` 和 status/metadata path helper 行为正确。
- `compileall time_router tests/smoke` 通过。

## 结论

本次只完成 P4b minimal path resolver helper extraction。新增 helper 不读取训练配置、不写 JSON、不创建正式输出目录、不接入 `/data2` 或 full-scale 输出目录、不替换 launcher / monitor / resume 行为，也没有实现 config system、logging framework 或 checkpoint index。

## 下一步方案

后续如继续 P4，应另起小步评估 launcher/status 兼容检查、配置默认值或 logging 边界；任何正式入口迁移仍需保留到后续阶段，并在迁移前后运行 golden smoke、oracle/TSF smoke、P4a/P4b smoke 和必要的小规模等价验证。
